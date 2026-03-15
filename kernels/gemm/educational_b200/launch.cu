// Launch harness for educational B200 matmul levels.
// Each level_XX.cu defines:
//   void matmul(bf16* A, bf16* B, bf16* C, size_t M, size_t N, size_t K)
// and then #includes this file to get main() + benchmarking.
//
// Benchmarking methodology (per MLPerf/NVIDIA conventions):
//   - Multiple input groups for L2 eviction (cold-cache simulation)
//   - 500 warmup iterations to reach power-steady state
//   - 100 profiling iterations, back-to-back without intermediate sync
//   - CUDA events for precise GPU-side timing
//   - 500ms thermal cooldown between problem sizes
//   - cuBLAS baseline for comparison

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common.cuh"

#ifndef CUDACHECK
#define CUDACHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return -1; \
    } \
} while(0)
#endif

#define CUBLASCHECK(err) do { \
    cublasStatus_t s = (err); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << s \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return -1; \
    } \
} while(0)

int run_benchmark(size_t M, size_t N, size_t K) {
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";

    // Determine number of input groups for L2 cache eviction
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t input_size = (M*K + K*N + M*N) * sizeof(__nv_bfloat16);
    const size_t ideal_size = size_t(l2_cache_size) * 3;
    const int num_groups = (input_size >= ideal_size) ? 1 : int(ideal_size / input_size) + 1;
    std::cout << "L2 cache: " << l2_cache_size / (1024*1024) << " MB, input groups: " << num_groups << std::endl;

    // Allocate device memory for each input group
    std::vector<__nv_bfloat16*> d_A(num_groups), d_B(num_groups), d_C(num_groups);
    __nv_bfloat16 *d_C_ref;
    for (int g = 0; g < num_groups; g++) {
        CUDACHECK(cudaMalloc(&d_A[g], M*K*sizeof(__nv_bfloat16)));
        CUDACHECK(cudaMalloc(&d_B[g], K*N*sizeof(__nv_bfloat16)));
        CUDACHECK(cudaMalloc(&d_C[g], M*N*sizeof(__nv_bfloat16)));
    }
    CUDACHECK(cudaMalloc(&d_C_ref, M*N*sizeof(__nv_bfloat16)));
    std::cout << "Allocated device memory" << std::endl;

    // Initialize with bitwise-identical random bf16 values on device
    uint64_t seed = 2024;
    for (int g = 0; g < num_groups; g++) {
        fill<__nv_bfloat16, FillMode::RANDOM>(d_A[g], M*K, seed + g*100,     -0.5f, 0.5f);
        fill<__nv_bfloat16, FillMode::RANDOM>(d_B[g], K*N, seed + g*100 + 1, -0.5f, 0.5f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_C[g], M*N, 0.0f);
    }
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_C_ref, M*N, 0.0f);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Initialized matrices" << std::endl;

    // Compute reference matmul on GPU (A * B^T)
    reference_gemm<__nv_bfloat16, __nv_bfloat16, true>(d_C_ref, d_A[0], d_B[0], M, N, K);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Computed reference matmul on device" << std::endl;

    constexpr int NUM_WARMUP = 500;
    constexpr int NUM_ITERS  = 100;
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    // ==================== TK Kernel ====================
    // Warmup (500 iterations)
    for (int i = 0; i < NUM_WARMUP; i++) {
        int g = i % num_groups;
        matmul(d_A[g], d_B[g], d_C[g], M, N, K);
    }
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaGetLastError());

    // Profile (100 iterations, CUDA events)
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERS; i++) {
        int g = i % num_groups;
        matmul(d_A[g], d_B[g], d_C[g], M, N, K);
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float tk_ms;
    CUDACHECK(cudaEventElapsedTime(&tk_ms, start, stop));
    double tk_us = (tk_ms * 1000.0) / NUM_ITERS;
    double flops = 2.0 * M * N * K;
    double tk_tflops = (flops / tk_us) / 1e6;
    std::cout << "TK kernel:   " << tk_us << " us  (" << tk_tflops << " TFLOPs)\n";

    CUDACHECK(cudaGetLastError());

    // Verify TK correctness
    check_correctness(d_C[0], d_C_ref, M * N);

    // ==================== cuBLAS ====================
    sleep_ms(500); // thermal cooldown

    cublasHandle_t handle;
    CUBLASCHECK(cublasCreate(&handle));
    CUBLASCHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    // cuBLAS computes C = alpha * A * B^T + beta * C in column-major.
    // Our layout: row-major A (M×K), row-major B (N×K), D = A * B^T.
    // Trick: interpret row-major as col-major transpose.
    //   col-major: C^T (N×M) = B (N×K) * A^T (K×M)
    //   => cublasGemmEx(N, M, K, B, N, A, K, C, N)
    float alpha = 1.0f, beta = 0.0f;

    // Warmup cuBLAS
    for (int i = 0; i < NUM_WARMUP; i++) {
        int g = i % num_groups;
        CUBLASCHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B[g], CUDA_R_16BF, K,
            d_A[g], CUDA_R_16BF, K,
            &beta,
            d_C[g], CUDA_R_16BF, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Profile cuBLAS
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERS; i++) {
        int g = i % num_groups;
        CUBLASCHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B[g], CUDA_R_16BF, K,
            d_A[g], CUDA_R_16BF, K,
            &beta,
            d_C[g], CUDA_R_16BF, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float cublas_ms;
    CUDACHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
    double cublas_us = (cublas_ms * 1000.0) / NUM_ITERS;
    double cublas_tflops = (flops / cublas_us) / 1e6;
    std::cout << "cuBLAS:      " << cublas_us << " us  (" << cublas_tflops << " TFLOPs)\n";
    std::cout << "TK/cuBLAS:   " << (tk_tflops / cublas_tflops * 100.0) << "%\n";

    cublasDestroy(handle);

    // Clean up
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    for (int g = 0; g < num_groups; g++) {
        cudaFree(d_A[g]);
        cudaFree(d_B[g]);
        cudaFree(d_C[g]);
    }
    cudaFree(d_C_ref);

    return 0;
}

int main() {
    size_t N;

    N = 4096;
    run_benchmark(N, N, N);

    sleep_ms(500); // thermal cooldown between sizes

    N = 8192;
    run_benchmark(N, N, N);

    return 0;
}
