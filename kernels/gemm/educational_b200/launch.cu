// Launch harness for educational B200 GEMM levels.
// Each level_XX.cu defines:
//   void matmul(bf16* A, bf16* B, bf16* C, size_t M, size_t N, size_t K)
// and then #includes this file to get main() + benchmarking.

#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <chrono>

#include "../common.cuh"

int run_benchmark(size_t M, size_t N, size_t K) {
    cudaError_t cudaStatus;
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (size_t i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (size_t i = 0; i < K * N; ++i) h_B[i] = dis(gen);
    std::cout << "Initialized matrices" << std::endl;

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C, *d_C_ref;
    cudaMalloc(&d_A, M*K*sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K*N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, M*N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_C_ref, M*N*sizeof(__nv_bfloat16));
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }
    std::cout << "Allocated device memory" << std::endl;

    // Convert to bf16 and copy to device
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    for (size_t i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (size_t i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);
    cudaMemcpy(d_A, h_A_bf16, M*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, K*N*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    std::cout << "Copied matrices to device" << std::endl;

    // Compute reference GEMM on GPU (A * B^T stored as row-major)
    reference_gemm<__nv_bfloat16, __nv_bfloat16, true>(d_C_ref, d_A, d_B, M, N, K);
    cudaDeviceSynchronize();
    std::cout << "Computed reference GEMM on device" << std::endl;

    // Warmup
    for (int i = 0; i < 2; i++) {
        matmul(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error after warmup: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    // Benchmark
    constexpr int ITERS = 10;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        matmul(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;
    double flops = double(2.0) * M * N * K;
    double tflops = (flops / useconds) / 1e6;
    std::cout << "Average kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error after benchmark: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    // Verify correctness
    check_correctness(d_C, d_C_ref, M * N);

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);

    return 0;
}

int main() {
    size_t N;

    N = 4096;
    run_benchmark(N, N, N);

    N = 8192;
    run_benchmark(N, N, N);

    return 0;
}
