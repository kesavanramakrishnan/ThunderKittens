// Level 08: Epilogue Pipelining + MMA/Epilogue Overlap
// ======================================================
// The final optimized version. Adds epilogue pipelining (EPI_PIPE_DEPTH)
// to break the output store into smaller chunks, and optionally overlaps
// MMA with epilogue in a single consumer warpgroup (OVERLAP_MMA_EPI).
//
// New concepts:
//   - EPI_PIPE_DEPTH: split the N-dimension of output into chunks for pipelined stores
//   - d_tile = st_bf<BM/2, BN/EPI_PIPE_DEPTH> — smaller output SMEM tiles
//   - NUM_D_TILES = EPI_PIPE_DEPTH > 1 ? 2 : 1 — double-buffer output SMEM
//   - OVERLAP_MMA_EPI mode:
//     * NUM_CONSUMERS=1, MMA_PIPE_DEPTH=2
//     * Single warpgroup handles both MMA and epilogue
//     * Only needs 2 warpgroups total (1 producer + 1 consumer/epilogue)
//   - Non-overlap mode:
//     * NUM_CONSUMERS=2 (separate MMA and epilogue warpgroups)
//     * MMA_PIPE_DEPTH=1
//     * Epilogue loads all TMEM slices at once, frees TMEM faster
//   - TMEM subtile reads: d_tt.subtile<tt<float, BM/2, BN/EPI_PIPE_DEPTH>>(0, offset)
//   - store_async_read_wait<NUM_D_TILES-1> to wait for prior TMA stores
//   - pdl::wait() / warpgroup::pdl::arrive() for dependent launch chaining
//   - Configurable template parameters like the production kernel
//
// This is equivalent to bf16_b200_gemm.cu with all features enabled.
//
// Tile: 256xNb output per cluster, CLUSTER_SIZE=2, fully parameterized
// Layout: A is (M, K) row-major, B is (N, K) row-major, D = A * B^T

#include "kittens.cuh"
#include "../common.cuh"
using namespace kittens;

template <int _Mb, int _Nb, int _Kb, int _SUPERGROUP_SIZE, bool _OVERLAP_MMA_EPI, int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH>
struct config {
    static_assert(_Mb == 256, "Mb must be 256");
    static_assert(_Nb >= 16 && _Nb <= 256 && _Nb % 16 == 0, "Nb must be 16, 32, ..., 256");
    static_assert(_Kb >= 16 && _Kb % 16 == 0, "Kb must be a multiple of 16");
    static_assert(_SUPERGROUP_SIZE >= 1 && _SUPERGROUP_SIZE <= 16, "SUPERGROUP_SIZE must be 1-16");
    static_assert(_LOAD_PIPE_DEPTH >= 1 && _LOAD_PIPE_DEPTH <= 16, "LOAD_PIPE_DEPTH must be 1-16");
    static_assert(_EPI_PIPE_DEPTH >= 1 && _EPI_PIPE_DEPTH <= 16, "EPI_PIPE_DEPTH must be 1-16");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = _Kb;
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;

    static constexpr bool OVERLAP_MMA_EPI = _OVERLAP_MMA_EPI;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int MMA_PIPE_DEPTH = OVERLAP_MMA_EPI ? 2 : 1;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr int CLC_PIPE_DEPTH = 1;

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_CONSUMERS = OVERLAP_MMA_EPI ? 1 : 2;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int NUM_D_TILES = EPI_PIPE_DEPTH > 1 ? 2 : 1;
};

template <typename C>
struct globals {
    using a_tile = st_bf<C::Mb/2, C::Kb>;
    using b_tile = st_bf<C::Nb/2, C::Kb>;
    using d_tile = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

    a_gl a;
    b_gl b;
    d_gl d;

    __host__ __inline__ dim3 grid() { return dim3(d.rows()/(C::NUM_CONSUMERS*C::Mb/2) * d.cols()/C::Nb); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() {
        constexpr size_t _dynamic_shared_memory = sizeof(a_tile) * C::LOAD_PIPE_DEPTH * C::NUM_CONSUMERS +
                                                  sizeof(b_tile) * C::LOAD_PIPE_DEPTH +
                                                  sizeof(d_tile) * C::NUM_D_TILES * C::NUM_CONSUMERS + 1024;
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel(const __grid_constant__ globals<C> g) {
    // TODO: Implement
    //
    // This is the full production kernel with all features.
    // Combine everything from levels 01-07:
    //
    // 1. All semaphores: tmem_provisioned, schedule_arrived/finished,
    //    inputs_arrived/finished, outputs_arrived/finished
    //    Bitfield: uint32_t bitfield = 0xFFFF0000;
    //
    // 2. Producer warpgroup (groupid == NUM_CONSUMERS):
    //    - Warp 3: TMA loader with ring buffer + CLC task loop
    //    - Warp 2: CLC scheduler
    //    - decrease_registers<56>
    //
    // 3. Consumer warpgroup (cta_rank == 0 && groupid < NUM_CONSUMERS):
    //    - TMEM accumulator d_tt[MMA_PIPE_DEPTH]
    //    - K-loop: mm2_ABt / mma2_ABt
    //    - detail::tcgen05::commit<CLUSTER_SIZE>(outputs_arrived)
    //
    // 4. Epilogue (the else branch, all CTAs):
    //    - if OVERLAP_MMA_EPI: single consumer handles both MMA and store
    //    - if !OVERLAP_MMA_EPI: separate epilogue warpgroup with more registers
    //    - TMEM -> registers -> SMEM -> TMA store
    //    - EPI_PIPE_DEPTH chunks along N dimension
    //    - NUM_D_TILES double-buffering for output SMEM
    //    - pdl::arrive() on last task
    //    - deprovision TMEM at end
}

template <typename C>
__host__ double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Template: Mb=" << C::Mb << " Nb=" << C::Nb << " Kb=" << C::Kb << " SUPERGROUP_SIZE=" << C::SUPERGROUP_SIZE
              << " OVERLAP_MMA_EPI=" << C::OVERLAP_MMA_EPI << " LOAD_PIPE_DEPTH=" << C::LOAD_PIPE_DEPTH << " EPI_PIPE_DEPTH=" << C::EPI_PIPE_DEPTH << "\n";

    sleep_ms(500);

    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = 2 * (size_t(M) * K + size_t(N) * K + size_t(M) * N);
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    std::vector<__nv_bfloat16*> d_A(arg_group_count);
    std::vector<__nv_bfloat16*> d_B(arg_group_count);
    std::vector<__nv_bfloat16*> d_C(arg_group_count);
    __nv_bfloat16* d_C_ref;
    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], M*K*sizeof(__nv_bfloat16)));
        CUDACHECK(cudaMalloc(&d_B[i], K*N*sizeof(__nv_bfloat16)));
        CUDACHECK(cudaMalloc(&d_C[i], M*N*sizeof(__nv_bfloat16)));
    }
    CUDACHECK(cudaMalloc(&d_C_ref, M*N*sizeof(__nv_bfloat16)));

    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<__nv_bfloat16, FillMode::RANDOM>(d_A[i], M*K, seed + i*100, -1.0f, 1.0f);
        fill<__nv_bfloat16, FillMode::RANDOM>(d_B[i], K*N, seed + i*100 + 1, -1.0f, 1.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_C[i], M*N, 0.0f);
    }
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_C_ref, M*N, 0.0f);
    CUDACHECK(cudaDeviceSynchronize());

    reference_gemm<__nv_bfloat16, __nv_bfloat16>(d_C_ref, d_A[0], d_B[0], M, N, K);
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<globals<C>> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename globals<C>::a_gl Ag{d_A[i], nullptr, nullptr, (int)M, (int)K};
        typename globals<C>::b_gl Bg{d_B[i], nullptr, nullptr, (int)N, (int)K};
        typename globals<C>::d_gl Dg{d_C[i], nullptr, nullptr, (int)M, (int)N};
        g.push_back(globals<C>{Ag, Bg, Dg});
    }

    CUDACHECK(cudaFuncSetAttribute(kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g[0].dynamic_shared_memory()));
    LaunchConfig<true, true> launch_config(g[0].grid(), g[0].block(), g[0].dynamic_shared_memory(), 0, C::CLUSTER_SIZE);

    int num_warmups = ncu ? 0 : 5;
    int num_iters = ncu ? 1 : 10;

    for (int i = 0; i < num_warmups; i++) {
        cudaLaunchKernelEx(launch_config, kernel<C>, g[i % arg_group_count]);
    }

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        cudaLaunchKernelEx(launch_config, kernel<C>, g[i % arg_group_count]);
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;
    double flops = double(2.0) * M * N * K;
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    check_correctness(d_C[0], d_C_ref, M * N);

    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }
    cudaFree(d_C_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return tflops;
}

__host__ int main() {
    int N;
    bool ncu = false;

    N = 4096;
    run_benchmark<config<256, 256, 64, 4, false, 4, 8>>(N, N, N, ncu);
    N = 8192;
    run_benchmark<config<256, 256, 64, 8, false, 4, 8>>(N, N, N, ncu);

    return 0;
}
