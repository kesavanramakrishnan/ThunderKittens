// Level 06: CLC (Cluster Launch Control) — Persistent Kernel
// ============================================================
// Replace fixed grid-based tile assignment with CLC hardware scheduling.
// The kernel runs persistently — each cluster requests the next tile from
// the CLC hardware unit instead of computing it from blockIdx.
//
// New concepts:
//   - clc::handle for storing schedule results
//   - clc::schedule(handle, semaphore) — ask hardware for next tile (CTA 0 only)
//   - clc::query(handle) — read the result (schedule.success, schedule.x)
//   - schedule_arrived / schedule_finished semaphores for CLC pipeline
//   - tma::expect_bytes for CLC async delivery (piggybacks on TMA mbarrier)
//   - Dedicated scheduler warp (warp 2 of producer warpgroup)
//   - Infinite loop: for (task_iter = 0; true; task_iter++) ... break on !success
//   - LaunchConfig<true, true> for CLC + PDL enabled launch
//
// Tile: 256x128 output per cluster, CLUSTER_SIZE=2
// Layout: A is (M, K) row-major, B is (N, K) row-major, D = A * B^T

#include "kittens.cuh"
using namespace kittens;

static constexpr int BM = 256;
static constexpr int BN = 128;
static constexpr int BK = 64;
static constexpr int LOAD_PIPE_DEPTH = 4;
static constexpr int CLC_PIPE_DEPTH = 1;
static constexpr int CLUSTER_SIZE = 2;

static constexpr int NUM_CONSUMER_WARPGROUPS = 1;
static constexpr int NUM_PRODUCER_WARPGROUPS = 1;
static constexpr int NUM_WARPS = (NUM_CONSUMER_WARPGROUPS + NUM_PRODUCER_WARPGROUPS) * 4;
static constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

struct matmul_globals {
    using a_tile = st_bf<BM/2, BK>;
    using b_tile = st_bf<BN/2, BK>;
    using d_tile = st_bf<BM/2, BN>;
    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;
    a_gl A;
    b_gl B;
    d_gl D;

    __host__ __inline__ dim3 grid() { return dim3(D.rows()/BM * D.cols()/BN * CLUSTER_SIZE); }
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {
    // TODO: Implement
    //
    // Key differences from Level 05:
    //
    // 1. New shared state:
    //    __shared__ clc::handle clc_handle[CLC_PIPE_DEPTH];
    //    __shared__ semaphore schedule_arrived[CLC_PIPE_DEPTH];
    //    __shared__ semaphore schedule_finished[CLC_PIPE_DEPTH];
    //    init schedule_finished with arrival count:
    //        (2 + NUM_CONSUMERS) * CLUSTER_SIZE + NUM_CONSUMERS
    //
    // 2. Initial tile from blockIdx (first task only):
    //    int2 tile_coord = {blockIdx.x / (CLUSTER_SIZE * (N/BN)),
    //                       (blockIdx.x / CLUSTER_SIZE) % (N/BN)};
    //    (or use a simple linear-to-2D mapping)
    //
    // 3. Producer warp 3 (TMA loader) — infinite loop:
    //    for (task_iter = 0; true; task_iter++) {
    //        // K-loop: same as Level 05
    //        // After K-loop:
    //        wait(schedule_arrived[task_iter % CLC_PIPE_DEPTH], ...);
    //        auto schedule = clc::query(clc_handle[...]);
    //        tma::cluster::arrive(schedule_finished[...], 0);
    //        if (schedule.success) tile_coord = ...(schedule.x);
    //        else break;
    //    }
    //
    // 4. Producer warp 2 (CLC scheduler) — infinite loop:
    //    for (task_iter = 0; true; task_iter++) {
    //        if (cta_rank == 0) {
    //            wait(schedule_finished[...], ...);
    //            clc::schedule(clc_handle[...], schedule_arrived[...]);
    //        }
    //        tma::expect_bytes(schedule_arrived[...], sizeof(clc_handle[...]));
    //        wait(schedule_arrived[...], ...);
    //        auto schedule = clc::query(clc_handle[...]);
    //        tma::cluster::arrive(schedule_finished[...], 0);
    //        if (!schedule.success) break;
    //    }
    //
    // 5. Consumer + Epilogue also wait on schedule_arrived and query schedule
    //    to know when to break.
}

void matmul(bf16* A, bf16* B, bf16* C, size_t M, size_t N, size_t K) {
    matmul_globals::a_gl a_arg{A, nullptr, nullptr, (int)M, (int)K};
    matmul_globals::b_gl b_arg{B, nullptr, nullptr, (int)N, (int)K};
    matmul_globals::d_gl d_arg{C, nullptr, nullptr, (int)M, (int)N};
    matmul_globals g{a_arg, b_arg, d_arg};

    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // CLC + PDL enabled launch
    LaunchConfig<true, true> launch_config(g.grid(), dim3(NUM_THREADS), mem_size, 0, CLUSTER_SIZE);
    cudaLaunchKernelEx(launch_config, kernel, g);
}

#include "launch.cu"
