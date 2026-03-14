// Level 03: 2-CTA Clusters + Distributed Shared Memory (DSMEM)
// ==============================================================
// Launch with CLUSTER_SIZE=2. Each CTA loads half the tile. MMA issued
// from CTA 0 only — it reads both CTAs' shared memory via DSMEM.
//
// New concepts:
//   - cluster_ctarank() to identify CTA 0 vs CTA 1
//   - tma::cluster::load_async with multicast mask (uint16_t)(1<<cta_rank)
//   - everyone::tma::cluster::arrive_aligned() / wait() for cluster barrier
//   - Tiles are halved: a_tile = st_bf<BM/2, BK>, b_tile = st_bf<BN/2, BK>
//   - Only CTA 0 issues mm2_ABt / mma2_ABt (tcgen05 reads both CTAs' SMEM)
//   - tensor_allocator<1, CLUSTER_SIZE, false> for cluster-aware TMEM
//
// Tile: 256x128 output per cluster (128 rows per CTA)
// Layout: A is (M, K) row-major, B is (N, K) row-major, D = A * B^T

#include "kittens.cuh"
using namespace kittens;

static constexpr int BM = 256; // per cluster (128 per CTA)
static constexpr int BN = 128;
static constexpr int BK = 64;
static constexpr int QSIZE = 2;
static constexpr int CLUSTER_SIZE = 2;

static constexpr int NUM_CONSUMER_WARPGROUPS = 1;
static constexpr int NUM_PRODUCER_WARPGROUPS = 1;
static constexpr int NUM_WARPS = (NUM_CONSUMER_WARPGROUPS + NUM_PRODUCER_WARPGROUPS) * 4;
static constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

struct matmul_globals {
    using a_tile = st_bf<BM/2, BK>;  // half: one per CTA
    using b_tile = st_bf<BN/2, BK>;  // half: one per CTA
    using d_tile = st_bf<BM/2, BN>;
    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;
    a_gl A;
    b_gl B;
    d_gl D;
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {
    // TODO: Implement
    //
    // Key differences from Level 02:
    //
    // 1. Cluster setup:
    //    const int cta_rank = cluster_ctarank();  // 0 or 1
    //    everyone::tma::cluster::arrive_aligned(); // after init
    //    everyone::tma::cluster::wait();           // before work
    //
    // 2. Producer TMA loads use cluster API:
    //    tma::cluster::load_async(a_smem[q], g.A,
    //        {tile_row*2 + cta_rank, k_idx},
    //        inputs_arrived[q], (uint16_t)(1<<cta_rank), 0);
    //    tma::cluster::load_async(b_smem[q], g.B,
    //        {tile_col*2 + cta_rank, k_idx},
    //        inputs_arrived[q], (uint16_t)(1<<cta_rank), 0);
    //
    // 3. Consumer MMA (CTA 0 only):
    //    if (cta_rank == 0 && ...) {
    //        mm2_ABt / mma2_ABt  // tcgen05 reads DSMEM from both CTAs
    //    }
    //
    // 4. tensor_allocator<1, CLUSTER_SIZE, false> for TMEM
    //
    // 5. Epilogue: each CTA stores its own half of output
    //    coord = {tile_row*2 + cta_rank, tile_col}
}

void matmul(bf16* A, bf16* B, bf16* C, size_t M, size_t N, size_t K) {
    matmul_globals::a_gl a_arg{A, nullptr, nullptr, (int)M, (int)K};
    matmul_globals::b_gl b_arg{B, nullptr, nullptr, (int)N, (int)K};
    matmul_globals::d_gl d_arg{C, nullptr, nullptr, (int)M, (int)N};
    matmul_globals g{a_arg, b_arg, d_arg};

    dim3 grid(M / BM * N / BN * CLUSTER_SIZE);  // 1D grid, 2 CTAs per cluster
    dim3 block(NUM_THREADS);
    unsigned long mem_size = 200000;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Cluster launch
    LaunchConfig<false, false> launch_config(grid, block, mem_size, 0, CLUSTER_SIZE);
    cudaLaunchKernelEx(launch_config, kernel, g);
}

#include "launch.cu"
