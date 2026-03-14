// Level 07: Supergroup Swizzling (L2 Cache Optimization)
// ========================================================
// Replace linear block-to-tile mapping with supergroup swizzled mapping
// for better L2 cache reuse across nearby CTAs.
//
// New concepts:
//   - get_swizzled_2d_idx<SUPERGROUP_SIZE>(rblks, cblks, cluster_idx)
//   - SUPERGROUP_SIZE controls how many clusters share L2-cached data
//   - rblks = D.rows() / BM, cblks = D.cols() / BN for tile grid dimensions
//   - blockIdx.x / CLUSTER_SIZE gives the cluster index
//   - Nearby clusters in a supergroup work on adjacent tiles to share A/B in L2
//
// Everything else is the same as Level 06 — this is purely a scheduling change.
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
static constexpr int SUPERGROUP_SIZE = 4;

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
    // Key differences from Level 06:
    //
    // 1. Compute grid dimensions for swizzled indexing:
    //    const int rblks = g.D.rows() / BM;  // row blocks
    //    const int cblks = g.D.cols() / BN;   // col blocks
    //
    // 2. Initial tile coordinate uses swizzled mapping:
    //    int2 tile_coord = get_swizzled_2d_idx<SUPERGROUP_SIZE>(
    //        rblks, cblks, blockIdx.x / CLUSTER_SIZE);
    //
    // 3. After CLC returns a new task, also use swizzled mapping:
    //    if (schedule.success)
    //        tile_coord = get_swizzled_2d_idx<SUPERGROUP_SIZE>(
    //            rblks, cblks, schedule.x / CLUSTER_SIZE);
    //
    // 4. A load coordinate:
    //    {tile_coord.x*2 + cta_rank, k_idx}
    //    B load coordinate:
    //    {tile_coord.y*2 + cta_rank, k_idx}
    //    D store coordinate:
    //    {tile_coord.x*2 + cta_rank, tile_coord.y}
    //
    // Everything else is identical to Level 06.
}

void matmul(bf16* A, bf16* B, bf16* C, size_t M, size_t N, size_t K) {
    matmul_globals::a_gl a_arg{A, nullptr, nullptr, (int)M, (int)K};
    matmul_globals::b_gl b_arg{B, nullptr, nullptr, (int)N, (int)K};
    matmul_globals::d_gl d_arg{C, nullptr, nullptr, (int)M, (int)N};
    matmul_globals g{a_arg, b_arg, d_arg};

    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    LaunchConfig<true, true> launch_config(g.grid(), dim3(NUM_THREADS), mem_size, 0, CLUSTER_SIZE);
    cudaLaunchKernelEx(launch_config, kernel, g);
}

#include "launch.cu"
