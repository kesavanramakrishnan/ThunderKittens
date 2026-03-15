// Level 03: 2-CTA Clusters + Distributed Shared Memory (DSMEM)
// CLUSTER_SIZE=2. MMA reads both CTAs' SMEM via DSMEM.
//
// New: cluster_ctarank(), tma::cluster::load_async with multicast,
//      mm2_ABt/mma2_ABt (ncta=2), cluster-aware tensor_allocator,
//      cluster barrier sync
//
// Tile: 256×128 per cluster (128×128 per CTA), CLUSTER_SIZE=2

#include "kittens.cuh"
using namespace kittens;

static constexpr int BM = 256;  // per cluster (128 per CTA)
static constexpr int BN = 128;
static constexpr int BK = 64;
static constexpr int CLUSTER_SIZE = 2;

static constexpr int NUM_WARPS = 4;  // single warpgroup per CTA
static constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

struct matmul_globals {
    // Per-CTA tiles: A is half of M rows, B is half of N rows
    // mm2 (ncta=2) reads both CTAs' descriptors for full 256×128 MMA
    using a_tile = st_bf<BM/CLUSTER_SIZE, BK>;   // 128×64 per CTA
    using b_tile = st_bf<BN/CLUSTER_SIZE, BK>;   // 64×64  per CTA
    using d_tile = st_bf<BM/CLUSTER_SIZE, BN>;   // 128×128 per CTA output
    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;
    a_gl A;
    b_gl B;
    d_gl D;
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {
    using G = matmul_globals;
    // Prefetch TMA descriptors
    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::a_tile>();
        g.B.template prefetch_tma<typename G::b_tile>();
        g.D.template prefetch_tma<typename G::d_tile>();
    }
    const int num_k_tiles = g.A.cols() / BK;
    const int num_m_clusters = g.D.rows() / (BM/CLUSTER_SIZE);  // in units of per-CTA tiles
    const int cluster_idx = blockIdx.x / CLUSTER_SIZE;
    const int cluster_row = cluster_idx / (g.D.cols() / BN);     // which 256-row cluster
    const int cluster_col = cluster_idx % (g.D.cols() / BN);     // which 128-col cluster

    const int cta_rank = cluster_ctarank();

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    typename G::a_tile (&a_smem) = al.allocate<G::a_tile>();
    typename G::b_tile (&b_smem) = al.allocate<G::b_tile>();
    typename G::d_tile (&d_smem) = al.allocate<G::d_tile>();


    tensor_allocator<1, CLUSTER_SIZE, false> tm_alloc{};
    using d_tt_t = tt<float, BM/CLUSTER_SIZE, BN>;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore inputs_arrived, inputs_finished, outputs_arrived, tmem_finished;

    // Initialize semaphores (single thread)
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);   // 1 TMA transaction group
        init_semaphore(inputs_finished, 0, 1);  // 1 MMA signals done reading smem
        init_semaphore(outputs_arrived, 0, 1);  // commit signals MMA results ready
        init_semaphore(tmem_finished, 0, 1);
    }
    // Provision TMEM (entire warp 0 must participate)
    if (warpid() == 0) {
        tm_alloc.provision(tmem_addr);
    }
    // Cluster barrier: ensures provision() is done on both CTAs before any thread reads tmem_addr
    everyone::tma::cluster::sync();

    // All threads: set up allocator with the provisioned address
    tm_alloc.set_addr(tmem_addr);
    d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(0);

    for (int k_iter = 0; k_iter < num_k_tiles; k_iter++) {
        // Wait for previous MMA to finish reading smem (both CTAs must wait
        // since mm2 on CTA 0 reads both CTAs' smem via DSMEM)
        if (k_iter > 0) {
            wait(inputs_finished, (k_iter - 1) % 2);
        }

        // Both CTAs load their own tiles, signaling CTA 0's barrier (dst_mbar_cta=0)
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            // Only CTA 0 sets expected bytes (all loads signal CTA 0's barrier)
            if (cta_rank == 0) {
                tma::expect_bytes(
                    inputs_arrived,
                    CLUSTER_SIZE * size_bytes<typeof(a_smem)> +
                    CLUSTER_SIZE * size_bytes<typeof(b_smem)>
                );
            }
            tma::cluster::load_async(a_smem, g.A, {0, 0, cluster_row*2 + cta_rank, k_iter}, inputs_arrived, (uint16_t)(1 << cta_rank), 0);
            tma::cluster::load_async(b_smem, g.B, {0, 0, cluster_col*2 + cta_rank, k_iter}, inputs_arrived, (uint16_t)(1 << cta_rank), 0);
        }

        // Only CTA 0 waits on inputs_arrived and issues MMA
        // mm2 reads both CTAs' smem via DSMEM, writes results to cluster TMEM
        // inputs_finished is multicast to both CTAs via commit<2>
        if (cta_rank == 0) {
            wait(inputs_arrived, k_iter % 2);
            if (threadIdx.x == 0) {
                if (k_iter == 0) mm2_ABt(d_tt, a_smem, b_smem, inputs_finished);
                else             mma2_ABt(d_tt, a_smem, b_smem, inputs_finished);
            }
        }
    }

    // Commit: CTA 0 flushes MMA pipeline, multicast signal to both CTAs
    if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
        detail::tcgen05::commit<CLUSTER_SIZE>(outputs_arrived);
    }
    // Both CTAs wait for outputs_arrived (commit<2> multicasts to both)
    wait(outputs_arrived, 0);

    // Epilogue: both CTAs read their own half from cluster TMEM
    rt_bf<BM/CLUSTER_SIZE/4, BN> d_reg; // divide by 4 per warpgroup
    warpgroup::load_async(d_reg, d_tt);
    tensor_load_wait();
    warpgroup::store(d_smem, d_reg);
    warpgroup::sync(1);
    if (warpgroup::laneid() == 0) {
        tma::store_async(g.D, d_smem, {0, 0, cluster_row*2 + cta_rank, cluster_col});
        tma::store_async_read_wait();
    }

    // Sync both CTAs before deprovision (cluster-scope operation)
    warpgroup::sync(1);
    if (warpid() == 0) {
        if (warp::elect_leader()) tma::cluster::arrive(tmem_finished, 1-cta_rank);
        wait(tmem_finished, 0);
        tm_alloc.deprovision();
    }

}

void matmul(bf16* A, bf16* B, bf16* C, size_t M, size_t N, size_t K) {
    matmul_globals::a_gl a_arg{A, nullptr, nullptr, M, K};
    matmul_globals::b_gl b_arg{B, nullptr, nullptr, N, K};
    matmul_globals::d_gl d_arg{C, nullptr, nullptr, M, N};
    matmul_globals g{a_arg, b_arg, d_arg};

    // Grid: 1D, 2 CTAs per cluster tile
    // Each cluster handles BM×BN = 256×128 of output
    dim3 grid((M / BM) * (N / BN) * CLUSTER_SIZE);
    dim3 block(NUM_THREADS);
    unsigned long mem_size = 200000;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Cluster launch
    LaunchConfig<true, false> launch_config(grid, block, mem_size, 0, CLUSTER_SIZE);
    cudaLaunchKernelEx(launch_config, kernel, g);
}

#include "launch.cu"
