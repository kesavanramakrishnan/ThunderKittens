// Level 06: Naive Persistent Kernel (Strided Tile Assignment)
// =============================================================
// Builds on Level 05 (3 warpgroups, bitfield phase tracking, LOAD_PIPE_DEPTH=4).
// Instead of one-shot execution where each cluster processes exactly one tile,
// the kernel runs persistently — each cluster loops over tiles with a stride
// equal to the number of clusters in the grid.
//
// New concepts (vs Level 05):
//   - Persistent outer task loop: for (tile = cluster_idx; tile < total; tile += num_clusters)
//   - Strided tile assignment — no atomics, no cross-CTA communication needed
//   - TMEM provisioned once, reused across all tasks
//   - outputs_finished semaphore: epilogue signals when store is done and
//     TMEM is safe to overwrite with next task's MMA results
//   - Semaphore phases keep advancing naturally across tasks (bitfield tracks)
//   - Launch with fewer clusters than tiles: grid = min(total, num_SMs/2) clusters
//
// Tile: 256x128 output per cluster, CLUSTER_SIZE=2
// Layout: A is (M, K) row-major, B is (N, K) row-major, D = A * B^T

#include "kittens.cuh"
using namespace kittens;

static constexpr int BM = 256;
static constexpr int BN = 128;
static constexpr int BK = 64;
static constexpr int LOAD_PIPE_DEPTH = 4;
static constexpr int CLUSTER_SIZE = 2;

static constexpr int NUM_CONSUMER_WARPGROUPS = 1;
static constexpr int NUM_EPILOGUE_WARPGROUPS = 1;
static constexpr int NUM_PRODUCER_WARPGROUPS = 1;
static constexpr int NUM_WARPS = (NUM_CONSUMER_WARPGROUPS + NUM_EPILOGUE_WARPGROUPS + NUM_PRODUCER_WARPGROUPS) * 4;
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
    int total_tiles;    // total number of cluster-sized tiles
    int num_clusters;   // number of clusters in the grid (stride)
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {
    using G = matmul_globals;
    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::a_tile>();
        g.B.template prefetch_tma<typename G::b_tile>();
        g.D.template prefetch_tma<typename G::d_tile>();
    }
    const int num_k_tiles   = g.A.cols() / BK;
    const int num_col_tiles = g.D.cols() / BN;
    const int cluster_start = blockIdx.x / CLUSTER_SIZE;
    const int cta_rank      = cluster_ctarank();
    const int wg_id         = warpgroup::groupid();

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    typename G::a_tile (&a_smem)[LOAD_PIPE_DEPTH] = al.allocate<G::a_tile, LOAD_PIPE_DEPTH>();
    typename G::b_tile (&b_smem)[LOAD_PIPE_DEPTH] = al.allocate<G::b_tile, LOAD_PIPE_DEPTH>();
    typename G::d_tile (&d_smem) = al.allocate<G::d_tile>();

    tensor_allocator<1, CLUSTER_SIZE, false> tm_alloc{};
    using d_tt_t = tt<float, BM/CLUSTER_SIZE, BN>;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore inputs_arrived[LOAD_PIPE_DEPTH], inputs_finished[LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived, outputs_finished;

    const bool is_producer = (wg_id == 1);
    const bool is_consumer = (wg_id == 0);

    // Initialize all semaphores ONCE
    if (threadIdx.x == 0) {
        init_semaphore(tmem_provisioned, 0, 1);
        #pragma unroll
        for (int i = 0; i < LOAD_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, 1);
    }
    uint32_t bitfield = 0xFFFF0000;
    everyone::tma::cluster::arrive_aligned();

    // ===================== PRODUCER (warpgroup 1) =====================
    if (wg_id == 1) {
        warpgroup::decrease_registers<56>();
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            int input_ring = 0;
            int outputs_finished_phase = 0;
            for (int tile = cluster_start; tile < g.total_tiles; tile += g.num_clusters) {
                // Wait for previous epilogue to finish reading TMEM (not first tile)
                if (tile != cluster_start) {
                    wait(outputs_finished, outputs_finished_phase);
                    outputs_finished_phase ^= 1;
                }
                int cluster_row = tile / num_col_tiles;
                int cluster_col = tile % num_col_tiles;
                for (int k_iter = 0; k_iter < num_k_tiles; k_iter++) {
                    wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                    tma::cluster::load_async(a_smem[input_ring], g.A, {0, 0, cluster_row*2 + cta_rank, k_iter}, inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    tma::cluster::load_async(b_smem[input_ring], g.B, {0, 0, cluster_col*2 + cta_rank, k_iter}, inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    update_phasebit<1>(bitfield, input_ring);
                    input_ring = ring_advance<LOAD_PIPE_DEPTH>(input_ring);
                }
            }
        }
    }
    // ===================== CONSUMER (warpgroup 0) =====================
    else if (wg_id == 0) {
        if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_alloc.set_addr(tmem_addr);
            d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(0);

            int input_ring = 0;
            int outputs_finished_phase = 0;
            for (int tile = cluster_start; tile < g.total_tiles; tile += g.num_clusters) {
                // Wait for previous epilogue to finish reading TMEM (not first tile)
                if (tile != cluster_start) {
                    wait(outputs_finished, outputs_finished_phase);
                    outputs_finished_phase ^= 1;
                }
                for (int k_iter = 0; k_iter < num_k_tiles; k_iter++) {
                    tma::expect_bytes(inputs_arrived[input_ring],
                        CLUSTER_SIZE * size_bytes<typeof(a_smem[0])> +
                        CLUSTER_SIZE * size_bytes<typeof(b_smem[0])>);
                    wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                    if (k_iter == 0) mm2_ABt(d_tt, a_smem[input_ring], b_smem[input_ring], inputs_finished[input_ring]);
                    else             mma2_ABt(d_tt, a_smem[input_ring], b_smem[input_ring], inputs_finished[input_ring]);
                    update_phasebit<0>(bitfield, input_ring);
                    input_ring = ring_advance<LOAD_PIPE_DEPTH>(input_ring);
                }
                detail::tcgen05::commit<CLUSTER_SIZE>(outputs_arrived);
            }
        }
    }
    // ===================== EPILOGUE (warpgroup 2) =====================
    else {
        warpgroup::increase_registers<224>();
        everyone::tma::cluster::wait_aligned();

        // Provision TMEM once
        if (warpgroup::warpid() == 0) {
            tm_alloc.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_alloc.set_addr(tmem_addr);
        d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(0);

        int outputs_arrived_phase = 0;
        for (int tile = cluster_start; tile < g.total_tiles; tile += g.num_clusters) {
            int cluster_row = tile / num_col_tiles;
            int cluster_col = tile % num_col_tiles;

            wait(outputs_arrived, outputs_arrived_phase);
            outputs_arrived_phase ^= 1;

            rt_bf<BM/CLUSTER_SIZE/4, BN> d_reg;
            warpgroup::load_async(d_reg, d_tt);
            tensor_load_wait();
            if (warpgroup::warpid() == 0) warp::arrive(outputs_finished);  // TMEM is free for next tile's MMA
            warpgroup::store(d_smem, d_reg);
            warpgroup::sync(1);
            if (warpgroup::laneid() == 0) {
                tma::store_async(g.D, d_smem, {0, 0, cluster_row*2 + cta_rank, cluster_col});
                tma::store_async_read_wait();
            }
            warpgroup::sync(1);
        }

        // Deprovision TMEM once
        if (warpgroup::warpid() == 0) tm_alloc.deprovision();
    }
}

void matmul(bf16* A, bf16* B, bf16* C, size_t M, size_t N, size_t K) {
    matmul_globals::a_gl a_arg{A, nullptr, nullptr, M, K};
    matmul_globals::b_gl b_arg{B, nullptr, nullptr, N, K};
    matmul_globals::d_gl d_arg{C, nullptr, nullptr, M, N};

    int total_tiles = (M / BM) * (N / BN);
    int num_sms = 148;  // B200
    int num_clusters = min(total_tiles, num_sms / (int)CLUSTER_SIZE);

    matmul_globals g{a_arg, b_arg, d_arg, total_tiles, num_clusters};

    dim3 grid(num_clusters * CLUSTER_SIZE);
    dim3 block(NUM_THREADS);
    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    LaunchConfig<true, false> launch_config(grid, block, mem_size, 0, CLUSTER_SIZE);
    cudaLaunchKernelEx(launch_config, kernel, g);
}

#include "launch.cu"
