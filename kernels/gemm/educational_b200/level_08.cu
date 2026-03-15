// Level 08: MMA Pipeline Depth
// ================================
// Builds on Level 07 (epilogue pipelining + naive persistent kernel).
// Adds MMA_PIPE_DEPTH=2 to double-buffer TMEM accumulators, allowing
// the consumer to start the next tile's MMA while the epilogue is still
// reading/storing the current tile's results.
//
// Why this helps:
//   - In Level 07, the consumer must wait for outputs_finished before
//     starting MMA for the next tile — TMEM is occupied by the epilogue
//   - With MMA_PIPE_DEPTH=2, two TMEM accumulators alternate:
//     * Consumer writes tile N into d_tt[0], commits
//     * Consumer immediately starts tile N+1 into d_tt[1] (no wait!)
//     * Epilogue reads d_tt[0] concurrently with tile N+1's MMA
//     * Consumer waits for d_tt[0]'s epilogue only before tile N+2
//
// New concepts (vs Level 07):
//   - MMA_PIPE_DEPTH = 2: number of TMEM accumulators
//   - d_tt[MMA_PIPE_DEPTH]: array of accumulators at different TMEM offsets
//     * d_tt[i] = tm_alloc.allocate<d_tt_t>(i * BN)
//   - outputs_finished[MMA_PIPE_DEPTH]: one semaphore per accumulator
//   - outputs_arrived[MMA_PIPE_DEPTH]: one semaphore per accumulator
//   - Consumer uses d_tt[task_iter % MMA_PIPE_DEPTH]
//   - Consumer waits on outputs_finished[task_iter % MMA_PIPE_DEPTH]
//     (first MMA_PIPE_DEPTH tiles pass immediately via bitfield)
//   - Epilogue reads d_tt[task_iter % MMA_PIPE_DEPTH] in chunks
//   - Epilogue signals outputs_finished[task_iter % MMA_PIPE_DEPTH]
//     after the last chunk is read from that accumulator
//
// Timeline with MMA_PIPE_DEPTH=2:
//   Tile 0: MMA → d_tt[0], commit(outputs_arrived[0])
//   Tile 1: MMA → d_tt[1], commit(outputs_arrived[1])  ← no wait!
//           Epilogue reads d_tt[0], signals outputs_finished[0]
//   Tile 2: wait(outputs_finished[0]), MMA → d_tt[0]
//           Epilogue reads d_tt[1], signals outputs_finished[1]
//
// Tile: 256x128 output per cluster, CLUSTER_SIZE=2
// Layout: A is (M, K) row-major, B is (N, K) row-major, D = A * B^T

#include "kittens.cuh"
using namespace kittens;

static constexpr int BM = 256;
static constexpr int BN = 256;
static constexpr int BK = 64;
static constexpr int LOAD_PIPE_DEPTH = 4;
static constexpr int EPI_PIPE_DEPTH = 4;
static constexpr int MMA_PIPE_DEPTH = 2;
static constexpr int CLUSTER_SIZE = 2;

static constexpr int NUM_D_TILES = EPI_PIPE_DEPTH > 1 ? 2 : 1;

static constexpr int NUM_CONSUMER_WARPGROUPS = 1;
static constexpr int NUM_EPILOGUE_WARPGROUPS = 1;
static constexpr int NUM_PRODUCER_WARPGROUPS = 1;
static constexpr int NUM_WARPS = (NUM_CONSUMER_WARPGROUPS + NUM_EPILOGUE_WARPGROUPS + NUM_PRODUCER_WARPGROUPS) * 4;
static constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

struct matmul_globals {
    using a_tile = st_bf<BM/2, BK>;
    using b_tile = st_bf<BN/2, BK>;
    using d_tile = st_bf<BM/2, BN/EPI_PIPE_DEPTH>;
    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;
    a_gl A;
    b_gl B;
    d_gl D;
    int total_tiles;
    int num_clusters;
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {
    // TODO: Implement
    //
    // Same structure as Level 07 (3 warpgroups, persistent strided loop,
    // epilogue pipelining), but now with MMA_PIPE_DEPTH=2 accumulators.
    //
    // Key differences from Level 07:
    //
    // 1. TMEM: allocate MMA_PIPE_DEPTH accumulators at different offsets:
    //    d_tt_t d_tt[MMA_PIPE_DEPTH];
    //    d_tt[0] = tm_alloc.allocate<d_tt_t>(0 * BN);
    //    d_tt[1] = tm_alloc.allocate<d_tt_t>(1 * BN);
    //
    // 2. Semaphores:
    //    outputs_arrived[MMA_PIPE_DEPTH] — one per accumulator
    //    outputs_finished[MMA_PIPE_DEPTH] — one per accumulator
    //    init_semaphore(outputs_finished[i], 0, CLUSTER_SIZE);
    //      (CLUSTER_SIZE because both CTAs' epilogues signal independently)
    //
    // 3. Producer:
    //    - Tracks mma_ring = 0 (indexes into MMA_PIPE_DEPTH)
    //    - wait(outputs_finished[mma_ring], get_phasebit<1>(bitfield, mma_ring))
    //      (first MMA_PIPE_DEPTH tiles pass immediately)
    //    - After K-loop: update_phasebit<1>(bitfield, mma_ring)
    //    - mma_ring = ring_advance<MMA_PIPE_DEPTH>(mma_ring)
    //
    // 4. Consumer:
    //    - Same mma_ring tracking
    //    - wait(outputs_finished[mma_ring], ...) before MMA
    //    - MMA into d_tt[mma_ring]
    //    - commit(outputs_arrived[mma_ring]) after K-loop
    //    - update_phasebit + advance mma_ring
    //
    // 5. Epilogue:
    //    - Tracks mma_ring for which accumulator to read
    //    - wait(outputs_arrived[mma_ring], ...) 
    //    - Chunked read from d_tt[mma_ring] via subtile (same as Level 07)
    //    - Signal outputs_finished[mma_ring] after last chunk
    //    - Advance mma_ring
    //
    // The key insight: with MMA_PIPE_DEPTH=2, the consumer can start
    // MMA for tile N+1 into d_tt[1] immediately after committing tile N
    // into d_tt[0], without waiting for the epilogue to finish reading
    // d_tt[0]. The wait only happens when wrapping back to d_tt[0].
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
    typename G::d_tile (&d_smem)[NUM_D_TILES]     = al.allocate<G::d_tile, NUM_D_TILES>();

    tensor_allocator<1, CLUSTER_SIZE, false> tm_alloc{};
    using d_tt_t = tt<float, BM/CLUSTER_SIZE, BN>;
    using d_tt_chunk_t = tt<float, BM/CLUSTER_SIZE, BN/EPI_PIPE_DEPTH>;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore inputs_arrived[LOAD_PIPE_DEPTH], inputs_finished[LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived[MMA_PIPE_DEPTH], outputs_finished[MMA_PIPE_DEPTH];

    // Initialize all semaphores ONCE
    if (threadIdx.x == 0) {
        init_semaphore(tmem_provisioned, 0, 1);
        #pragma unroll
        for (int i = 0; i < LOAD_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < MMA_PIPE_DEPTH; i++) {
            init_semaphore(outputs_arrived[i], 0, 1);
            init_semaphore(outputs_finished[i], 0, 1);
        }
    }
    uint32_t bitfield = 0xFFFF0000;
    everyone::tma::cluster::arrive_aligned();

    // ===================== PRODUCER (warpgroup 1) =====================
    // Identical to Level 06 — epilogue pipelining doesn't affect the producer.
    if (wg_id == 1) {
        warpgroup::decrease_registers<56>();
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            int input_ring = 0;
            for (int task_iter = 0; task_iter < g.total_tiles; task_iter++) {
                int tile = cluster_start + task_iter * g.num_clusters;
                if (tile >= g.total_tiles) break;
                wait(outputs_finished[task_iter % MMA_PIPE_DEPTH], ((task_iter + MMA_PIPE_DEPTH) / MMA_PIPE_DEPTH) % 2);
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
    // Identical to Level 06 — one TMEM accumulator, same K-loop.
    else if (wg_id == 0) {
        if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_alloc.set_addr(tmem_addr);
            d_tt_t d_tt[MMA_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < MMA_PIPE_DEPTH; i++) {
                d_tt[i] = tm_alloc.allocate<d_tt_t>(i * BN);
            }

            int input_ring = 0;
            for (int task_iter = 0; task_iter < g.total_tiles; task_iter++) {
                int tile = cluster_start + task_iter * g.num_clusters;
                if (tile >= g.total_tiles) break;
                wait(outputs_finished[task_iter % MMA_PIPE_DEPTH], ((task_iter + MMA_PIPE_DEPTH) / MMA_PIPE_DEPTH) % 2);
                for (int k_iter = 0; k_iter < num_k_tiles; k_iter++) {
                    tma::expect_bytes(inputs_arrived[input_ring],
                        CLUSTER_SIZE * size_bytes<typeof(a_smem[0])> +
                        CLUSTER_SIZE * size_bytes<typeof(b_smem[0])>);
                    wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                    if (k_iter == 0) mm2_ABt(d_tt[task_iter % MMA_PIPE_DEPTH], a_smem[input_ring], b_smem[input_ring], inputs_finished[input_ring]);
                    else             mma2_ABt(d_tt[task_iter % MMA_PIPE_DEPTH], a_smem[input_ring], b_smem[input_ring], inputs_finished[input_ring]);
                    update_phasebit<0>(bitfield, input_ring);
                    input_ring = ring_advance<LOAD_PIPE_DEPTH>(input_ring);
                }
                detail::tcgen05::commit<CLUSTER_SIZE>(outputs_arrived[task_iter % MMA_PIPE_DEPTH]);
            }
        }
    }
    // ===================== EPILOGUE (warpgroup 2) =====================
    // NEW: instead of loading the entire 128×128 at once, read EPI_PIPE_DEPTH
    // chunks of 128×(128/EPI_PIPE_DEPTH) using subtile, double-buffering SMEM.
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
        d_tt_t d_tt[MMA_PIPE_DEPTH];
        #pragma unroll
        for (int i = 0; i < MMA_PIPE_DEPTH; i++) {
            d_tt[i] = tm_alloc.allocate<d_tt_t>(i * BN);
        }

        for (int task_iter = 0; task_iter < g.total_tiles; task_iter++) {
            int tile = cluster_start + task_iter * g.num_clusters;
            if (tile >= g.total_tiles) break;
            int cluster_row = tile / num_col_tiles;
            int cluster_col = tile % num_col_tiles;

            wait(outputs_arrived[task_iter % MMA_PIPE_DEPTH], (task_iter / MMA_PIPE_DEPTH) % 2);

            rt_bf<BM/CLUSTER_SIZE/4, BN/EPI_PIPE_DEPTH> d_reg;
            #pragma unroll
            for (int i = 0; i < EPI_PIPE_DEPTH; i++) {
                warpgroup::load_async(d_reg, d_tt[task_iter % MMA_PIPE_DEPTH].template subtile<d_tt_chunk_t>(0, BN/EPI_PIPE_DEPTH * i));
                tensor_load_wait();
                if (i == EPI_PIPE_DEPTH - 1) {
                    if (warpgroup::warpid() == 0) warp::arrive(outputs_finished[task_iter % MMA_PIPE_DEPTH]);
                }
                warpgroup::tma::store_async_read_wait<NUM_D_TILES-1>();
                warpgroup::sync(1);
                warpgroup::store(d_smem[i % NUM_D_TILES], d_reg);
                warpgroup::sync(1);
                tma::store_async(g.D, d_smem[i % NUM_D_TILES], {0, 0, cluster_row*2 + cta_rank, EPI_PIPE_DEPTH * cluster_col + i});
            }
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
    int num_sms = 148;
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
