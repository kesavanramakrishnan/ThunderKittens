// Level 09: CLC (Cluster Launch Control)
// =========================================
// Builds on Level 08 (MMA pipeline depth + epilogue pipelining).
// Replaces the strided tile loop with hardware-scheduled persistent
// execution via CLC — the GPU's built-in work distribution mechanism.
//
// Why CLC instead of strided loops:
//   - Strided assignment is static — can't adapt to runtime load imbalance
//   - CLC dynamically assigns tiles to clusters as they become available
//   - CLC handles the "persistent kernel" pattern in hardware, so we don't
//     need to manage a global atomic counter or stride calculation
//   - CLC also enables dependent kernel launches for back-to-back matmuls
//
// New concepts (vs Level 08):
//   - clc::handle: shared memory object that receives scheduling info from HW
//   - clc::schedule(handle, sem): CTA 0 requests next tile from HW scheduler
//   - clc::query(handle): all CTAs read the assigned tile index
//     * Returns {success, x} — success=false means no more tiles
//   - schedule_arrived[CLC_PIPE_DEPTH]: HW signals when a new tile is assigned
//   - schedule_finished[CLC_PIPE_DEPTH]: kernel signals when done processing
//     * Arrival count = (2 + NUM_CONSUMER_WARPGROUPS) * CLUSTER_SIZE + NUM_CONSUMER_WARPGROUPS
//       because producer (2 warps: loader + scheduler), consumer(s), and
//       epilogue all need to acknowledge the schedule
//   - LaunchConfig<true, true>: cluster=true, CLC=true
//   - Grid size = total_tiles * CLUSTER_SIZE (HW schedules all tiles)
//
// Producer warpgroup now has 2 active warps:
//   - Warp 3: TMA loader (same K-loop as before, but inside CLC task loop)
//     * After K-loop, waits on schedule_arrived to get NEXT tile index
//     * Queries clc::handle for tile coords
//     * Signals schedule_finished when done reading handle
//     * Breaks when schedule.success == false
//   - Warp 2: CLC scheduler
//     * CTA 0 only: calls clc::schedule to request next tile
//     * Waits on schedule_finished (previous tile fully acknowledged)
//     * All CTAs: waits on schedule_arrived, queries handle, signals finished
//     * Breaks when schedule.success == false
//
// Consumer and epilogue also participate in CLC protocol:
//   - Both wait on schedule_arrived to get tile index
//   - Both signal schedule_finished after reading the handle
// Key structural change: the task loop is now `for (task_iter = 0; true; ...)`
// with a `break` when `!schedule.success`, instead of iterating over a
// known tile count with stride.
//
// Tile: 256x256 output per cluster, CLUSTER_SIZE=2
// Layout: A is (M, K) row-major, B is (N, K) row-major, D = A * B^T

#include "kittens.cuh"
using namespace kittens;

static constexpr int BM = 256;
static constexpr int BN = 256;
static constexpr int BK = 64;
static constexpr int LOAD_PIPE_DEPTH = 4;
static constexpr int EPI_PIPE_DEPTH = 4;
static constexpr int MMA_PIPE_DEPTH = 2;
static constexpr int CLC_PIPE_DEPTH = 1;
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
    typename G::d_tile (&d_smem)[NUM_D_TILES]     = al.allocate<G::d_tile, NUM_D_TILES>();

    tensor_allocator<1, CLUSTER_SIZE, false> tm_alloc{};
    using d_tt_t = tt<float, BM/CLUSTER_SIZE, BN>;
    using d_tt_chunk_t = tt<float, BM/CLUSTER_SIZE, BN/EPI_PIPE_DEPTH>;

    __shared__ clc::handle clc_handle[CLC_PIPE_DEPTH];

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned, tmem_finished;
    __shared__ semaphore inputs_arrived[LOAD_PIPE_DEPTH], inputs_finished[LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived[MMA_PIPE_DEPTH], outputs_finished[MMA_PIPE_DEPTH];
    __shared__ semaphore schedule_arrived[CLC_PIPE_DEPTH], schedule_finished[CLC_PIPE_DEPTH];

    // Initialize all semaphores ONCE
    if (threadIdx.x == 0) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(tmem_finished, 0, 1);
        #pragma unroll
        for (int i = 0; i < CLC_PIPE_DEPTH; i++) {
            init_semaphore(schedule_arrived[i], 0, 1);
            init_semaphore(schedule_finished[i], 0, (3)*CLUSTER_SIZE+1);
        }
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
    if (wg_id == 1) {
        warpgroup::decrease_registers<56>();

        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            int input_ring = 0;
            int tile_coord = cluster_start;
            everyone::tma::cluster::wait();
            for (int task_iter = 0; true; task_iter++) {
                int cluster_row = tile_coord / num_col_tiles;
                int cluster_col = tile_coord % num_col_tiles;
                for (int k_iter = 0; k_iter < num_k_tiles; k_iter++) {
                    wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                    tma::cluster::load_async(a_smem[input_ring], g.A, {0, 0, cluster_row*2 + cta_rank, k_iter}, inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    tma::cluster::load_async(b_smem[input_ring], g.B, {0, 0, cluster_col*2 + cta_rank, k_iter}, inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    update_phasebit<1>(bitfield, input_ring);
                    input_ring = ring_advance<LOAD_PIPE_DEPTH>(input_ring);
                }
                wait(schedule_arrived[task_iter % CLC_PIPE_DEPTH], (task_iter / CLC_PIPE_DEPTH) % 2);
                auto schedule = clc::query(clc_handle[task_iter % CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter % CLC_PIPE_DEPTH], 0);
                if (schedule.success) tile_coord = schedule.x / CLUSTER_SIZE;
                else break;
            }
        } else if (warpgroup::warpid() == 2 && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            for (int task_iter = 0; true; task_iter++) {
                if (cta_rank == 0) {
                    wait(schedule_finished[task_iter % CLC_PIPE_DEPTH], ((task_iter + CLC_PIPE_DEPTH) / CLC_PIPE_DEPTH) % 2);
                    clc::schedule(clc_handle[task_iter % CLC_PIPE_DEPTH], schedule_arrived[task_iter % CLC_PIPE_DEPTH]);
                }
                tma::expect_bytes(schedule_arrived[task_iter % CLC_PIPE_DEPTH], sizeof(clc_handle[task_iter % CLC_PIPE_DEPTH]));
                wait(schedule_arrived[task_iter % CLC_PIPE_DEPTH], (task_iter / CLC_PIPE_DEPTH) % 2);
                auto schedule = clc::query(clc_handle[task_iter % CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter % CLC_PIPE_DEPTH], 0);
                if (!schedule.success) break;
            }
        }
    }
    // ===================== CONSUMER (warpgroup 0) =====================
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
            for (int task_iter = 0; true; task_iter++) {
                wait(schedule_arrived[task_iter % CLC_PIPE_DEPTH], (task_iter / CLC_PIPE_DEPTH) % 2);
                auto schedule = clc::query(clc_handle[task_iter % CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter % CLC_PIPE_DEPTH], 0);
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
                if (!schedule.success) break;
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
        d_tt_t d_tt[MMA_PIPE_DEPTH];
        #pragma unroll
        for (int i = 0; i < MMA_PIPE_DEPTH; i++) {
            d_tt[i] = tm_alloc.allocate<d_tt_t>(i * BN);
        }

        int tile_coord = cluster_start;
        for (int task_iter = 0; true; task_iter++) {
            int cluster_row = tile_coord / num_col_tiles;
            int cluster_col = tile_coord % num_col_tiles;

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

            // Read CLC schedule for next tile
            wait(schedule_arrived[task_iter % CLC_PIPE_DEPTH], (task_iter / CLC_PIPE_DEPTH) % 2);
            auto schedule = clc::query(clc_handle[task_iter % CLC_PIPE_DEPTH]);
            warpgroup::sync(1);
            warpgroup::tma::cluster::arrive(schedule_finished[task_iter % CLC_PIPE_DEPTH], 0);
            if (schedule.success) tile_coord = schedule.x / CLUSTER_SIZE;
            else break;
        }

        // Sync both CTAs before deprovision (cluster-scope operation)
        if (warpgroup::warpid() == 0) {
            if (warp::elect_leader()) tma::cluster::arrive(tmem_finished, 1-cta_rank);
            wait(tmem_finished, 0);
            tm_alloc.deprovision();
        }
    }
}

void matmul(bf16* A, bf16* B, bf16* C, size_t M, size_t N, size_t K) {
    matmul_globals::a_gl a_arg{A, nullptr, nullptr, M, K};
    matmul_globals::b_gl b_arg{B, nullptr, nullptr, N, K};
    matmul_globals::d_gl d_arg{C, nullptr, nullptr, M, N};
    matmul_globals g{a_arg, b_arg, d_arg};

    int total_tiles = (M / BM) * (N / BN);

    dim3 grid(total_tiles * CLUSTER_SIZE);
    dim3 block(NUM_THREADS);
    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    LaunchConfig<true, true> launch_config(grid, block, mem_size, 0, CLUSTER_SIZE);
    cudaLaunchKernelEx(launch_config, kernel, g);
}

#include "launch.cu"
