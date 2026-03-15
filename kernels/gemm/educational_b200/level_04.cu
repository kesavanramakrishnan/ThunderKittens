// Level 04: Separate Epilogue Warpgroup + Double Buffering
// ==========================================================
// Split the CTA into 3 roles: producer (loads), consumer (MMA), epilogue (stores).
// The consumer writes MMA results to TMEM, the epilogue reads TMEM and stores to
// global memory. This decouples MMA from stores.
//
// New concepts (vs Level 03):
//   - 3 warpgroups: producer (loads), consumer (MMA), epilogue (stores)
//   - LOAD_PIPE_DEPTH=2 double-buffered input tiles
//   - outputs_arrived: consumer signals epilogue that TMEM accumulator is ready
//   - tmem_provisioned: epilogue signals consumer that TMEM is allocated
//   - increase_registers / decrease_registers for register budget balancing
//   - warpgroup::load_async from TMEM into registers
//   - warpgroup::store from registers to SMEM, then TMA store to global
//
// Tile: 256x128 output per cluster (128 rows per CTA), CLUSTER_SIZE=2
// Layout: A is (M, K) row-major, B is (N, K) row-major, D = A * B^T

#include "kittens.cuh"
using namespace kittens;

static constexpr int BM = 256;
static constexpr int BN = 128;
static constexpr int BK = 64;
static constexpr int LOAD_PIPE_DEPTH = 2;
static constexpr int CLUSTER_SIZE = 2;

static constexpr int NUM_CONSUMER_WARPGROUPS = 1;  // MMA only
static constexpr int NUM_EPILOGUE_WARPGROUPS = 1;  // store only
static constexpr int NUM_PRODUCER_WARPGROUPS = 1;  // loads
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
        
    const int wg_id = warpgroup::groupid();  // 0, 1, or 2


    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    typename G::a_tile (&a_smem)[LOAD_PIPE_DEPTH] = al.allocate<G::a_tile, LOAD_PIPE_DEPTH>();
    typename G::b_tile (&b_smem)[LOAD_PIPE_DEPTH] = al.allocate<G::b_tile, LOAD_PIPE_DEPTH>();
    typename G::d_tile (&d_smem) = al.allocate<G::d_tile>();


    tensor_allocator<1, CLUSTER_SIZE, false> tm_alloc{};
    using d_tt_t = tt<float, BM/CLUSTER_SIZE, BN>;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore inputs_arrived[LOAD_PIPE_DEPTH], inputs_finished[LOAD_PIPE_DEPTH], outputs_arrived;

    if (threadIdx.x == 0) {
        init_semaphore(tmem_provisioned, 0, 1);
        for(int i = 0; i < LOAD_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 1);   // 1 TMA transaction group
            init_semaphore(inputs_finished[i], 0, 1);  // 1 MMA signals done reading smem
        }
        init_semaphore(outputs_arrived, 0, 1);  // commit signals MMA results ready
    }
    // Cluster arrive: signal init is done, each branch waits when needed
    everyone::tma::cluster::arrive_aligned();

    // ===================== PRODUCER (warpgroup 1) =====================
    // Issues TMA loads into double-buffered smem. Single thread does TMA.
    if (wg_id == 1) {
        warpgroup::decrease_registers<56>();
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            everyone::tma::cluster::wait();  // wait for semaphore init
            for (int k_iter = 0; k_iter < num_k_tiles; k_iter++) {
                int q = k_iter % LOAD_PIPE_DEPTH;
                if (k_iter >= LOAD_PIPE_DEPTH) {
                    wait(inputs_finished[q], (k_iter / LOAD_PIPE_DEPTH) - 1);
                }
                if (cta_rank == 0) {
                    tma::expect_bytes(
                        inputs_arrived[q],
                        CLUSTER_SIZE * size_bytes<typeof(a_smem[0])> +
                        CLUSTER_SIZE * size_bytes<typeof(b_smem[0])>
                    );
                }
                tma::cluster::load_async(a_smem[q], g.A, {0, 0, cluster_row*2 + cta_rank, k_iter}, inputs_arrived[q], (uint16_t)(1 << cta_rank), 0);
                tma::cluster::load_async(b_smem[q], g.B, {0, 0, cluster_col*2 + cta_rank, k_iter}, inputs_arrived[q], (uint16_t)(1 << cta_rank), 0);
            }
        }
    }
    // ===================== CONSUMER (warpgroup 0) =====================
    // Waits for data, issues MMA. Only CTA 0 runs MMA (mm2 reads both CTAs via DSMEM).
    // After K-loop, commits to signal epilogue.
    else if (wg_id == 0) {
        if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
            everyone::tma::cluster::wait();  // wait for semaphore init
            wait(tmem_provisioned, 0);       // wait for epilogue to provision TMEM
            tm_alloc.set_addr(tmem_addr);
            d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(0);

            for (int k_iter = 0; k_iter < num_k_tiles; k_iter++) {
                int q = k_iter % LOAD_PIPE_DEPTH;
                wait(inputs_arrived[q], k_iter / LOAD_PIPE_DEPTH);
                    if (k_iter == 0) mm2_ABt(d_tt, a_smem[q], b_smem[q], inputs_finished[q]);
                    else             mma2_ABt(d_tt, a_smem[q], b_smem[q], inputs_finished[q]);
            }
            // Flush MMA pipeline, multicast completion to both CTAs
                detail::tcgen05::commit<CLUSTER_SIZE>(outputs_arrived);
            }
        // CTA 1's consumer warpgroup has no work
    }
    // ===================== EPILOGUE (warpgroup 2) =====================
    // Provisions TMEM, waits for MMA results, stores to global memory, deprovisions.
    else {
        warpgroup::increase_registers<224>();
        everyone::tma::cluster::wait_aligned();  // wait for semaphore init

        // Epilogue warpgroup owns TMEM lifecycle
        if (warpgroup::warpid() == 0) {
            tm_alloc.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_alloc.set_addr(tmem_addr);
        d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(0);

        // Wait for MMA results (commit<2> multicasts to both CTAs)
        wait(outputs_arrived, 0);

        // TMEM -> registers -> shared -> global
        rt_bf<BM/CLUSTER_SIZE/4, BN> d_reg;
        warpgroup::load_async(d_reg, d_tt);
        tensor_load_wait();
        warpgroup::store(d_smem, d_reg);
        warpgroup::sync(1);
        if (warpgroup::laneid() == 0) {
            tma::store_async(g.D, d_smem, {0, 0, cluster_row*2 + cta_rank, cluster_col});
            tma::store_async_read_wait();
        }

        // Free TMEM (entire warp must participate)
        warpgroup::sync(1);
        if (warpgroup::warpid() == 0) tm_alloc.deprovision();
    }
}

void matmul(bf16* A, bf16* B, bf16* C, size_t M, size_t N, size_t K) {
    matmul_globals::a_gl a_arg{A, nullptr, nullptr, M, K};
    matmul_globals::b_gl b_arg{B, nullptr, nullptr, N, K};
    matmul_globals::d_gl d_arg{C, nullptr, nullptr, M, N};
    matmul_globals g{a_arg, b_arg, d_arg};

    dim3 grid(M / BM * N / BN * CLUSTER_SIZE);
    dim3 block(NUM_THREADS);
    unsigned long mem_size = 200000;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    LaunchConfig<true, false> launch_config(grid, block, mem_size, 0, CLUSTER_SIZE);
    cudaLaunchKernelEx(launch_config, kernel, g);
}

#include "launch.cu"
