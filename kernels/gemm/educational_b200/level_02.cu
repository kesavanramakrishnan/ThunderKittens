// Level 02: Tensor Memory (TMEM) + tcgen05 MMA
// ===============================================
// Replace register accumulators with TMEM. Use tcgen05 MMA instructions
// (mm2_ABt / mma2_ABt) instead of warpgroup::mma_ABt.
//
// New concepts:
//   - tensor_allocator for TMEM allocation
//   - tt<float, M, N> tensor tiles (accumulators live in TMEM, not registers)
//   - mm2_ABt (first iteration: zero + multiply) vs mma2_ABt (accumulate)
//   - tm_alloc.provision() / deprovision() lifecycle
//   - tmem_provisioned semaphore to broadcast TMEM address
//   - detail::tcgen05::commit to signal MMA completion
//   - Epilogue: TMEM -> registers -> shared memory -> TMA store
//
// Tile: 128x128 output per CTA, single warpgroup consumer
// Layout: A is (M, K) row-major, B is (N, K) row-major, D = A * B^T

#include "kittens.cuh"
using namespace kittens;

static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 64;

static constexpr int NUM_WARPS = 4;
static constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

struct matmul_globals {
    using a_tile = st_bf<BM, BK>;
    using b_tile = st_bf<BN, BK>;
    using d_tile = st_bf<BM, BN>;
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
    int row = blockIdx.y;
    int col = blockIdx.x;

    // Shared memory allocation
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    typename G::a_tile (&a_smem) = al.allocate<G::a_tile>();
    typename G::b_tile (&b_smem) = al.allocate<G::b_tile>();
    typename G::d_tile (&d_smem) = al.allocate<G::d_tile>();

    // TMEM allocation
    tensor_allocator<1, 1, false> tm_alloc{};
    using d_tt_t = tt<float, BM, BN>;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore inputs_arrived, inputs_finished, outputs_arrived;

    // Initialize semaphores (single thread)
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);   // 1 TMA transaction group
        init_semaphore(inputs_finished, 0, 1);  // 1 MMA signals done reading smem
        init_semaphore(outputs_arrived, 0, 1);  // commit signals MMA results ready
    }
    // Provision TMEM (entire warp 0 must participate)
    if (warpid() == 0) {
        tm_alloc.provision(tmem_addr);
    }
    __syncthreads();

    // All threads: set up allocator with the provisioned address
    tm_alloc.set_addr(tmem_addr);
    d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(0);

    // K-loop: TMA load then MMA (fully serial, no overlap)
    for (int k_iter = 0; k_iter < num_k_tiles; k_iter++) {
        // Wait for previous MMA to finish reading smem (skip first iter)
        if (k_iter > 0) {
            wait(inputs_finished, (k_iter - 1) % 2);
        }

        // TMA load (single thread only)
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            tma::expect_bytes(
                inputs_arrived,
                size_bytes<typeof(a_smem)> +
                size_bytes<typeof(b_smem)>
            );
            tma::load_async(a_smem, g.A, {0, 0, row, k_iter}, inputs_arrived);
            tma::load_async(b_smem, g.B, {0, 0, col, k_iter}, inputs_arrived);
        
        wait(inputs_arrived, k_iter % 2);

        // MMA: shared -> TMEM (single thread for tcgen05)
        // sem version: signals inputs_finished when MMA is done reading smem
        // if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            if (k_iter == 0) mm_ABt(d_tt, a_smem, b_smem, inputs_finished);
            else             mma_ABt(d_tt, a_smem, b_smem, inputs_finished);
        }
    }

    // Commit: flush MMA pipeline, signal outputs_arrived when TMEM is ready
    if (threadIdx.x == 0) {
        detail::tcgen05::commit<1>(outputs_arrived);
    }
    wait(outputs_arrived, 0);

    // Epilogue: TMEM -> registers -> shared -> global
    rt_bf<BM/4, BN> d_reg;
    warpgroup::load_async(d_reg, d_tt);
    tensor_load_wait();
    warpgroup::store(d_smem, d_reg);
    warpgroup::sync(1);
    if (warpgroup::laneid() == 0) {
        tma::store_async(g.D, d_smem, {0, 0, row, col});
        tma::store_async_read_wait();
    }

    // Free TMEM (entire warp must participate)
    warpgroup::sync(1);
    if (warpid() == 0) tm_alloc.deprovision();
}

void matmul(bf16* A, bf16* B, bf16* C, size_t M, size_t N, size_t K) {
    matmul_globals::a_gl a_arg{A, nullptr, nullptr, M, K};
    matmul_globals::b_gl b_arg{B, nullptr, nullptr, N, K};
    matmul_globals::d_gl d_arg{C, nullptr, nullptr, M, N};
    matmul_globals g{a_arg, b_arg, d_arg};

    dim3 grid(N / BN, M / BM);
    dim3 block(NUM_THREADS);
    unsigned long mem_size = 200000;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<grid, block, mem_size>>>(g);
}

#include "launch.cu"
