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
    // TODO: Implement
    //
    // Key differences from Level 01:
    //
    // 1. TMEM allocation:
    //    tensor_allocator<1, 1, false> tm_alloc{};
    //    using d_tt_t = tt<float, BM, BN>;  // accumulator in TMEM
    //    __shared__ uint32_t tmem_addr;
    //    __shared__ semaphore tmem_provisioned;

    using G = matmul_globals;

    if (threadIdx.x == 0) {
        g.a.template prefetch_tma<typename G::a_tile>(); // prefetch tensor maps at start of kernel 
        g.b.template prefetch_tma<typename G::b_tile>();
        g.d.template prefetch_tma<typename G::d_tile>();
    }

    const int iters_per_task = g.a.cols() / BK; // number of iterations of mma_AB per task
    const int rblks = g.d.rows() / BM;
    const int cblks = g.d.cols() / BN;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    typename G::a_tile (&a_smem) = al.allocate<G::a_tile>();
    typename G::b_tile (&b_smem) = al.allocate<G::b_tile>();
    typename G::d_tile (&d_smem) = al.allocate<G::d_tile>();

    tensor_allocator<1, 1, false> tm_alloc{};
    using d_tt_t = tt<float, BM, BN>;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    int tile_coord_row = blockIdx.x;
    int tile_coord_col = blockIdx.y;

    //
    // 2. Epilogue warpgroup provisions TMEM:
    //    tm_alloc.provision(tmem_addr);
    //    warp::arrive(tmem_provisioned);

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        tm_alloc.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }

    __syncthreads();

    // TODO: Implement consumer logic
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);
    d_tt = tm_alloc.allocate<d_tt_t>(0, 0);

    for (int k_iter = 0; k_iter < iters_per_task; k_iter++) {
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            tma::cluster::load_async(a_smem, g.a, {tile_coord_row, idx});
            wait(inputs_arrived)
        }
    }



    //
    // 3. Consumer warpgroup (CTA rank 0 only for tcgen05):
    //    - wait(tmem_provisioned)
    //    - d_tt = tm_alloc.allocate<d_tt_t>(...)
    //    - K-loop: mm2_ABt (idx==0) / mma2_ABt (idx>0) instead of warpgroup::mma_ABt
    //    - detail::tcgen05::commit<1>(outputs_arrived) to signal completion
    //
    // 4. Epilogue warpgroup:
    //    - wait(outputs_arrived)
    //    - warpgroup::load_async(d_reg, d_tt)  // TMEM -> registers (auto bf16 convert)
    //    - tensor_load_wait()
    //    - warpgroup::store(d_smem, d_reg)      // registers -> SMEM
    //    - warpgroup::tma::store_async(...)      // SMEM -> global
    //    - arrive(outputs_finished)
    //    - tm_alloc.deprovision()
}

void matmul(bf16* A, bf16* B, bf16* C, size_t M, size_t N, size_t K) {
    matmul_globals::a_gl a_arg{A, nullptr, nullptr, (int)M, (int)K};
    matmul_globals::b_gl b_arg{B, nullptr, nullptr, (int)N, (int)K};
    matmul_globals::d_gl d_arg{C, nullptr, nullptr, (int)M, (int)N};
    matmul_globals g{a_arg, b_arg, d_arg};

    dim3 grid(N / BN, M / BM);
    dim3 block(NUM_THREADS);
    unsigned long mem_size = 200000;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<grid, block, mem_size>>>(g);
}

#include "launch.cu"
