// Level 04: Separate Epilogue Warpgroup
// =======================================
// Split the CTA into 3 roles: producer (loads), consumer (MMA), epilogue (stores).
// The consumer writes MMA results to TMEM, the epilogue reads TMEM and stores to
// global memory. This allows overlapping the next task's MMA with the current store.
//
// New concepts:
//   - 3 warpgroups: producer (loads), consumer (MMA), epilogue (stores)
//   - outputs_arrived: consumer signals epilogue that TMEM accumulator is ready
//   - outputs_finished: epilogue signals consumer that TMEM buffer is free to reuse
//   - warpgroup::load_async from TMEM subtile into registers
//   - warpgroup::store from registers to SMEM
//   - warpgroup::tma::store_async from SMEM to global
//   - increase_registers / decrease_registers for register budget balancing
//
// Tile: 256x128 output per cluster (128 rows per CTA), CLUSTER_SIZE=2
// Layout: A is (M, K) row-major, B is (N, K) row-major, D = A * B^T

#include "kittens.cuh"
using namespace kittens;

static constexpr int BM = 256;
static constexpr int BN = 128;
static constexpr int BK = 64;
static constexpr int QSIZE = 2;
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
    // TODO: Implement
    //
    // Key differences from Level 03:
    //
    // 1. Three warpgroup roles (by warpgroup::groupid()):
    //    - groupid 0: consumer (MMA) — only on CTA 0
    //    - groupid 1: producer (TMA loads)
    //    - groupid 2 (the else branch): epilogue (stores) — all CTAs
    //
    // 2. New semaphores:
    //    outputs_arrived[1]: consumer -> epilogue (TMEM result ready)
    //    outputs_finished[1]: epilogue -> consumer (TMEM buffer free)
    //
    // 3. Consumer warpgroup:
    //    - wait(outputs_finished) before starting new task's MMA
    //    - K-loop: mm2_ABt / mma2_ABt as before
    //    - After K-loop: detail::tcgen05::commit<CLUSTER_SIZE>(outputs_arrived)
    //
    // 4. Epilogue warpgroup:
    //    - Provisions TMEM, signals tmem_provisioned
    //    - wait(outputs_arrived) to know TMEM is ready
    //    - warpgroup::load_async(d_reg, d_tt.subtile(...)) // TMEM -> registers
    //    - tensor_load_wait()
    //    - arrive(outputs_finished) // free TMEM for next MMA
    //    - warpgroup::store(d_smem, d_reg) // registers -> SMEM
    //    - warpgroup::tma::store_async(...) // SMEM -> global
    //    - tm_alloc.deprovision() at end
    //
    // 5. Register budget:
    //    - Producer: decrease_registers<56>
    //    - Epilogue: increase_registers<224>
}

void matmul(bf16* A, bf16* B, bf16* C, size_t M, size_t N, size_t K) {
    matmul_globals::a_gl a_arg{A, nullptr, nullptr, (int)M, (int)K};
    matmul_globals::b_gl b_arg{B, nullptr, nullptr, (int)N, (int)K};
    matmul_globals::d_gl d_arg{C, nullptr, nullptr, (int)M, (int)N};
    matmul_globals g{a_arg, b_arg, d_arg};

    dim3 grid(M / BM * N / BN * CLUSTER_SIZE);
    dim3 block(NUM_THREADS);
    unsigned long mem_size = 200000;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    LaunchConfig<false, false> launch_config(grid, block, mem_size, 0, CLUSTER_SIZE);
    cudaLaunchKernelEx(launch_config, kernel, g);
}

#include "launch.cu"
