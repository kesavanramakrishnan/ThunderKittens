// Level 05: Deeper Load Pipeline
// ================================
// Go from double buffering (QSIZE=2) to a deeper pipeline (LOAD_PIPE_DEPTH=4+).
// Use a ring buffer with bitfield-based phase tracking for semaphores.
//
// New concepts:
//   - LOAD_PIPE_DEPTH > 2 ring buffer for input tiles
//   - ring_advance<LOAD_PIPE_DEPTH>(ring_idx) to advance ring index
//   - Bitfield phase tracking: uint32_t bitfield = 0xFFFF0000
//     * Upper 16 bits = finished phases (start at 1)
//     * Lower 16 bits = arrived phases (start at 0)
//   - get_phasebit<bit_offset>(bitfield, ring_idx) to read current phase
//   - update_phasebit<bit_offset>(bitfield, ring_idx) to flip phase
//   - tma::expect_bytes() to tell mbarrier how many bytes to expect
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
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {
    // TODO: Implement
    //
    // Key differences from Level 04:
    //
    // 1. Shared memory now has LOAD_PIPE_DEPTH slots:
    //    a_smem[LOAD_PIPE_DEPTH], b_smem[LOAD_PIPE_DEPTH]
    //
    // 2. Semaphores per pipeline slot:
    //    inputs_arrived[LOAD_PIPE_DEPTH], inputs_finished[LOAD_PIPE_DEPTH]
    //
    // 3. Phase tracking via bitfield:
    //    uint32_t bitfield = 0xFFFF0000;
    //    // Producer uses get_phasebit<1>(bitfield, ring) for inputs_finished
    //    // Consumer uses get_phasebit<0>(bitfield, ring) for inputs_arrived
    //    // After each wait, call update_phasebit<bit>(bitfield, ring) to flip
    //
    // 4. Producer loop:
    //    int input_ring = 0;
    //    for (int idx = 0; idx < iters_per_task; idx++) {
    //        wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
    //        tma::cluster::load_async(..., inputs_arrived[input_ring], ...);
    //        update_phasebit<1>(bitfield, input_ring);
    //        input_ring = ring_advance<LOAD_PIPE_DEPTH>(input_ring);
    //    }
    //
    // 5. Consumer loop:
    //    int input_ring = 0;
    //    for (int idx = 0; idx < iters_per_task; idx++) {
    //        tma::expect_bytes(inputs_arrived[input_ring], ...);
    //        wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
    //        mm2_ABt / mma2_ABt(...)
    //        // mma arrives on inputs_finished automatically via last arg
    //        update_phasebit<0>(bitfield, input_ring);
    //        input_ring = ring_advance<LOAD_PIPE_DEPTH>(input_ring);
    //    }
}

void matmul(bf16* A, bf16* B, bf16* C, size_t M, size_t N, size_t K) {
    matmul_globals::a_gl a_arg{A, nullptr, nullptr, (int)M, (int)K};
    matmul_globals::b_gl b_arg{B, nullptr, nullptr, (int)N, (int)K};
    matmul_globals::d_gl d_arg{C, nullptr, nullptr, (int)M, (int)N};
    matmul_globals g{a_arg, b_arg, d_arg};

    dim3 grid(M / BM * N / BN * CLUSTER_SIZE);
    dim3 block(NUM_THREADS);
    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    LaunchConfig<false, false> launch_config(grid, block, mem_size, 0, CLUSTER_SIZE);
    cudaLaunchKernelEx(launch_config, kernel, g);
}

#include "launch.cu"
