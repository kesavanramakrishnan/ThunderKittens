// Level 01: H100-style baseline on B200
// =======================================
// WGMMA + TMA + producer/consumer warpgroup split + double buffering.
// No B200-specific features yet — just a working GEMM using H100-era patterns.
//
// Concepts:
//   - warpgroup::mma_ABt (WGMMA on shared memory tiles)
//   - TMA loads with tma::load_async
//   - Producer/consumer warpgroup partitioning
//   - Semaphore-based synchronization (full/empty)
//   - Double buffering (QSIZE=2) to overlap loads with compute
//
// Tile: 128x128 output per CTA, single warpgroup consumer
// Layout: A is (M, K) row-major, B is (N, K) row-major, D = A * B^T

#include "kittens.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 64;

static constexpr int BM = 64;
static constexpr int BN = 256;
static constexpr int BK = 64;
static constexpr int QSIZE = 2; // double buffering

static constexpr int NUM_CONSUMER_WARPGROUPS = 1;
static constexpr int NUM_PRODUCER_WARPGROUPS = 1;
static constexpr int NUM_WARPS = (NUM_CONSUMER_WARPGROUPS + NUM_PRODUCER_WARPGROUPS) * 4;
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
    // 1. Allocate shared memory tiles:
    //    - a_smem[QSIZE], b_smem[QSIZE] for double-buffered inputs
    //    - d_smem for output staging
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]); // TODO: How does this work?

    typename matmul_globals::a_tile (&a_smem)[QSIZE] = al.allocate<matmul_globals::a_tile, QSIZE>();
    typename matmul_globals::b_tile (&b_smem)[QSIZE] = al.allocate<matmul_globals::b_tile, QSIZE>();
    typename matmul_globals::d_tile (&d_smem) = al.allocate<matmul_globals::d_tile>();
    //
    // 2. Allocate register accumulator (done inside consumer branch)

    int row = blockIdx.y; 
    int col = blockIdx.x; 

    const int warpid = kittens::warpid();
    const int warpgroupid = warpid/4;

    // Determine type of warp group
    bool is_producer = (warpgroupid == 0);
    bool is_consumer = (warpgroupid > 0);
    int num_k_tiles = g.A.cols() / BK;
    int num_consumers = NUM_CONSUMER_WARPGROUPS;
    
    // Consumer index (0-based) for consumer warp groups
    //
    // 3. Initialize semaphores: inputs_arrived[QSIZE], inputs_finished[QSIZE]
    __shared__ semaphore full[QSIZE], empty[QSIZE];

    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; i++) {
            init_semaphore(full[i], 0, 1);
            init_semaphore(empty[i], num_consumers, 0);
        }
    }
    __syncthreads();

    //
    // 4. Producer warpgroup (warpgroup 1):
    //    - decrease_registers to give registers to consumer
    //    - Single leader thread loops over K tiles:
    //      wait(inputs_finished[q]) -> tma::load_async A,B -> arrive(inputs_arrived[q])

    if (is_producer) {
        warpgroup::decrease_registers<40>();
        if (warpgroup::laneid() == 0) {
            int p = 0; int q_idx = 0;
            for (int i = 0; i < num_k_tiles; i++, q_idx++) {
                if (q_idx == QSIZE) { q_idx = 0; p ^= 1; }
                wait(empty[q_idx], p);
                tma::expect_bytes(
                    full[q_idx], 
                    size_bytes<typeof(a_smem[0])> +
                    size_bytes<typeof(b_smem[0])>
                );
                tma::load_async(a_smem[q_idx], g.A, {0, 0, row, i}, full[q_idx]);
                tma::load_async(b_smem[q_idx], g.B, {0, 0, col, i}, full[q_idx]);
            }
        }
    } else {
        warpgroup::increase_registers<232>();
        rt_fl<16, BN> d_reg;
        kittens::warp::zero(d_reg);

        if (warpgroup::laneid() == 0)
            for (int i = 0; i < QSIZE; ++i) arrive(empty[i], 1);

        int p = 0, q_idx = 0;
        for (int tile = 0; tile < num_k_tiles; ++tile, ++q_idx) {
            if (q_idx == QSIZE) { q_idx = 0; p ^= 1; }
            wait(full[q_idx], p);
            warpgroup::mma_ABt(
                d_reg,
                a_smem[q_idx],
                b_smem[q_idx]
            );
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) arrive(empty[q_idx], 1);
        }

        // Store accumulator to shared memory
        warpgroup::store(d_smem, d_reg);
        warpgroup::sync(warpgroupid + 4);

        // TMA store to global memory
        if (warpid % 4 == 0) {
            tma::store_async(g.D, d_smem, {0, 0, row, col});
            tma::store_async_read_wait();
        }
    }
}

void matmul(bf16* A, bf16* B, bf16* C, size_t M, size_t N, size_t K) {
    matmul_globals::a_gl a_arg{A, nullptr, nullptr, (int)M, (int)K};
    matmul_globals::b_gl b_arg{B, nullptr, nullptr, (int)N, (int)K};
    matmul_globals::d_gl d_arg{C, nullptr, nullptr, (int)M, (int)N};
    matmul_globals g{a_arg, b_arg, d_arg};

    dim3 grid(N / BN, M / BM);
    dim3 block(NUM_THREADS);
    unsigned long mem_size = 200000;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        attr_set = true;
    }
    kernel<<<grid, block, mem_size>>>(g);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Launch error: %s\n", cudaGetErrorString(err));
    }
}

#include "launch.cu"
