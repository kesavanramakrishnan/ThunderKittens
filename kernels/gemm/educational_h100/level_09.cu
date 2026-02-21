#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <chrono>
#include <cuda_runtime.h>

#include "kittens.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 64;
constexpr int M_BLOCK = 2;  // Number of consumer warp groups
constexpr int N_BLOCK = 4;  // Number of output tiles per row

static constexpr int NUM_PRODUCER_WORKERS = (4);
static constexpr int NUM_CONSUMER_WORKERS = (M_BLOCK*4);
static constexpr int QSIZE = 3;
static constexpr int NUM_THREADS = ((NUM_PRODUCER_WORKERS+NUM_CONSUMER_WORKERS)*kittens::WARP_THREADS);

struct matmul_globals { 
    using sub_tile = st_bf<BLOCK_SIZE,BLOCK_SIZE>;
    using tile_gl =  gl<bf16,  1, 1, -1, -1, sub_tile>;
    tile_gl A, B, C;
    int N;
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {
    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);

    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&As)[QSIZE][M_BLOCK] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, QSIZE, M_BLOCK>();
    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&Bs)[QSIZE][N_BLOCK] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, QSIZE, N_BLOCK>();
    
    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&C_tiles)[M_BLOCK][N_BLOCK] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, M_BLOCK, N_BLOCK>();

    // Accumulator for each consumer warp group
    using wide_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE*N_BLOCK>;
    rt_fl<16, BLOCK_SIZE*N_BLOCK> C_accum;

    int row = blockIdx.y * M_BLOCK; 
    int col = blockIdx.x * N_BLOCK; 

    const int warpid = kittens::warpid();
    const int warpgroupid = warpid/4;

    // Determine type of warp group
    bool is_producer = (warpgroupid == 0);
    bool is_consumer = (warpgroupid > 0 && warpgroupid <= M_BLOCK);
    int num_tiles = (g.N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_consumers = (NUM_THREADS / 128) - 1;
    
    // Consumer index (0-based) for consumer warp groups
    int consumer_idx = is_consumer ? (warpgroupid - 1) : 0;

    __shared__ semaphore full[QSIZE], empty[QSIZE];

    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; i++) {
            init_semaphore(full[i], 0, 1);
            init_semaphore(empty[i], num_consumers, 0);
        }
    }
    __syncthreads();

    // __shared__ semaphore bar;
    if (is_producer) {
        warpgroup::decrease_registers<40>();
        if (warpgroup::laneid() == 0) {
            int p = 0; int q_idx = 0;
            for (int i = 0; i < num_tiles; i++, q_idx++) {
                if (q_idx == QSIZE) { q_idx = 0; p ^= 1; }
                wait(empty[q_idx], p);
                tma::expect_bytes(
                    full[q_idx], 
                    M_BLOCK * size_bytes<typeof(As[0][0])> +
                    N_BLOCK * size_bytes<typeof(Bs[0][0])>
                );
                
                // Load initial A tiles (one row per consumer)
                for (int m = 0; m < M_BLOCK; m++) {
                    tma::load_async(As[q_idx][m], g.A, {0, 0, row + m, i}, full[q_idx]);
                }
                
                // Load initial B tiles (all columns for this thread block)
                for (int n = 0; n < N_BLOCK; n++) {
                    tma::load_async(Bs[q_idx][n], g.B, {0, 0, i, col + n}, full[q_idx]);
                }
            }
        }


    } else {
        warpgroup::increase_registers<232>();
        rt_fl<16,BLOCK_SIZE*N_BLOCK> C_accum;
        kittens::warp::zero(C_accum);

        if (warpgroup::laneid() == 0)
            for (int i = 0; i < QSIZE; ++i) arrive(empty[i], 1);

        int p = 0, q_idx = 0;
        for (int tile = 0; tile < num_tiles; ++tile, ++q_idx) {
            if (q_idx == QSIZE) { q_idx = 0; p ^= 1; }
            wait(full[q_idx], p);
            warpgroup::mma_AB(
                C_accum,
                As[q_idx][consumer_idx],                // Get this consumer's A tile
                reinterpret_cast<wide_tile&>(Bs[q_idx][0])  // Get all B tiles as a wide tile
            );
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) arrive(empty[q_idx], 1);
        }



        wide_tile& wide_C_temp = reinterpret_cast<wide_tile&>(C_tiles[consumer_idx][0]);
        warpgroup::store(wide_C_temp, C_accum);        
        warpgroup::sync(warpgroupid+4);
        
        // Only first warp in each consumer group stores to global memory
        if (warpid % 4 == 0) {
            for (int n = 0; n < N_BLOCK; n++) {
                tma::store_async(g.C, C_tiles[consumer_idx][n], {0, 0, row + consumer_idx, col + n});
                tma::store_async_read_wait();
            }
        }
    }
}

// Launch kernel
void matmul(bf16* A, bf16* B, bf16* C, size_t N) { 
    
    // Global pointers
    using tile_gl = matmul_globals::tile_gl;
    tile_gl a_arg{A, nullptr, nullptr, N, N};
    tile_gl b_arg{B, nullptr, nullptr, N, N};
    tile_gl c_arg{C, nullptr, nullptr, N, N};
    matmul_globals g{a_arg, b_arg, c_arg, (int)N}; 

    // Launch 
    int NEW_ROW_BLOCK_SIZE = BLOCK_SIZE*M_BLOCK;
    int NEW_COL_BLOCK_SIZE = BLOCK_SIZE*N_BLOCK;
    dim3 blocks(
        (N + NEW_COL_BLOCK_SIZE - 1) / (NEW_COL_BLOCK_SIZE),
        (N + NEW_ROW_BLOCK_SIZE - 1) / (NEW_ROW_BLOCK_SIZE)
    );
    
    unsigned long mem_size = 220000;
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<blocks, NUM_THREADS, mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

#include "launch.cu"
