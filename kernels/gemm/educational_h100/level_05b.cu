#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <chrono>
#include <cuda_runtime.h>

#include "kittens.cuh"
using namespace kittens;

static constexpr int BLOCK_SIZE = 64;
static constexpr int NUM_WORKERS =  (4);
static constexpr int NUM_THREADS = (NUM_WORKERS*kittens::WARP_THREADS);

struct matmul_globals { 
    using sub_tile = st_bf<BLOCK_SIZE,BLOCK_SIZE>;
    using tile_gl =  gl<bf16,  1, 1, -1, -1, sub_tile>;
    tile_gl A;
    tile_gl B_col;  // B stored in column-major (i.e. transposed: N x K)
    tile_gl C;
    int N;
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {

    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE,BLOCK_SIZE> &As = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>>(); 
    st_bf<BLOCK_SIZE,BLOCK_SIZE> &Bs = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>>(); 
    
    rt_fl<16,BLOCK_SIZE> C_accum;
    rt_fl<16,BLOCK_SIZE> C_accum_cpy;

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int row = by; 
    int col = bx; 

    kittens::warp::zero(C_accum_cpy);
    int num_tiles = (g.N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile) {
        warpgroup::load(As, g.A, {0, 0, row, tile});
        warpgroup::load(Bs, g.B_col, {0, 0, col, tile});  // swapped: B is now (N, K)
        __syncthreads();
        warpgroup::mma_ABt(C_accum, As, Bs);               // A × Bᵀ
        warpgroup::mma_async_wait();
        kittens::warp::add(C_accum_cpy, C_accum_cpy, C_accum);
        kittens::warp::zero(C_accum);
    }
    warpgroup::store(g.C, C_accum_cpy, {0, 0, row, col});
}

// Simple transpose kernel
__global__ void transpose_kernel(__nv_bfloat16* out, const __nv_bfloat16* in, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int r = idx / cols;
        int c = idx % cols;
        out[c * rows + r] = in[r * cols + c];  // transpose: out[c][r] = in[r][c]
    }
}

// launch kernel
void matmul(bf16* A, bf16* B, bf16* C, size_t N) { 

    // Transpose B (K×N row-major) → B_T (N×K row-major, i.e. K×N column-major)
    bf16* B_T;
    cudaMalloc(&B_T, N * N * sizeof(bf16));
    int total = N * N;
    int threads = 256;
    int blocks_t = (total + threads - 1) / threads;
    transpose_kernel<<<blocks_t, threads>>>(B_T, B, N, N);
    cudaDeviceSynchronize();

    // global pointers
    using a_gl = matmul_globals::tile_gl;
    using b_gl = matmul_globals::tile_gl; 
    using c_gl = matmul_globals::tile_gl;
    a_gl  a_arg{A, nullptr, nullptr, N, N};
    b_gl  b_arg{B_T, nullptr, nullptr, N, N};  // B_T is (N, K)
    c_gl  c_arg{C, nullptr, nullptr, N, N};
    matmul_globals g{a_arg, b_arg, c_arg, (int)N}; 

    // launch
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    unsigned long mem_size = 100000;
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<blocks, NUM_THREADS, mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();

    cudaFree(B_T);
}

#include "launch.cu"
