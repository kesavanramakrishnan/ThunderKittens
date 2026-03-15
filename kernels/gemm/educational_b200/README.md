# ThunderKittens Educational B200 BF16 Matmul Kernels

This folder builds up a B200 (Blackwell) BF16 matmul (D = A × Bᵀ) piece-by-piece.

## Running

```bash
# Single level on Modal (B200 GPU):
modal run test_modal.py --level 10

# All levels:
modal run test_modal.py

# Local build (set LEVEL in Makefile):
make clean && make
```

## Levels

| # | Name | What's New | Output Tile |
|---|------|-----------|-------------|
| 01 | H100-style baseline | WGMMA + TMA + producer/consumer + double buffering (QSIZE=2) | 64×256 |
| 02 | TMEM + tcgen05 MMA | Replace register accumulators with Tensor Memory; `mm_ABt`/`mma_ABt` | 128×128 |
| 03 | 2-CTA Clusters + DSMEM | CLUSTER_SIZE=2, `mm2_ABt`/`mma2_ABt`, multicast TMA loads | 256×128 |
| 04 | Separate epilogue warpgroup | 3 warpgroups (producer/consumer/epilogue), LOAD_PIPE_DEPTH=2 | 256×128 |
| 05 | Deeper load pipeline | LOAD_PIPE_DEPTH=4, bitfield phase tracking, `ring_advance` | 256×128 |
| 06 | Persistent kernel (strided) | Outer task loop, `outputs_finished` semaphore, SM-count grid | 256×128 |
| 07 | Epilogue pipelining | EPI_PIPE_DEPTH=4, chunked TMEM→SMEM→global, double-buffered d_smem | 256×128 |
| 08 | MMA pipeline depth | MMA_PIPE_DEPTH=2, double-buffered TMEM accumulators, BN=256 | 256×256 |
| 09 | CLC (Cluster Launch Control) | Hardware work scheduling, `clc::handle`/`schedule`/`query` | 256×256 |
| 10 | Full production kernel | Supergroup swizzling, OVERLAP_MMA_EPI, NUM_CONSUMERS=2, templated config | 256×256 |

## Key B200 concepts introduced per level

| Level | New Concept                          | Key APIs / Patterns                                          |
|-------|--------------------------------------|--------------------------------------------------------------|
| 01    | Baseline (H100-style on B200)        | `warpgroup::mma_ABt`, `tma::load_async`, semaphores          |
| 02    | TMEM + tcgen05                       | `tensor_allocator`, `mm_ABt`, `mma_ABt`, `tt<float,...>`     |
| 03    | Clusters + DSMEM                     | `cluster_ctarank()`, `tma::cluster::load_async`, multicast   |
| 04    | Epilogue warpgroup + double buffer   | TMEM→reg→SMEM→TMA store, `tmem_provisioned`, 3 warpgroups    |
| 05    | Deep pipeline + bitfield             | `ring_advance`, `get_phasebit`, `update_phasebit`, bitfield  |
| 06    | Persistent kernel (strided)          | Task loop, `outputs_finished`, SM-count grid sizing           |
| 07    | Epilogue pipelining                  | `EPI_PIPE_DEPTH`, subtile reads, `store_async_read_wait`     |
| 08    | MMA pipeline depth                   | `MMA_PIPE_DEPTH=2`, d_tt[2] accumulators, task_iter phases   |
| 09    | CLC hardware scheduling              | `clc::schedule`, `clc::query`, `clc::handle`, `LaunchConfig<true,true>` |
| 10    | Production optimizations             | `get_swizzled_2d_idx`, `OVERLAP_MMA_EPI`, `NUM_CONSUMERS=2`  |

## B matrix layout

All levels use B stored as **(N, K) row-major** (i.e., B transposed), so the matmul computes **D = A × Bᵀ**.
This matches the production kernel's `mma_ABt` / `mm2_ABt` convention and is the natural layout for
Blackwell tensor cores.

## Benchmark results (B200, BF16)

100 profiling iterations, 500 warmup, L2 eviction groups.

### 4096×4096×4096

| Level | TFLOPs | Time (µs) | % cuBLAS |
|-------|--------|-----------|----------|
| 02 | 279 | 492.1 | 19.9% |
| 03 | 321 | 428.4 | 22.4% |
| 04 | 612 | 224.6 | 42.1% |
| 05 | 1026 | 134.0 | 72.0% |
| 06 | 1115 | 123.2 | 78.3% |
| 07 | 1010 | 136.0 | 72.1% |
| 08 | 1289 | 106.6 | 83.6% |
| 09 | 1216 | 113.1 | 78.3% |
| 10 | **1559** | **88.1** | **104.7%** |

### 8192×8192×8192

| Level | TFLOPs | Time (µs) | % cuBLAS |
|-------|--------|-----------|----------|
| 02 | 263 | 4186.6 | 17.1% |
| 03 | 272 | 4037.3 | 17.7% |
| 04 | 523 | 2103.0 | 33.7% |
| 05 | 828 | 1327.5 | 53.4% |
| 06 | 846 | 1299.6 | 55.0% |
| 07 | 818 | 1345.0 | 53.2% |
| 08 | 1220 | 900.9 | 79.1% |
| 09 | 1216 | 904.4 | 78.4% |
| 10 | **1554** | **707.6** | **101.2%** |
