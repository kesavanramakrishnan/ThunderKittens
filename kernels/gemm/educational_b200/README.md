# ThunderKittens Educational B200 BF16 GEMM Kernels

This folder builds up the B200 (Blackwell) BF16 GEMM piece-by-piece. Educational purposes only.

Change the `LEVEL` field in the `Makefile` to `01` - `08`, then `make clean && make`.

To run on Modal: `modal run test_modal.py -- 01` (or omit the level to run all).

## Levels

- **Level 01**: H100-style baseline on B200 — WGMMA + TMA + producer/consumer + double buffering
- **Level 02**: Tensor Memory (TMEM) + tcgen05 MMA — replace register accumulators with TMEM
- **Level 03**: 2-CTA Clusters + Distributed Shared Memory (DSMEM)
- **Level 04**: Separate epilogue warpgroup — split MMA and store into different warpgroups
- **Level 05**: Deeper load pipeline — ring buffer with bitfield phase tracking
- **Level 06**: CLC (Cluster Launch Control) — persistent kernel with hardware work scheduling
- **Level 07**: Supergroup swizzling — L2 cache-friendly tile ordering
- **Level 08**: Epilogue pipelining + MMA/epilogue overlap — final optimized version

## Key B200 concepts introduced per level

| Level | New Concept                        | Key APIs                                                    |
|-------|------------------------------------|-------------------------------------------------------------|
| 01    | Baseline (H100-style)              | `warpgroup::mma_AB`, `tma::load_async`, semaphores          |
| 02    | TMEM + tcgen05                     | `tensor_allocator`, `mm2_ABt`, `mma2_ABt`, `tt<float,...>`  |
| 03    | Clusters + DSMEM                   | `cluster_ctarank()`, `tma::cluster::load_async`, multicast  |
| 04    | Epilogue warpgroup                 | TMEM→reg→SMEM→TMA store, `outputs_arrived/finished`         |
| 05    | Deep pipeline                      | `ring_advance`, `get_phasebit`, `update_phasebit`, bitfield |
| 06    | CLC persistent kernel              | `clc::schedule`, `clc::query`, `clc::handle`                |
| 07    | Supergroup swizzle                 | `get_swizzled_2d_idx<SUPERGROUP_SIZE>`                      |
| 08    | Epilogue pipeline + MMA overlap    | `EPI_PIPE_DEPTH`, `OVERLAP_MMA_EPI`, `NUM_D_TILES`          |

## B matrix layout

All levels use B stored as **(N, K) row-major** (i.e., B transposed), so the GEMM computes **D = A × Bᵀ**.
This matches the production kernel's `mma_ABt` / `mm2_ABt` convention and is the natural layout for
Blackwell tensor cores.
