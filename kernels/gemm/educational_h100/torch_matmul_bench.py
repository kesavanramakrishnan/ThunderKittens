import torch
import time

M, N, K = 8192, 8192, 8192
dtype = torch.bfloat16
device = "cuda"

A = torch.randn(M, K, dtype=dtype, device=device)
B = torch.randn(K, N, dtype=dtype, device=device)

# Warmup
for _ in range(10):
    C = torch.matmul(A, B)
torch.cuda.synchronize()

# Benchmark
num_iters = 100
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(num_iters):
    C = torch.matmul(A, B)
torch.cuda.synchronize()
end = time.perf_counter()

avg_time_us = (end - start) / num_iters * 1e6
flops = 2 * M * N * K
tflops = flops / (avg_time_us * 1e-6) / 1e12

print(f"M={M} N={N} K={K}  dtype={dtype}")
print(f"Avg time: {avg_time_us:.1f} us")
print(f"Performance: {tflops:.1f} TFLOPs")
