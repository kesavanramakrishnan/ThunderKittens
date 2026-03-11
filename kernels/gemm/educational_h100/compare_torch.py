"""
Compare your GEMM kernel output against torch.matmul (cuBLAS).

Usage:
  1. Run `make run` to execute the kernel (saves binary files to /tmp/)
  2. Run `python3 compare_torch.py` to compare against torch
"""
import torch
import numpy as np

M, N, K = 4096, 4096, 4096

def load_bf16(path, shape):
    with open(path, 'rb') as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape).copy()
    return torch.from_numpy(arr).view(torch.bfloat16).cuda()

# Load the matrices saved by launch.cu
A = load_bf16('/tmp/gemm_A.bin', (M, K))
B = load_bf16('/tmp/gemm_B.bin', (K, N))
C_kernel = load_bf16('/tmp/gemm_C_kernel.bin', (M, N))
C_cuda_ref = load_bf16('/tmp/gemm_C_ref.bin', (M, N))

# Torch reference (cuBLAS bf16)
C_torch = torch.matmul(A, B)

# fp32 ground truth
C_fp32 = torch.matmul(A.float(), B.float())

def report(name, C_test, C_ref):
    diff = (C_test.float() - C_ref.float()).abs()
    print(f"\n--- {name} ---")
    print(f"  Max abs error:  {diff.max().item():.4f}")
    print(f"  Mean abs error: {diff.mean().item():.6f}")
    print(f"  Elements > 0.5: {(diff > 0.5).sum().item()}")
    print(f"  Elements > 1.0: {(diff > 1.0).sum().item()}")
    return diff.max().item()

print(f"M={M} N={N} K={K}")

report("Your kernel vs torch.matmul (cuBLAS)", C_kernel, C_torch)
report("Your kernel vs fp32 ground truth",     C_kernel, C_fp32)
report("torch.matmul vs fp32 ground truth",    C_torch,  C_fp32)
max_err = report("Your kernel vs CUDA naive ref",       C_kernel, C_cuda_ref)

if max_err < 1.0:
    print("\nPASS: Kernel is correct.")
else:
    print("\nFAIL: Kernel output has significant errors.")
