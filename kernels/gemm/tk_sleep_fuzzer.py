#!/usr/bin/env python3
"""
TK Kernel Sleep Injection Fuzzer

Injects __nanosleep at every sync-sensitive point in a TK kernel to find races.
Runs all variants in parallel on Modal B200 GPUs.

Usage:
    modal run tk_sleep_fuzzer.py --kernel nvfp4
    modal run tk_sleep_fuzzer.py --kernel mxfp8
"""

import modal, subprocess, re, sys, traceback

app = modal.App("tk-sleep-fuzzer")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .add_local_dir("/Users/kesavanramakrishnan/ThunderKittens-2", remote_path="/root/ThunderKittens")
)

KERNEL_CONFIGS = {
    "nvfp4": dict(
        src_dir="kernels/gemm/nvfp4_b200", src_file="nvfp4_b200_gemm.cu",
        binary="nvfp4_b200_gemm.out", test_config="nvfp4_gemm::config<256, 5, 8, 4, 2, false>",
        test_size=4096,
    ),
    "mxfp8": dict(
        src_dir="kernels/gemm/mxfp8_b200", src_file="mxfp8_b200_gemm.cu",
        binary="mxfp8_b200_gemm.out", test_config="mxfp8_gemm::config<256, 5, 8, 8, 2, false>",
        test_size=4096,
    ),
}

INJECTION_PATTERNS = [
    (r'^\s*wait\s*\(', "before_wait"),
    (r'tma::cluster::arrive\s*\(', "before_arrive"),
    (r'\.deprovision\s*\(', "before_deprovision"),
    (r'\.provision\s*\(', "before_provision"),
    (r'warpgroup::load_async\s*\(', "before_tmem_read"),
    (r'(?:mm2_ABt|mma2_ABt)\s*\(', "before_mma"),
    (r'tensor_commit', "before_tensor_commit"),
    (r'load_mxnv_scale_async', "before_scale_load"),
]

SLEEP_NS = 2_000_000


def find_kernel_range(lines):
    start, depth = None, 0
    for i, ln in enumerate(lines):
        if re.search(r'__device__\s+inline\s+void\s+kernel\s*\(', ln):
            start = i
        if start is not None:
            depth += ln.count('{') - ln.count('}')
            if depth == 0 and i > start:
                return start, i
    return (start or 0), len(lines) - 1


def find_injection_points(source):
    lines = source.split('\n')
    k_start, k_end = find_kernel_range(lines)
    points, seen = [], set()
    for i in range(k_start, k_end + 1):
        ln = lines[i]
        if ln.strip().startswith('//') or ln.strip().startswith('#pragma'):
            continue
        for pat, label in INJECTION_PATTERNS:
            if re.search(pat, ln) and i not in seen:
                points.append((i, label, ln.strip()[:80]))
                seen.add(i)
                break
    return points


def inject_sleep(source, line_idx, cta_id):
    lines = source.split('\n')
    indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
    pad = ' ' * indent
    sleep = f'{pad}if (cluster_ctarank() == {cta_id}) {{ __nanosleep({SLEEP_NS}); }}\n'
    lines.insert(line_idx, sleep)
    return '\n'.join(lines)


def patch_main_for_stress(source, cfg):
    lines = source.split('\n')
    ms, me, depth = None, None, 0
    for i, ln in enumerate(lines):
        if re.match(r'\s*int main\(\)\s*\{', ln):
            ms = i
        if ms is not None:
            depth += ln.count('{') - ln.count('}')
            if depth == 0 and i > ms:
                me = i; break
    if ms is None:
        return source
    new = (f'int main() {{\n'
           f'    bool ncu = false; int N = {cfg["test_size"]};\n'
           f'    std::cout << "=== FUZZER (5 trials) ===" << std::endl;\n'
           f'    for (int t = 0; t < 5; t++) {{\n'
           f'        std::cout << "Trial " << t << "..." << std::flush;\n'
           f'        run_benchmark<{cfg["test_config"]}>(N, N, N, ncu);\n'
           f'        std::cout << " OK" << std::endl;\n'
           f'    }}\n'
           f'    std::cout << "=== ALL TRIALS PASSED ===" << std::endl;\n'
           f'    return 0;\n'
           f'}}')
    lines[ms:me+1] = new.split('\n')
    return '\n'.join(lines)


def patch_fast_bench(source):
    source = re.sub(r'int num_warmups = ncu \? 0 : \d+;', 'int num_warmups = ncu ? 0 : 1;', source)
    source = re.sub(r'int num_iters = ncu \? 1 : \d+;', 'int num_iters = ncu ? 1 : 1;', source)
    return source


@app.function(image=image, gpu="B200", timeout=900, retries=0)
def test_injection(kernel_name: str, point_idx: int, line_idx: int,
                   label: str, cta_id: int):
    cfg = KERNEL_CONFIGS[kernel_name]
    workdir = f"/root/ThunderKittens/{cfg['src_dir']}"
    src_path = f"{workdir}/{cfg['src_file']}"
    with open(src_path) as f:
        source = f.read()

    patched = inject_sleep(source, line_idx, cta_id)
    patched = patch_fast_bench(patched)
    patched = patch_main_for_stress(patched, cfg)
    with open(src_path, 'w') as f:
        f.write(patched)

    subprocess.run(["make", "clean"], cwd=workdir, capture_output=True, timeout=60)
    build = subprocess.run(["make"], cwd=workdir, capture_output=True, text=True, timeout=300)
    if build.returncode != 0:
        with open(src_path, 'w') as f:
            f.write(source)
        return dict(point_idx=point_idx, line=line_idx+1, label=label,
                    cta=cta_id, passed=False, rc=-100, error=f"BUILD: {build.stderr[-500:]}")

    run = subprocess.run([f"./{cfg['binary']}"], cwd=workdir,
                         capture_output=True, text=True, timeout=300)
    with open(src_path, 'w') as f:
        f.write(source)

    passed = run.returncode == 0 and run.stdout and "ALL TRIALS PASSED" in run.stdout
    err = ""
    if not passed:
        err = (run.stderr[-300:] if run.stderr else "") + (run.stdout[-300:] if run.stdout else "")
    return dict(point_idx=point_idx, line=line_idx+1, label=label,
                cta=cta_id, passed=passed, rc=run.returncode, error=err)


@app.function(image=image, gpu="B200", timeout=60)
def get_injection_points(kernel_name: str):
    cfg = KERNEL_CONFIGS[kernel_name]
    with open(f"/root/ThunderKittens/{cfg['src_dir']}/{cfg['src_file']}") as f:
        return find_injection_points(f.read())


@app.local_entrypoint()
def main(kernel: str = "nvfp4"):
    if kernel not in KERNEL_CONFIGS:
        print(f"Unknown kernel: {kernel}. Options: {list(KERNEL_CONFIGS.keys())}")
        sys.exit(1)

    cfg = KERNEL_CONFIGS[kernel]
    print(f"\n{'='*70}")
    print(f"  TK Sleep Injection Fuzzer — {cfg['src_file']}")
    print(f"{'='*70}\n")

    print("Finding injection points...")
    points = get_injection_points.remote(kernel)
    print(f"Found {len(points)} points:\n")
    for i, (li, label, text) in enumerate(points):
        print(f"  [{i:2d}] line {li+1:4d}  {label:25s}  {text[:60]}")

    total = len(points) * 2
    print(f"\nLaunching {total} variants in parallel (each point × CTA 0,1)...\n")

    handles = []
    args_list = []
    for i, (li, label, _) in enumerate(points):
        for cta in [0, 1]:
            handles.append(test_injection.spawn(kernel, i, li, label, cta))
            args_list.append((i, li+1, label, cta))

    results = []
    for idx, h in enumerate(handles):
        pi, ln, lb, ct = args_list[idx]
        try:
            r = h.get()
            results.append(r)
        except Exception as e:
            err_msg = str(e)[:200]
            print(f"  [!] Task crashed: line {ln} {lb} CTA {ct}: {err_msg}")
            results.append(dict(point_idx=pi, line=ln, label=lb,
                                cta=ct, passed=False, rc=-999,
                                error=f"Modal task error: {err_msg}"))

    # Report
    failures = [r for r in results if not r["passed"]]
    passes = [r for r in results if r["passed"]]

    print(f"\n{'='*70}")
    print(f"  RESULTS: {len(passes)} PASSED, {len(failures)} FAILED out of {total}")
    print(f"{'='*70}\n")

    if failures:
        print("FAILURES (potential race conditions):\n")
        for r in sorted(failures, key=lambda x: x["line"]):
            print(f"  ✗ line {r['line']:4d}  {r['label']:25s}  CTA {r['cta']}  rc={r['rc']}")
            if r["error"]:
                for el in r["error"].strip().split('\n')[-3:]:
                    print(f"      {el.strip()}")
            print()
    else:
        print("  ✓ ALL INJECTION POINTS PASSED — no races detected!\n")

    if passes:
        print(f"Passing points ({len(passes)}):")
        for r in sorted(passes, key=lambda x: x["line"]):
            print(f"  ✓ line {r['line']:4d}  {r['label']:25s}  CTA {r['cta']}")

    print()
