import modal
import subprocess
import sys
import re

app = modal.App("tk-educational-b200-gemm")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.11",
    )
    .add_local_dir(
        "/Users/kesavanramakrishnan/ThunderKittens-2",
        remote_path="/root/ThunderKittens",
    )
)


def build_and_run(level: str, workdir: str):
    """Build and run a specific level, return stdout or None on failure."""
    # Clean
    subprocess.run(
        ["make", "clean"], cwd=workdir, capture_output=True, text=True, timeout=60
    )

    # Write the level into the Makefile
    makefile_path = f"{workdir}/Makefile"
    with open(makefile_path, "r") as f:
        content = f.read()
    content = re.sub(r"^LEVEL := \d+", f"LEVEL := {level}", content, flags=re.MULTILINE)
    with open(makefile_path, "w") as f:
        f.write(content)

    # Build
    build = subprocess.run(
        ["make"], cwd=workdir, capture_output=True, text=True, timeout=300
    )
    if build.returncode != 0:
        print(f"=== Level {level} BUILD FAILED ===")
        print(build.stderr[-3000:] if len(build.stderr) > 3000 else build.stderr)
        return None

    # Run
    try:
        run = subprocess.run(
            [f"./level_{level}.out"], cwd=workdir, capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired:
        print(f"=== Level {level} TIMED OUT (60s) ===")
        return None
    if run.stderr:
        print(f"=== Level {level} STDERR ===")
        print(run.stderr[-3000:])
    if run.returncode != 0:
        print(f"=== Level {level} RUN FAILED (exit {run.returncode}) ===")
        print(run.stdout[-2000:])
        return None
    return run.stdout


def parse_results(output):
    """Parse benchmark output into list of result dicts."""
    results = []
    lines = output.strip().split("\n")
    current_m, current_n, current_k = None, None, None
    tk_us, tk_tflops = None, None
    cublas_us, cublas_tflops = None, None
    for line in lines:
        m = re.search(r"M=(\d+)\s+N=(\d+)\s+K=(\d+)", line)
        if m:
            # Emit previous block if it had TK data but no cuBLAS
            if current_m is not None and tk_tflops is not None:
                results.append({
                    "M": current_m, "N": current_n, "K": current_k,
                    "tk_us": tk_us, "tk_tflops": tk_tflops,
                    "cublas_us": cublas_us, "cublas_tflops": cublas_tflops,
                })
            current_m = int(m.group(1))
            current_n = int(m.group(2))
            current_k = int(m.group(3))
            tk_us = tk_tflops = cublas_us = cublas_tflops = None
        # New format: "TK kernel:   91.07 us  (1509.1 TFLOPs)"
        t = re.search(r"TK kernel:\s+([\d.]+)\s+us\s+\(([\d.]+)\s+TFLOPs\)", line)
        if t and current_m is not None:
            tk_us = float(t.group(1))
            tk_tflops = float(t.group(2))
        # New format: "cuBLAS:      95.2 us  (1443.5 TFLOPs)"
        c = re.search(r"cuBLAS:\s+([\d.]+)\s+us\s+\(([\d.]+)\s+TFLOPs\)", line)
        if c and current_m is not None:
            cublas_us = float(c.group(1))
            cublas_tflops = float(c.group(2))
        # Old format fallback: "Average kernel execution time: 91.07 us"
        t_old = re.search(r"Average kernel execution time:\s+([\d.]+)\s+us", line)
        if t_old and current_m is not None and tk_us is None:
            tk_us = float(t_old.group(1))
        p_old = re.search(r"Achieved performance:\s+([\d.]+)\s+TFLOPs", line)
        if p_old and current_m is not None and tk_tflops is None:
            tk_tflops = float(p_old.group(1))
        # Emit result when we have TK data and either cuBLAS or end of block
        if tk_tflops is not None and cublas_tflops is not None and current_m is not None:
            results.append({
                "M": current_m, "N": current_n, "K": current_k,
                "tk_us": tk_us, "tk_tflops": tk_tflops,
                "cublas_us": cublas_us, "cublas_tflops": cublas_tflops,
            })
            current_m = None
    # Handle case where cuBLAS is not present (old levels)
    if current_m is not None and tk_tflops is not None:
        results.append({
            "M": current_m, "N": current_n, "K": current_k,
            "tk_us": tk_us, "tk_tflops": tk_tflops,
            "cublas_us": cublas_us, "cublas_tflops": cublas_tflops,
        })
    return results


@app.function(
    image=image,
    gpu="B200",
    timeout=1800,
)
def run_level(level: str):
    workdir = "/root/ThunderKittens/kernels/gemm/educational_b200"
    print(f"\n{'='*60}")
    print(f"  LEVEL {level}")
    print(f"{'='*60}")

    output = build_and_run(level, workdir)
    if output is None:
        return {"level": level, "status": "failed", "results": []}

    print(output)
    results = parse_results(output)

    # Summary
    print(f"\n--- Level {level} Summary ---")
    for r in results:
        line = f"  {r['M']}x{r['N']}x{r['K']}: TK {r['tk_tflops']:.2f} TFLOPs ({r['tk_us']:.1f} us)"
        if r.get('cublas_tflops'):
            pct = r['tk_tflops'] / r['cublas_tflops'] * 100
            line += f"  |  cuBLAS {r['cublas_tflops']:.2f} TFLOPs  |  TK/cuBLAS {pct:.1f}%"
        print(line)

    return {"level": level, "status": "ok", "results": results}


@app.function(
    image=image,
    gpu="B200",
    timeout=3600,
)
def run_all_levels(levels: list[str]):
    workdir = "/root/ThunderKittens/kernels/gemm/educational_b200"
    all_results = {}

    for level in levels:
        print(f"\n{'='*60}")
        print(f"  LEVEL {level}")
        print(f"{'='*60}")

        output = build_and_run(level, workdir)
        if output is None:
            all_results[level] = []
            continue

        print(output)
        all_results[level] = parse_results(output)

    # Print comparison table
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON ACROSS LEVELS")
    print(f"{'='*80}")

    # Collect all problem sizes
    all_sizes = set()
    for results in all_results.values():
        for r in results:
            all_sizes.add((r['M'], r['N'], r['K']))

    for size in sorted(all_sizes):
        m, n, k = size
        print(f"\n  {m}x{n}x{k}:")
        for level in levels:
            results = all_results.get(level, [])
            for r in results:
                if (r['M'], r['N'], r['K']) == size:
                    line = f"    Level {level}: {r['tk_tflops']:8.2f} TFLOPs  ({r['tk_us']:.1f} us)"
                    if r.get('cublas_tflops'):
                        pct = r['tk_tflops'] / r['cublas_tflops'] * 100
                        line += f"  [{pct:.1f}% of cuBLAS]"
                    print(line)
                    break

    return all_results


@app.local_entrypoint()
def main(level: str = ""):
    # Usage:
    #   modal run test_modal.py                    # runs all levels
    #   modal run test_modal.py --level 01         # runs level 01 only
    #   modal run test_modal.py --level 01,03      # runs levels 01 and 03

    if level == "":
        # Run all levels that exist
        import os
        import glob
        pattern = os.path.join(os.path.dirname(__file__), "level_*.cu")
        files = sorted(glob.glob(pattern))
        levels = [re.search(r"level_(\d+)\.cu", f).group(1) for f in files]
        if not levels:
            print("No level_*.cu files found!")
            return
        print(f"Running all levels: {levels}")
        run_all_levels.remote(levels)
    elif "," in level:
        levels = [l.strip() for l in level.split(",")]
        run_all_levels.remote(levels)
    else:
        run_level.remote(level)
