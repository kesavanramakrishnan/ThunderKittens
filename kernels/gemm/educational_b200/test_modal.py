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
    """Parse benchmark output into list of (M, N, K, tflops, time_us) tuples."""
    results = []
    lines = output.strip().split("\n")
    current_m, current_n, current_k = None, None, None
    time_us = None
    for line in lines:
        m = re.search(r"M=(\d+)\s+N=(\d+)\s+K=(\d+)", line)
        if m:
            current_m = int(m.group(1))
            current_n = int(m.group(2))
            current_k = int(m.group(3))
        t = re.search(r"Average kernel execution time:\s+([\d.]+)\s+us", line)
        if t and current_m is not None:
            time_us = float(t.group(1))
        p = re.search(r"Achieved performance:\s+([\d.]+)\s+TFLOPs", line)
        if p and current_m is not None:
            tflops = float(p.group(1))
            results.append((current_m, current_n, current_k, tflops, time_us))
            current_m = None
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
    for m, n, k, tflops, t_us in results:
        print(f"  {m}x{n}x{k}: {tflops:.2f} TFLOPs ({t_us:.1f} us)")

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
        for m, n, k, _, _ in results:
            all_sizes.add((m, n, k))

    for size in sorted(all_sizes):
        m, n, k = size
        print(f"\n  {m}x{n}x{k}:")
        for level in levels:
            results = all_results.get(level, [])
            for rm, rn, rk, tflops, t_us in results:
                if (rm, rn, rk) == size:
                    print(f"    Level {level}: {tflops:8.2f} TFLOPs  ({t_us:.1f} us)")
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
