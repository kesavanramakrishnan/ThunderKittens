import modal
import subprocess
import re

app = modal.App("tk-nvfp4-b200-perf-compare")

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

###############################################################################
# Source patches — we start from the MBARRIER version (current in-tree) and
# derive the other two variants by string replacement.
###############################################################################

# --- Semaphore declaration / init for mbarrier version (current) ---
MBARRIER_SEMA_DECL = """\
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ semaphore tmem_dealloc_ready;"""

MBARRIER_SEMA_INIT = """\
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
        init_semaphore(tmem_dealloc_ready, 0, C::CLUSTER_SIZE);"""

# --- Deprovision block for each variant ---
MBARRIER_DEPROVISION = """\
        warpgroup::sync(1);
        warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) {
            warpgroup::tma::cluster::arrive(tmem_dealloc_ready, 0, 1);
            warpgroup::tma::cluster::arrive(tmem_dealloc_ready, 1, 1);
            tma::cluster::wait(tmem_dealloc_ready, 0);
            tm_allocator.deprovision();
        }
    }
}"""

NO_SYNC_DEPROVISION = """\
        warpgroup::sync(1);
        warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }
}"""

NO_SYNC_SEMA_DECL = """\
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;"""

NO_SYNC_SEMA_INIT = """\
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);"""

# NOTE: A full cluster barrier (barrier.cluster.arrive/wait) is NOT viable
# here because it requires ALL threads in the CTA to participate, but the
# producer warpgroup diverges via warp::elect_leader() and never reconverges.
# The mbarrier (semaphore) approach is the only correct fix for this kernel
# structure — it allows warp-level signaling without requiring all threads.


def build_and_bench(label, workdir):
    """Build and run the benchmark, return stdout."""
    subprocess.run(
        ["make", "clean"], cwd=workdir, capture_output=True, text=True, timeout=60
    )
    build = subprocess.run(
        ["make"], cwd=workdir, capture_output=True, text=True, timeout=300
    )
    if build.returncode != 0:
        print(f"=== {label} BUILD FAILED ===")
        print(build.stderr[-3000:] if len(build.stderr) > 3000 else build.stderr)
        return None
    run = subprocess.run(
        ["./nvfp4_b200_gemm.out"], cwd=workdir, capture_output=True, text=True, timeout=600
    )
    if run.returncode != 0:
        print(f"=== {label} RUN FAILED (exit {run.returncode}) ===")
        print(run.stdout)
        print(run.stderr)
        return None
    return run.stdout


def parse_results(output):
    """Parse benchmark output into list of (size, tflops, time_us) tuples."""
    results = []
    lines = output.strip().split("\n")
    current_m = None
    time_us = None
    for line in lines:
        m = re.search(r"M=(\d+)\s+N=(\d+)\s+K=(\d+)", line)
        if m:
            current_m = int(m.group(1))
        t = re.search(r"Average kernel execution time:\s+([\d.]+)\s+us", line)
        if t and current_m is not None:
            time_us = float(t.group(1))
        p = re.search(r"Achieved performance:\s+([\d.]+)\s+TFLOPs", line)
        if p and current_m is not None:
            tflops = float(p.group(1))
            results.append((current_m, tflops, time_us))
            current_m = None
    return results


def write_variant(src_path, base_src, variant):
    """Patch the base (mbarrier) source into the requested variant."""
    if variant == "mbarrier":
        patched = base_src
    elif variant == "no_sync":
        patched = base_src
        patched = patched.replace(MBARRIER_SEMA_DECL, NO_SYNC_SEMA_DECL)
        patched = patched.replace(MBARRIER_SEMA_INIT, NO_SYNC_SEMA_INIT)
        patched = patched.replace(MBARRIER_DEPROVISION, NO_SYNC_DEPROVISION)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    with open(src_path, "w") as f:
        f.write(patched)


@app.function(
    image=image,
    gpu="B200",
    timeout=1800,
)
def compare_perf():
    workdir = "/root/ThunderKittens/kernels/gemm/nvfp4_b200"
    src = f"{workdir}/nvfp4_b200_gemm.cu"

    with open(src) as f:
        base_src = f.read()

    variants = [
        ("no_sync",  "OLD: no cluster sync before deprovision"),
        ("mbarrier", "FIXED: semaphore-based cluster sync (warp-level)"),
    ]

    all_results = {}
    for variant_key, variant_label in variants:
        print("\n" + "=" * 70)
        print(f"RUNNING {variant_label}")
        print("=" * 70)
        write_variant(src, base_src, variant_key)
        output = build_and_bench(variant_key, workdir)
        if output:
            print(output)
            all_results[variant_key] = parse_results(output)
        else:
            all_results[variant_key] = []

    # Restore original
    with open(src, "w") as f:
        f.write(base_src)

    # --- Print comparison table ---
    if all(all_results.get(k) for k, _ in variants):
        no_sync  = all_results["no_sync"]
        mbarrier = all_results["mbarrier"]

        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON: OLD (no sync) vs FIXED (mbarrier)")
        print("=" * 80)
        hdr = f"{'Size':>6} | {'Old (TFLOPs)':>14} | {'Fixed (TFLOPs)':>14} | {'Old (us)':>10} | {'Fixed (us)':>10} | {'Diff':>7}"
        print(hdr)
        print("-" * len(hdr))
        for (m0, tf0, t0), (m1, tf1, t1) in zip(no_sync, mbarrier):
            d = (tf1 - tf0) / tf0 * 100
            print(f"{m0:>6} | {tf0:>12.2f} TF | {tf1:>12.2f} TF | {t0:>10.2f} | {t1:>10.2f} | {d:>+6.2f}%")

    return 0


@app.local_entrypoint()
def main():
    exit_code = compare_perf.remote()
    if exit_code != 0:
        print(f"Failed with exit code {exit_code}")
    else:
        print("Done!")
