"""
Reproducer for the nvfp4 tensor memory deallocation race condition.

Injects a __nanosleep delay on CTA 1 before deprovision to deterministically
widen the race window.  Two variants are compiled and run:

  1. UNFIXED + nanosleep  → CTA 0 deprovisions while CTA 1 is still using tmem
                            → expect crash (illegal memory access) or corruption
  2. FIXED   + nanosleep  → mbarrier ensures both CTAs finish before deprovision
                            → should work correctly despite the delay

Run:
    modal run reproduce_race.py
"""

import modal
import subprocess
import re

app = modal.App("tk-nvfp4-race-reproducer")

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
# Source patches
###############################################################################

# ── The current in-tree source has the mbarrier fix.  We derive variants. ──

# Current (fixed) deprovision block
FIXED_DEPROVISION = """\
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

# Current semaphore declarations (fixed has tmem_dealloc_ready)
FIXED_SEMA_DECL = """\
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ semaphore tmem_dealloc_ready;"""

FIXED_SEMA_INIT = """\
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
        init_semaphore(tmem_dealloc_ready, 0, C::CLUSTER_SIZE);"""

# ── Nanosleep injection point: delay CTA 1's EPILOGUE (tmem reads) ──
# We inject a nanosleep before the epilogue's wait(outputs_arrived, ...)
# on CTA 1 only.  This makes CTA 1's epilogue start late, so CTA 0 finishes
# its epilogue and reaches deprovision() while CTA 1 is still reading tmem.

# The epilogue wait line (present in both variants)
EPI_WAIT_LINE = "            // Wait for the last matmul to complete"
EPI_WAIT_LINE_WITH_SLEEP = """\
            // RACE REPRODUCER: delay CTA 1's epilogue so CTA 0 reaches
            // deprovision() while CTA 1 is still reading tensor memory
            if (cluster_ctarank() == 1) {
                __nanosleep(2000000); // 2ms delay on CTA 1 epilogue
            }
            // Wait for the last matmul to complete"""

# ── Variant 1: UNFIXED + nanosleep (should crash/corrupt) ──
UNFIXED_NANOSLEEP_SEMA_DECL = """\
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;"""

UNFIXED_NANOSLEEP_SEMA_INIT = """\
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);"""

UNFIXED_NANOSLEEP_DEPROVISION = """\
        warpgroup::sync(1);
        warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }
}"""

# ── Variant 2: FIXED + nanosleep (should work correctly) ──
# Keep the mbarrier fix.  Even with CTA 1 delayed, the barrier
# prevents CTA 0 from deprovisioning until CTA 1 arrives.
FIXED_NANOSLEEP_DEPROVISION = FIXED_DEPROVISION  # no change needed

# Reduce benchmark to 1 warmup + 1 iter so we're not waiting forever with nanosleep
BENCH_ORIG_WARMUP = "int num_warmups = ncu ? 0 : 500;"
BENCH_ORIG_ITERS  = "int num_iters = ncu ? 1 : 100;"
BENCH_FAST_WARMUP = "int num_warmups = ncu ? 0 : 1;"
BENCH_FAST_ITERS  = "int num_iters = ncu ? 1 : 1;"

# Replace the full main() body to stress-test a single size
# Original main runs 5 sizes; we run 8192 x 20 trials
ORIG_MAIN = """\
int main() {
    int N;
    bool ncu = false;

    // Template parameters: Nb, LOAD_PIPE_DEPTH, EPI_PIPE_DEPTH, SUPERGROUP_SIZE, NUM_D_TILES, OVERLAP_EPI
    N = 1024;
    run_benchmark<nvfp4_gemm::config<128, 5, 4, 12, 2, true>>(N, N, N, ncu);
    N = 2048;
    run_benchmark<nvfp4_gemm::config<256, 5, 8, 4, 2, true>>(N, N, N, ncu);
    N = 4096;
    run_benchmark<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>(N, N, N, ncu);
    N = 8192;
    run_benchmark<nvfp4_gemm::config<256, 4, 16, 1, 2, false>>(N, N, N, ncu);
    N = 16384;
    run_benchmark<nvfp4_gemm::config<256, 4, 16, 12, 2, false>>(N, N, N, ncu);

    return 0;
}"""

REPRO_MAIN = """\
int main() {
    bool ncu = false;
    int N = 8192;
    std::cout << "=== RUNNING RACE REPRODUCER (20 trials) ===" << std::endl;
    for (int trial = 0; trial < 20; trial++) {
        std::cout << "Trial " << trial << "..." << std::flush;
        run_benchmark<nvfp4_gemm::config<256, 4, 16, 1, 2, false>>(N, N, N, ncu);
        std::cout << " OK" << std::endl;
    }
    std::cout << "=== ALL 20 TRIALS PASSED ===" << std::endl;
    return 0;
}"""


def patch_source(base_src, variant):
    """Generate source for the given variant."""
    src = base_src

    # Speed up benchmark
    src = src.replace(BENCH_ORIG_WARMUP, BENCH_FAST_WARMUP)
    src = src.replace(BENCH_ORIG_ITERS, BENCH_FAST_ITERS)

    # Inject nanosleep to delay CTA 1's epilogue (both variants)
    src = src.replace(EPI_WAIT_LINE, EPI_WAIT_LINE_WITH_SLEEP)

    if variant == "unfixed_nanosleep":
        # Remove the mbarrier fix → bare deprovision, no cluster sync
        src = src.replace(FIXED_SEMA_DECL, UNFIXED_NANOSLEEP_SEMA_DECL)
        src = src.replace(FIXED_SEMA_INIT, UNFIXED_NANOSLEEP_SEMA_INIT)
        src = src.replace(FIXED_DEPROVISION, UNFIXED_NANOSLEEP_DEPROVISION)
    elif variant == "fixed_nanosleep":
        # Keep the mbarrier fix — deprovision is already protected
        pass
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Replace main() to run stress test
    src = src.replace(ORIG_MAIN, REPRO_MAIN)

    return src


def build_and_run(label, workdir, timeout=300):
    """Build and run, return (stdout, stderr, returncode)."""
    subprocess.run(["make", "clean"], cwd=workdir, capture_output=True, timeout=60)
    build = subprocess.run(["make"], cwd=workdir, capture_output=True, text=True, timeout=300)
    if build.returncode != 0:
        print(f"\n{'='*70}")
        print(f"{label}: BUILD FAILED")
        print(f"{'='*70}")
        print(build.stderr[-3000:])
        return None, build.stderr, build.returncode

    run = subprocess.run(
        ["./nvfp4_b200_gemm.out"], cwd=workdir,
        capture_output=True, text=True, timeout=timeout
    )
    return run.stdout, run.stderr, run.returncode


@app.function(image=image, gpu="B200", timeout=1800)
def reproduce_race():
    workdir = "/root/ThunderKittens/kernels/gemm/nvfp4_b200"
    src_path = f"{workdir}/nvfp4_b200_gemm.cu"

    with open(src_path) as f:
        base_src = f.read()

    variants = [
        ("unfixed_nanosleep", "UNFIXED + nanosleep (should CRASH or corrupt)"),
        ("fixed_nanosleep",   "FIXED + nanosleep (should work correctly)"),
    ]

    results = {}
    for variant_key, variant_label in variants:
        print("\n" + "=" * 70)
        print(f"  {variant_label}")
        print("=" * 70 + "\n")

        patched = patch_source(base_src, variant_key)
        with open(src_path, "w") as f:
            f.write(patched)

        stdout, stderr, rc = build_and_run(variant_key, workdir, timeout=600)
        results[variant_key] = (stdout, stderr, rc)

        if stdout:
            # Print last 2000 chars
            print(stdout[-2000:] if len(stdout) > 2000 else stdout)
        if rc != 0:
            print(f"\n>>> EXIT CODE: {rc}")
            if stderr:
                print(f">>> STDERR (last 1000 chars):\n{stderr[-1000:]}")

    # Restore original source
    with open(src_path, "w") as f:
        f.write(base_src)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for variant_key, variant_label in variants:
        stdout, stderr, rc = results[variant_key]
        if rc == 0:
            status = "PASSED (all trials OK)"
        elif rc is None:
            status = "BUILD FAILED"
        elif rc < 0:
            import signal
            try:
                sig = signal.Signals(-rc).name
            except (ValueError, AttributeError):
                sig = f"signal {-rc}"
            status = f"CRASHED ({sig})"
        else:
            status = f"FAILED (exit code {rc})"

        print(f"  {variant_label}")
        print(f"    Result: {status}")

    print()
    return 0


@app.local_entrypoint()
def main():
    exit_code = reproduce_race.remote()
    if exit_code != 0:
        print(f"Failed with exit code {exit_code}")
    else:
        print("Done!")
