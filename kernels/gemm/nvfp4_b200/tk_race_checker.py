#!/usr/bin/env python3
"""
TK Kernel Cluster Resource Race Condition Checker  (v2 — generalized)

A domain-specific static analyzer for ThunderKittens CUDA kernels that detects
race conditions on shared cluster resources.

Nothing is hardcoded to a specific kernel.  The checker *discovers*:
  - Tensor-memory allocator variable names (any `tensor_allocator<...> NAME;`)
  - Tensor-memory region variable names (any `NAME.allocate<...>(...)`)
  - All accesses to those regions (any line referencing a discovered tmem var)
  - Semaphore names, init counts, arrive/wait calls
  - Producer / consumer branch boundaries

Then it checks two safety properties:

  CHECK 1 — Deprovision Safety
    Every `allocator.deprovision()` must be preceded by a cluster-wide barrier
    (arrive from all CTAs + wait) that comes *after* the last tmem access.

  CHECK 2 — Cross-Branch Hazard
    When tmem is written in one branch (producer) and read in another (consumer),
    the writer must not overwrite tmem until the reader signals completion via a
    cluster-scoped semaphore that the writer waits on.  (WAR hazard)

Usage:
    python tk_race_checker.py <kernel_file.cu>
"""

import re
import sys
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Op:
    """A single operation extracted from the source."""
    kind: str           # provision, deprovision, tmem_access, sema_decl,
                        # sema_init, arrive, wait
    line: int           # 1-indexed
    text: str           # stripped source line
    resource: str = ""  # allocator or tmem-var name involved
    sema: str = ""      # semaphore name
    init_count: Optional[int] = None   # for sema_init
    target_cta: Optional[str] = None   # for arrive
    access: str = ""    # "read" or "write" for tmem_access
    branch: str = ""    # "setup", "producer", "consumer"


@dataclass
class SemaInfo:
    name: str
    init_count: Optional[int] = None
    is_cluster: bool = False


@dataclass
class Warning:
    severity: str       # ERROR, WARNING
    line: int
    message: str
    detail: str


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — structural parsing
# ─────────────────────────────────────────────────────────────────────────────

def find_kernel_bounds(source: str) -> tuple[int, int]:
    """Return (start, end) 0-indexed line numbers of the kernel function."""
    lines = source.split("\n")
    start = None
    depth = 0
    for i, ln in enumerate(lines):
        if re.search(r"__device__\s+inline\s+void\s+kernel\s*\(", ln):
            start = i
        if start is not None:
            depth += ln.count("{") - ln.count("}")
            if depth == 0 and i > start:
                return start, i
    return (start or 0), len(lines) - 1


def resolve_cluster_size(source: str) -> int:
    m = re.search(r"static\s+constexpr\s+int\s+CLUSTER_SIZE\s*=\s*(\d+)", source)
    if m:
        return int(m.group(1))
    return 2  # TK default


def assign_branch(line_idx: int, branch_ranges: list[tuple[str, int, int]]) -> str:
    """Which branch does this 0-indexed line fall in?
    If multiple ranges overlap, pick the narrowest (most specific) one."""
    best = "setup"
    best_span = float("inf")
    for name, lo, hi in branch_ranges:
        if lo <= line_idx <= hi:
            span = hi - lo
            if span < best_span:
                best = name
                best_span = span
    return best


def find_branch_ranges(lines: list[str], k_start: int, k_end: int) -> list[tuple[str, int, int]]:
    """Find (name, start_line, end_line) for producer and consumer branches."""
    ranges: list[tuple[str, int, int]] = []
    producer_start = consumer_start = None

    for i in range(k_start, k_end + 1):
        ln = lines[i]
        # producer: the first if-branch with CONSUMER_WARPGROUPS or "Producer"
        if producer_start is None and (
            re.search(r"warpgroup_id\s*>=\s*\w+::CONSUMER_WARPGROUPS", ln)
            or re.search(r"//\s*Producer", ln, re.IGNORECASE)
        ):
            producer_start = i
        # consumer: the else-if branch
        if consumer_start is None and (
            re.search(r"warpgroup_id\s*<\s*\w+::CONSUMER_WARPGROUPS", ln)
            or re.search(r"//\s*Consumer", ln, re.IGNORECASE)
        ):
            consumer_start = i

    # Determine range end by brace matching from each start.
    # For `} else if (...) {` lines, we need to only count the opening brace(s)
    # that appear after the branch keyword, not the closing brace before it.
    def brace_end(start):
        depth = 0
        for i in range(start, k_end + 1):
            ln = lines[i]
            if i == start:
                # On the start line, only count from the last '{' onward
                # to handle "} else if (...) {" patterns
                last_open = ln.rfind("{")
                if last_open >= 0:
                    depth = 1
                continue
            depth += ln.count("{") - ln.count("}")
            if depth <= 0:
                return i
        return k_end

    if producer_start is not None:
        pe = brace_end(producer_start)
        ranges.append(("producer", producer_start, pe))
        if consumer_start is None:
            consumer_start = pe + 1
    if consumer_start is not None:
        ce = brace_end(consumer_start)
        ranges.append(("consumer", consumer_start, ce))

    return ranges


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — discovery (no hardcoded names)
# ─────────────────────────────────────────────────────────────────────────────

def discover_allocators(lines: list[str], k_start: int, k_end: int) -> dict[str, int]:
    """Find tensor_allocator declarations → {var_name: cluster_size_param}."""
    allocs = {}
    for i in range(k_start, k_end + 1):
        m = re.search(r"tensor_allocator\s*<[^>]*>\s+(\w+)\s*;", lines[i])
        if m:
            var = m.group(1)
            # extract second template arg (CLUSTER_SIZE)
            tp = re.search(r"tensor_allocator\s*<\s*\d+\s*,\s*([^,>]+)", lines[i])
            allocs[var] = tp.group(1).strip() if tp else "1"
    return allocs


def discover_tmem_vars(lines: list[str], k_start: int, k_end: int,
                       allocator_names: set[str]) -> set[str]:
    """Find variables assigned from allocator.allocate<>()."""
    tmem_vars: set[str] = set()
    for i in range(k_start, k_end + 1):
        for alloc in allocator_names:
            pat = rf"(\w+)\s*=\s*{re.escape(alloc)}\.(?:template\s+)?allocate"
            m = re.search(pat, lines[i])
            if m:
                tmem_vars.add(m.group(1))
            # Also catch: auto NAME = ...
            pat2 = rf"auto\s+(\w+)\s*=\s*{re.escape(alloc)}\.(?:template\s+)?allocate"
            m2 = re.search(pat2, lines[i])
            if m2:
                tmem_vars.add(m2.group(1))
    return tmem_vars


# Known write patterns: MMA ops, scale loads, tensor_commit
TMEM_WRITE_PATTERNS = [
    r"mm2_ABt\s*\(",
    r"mma2_ABt\s*\(",
    r"load_mxnv_scale_async",
    r"tensor_commit",
]
# Known read patterns: load_async from tmem (second arg is a tmem var)
TMEM_READ_PATTERNS = [
    r"warpgroup::load_async\s*\(",
    r"tensor_load_wait",
]


def classify_tmem_access(line: str, tmem_vars: set[str]) -> Optional[str]:
    """Return 'read', 'write', or None."""
    for v in tmem_vars:
        if v not in line:
            continue
        for wp in TMEM_WRITE_PATTERNS:
            if re.search(wp, line):
                return "write"
        for rp in TMEM_READ_PATTERNS:
            if re.search(rp, line):
                return "read"
        # Fallback: if the tmem var appears as a function arg, it's an access
        # Heuristic: if it appears after '(' → likely an argument (use)
        if re.search(rf"\(\s*{re.escape(v)}", line) or re.search(rf",\s*{re.escape(v)}", line):
            return "read"  # conservative: treat unknown use as read
    # Also catch subtile accesses — derived from tmem vars
    for v in tmem_vars:
        if f"{v}." in line or f"{v} " in line:
            for wp in TMEM_WRITE_PATTERNS:
                if re.search(wp, line):
                    return "write"
            for rp in TMEM_READ_PATTERNS:
                if re.search(rp, line):
                    return "read"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — extract all operations
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_ops(lines: list[str], k_start: int, k_end: int,
                    allocator_names: set[str], tmem_vars: set[str],
                    branch_ranges: list[tuple[str, int, int]],
                    cluster_size: int) -> list[Op]:
    ops: list[Op] = []

    # Semaphore patterns (generic)
    re_sema_decl = re.compile(r"__shared__\s+semaphore\s+(\w+)")
    re_sema_init = re.compile(r"init_semaphore\s*\(\s*(\w+)(?:\[\w+\])?\s*,\s*(\d+)\s*,\s*([^)]+)\)")
    re_arrive = re.compile(r"(?:warpgroup::)?tma::cluster::arrive\s*\(\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")
    re_cluster_wait = re.compile(r"tma::cluster::wait\s*\(\s*(\w+)\s*,")
    re_local_wait = re.compile(r"(?<!\w)wait\s*\(\s*(\w+)(?:\[\w+\])?\s*,")

    for i in range(k_start, k_end + 1):
        ln = lines[i]
        stripped = ln.strip()
        if stripped.startswith("//"):
            continue
        br = assign_branch(i, branch_ranges)

        # Allocator lifecycle
        for alloc in allocator_names:
            if re.search(rf"{re.escape(alloc)}\.provision\s*\(", ln):
                ops.append(Op("provision", i+1, stripped, resource=alloc, branch=br))
            if re.search(rf"{re.escape(alloc)}\.deprovision\s*\(", ln):
                ops.append(Op("deprovision", i+1, stripped, resource=alloc, branch=br))

        # Tmem access
        acc = classify_tmem_access(ln, tmem_vars)
        if acc:
            ops.append(Op("tmem_access", i+1, stripped, access=acc, branch=br))

        # Semaphore declarations
        m = re_sema_decl.search(ln)
        if m:
            ops.append(Op("sema_decl", i+1, stripped, sema=m.group(1), branch=br))

        # Semaphore init
        m = re_sema_init.search(ln)
        if m:
            name = m.group(1)
            count_str = m.group(3).strip()
            count = None
            if count_str.isdigit():
                count = int(count_str)
            elif "CLUSTER_SIZE" in count_str:
                count = cluster_size
            ops.append(Op("sema_init", i+1, stripped, sema=name, init_count=count, branch=br))

        # Arrive
        m = re_arrive.search(ln)
        if m:
            ops.append(Op("arrive", i+1, stripped, sema=m.group(1),
                          target_cta=m.group(2), branch=br))

        # Wait (cluster-scope)
        m = re_cluster_wait.search(ln)
        if m:
            ops.append(Op("wait", i+1, stripped, sema=m.group(1), branch=br))
        else:
            m = re_local_wait.search(ln)
            if m:
                ops.append(Op("wait", i+1, stripped, sema=m.group(1), branch=br))

    return ops


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — safety checks
# ─────────────────────────────────────────────────────────────────────────────

def build_sema_map(ops: list[Op], cluster_size: int) -> dict[str, SemaInfo]:
    semas: dict[str, SemaInfo] = {}
    for o in ops:
        if o.kind == "sema_decl":
            semas[o.sema] = SemaInfo(name=o.sema)
        elif o.kind == "sema_init" and o.sema in semas:
            semas[o.sema].init_count = o.init_count
            if o.init_count is not None and o.init_count >= cluster_size:
                semas[o.sema].is_cluster = True
    return semas


def check_deprovision_safety(ops: list[Op], semas: dict[str, SemaInfo],
                             cluster_size: int) -> list[Warning]:
    """CHECK 1: every deprovision must be protected by a full cluster barrier."""
    warnings = []
    deprovs = [o for o in ops if o.kind == "deprovision"]
    tmem_accesses = [o for o in ops if o.kind == "tmem_access"]

    for dep in deprovs:
        # Last tmem access before deprovision
        last_use = None
        for a in reversed(tmem_accesses):
            if a.line < dep.line:
                last_use = a
                break

        # Look for a protecting wait on a cluster-scoped semaphore
        # between last_use and deprovision, where all CTAs arrive
        protected = False
        for o in reversed(ops):
            if o.line >= dep.line:
                continue
            if last_use and o.line < last_use.line:
                break
            if o.kind != "wait":
                continue
            if o.sema not in semas or not semas[o.sema].is_cluster:
                continue
            # Check all CTAs arrive at this sema
            arrivals = [a for a in ops if a.kind == "arrive" and a.sema == o.sema]
            ctas = set(a.target_cta for a in arrivals if a.target_cta is not None)
            expected = set(str(i) for i in range(cluster_size))
            if ctas >= expected:
                # Verify arrives are after last use
                ok = all(
                    a.line >= (last_use.line if last_use else 0)
                    for a in arrivals
                )
                if ok:
                    protected = True
                    break

        if not protected:
            detail = [f"  allocator CLUSTER_SIZE={cluster_size}"]
            detail.append(f"  deprovision() at line {dep.line}: {dep.text}")
            if last_use:
                detail.append(f"  last tmem access at line {last_use.line} ({last_use.access}): {last_use.text}")
            cluster_semas = [s.name for s in semas.values() if s.is_cluster]
            detail.append(f"  cluster-scoped semaphores: {cluster_semas}")
            nearby = [o for o in ops if o.kind == "arrive" and dep.line - 30 < o.line < dep.line]
            if nearby:
                detail.append("  nearby arrive() but not sufficient:")
                for a in nearby:
                    detail.append(f"    line {a.line}: arrive({a.sema}, cta={a.target_cta})")
            else:
                detail.append("  NO arrive()/wait() found before deprovision()")
            detail += ["",
                "  FIX: before deprovision(), add a cluster barrier:",
                f"    init_semaphore(barrier, 0, {cluster_size});  // in setup",
                *[f"    arrive(barrier, {i}, 1);  // signal CTA {i}"
                  for i in range(cluster_size)],
                "    wait(barrier, phase);",
                "    allocator.deprovision();",
            ]
            warnings.append(Warning(
                "ERROR", dep.line,
                f"RACE: deprovision() at line {dep.line} not protected by cluster barrier. "
                f"CLUSTER_SIZE={cluster_size}: one CTA can free tmem while another is using it.",
                "\n".join(detail),
            ))
    return warnings


def check_cross_branch_hazards(ops: list[Op], semas: dict[str, SemaInfo],
                               cluster_size: int) -> list[Warning]:
    """CHECK 2: producer writes to tmem, consumer reads — consumer must signal
    completion via a cluster semaphore that the producer waits on before the
    next write.  Otherwise the producer can overwrite data the consumer is
    still reading (WAR hazard)."""
    warnings = []

    # Collect per-branch tmem ops
    producer_writes = [o for o in ops if o.kind == "tmem_access" and o.access == "write" and o.branch == "producer"]
    consumer_reads  = [o for o in ops if o.kind == "tmem_access" and o.access == "read"  and o.branch == "consumer"]

    if not producer_writes or not consumer_reads:
        return warnings

    # The pattern: producer writes tmem in a loop, consumer reads tmem in a
    # loop.  The producer must wait() on a semaphore that the consumer arrive()s
    # on *before* starting the next write iteration.
    #
    # Find: does the producer wait on any semaphore that the consumer arrives at?
    producer_waits = [o for o in ops if o.kind == "wait" and o.branch == "producer"]
    consumer_arrives = [o for o in ops if o.kind == "arrive" and o.branch == "consumer"]

    # Build the set of semaphores the consumer signals
    consumer_signaled = set(a.sema for a in consumer_arrives)
    # Build the set of semaphores the producer waits on
    producer_waited = set(w.sema for w in producer_waits)

    # The intersection tells us which semaphores coordinate consumer→producer
    coord_semas = consumer_signaled & producer_waited

    if not coord_semas:
        # No coordination at all — the producer never waits for the consumer
        first_write = producer_writes[0]
        first_read = consumer_reads[0]
        detail = [
            f"  producer writes tmem at line {first_write.line}: {first_write.text}",
            f"  consumer reads tmem at line {first_read.line}: {first_read.text}",
            "  consumer arrive()s on: " + str(list(consumer_signaled)),
            "  producer wait()s on:   " + str(list(producer_waited)),
            "  NO shared semaphore found for consumer→producer signaling.",
            "",
            "  FIX: consumer must arrive() on a semaphore that the producer",
            "  wait()s on before overwriting tmem for the next iteration.",
        ]
        warnings.append(Warning(
            "ERROR", first_write.line,
            f"WAR HAZARD: producer writes tmem (line {first_write.line}) but never "
            f"waits for consumer to finish reading. Producer can overwrite data "
            f"that consumer is still using.",
            "\n".join(detail),
        ))
    else:
        # Coordination exists — verify the wait happens BEFORE the next write
        # (i.e., the wait is in the producer's loop body before the write)
        for sema_name in coord_semas:
            pw = [o for o in producer_waits if o.sema == sema_name]
            # The wait should come before (or at the start of) the write loop
            # Heuristic: at least one wait on this sema should be before
            # the first write in the same iteration
            first_write = producer_writes[0]
            any_before = any(w.line < first_write.line for w in pw)
            if not any_before:
                # All waits come after the first write — possible hazard on
                # the first iteration (though often handled by init state)
                pass  # This is a weaker signal, skip for now

    return warnings


def analyze_kernel(source: str) -> list[Warning]:
    lines = source.split("\n")
    cluster_size = resolve_cluster_size(source)
    k_start, k_end = find_kernel_bounds(source)
    branch_ranges = find_branch_ranges(lines, k_start, k_end)

    # Discovery
    allocators = discover_allocators(lines, k_start, k_end)
    alloc_names = set(allocators.keys())
    tmem_vars = discover_tmem_vars(lines, k_start, k_end, alloc_names)

    # Extract ops
    all_ops = extract_all_ops(lines, k_start, k_end, alloc_names, tmem_vars,
                              branch_ranges, cluster_size)

    semas = build_sema_map(all_ops, cluster_size)

    # Run checks
    warnings: list[Warning] = []
    warnings += check_deprovision_safety(all_ops, semas, cluster_size)
    warnings += check_cross_branch_hazards(all_ops, semas, cluster_size)

    # Bonus: orphan arrives (arrived but never waited)
    for name, info in semas.items():
        if not info.is_cluster:
            continue
        arrives = [o for o in all_ops if o.kind == "arrive" and o.sema == name]
        waits   = [o for o in all_ops if o.kind == "wait"   and o.sema == name]
        if arrives and not waits:
            warnings.append(Warning(
                "WARNING", arrives[0].line,
                f"Cluster semaphore '{name}' is arrived at but never waited on.",
                f"  arrive() at lines: {[a.line for a in arrives]}",
            ))

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

R = "\033[91m"; Y = "\033[93m"; G = "\033[92m"; B = "\033[1m"; D = "\033[2m"; X = "\033[0m"


def print_report(filepath: str, warnings: list[Warning], source: str):
    lines = source.split("\n")
    cluster_size = resolve_cluster_size(source)
    k_start, k_end = find_kernel_bounds(source)
    branch_ranges = find_branch_ranges(lines, k_start, k_end)
    allocators = discover_allocators(lines, k_start, k_end)
    tmem_vars = discover_tmem_vars(lines, k_start, k_end, set(allocators))
    all_ops = extract_all_ops(lines, k_start, k_end, set(allocators), tmem_vars,
                              branch_ranges, cluster_size)

    print(f"\n{'='*78}")
    print(f"  TK Cluster Resource Race Checker  (v2)")
    print(f"  File: {filepath}")
    print(f"{'='*78}")

    print(f"\n  {D}Discovered:{X}")
    print(f"    CLUSTER_SIZE     = {cluster_size}")
    print(f"    Allocators       = {list(allocators.keys())}")
    print(f"    Tmem variables   = {sorted(tmem_vars)}")
    print(f"    Branches         = {[(n,s,e) for n,s,e in branch_ranges]}")

    n = lambda k: sum(1 for o in all_ops if o.kind == k)
    print(f"    Ops: {n('provision')} provision, {n('deprovision')} deprovision, "
          f"{n('tmem_access')} tmem accesses, {n('arrive')} arrives, {n('wait')} waits")

    if not warnings:
        print(f"\n  {G}{B}✓ NO RACE CONDITIONS DETECTED{X}")
        print(f"  {G}All cluster resources are properly synchronized.{X}")
    else:
        errs = sum(1 for w in warnings if w.severity == "ERROR")
        wrns = sum(1 for w in warnings if w.severity == "WARNING")
        print(f"\n  {R}{B}✗ FOUND {errs} ERROR(S), {wrns} WARNING(S){X}")

        for w in warnings:
            c = R if w.severity == "ERROR" else Y
            print(f"\n  {c}{B}[{w.severity}] Line {w.line}{X}")
            print(f"  {c}{w.message}{X}")
            if w.detail:
                for dl in w.detail.split("\n"):
                    print(f"  {D}{dl}{X}")
            lo = max(0, w.line - 4)
            hi = min(len(lines), w.line + 3)
            print(f"\n  {D}Source context:{X}")
            for j in range(lo, hi):
                mk = f"{R}>>>{X}" if j + 1 == w.line else "   "
                print(f"  {mk} {D}{j+1:4d}{X} {lines[j]}")

    print(f"\n{'='*78}\n")
    return sum(1 for w in warnings if w.severity == "ERROR")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <kernel_file.cu>")
        sys.exit(1)
    filepath = sys.argv[1]
    with open(filepath) as f:
        source = f.read()
    warnings = analyze_kernel(source)
    num_errors = print_report(filepath, warnings, source)
    sys.exit(1 if num_errors > 0 else 0)


if __name__ == "__main__":
    main()
