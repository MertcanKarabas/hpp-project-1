"""
One timing function for the whole project, so every benchmark follows the rules:
warmup runs first (results discarded), then >= 5 timed runs, report the median.

Why the median: one slow run (OS scheduling, another app waking up) moves the mean
but barely moves the median. We still save every raw time so the report can show
the spread.
"""

import csv
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

MIN_RUNS = 5


def benchmark(func, *args, warmup=3, runs=5, **kwargs):
    """Time func(*args, **kwargs). Returns timing stats and the last result."""
    if runs < MIN_RUNS or warmup < 1:
        raise ValueError(f"project rule: warmup >= 1 and runs >= {MIN_RUNS}")

    for _ in range(warmup):
        result = func(*args, **kwargs)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - t0)

    return {
        "median_s": statistics.median(times),
        "mean_s": statistics.mean(times),
        "std_s": statistics.stdev(times),
        "min_s": min(times),
        "max_s": max(times),
        "warmup": warmup,
        "runs": runs,
        "times": times,
        "result": result,
    }


def save_row(path, row):
    """Append one result row to a CSV file (header is written on first use)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: (";".join(f"{t:.6f}" for t in v) if isinstance(v, list) else v) for k, v in row.items()}
    new = not path.exists()
    if not new:
        with open(path) as f:
            header = next(csv.reader(f))
        if header != list(row.keys()):
            raise ValueError(f"{path.name}: row columns {list(row.keys())} do not match header {header}")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def _sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def environment_info():
    """Hardware + software facts for the report's introduction."""
    import numpy as np

    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "logical_cores": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    if sys.platform == "darwin":
        info["cpu"] = _sysctl("machdep.cpu.brand_string") or info["cpu"]
        info["performance_cores"] = _sysctl("hw.perflevel0.physicalcpu")
        info["efficiency_cores"] = _sysctl("hw.perflevel1.physicalcpu")
        mem = _sysctl("hw.memsize")
        info["ram_gb"] = round(int(mem) / 2**30, 1) if mem else None
    elif Path("/proc/meminfo").exists():
        kb = int(Path("/proc/meminfo").read_text().split()[1])
        info["ram_gb"] = round(kb / 2**20, 1)
    try:
        info["blas"] = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception:
        info["blas"] = "unknown"
    return info
