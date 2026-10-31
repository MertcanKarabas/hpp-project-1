"""
Phase 2C: the same operation -- the similarity matrix S = Q X^T -- computed four ways
(plus two Numba extras that explain *why* NumPy wins on big matrices).

  1. Python loops               element-wise, Python lists
  2. NumPy                      Q @ X.T  (one BLAS call; Apple Accelerate on this Mac)
  3. Numba @njit                the Python loops compiled to machine code
  4. Numba parallel + prange    documents split across all cores
  extra: Numba fastmath         float32 accumulator, lets the compiler use SIMD
  extra: Numba parallel+fastmath

Problem sizes go from tiny to large so we can see where each one wins:
  (queries x docs) = 1 x 1K, 20 x 1K, 20 x 20K, 200 x 20K   (768 dims, mpnet)
Pure Python is skipped at 200 x 20K (~1 min per run); its time there is extrapolated
from 20 x 20K (cost is linear in queries x docs x dims).

Second experiment -- an operation with no BLAS behind it: top-10 selection per query
from the 200 x 20,000 score matrix. NumPy (argpartition + sort of the 10) vs our Numba
loop (one pass, small sorted buffer) vs the same loop with prange over queries.

Outputs: results/phase2c_vectorise.csv, results/phase2c_topk.csv, figures/fig4_vectorise.png

Usage:
  .venv/bin/python src/phase2c_vectorise.py              # both experiments (~3 min)
  .venv/bin/python src/phase2c_vectorise.py --topk-only  # only the top-k experiment
  .venv/bin/python src/phase2c_vectorise.py --plot-only
"""

import csv
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np

import kernels as K
from benchmark_utils import benchmark, save_row
from data_utils import FIGURES, RESULTS, load_dense

SIZES = [(1, 1_000), (20, 1_000), (20, 20_000), (200, 20_000)]
PYTHON_SKIP = {(200, 20_000)}
OUT = RESULTS / "phase2c_vectorise.csv"
OUT_TOPK = RESULTS / "phase2c_topk.csv"
TOPK = 10
TOPK_VARIANTS = [
    ("NumPy argpartition + sort", K.topk_numpy),
    ("Numba @njit", K.topk_rows),
    ("Numba parallel (prange)", K.topk_rows_parallel),
]

VARIANTS = [
    # name, function, required by the assignment?
    ("Python loops", None, True),
    ("NumPy (Q @ X.T)", lambda Q, X: Q @ X.T, True),
    ("Numba @njit", K.dense_numba, True),
    ("Numba parallel (prange)", K.dense_numba_parallel, True),
    ("Numba fastmath", K.dense_numba_fastmath, False),
    ("Numba parallel + fastmath", K.dense_numba_parallel_fastmath, False),
]

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
EXTRA = "#a3a29c"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def compile_time(func, Q, X):
    """First call = Numba compiles the function for these argument types."""
    t0 = time.perf_counter()
    func(Q[:1], X[:4])
    return time.perf_counter() - t0


def run():
    X_all, Q_all = load_dense("mpnet")
    if OUT.exists():
        OUT.unlink()
    threads = numba.get_num_threads()

    compile_s = {name: compile_time(f, Q_all, X_all) for name, f, _ in VARIANTS if "Numba" in name}
    for name, s in compile_s.items():
        print(f"compile {name:28s} {s:.2f}s")

    python_s_per_mac = None
    for nq, n in SIZES:
        Q, X = Q_all[:nq], X_all[:n]
        d = Q.shape[1]
        macs = nq * n * d
        ref = Q.astype(np.float64) @ X.astype(np.float64).T
        times = {}
        for name, func, required in VARIANTS:
            if name == "Python loops":
                if (nq, n) in PYTHON_SKIP:
                    times[name] = python_s_per_mac * macs
                    print(f"  {nq:>3} x {n:>6}  {name:28s} skipped, estimated {times[name]:.1f}s")
                    save_row(OUT, row(name, nq, n, d, None, times[name], np.nan, threads, required, True, None))
                    continue
                Ql, Xl = Q.tolist(), X.tolist()
                r = benchmark(K.dense_python, Ql, Xl, warmup=1, runs=5)
                S = np.array(r["result"])
                python_s_per_mac = r["median_s"] / macs
            else:
                r = benchmark(func, Q, X, warmup=2, runs=5)
                S = r["result"]
            err = float(np.max(np.abs(S - ref)))
            times[name] = r["median_s"]
            save_row(OUT, row(name, nq, n, d, r, r["median_s"], err, threads, required, False, compile_s.get(name)))
            print(f"  {nq:>3} x {n:>6}  {name:28s} {r['median_s'] * 1000:10.3f} ms  "
                  f"{2 * macs / r['median_s'] / 1e9:8.2f} GFLOP/s  err {err:.1e}", flush=True)

    add_speedups()
    run_topk()
    plot()


def run_topk():
    """Top-10 per query from the full 200 x 20K score matrix: NumPy vs our Numba loop."""
    X, Q = load_dense("mpnet")
    S = Q @ X.T
    ref = -np.sort(-S.astype(np.float64), axis=1)[:, :TOPK]
    matmul_s = benchmark(lambda A, B: A @ B.T, Q, X, warmup=2, runs=5)["median_s"]
    if OUT_TOPK.exists():
        OUT_TOPK.unlink()
    print(f"top-{TOPK} from a {S.shape[0]} x {S.shape[1]} score matrix (matmul itself: {matmul_s * 1000:.2f} ms)")
    for name, func in TOPK_VARIANTS:
        func(S[:2, :100], TOPK)                          # compile / warm the code path
        r = benchmark(func, S, TOPK, warmup=2, runs=5)
        err = float(np.max(np.abs(r["result"][1] - ref)))
        save_row(OUT_TOPK, {
            "variant": name, "n_queries": S.shape[0], "n_docs": S.shape[1], "k": TOPK,
            "warmup": r["warmup"], "runs": r["runs"], "median_s": r["median_s"], "std_s": r["std_s"],
            "matmul_s": matmul_s, "max_abs_err": err, "times": r["times"],
        })
        print(f"  {name:28s} {r['median_s'] * 1000:8.3f} ms   err {err:.1e}", flush=True)


def row(name, nq, n, d, r, median, err, threads, required, estimated, compile_s):
    macs = nq * n * d
    return {
        "variant": name, "required": required, "n_queries": nq, "n_docs": n, "dim": d,
        "warmup": r["warmup"] if r else "", "runs": r["runs"] if r else "",
        "median_s": median, "std_s": r["std_s"] if r else "", "min_s": r["min_s"] if r else "",
        "gflops": 2 * macs / median / 1e9, "max_abs_err": err, "estimated": estimated,
        "numba_threads": threads if "parallel" in name else 1,
        "compile_s": compile_s if compile_s is not None else "",
        "times": r["times"] if r else "",
    }


def load_rows():
    with open(OUT) as f:
        return list(csv.DictReader(f))


def add_speedups():
    rows = load_rows()
    base = {(r["n_queries"], r["n_docs"]): float(r["median_s"]) for r in rows if r["variant"] == "Python loops"}
    npy = {(r["n_queries"], r["n_docs"]): float(r["median_s"]) for r in rows if r["variant"].startswith("NumPy")}
    for r in rows:
        key = (r["n_queries"], r["n_docs"])
        r["speedup_vs_python"] = base[key] / float(r["median_s"])
        r["speedup_vs_numpy"] = npy[key] / float(r["median_s"])
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def style(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)


def plot():
    rows = load_rows()
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw={"width_ratios": [1.15, 1, 0.8]})
    fig.patch.set_facecolor(SURFACE)

    # (a) one size, every variant
    sel = [r for r in rows if r["n_queries"] == "20" and r["n_docs"] == "20000"][::-1]
    names = [r["variant"] for r in sel]
    vals = [float(r["median_s"]) * 1000 for r in sel]
    colors = [BLUE if r["required"] == "True" else EXTRA for r in sel]
    bars = a.barh(names, vals, color=colors, height=0.55)
    a.set_xscale("log")
    for bar, r in zip(bars, sel):
        ms = float(r["median_s"]) * 1000
        label = f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:.2f} ms"
        a.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height() / 2,
               f"{label}   {float(r['speedup_vs_python']):,.0f}x", va="center", color=INK, fontsize=10)
    a.set_xlim(min(vals) * 0.5, max(vals) * 15)
    a.set_xlabel("Milliseconds (log scale), 20 queries x 20,000 docs x 768 dims", color=INK2)
    a.set_title("(a) Time and speedup vs Python loops", color=INK, fontsize=11, loc="left")
    style(a)
    a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=EXTRA)],
             labels=["Required by the assignment", "Extra (explains the gap)"],
             loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=9, labelcolor=INK2)

    # (b) throughput across problem sizes
    labels = [f"{nq} x {n // 1000}K" for nq, n in SIZES]
    series = [("NumPy (Q @ X.T)", BLUE), ("Numba @njit", ORANGE),
              ("Numba parallel (prange)", AQUA), ("Numba parallel + fastmath", YELLOW)]
    for name, color in series:
        ys = []
        for nq, n in SIZES:
            r = next(r for r in rows if r["variant"] == name and r["n_queries"] == str(nq) and r["n_docs"] == str(n))
            ys.append(float(r["gflops"]))
        b.plot(labels, ys, "-o", color=color, linewidth=2, markersize=8, label=name,
               markeredgecolor=SURFACE, markeredgewidth=2)
    b.set_yscale("log")
    b.set_xlabel("Problem size (queries x docs), 768 dims", color=INK2)
    b.set_ylabel("GFLOP/s (log scale)", color=INK2)
    b.set_title("(b) Matrix multiply throughput by problem size", color=INK, fontsize=11, loc="left")
    style(b, "y")
    b.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)

    # (c) top-k: a loop NumPy cannot hand to BLAS
    with open(OUT_TOPK) as f:
        tk = list(csv.DictReader(f))
    names = [r["variant"] for r in tk][::-1]
    vals = [float(r["median_s"]) * 1000 for r in tk][::-1]
    colors = [BLUE if n.startswith("NumPy") else AQUA if "parallel" in n else ORANGE for n in names]
    bars = c.barh(names, vals, color=colors, height=0.55)
    mm = float(tk[0]["matmul_s"]) * 1000
    for bar, v in zip(bars, vals):
        c.text(max(bar.get_width(), mm) + max(vals) * 0.03, bar.get_y() + bar.get_height() / 2, f"{v:.2f} ms",
               va="center", color=INK, fontsize=10)
    c.axvline(mm, color=INK2, linewidth=1, linestyle="--")
    c.text(mm, len(names) - 1.5, f"  Q @ X.T itself: {mm:.1f} ms", color=INK2, fontsize=9, va="center")
    c.set_xlim(0, max(vals) * 1.35)
    c.set_xlabel(f"Milliseconds, top-{TOPK} of 200 x 20,000 scores", color=INK2)
    c.set_title("(c) Top-k selection: no BLAS to call", color=INK, fontsize=11, loc="left")
    style(c)

    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig4_vectorise.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("saved figures/fig4_vectorise.png")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot()
    elif "--topk-only" in sys.argv:
        run_topk()
        plot()
    else:
        run()
