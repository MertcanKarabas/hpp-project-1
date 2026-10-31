"""
Phase 3 figures and Amdahl's Law fit (reads results/phase3_parallel.csv).

Amdahl's Law: if a fraction P of the work can run in parallel, W workers give at most
    speedup(W) = 1 / ((1 - P) + P / W)        and never more than 1 / (1 - P).
We fit P to the measured speedups by least squares (grid search over P in [0, 1]).

Outputs: results/phase3_amdahl.csv, results/phase3_summary.csv, figures/fig5_parallel.png
"""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_utils import FIGURES, RESULTS

IN = RESULTS / "phase3_parallel.csv"
OUT_AMDAHL = RESULTS / "phase3_amdahl.csv"
OUT_SUMMARY = RESULTS / "phase3_summary.csv"

NAMES = {
    "numba_prange": "Numba prange", "threadpool": "ThreadPoolExecutor", "processpool": "ProcessPoolExecutor",
    "dask": "Dask (threads)", "ray": "Ray tasks", "processpool_pickle": "ProcessPool, chunk pickled per task",
    "asyncio_executor": "asyncio + executor", "blas_threads": "BLAS threads (one matmul)",
    "asyncio": "asyncio, coroutines only", "phase2_best": "Phase 2 best (default BLAS + Numba top-k)",
}
COLORS = {"numba_prange": "#2a78d6", "threadpool": "#eb6834", "processpool": "#1baf7a", "dask": "#eda100",
          "ray": "#e87ba4", "processpool_pickle": "#008300", "asyncio_executor": "#6250d6",
          "blas_threads": "#e34948"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
P_CORES = 4


def load():
    with open(IN) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["workers"] = int(r["workers"])
        r["median_s"] = float(r["median_s"])
        r["setup_s"] = float(r["setup_s"])
    return rows


def series(rows, workload, strategy):
    pts = sorted((r["workers"], r["median_s"]) for r in rows if r["workload"] == workload and r["strategy"] == strategy)
    return [w for w, _ in pts], [t for _, t in pts]


def amdahl(W, P):
    return 1.0 / ((1.0 - P) + P / np.asarray(W, dtype=float))


def fit_P(W, speedup):
    grid = np.linspace(0, 1, 10_001)
    err = [np.sum((amdahl(W, P) - speedup) ** 2) for P in grid]
    return float(grid[int(np.argmin(err))])


def analyse(rows):
    out = []
    for workload in ("cpu", "python", "blas"):
        for strategy in NAMES:
            W, T = series(rows, workload, strategy)
            if len(W) < 3:
                continue
            s = np.array(T[0]) / np.array(T)
            P = fit_P(W, s)
            best = int(np.argmax(s))
            out.append({"workload": workload, "strategy": strategy, "t1_ms": T[0] * 1000,
                        "best_workers": W[best], "best_ms": T[best] * 1000, "max_speedup": float(s[best]),
                        "speedup_at_4": float(s[W.index(4)]), "speedup_at_10": float(s[W.index(10)]),
                        "amdahl_P": P, "amdahl_limit": (1 / (1 - P)) if P < 1 else float("inf"),
                        "speedups": json.dumps([round(x, 3) for x in s])})
    with open(OUT_AMDAHL, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    return out


def summary(rows):
    """Best configuration of every strategy vs the Phase 1 baseline and the best Phase 2 version."""
    with open(RESULTS / "phase1_baseline.csv") as f:
        base = next(r for r in csv.DictReader(f) if r["n_docs"] == "20000")
    baseline_s = float(base["per_query_ms"]) * 200 / 1000          # naive Python, all 200 queries
    p2 = next(r for r in rows if r["strategy"] == "phase2_best")["median_s"]
    out = []
    for workload in ("blas", "cpu"):
        for strategy in NAMES:
            sel = [r for r in rows if r["workload"] == workload and r["strategy"] == strategy]
            if not sel:
                continue
            b = min(sel, key=lambda r: r["median_s"])
            out.append({"workload": workload, "strategy": strategy, "best_workers": b["workers"],
                        "median_ms": b["median_s"] * 1000, "setup_s": b["setup_s"],
                        "speedup_vs_naive_baseline": baseline_s / b["median_s"],
                        "speedup_vs_phase2_best": p2 / b["median_s"]})
    with open(OUT_SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    return out, baseline_s, p2


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def cores_marker(ax):
    ax.axvspan(P_CORES + 0.05, 10.5, color=GRID, alpha=0.35, linewidth=0)
    ax.text(P_CORES + 0.2, ax.get_ylim()[1] * 0.97, "efficiency cores join", color=INK2, fontsize=8.5, va="top")


def line(ax, W, y, strategy, **kw):
    ax.plot(W, y, "-o", color=COLORS[strategy], linewidth=2, markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=NAMES[strategy], **kw)


def plot(rows, fits):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(19, 5.8))
    fig.patch.set_facecolor(SURFACE)
    fitted = {(f["workload"], f["strategy"]): f["amdahl_P"] for f in fits}

    # (a) CPU-bound search: speedup vs workers
    a.plot([1, 10], [1, 10], color=INK2, linestyle="--", linewidth=1, label="Ideal (speedup = workers)")
    for s in ("numba_prange", "threadpool", "processpool", "dask", "ray"):
        W, T = series(rows, "cpu", s)
        line(a, W, T[0] / np.array(T), s)
    best = max((f for f in fits if f["workload"] == "cpu"), key=lambda f: f["max_speedup"])
    Wf = np.linspace(1, 10, 100)
    a.plot(Wf, amdahl(Wf, best["amdahl_P"]), color=COLORS[best["strategy"]], linestyle=":", linewidth=1.5,
           label=f"Amdahl fit, {NAMES[best['strategy']]}: P = {best['amdahl_P']:.2f}")
    a.set_xlim(0.5, 10.5)
    a.set_ylim(0, 10.5)
    a.set_xlabel("Workers", color=INK2)
    a.set_ylabel("Speedup vs 1 worker", color=INK2)
    a.set_title("(a) CPU-bound search (Numba scoring + top-k)", color=INK, fontsize=11, loc="left")
    style(a)
    cores_marker(a)
    a.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left", bbox_to_anchor=(0, 0.93))

    # (b) pure Python: the GIL
    b.plot([1, 10], [1, 10], color=INK2, linestyle="--", linewidth=1, label="Ideal")
    for s in ("threadpool", "processpool"):
        W, T = series(rows, "python", s)
        line(b, W, T[0] / np.array(T), s)
        P = fitted.get(("python", s))
        b.annotate(f"P = {P:.2f}", (W[-1], T[0] / T[-1]), xytext=(6, 0), textcoords="offset points",
                   color=COLORS[s], fontsize=9, va="center")
    b.set_xlim(0.5, 11.3)
    b.set_ylim(0, 10.5)
    b.set_xlabel("Workers", color=INK2)
    b.set_ylabel("Speedup vs 1 worker", color=INK2)
    b.set_title("(b) Pure-Python search: threads vs processes (GIL)", color=INK, fontsize=11, loc="left")
    style(b)
    cores_marker(b)
    b.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left", bbox_to_anchor=(0, 0.93))

    # (c) BLAS search: absolute time
    for s in ("blas_threads", "threadpool", "processpool", "dask", "ray", "asyncio_executor", "processpool_pickle"):
        W, T = series(rows, "blas", s)
        line(c, W, np.array(T) * 1000, s)
    p2 = next(r for r in rows if r["strategy"] == "phase2_best")["median_s"] * 1000
    c.axhline(p2, color=INK, linestyle="--", linewidth=1)
    c.text(10.4, p2, f"Phase 2 best: {p2:.1f} ms", color=INK, fontsize=9, va="bottom", ha="right")
    aio = [r for r in rows if r["strategy"] == "asyncio"]
    if aio:
        t = min(r["median_s"] for r in aio) * 1000
        c.axhline(t, color=COLORS["asyncio_executor"], linestyle=":", linewidth=1)
        c.text(10.4, t, f"asyncio, coroutines only: {t:.1f} ms", color=COLORS["asyncio_executor"], fontsize=9,
               va="bottom", ha="right")
    c.set_yscale("log")
    c.set_xlim(0.5, 10.5)
    c.set_xlabel("Workers", color=INK2)
    c.set_ylabel("Milliseconds for 200 queries (log scale)", color=INK2)
    c.set_title("(c) BLAS search (final system): time vs workers", color=INK, fontsize=11, loc="left")
    style(c)
    c.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig5_parallel.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("saved figures/fig5_parallel.png")


def main():
    rows = load()
    fits = analyse(rows)
    summ, baseline_s, p2 = summary(rows)
    print(f"naive baseline (200 queries): {baseline_s:.1f} s | Phase 2 best: {p2 * 1000:.2f} ms")
    for f in fits:
        print(f"  {f['workload']:6s} {f['strategy']:18s} T1 {f['t1_ms']:9.2f} ms  best {f['best_ms']:9.2f} ms "
              f"@{f['best_workers']:>2}  max speedup {f['max_speedup']:5.2f}x  Amdahl P {f['amdahl_P']:.3f}")
    plot(rows, fits)


if __name__ == "__main__":
    main()
