"""
Phase 4: the final system -- correctness, scaling, and the overall comparison.

A. Correctness
   - exact mode, 20K docs, all 200 queries vs the Phase 1 naive reference (tolerance 1e-5)
   - exact mode at 1K / 5K / 10K / 20K vs the naive Python search run now on 20 queries
   - fast mode: recall@10 vs the reference (it is approximate by design)
B. Scaling
   - real corpus: 1K, 5K, 10K, 20K documents
   - synthetic extension: 50K, 100K, 200K (real vectors repeated with small noise, re-normalised;
     used ONLY for timing, to see where the system bends -- the 10x question)
   - batch of 200 queries and single-query latency, exact and fast mode
   - where the time goes: matrix multiply vs top-k
   - chunked vs unchunked at 200K: time and peak memory
C. Overview ("money shot"): every version on 200 queries x 20K docs, top-10.

Outputs: results/phase4_*.{csv,json}, figures/fig6_scaling.png, figures/fig7_overview.png

Usage:
  .venv/bin/python src/phase4_final.py
  .venv/bin/python src/phase4_final.py --plot-only
"""

import csv
import json
import sys
import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import kernels as K
from benchmark_utils import benchmark, save_row
from data_utils import CORPUS_SIZES, FIGURES, RESULTS, load_dense
from phase1_naive_search import search_naive
from search_engine import SearchEngine, normalise, topk

TOP = 10
SYNTH_SIZES = [50_000, 100_000, 200_000]
OUT_CORR = RESULTS / "phase4_correctness.json"
OUT_SCALE = RESULTS / "phase4_scaling.csv"
OUT_CHUNK = RESULTS / "phase4_chunked.json"
OUT_OVER = RESULTS / "phase4_overview.csv"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def synthetic(X, n, seed=0):
    """n vectors built from the real ones: pick a random real vector, add 5% noise, re-normalise."""
    rng = np.random.default_rng(seed)
    base = X[rng.integers(0, X.shape[0], n)]
    noise = rng.standard_normal(base.shape).astype(np.float32) * (0.05 / np.sqrt(X.shape[1]))
    return np.ascontiguousarray(normalise(base + noise))


# --- A. correctness --------------------------------------------------------------------------------

def correctness(X, Q):
    ref = np.load(RESULTS / "phase1_reference_top10_mpnet.npz")
    out = {}
    exact = SearchEngine(X, mode="exact")
    idx, val = exact.search(Q, TOP)
    out["exact_20k_vs_phase1_reference"] = {
        "n_queries": int(Q.shape[0]),
        "max_abs_score_diff": float(np.max(np.abs(val - ref["scores"]))),
        "within_1e-5": bool(np.max(np.abs(val - ref["scores"])) <= 1e-5),
        "identical_top10_lists": float(np.mean([(a == b).all() for a, b in zip(idx, ref["indices"])])),
        "identical_top10_sets": float(np.mean([set(a) == set(b) for a, b in zip(idx, ref["indices"])])),
    }
    out["exact_vs_naive_run_now_20_queries"] = []
    for n in CORPUS_SIZES:
        naive = search_naive(Q[:20].tolist(), X[:n].tolist(), k=TOP)
        n_idx = np.array([[i for i, _ in r] for r in naive])
        n_val = np.array([[s for _, s in r] for r in naive])
        idx, val = SearchEngine(X[:n], mode="exact").search(Q[:20], TOP)
        out["exact_vs_naive_run_now_20_queries"].append({
            "n_docs": n, "max_abs_score_diff": float(np.max(np.abs(val - n_val))),
            "identical_top10_lists": float(np.mean([(a == b).all() for a, b in zip(idx, n_idx)])),
        })
    fast = SearchEngine(X, mode="fast")
    idx, val = fast.search(Q, TOP)
    out["fast_20k_recall_at_10"] = float(np.mean([len(set(a) & set(b)) / TOP for a, b in zip(idx, ref["indices"])]))
    OUT_CORR.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


# --- B. scaling ------------------------------------------------------------------------------------

def scaling(X, Q):
    if OUT_SCALE.exists():
        OUT_SCALE.unlink()
    with open(RESULTS / "phase1_baseline.csv") as f:
        naive_ms = {int(r["n_docs"]): float(r["per_query_ms"]) for r in csv.DictReader(f)}
    sizes = [(n, "real") for n in CORPUS_SIZES] + [(n, "synthetic") for n in SYNTH_SIZES]
    for n, kind in sizes:
        Xn = X[:n] if kind == "real" else synthetic(X, n)
        for mode in ("exact", "fast"):
            engine = SearchEngine(Xn, mode=mode)
            build = benchmark(lambda: SearchEngine(Xn, mode=mode), warmup=1, runs=5)["median_s"]
            batch = benchmark(engine.search, Q, TOP, warmup=2, runs=5)
            single = benchmark(lambda: [engine.search(Q[i], TOP) for i in range(20)], warmup=2, runs=5)
            Qp = engine.prepare(Q)
            mm = benchmark(lambda: Qp @ engine.X.T, warmup=2, runs=5)["median_s"]
            S = Qp @ engine.X.T
            tk = benchmark(topk, S, TOP, warmup=2, runs=5)["median_s"]
            pr = benchmark(engine.prepare, Q, warmup=2, runs=5)["median_s"]
            row = {
                "n_docs": n, "corpus": kind, "mode": mode,
                "batch200_ms": batch["median_s"] * 1000, "batch200_std_ms": batch["std_s"] * 1000,
                "single_query_ms": single["median_s"] / 20 * 1000,
                "matmul_ms": mm * 1000, "topk_ms": tk * 1000, "prepare_ms": pr * 1000,
                "build_s": build, "corpus_mb": engine.X.nbytes / 1e6,
                "naive_ms_200_queries": naive_ms[n] * 200 if n in naive_ms else "",
                "us_per_1k_docs_batch": batch["median_s"] * 1e6 / (n / 1000),
                "times": batch["times"],
            }
            save_row(OUT_SCALE, row)
            print(f"  {kind:9s} N={n:>7} {mode:5s}  batch {row['batch200_ms']:8.2f} ms  single {row['single_query_ms']:6.3f} ms"
                  f"  (matmul {row['matmul_ms']:.2f} + top-k {row['topk_ms']:.2f})  build {build:.2f}s", flush=True)


def chunked(X, Q, n=200_000, chunk=20_000):
    Xn = synthetic(X, n)
    out = {"n_docs": n, "chunk_docs": chunk}
    for name, engine in (("unchunked", SearchEngine(Xn)), ("chunked", SearchEngine(Xn, chunk_docs=chunk))):
        r = benchmark(engine.search, Q, TOP, warmup=2, runs=5)
        tracemalloc.start()
        engine.search(Q, TOP)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        out[name] = {"batch200_ms": r["median_s"] * 1000, "peak_search_memory_mb": peak / 1e6}
        print(f"  {name:9s} N={n}: {r['median_s'] * 1000:.2f} ms, peak search memory {peak / 1e6:.1f} MB")
    a, b = SearchEngine(Xn).search(Q, TOP), SearchEngine(Xn, chunk_docs=chunk).search(Q, TOP)
    out["max_abs_diff_chunked_vs_unchunked"] = float(np.max(np.abs(a[1] - b[1])))
    out["identical_indices_chunked_vs_unchunked"] = bool(np.array_equal(a[0], b[0]))
    OUT_CHUNK.write_text(json.dumps(out, indent=2))


# --- C. overview -------------------------------------------------------------------------------------

def overview(X, Q):
    if OUT_OVER.exists():
        OUT_OVER.unlink()
    ref = np.load(RESULTS / "phase1_reference_top10_mpnet.npz")["scores"]

    def row(name, phase, seconds, err, note=""):
        save_row(OUT_OVER, {"version": name, "phase": phase, "seconds_200_queries": seconds,
                            "max_abs_err_vs_reference": err, "note": note})
        print(f"  {name:44s} {seconds * 1000:12.2f} ms  err {err}")

    # naive Python: 5 queries timed, scaled to 200 (linear in queries, Phase 1)
    Xl, Ql = X.tolist(), Q[:5].tolist()
    r = benchmark(search_naive, Ql, Xl, TOP, warmup=1, runs=5)
    got = np.array([[s for _, s in q] for q in r["result"]])
    row("Naive Python loops (Phase 1)", 1, r["median_s"] * 40, float(np.max(np.abs(got - ref[:5]))),
        "5 queries timed, x40")

    exact, fast = SearchEngine(X), SearchEngine(X, mode="fast")
    versions = [
        ("Numba @njit loops + top-k", 2, lambda: K.topk_rows(K.dense_numba(Q, X), TOP)),
        ("Numba parallel + fastmath, parallel top-k", 2,
         lambda: K.topk_rows_parallel(K.dense_numba_parallel_fastmath(Q, X), TOP)),
        ("NumPy matmul + NumPy argpartition", 2, lambda: K.topk_numpy(Q @ X.T, TOP)),
        ("Final, exact (matmul + Numba parallel top-k)", 4, lambda: exact.search(Q, TOP)),
        ("Final, fast (SVD k=300)", 4, lambda: fast.search(Q, TOP)),
    ]
    for name, phase, f in versions:
        r = benchmark(f, warmup=2, runs=5)
        err = float(np.max(np.abs(r["result"][1] - ref)))
        row(name, phase, r["median_s"], err, "approximate (recall@10 ~0.95)" if name.startswith("Final, fast") else "")


# --- figures -----------------------------------------------------------------------------------------

def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_scaling():
    rows = load_csv(OUT_SCALE)
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(SURFACE)

    real = [r for r in rows if r["corpus"] == "real" and r["mode"] == "exact"]
    ns = [int(r["n_docs"]) for r in real]
    a.plot(ns, [float(r["naive_ms_200_queries"]) for r in real], "-o", color=INK2, linewidth=2, markersize=7,
           label="Naive Python baseline (Phase 1)")
    for mode, key, color, label in [("exact", "batch200_ms", BLUE, "Final exact, batch of 200"),
                                    ("fast", "batch200_ms", ORANGE, "Final fast (k=300), batch of 200"),
                                    ("exact", "single_query_ms", AQUA, "Final exact, one query"),
                                    ("fast", "single_query_ms", YELLOW, "Final fast, one query")]:
        real_sel = [r for r in rows if r["mode"] == mode and r["corpus"] == "real"]
        syn_sel = [r for r in rows if r["mode"] == mode and r["corpus"] == "synthetic"]
        a.plot([int(r["n_docs"]) for r in real_sel], [float(r[key]) for r in real_sel], "-o", color=color,
               linewidth=2, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        link = real_sel[-1:] + syn_sel                       # dotted line continues from the 20K real point
        a.plot([int(r["n_docs"]) for r in link], [float(r[key]) for r in link], ":", color=color, linewidth=2)
        a.plot([int(r["n_docs"]) for r in syn_sel], [float(r[key]) for r in syn_sel], "o", color=color,
               markersize=7, markerfacecolor=SURFACE, markeredgewidth=1.8)
    # one query reads the whole corpus once: it gets much slower when the corpus leaves the 16 MB L2 cache
    l2_docs = 16 * 2**20 / (768 * 4)
    a.axvline(l2_docs, color=AQUA, linewidth=1, linestyle="--")
    a.text(l2_docs * 1.06, 1.5e2, "right of this line the exact\ncorpus no longer fits the\n16 MB L2 cache", color=AQUA,
           fontsize=8.5, va="center")
    a.axvspan(25_000, 260_000, color=GRID, alpha=0.35, linewidth=0)
    a.text(30_000, 3e5, "synthetic corpus\n(10x question)", color=INK2, fontsize=9, va="top")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xticks([1e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5], ["1K", "5K", "10K", "20K", "50K", "100K", "200K"])
    a.set_xlabel("Documents", color=INK2)
    a.set_ylabel("Milliseconds (log scale)", color=INK2)
    a.set_title("(a) Search time vs corpus size", color=INK, fontsize=11, loc="left")
    style(a)
    a.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

    ex = [r for r in rows if r["mode"] == "exact"]
    labels = [f"{int(r['n_docs']) // 1000}K" for r in ex]
    mm = np.array([float(r["matmul_ms"]) for r in ex])
    tk = np.array([float(r["topk_ms"]) for r in ex])
    tot = mm + tk
    b.bar(labels, mm / tot * 100, color=BLUE, width=0.6, label="Matrix multiply Q @ X.T")
    b.bar(labels, tk / tot * 100, bottom=mm / tot * 100, color=ORANGE, width=0.6, label="Top-k (Numba, parallel)")
    for i, t in enumerate(tot):
        b.text(i, 101, f"{t:.1f} ms", ha="center", va="bottom", fontsize=9, color=INK)
    b.set_ylim(0, 112)
    b.set_ylabel("Share of search time (%)", color=INK2)
    b.set_xlabel("Documents (exact mode, 200 queries; 50K+ synthetic)", color=INK2)
    b.set_title("(b) Where the time goes", color=INK, fontsize=11, loc="left")
    style(b)
    b.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig6_scaling.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("saved figures/fig6_scaling.png")


def plot_overview():
    rows = load_csv(OUT_OVER)[::-1]
    names = [r["version"] for r in rows]
    secs = [float(r["seconds_200_queries"]) for r in rows]
    base = float(next(r for r in rows if r["phase"] == "1")["seconds_200_queries"])
    colors = {"1": INK2, "2": BLUE, "4": ORANGE}
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(SURFACE)
    bars = ax.barh(names, [s * 1000 for s in secs], color=[colors[r["phase"]] for r in rows], height=0.6)
    ax.set_xscale("log")
    for bar, s, r in zip(bars, secs, rows):
        label = f"{s:.1f} s" if s >= 1 else f"{s * 1000:.2f} ms"
        ax.text(bar.get_width() * 1.2, bar.get_y() + bar.get_height() / 2,
                f"{label}   {base / s:,.0f}x", va="center", fontsize=10, color=INK)
    ax.set_xlim(1, max(secs) * 1000 * 60)
    ax.set_xlabel("Milliseconds for 200 queries x 20,000 documents, top-10 (log scale)", color=INK2)
    ax.set_title(f"From {base:.0f} seconds to a few milliseconds", color=INK, fontsize=12, loc="left")
    style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()],
              labels=["Phase 1 baseline", "Phase 2 versions", "Final system (Phase 4)"],
              frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_overview.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("saved figures/fig7_overview.png")


def main():
    if "--plot-only" in sys.argv:
        plot_scaling()
        plot_overview()
        return
    X, Q = load_dense("mpnet")
    print("A. correctness")
    correctness(X, Q)
    print("B. scaling")
    scaling(X, Q)
    print("   chunked vs unchunked at 200K (synthetic)")
    chunked(X, Q)
    print("C. overview")
    overview(X, Q)
    plot_scaling()
    plot_overview()


if __name__ == "__main__":
    main()
