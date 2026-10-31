"""
Phase 1: where does the naive search spend its time?

Four views, each answering a different question:
  1. Manual breakdown (no profiler overhead): how the time splits between Python call
     overhead, the dot product, the norms and the sort. Honest shares for Amdahl's Law.
  2. cProfile: time per *function* and how many times each one is called.
  3. line_profiler: time per *line* inside cosine_similarity_python and search_naive.
  4. Data layout: the same pure-Python loop over NumPy rows instead of Python lists.

Outputs:
  results/phase1_breakdown.json, results/phase1_cprofile.txt, results/phase1_line_profiler.txt
  figures/fig1_profile.png, figures/fig1b_cprofile.png

Usage:
  .venv/bin/python src/phase1_profiling.py               # profile + figures (~2 min)
  .venv/bin/python src/phase1_profiling.py --plot-only   # redraw figures from saved results
"""

import cProfile
import io
import json
import linecache
import pstats
import statistics
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from line_profiler import LineProfiler

from benchmark_utils import benchmark
from data_utils import FIGURES, RESULTS, load_dense
from phase1_naive_search import cosine_similarity_python, search_naive

K = 10

# Chart palette (validated reference palette, light mode)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# --- 1. Manual breakdown -----------------------------------------------------------

def no_op(a, b):
    """Does no math: what is left is the cost of the loop, the call and the append."""
    return 0.0


def dot_only_python(a, b):
    """cosine_similarity_python without the two norm accumulators."""
    dot = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
    return dot


def timed_search(queries, corpus, sim):
    """search_naive with two stopwatches: scoring loop and sort, per query."""
    t_score = t_sort = 0.0
    for query in queries:
        t0 = time.perf_counter()
        scores = []
        for d_idx, doc in enumerate(corpus):
            scores.append((d_idx, sim(query, doc)))
        t1 = time.perf_counter()
        scores.sort(key=lambda x: x[1], reverse=True)
        scores[:K]
        t2 = time.perf_counter()
        t_score += t1 - t0
        t_sort += t2 - t1
    return t_score, t_sort


def median_breakdown(queries, corpus, sim, warmup=1, runs=5):
    for _ in range(warmup):
        timed_search(queries, corpus, sim)
    runs_ = [timed_search(queries, corpus, sim) for _ in range(runs)]
    return statistics.median(r[0] for r in runs_), statistics.median(r[1] for r in runs_)


def manual_breakdown(queries, corpus):
    nq = len(queries)
    score_full, sort_full = median_breakdown(queries, corpus, cosine_similarity_python)
    score_dot, _ = median_breakdown(queries, corpus, dot_only_python)
    score_noop, _ = median_breakdown(queries, corpus, no_op)

    ms = lambda s: s / nq * 1000  # seconds for all queries -> ms per query
    parts = {
        "Per-doc call + append": ms(score_noop),
        "Inner loop + dot product": ms(score_dot - score_noop),
        "Norms (a[i]*a[i], b[i]*b[i])": ms(score_full - score_dot),
        "Full sort + top-k": ms(sort_full),
    }
    total = sum(parts.values())
    return {
        "n_docs": len(corpus), "n_queries": nq, "dim": len(corpus[0]),
        "total_ms_per_query": total,
        "parts_ms_per_query": parts,
        "share": {name: v / total for name, v in parts.items()},
        # Amdahl: removing a part that takes fraction p of the time caps the speedup at 1/(1-p)
        "max_speedup_if_removed": {name: 1 / (1 - v / total) for name, v in parts.items()},
    }


# --- 2. cProfile ------------------------------------------------------------------

def run_cprofile(queries, corpus):
    pr = cProfile.Profile()
    pr.enable()
    search_naive(queries, corpus, k=K)
    pr.disable()

    stats = pstats.Stats(pr).strip_dirs()
    text = io.StringIO()
    stats.stream = text
    print(f"cProfile: search_naive, {len(queries)} queries x {len(corpus)} docs x {len(corpus[0])} dims\n",
          file=text)
    stats.sort_stats("tottime").print_stats(12)
    stats.sort_stats("cumtime").print_stats(8)
    (RESULTS / "phase1_cprofile.txt").write_text(text.getvalue())

    rows = []
    for (file, line, func), (cc, ncalls, tottime, cumtime, _) in stats.stats.items():
        rows.append({"function": func, "file": file, "ncalls": ncalls,
                     "tottime_s": tottime, "cumtime_s": cumtime})
    rows.sort(key=lambda r: -r["tottime_s"])
    return rows[:10], text.getvalue()


# --- 3. line_profiler ----------------------------------------------------------------

def run_line_profiler(queries, corpus):
    lp = LineProfiler(cosine_similarity_python, search_naive)
    lp.runcall(search_naive, queries, corpus, K)

    text = io.StringIO()
    lp.print_stats(stream=text, output_unit=1e-3, stripzeros=True)
    (RESULTS / "phase1_line_profiler.txt").write_text(
        f"line_profiler: {len(queries)} queries x {len(corpus)} docs\n\n" + text.getvalue())

    stats = lp.get_stats()
    out = {}
    for (filename, _, func), lines in stats.timings.items():
        total = sum(t for _, _, t in lines)
        out[func] = [{
            "line": lineno,
            "code": linecache.getline(filename, lineno).strip(),
            "hits": hits,
            "seconds": t * stats.unit,
            "share": t / total if total else 0.0,
        } for lineno, hits, t in lines]
    return out


# --- 4. Data layout --------------------------------------------------------------------

def layout_test(X, Q, n_docs=1000):
    """Same pure-Python loop; only the container changes (list vs NumPy row)."""
    docs_np, q_np = X[:n_docs], Q[0]
    docs_list, q_list = docs_np.tolist(), q_np.tolist()

    def loop(q, docs):
        return [cosine_similarity_python(q, d) for d in docs]

    t_list = benchmark(loop, q_list, docs_list, warmup=1, runs=5)["median_s"]
    t_np = benchmark(loop, q_np, docs_np, warmup=1, runs=5)["median_s"]
    return {"n_docs": n_docs, "python_list_s": t_list, "numpy_rows_s": t_np, "numpy_slower_by": t_np / t_list}


# --- Figures -------------------------------------------------------------------------

def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figure(bd, lines):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.patch.set_facecolor(SURFACE)

    # (a) manual breakdown
    names = list(bd["parts_ms_per_query"])[::-1]
    vals = [bd["parts_ms_per_query"][n] for n in names]
    colors = [ORANGE if n.startswith("Norms") else BLUE for n in names]
    bars = a.barh(names, vals, color=colors, height=0.55)
    for bar, n in zip(bars, names):
        a.text(bar.get_width() + max(vals) * 0.02, bar.get_y() + bar.get_height() / 2,
               f"{bd['parts_ms_per_query'][n]:.0f} ms  ({bd['share'][n]:.0%})",
               va="center", color=INK, fontsize=10)
    a.set_xlim(0, max(vals) * 1.45)
    a.set_xlabel("Milliseconds per query", color=INK2)
    a.set_title(f"(a) {bd['total_ms_per_query']:.0f} ms per query, {bd['n_docs']:,} docs x {bd['dim']} dims "
                f"(no profiler)", color=INK, fontsize=11, loc="left")
    style(a)

    # (b) line_profiler, cosine_similarity_python
    rows = [r for r in lines["cosine_similarity_python"] if r["share"] >= 0.005][::-1]
    labels = [r["code"] for r in rows]
    shares = [r["share"] * 100 for r in rows]
    colors = [ORANGE if "norm_a +=" in c or "norm_b +=" in c else BLUE for c in labels]
    bars = b.barh(range(len(rows)), shares, color=colors, height=0.55)
    b.set_yticks(range(len(rows)), labels, family="monospace", fontsize=9)
    for bar, s in zip(bars, shares):
        b.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2, f"{s:.0f}%",
               va="center", color=INK, fontsize=10)
    b.set_xlim(0, max(shares) * 1.25)
    b.set_xlabel("Share of time inside cosine_similarity_python (%)", color=INK2)
    b.set_title("(b) line_profiler: time per line", color=INK, fontsize=11, loc="left")
    style(b)

    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
               labels=["Work the search needs", "Redundant work (norms of unit vectors)"],
               loc="lower center", ncol=2, frameon=False, fontsize=10, labelcolor=INK2,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig1_profile.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_cprofile_screenshot(text):
    # keep only the header and the first table (sorted by tottime); tables end with a blank gap
    head, rest = text.split("Ordered by: internal time", 1)
    keep = (head + "Ordered by: internal time" + rest.split("\n\n\n")[0]).rstrip().splitlines()
    fig = plt.figure(figsize=(13, 0.19 * len(keep) + 0.4))
    fig.patch.set_facecolor("white")
    fig.text(0.01, 0.98, "\n".join(keep), family="monospace", fontsize=8.5, va="top", color=INK)
    fig.savefig(FIGURES / "fig1b_cprofile.png", dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_only():
    """Redraw the figures from saved results, without re-profiling."""
    saved = json.loads((RESULTS / "phase1_breakdown.json").read_text())
    make_figure(saved["manual_breakdown"], saved["line_profiler"])
    make_cprofile_screenshot((RESULTS / "phase1_cprofile.txt").read_text())
    print("redrawn figures/fig1_profile.png, figures/fig1b_cprofile.png")


def main():
    if "--plot-only" in sys.argv:
        return plot_only()
    RESULTS.mkdir(exist_ok=True)
    X, Q = load_dense("mpnet")
    corpus = X.tolist()

    print("1/4 manual breakdown (20K docs, 5 queries, 3 variants x 6 runs)...", flush=True)
    bd = manual_breakdown(Q[:5].tolist(), corpus)
    for name, v in bd["parts_ms_per_query"].items():
        print(f"    {name:32s} {v:7.1f} ms/query  {bd['share'][name]:5.1%}  "
              f"(max speedup if removed: {bd['max_speedup_if_removed'][name]:.2f}x)")

    print("2/4 cProfile (20K docs, 2 queries)...", flush=True)
    top, ctext = run_cprofile(Q[:2].tolist(), corpus)
    for r in top[:6]:
        print(f"    {r['function']:28s} ncalls={r['ncalls']:>9,}  tottime={r['tottime_s']:.2f}s")

    print("3/4 line_profiler (1K docs, 5 queries)...", flush=True)
    lines = run_line_profiler(Q[:5].tolist(), corpus[:1000])
    for r in lines["cosine_similarity_python"]:
        print(f"    {r['share']:5.1%}  {r['code']}")
    for r in lines["search_naive"]:
        if r["share"] >= 0.01:
            print(f"    [search_naive] {r['share']:5.1%}  {r['code']}")

    print("4/4 data layout (1 query x 1K docs)...", flush=True)
    lay = layout_test(X, Q)
    print(f"    list {lay['python_list_s'] * 1000:.1f} ms vs NumPy rows {lay['numpy_rows_s'] * 1000:.1f} ms "
          f"-> NumPy rows {lay['numpy_slower_by']:.1f}x slower in a Python loop")

    (RESULTS / "phase1_breakdown.json").write_text(json.dumps(
        {"manual_breakdown": bd, "cprofile_top": top, "line_profiler": lines, "layout_test": lay}, indent=2))
    make_figure(bd, lines)
    make_cprofile_screenshot(ctext)
    print("saved figures/fig1_profile.png, figures/fig1b_cprofile.png, results/phase1_*.{json,txt}")


if __name__ == "__main__":
    main()
