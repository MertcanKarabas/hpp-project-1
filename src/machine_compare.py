"""
Bonus: laptop vs Google Colab, and the roofline model of both machines.

Reads the JSON files written by src/machine_bench.py:
  results/machine_laptop.json   results/machine_colab.json
and writes
  results/machine_comparison.csv   one row per metric: laptop, colab, colab / laptop
  figures/fig8_machines.png        (a) the six versions  (b) final engine vs corpus size
                                   (c) thread scaling     (d) how much slower Colab is, per metric
  figures/fig9_roofline.png        roofline of each machine with the measured kernels

With only one JSON present it draws that machine alone (useful before the Colab run).

Usage:
  .venv/bin/python src/machine_compare.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import FIGURES, RESULTS  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MACHINES = [("laptop", BLUE), ("colab", ORANGE)]

SHORT = {
    "Naive Python loops (Phase 1)": "Naive Python (Phase 1)",
    "Numba @njit loops + top-k": "Numba @njit, 1 thread",
    "Numba parallel + fastmath, parallel top-k": "Numba prange + fastmath",
    "NumPy matmul + NumPy argpartition": "NumPy matmul + argpartition",
    "Final, exact (matmul + Numba parallel top-k)": "Final, exact",
    "Final, fast (SVD k=300)": "Final, fast (SVD k=300)",
}
KERNEL_LABEL = {
    "matmul_q1": "Q @ X.T, 1 query",
    "matmul_q20": "Q @ X.T, 20 queries",
    "matmul_q200": "Q @ X.T, 200 queries",
    "numba_njit_q200": "Numba @njit (1 thread)",
    "numba_prange_fastmath_q200": "Numba prange + fastmath",
    "topk_numba_parallel": "top-k, Numba (ops = comparisons)",
}


def load():
    out = {}
    for name, _ in MACHINES:
        p = RESULTS / f"machine_{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    if not out:
        sys.exit("no results/machine_*.json yet: run src/machine_bench.py first")
    return out


def label(m):
    e = m["env"]
    cores = f"{e['logical_cores']} logical cores" if e.get("threads_per_core", 1) == 1 else \
        f"{e['physical_cores']} core(s) x {e['threads_per_core']} threads"
    return f"{m['name'].capitalize()}: {e['cpu']}, {cores}, BLAS {e['blas']}"


def style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, which="major")
    if grid in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)


def fmt_ms(ms):
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.1f} ms" if ms >= 10 else f"{ms:.2f} ms"


# --- metrics table ------------------------------------------------------------------------------

def metrics(m):
    """(metric, unit, value, higher_is_better)"""
    r, k = m["roofs"], {x["name"]: x for x in m["kernels"]}
    ov = {x["version"]: x["seconds_200_queries"] * 1e3 for x in m["overview"]}
    eng = {(x["n_docs"], x["mode"]): x for x in m["engine"]}
    th, gil = m["threads"], m["threads"]["gil"]
    blas1 = next(x["gflops"] for x in th["blas"] if x["threads"] == "1")
    numba1 = th["numba"][0]["ms"]
    rows = [
        ("Peak float32 compute (SGEMM)", "GFLOP/s", r["peak_gflops"], True),
        ("Memory bandwidth (best of triad / read)", "GB/s", r["bandwidth_gbs"], True),
        ("Memory bandwidth, one thread (NumPy copy)", "GB/s", r["copy_1thread_gbs"], True),
        ("Ridge point (peak / bandwidth)", "FLOP/byte", r["ridge_flop_per_byte"], None),
        ("BLAS, 1 thread (Q @ X.T, 200 x 20K)", "GFLOP/s", blas1, True),
        ("Q @ X.T, 1 query (memory bound)", "GB/s", k["matmul_q1"]["gbs"], True),
    ]
    rows += [(f"{SHORT[v]} (200 queries)", "ms", ov[v], False) for v in SHORT]
    rows += [
        ("Final exact, one query, 20K docs", "ms", eng[(20000, "exact")]["single_query_ms"], False),
        ("Final fast, build (SVD, 20K docs)", "s", eng[(20000, "fast")]["build_s"], False),
        ("Numba speedup, all cores vs 1", "x", numba1 / th["numba"][-1]["ms"], True),
        ("Pure Python, thread pool speedup", "x", gil["serial_s"] / gil["threadpool_s"], True),
        ("Pure Python, process pool speedup", "x", gil["serial_s"] / gil["processpool_s"], True),
        ("Exact mode: identical top-10 lists vs Phase 1", "share", m["correctness"]["exact_identical_top10_lists"], None),
        ("Exact mode: max score difference vs Phase 1", "abs", m["correctness"]["exact_max_abs_score_diff"], None),
    ]
    return rows


def write_table(ms):
    names = list(ms)
    per = {n: metrics(ms[n]) for n in names}
    out = RESULTS / "machine_comparison.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "unit"] + names + (["colab_over_laptop"] if len(names) == 2 else []))
        for i, (metric, unit, _, _) in enumerate(per[names[0]]):
            vals = [per[n][i][2] for n in names]
            w.writerow([metric, unit] + [f"{v:.6g}" for v in vals] + ([f"{vals[1] / vals[0]:.4g}"] if len(vals) == 2 else []))
    print(f"saved {out.relative_to(RESULTS.parent)}")
    print("\n| metric | unit | " + " | ".join(names) + " |" + (" Colab / laptop |" if len(names) == 2 else ""))
    print("|---|---|" + "---:|" * (len(names) + (len(names) == 2)))
    for i, (metric, unit, _, _) in enumerate(per[names[0]]):
        vals = [per[n][i][2] for n in names]
        cells = [f"{v:.4g}" for v in vals] + ([f"{vals[1] / vals[0]:.3g}"] if len(vals) == 2 else [])
        print(f"| {metric} | {unit} | " + " | ".join(cells) + " |")


# --- figure 8 -----------------------------------------------------------------------------------

def plot_machines(ms):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor(SURFACE)
    (a, b), (c, d) = axes
    present = [(n, col) for n, col in MACHINES if n in ms]

    # (a) the six versions on each machine
    versions = list(SHORT)[::-1]
    y = np.arange(len(versions))
    h = 0.8 / len(present)
    for i, (n, col) in enumerate(present):
        ov = {x["version"]: x["seconds_200_queries"] * 1e3 for x in ms[n]["overview"]}
        vals = [ov[v] for v in versions]
        pos = y + (i - (len(present) - 1) / 2) * h
        a.barh(pos, vals, height=h * 0.9, color=col, label=n.capitalize())
        for p, v in zip(pos, vals):
            a.text(v * 1.15, p, fmt_ms(v), va="center", fontsize=8.5, color=INK)
    a.set_yticks(y, [SHORT[v] for v in versions])
    a.set_xscale("log")
    a.set_xlim(1, 3e6)
    a.set_xlabel("Milliseconds, 200 queries x 20,000 documents (log scale)", color=INK2)
    a.set_title("(a) The same six versions on both machines", color=INK, fontsize=11, loc="left")
    style(a, "x")
    a.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")

    # (b) final engine vs corpus size
    for n, col in present:
        eng = ms[n]["engine"]
        for mode, key, ls, lab in [("exact", "batch200_ms", "-", "exact, batch of 200"),
                                   ("fast", "batch200_ms", "--", "fast, batch of 200"),
                                   ("exact", "single_query_ms", ":", "exact, one query")]:
            sel = [x for x in eng if x["mode"] == mode]
            b.plot([x["n_docs"] for x in sel], [x[key] for x in sel], ls, marker="o", color=col, linewidth=2,
                   markersize=6, markeredgecolor=SURFACE, label=f"{n.capitalize()}, {lab}")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xticks([1e3, 5e3, 1e4, 2e4], ["1K", "5K", "10K", "20K"])
    b.set_xlabel("Documents", color=INK2)
    b.set_ylabel("Milliseconds (log scale)", color=INK2)
    b.set_title("(b) Final engine vs corpus size", color=INK, fontsize=11, loc="left")
    b.set_ylim(5e-3, 2e3)                                   # headroom so the legend sits above the lines
    style(b, "both")
    b.legend(frameon=False, fontsize=8.5, labelcolor=INK2, ncol=2, loc="upper left", handlelength=3.2)

    # (c) thread scaling: BLAS and Numba, speedup over 1 thread
    for n, col in present:
        th = ms[n]["threads"]
        bl = [x for x in th["blas"] if x["threads"] != "default"]
        w = [int(x["threads"]) for x in bl]
        c.plot(w, [bl[0]["ms"] / x["ms"] for x in bl], "-o", color=col, linewidth=2, markersize=6,
               markeredgecolor=SURFACE, label=f"{n.capitalize()}, BLAS matmul")
        nb = th["numba"]
        c.plot([x["threads"] for x in nb], [nb[0]["ms"] / x["ms"] for x in nb], "--s", color=col, linewidth=2,
               markersize=6, markeredgecolor=SURFACE, label=f"{n.capitalize()}, Numba prange kernel")
    top = max(max(x["threads"] for x in ms[n]["threads"]["numba"]) for n, _ in present)
    c.plot([1, top], [1, top], ":", color=INK2, linewidth=1, label="ideal")
    if "laptop" in ms:
        c.text(3.2, 1.55, "laptop BLAS: one thread already drives the matrix unit", color=BLUE, fontsize=8.5)
    if "colab" in ms:
        c.text(2.3, 0.45, "Colab: 2 vCPUs are 1 physical core, so no speedup", color=ORANGE, fontsize=8.5)
    c.set_ylim(0, top + 0.5)
    c.set_xlabel("Threads", color=INK2)
    c.set_ylabel("Speedup over 1 thread", color=INK2)
    c.set_title("(c) Thread scaling", color=INK, fontsize=11, loc="left")
    style(c, "both")
    c.legend(frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=3.2)

    # (d) Colab / laptop, per metric
    if len(present) == 2:
        L, Cm = metrics(ms["laptop"]), metrics(ms["colab"])
        keep = [i for i, r in enumerate(L) if r[3] is not None and r[1] in ("GFLOP/s", "GB/s", "ms", "s")]
        names = [L[i][0] for i in keep][::-1]
        # "how many times slower is Colab": time ratio for times, inverse ratio for rates
        slower = [(Cm[i][2] / L[i][2]) if not L[i][3] else (L[i][2] / Cm[i][2]) for i in keep][::-1]
        bars = d.barh(names, slower, color=[ORANGE if s >= 1 else AQUA for s in slower], height=0.65)
        for bar, s in zip(bars, slower):
            d.text(bar.get_width() * 1.08, bar.get_y() + bar.get_height() / 2, f"{s:.1f}x", va="center",
                   fontsize=9, color=INK)
        d.axvline(1, color=INK2, linewidth=1)
        d.set_xscale("log")
        d.set_xlim(min(0.5, min(slower) / 1.5), max(slower) * 3)
        d.set_xlabel("How many times slower Colab is (log scale; 1 = same)", color=INK2)
        d.tick_params(axis="y", labelsize=8.5)
    else:
        d.text(0.5, 0.5, "waiting for results/machine_colab.json", ha="center", va="center", color=INK2,
               transform=d.transAxes)
        d.set_xticks([])
        d.set_yticks([])
    d.set_title("(d) Colab vs laptop, metric by metric", color=INK, fontsize=11, loc="left")
    style(d, "x")

    fig.text(0.01, 0.005, "   |   ".join(label(ms[n]) for n, _ in present), fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig8_machines.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("saved figures/fig8_machines.png")


# --- figure 9: roofline -------------------------------------------------------------------------

def plot_roofline(ms):
    present = [(n, col) for n, col in MACHINES if n in ms]
    fig, axes = plt.subplots(1, len(present), figsize=(7.5 * len(present), 6.2), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    ai = np.logspace(-1, 3, 400)
    markers = {"matmul_q1": "o", "matmul_q20": "o", "matmul_q200": "o", "numba_njit_q200": "s",
               "numba_prange_fastmath_q200": "D", "topk_numba_parallel": "^"}
    colors = {"matmul_q1": BLUE, "matmul_q20": BLUE, "matmul_q200": BLUE, "numba_njit_q200": MAGENTA,
              "numba_prange_fastmath_q200": AQUA, "topk_numba_parallel": YELLOW}

    for ax, (n, col) in zip(axes[0], present):
        r = ms[n]["roofs"]
        for other, ocol in present:                          # the other machine's roof, faint, for scale
            if other != n:
                ro = ms[other]["roofs"]
                ax.plot(ai, np.minimum(ro["peak_gflops"], ro["bandwidth_gbs"] * ai), "--", color=ocol,
                        linewidth=1.2, alpha=0.6, label=f"{other.capitalize()} roof")
        ax.plot(ai, np.minimum(r["peak_gflops"], r["bandwidth_gbs"] * ai), "-", color=col, linewidth=2.5,
                label=f"{n.capitalize()} roof")
        ax.axvline(r["ridge_flop_per_byte"], color=col, linewidth=1, linestyle=":")
        ax.text(r["ridge_flop_per_byte"] * 1.1, r["peak_gflops"] * 0.012,
                f"ridge point\n{r['ridge_flop_per_byte']:.1f} FLOP/byte", color=col, fontsize=8.5)
        ax.text(950, r["peak_gflops"] * 1.15, f"compute: {r['peak_gflops']:,.0f} GFLOP/s", color=col,
                fontsize=9, ha="right")
        for k in ms[n]["kernels"]:
            if k["name"] not in KERNEL_LABEL:
                continue
            ax.plot(k["ai"], k["gflops"], markers[k["name"]], color=colors[k["name"]], markersize=9,
                    markeredgecolor=INK, markeredgewidth=0.8, linestyle="none")
            ax.annotate(f"{KERNEL_LABEL[k['name']]}\n{k['gflops']:,.1f} GFLOP/s", (k["ai"], k["gflops"]),
                        xytext=(8, -4), textcoords="offset points", fontsize=8, color=INK, va="top")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, 1000)
        ax.set_ylim(0.3, 6000)
        ax.set_xlabel("Arithmetic intensity (FLOP per byte of memory traffic)", color=INK2)
        ax.set_title(label(ms[n]), color=INK, fontsize=9.5, loc="left")
        style(ax, "both")
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
    axes[0][0].set_ylabel("GFLOP/s (log scale)", color=INK2)
    fig.suptitle("Roofline: batching queries moves Q @ X.T from memory bound to compute bound",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    # memory label written along the sloped roof: the angle needs the final axes geometry
    for ax, (n, col) in zip(axes[0], present):
        r = ms[n]["roofs"]
        p1, p2 = ax.transData.transform([(1, r["bandwidth_gbs"]), (10, r["bandwidth_gbs"] * 10)])
        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        x0 = r["ridge_flop_per_byte"] / 8
        ax.text(x0, r["bandwidth_gbs"] * x0 * 1.5, f"memory: {r['bandwidth_gbs']:.0f} GB/s", color=col, fontsize=9,
                rotation=angle, rotation_mode="anchor")
    fig.savefig(FIGURES / "fig9_roofline.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("saved figures/fig9_roofline.png")


if __name__ == "__main__":
    ms = load()
    write_table(ms)
    plot_machines(ms)
    plot_roofline(ms)
