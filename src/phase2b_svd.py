"""
Phase 2B: dimensionality reduction -- choose a method and justify it with data.

Matrix: mpnet corpus embeddings A (20,000 x 768, float32).

Methods
  ours_power    our power iteration, one singular vector at a time (src/svd_power.py)
  ours_block    our block power iteration, all k vectors at once (src/svd_power.py)
  numpy_full    np.linalg.svd: every one of the 768 components (ground truth)
  sklearn_tsvd  sklearn TruncatedSVD: randomized, only the top k
  sklearn_pca   sklearn PCA, solver 'covariance_eigh': centre the data, then
                eigen-decompose the 768 x 768 covariance matrix

Experiments
  1. time vs k                     N = 20K, k = 10 ... 700
  2. time vs matrix size           k = 100, N = 1K ... 20K
  3. reconstruction error          relative Frobenius error, plus the best possible (Eckart-Young)
  4. peak memory                   tracemalloc, k = 100
  5. our block power iteration     error vs number of iterations
  6. search quality                recall@10 of reduced-space search vs the Phase 1 reference
  7. PCA vs SVD                    how far the data mean is from the origin

Outputs: results/phase2b_*.csv, results/phase2b_summary.json, figures/fig3_svd.png

Usage:
  .venv/bin/python src/phase2b_svd.py
  .venv/bin/python src/phase2b_svd.py --plot-only
"""

import csv
import json
import sys
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD

import kernels as K
from benchmark_utils import benchmark, save_row
from data_utils import CORPUS_SIZES, FIGURES, RESULTS, load_dense
from svd_power import block_power_svd, power_iteration_svd, reconstruction_error

K_GRID = [10, 25, 50, 100, 200, 300, 400, 500, 600, 700]
K_GRID_POWER = [10, 25, 50, 100]               # one-vector-at-a-time: ~35 s at k=100, so timed up to 100;
                                               # its k=200 error comes from one (untimed) run
K_ERROR = [50, 100, 200]
K_RECALL = [25, 50, 100, 200, 300, 400, 600]
BLOCK_ITERS = 6
BLOCK_ITER_GRID = [0, 1, 2, 4, 6, 8, 12, 16]

OUT_K = RESULTS / "phase2b_time_vs_k.csv"
OUT_N = RESULTS / "phase2b_time_vs_n.csv"
OUT_ERR = RESULTS / "phase2b_error.csv"
OUT_ITER = RESULTS / "phase2b_block_iters.csv"
OUT_RECALL = RESULTS / "phase2b_recall.csv"
OUT_SUMMARY = RESULTS / "phase2b_summary.json"

NAMES = {
    "numpy_full": "NumPy full SVD",
    "sklearn_tsvd": "sklearn TruncatedSVD",
    "sklearn_pca": "sklearn PCA (covariance_eigh)",
    "ours_block": "Ours: block power iteration",
    "ours_power": "Ours: power iteration (1 vector at a time)",
}
COLORS = {"numpy_full": "#2a78d6", "sklearn_tsvd": "#eb6834", "sklearn_pca": "#1baf7a",
          "ours_block": "#eda100", "ours_power": "#e87ba4"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# --- the five methods, same interface: (A, k) -> U, S, Vt, mean ------------------------------

def ours_power(A, k):
    U, S, Vt, _ = power_iteration_svd(A, k)
    return U, S, Vt, None


def ours_block(A, k, n_iter=BLOCK_ITERS):
    U, S, Vt = block_power_svd(A, k, n_iter=n_iter)
    return U, S, Vt, None


def numpy_full(A, k):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], S[:k], Vt[:k], None


def sklearn_tsvd(A, k):
    t = TruncatedSVD(n_components=k, algorithm="randomized", random_state=0)
    Z = t.fit_transform(A)                      # Z = U * S
    S = t.singular_values_
    return Z / S, S, t.components_, None


def sklearn_pca(A, k):
    p = PCA(n_components=k, svd_solver="covariance_eigh")
    Z = p.fit_transform(A)                      # Z = (A - mean) V
    S = p.singular_values_
    return Z / S, S, p.components_, p.mean_


METHODS = {"numpy_full": numpy_full, "sklearn_tsvd": sklearn_tsvd, "sklearn_pca": sklearn_pca,
           "ours_block": ours_block, "ours_power": ours_power}


def fresh(path):
    if path.exists():
        path.unlink()


# --- 1 + 3. time vs k, reconstruction error ----------------------------------------------------

def time_vs_k(A, sigma_full):
    fresh(OUT_K)
    fresh(OUT_ERR)
    total = float(np.sum(sigma_full.astype(np.float64) ** 2))
    full_r = benchmark(numpy_full, A, 768, warmup=1, runs=5)   # cost does not depend on k
    for k in K_GRID:
        for key, func in METHODS.items():
            if key == "ours_power" and k not in K_GRID_POWER:
                continue
            r = full_r if key == "numpy_full" else benchmark(func, A, k, warmup=1, runs=5)
            U, S, Vt, mean = r["result"] if key != "numpy_full" else numpy_full(A, k)
            err = reconstruction_error(A, U, S, Vt, mean)
            optimal = float(np.sqrt(np.sum(sigma_full[k:].astype(np.float64) ** 2) / total))
            save_row(OUT_K, {"method": key, "k": k, "n_docs": A.shape[0], "median_s": r["median_s"],
                             "std_s": r["std_s"], "warmup": r["warmup"], "runs": r["runs"], "times": r["times"]})
            save_row(OUT_ERR, {"method": key, "k": k, "rel_error": err, "optimal_svd_error": optimal,
                               "gap_vs_optimal": err - optimal})
            print(f"  k={k:>3} {NAMES[key]:42s} {r['median_s']:8.3f} s   error {err:.5f} (optimal {optimal:.5f})",
                  flush=True)


# --- 2. time vs N --------------------------------------------------------------------------

def time_vs_n(A_all, k=100):
    fresh(OUT_N)
    for n in CORPUS_SIZES:
        A = A_all[:n]
        for key, func in METHODS.items():
            r = benchmark(func, A, k, warmup=1, runs=5)
            save_row(OUT_N, {"method": key, "k": k, "n_docs": n, "median_s": r["median_s"], "std_s": r["std_s"],
                             "warmup": r["warmup"], "runs": r["runs"], "times": r["times"]})
            print(f"  N={n:>6} {NAMES[key]:42s} {r['median_s']:8.3f} s", flush=True)


# --- 4. memory ------------------------------------------------------------------------------

def peak_memory(A, k=100):
    out = {}
    for key, func in METHODS.items():
        tracemalloc.start()
        func(A, k)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        out[key] = peak / 1e6
        print(f"  {NAMES[key]:42s} peak {peak / 1e6:8.1f} MB")
    return out


# --- 5. block power: accuracy vs iterations --------------------------------------------------

def block_iterations(A, sigma_full, k=100):
    fresh(OUT_ITER)
    total = float(np.sum(sigma_full.astype(np.float64) ** 2))
    optimal = float(np.sqrt(np.sum(sigma_full[k:].astype(np.float64) ** 2) / total))
    for it in BLOCK_ITER_GRID:
        r = benchmark(ours_block, A, k, it, warmup=1, runs=5)
        U, S, Vt, _ = r["result"]
        err = reconstruction_error(A, U, S, Vt)
        sv_err = float(np.max(np.abs(S - sigma_full[:k]) / sigma_full[:k]))
        save_row(OUT_ITER, {"n_iter": it, "k": k, "median_s": r["median_s"], "rel_error": err,
                            "optimal": optimal, "max_rel_sigma_error": sv_err, "times": r["times"]})
        print(f"  n_iter={it:>2}: {r['median_s']:.3f} s, error {err:.5f} (optimal {optimal:.5f}), "
              f"worst sigma error {sv_err:.1e}")


# --- 6. search quality after reduction --------------------------------------------------------

def normalise(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def recall_at_10(A, Qm, ref_idx):
    fresh(OUT_RECALL)
    full_time = benchmark(lambda Q, X: Q @ X.T, Qm, A, warmup=2, runs=5)["median_s"]
    for k in K_RECALL:
        for key in ("sklearn_tsvd", "sklearn_pca"):
            _, _, Vt, mean = METHODS[key](A, k)
            m = mean if mean is not None else 0.0
            Z = np.ascontiguousarray(normalise((A - m) @ Vt.T), dtype=np.float32)
            Qz = np.ascontiguousarray(normalise((Qm - m) @ Vt.T), dtype=np.float32)
            t = benchmark(lambda Q, X: Q @ X.T, Qz, Z, warmup=2, runs=5)["median_s"]
            idx, _ = K.topk_rows_parallel(Qz @ Z.T, 10)
            rec = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(idx, ref_idx)]))
            save_row(OUT_RECALL, {"method": key, "k": k, "recall_at_10": rec, "matmul_s": t,
                                  "full_dim_matmul_s": full_time, "reduced_corpus_mb": Z.nbytes / 1e6})
            print(f"  k={k:>3} {NAMES[key]:32s} recall@10 {rec:.3f}   matmul {t * 1000:.2f} ms "
                  f"(768 dims: {full_time * 1000:.2f} ms)")


# --- 7. PCA vs SVD: is the data centred? -----------------------------------------------------

def mean_analysis(A, Vt_full, sigma_full):
    m = A.mean(axis=0).astype(np.float64)
    v1 = Vt_full[0].astype(np.float64)
    s2 = sigma_full.astype(np.float64) ** 2
    return {
        "mean_norm": float(np.linalg.norm(m)),            # rows are unit length, so 0 = centred, 1 = all identical
        "cos_first_singular_vector_vs_mean": float(abs(v1 @ m) / np.linalg.norm(m)),
        "energy_share_first_component": float(s2[0] / s2.sum()),
        "energy_share_second_component": float(s2[1] / s2.sum()),
    }


# --- figure ------------------------------------------------------------------------------------

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def line(ax, xs, ys, key, **kw):
    ax.plot(xs, ys, "-o", color=COLORS[key], linewidth=2, markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=NAMES[key], **kw)


def plot():
    tk, tn, er, rc = load_csv(OUT_K), load_csv(OUT_N), load_csv(OUT_ERR), load_csv(OUT_RECALL)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(SURFACE)
    (a, b), (c, d) = axes

    for key in METHODS:
        rows = [r for r in tk if r["method"] == key]
        line(a, [int(r["k"]) for r in rows], [float(r["median_s"]) for r in rows], key)
    a.set_yscale("log")
    a.set_xlabel("Number of components k", color=INK2)
    a.set_ylabel("Seconds (log scale)", color=INK2)
    a.set_title("(a) Time vs k, 20,000 x 768", color=INK, fontsize=11, loc="left")
    style(a)
    handles, labels = a.get_legend_handles_labels()

    for key in METHODS:
        rows = [r for r in tn if r["method"] == key]
        line(b, [int(r["n_docs"]) for r in rows], [float(r["median_s"]) for r in rows], key)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xticks(CORPUS_SIZES, [f"{n // 1000}K" for n in CORPUS_SIZES])
    b.set_xlabel("Documents (rows of the matrix)", color=INK2)
    b.set_ylabel("Seconds (log scale)", color=INK2)
    b.set_title("(b) Time vs matrix size, k = 100", color=INK, fontsize=11, loc="left")
    style(b)

    # all methods sit within 1% of the optimum, so plot the distance to it
    c.axhline(0, color=INK2, linewidth=1.2, linestyle="--")
    c.text(K_GRID[-1], 0, "best possible rank-k error (Eckart-Young)  ", color=INK2, fontsize=9,
           ha="right", va="bottom")
    for key in METHODS:
        rows = [r for r in er if r["method"] == key]
        gaps = [(float(r["rel_error"]) - float(r["optimal_svd_error"])) / float(r["optimal_svd_error"]) * 100
                for r in rows]
        line(c, [int(r["k"]) for r in rows], gaps, key)
    c.set_xlabel("Number of components k", color=INK2)
    c.set_ylabel("Error above the optimum (%)", color=INK2)
    c.set_title("(c) Reconstruction error vs the best possible", color=INK, fontsize=11, loc="left")
    style(c)

    for key in ("sklearn_tsvd", "sklearn_pca"):
        rows = [r for r in rc if r["method"] == key]
        line(d, [int(r["k"]) for r in rows], [float(r["recall_at_10"]) for r in rows], key)
    d.axhline(1.0, color=INK2, linewidth=1, linestyle="--")
    d.text(K_RECALL[0], 1.0, " full 768-dim search = 1.0", color=INK2, fontsize=9, va="bottom")
    d.set_ylim(0, 1.08)
    d.set_xlabel("Number of components k", color=INK2)
    d.set_ylabel("Recall@10 vs exact 768-dim search", color=INK2)
    d.set_title("(d) Search quality after reduction", color=INK, fontsize=11, loc="left")
    style(d)

    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, labelcolor=INK2,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig3_svd.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("saved figures/fig3_svd.png")


def main():
    if "--plot-only" in sys.argv:
        return plot()
    A, Qm = load_dense("mpnet")
    ref_idx = np.load(RESULTS / "phase1_reference_top10_mpnet.npz")["indices"]
    t0 = time.perf_counter()
    _, sigma_full, Vt_full = np.linalg.svd(A, full_matrices=False)

    print("7. PCA vs SVD (data mean):")
    summary = {"mean_analysis": mean_analysis(A, Vt_full, sigma_full)}
    print("  ", summary["mean_analysis"])
    print("5. block power: error vs iterations (k=100):")
    block_iterations(A, sigma_full)
    print("1+3. time vs k and reconstruction error:")
    time_vs_k(A, sigma_full)
    print("2. time vs matrix size (k=100):")
    time_vs_n(A)
    print("4. peak memory (k=100):")
    summary["peak_memory_mb_k100"] = peak_memory(A)
    print("   our power iteration at k=200 (one untimed run: error + iterations per vector)")
    t1 = time.perf_counter()
    U, S, Vt, iters = power_iteration_svd(A, 200)
    single_run_s = time.perf_counter() - t1
    err200 = reconstruction_error(A, U, S, Vt)
    total = float(np.sum(sigma_full.astype(np.float64) ** 2))
    opt200 = float(np.sqrt(np.sum(sigma_full[200:].astype(np.float64) ** 2) / total))
    save_row(OUT_ERR, {"method": "ours_power", "k": 200, "rel_error": err200, "optimal_svd_error": opt200,
                       "gap_vs_optimal": err200 - opt200})
    summary["power_iteration_k200"] = {
        "single_run_s": single_run_s, "rel_error": err200, "optimal": opt200,
        "iters_mean": float(np.mean(iters)), "iters_max": int(max(iters)),
        "hit_max_iter": int(sum(i >= 300 for i in iters)),
        "max_rel_sigma_error": float(np.max(np.abs(S - sigma_full[:200]) / sigma_full[:200])),
        "iters_first_10": iters[:10], "iters_last_10": iters[-10:],
    }
    print("  ", {k: v for k, v in summary["power_iteration_k200"].items() if "iters_" not in k or k == "iters_mean"})
    print("6. search quality:")
    recall_at_10(A, Qm, ref_idx)
    summary["total_minutes"] = (time.perf_counter() - t0) / 60
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    plot()


if __name__ == "__main__":
    main()
