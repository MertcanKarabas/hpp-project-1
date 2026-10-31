"""
Phase 2A: sparse vs dense -- does the representation matter?

Representations of the same 20,000 abstracts:
  TF-IDF CSR      sparse, 25,933 columns, ~96 nonzeros per document
  TF-IDF dense    the same matrix with all the zeros stored (float32)
  mpnet           dense embedding, 768 dims
  MiniLM          dense embedding, 384 dims (memory only)

1) Memory footprint at 1K / 5K / 10K / 20K documents.
2) Score-matrix time with OUR OWN kernels (src/kernels.py, Numba, single thread):
     sparse . sparse   merge        TF-IDF query (CSR)  x TF-IDF docs (CSR)
     sparse . sparse   inverted     same, through an inverted index (CSC of the corpus)
     sparse . dense                 TF-IDF docs (CSR)   x TF-IDF query as a dense vector
     dense  . dense    TF-IDF       both densified (25,933 dims)  -- 20 queries only, it is slow
     dense  . dense    mpnet        768 dims
   Library versions (scipy sparse, NumPy/BLAS) are timed as a reference point.

All kernels are checked against the library result (scipy for TF-IDF, float64 NumPy for mpnet),
tolerance 1e-5.

Outputs: results/phase2a_memory.csv, results/phase2a_sparse_dense.csv, figures/fig2_sparse_dense.png

Usage:
  .venv/bin/python src/phase2a_sparse_dense.py
  .venv/bin/python src/phase2a_sparse_dense.py --plot-only
"""

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

import kernels as K
from benchmark_utils import benchmark, save_row
from data_utils import CORPUS_SIZES, EMB, FIGURES, RESULTS, load_dense

N_QUERIES = 200
N_QUERIES_DENSE_TFIDF = 20      # 25,933-dim dense dot products: 34x more work per pair than mpnet
OUT_MEM = RESULTS / "phase2a_memory.csv"
OUT_TIME = RESULTS / "phase2a_sparse_dense.csv"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def csr_bytes(M):
    return M.data.nbytes + M.indices.nbytes + M.indptr.nbytes


def load_tfidf():
    X = sp.load_npz(EMB / "tfidf_corpus.npz").tocsr()
    Q = sp.load_npz(EMB / "tfidf_queries.npz").tocsr()
    X.sort_indices()
    Q.sort_indices()
    return X, Q


def memory_table(X_tfidf, X_mpnet, X_minilm):
    if OUT_MEM.exists():
        OUT_MEM.unlink()
    V = X_tfidf.shape[1]
    for n in CORPUS_SIZES:
        Xs = X_tfidf[:n]
        save_row(OUT_MEM, {
            "n_docs": n,
            "tfidf_csr_mb": csr_bytes(Xs) / 1e6,
            "tfidf_dense_mb": n * V * 4 / 1e6,
            "mpnet_mb": X_mpnet[:n].nbytes / 1e6,
            "minilm_mb": X_minilm[:n].nbytes / 1e6,
            "tfidf_nnz": Xs.nnz,
            "tfidf_density": Xs.nnz / (n * V),
        })
        print(f"N={n:>6}: CSR {csr_bytes(Xs) / 1e6:6.1f} MB | dense TF-IDF {n * V * 4 / 1e6:7.1f} MB | "
              f"mpnet {X_mpnet[:n].nbytes / 1e6:5.1f} MB | MiniLM {X_minilm[:n].nbytes / 1e6:5.1f} MB")


def record(name, kind, n, nq, r, ref, extra=None):
    err = float(np.max(np.abs(np.asarray(r["result"], dtype=np.float64) - ref)))
    row = {
        "method": name, "kind": kind, "n_docs": n, "n_queries": nq,
        "warmup": r["warmup"], "runs": r["runs"],
        "median_s": r["median_s"], "std_s": r["std_s"], "min_s": r["min_s"],
        "ms_per_query": r["median_s"] / nq * 1000, "max_abs_err": err,
        "index_build_s": "", "postings_per_query": "",   # filled only for the inverted index
    }
    row.update(extra or {})
    row["times"] = r["times"]
    save_row(OUT_TIME, row)
    print(f"  N={n:>6} {name:34s} {row['ms_per_query']:9.3f} ms/query   err {err:.1e}", flush=True)
    if err > 1e-5:
        raise AssertionError(f"{name} differs from the reference by {err}")


def timing(X_tfidf, Q_tfidf, X_mpnet, Q_mpnet):
    if OUT_TIME.exists():
        OUT_TIME.unlink()
    Qs = Q_tfidf[:N_QUERIES]
    qp, qi, qv = K.csr_arrays(Qs)
    Qd = Qs.toarray()                                   # dense queries for sparse . dense
    Qd_small = Qd[:N_QUERIES_DENSE_TFIDF]
    Qm = Q_mpnet[:N_QUERIES]
    X_dense_all = X_tfidf.toarray()                     # 2.07 GB at 20K, allocated once

    for n in CORPUS_SIZES:
        Xs = X_tfidf[:n]
        dp, di, dv = K.csr_arrays(Xs)
        ref_tfidf = (Qs @ Xs.T).toarray().astype(np.float64)

        # sparse . sparse (merge)
        r = benchmark(K.csr_csr_merge, qp, qi, qv, dp, di, dv, warmup=2, runs=5)
        record("sparse.sparse merge (ours)", "ours", n, N_QUERIES, r, ref_tfidf)

        # sparse . sparse (inverted index); building the index is a one-off cost, timed separately
        T = Xs.tocsc()
        T.sort_indices()
        build = benchmark(lambda M: M.tocsc(), Xs, warmup=1, runs=5)["median_s"]
        postings = int(sum(T.indptr[w + 1] - T.indptr[w] for w in qi)) / N_QUERIES
        r = benchmark(K.csr_csr_inverted, qp, qi, qv, T.indptr, T.indices, T.data, n, warmup=2, runs=5)
        record("sparse.sparse inverted (ours)", "ours", n, N_QUERIES, r, ref_tfidf,
               {"index_build_s": build, "postings_per_query": postings})

        # sparse . dense
        r = benchmark(K.csr_dense, dp, di, dv, Qd, warmup=2, runs=5)
        record("sparse.dense (ours)", "ours", n, N_QUERIES, r, ref_tfidf)

        # dense . dense, TF-IDF densified (fewer queries)
        Xd = X_dense_all[:n]
        ref_small = ref_tfidf[:N_QUERIES_DENSE_TFIDF]
        r = benchmark(K.dense_numba, Qd_small, Xd, warmup=2, runs=5)
        record("dense.dense TF-IDF (ours)", "ours", n, N_QUERIES_DENSE_TFIDF, r, ref_small)

        # dense . dense, mpnet
        Xm = X_mpnet[:n]
        ref_m = Qm.astype(np.float64) @ Xm.astype(np.float64).T
        r = benchmark(K.dense_numba, Qm, Xm, warmup=2, runs=5)
        record("dense.dense mpnet (ours)", "ours", n, N_QUERIES, r, ref_m)

        # library reference points
        r = benchmark(lambda A, B: (A @ B.T).toarray(), Qs, Xs, warmup=2, runs=5)
        record("sparse.sparse scipy", "library", n, N_QUERIES, r, ref_tfidf)
        r = benchmark(lambda A, B: (A @ B.T).T, Xs, Qd, warmup=2, runs=5)
        record("sparse.dense scipy", "library", n, N_QUERIES, r, ref_tfidf)
        r = benchmark(lambda A, B: A @ B.T, Qd_small, Xd, warmup=2, runs=5)
        record("dense.dense TF-IDF NumPy", "library", n, N_QUERIES_DENSE_TFIDF, r, ref_small)
        r = benchmark(lambda A, B: A @ B.T, Qm, Xm, warmup=2, runs=5)
        record("dense.dense mpnet NumPy", "library", n, N_QUERIES, r, ref_m)


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
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)


def plot():
    mem = load_csv(OUT_MEM)
    rows = load_csv(OUT_TIME)
    ns = [int(r["n_docs"]) for r in mem]
    xlabels = [f"{n // 1000}K" for n in ns]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    fig.patch.set_facecolor(SURFACE)

    # (a) memory
    for key, label, color in [("tfidf_dense_mb", "TF-IDF dense (25,933 dims)", BLUE),
                              ("mpnet_mb", "mpnet dense (768 dims)", ORANGE),
                              ("minilm_mb", "MiniLM dense (384 dims)", AQUA),
                              ("tfidf_csr_mb", "TF-IDF sparse CSR", YELLOW)]:
        ys = [float(r[key]) for r in mem]
        a.plot(xlabels, ys, "-o", color=color, linewidth=2, markersize=8,
               markeredgecolor=SURFACE, markeredgewidth=2)
        a.annotate(f"{label}: {ys[-1]:,.0f} MB", (len(ys) - 1, ys[-1]), xytext=(8, 0),
                   textcoords="offset points", va="center", fontsize=9, color=INK)
    a.set_yscale("log")
    a.set_xlim(-0.2, len(ns) - 1 + 2.1)
    a.set_xlabel("Documents", color=INK2)
    a.set_ylabel("Memory (MB, log scale)", color=INK2)
    a.set_title("(a) Memory footprint", color=INK, fontsize=11, loc="left")
    style(a)

    # (b) time per query, our kernels
    for name, color in [("dense.dense TF-IDF (ours)", BLUE), ("dense.dense mpnet (ours)", ORANGE),
                        ("sparse.sparse merge (ours)", MAGENTA), ("sparse.dense (ours)", AQUA),
                        ("sparse.sparse inverted (ours)", YELLOW)]:
        ys = [float(next(r for r in rows if r["method"] == name and int(r["n_docs"]) == n)["ms_per_query"])
              for n in ns]
        b.plot(xlabels, ys, "-o", color=color, linewidth=2, markersize=8, label=name.replace(" (ours)", ""),
               markeredgecolor=SURFACE, markeredgewidth=2)
    b.set_yscale("log")
    b.set_xlabel("Documents", color=INK2)
    b.set_ylabel("Milliseconds per query (log scale)", color=INK2)
    b.set_title("(b) Our kernels (Numba, 1 thread): time per query", color=INK, fontsize=11, loc="left")
    style(b)
    b.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)

    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig2_sparse_dense.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("saved figures/fig2_sparse_dense.png")


def main():
    if "--plot-only" in sys.argv:
        return plot()
    X_tfidf, Q_tfidf = load_tfidf()
    X_mpnet, Q_mpnet = load_dense("mpnet")
    X_minilm, _ = load_dense("minilm")
    print("memory:")
    memory_table(X_tfidf, X_mpnet, X_minilm)
    print("timing:")
    timing(X_tfidf, Q_tfidf, X_mpnet, Q_mpnet)
    plot()


if __name__ == "__main__":
    main()
