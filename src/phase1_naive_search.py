"""
Phase 1: naive baseline search -- intentionally slow, pure Python.

Same structure as the course starter code (starter_code/phase1_baseline/naive_search.py):
  cosine_similarity_python(a, b)   element-wise loop, no NumPy inside
  search_naive(queries, corpus, k) score every document, full sort, keep top-k

Vectors are converted to plain Python lists first (X.tolist()), exactly like the
starter code. Every later phase is compared against this function.

Usage:
  .venv/bin/python src/phase1_naive_search.py bench       # timing at 1K/5K/10K/20K
  .venv/bin/python src/phase1_naive_search.py reference   # top-k for all 200 queries (~3 min, run once)
"""

import argparse
import json
import math
import time
import tracemalloc

import numpy as np

from benchmark_utils import benchmark, environment_info, save_row
from data_utils import CORPUS_SIZES, RESULTS, load_dense


def cosine_similarity_python(a, b):
    """Cosine similarity of two Python lists, one element at a time."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def search_naive(queries, corpus, k=10):
    """For each query: score all documents, sort all scores, return top-k (doc_index, score)."""
    results = []
    for query in queries:
        scores = []
        for d_idx, doc in enumerate(corpus):
            sim = cosine_similarity_python(query, doc)
            scores.append((d_idx, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        results.append(scores[:k])
    return results


def check_against_numpy(results, Q, X, k):
    """Sanity check: naive scores must equal a float64 NumPy computation.
    Scores are compared (not indices) so ties cannot cause a false alarm."""
    Q64, X64 = Q.astype(np.float64), X.astype(np.float64)
    S = (Q64 @ X64.T) / np.outer(np.linalg.norm(Q64, axis=1), np.linalg.norm(X64, axis=1))
    ref = -np.sort(-S, axis=1)[:, :k]
    got = np.array([[s for _, s in r] for r in results])
    return float(np.max(np.abs(got - ref)))


def measure_list_conversion(X):
    """Cost of the Python-list layout: conversion time and memory vs the NumPy array."""
    t0 = time.perf_counter()
    X.tolist()
    seconds = time.perf_counter() - t0
    tracemalloc.start()
    as_list = X.tolist()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del as_list
    return seconds, peak


def run_bench(args):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "environment.json").write_text(json.dumps(environment_info(), indent=2))
    out = RESULTS / "phase1_baseline.csv"
    X_all, Q_all = load_dense(args.model)
    Q = Q_all[: args.queries]
    queries = Q.tolist()

    for n in args.sizes:
        X = X_all[:n]
        corpus = X.tolist()  # conversion is setup, not part of the timed search
        r = benchmark(search_naive, queries, corpus, k=args.k, warmup=args.warmup, runs=args.runs)
        err = check_against_numpy(r["result"], Q, X, args.k)
        per_query_ms = r["median_s"] / len(queries) * 1000
        save_row(out, {
            "method": "naive_python", "model": args.model, "n_docs": n, "dim": X.shape[1],
            "n_queries": len(queries), "k": args.k, "warmup": r["warmup"], "runs": r["runs"],
            "median_s": round(r["median_s"], 6), "mean_s": round(r["mean_s"], 6),
            "std_s": round(r["std_s"], 6), "min_s": round(r["min_s"], 6), "max_s": round(r["max_s"], 6),
            "per_query_ms": round(per_query_ms, 3),
            "us_per_doc": round(per_query_ms * 1000 / n, 3),
            "est_all_200_queries_s": round(per_query_ms * 200 / 1000, 1),
            "max_abs_err_vs_numpy": err, "times": r["times"],
        })
        print(f"N={n:>6}: median {r['median_s']:.3f}s for {len(queries)} queries "
              f"-> {per_query_ms:.1f} ms/query, {per_query_ms * 1000 / n:.2f} us/doc, "
              f"std {r['std_s']:.3f}s, err vs numpy {err:.1e}", flush=True)

    secs, peak = measure_list_conversion(X_all)
    layout = {
        "n_docs": X_all.shape[0], "dim": X_all.shape[1],
        "numpy_float32_mb": round(X_all.nbytes / 1e6, 1),
        "python_list_mb": round(peak / 1e6, 1),
        "ratio": round(peak / X_all.nbytes, 1),
        "tolist_seconds": round(secs, 3),
    }
    (RESULTS / "phase1_memory_layout.json").write_text(json.dumps(layout, indent=2))
    print(f"memory: NumPy {layout['numpy_float32_mb']} MB vs list-of-lists {layout['python_list_mb']} MB "
          f"({layout['ratio']}x), tolist {secs:.2f}s")


def run_reference(args):
    """Naive top-k for every query on the full corpus: the ground truth for later phases."""
    X, Q = load_dense(args.model)
    t0 = time.perf_counter()
    res = search_naive(Q.tolist(), X.tolist(), k=args.k)
    elapsed = time.perf_counter() - t0
    idx = np.array([[i for i, _ in r] for r in res], dtype=np.int64)
    scores = np.array([[s for _, s in r] for r in res], dtype=np.float64)
    np.savez(RESULTS / f"phase1_reference_top{args.k}_{args.model}.npz", indices=idx, scores=scores)
    print(f"reference: {len(res)} queries x {X.shape[0]} docs in {elapsed:.0f}s (single run), "
          f"err vs numpy {check_against_numpy(res, Q, X, args.k):.1e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "reference"])
    ap.add_argument("--model", default="mpnet")
    ap.add_argument("--sizes", type=int, nargs="*", default=CORPUS_SIZES)
    ap.add_argument("--queries", type=int, default=20, help="queries per timed run (pure Python is slow)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    run_bench(args) if args.mode == "bench" else run_reference(args)
