"""
Three supporting measurements that the write-ups quote, made reproducible.

  overlap   top-10 agreement between representations on the 200 queries:
            mpnet vs TF-IDF and mpnet vs MiniLM                      -> results/extra_model_overlap.json
  encode    time to turn one query sentence into a vector (the demo's first step),
            on the CPU and on the Apple GPU (mps):
              repeated: the same sentence, warmup 2 + 5 runs (median)
              new:      20 different sentences, each encoded once (Apple's GPU backend
                        compiles new kernels for every new input length)   -> results/extra_encode_latency.json
  blas      the matrix multiply Q @ X.T alone (200 x 20K x 768) with Accelerate limited to
            1, 2, 4, 10 threads or left at its default; one fresh process per setting,
            because the limit is read when NumPy loads                   -> results/extra_blas_threads.csv

Usage:
  .venv/bin/python src/extra_measurements.py             # all three
  .venv/bin/python src/extra_measurements.py overlap     # or encode / blas
"""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

MODEL = "sentence-transformers/all-mpnet-base-v2"
BLAS_THREADS = ["1", "2", "4", "10", "default"]


def top10(S):
    import numpy as np
    part = np.argpartition(-S, 10, axis=1)[:, :10]
    return part


def overlap():
    import numpy as np
    import scipy.sparse as sp
    from data_utils import EMB, RESULTS, load_dense

    Xm, Qm = load_dense("mpnet")
    Xs, Qs = load_dense("minilm")
    Xt, Qt = sp.load_npz(EMB / "tfidf_corpus.npz"), sp.load_npz(EMB / "tfidf_queries.npz")
    tm = top10(Qm @ Xm.T)
    ts = top10(Qs @ Xs.T)
    tt = top10((Qt @ Xt.T).toarray())

    def agree(a, b):
        return float(np.mean([len(set(x) & set(y)) / 10 for x, y in zip(a, b)]))

    out = {"n_queries": int(Qm.shape[0]), "k": 10,
           "mpnet_vs_tfidf": agree(tm, tt), "mpnet_vs_minilm": agree(tm, ts)}
    (RESULTS / "extra_model_overlap.json").write_text(json.dumps(out, indent=2))
    print(out)


def encode():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import warnings
    warnings.filterwarnings("ignore")
    from sentence_transformers import SentenceTransformer
    from benchmark_utils import benchmark
    from data_utils import RESULTS, load_query_texts

    texts = load_query_texts()[:20]                 # 20 real titles of different lengths
    out = {"model": MODEL, "sentences_new": len(texts)}
    for device in ("cpu", "mps"):
        model = SentenceTransformer(MODEL, device=device)
        warm = "a short warm-up sentence"
        r = benchmark(model.encode, [warm], warmup=2, runs=5)
        new = []
        for t in texts:                             # each sentence encoded exactly once
            t0 = time.perf_counter()
            model.encode([t])
            new.append(time.perf_counter() - t0)
        out[device] = {"repeated_median_ms": r["median_s"] * 1000,
                       "new_sentences_median_ms": statistics.median(new) * 1000,
                       "new_sentences_min_ms": min(new) * 1000, "new_sentences_max_ms": max(new) * 1000}
        print(device, {k: round(v, 1) for k, v in out[device].items()})
    (RESULTS / "extra_encode_latency.json").write_text(json.dumps(out, indent=2))


def blas_one():
    """Runs in a child process whose VECLIB_MAXIMUM_THREADS was set before NumPy loaded."""
    from benchmark_utils import benchmark
    from data_utils import load_dense
    X, Q = load_dense("mpnet")
    r = benchmark(lambda: Q @ X.T, warmup=3, runs=7)
    print(json.dumps({"median_s": r["median_s"], "std_s": r["std_s"], "times": r["times"]}))


def blas():
    from benchmark_utils import save_row
    from data_utils import RESULTS
    out = RESULTS / "extra_blas_threads.csv"
    if out.exists():
        out.unlink()
    base = None
    for t in BLAS_THREADS:
        env = os.environ.copy()
        env.pop("VECLIB_MAXIMUM_THREADS", None)
        if t != "default":
            env["VECLIB_MAXIMUM_THREADS"] = t
        p = subprocess.run([sys.executable, __file__, "blas-one"], env=env, capture_output=True, text=True)
        r = json.loads(p.stdout.strip().splitlines()[-1])
        base = base or r["median_s"]
        gflops = 2 * 200 * 20_000 * 768 / r["median_s"] / 1e9
        save_row(out, {"veclib_threads": t, "median_ms": r["median_s"] * 1000, "std_ms": r["std_s"] * 1000,
                       "gflops": gflops, "speedup_vs_1_thread": base / r["median_s"], "warmup": 3, "runs": 7,
                       "times": r["times"]})
        print(f"  VECLIB_MAXIMUM_THREADS={t:>7}: {r['median_s'] * 1000:.2f} ms  {gflops:,.0f} GFLOP/s  "
              f"{base / r['median_s']:.2f}x")
        time.sleep(2)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "blas-one":
        blas_one()
    else:
        for name, f in (("overlap", overlap), ("encode", encode), ("blas", blas)):
            if what in (name, "all"):
                print(f"--- {name}")
                f()
