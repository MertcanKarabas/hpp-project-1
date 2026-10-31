"""
Phase 3: parallelism and concurrency.

Every strategy runs the SAME unit of work -- "top-10 of every query inside documents a..b" --
and only the way the units are spread over workers changes. The corpus is split into
4 x workers chunks (more chunks than workers, so fast performance cores can take extra
chunks while slower efficiency cores finish theirs). Partial top-10 lists are merged at the end.

Three workloads
  blas    S = Q @ X_chunk.T (Apple Accelerate)  + top-k   -- what the final system does
  cpu     S = our Numba fastmath kernel           + top-k   -- the same search on CPU cores only
  python  naive pure-Python search (Phase 1), 20 queries x 5,000 docs, split by query

Strategies (workers = 1, 2, 4, 6, 8, 10)
  blas_threads        one big matmul, BLAS allowed W threads (VECLIB_MAXIMUM_THREADS)   [blas]
  numba_prange        Numba parallel kernels, numba.set_num_threads(W)                  [cpu]
  threadpool          concurrent.futures.ThreadPoolExecutor                              [all]
  processpool         ProcessPoolExecutor, corpus memory-mapped once per worker         [all]
  processpool_pickle  ProcessPoolExecutor, every task ships its chunk (pickled)         [blas]
  dask                dask.delayed + threaded scheduler                                  [blas, cpu]
  ray                 Ray tasks, corpus placed once in Ray's shared object store        [blas, cpu]
  asyncio_executor    asyncio + run_in_executor (a thread pool under the hood)          [blas]
  asyncio             plain coroutines, no threads (concurrency without parallelism)    [blas]
  phase2_best         best Phase 2 version: default BLAS + Numba parallel top-k        [blas]

Each configuration runs in a fresh subprocess, so no threads or worker processes leak from
one measurement into the next, and the BLAS thread limit (an environment variable read at
start-up) can be set per run. All pools use 1 BLAS thread to avoid oversubscription.

Usage:
  .venv/bin/python src/phase3_parallel.py --all          # every configuration (~15 min)
  .venv/bin/python src/phase3_parallel.py --one blas threadpool 4
  .venv/bin/python src/phase3_parallel.py --plot-only
"""

import os
import sys

# The BLAS thread limit is read when NumPy loads, so it has to be set before the import below.
# --all passes it explicitly to every subprocess; this default covers a manual --one run.
# phase2_best deliberately keeps Accelerate's default threading.
if "--one" in sys.argv and "phase2_best" not in sys.argv:
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import kernels as K  # noqa: E402
from benchmark_utils import benchmark, save_row  # noqa: E402
from data_utils import EMB, FIGURES, RESULTS  # noqa: E402

K_TOP = 10
WORKERS = [1, 2, 4, 6, 8, 10]
PY_DOCS, PY_QUERIES = 5_000, 20
OUT = RESULTS / "phase3_parallel.csv"
COOLDOWN_S = 2.0

# Module-level state. Threads share it; every worker process builds its own copy (init_worker).
G = {}


# --- the unit of work ---------------------------------------------------------------------------

def load_globals(workload):
    G["workload"] = workload
    X = np.load(EMB / "mpnet_corpus.npy", mmap_mode="r")      # memory-mapped: the OS page cache is
    G["X"] = np.asarray(X)                                      # shared by every process on the machine
    G["Q"] = np.load(EMB / "mpnet_queries.npy")
    if workload == "python":
        G["X"] = np.ascontiguousarray(G["X"][:PY_DOCS])
        G["Q"] = G["Q"][:PY_QUERIES]
        G["Xl"], G["Ql"] = G["X"].tolist(), G["Q"].tolist()


def init_worker(workload):
    """ProcessPool initializer: runs once in every worker process, not once per task.
    Workers inherit VECLIB_MAXIMUM_THREADS=1 from the parent environment (set before NumPy loads)."""
    load_globals(workload)


def chunk_search(a, b):
    """Top-k of every query among documents a..b-1. Returns global document ids."""
    Q, Xc = G["Q"], G["X"][a:b]
    S = Q @ Xc.T if G["workload"] == "blas" else K.dense_fastmath_nogil(Q, Xc)
    idx, val = K.topk_rows_nogil(S, K_TOP)
    return idx + a, val


def chunk_python(qa, qb):
    """Phase 1 naive search for queries qa..qb-1 over the whole (small) corpus."""
    from phase1_naive_search import search_naive
    res = search_naive(G["Ql"][qa:qb], G["Xl"], k=K_TOP)
    return (np.array([[i for i, _ in r] for r in res]), np.array([[s for _, s in r] for r in res]))


def run_task(bounds):
    """Top-level dispatcher so ProcessPoolExecutor can pickle it by name."""
    return (chunk_python if G["workload"] == "python" else chunk_search)(*bounds)


def chunk_with_data(args):
    """processpool_pickle: the chunk itself travels to the worker with every task."""
    Q, Xc, a = args
    idx, val = K.topk_rows_nogil(Q @ Xc.T, K_TOP)
    return idx + a, val


def ray_chunk(Q, X, a, b, workload):
    """Ray task: Q and X arrive as zero-copy read-only views of Ray's shared object store."""
    S = Q @ X[a:b].T if workload == "blas" else K.dense_fastmath_nogil(Q, X[a:b])
    idx, val = K.topk_rows_nogil(S, K_TOP)
    return idx + a, val


def task_bounds(workload, workers):
    if workload == "python":
        return [(i, i + 1) for i in range(PY_QUERIES)]          # one query per task
    edges = np.linspace(0, G["X"].shape[0], 4 * workers + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]


def merge(workload, parts):
    if workload == "python":
        return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])
    return K.merge_topk(parts, K_TOP)


# --- strategies ------------------------------------------------------------------------------------

async def _coroutines(T, fn):
    async def one(a, b):
        return fn(a, b)                                        # no await inside: runs start to finish
    return await asyncio.gather(*[one(a, b) for a, b in T])


async def _executor(T, fn, ex):
    loop = asyncio.get_running_loop()
    return await asyncio.gather(*[loop.run_in_executor(ex, fn, a, b) for a, b in T])


def make_search(strategy, workload, W):
    """Returns (search(), cleanup()). Pool / cluster start-up happens here, outside the timing."""
    fn = chunk_python if workload == "python" else chunk_search
    T = task_bounds(workload, W)
    done = lambda: None

    if strategy == "blas_threads":
        return (lambda: K.topk_rows_nogil(G["Q"] @ G["X"].T, K_TOP)), done

    if strategy == "phase2_best":
        return (lambda: K.topk_rows_parallel(G["Q"] @ G["X"].T, K_TOP)), done

    if strategy == "numba_prange":
        import numba
        numba.set_num_threads(W)
        return (lambda: K.topk_rows_parallel(K.dense_numba_parallel_fastmath(G["Q"], G["X"]), K_TOP)), done

    if strategy == "threadpool":
        ex = ThreadPoolExecutor(W)
        return (lambda: merge(workload, list(ex.map(lambda t: fn(*t), T)))), ex.shutdown

    if strategy == "processpool":
        ex = ProcessPoolExecutor(W, initializer=init_worker, initargs=(workload,))
        list(ex.map(run_task, T))                              # start every worker, load data, compile
        return (lambda: merge(workload, list(ex.map(run_task, T)))), ex.shutdown

    if strategy == "processpool_pickle":
        ex = ProcessPoolExecutor(W)
        Q, X = G["Q"], G["X"]
        payload = lambda: [(Q, X[a:b], a) for a, b in T]
        list(ex.map(chunk_with_data, payload()))
        return (lambda: merge(workload, list(ex.map(chunk_with_data, payload())))), ex.shutdown

    if strategy == "dask":
        import dask
        from dask import delayed
        return (lambda: merge(workload, dask.compute(*[delayed(fn)(a, b) for a, b in T],
                                                     scheduler="threads", num_workers=W))), done

    if strategy == "ray":
        import ray
        ray.init(num_cpus=W, include_dashboard=False, log_to_driver=False, logging_level="ERROR",
                 runtime_env={"env_vars": {"VECLIB_MAXIMUM_THREADS": "1", "PYTHONPATH": str(SRC)}})
        Q_ref, X_ref = ray.put(G["Q"]), ray.put(np.ascontiguousarray(G["X"]))   # copied in once
        remote = ray.remote(ray_chunk)
        search = lambda: merge(workload, ray.get([remote.remote(Q_ref, X_ref, a, b, workload) for a, b in T]))
        search()
        return search, ray.shutdown

    if strategy == "asyncio":
        return (lambda: merge(workload, asyncio.run(_coroutines(T, fn)))), done

    if strategy == "asyncio_executor":
        ex = ThreadPoolExecutor(W)
        return (lambda: merge(workload, asyncio.run(_executor(T, fn, ex)))), ex.shutdown

    raise ValueError(strategy)


def reference_scores(workload):
    if workload == "python":
        S = G["Q"].astype(np.float64) @ G["X"].astype(np.float64).T
        return -np.sort(-S, axis=1)[:, :K_TOP]
    return np.load(RESULTS / "phase1_reference_top10_mpnet.npz")["scores"]


def run_one(workload, strategy, W):
    load_globals(workload)
    t0 = time.perf_counter()
    search, cleanup = make_search(strategy, workload, W)
    setup_s = time.perf_counter() - t0
    try:
        warmup = 1 if workload == "python" else 2
        r = benchmark(search, warmup=warmup, runs=5)
        err = float(np.max(np.abs(r["result"][1].astype(np.float64) - reference_scores(workload))))
    finally:
        cleanup()
    return {
        "workload": workload, "strategy": strategy, "workers": W,
        "median_s": r["median_s"], "std_s": r["std_s"], "min_s": r["min_s"],
        "warmup": r["warmup"], "runs": r["runs"], "setup_s": setup_s,
        "n_tasks": len(task_bounds(workload, W)) if strategy not in ("blas_threads", "phase2_best", "numba_prange") else 1,
        "max_abs_err": err,
        "blas_threads": os.environ.get("VECLIB_MAXIMUM_THREADS", "default"),
        "times": r["times"],
    }


# --- orchestration ---------------------------------------------------------------------------------

def configs():
    out = []
    for W in WORKERS:
        for s in ("blas_threads", "threadpool", "processpool", "processpool_pickle", "dask", "ray", "asyncio_executor"):
            out.append(("blas", s, W))
        for s in ("numba_prange", "threadpool", "processpool", "dask", "ray"):
            out.append(("cpu", s, W))
        for s in ("threadpool", "processpool"):
            out.append(("python", s, W))
    out += [("blas", "asyncio", 1), ("blas", "asyncio", 10), ("blas", "phase2_best", 10)]
    return out


def run_all():
    if OUT.exists():
        OUT.unlink()
    todo = configs()
    for n, (workload, strategy, W) in enumerate(todo, 1):
        env = os.environ.copy()
        if strategy == "phase2_best":
            env.pop("VECLIB_MAXIMUM_THREADS", None)
        else:
            env["VECLIB_MAXIMUM_THREADS"] = str(W) if strategy == "blas_threads" else "1"
        p = subprocess.run([sys.executable, str(Path(__file__)), "--one", workload, strategy, str(W)],
                           env=env, capture_output=True, text=True, timeout=600)
        line = next((l for l in p.stdout.splitlines()[::-1] if l.startswith("RESULT ")), None)
        if line is None:
            print(f"[{n}/{len(todo)}] FAILED {workload} {strategy} W={W}\n{p.stderr[-2000:]}", flush=True)
            continue
        row = json.loads(line[len("RESULT "):])
        save_row(OUT, row)
        print(f"[{n}/{len(todo)}] {workload:6s} {strategy:18s} W={W:>2}  {row['median_s'] * 1000:10.2f} ms  "
              f"setup {row['setup_s']:.2f}s  err {row['max_abs_err']:.1e}", flush=True)
        time.sleep(COOLDOWN_S)
    import phase3_plot
    phase3_plot.main()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--one", nargs=3, metavar=("WORKLOAD", "STRATEGY", "WORKERS"))
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if a.one:
        row = run_one(a.one[0], a.one[1], int(a.one[2]))
        print("RESULT " + json.dumps(row))
    elif a.all:
        run_all()
    elif a.plot_only:
        import phase3_plot
        phase3_plot.main()
