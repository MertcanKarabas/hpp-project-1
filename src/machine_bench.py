"""
Bonus: the same benchmark suite on two machines (laptop + Google Colab), plus the numbers
a roofline model needs.

A compact, portable version of the project's key measurements. It uses the project's own
kernels, the final SearchEngine and the same benchmark() rules (warmup, >= 5 runs, median),
and writes one JSON file per machine:

  results/machine_<name>.json

  1. env        CPU model, physical / logical cores, caches, RAM, BLAS library, versions
  2. roofs      peak float32 compute (large SGEMM) and memory bandwidth (STREAM-style kernels)
  3. kernels    Q @ X.T for 1 / 20 / 200 queries (arithmetic intensity 0.5 -> 79 FLOP/byte),
                our Numba kernels and both top-k versions: the points of the roofline plot
  4. overview   the six versions of the Phase 4 overview (200 queries x 20,000 documents)
  5. engine     final SearchEngine, exact and fast mode, 1K-20K documents, and correctness
                against the Phase 1 reference (results/phase1_reference_top10_mpnet.npz)
  6. threads    BLAS threads, Numba threads, pure-Python thread pool vs process pool (GIL)

Usage:
  .venv/bin/python src/machine_bench.py --name laptop     # on the laptop (~1 min)
  python src/machine_bench.py --name colab                # inside notebooks/colab_machine_bench.ipynb
  .venv/bin/python src/machine_bench.py --pack            # colab_pack.zip: code + embeddings for Colab
  .venv/bin/python src/machine_compare.py                 # figures + table from both JSON files
"""

import os
import sys

# Every BLAS library reads its thread limit once, when NumPy loads. The --blas-probe child process
# gets these from its parent (run_blas_threads), so nothing may import NumPy before that happens.
BLAS_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit, prange  # noqa: E402

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import kernels as K  # noqa: E402
from benchmark_utils import benchmark  # noqa: E402
from data_utils import CORPUS_SIZES, EMB, RESULTS, ROOT, load_dense  # noqa: E402
from phase1_naive_search import search_naive  # noqa: E402
from search_engine import SearchEngine  # noqa: E402

TOP = 10
D = 768
N_FULL = 20_000
NAIVE_QUERIES = 4                 # naive Python over 20K docs: time 4 queries, scale x50 to 200 (as Phase 4)
GIL_QUERIES, GIL_DOCS = 20, 1_000
STREAM_N = 2 ** 26                # 64M float32 = 256 MB per array, far bigger than any cache
SGEMM_SIZES = (2048, 4096)
PACK = ROOT / "colab_pack.zip"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- 1. environment -------------------------------------------------------------------------------

def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def env_info():
    import numba
    import scipy

    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "os": platform.platform(),
        "machine": platform.machine(),
        "logical_cores": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "numba_threads": numba.config.NUMBA_NUM_THREADS,
    }
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        version = blas.get("version", "")
        info["blas"] = blas.get("name", "?") + (f" {version}" if version and version != "unknown" else "")
    except Exception:
        info["blas"] = "unknown"

    if sys.platform == "darwin":
        g = lambda key: sh(["sysctl", "-n", key])                          # noqa: E731
        p_cores, e_cores = g("hw.perflevel0.physicalcpu"), g("hw.perflevel1.physicalcpu")
        info.update({
            "cpu": g("machdep.cpu.brand_string"),
            "physical_cores": int(g("hw.physicalcpu") or 0),
            "threads_per_core": 1,
            "core_note": f"{p_cores} performance + {e_cores} efficiency cores",
            "l1d": f"{int(g('hw.perflevel0.l1dcachesize') or 0) // 1024} KB per P-core",
            "l2": f"{int(g('hw.perflevel0.l2cachesize') or 0) // 2**20} MB shared by the P-cores",
            "l3": "none (a system-level cache is shared with the GPU)",
            "simd": "NEON 128-bit + SME matrix unit",
            "ram_gb": round(int(g("hw.memsize") or 0) / 2**30, 1),
        })
    else:
        fields = {}
        for line in sh(["lscpu"]).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fields[k.strip()] = v.strip()
        num = lambda key: int(fields.get(key, "0").split()[0] or 0)      # noqa: E731
        sockets, cores, tpc = num("Socket(s)"), num("Core(s) per socket"), num("Thread(s) per core")
        flags = fields.get("Flags", "").split()
        info.update({
            "cpu": fields.get("Model name", platform.processor()),
            "physical_cores": sockets * cores,
            "threads_per_core": tpc,
            "core_note": f"{sockets} socket x {cores} core(s) x {tpc} thread(s) per core",
            "l1d": fields.get("L1d cache", "?"),
            "l2": fields.get("L2 cache", "?"),
            "l3": fields.get("L3 cache", "?"),
            "simd": ", ".join(f for f in ("sse4_2", "avx", "avx2", "fma", "avx512f", "avx512_vnni", "amx_tile")
                              if f in flags) or "?",
            "ram_gb": None,
        })
        if Path("/proc/meminfo").exists():
            info["ram_gb"] = round(int(Path("/proc/meminfo").read_text().split()[1]) / 2**20, 1)
    return info


# --- 2. roofs: peak compute and memory bandwidth --------------------------------------------------

@njit(parallel=True)
def stream_triad(a, b, c, s):
    for i in prange(a.shape[0]):
        a[i] = b[i] + s * c[i]


@njit(parallel=True, fastmath=True)
def stream_read(b):
    total = 0.0
    for i in prange(b.shape[0]):
        total += b[i]
    return total


def roofs():
    out = {"sgemm": []}
    rng = np.random.default_rng(0)
    for n in SGEMM_SIZES:
        A = rng.standard_normal((n, n), dtype=np.float32)
        B = rng.standard_normal((n, n), dtype=np.float32)
        r = benchmark(np.matmul, A, B, warmup=2, runs=5)
        gf = 2 * n ** 3 / r["median_s"] / 1e9
        out["sgemm"].append({"n": n, "ms": r["median_s"] * 1e3, "gflops": gf})
        log(f"  SGEMM {n}x{n}: {gf:8.1f} GFLOP/s")
        del A, B
    out["peak_gflops"] = max(s["gflops"] for s in out["sgemm"])

    a = np.zeros(STREAM_N, dtype=np.float32)
    b = np.ones(STREAM_N, dtype=np.float32)
    c = np.full(STREAM_N, 2.0, dtype=np.float32)
    r = benchmark(stream_triad, a, b, c, np.float32(3.0), warmup=2, runs=7)
    out["triad_gbs"] = 12 * STREAM_N / r["median_s"] / 1e9               # read b, c + write a (STREAM count)
    r = benchmark(stream_read, b, warmup=2, runs=7)
    out["read_gbs"] = 4 * STREAM_N / r["median_s"] / 1e9
    r = benchmark(np.copyto, a, b, warmup=2, runs=7)
    out["copy_1thread_gbs"] = 8 * STREAM_N / r["median_s"] / 1e9          # NumPy copy: one thread
    out["bandwidth_gbs"] = max(out["triad_gbs"], out["read_gbs"])
    out["ridge_flop_per_byte"] = out["peak_gflops"] / out["bandwidth_gbs"]
    log(f"  bandwidth: triad {out['triad_gbs']:.1f}, read {out['read_gbs']:.1f}, "
        f"1-thread copy {out['copy_1thread_gbs']:.1f} GB/s; ridge {out['ridge_flop_per_byte']:.1f} FLOP/byte")
    return out


# --- 3. kernels: the points of the roofline -------------------------------------------------------

def kernel_rows(X, Q):
    N = X.shape[0]
    rows = []

    def add(name, f, flops, nbytes, warmup=2, runs=5):
        r = benchmark(f, warmup=warmup, runs=runs)
        s = r["median_s"]
        rows.append({"name": name, "median_s": s, "std_s": r["std_s"], "flops": flops, "bytes": nbytes,
                     "ai": flops / nbytes, "gflops": flops / s / 1e9, "gbs": nbytes / s / 1e9})
        log(f"  {name:28s} {s * 1e3:10.3f} ms  {flops / s / 1e9:8.1f} GFLOP/s  AI {flops / nbytes:6.2f}")

    for q in (1, 20, 200):
        Qq = np.ascontiguousarray(Q[:q])
        # compulsory traffic: read Q and X once, write S once
        add(f"matmul_q{q}", lambda Qq=Qq: Qq @ X.T, 2 * q * N * D, 4 * (q * D + N * D + q * N))
    full = (2 * 200 * N * D, 4 * (200 * D + N * D + 200 * N))
    add("numba_njit_q200", lambda: K.dense_numba(Q, X), *full, warmup=1)
    add("numba_prange_fastmath_q200", lambda: K.dense_numba_parallel_fastmath(Q, X), *full)
    S = Q @ X.T
    # top-k does ~1 comparison per score and reads every score once: "operations" = comparisons
    add("topk_numba_parallel", lambda: K.topk_rows_parallel(S, TOP), 200 * N, 4 * 200 * N)
    add("topk_numpy_argpartition", lambda: K.topk_numpy(S, TOP), 200 * N, 4 * 200 * N)
    return rows


# --- 4. overview: the six versions of Phase 4 -----------------------------------------------------

def overview(X, Q, ref_scores):
    rows = []

    def add(name, seconds, err, note=""):
        rows.append({"version": name, "seconds_200_queries": seconds, "max_abs_err": err, "note": note})
        log(f"  {name:44s} {seconds * 1e3:12.2f} ms")

    Xl, Ql = X.tolist(), Q[:NAIVE_QUERIES].tolist()
    r = benchmark(search_naive, Ql, Xl, TOP, warmup=1, runs=5)
    got = np.array([[s for _, s in q] for q in r["result"]])
    add("Naive Python loops (Phase 1)", r["median_s"] * 200 / NAIVE_QUERIES,
        float(np.max(np.abs(got - ref_scores[:NAIVE_QUERIES]))), f"{NAIVE_QUERIES} queries timed, scaled to 200")
    del Xl

    exact, fast = SearchEngine(X), SearchEngine(X, mode="fast")
    versions = [
        ("Numba @njit loops + top-k", lambda: K.topk_rows(K.dense_numba(Q, X), TOP), 1),
        ("Numba parallel + fastmath, parallel top-k",
         lambda: K.topk_rows_parallel(K.dense_numba_parallel_fastmath(Q, X), TOP), 2),
        ("NumPy matmul + NumPy argpartition", lambda: K.topk_numpy(Q @ X.T, TOP), 2),
        ("Final, exact (matmul + Numba parallel top-k)", lambda: exact.search(Q, TOP), 2),
        ("Final, fast (SVD k=300)", lambda: fast.search(Q, TOP), 2),
    ]
    for name, f, warm in versions:
        r = benchmark(f, warmup=warm, runs=5)
        add(name, r["median_s"], float(np.max(np.abs(r["result"][1] - ref_scores))),
            "approximate (recall@10 ~0.95)" if name.startswith("Final, fast") else "")
    return rows


# --- 5. engine: scaling and correctness -----------------------------------------------------------

def engine(X, Q, ref):
    rows = []
    for n in CORPUS_SIZES:
        for mode in ("exact", "fast"):
            t0 = time.perf_counter()
            eng = SearchEngine(X[:n], mode=mode)
            build_s = time.perf_counter() - t0
            batch = benchmark(eng.search, Q, TOP, warmup=2, runs=5)
            single = benchmark(lambda: [eng.search(Q[i:i + 1], TOP) for i in range(20)], warmup=1, runs=5)
            rows.append({"n_docs": n, "mode": mode, "build_s": build_s, "batch200_ms": batch["median_s"] * 1e3,
                         "single_query_ms": single["median_s"] * 1e3 / 20})
            log(f"  {mode:5s} {n:6d} docs: batch {rows[-1]['batch200_ms']:8.3f} ms, "
                f"single {rows[-1]['single_query_ms']:7.3f} ms, build {build_s:6.3f} s")

    idx, val = SearchEngine(X).search(Q, TOP)
    fidx, _ = SearchEngine(X, mode="fast").search(Q, TOP)
    correct = {
        "exact_max_abs_score_diff": float(np.max(np.abs(val - ref["scores"]))),
        "exact_identical_top10_lists": float(np.mean(np.all(idx == ref["indices"], axis=1))),
        "exact_identical_top10_sets": float(np.mean([set(a) == set(b) for a, b in zip(idx, ref["indices"])])),
        "fast_recall_at_10": float(np.mean([len(set(a) & set(b)) / TOP for a, b in zip(fidx, ref["indices"])])),
    }
    log(f"  correctness vs Phase 1 reference: {correct}")
    return rows, correct


# --- 6. threads -----------------------------------------------------------------------------------

def worker_counts():
    n = os.cpu_count() or 1
    return sorted({w for w in (1, 2, 4, 8, n) if w <= n})


def blas_probe():
    """Child process: one Q @ X.T (200 x 20K) with the BLAS thread limit set by the parent."""
    X, Q = load_dense("mpnet")
    r = benchmark(lambda: Q @ X.T, warmup=3, runs=7)
    print(json.dumps({"ms": r["median_s"] * 1e3, "gflops": 2 * 200 * N_FULL * D / r["median_s"] / 1e9}))


def run_blas_threads():
    rows = []
    for w in worker_counts() + ["default"]:
        env = os.environ.copy()
        for v in BLAS_THREAD_VARS:
            env.pop(v, None)
            if w != "default":
                env[v] = str(w)
        out = subprocess.run([sys.executable, __file__, "--blas-probe"], env=env, capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(out.stderr)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        rows.append({"threads": str(w), **r})
        log(f"  BLAS threads {str(w):7s}: {r['ms']:8.3f} ms  {r['gflops']:8.1f} GFLOP/s")
    return rows


def run_numba_threads(X, Q):
    import numba

    rows = []
    for w in worker_counts():
        numba.set_num_threads(w)
        r = benchmark(lambda: K.topk_rows_parallel(K.dense_numba_parallel_fastmath(Q, X), TOP), warmup=2, runs=5)
        rows.append({"threads": w, "ms": r["median_s"] * 1e3})
        log(f"  Numba threads {w:3d}: {rows[-1]['ms']:8.2f} ms")
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    return rows


G = {}


def gil_init():
    X, Q = load_dense("mpnet", n_docs=GIL_DOCS, n_queries=GIL_QUERIES)
    G["X"], G["Q"] = X.tolist(), Q.tolist()


def gil_task(bounds):
    a, b = bounds
    return search_naive(G["Q"][a:b], G["X"], TOP)


def run_gil():
    """Pure Python search split by query: serial, thread pool, process pool (W = all cores)."""
    gil_init()
    W = os.cpu_count() or 1
    edges = np.linspace(0, GIL_QUERIES, W + 1, dtype=int)
    tasks = [(int(edges[i]), int(edges[i + 1])) for i in range(W)]
    out = {"workers": W, "queries": GIL_QUERIES, "docs": GIL_DOCS}
    out["serial_s"] = benchmark(gil_task, (0, GIL_QUERIES), warmup=1, runs=5)["median_s"]
    with ThreadPoolExecutor(W) as ex:
        out["threadpool_s"] = benchmark(lambda: list(ex.map(gil_task, tasks)), warmup=1, runs=5)["median_s"]
    # spawn on both machines (macOS default; Linux would fork a process that already runs Numba threads)
    with ProcessPoolExecutor(W, mp_context=multiprocessing.get_context("spawn"), initializer=gil_init) as ex:
        out["processpool_s"] = benchmark(lambda: list(ex.map(gil_task, tasks)), warmup=1, runs=5)["median_s"]
    log(f"  pure Python, {W} workers: serial {out['serial_s']:.3f} s, threads {out['threadpool_s']:.3f} s, "
        f"processes {out['processpool_s']:.3f} s")
    return out


# --- orchestration --------------------------------------------------------------------------------

def run(name):
    t_start = time.perf_counter()
    X, Q = load_dense("mpnet")
    X, Q = np.ascontiguousarray(X), np.ascontiguousarray(Q)
    ref = np.load(RESULTS / "phase1_reference_top10_mpnet.npz")
    res = {"name": name}

    log("1/6 environment")
    res["env"] = env_info()
    log(f"  {res['env']['cpu']} | {res['env']['core_note']} | BLAS {res['env']['blas']}")
    log("2/6 roofs: peak compute and memory bandwidth")
    res["roofs"] = roofs()
    log("3/6 kernels")
    res["kernels"] = kernel_rows(X, Q)
    log("4/6 overview (200 queries x 20,000 docs)")
    res["overview"] = overview(X, Q, ref["scores"])
    log("5/6 final engine")
    res["engine"], res["correctness"] = engine(X, Q, ref)
    log("6/6 threads")
    res["threads"] = {"blas": run_blas_threads(), "numba": run_numba_threads(X, Q), "gil": run_gil()}
    res["total_runtime_s"] = time.perf_counter() - t_start

    out = RESULTS / f"machine_{name}.json"
    out.write_text(json.dumps(res, indent=1))
    log(f"done in {res['total_runtime_s'] / 60:.1f} min -> {out}")


def pack():
    """Everything the Colab notebook needs, in one zip: code, embeddings, Phase 1 reference."""
    files = sorted(p for p in SRC.glob("*.py"))
    files += [EMB / "mpnet_corpus.npy", EMB / "mpnet_queries.npy", RESULTS / "phase1_reference_top10_mpnet.npz"]
    with zipfile.ZipFile(PACK, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT))
    print(f"{PACK} ({PACK.stat().st_size / 2**20:.0f} MB, {len(files)} files)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="machine label, e.g. laptop or colab")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--blas-probe", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.blas_probe:
        blas_probe()
    elif a.pack:
        pack()
    elif a.name:
        run(a.name)
    else:
        ap.print_help()
