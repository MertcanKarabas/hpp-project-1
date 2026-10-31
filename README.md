# High-Performance Similarity Search Engine

YZM 513 – Project 1. Top-10 semantic search over **20,000 arXiv abstracts (cs.AI / cs.CL / cs.LG, 2025–2026)** with 200 queries, taken from a naive pure-Python baseline to a system that is **33,000× faster with identical results**.

| Version (200 queries × 20,000 docs, top-10) | Time | Speedup |
|---|---:|---:|
| Naive Python loops (Phase 1 baseline) | 155 s | 1× |
| Numba `@njit` loops | 1.38 s | 113× |
| Numba parallel + fastmath | 37.9 ms | 4,100× |
| NumPy matmul + NumPy argpartition | 33.9 ms | 4,600× |
| **Final, exact** (NumPy matmul + Numba parallel top-k) | **4.67 ms** | **33,000×** |
| Final, fast (our SVD, k = 300, recall@10 = 0.95) | 2.42 ms | 64,000× |

The exact mode returns the same top-10 lists as the baseline for all 200 queries (max score difference 8.4e-7, tolerance 1e-5). Live demo: ~18 ms to embed a query + **0.9 ms to search**.

![Overview](figures/fig7_overview.png)

## Hardware and software
Apple M4 (4 performance + 6 efficiency cores), 16 GB RAM, macOS; Python 3.13.7, NumPy 2.5.4 with Apple Accelerate (BLAS), Numba 0.68.1. Details: `results/environment.json`.

On Apple Silicon, Accelerate runs matrix multiplication on a dedicated matrix unit (AMX/SME). This explains two findings: NumPy beats every hand-written loop, and extra CPU threads barely speed up the matrix multiply (see Phase 3).

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. dataset: arXiv API, ~10 min (rate-limited), see data/dataset_card.md
.venv/bin/python src/data_prep.py

# 2. vectors: TF-IDF + mpnet (768-d) + MiniLM (384-d); ~20 min on Apple GPU (mps), longer on CPU
.venv/bin/python src/embed.py

# 3. live demo
.venv/bin/python demo.py               # exact mode
.venv/bin/python demo.py --fast        # SVD k=300 mode
```

## Reproducing every result

All benchmarks use warmup runs and ≥ 5 timed runs (median), enforced by `src/benchmark_utils.py`. Raw timings go to `results/`, figures to `figures/`.

| Phase | Command | Time | Output |
|---|---|---:|---|
| 1 Baseline + profiling | `.venv/bin/python src/phase1_naive_search.py bench` | ~3 min | `results/phase1_baseline.csv` |
| | `.venv/bin/python src/phase1_profiling.py` | ~2 min | Figure 1, cProfile / line_profiler output |
| | `.venv/bin/python src/phase1_naive_search.py reference` | ~3 min | top-10 reference for the correctness checks |
| 2A Sparse vs dense | `.venv/bin/python src/phase2a_sparse_dense.py` | ~4 min | Figure 2 (needs ~2 GB RAM) |
| 2B Dimensionality reduction | `.venv/bin/python src/phase2b_svd.py` | ~12 min | Figure 3 |
| 2C Loops vs NumPy vs Numba | `.venv/bin/python src/phase2c_vectorise.py` | ~3 min | Figure 4 |
| 3 Parallelism | `.venv/bin/python src/phase3_parallel.py --all` | ~15 min | Figure 5 (87 configurations) |
| 4 Final system | `.venv/bin/python src/phase4_final.py` | ~4 min | Figures 6 and 7 |

Every script also accepts `--plot-only` to redraw its figure from the saved results.

## Figures

| # | File | Conclusion |
|---|---|---|
| 1 | `fig1_profile.png`, `fig1b_cprofile.png` | 99.6% of the naive time is the element-wise cosine loop; 58% is recomputing norms of unit vectors; the full sort is only 0.3%. |
| 2 | `fig2_sparse_dense.png` | TF-IDF takes 15 MB as CSR vs 2,075 MB dense; an inverted index searches it 730× faster than our dense mpnet loop. |
| 3 | `fig3_svd.png` | Our block power iteration is 8× faster than full SVD at k = 100; PCA's centring caps recall@10 at 0.74, uncentred SVD reaches 0.95 at k = 300. |
| 4 | `fig4_vectorise.png` | NumPy is 3,700× faster than Python loops and beats Numba on matrix multiply; Numba wins 22–98× on top-k, which has no BLAS equivalent. |
| 5 | `fig5_parallel.png` | CPU-bound search scales 4.5× on 10 asymmetric cores; threads give 1.0× on pure Python (GIL); no explicit parallelism beats a single BLAS call on the matrix unit. |
| 6 | `fig6_scaling.png` | Batch search scales linearly (~250 µs per 1,000 docs) up to 200K docs; single queries jump 4.8× when the corpus outgrows the 16 MB L2 cache. |
| 7 | `fig7_overview.png` | Every step of the story on one task: 113× → 4,100× → 4,600× → 33,000× (exact) → 64,000× (fast). |

## Repository layout

```
├── demo.py                      live demo
├── requirements.txt
├── src/
│   ├── data_prep.py             download + clean the arXiv dataset
│   ├── embed.py                 TF-IDF, mpnet, MiniLM vectors (offline)
│   ├── benchmark_utils.py       warmup + >= 5 runs + median, CSV rows, environment info
│   ├── data_utils.py            shared paths and loaders
│   ├── kernels.py               our similarity / top-k kernels (Python, Numba)
│   ├── svd_power.py             our power-iteration and block power-iteration SVD
│   ├── phase1_naive_search.py   naive baseline + reference results
│   ├── phase1_profiling.py      cProfile, line_profiler, time breakdown
│   ├── phase2a_sparse_dense.py  sparse vs dense
│   ├── phase2b_svd.py           SVD / TruncatedSVD / PCA comparison
│   ├── phase2c_vectorise.py     Python vs NumPy vs Numba (+ top-k)
│   ├── phase3_parallel.py       7 parallel strategies x 3 workloads
│   ├── phase3_plot.py           Amdahl fit + Figure 5
│   ├── search_engine.py         the final system
│   └── phase4_final.py          correctness, scaling, overview
├── data/                        dataset_card.md, dataset_stats.json (large files rebuilt by the scripts)
├── results/                     raw benchmark data (CSV / JSON)
├── figures/                     all figures, generated by code
└── docs/                        step-by-step explanations (Turkish)
```

## Detailed write-ups (Turkish)
1. [Vectorisation](docs/01_vektorlestirme.md): dataset, TF-IDF, CSR, mpnet
2. [Phase 1](docs/02_phase1_baseline.md): baseline and profiling
3. [Phase 2A](docs/03_phase2a_sparse_dense.md): sparse vs dense
4. [Phase 2B](docs/04_phase2b_svd.md): dimensionality reduction
5. [Phase 2C](docs/05_phase2c_vectorise.md): loops vs NumPy vs Numba
6. [Phase 3](docs/06_phase3_parallel.md): parallelism and concurrency
7. [Phase 4](docs/07_phase4_final.md): final system and scaling

## Notes
- **Dataset:** `src/data_prep.py` queries the arXiv API (3 categories × 22 months × 3 windows). Raw responses are cached in `data/raw/`, so reruns do not hit the API again. Preprocessing and statistics: `data/dataset_card.md`.
- **SSL on python.org builds of Python (macOS):** `data_prep.py` uses `certifi` certificates automatically; verification stays on.
- **Embedding time** is excluded from all benchmarks (project FAQ). The demo prints it separately.
- **Synthetic data** (50K–200K documents) is used only in Phase 4, only to time the 10× scale-up question. All quality and correctness numbers come from the real corpus.
