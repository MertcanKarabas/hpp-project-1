"""
Live demo: type a sentence, get the most similar 2025-26 arXiv abstracts in milliseconds.

Usage:
  .venv/bin/python demo.py                       # interactive, exact mode (matches the baseline)
  .venv/bin/python demo.py --fast                # SVD k=300 mode (recall@10 ~ 0.95, faster)
  .venv/bin/python demo.py --query "your text"   # one query, then exit
  .venv/bin/python demo.py -k 5                  # results per query (default 10, the same top-10 as the benchmarks)

Everything runs on this machine: the embedding model comes from the local HuggingFace
cache (offline mode), so the demo does not need an internet connection in class.
Two times are printed per query: turning the sentence into a vector (the embedding model,
not part of the benchmarks per the project FAQ) and the search itself.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")              # never touch the network
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_utils import DATA, load_dense  # noqa: E402
from search_engine import SearchEngine  # noqa: E402

MODEL = "sentence-transformers/all-mpnet-base-v2"


def load_docs():
    with open(DATA / "corpus.jsonl") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="use the SVD (k=300) index")
    ap.add_argument("--query", help="run one query and exit")
    ap.add_argument("-k", type=int, default=10, help="results per query")
    # cpu by default: on Apple's GPU (mps) every new sentence length triggers a recompile (~300 ms),
    # while the CPU encodes a short query in a steady ~15-20 ms
    ap.add_argument("--device", default="cpu", help="cpu (default) or mps for the embedding model")
    args = ap.parse_args()

    t0 = time.perf_counter()
    X, _ = load_dense("mpnet")
    docs = load_docs()
    engine = SearchEngine(X, mode="fast" if args.fast else "exact")

    from sentence_transformers import SentenceTransformer
    device = args.device
    model = SentenceTransformer(MODEL, device=device)

    # warm-up: the first encode and the first search initialise the model and the Numba kernels
    for _ in range(2):
        engine.search(model.encode(["warm up"], normalize_embeddings=True), k=args.k)
    print(f"Ready: {len(docs):,} arXiv abstracts (2025-26), mode = {engine.mode}, "
          f"model on {device}, loaded in {time.perf_counter() - t0:.1f} s")

    def run(q):
        t1 = time.perf_counter()
        qv = model.encode([q], normalize_embeddings=True)
        t2 = time.perf_counter()
        idx, val = engine.search(qv, k=args.k)
        t3 = time.perf_counter()
        print(f'\n"{q}"\n  encode {(t2 - t1) * 1000:.1f} ms | search {(t3 - t2) * 1000:.2f} ms '
              f'over {len(docs):,} documents')
        for rank, (i, s) in enumerate(zip(idx[0], val[0]), 1):
            d = docs[i]
            print(f"  {rank}. [{s:.3f}] {d['title'][:95]}")
            print(f"       arXiv:{d['id']}  {d['primary_category']}  {d['published'][:10]}")

    if args.query:
        run(args.query)
        return
    print("Type a query and press Enter (empty line to quit).")
    while True:
        try:
            q = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        run(q)


if __name__ == "__main__":
    main()
