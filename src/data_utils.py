"""
Shared paths and loaders, so every phase reads the same data the same way.

Row i of every matrix = line i of data/corpus.jsonl (queries: data/queries.jsonl).
The corpus order is already random, so X[:n] is a random subset of size n
(used for the 1K / 5K / 10K / 20K scaling experiments).
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EMB = DATA / "embeddings"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

CORPUS_SIZES = [1_000, 5_000, 10_000, 20_000]


def load_dense(model="mpnet", n_docs=None, n_queries=None):
    """Return (corpus, queries) as float32 arrays of shape (N, d) and (Q, d)."""
    X = np.load(EMB / f"{model}_corpus.npy")
    Q = np.load(EMB / f"{model}_queries.npy")
    if n_docs is not None:
        X = X[:n_docs]
    if n_queries is not None:
        Q = Q[:n_queries]
    return X, Q


def load_titles():
    with open(DATA / "corpus.jsonl") as f:
        return [json.loads(line)["title"] for line in f]


def load_query_texts():
    with open(DATA / "queries.jsonl") as f:
        return [json.loads(line)["query"] for line in f]
