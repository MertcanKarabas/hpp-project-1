"""
The final search system (Phase 4):

  data -> embedding (mpnet, offline) -> optional SVD reduction -> scoring (one BLAS matmul) -> parallel top-k

Each design choice comes from an earlier measurement:
  - dense mpnet vectors, normalised once at build time, so cosine = dot product
    and no norms are computed per query                                 (Phase 1: norms were 58%)
  - all queries of a batch scored by ONE matrix multiply Q @ X.T          (Phase 2C, Phase 3: the
    matrix unit is fastest with one big call; extra threads did not help)
  - top-k with our one-pass Numba kernel, parallel over queries          (Phase 2C: 98x faster than
                                                                          NumPy argpartition)
  - optional "fast" mode: uncentred truncated SVD with our block power
    iteration, 300 dims, recall@10 ~ 0.95                                (Phase 2B)
  - optional chunking over documents: bounded memory for big corpora     (Phase 3, 10x question)
"""

import numpy as np

import kernels as K
from svd_power import block_power_svd

PARALLEL_TOPK_MIN_QUERIES = 8      # below this, thread start-up costs more than it saves


def normalise(M):
    M = np.asarray(M, dtype=np.float32)
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def topk(S, k):
    if S.shape[0] >= PARALLEL_TOPK_MIN_QUERIES:
        return K.topk_rows_parallel(S, k)
    return K.topk_rows(S, k)


class SearchEngine:
    def __init__(self, corpus, mode="exact", dims=300, chunk_docs=None, svd_iters=6):
        """corpus: (n_docs, 768) embeddings. mode: 'exact' (768 dims) or 'fast' (SVD, `dims` dims)."""
        if mode not in ("exact", "fast"):
            raise ValueError(mode)
        self.mode = mode
        self.chunk_docs = chunk_docs
        X = normalise(corpus)
        self.P = None
        if mode == "fast":
            _, _, Vt = block_power_svd(X, dims, n_iter=svd_iters)
            self.P = np.ascontiguousarray(Vt.T)                 # 768 x dims projection, computed once
            X = normalise(X @ self.P)
        self.X = np.ascontiguousarray(X, dtype=np.float32)

    def prepare(self, Q):
        """Queries -> unit vectors in the same space as the corpus."""
        Q = normalise(np.atleast_2d(Q))
        if self.P is not None:
            Q = normalise(Q @ self.P)
        return np.ascontiguousarray(Q)

    def search(self, Q, k=10):
        """Top-k (document ids, cosine scores) for every query row, best first."""
        if not 1 <= k <= self.X.shape[0]:
            raise ValueError(f"k must be between 1 and {self.X.shape[0]}, got {k}")
        Q = self.prepare(Q)
        if self.chunk_docs is None:
            return topk(Q @ self.X.T, k)
        parts = []
        for a in range(0, self.X.shape[0], self.chunk_docs):     # never holds more than one chunk of scores
            idx, val = topk(Q @ self.X[a:a + self.chunk_docs].T, k)
            parts.append((idx + a, val))
        return K.merge_topk(parts, k)
