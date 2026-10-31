"""
Similarity kernels we wrote ourselves (no sklearn / scipy inside), shared by Phase 2 and later.

Every kernel returns the score matrix S of shape (n_queries, n_docs) with
S[i, j] = <query i, document j>. All our vectors are unit length (TF-IDF rows are
L2-normalised, mpnet ends with a Normalize layer), so this dot product IS the cosine.

Dense kernels (Phase 2C):
  dense_python            pure Python lists, triple loop
  dense_numba             @njit, same loops compiled to machine code
  dense_numba_parallel    @njit(parallel=True), documents split across threads (prange)
  dense_numba_fastmath    @njit(fastmath=True) + float32 accumulator -> compiler may use SIMD
  dense_numba_parallel_fastmath   both

Sparse kernels (Phase 2A), on CSR arrays (indptr, indices, data):
  csr_csr_merge           sparse query . sparse doc, two-pointer merge of sorted indices
  csr_csr_inverted        sparse query . sparse docs via an inverted index (CSC of the corpus)
  csr_dense               sparse doc . dense query

Loop order: documents are the outer loop for the dense kernels, so each document row
(768 floats = 3 KB) is read from memory once and reused for all queries while it sits in cache.
"""

import numpy as np
from numba import njit, prange


# --- Dense . dense ---------------------------------------------------------------------

def dense_python(Q, X):
    """Q, X: Python lists of lists. The reference 'element-wise' implementation."""
    nq, n, d = len(Q), len(X), len(Q[0])
    S = [[0.0] * n for _ in range(nq)]
    for j in range(n):
        x = X[j]
        for i in range(nq):
            q = Q[i]
            acc = 0.0
            for t in range(d):
                acc += q[t] * x[t]
            S[i][j] = acc
    return S


@njit
def dense_numba(Q, X):
    nq, d = Q.shape
    n = X.shape[0]
    S = np.empty((nq, n), dtype=np.float32)
    for j in range(n):
        for i in range(nq):
            acc = 0.0                      # float64 sum (each float32*float32 product is rounded to float32 first)
            for t in range(d):
                acc += Q[i, t] * X[j, t]
            S[i, j] = acc
    return S


@njit(parallel=True)
def dense_numba_parallel(Q, X):
    nq, d = Q.shape
    n = X.shape[0]
    S = np.empty((nq, n), dtype=np.float32)
    for j in prange(n):                    # each thread gets a block of documents
        for i in range(nq):
            acc = 0.0
            for t in range(d):
                acc += Q[i, t] * X[j, t]
            S[i, j] = acc                  # every (i, j) written by exactly one thread: no race
    return S


@njit(fastmath=True)
def dense_numba_fastmath(Q, X):
    nq, d = Q.shape
    n = X.shape[0]
    S = np.empty((nq, n), dtype=np.float32)
    for j in range(n):
        for i in range(nq):
            acc = np.float32(0.0)          # float32 + fastmath lets LLVM reorder the sum into SIMD lanes
            for t in range(d):
                acc += Q[i, t] * X[j, t]
            S[i, j] = acc
    return S


@njit(parallel=True, fastmath=True)
def dense_numba_parallel_fastmath(Q, X):
    nq, d = Q.shape
    n = X.shape[0]
    S = np.empty((nq, n), dtype=np.float32)
    for j in prange(n):
        for i in range(nq):
            acc = np.float32(0.0)
            for t in range(d):
                acc += Q[i, t] * X[j, t]
            S[i, j] = acc
    return S


# --- Sparse . sparse --------------------------------------------------------------------

@njit
def csr_csr_merge(q_ptr, q_idx, q_val, d_ptr, d_idx, d_val):
    """For every (query, doc) pair walk both sorted index lists at once and multiply
    only where the word ids match. Cost per pair ~ nnz(query) + nnz(doc)."""
    nq = q_ptr.shape[0] - 1
    n = d_ptr.shape[0] - 1
    S = np.zeros((nq, n), dtype=np.float32)
    for i in range(nq):
        a0, a1 = q_ptr[i], q_ptr[i + 1]
        for j in range(n):
            p, r = a0, d_ptr[j]
            r1 = d_ptr[j + 1]
            acc = 0.0
            while p < a1 and r < r1:
                ia, ib = q_idx[p], d_idx[r]
                if ia == ib:
                    acc += q_val[p] * d_val[r]
                    p += 1
                    r += 1
                elif ia < ib:
                    p += 1
                else:
                    r += 1
            S[i, j] = acc
    return S


@njit
def csr_csr_inverted(q_ptr, q_idx, q_val, t_ptr, t_doc, t_val, n_docs):
    """Inverted index: t_* is the corpus in CSC form, i.e. for every word the list of
    (doc, weight) it appears in. For each query word we only touch documents that
    contain it. Cost per query ~ sum of document frequencies of its words."""
    nq = q_ptr.shape[0] - 1
    S = np.zeros((nq, n_docs), dtype=np.float32)
    for i in range(nq):
        for p in range(q_ptr[i], q_ptr[i + 1]):
            word, w = q_idx[p], q_val[p]
            for r in range(t_ptr[word], t_ptr[word + 1]):
                S[i, t_doc[r]] += w * t_val[r]
    return S


# --- Sparse . dense ----------------------------------------------------------------------

@njit
def csr_dense(d_ptr, d_idx, d_val, Qd):
    """Sparse documents times dense query vectors (Qd: n_queries x vocab).
    For each document only its nonzeros are visited: sum data[r] * q[indices[r]]."""
    nq = Qd.shape[0]
    n = d_ptr.shape[0] - 1
    S = np.empty((nq, n), dtype=np.float32)
    for i in range(nq):                    # query row (vocab floats = ~100 KB) stays in L1/L2
        for j in range(n):
            acc = 0.0
            for r in range(d_ptr[j], d_ptr[j + 1]):
                acc += d_val[r] * Qd[i, d_idx[r]]
            S[i, j] = acc
    return S


# --- Top-k selection (a loop with no BLAS equivalent) -------------------------------------

@njit
def _topk_row(row, k, out_idx, out_val):
    """One pass over the row, keeping the k best in a small sorted buffer (insertion)."""
    for p in range(k):
        out_val[p] = -np.inf
        out_idx[p] = -1
    for j in range(row.shape[0]):
        s = row[j]
        if s > out_val[k - 1]:              # most scores fail this test and cost one comparison
            p = k - 1
            while p > 0 and out_val[p - 1] < s:
                out_val[p] = out_val[p - 1]
                out_idx[p] = out_idx[p - 1]
                p -= 1
            out_val[p] = s
            out_idx[p] = j


@njit
def topk_rows(S, k):
    nq = S.shape[0]
    idx = np.empty((nq, k), dtype=np.int64)
    val = np.empty((nq, k), dtype=S.dtype)
    for i in range(nq):
        _topk_row(S[i], k, idx[i], val[i])
    return idx, val


@njit(parallel=True)
def topk_rows_parallel(S, k):
    nq = S.shape[0]
    idx = np.empty((nq, k), dtype=np.int64)
    val = np.empty((nq, k), dtype=S.dtype)
    for i in prange(nq):                    # rows are independent
        _topk_row(S[i], k, idx[i], val[i])
    return idx, val


# --- GIL-free versions for thread pools (Phase 3) ------------------------------------------
# nogil=True: while the compiled code runs, other Python threads can run too.
# cache=True: compiled code is saved to disk, so new worker processes do not recompile.

@njit(nogil=True, cache=True)
def topk_rows_nogil(S, k):
    nq = S.shape[0]
    idx = np.empty((nq, k), dtype=np.int64)
    val = np.empty((nq, k), dtype=S.dtype)
    for i in range(nq):
        _topk_row(S[i], k, idx[i], val[i])
    return idx, val


@njit(fastmath=True, nogil=True, cache=True)
def dense_fastmath_nogil(Q, X):
    nq, d = Q.shape
    n = X.shape[0]
    S = np.empty((nq, n), dtype=np.float32)
    for j in range(n):
        for i in range(nq):
            acc = np.float32(0.0)
            for t in range(d):
                acc += Q[i, t] * X[j, t]
            S[i, j] = acc
    return S


def merge_topk(parts, k):
    """Combine per-chunk top-k lists (each n_queries x k) into the global top-k."""
    idx = np.concatenate([p[0] for p in parts], axis=1)
    val = np.concatenate([p[1] for p in parts], axis=1)
    order = np.argsort(-val, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(val, order, axis=1)


def topk_numpy(S, k):
    """NumPy way: argpartition finds the k best (unordered), then sort only those k."""
    part = np.argpartition(-S, k, axis=1)[:, :k]
    vals = np.take_along_axis(S, part, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(vals, order, axis=1)


def csr_arrays(M):
    """(indptr, indices, data) of a scipy CSR matrix, with sorted indices (needed by the merge)."""
    M = M.tocsr()
    M.sort_indices()
    return M.indptr, M.indices, M.data
