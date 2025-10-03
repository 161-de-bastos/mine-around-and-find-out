import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def pagerank_power(A, alpha=0.85, tol=1e-6, max_iter=100):
    n = A.shape[0]
    if n == 0:
        return np.array([])

    row_sums = np.array(A.sum(axis=1)).ravel()
    dangling = (row_sums == 0).astype(float)
    inv = np.where(row_sums == 0, 1.0, 1.0 / row_sums)
    Dinv = csr_matrix((inv, (np.arange(n), np.arange(n))), shape=(n, n))
    P = Dinv @ A

    v = np.full(n, 1.0 / n)
    teleport = np.full(n, (1.0 - alpha) / n)

    for _ in range(max_iter):
        v_next = alpha * (P @ v + (dangling @ v) * (1.0 / n)) + teleport
        if np.linalg.norm(v_next - v, 1) < tol:
            v = v_next
            break
        v = v_next
    return v / v.sum()

def eigenvector_centrality(A, k=1, tol=1e-6, max_iter=200):
    if A.shape[0] == 0:
        return np.array([])
    vals, vecs = eigsh(A, k=k, which="LA", tol=tol, maxiter=max_iter)
    if k == 1:
        v = np.abs(vecs[:, -1])
        s = v.sum()
        return v / s if s != 0 else v
    V = np.abs(vecs)
    return V / (np.sum(V, axis=0, keepdims=True) + 1e-12)

