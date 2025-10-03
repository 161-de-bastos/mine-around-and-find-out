import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
import scipy.spatial
import scipy.sparse.linalg

def pagerank_power(A, p, alpha, max_iter, tol):
    n = A.shape[0]
    if n == 0:
        return np.array([])
    
    r = np.array(A.sum(axis=1)).ravel()
    k = r.nonzero()[0]
    
    D_1 = csr_matrix((1 / r[k], (k,k)), shape = (n,n))

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