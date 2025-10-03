import numpy as np

def graph_stats(A):
    n = A.shape[0]
    symmetric = (A - A.T).nnz == 0
    m = int(A.nnz / 2) if symmetric else int(A.nnz)
    deg = np.array(A.sum(axis=1)).ravel()
    return {"nodes": n, "edges": m, "avg_degree": float(deg.mean() if n else 0.0)}

def degree_centrality(A):
    return np.array(A.sum(axis=1)).ravel()