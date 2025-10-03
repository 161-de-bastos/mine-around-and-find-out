import numpy as np
from scipy.sparse import csr_matrix, issparse

def topic_graph(txts, vectorizer, model, tau = 0.0, normalize_rows = True):
    X = vectorizer.transform(txts)
    T = model.transform(X)
    if normalize_rows:
        rs = T.sum(axis=1, keepdims=True); rs[rs==0] = 1.0 
        T = T / rs
    if tau > 0.0:
        T = np.where(T >= tau, T, 0.0)
    if (T == 0).mean() > 0.8:
        T_sp = csr_matrix(T)
        A_tt = (T_sp.T @ T_sp).tocsr()
    else:
        A_tt = csr_matrix(T.T @ T)
    return A_tt, T