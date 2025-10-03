import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score
)
from gensim.models.coherencemodel import CoherenceModel

def plot_class_distribution(scores_avg, title="Distribución de sentimiento (promedio corpus)"):
    labels = list(scores_avg.keys())
    vals = [scores_avg[k] for k in labels]
    
    plt.figure()
    sns.barplot(x=labels, y=vals)
    plt.title(title)
    plt.ylabel("Probabilidad media")
    plt.tight_layout()

def topic_coherence(model, vectorizer, docs_tokens, measure="c_v", topn=10):
    vocab = {i: t for t, i in vectorizer.vocabulary_.items()}
    topics = []
    for k in range(model.components_.shape[0]):
        row = model.components_[k]
        idx = np.argpartition(row, -topn)[-topn:]
        idx = idx[np.argsort(row[idx])[::-1]]
        topics.append([vocab[i] for i in idx])
    cm = CoherenceModel(topics=topics, texts=docs_tokens, coherence=measure)
    return float(cm.get_coherence())

def topic_diversity(model, vectorizer, topn=25):
    vocab = {i: t for t, i in vectorizer.vocabulary_.items()}
    all_terms = []
    for k in range(model.components_.shape[0]):
        row = model.components_[k]
        idx = np.argpartition(row, -topn)[-topn:]
        idx = idx[np.argsort(row[idx])[::-1]]
        all_terms.extend([vocab[i] for i in idx])
    unique = len(set(all_terms))
    total = len(all_terms)
    return float(unique / total) if total else 0.0

def graph_stats(A):
    n = A.shape[0]
    symmetric = (A - A.T).nnz == 0
    m = int(A.nnz / 2) if symmetric else int(A.nnz)
    deg = np.array(A.sum(axis=1)).ravel()
    return {"nodes": n, "edges": m, "avg_degree": float(deg.mean() if n else 0.0)}

def plot_bar_topk(scores_dict, k=10, title="Top-k PageRank"):
    items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    names = [a for a,_ in items]
    vals = [b for _,b in items]

    plt.figure()
    sns.barplot(x=names, y=vals)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()

def plot_degree_hist(A, title="Distribución de grados"):
    deg = np.array(A.sum(axis=1)).ravel()
    
    plt.figure()
    sns.histplot(deg, bins=20, kde=False)
    plt.title(title)
    plt.xlabel("grado")
    plt.ylabel("frecuencia")
    plt.tight_layout()