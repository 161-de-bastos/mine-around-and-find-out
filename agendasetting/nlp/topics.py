import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def single_lda(txts, n_topics = 10, max_features = 6000, random_state = 42):
    cv = CountVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
    X = cv.fit_transform(txts)
    lda = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method="batch",
            random_state=random_state
        ).fit(X)

    comp = lda.components_
    vocab = {i: t for t, i in cv.vocabulary_.items()}
    top = {}

    for k in range(comp.shape[0]):
        idx = np.argpartition(comp[k], -10)[-10:]
        idx = idx[np.argsort(comp[k][idx])[::-1]]
        top[k] = [(vocab[i], float(comp[k][i])) for i in idx]

    return lda, cv, top