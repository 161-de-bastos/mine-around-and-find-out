from .configs import (
    PREPROCESS_CFG, 
    SPACY_ESP_MODEL, 
    TRANSFORMERS_SENTIMENT_MODEL,
    TORCH_CFG,
    WORKABLE_CSV_PATH,
    SENTIMENT_INFERENCE,
    LDA_CFG,
    TOPIC_CFG,
    PAGERANK_CFG     
)

def get_bodies(path):
    import pandas as pd

    return pd.read_csv(path)[['body']].values.tolist()

def load_models():
    from .nlp.preprocess import load_spacy
    from .nlp.sentiment import load_emotions

    nlp = load_spacy(model = SPACY_ESP_MODEL)
    mdl = load_emotions(model = TRANSFORMERS_SENTIMENT_MODEL, cfg = TORCH_CFG)
    return nlp, mdl

def preprocess(txts, nlp, cfg):
    from .nlp.preprocess import preprocess_docs

    return preprocess_docs(
        txts = txts, 
        nlp = nlp, 
        cfg = cfg,
        n_process = cfg['n_process'],
        batch_size = cfg['batch_size']
    )

def run_sentiments(txts, mdl, cfg):
    from .nlp.sentiment import predict_sentiment

    return predict_sentiment(
        txts = txts,
        mdl = mdl,
        **cfg
    )

def run_lda(txts, cfg):
    from .nlp.topics import single_lda

    return single_lda(txts = txts, **cfg)

def run_topic_correlation(txts, vct, mdl, graph_cfg, centrality_cfg):
    from .network.graph import topic_graph
    from .network.centrality import pagerank_power

    sparse, nodes = topic_graph(
        txts = txts,
        vectorizer = vct,
        model = mdl,
        **graph_cfg
    )

    pr = pagerank_power(
        A = sparse,
        **centrality_cfg
    )

    return sparse, nodes, pr

if __name__=='__main__':
    SKIP_SEQUENCE = [ 
        False, 
        False,
        False
    ]
    
    if not SKIP_SEQUENCE[0] or True:
        nlp, mdl = load_models()
        docs = get_bodies(WORKABLE_CSV_PATH)
        txts = preprocess(
            txts = docs,
            nlp = nlp,
            cfg = PREPROCESS_CFG
        )

    if not SKIP_SEQUENCE[1]:
        from .figures import plot_class_distribution

        sentiments = run_sentiments(
            txts = docs, 
            mdl = mdl
        )
        plot_class_distribution(sentiments['avg'])
    
    if not SKIP_SEQUENCE[2]:
        topic, vct, top = run_lda(
            txts = docs, 
            cfg = LDA_CFG
        )

    if not SKIP_SEQUENCE[3] and not SKIP_SEQUENCE[2]:
        from .metrics import graph_stats, degree_centrality
        from .figures import plot_degree_hist

        spr, ns, pr = run_topic_correlation(
            txts = txts,
            vct = vct,
            mdl = topic,
            graph_cfg = TOPIC_CFG,
            centrality_cfg = PAGERANK_CFG
        )

        print(graph_stats(spr))
        plot_degree_hist(degree_centrality(spr))
