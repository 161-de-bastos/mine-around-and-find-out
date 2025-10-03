from .configs import (
    PREPROCESS_CFG, 
    SPACY_ESP_MODEL, 
    TRANSFORMERS_SENTIMENT_MODEL,
    TORCH_CFG,
    WORKABLE_CSV_PATH,
    SENTIMENT_INFERENCE,
    LDA_CFG,
    PAGERANK_CFG     
)

SKIP_SEQUENCE = [ 
    False, 
    False,
    False
]

def get_bodies(path):
    import pandas as pd

    return pd.read_csv(path)[['body']].values.tolist()

def load_models():
    from .nlp.preprocess import load_spacy
    from .nlp.sentiment import load_emotions

    nlp = load_spacy(model = SPACY_ESP_MODEL)
    mdl = load_emotions(model = TRANSFORMERS_SENTIMENT_MODEL, cfg = TORCH_CFG)
    return nlp, mdl

def preprocess(txts, nlp):
    from .nlp.preprocess import preprocess_docs

    return preprocess_docs(
        txts = txts, 
        nlp = nlp, 
        n_process = PREPROCESS_CFG['n_process'],
        batch_size = PREPROCESS_CFG['batch_size']
    )

def run_sentiments(txts, mdl):
    from .nlp.sentiment import predict_sentiment

    return predict_sentiment(
        txts = txts,
        mdl = mdl,
        **SENTIMENT_INFERENCE
    )

def run_lda(txts, cfg):
    from .nlp.topics import single_lda

    return single_lda(txts = txts, **cfg)

def run_topic_correlation(txts, vct, mdl, graph_cfg, centrality_cfg):
    from .network.graph import topic_graph
    from .network.centrality import pagerank_power

    nodes, sparse = topic_graph(
        txts = txts,
        vectorizer = vct,
        model = mdl,
        **graph_cfg
    )

    pr = pagerank_power(
        A = sparse,
        **centrality_cfg
    )

    return nodes, sparse, pr

if __name__=='__main__':
    if not SKIP_SEQUENCE[0] or True:
        nlp, mdl = load_models()
        txts = preprocess(
            txts = get_bodies(WORKABLE_CSV_PATH),
            nlp = nlp
        )

    if not SKIP_SEQUENCE[1]:
        sentiments = run_sentiments(
            txts = txts, 
            mdl = mdl
        )
    
    if not SKIP_SEQUENCE[2]:
        topic, vct, top = run_lda(
            txts = txts, 
            cfg = LDA_CFG
        )

    if not SKIP_SEQUENCE[3] and not SKIP_SEQUENCE[2]:
        ns, spr, pr = run_topic_correlation(
            txts = txts,
            vct = vct,
            mdl = topic,
            graph_cfg = LDA_CFG,
            centrality_cfg = PAGERANK_CFG
        )
