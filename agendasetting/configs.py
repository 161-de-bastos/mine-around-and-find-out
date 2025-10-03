PREPROCESS_CFG = {
    'lowercase': True,                  
    'strip_accents': False,             
    'keep_pos': ('NOUN', 'PROPN', 'VERB', 'ADJ'),
    'remove_stop': True,                
    'min_len': 2,                       
    'lemmatize': True,
    'n_process': 1,
    'batch_size':512                  
}

SPACY_ESP_MODEL = 'es_dep_news_trf'
TRANSFORMERS_SENTIMENT_MODEL = 'bardsai/finance-sentiment-es-base'

TORCH_CFG = {
    'force_cpu': False,
    'dtype': 'auto',
    'bit8': False,
    'bit4': False
}

WORKABLE_CSV_PATH = 'data/retrieval_v1.2.csv'

SENTIMENT_INFERENCE = {
    'batch_size': 64,
    'max_length': 256,
    'truncation': True
}

LDA_CFG = {
    'n_topics': 10, 
    'max_features': 6000, 
    'random_state': 42
}

TOPIC_CFG = {
    'tau': 0.0,
    'normalize_rows': True
}

PAGERANK_CFG = {
    'alpha': 0.85, 
    'tol': 1e-6, 
    'max_iter': 100
}