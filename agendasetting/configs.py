
PREPROCESS_CFG = {
    'lowercase': True,                  
    'strip_accents': False,             
    'keep_pos': ('NOUN', 'PROPN', 'VERB', 'ADJ'),
    'remove_stop': True,                
    'min_len': 2,                       
    'lemmatize': True,                  
}

SPACY_ESP_MODEL = 'es_core_news_md'
TRANSFORMERS_SENTIMENT_MODEL = 'bardsai/finance-sentiment-es-base'