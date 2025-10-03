import numpy as np
from spacy.attrs import IS_SPACE, IS_PUNCT, LIKE_NUM, IS_STOP, POS, LENGTH
import spacy
from unidecode import unidecode

from ..configs import PREPROCESS_CFG, SPACY_ESP_MODEL

def load_spacy(model: str = SPACY_ESP_MODEL):
    return spacy.load(model)

def normalize (txt: str, cfg: dict):
    s = txt.lower() if cfg.get('lowercase', True) else txt
    return unidecode(s) if cfg.get('strip_accents', False) else s

def token_pos(doc, cfg):
    arr = doc.to_array([IS_SPACE, IS_PUNCT, LIKE_NUM, IS_STOP, POS, LENGTH])
    c_space, c_punct, c_num, c_stop, c_pos, c_len = 0, 1, 2, 3, 4, 5

    mask = (arr[:, c_space] == 0) & (arr[:, c_punct] == 0) & (arr[:, c_num] == 0)
    if cfg.get("remove_stop", True):
        mask &= (arr[:, c_stop] == 0)
    min_len = int(cfg.get("min_len", 2))
    if min_len > 1:
        mask &= (arr[:, c_len] >= min_len)

    keep = cfg.get("keep_pos", ())
    if keep:
        pos_ids = {doc.vocab.strings[p] for p in keep}
        mask &= np.isin(arr[:, c_pos], np.fromiter(pos_ids, dtype=arr.dtype))

    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    
    strings = doc.vocab.strings
    lemmatize = cfg.get("lemmatize", True)

    out = []
    for i in idx.tolist():
        tok = doc[i]
        pos_str = strings[arr[i, c_pos]]
        w = tok.lemma_ if (lemmatize and tok.lemma_) else tok.text
        out.append((w, pos_str))
    return out

def preprocess_docs(txts, nlp, cfg, n_process = 1, batch_size = 512):
    normies = [normalize(t, cfg) for t in txts]
    docs_tokens = []

    for doc in nlp.pipe(normies, n_process = n_process, batch_size = batch_size):
        docs_tokens.append(token_pos(doc, cfg))

    return docs_tokens