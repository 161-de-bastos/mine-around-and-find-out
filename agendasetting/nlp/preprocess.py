from unidecode import unidecode
import spacy

from ..configs import PREPROCESS_CFG, SPACY_ESP_MODEL

def load_spacy(model: str = SPACY_ESP_MODEL):
    return spacy.load(model)

def normalize (txt: str, cfg: dict):
    s = txt.lower() if cfg.get('lowercase', True) else txt
    return unidecode(s) if cfg.get('strip_accents', False) else s
