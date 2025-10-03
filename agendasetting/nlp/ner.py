import numpy as np
from spacy.attrs import DEP

def actors(txts, nlp):
    doc = nlp('\n\n'.join(txts))
    entities = list(dict.fromkeys([
        (e.text, e.label) for e in doc.ents
    ]))