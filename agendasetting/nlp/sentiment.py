import torch as tch
import numpy as np
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from ..configs import TRANSFORMERS_SENTIMENT_MODEL, TORCH_CFG
from ..torchutils import torchesque

def load_emotions(model: str = TRANSFORMERS_SENTIMENT_MODEL, cfg: dict = TORCH_CFG):
    tok = AutoTokenizer.from_pretrained(model, use_fast = True)
    backend = torchesque(**cfg)

    mdl = AutoModelForSequenceClassification.from_pretrained(model, **backend['quantargs'])
    mdl.eval()

    if backend['device'].type == 'cuda' and not (cfg['bit8'] or cfg['bit4']):
        mdl.to(backend['device'], dtype = backend['dtype'])
    else:
        mdl.to(backend['device'])

    return {
        'tokenizer': tok,
        'model': mdl,
        'device': backend['device'],
        'id2label': mdl.config.id2label
    }

def standard_labels(rawscores):
    out = {}
    for s in rawscores:
        lbl = s['label'].lower()
        if      'neg' in lbl:   key = 'negative'
        elif    'neu' in lbl:   key = 'neutral'
        elif    'pos' in lbl:   key = 'positive'
        else:                   key = lbl
        out[key] = float(s['score'])
    return out

def softmax(logits):
    z = logits - logits.max(axis = 1, keepdims = True)
    e = np.exp(z)
    return e / e.sum(axis = 1, keepdims = True)

def predict_sentiment(txts, mdl, batch_size = 64, max_length = 256, truncation = True):
    tok = mdl['tokenizer']
    model = mdl['model']
    dev = mdl['device']
    id2label = mdl['id2label']
    preds = [None] * len(txts)

    with tch.inference_mode():
        tch.autocast
        for i in range(0, len(txts), batch_size):
            batch = txts[i:i + batch_size]
            enc = tok(batch, padding = True, truncation = truncation, max_length = max_length, return_tensors = "pt").to(dev)
            logits = model(**enc).logits
            probs = softmax(logits.detach().float().cpu().numpy())
            for j in range(probs.shape[0]):
                row = probs[j]
                raw = [
                    {"label": id2label[k], "score": float(row[k])} for k in range(row.shape[0])
                ]
                preds[i + j] = standard_labels(raw)

    avg = {}
    for r in preds:
        for k, v in r.items():
            avg[k] = avg.get(k, 0.0) + v
    if preds:
        for k in list(avg.keys()):
            avg[k] /= len(preds)

    return {
        "preds": preds, 
        "avg": avg
    }