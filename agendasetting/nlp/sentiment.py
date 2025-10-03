import torch as tch
import numpy as np
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from ..configs import TRANSFORMERS_SENTIMENT_MODEL, TORCH_CFG
from ..torchutils import torchesque

def load_emotions(model: str = TRANSFORMERS_SENTIMENT_MODEL, cfg: dict = TORCH_CFG):
    tok = AutoTokenizer.from_pretrained(model, use_fast = True)
    backend = torchesque(**TORCH_CFG)

    mdl = AutoModelForSequenceClassification.from_pretrained(model, **backend['quantargs'])
    mdl.eval()

    if backend['device'].type == 'cuda' and not (TORCH_CFG['bit8'] or TORCH_CFG['bit4']):
        mdl.to(backend['device'], dtype = backend['dtype'])
    else:
        mdl.to(backend['device'])

    return {
        'tokenizer': tok,
        'model': mdl,
        'device': backend['device'],
        'id2label': mdl.config.id2label
    }


