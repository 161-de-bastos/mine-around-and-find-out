import torch as tch
import numpy as np
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from ..configs import TRANSFORMERS_SENTIMENT_MODEL

tch.set_float32_matmul_precision("high")

def load_emotions(model: str = TRANSFORMERS_SENTIMENT_MODEL, device: int = -1): # -1 = CPU; >=0 CUDA
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForSequenceClassification.from_pretrained(model)
    pipe = TextClassificationPipeline(
        model = mdl, tokenizer = tok, device = device, return_all_score = True, truncation = True
    )
    return {'pipeline': pipe, 'id2label': mdl.config.id2label}


