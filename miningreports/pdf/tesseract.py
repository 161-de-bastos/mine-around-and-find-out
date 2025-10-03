import os
import re
import json
import tempfile
import pandas as pd
from rapidfuzz import fuzz, process
from pdf2image import convert_from_path
from PIL import Image, ImageOps, ImageFilter
import cv2
import numpy as np
import pytesseract

LANGS = "spa" 
DPI = 250          
OCR_PSM = "6"      # 6 = Assume a single uniform block of text
OCR_OEM = "1"      # 1 = LSTM only
PREPROC_DENOISE = True
PREPROC_BW = True
PREPROC_CONTRAST = 1.4 
PREPROC_SHARPEN = True
SECTION_CODE_RE = re.compile(r'\b([A-J]\d{1,2})\b', flags=re.IGNORECASE)

def enhance(img: Image.Image) -> Image.Image:
    if PREPROC_BW:
        img = ImageOps.grayscale(img)
    if PREPROC_CONTRAST and PREPROC_CONTRAST != 1.0:
        img = ImageOps.autocontrast(img)
    if PREPROC_DENOISE:
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR) if img.mode != "RGB" else np.array(img)
        arr = cv2.bilateralFilter(arr, 5, 30, 30)
        img = Image.fromarray(arr if img.mode == "RGB" else cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY))
    if PREPROC_SHARPEN:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=6))
    return img

def ocr_image(img):
    cfg = f'--oem {OCR_OEM} --psm {OCR_PSM}'
    text = pytesseract.image_to_string(img, lang=LANGS, config=cfg)
    text = text.replace('\x0c', ' ').strip()
    return text

def normalize(s):
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[“”"«»/\\|*_:;.,()\[\]-]+', ' ', s)
    return s

def read_targets(path):
    targets = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        targets.append(line)
    return targets

def best_match(page_text, targets_norm, thresh):
    hits = []
    if not page_text:
        return hits
    for t in targets_norm:
        score = fuzz.partial_ratio(t, page_text)
        if score >= thresh:
            hits.append((t, score))
    return hits