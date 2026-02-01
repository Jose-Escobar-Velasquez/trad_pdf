import os
import re
from typing import List, Literal, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
# -----------------------------
# Esta es la rama de la API de traducción
# -----------------------------

app = FastAPI(title="PDF Translator API (EN->ES)", version="1.0")
"""
con este comando ejecutamos la api

$env:TORCH_NUM_THREADS="2"
>> $env:TORCH_NUM_INTEROP_THREADS="1"
>> $env:TOKENIZERS_PARALLELISM="false"
>> uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
# -----------------------------
# Config
# -----------------------------
DEFAULT_ENGINE = os.getenv("TRANSLATION_ENGINE", "opus")  # opus | nllb
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Modelos (descarga 1a vez, luego queda en cache local)
OPUS_MODEL = "Helsinki-NLP/opus-mt-en-es"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

# NLLB language codes
NLLB_SRC = "eng_Latn"
NLLB_TGT = "spa_Latn"

# Carga global (lazy) para no cargar ambos si no los usas
_tokenizer = {}
_model = {}


# -----------------------------
# Utils: chunking para textos largos
# -----------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

def split_into_chunks(text: str, max_chars: int = 2500) -> List[str]:
    """
    Chunking simple por oraciones (y fallback por tamaño).
    max_chars es heurístico: evita entradas gigantes.
    """
    text = (text or "").strip()
    if not text:
        return []

    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    chunks = []
    cur = ""

    for p in parts:
        if not cur:
            cur = p
            continue

        if len(cur) + 1 + len(p) <= max_chars:
            cur += " " + p
        else:
            chunks.append(cur)
            cur = p

    if cur:
        chunks.append(cur)

    # Si quedó algo todavía demasiado grande (párrafos enormes), corta por bruto
    final = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars):
                final.append(c[i:i+max_chars])
    return final


# -----------------------------
# Loaders
# -----------------------------
def get_engine(engine: str):
    engine = engine.lower().strip()
    if engine not in ("opus", "nllb"):
        engine = "opus"
    if engine in _model and engine in _tokenizer:
        return _tokenizer[engine], _model[engine], engine

    if engine == "opus":
        tok = AutoTokenizer.from_pretrained(OPUS_MODEL)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(OPUS_MODEL)
    else:
        tok = AutoTokenizer.from_pretrained(NLLB_MODEL)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)

    mdl = mdl.to(DEVICE)
    mdl.eval()

    _tokenizer[engine] = tok
    _model[engine] = mdl
    return tok, mdl, engine


# -----------------------------
# Translation core
# -----------------------------
@torch.inference_mode()
def translate_batch_opus(tokenizer, model, texts: List[str], max_new_tokens=512) -> List[str]:
    # MarianMT no necesita forced BOS
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.batch_decode(out, skip_special_tokens=True)


@torch.inference_mode()
def translate_batch_nllb(tokenizer, model, texts: List[str], max_new_tokens=512) -> List[str]:
    # NLLB requiere src_lang + forced_bos_token_id del target
    tokenizer.src_lang = NLLB_SRC
    forced_id = tokenizer.convert_tokens_to_ids(NLLB_TGT)

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    out = model.generate(**inputs, forced_bos_token_id=forced_id, max_new_tokens=max_new_tokens)
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def translate_texts(texts: List[str], engine: str, batch_size: int = 8) -> List[str]:
    tokenizer, model, engine = get_engine(engine)

    # Expand chunks
    # guardamos un mapa para reconstruir cada texto original
    mapping = []
    flat_chunks = []
    for i, t in enumerate(texts):
        chunks = split_into_chunks(t, max_chars=2500)
        if not chunks:
            mapping.append((i, 0))
            continue
        mapping.append((i, len(chunks)))
        flat_chunks.extend(chunks)

    # Traducir chunks en batches
    translated_chunks = []
    if flat_chunks:
        for i in range(0, len(flat_chunks), batch_size):
            batch = flat_chunks[i:i+batch_size]
            if engine == "nllb":
                translated_chunks.extend(translate_batch_nllb(tokenizer, model, batch))
            else:
                translated_chunks.extend(translate_batch_opus(tokenizer, model, batch))

    # Re-armar por texto original
    result = [""] * len(texts)
    ptr = 0
    for (idx, n) in mapping:
        if n == 0:
            result[idx] = ""
        else:
            result[idx] = " ".join(translated_chunks[ptr:ptr+n]).strip()
            ptr += n

    return result


# -----------------------------
# API Schemas
# -----------------------------
class TranslateRequest(BaseModel):
    texts: List[str] = Field(..., description="Lista de textos (ej. secciones del PDF)")
    engine: Optional[Literal["opus", "nllb"]] = Field(default=None, description="Motor de traducción")
    batch_size: int = Field(default=8, ge=1, le=64)
    # si quieres ajustar lo “largo” que deja el modelo
    # max_new_tokens: int = Field(default=512, ge=16, le=1024)


class TranslateResponse(BaseModel):
    engine: str
    translations: List[str]


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    engine = req.engine or DEFAULT_ENGINE
    translations = translate_texts(req.texts, engine=engine, batch_size=req.batch_size)
    return TranslateResponse(engine=engine, translations=translations)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "default_engine": DEFAULT_ENGINE}
