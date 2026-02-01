from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app.config import (
    DEVICE, OPUS_MODEL, NLLB_MODEL, NLLB_SRC, NLLB_TGT,
    MAX_INPUT_TOKENS, MAX_NEW_TOKENS
)
from app.utils.chunking import split_into_chunks


@dataclass
class EnginePack:
    tokenizer: any
    model: any
    engine: str


@lru_cache(maxsize=2)
def load_engine(engine: str) -> EnginePack:
    engine = (engine or "opus").lower().strip()
    if engine not in ("opus", "nllb"):
        engine = "opus"

    if engine == "opus":
        tok = AutoTokenizer.from_pretrained(OPUS_MODEL)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(OPUS_MODEL)
    else:
        tok = AutoTokenizer.from_pretrained(NLLB_MODEL)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)

    mdl = mdl.to(DEVICE)
    mdl.eval()
    return EnginePack(tokenizer=tok, model=mdl, engine=engine)


@torch.inference_mode()
def _translate_batch_opus(pack: EnginePack, texts: List[str]) -> List[str]:
    tok, mdl = pack.tokenizer, pack.model
    inputs = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(DEVICE)

    out = mdl.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    return tok.batch_decode(out, skip_special_tokens=True)


@torch.inference_mode()
def _translate_batch_nllb(pack: EnginePack, texts: List[str]) -> List[str]:
    tok, mdl = pack.tokenizer, pack.model
    tok.src_lang = NLLB_SRC
    forced_id = tok.convert_tokens_to_ids(NLLB_TGT)

    inputs = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(DEVICE)

    out = mdl.generate(**inputs, forced_bos_token_id=forced_id, max_new_tokens=MAX_NEW_TOKENS)
    return tok.batch_decode(out, skip_special_tokens=True)


def translate_texts(texts: List[str], engine: str = "opus", batch_size: int = 8) -> Tuple[str, List[str]]:
    """
    Traduce una lista de textos. Maneja chunking y rearmado.
    """
    pack = load_engine(engine)

    # 1) Expandir a chunks
    mapping: List[Tuple[int, int]] = []
    flat_chunks: List[str] = []

    for i, t in enumerate(texts):
        chunks = split_into_chunks(t, max_chars=2500)
        mapping.append((i, len(chunks)))
        flat_chunks.extend(chunks)

    # 2) Traducir chunks en batches
    translated_chunks: List[str] = []
    if flat_chunks:
        for i in range(0, len(flat_chunks), batch_size):
            batch = flat_chunks[i:i + batch_size]
            if pack.engine == "nllb":
                translated_chunks.extend(_translate_batch_nllb(pack, batch))
            else:
                translated_chunks.extend(_translate_batch_opus(pack, batch))

    # 3) Rearmar por texto original
    result = [""] * len(texts)
    ptr = 0
    for (idx, n) in mapping:
        if n == 0:
            result[idx] = ""
        else:
            result[idx] = " ".join(translated_chunks[ptr:ptr + n]).strip()
            ptr += n

    return pack.engine, result
