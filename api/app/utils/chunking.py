import re
from typing import List

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

def split_into_chunks(text: str, max_chars: int = 2500) -> List[str]:
    """
    Divide texto largo en chunks por oraciones (y fallback por tamaño).
    max_chars evita meter párrafos gigantes al modelo.
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

    # Corte bruto si aún queda demasiado grande
    final = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars):
                final.append(c[i:i + max_chars])
    return final
