import fitz  # PyMuPDF
from app.text_utils import normalize_text, looks_like_header_footer


def extract_full_text(
    doc: fitz.Document,
    include_headers: bool = False,
    two_columns: bool = True,
) -> str:
    """
    Extrae TODO el texto del PDF en orden de lectura.
    Para papers: 2 columnas (izq -> der).
    Retorna 1 solo string.
    """
    pages_text = []

    for pno in range(doc.page_count):
        page = doc[pno]
        W = float(page.rect.width)
        H = float(page.rect.height)
        mid = W / 2.0

        items = []  # (x0, y0, text)
        d = page.get_text("dict")

        for block in d.get("blocks", []):
            if block.get("type", 0) != 0:
                continue

            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                text = normalize_text("".join(s.get("text", "") for s in spans))
                if not text:
                    continue

                y0 = min(s["bbox"][1] for s in spans)
                x0 = min(s["bbox"][0] for s in spans)

                if (not include_headers) and looks_like_header_footer(text, y0, H):
                    continue

                items.append((x0, y0, text))

        if not items:
            pages_text.append("")
            continue

        if two_columns:
            left = [t for t in items if t[0] < mid]
            right = [t for t in items if t[0] >= mid]
            left.sort(key=lambda t: t[1])
            right.sort(key=lambda t: t[1])
            ordered = left + right
        else:
            ordered = sorted(items, key=lambda t: t[1])

        page_text = "\n".join(t[2] for t in ordered)
        pages_text.append(page_text)

    # Separador simple entre páginas (no lo usamos para reconstruir layout, solo para lectura)
    return "\n\n".join(pages_text).strip()
