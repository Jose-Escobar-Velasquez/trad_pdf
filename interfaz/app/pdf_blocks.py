import re
import statistics
import fitz  # PyMuPDF

from app.text_utils import normalize_text, looks_like_header_footer, reflow_lines, de_spaced_word


_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")

def _norm_font_name(name: str) -> str:
    name = (name or "").strip()
    name = _SUBSET_PREFIX.sub("", name)  # quita ABCDEF+
    return name.lower()

def _int_to_rgb01(c: int):
    # c suele venir como 0xRRGGBB
    r = ((c >> 16) & 255) / 255.0
    g = ((c >> 8) & 255) / 255.0
    b = (c & 255) / 255.0
    return (r, g, b)

def find_references_cutoff(doc: fitz.Document, include_headers: bool = False):
    """
    Busca la primera aparición de 'References' y devuelve (page_index, y0).
    Todo lo que esté en esa página con y >= y0, y páginas posteriores, se omite.
    """
    for pno in range(doc.page_count):
        page = doc[pno]
        H = float(page.rect.height)
        d = page.get_text("dict")

        for block in d.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                raw = "".join(s.get("text", "") for s in spans)
                text = normalize_text(de_spaced_word(raw))
                if not text:
                    continue

                y0 = min(s["bbox"][1] for s in spans)
                if (not include_headers) and looks_like_header_footer(text, y0, H):
                    continue

                if text.strip().lower() == "references":
                    return (pno, float(y0))

    return None


def extract_text_blocks(
    doc: fitz.Document,
    include_headers: bool = False,
    min_chars: int = 8,
    skip_after_references: bool = True,
):
    """
    Extrae bloques de texto del PDF y devuelve:
      {page, bbox, text, fontsize, fontname, font_xref, color}
    - font_xref: xref del font embebido si se puede resolver (para mantener tipografía)
    - color: RGB 0..1 desde spans
    - skip_after_references: si True, no extrae nada desde References en adelante
    """
    cutoff = find_references_cutoff(doc, include_headers=include_headers) if skip_after_references else None

    blocks_out = []

    for pno in range(doc.page_count):
        page = doc[pno]
        H = float(page.rect.height)

        # map de fuentes de la página: nombre_normalizado -> xref
        font_entries = page.get_fonts(full=True)  # (xref, ..., type, name, refname, enc, ...)
        font_map = {}
        for entry in font_entries:
            xref = entry[0]
            name = entry[3]  # ejemplo: 'Helvetica' o 'ABCDEE+TimesNewRomanPSMT'
            font_map[_norm_font_name(name)] = xref

        d = page.get_text("dict")

        for block in d.get("blocks", []):
            if block.get("type", 0) != 0:
                continue

            bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
            b_y0 = float(bbox[1])

            # corta después de References
            if cutoff is not None:
                c_page, c_y = cutoff
                if pno > c_page:
                    continue
                if pno == c_page and b_y0 >= c_y:
                    continue

            lines = []
            sizes = []
            font_hits = {}
            colors = {}
            bold_hits = 0
            span_count = 0

            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                raw = "".join(s.get("text", "") for s in spans)
                line_text = normalize_text(raw)
                if not line_text:
                    continue

                y0 = min(s["bbox"][1] for s in spans)

                # filtro header/footer
                if (not include_headers) and looks_like_header_footer(line_text, y0, H):
                    continue

                lines.append(line_text)

                for s in spans:
                    txt = (s.get("text") or "").strip()
                    if not txt:
                        continue

                    span_count += 1
                    sizes.append(float(s.get("size", 10.0)))

                    f = s.get("font") or ""
                    fn = _norm_font_name(f)
                    font_hits[fn] = font_hits.get(fn, 0) + 1

                    color_int = int(s.get("color", 0))
                    colors[color_int] = colors.get(color_int, 0) + 1

                    font_low = (f or "").lower()
                    if "bold" in font_low:
                        bold_hits += 1

            text = reflow_lines(lines)
            if len(text) < int(min_chars):
                continue

            # evita bloques de solo números
            if text.strip().isdigit() and len(text.strip()) <= 4:
                continue

            fontsize = statistics.median(sizes) if sizes else 10.0
            is_bold = (bold_hits / span_count) > 0.5 if span_count else False

            # fuente dominante
            dom_font = None
            if font_hits:
                dom_font = max(font_hits.items(), key=lambda kv: kv[1])[0]  # ya normalizada

            # xref de fuente (si existe)
            font_xref = font_map.get(dom_font) if dom_font else None

            # fallback fontname (solo por si no logramos embebida)
            fontname = "helvB" if is_bold else "helv"

            # color dominante
            dom_color_int = max(colors.items(), key=lambda kv: kv[1])[0] if colors else 0
            color = _int_to_rgb01(dom_color_int)

            blocks_out.append(
                {
                    "page": pno,
                    "bbox": bbox,
                    "text": text,
                    "fontsize": float(fontsize),
                    "fontname": fontname,
                    "font_xref": font_xref,   # <- clave para mantener tipografía real
                    "color": color,
                }
            )

    return blocks_out
