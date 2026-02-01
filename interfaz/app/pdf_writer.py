import fitz  # PyMuPDF

LINEHEIGHT = 1.20


def _wrap_count_lines(text: str, max_width: float, measure_fontname: str, fontsize: float) -> int:
    """
    Cuenta líneas aproximadas con word-wrap usando una fuente SOPORTADA (Base14).
    measure_fontname debe ser 'helv', 'times-roman', 'cour', etc.
    """
    if not text:
        return 1

    total_lines = 0
    for para in text.split("\n"):
        words = para.strip().split()
        if not words:
            total_lines += 1
            continue

        cur = words[0]
        for w in words[1:]:
            test = cur + " " + w
            if fitz.get_text_length(test, fontname=measure_fontname, fontsize=fontsize) <= max_width:
                cur = test
            else:
                total_lines += 1
                cur = w

        total_lines += 1

    return max(total_lines, 1)


def _fit_fontsize_to_rect(
    text: str,
    rect: fitz.Rect,
    measure_fontname: str,   # <- OJO: fuente para medir (Base14)
    target_size: float,
    min_size: float = 5.0
):
    fs = max(float(target_size), min_size)
    width = max(rect.width - 2, 10)
    height = max(rect.height - 2, 10)

    for _ in range(4):
        n_lines = _wrap_count_lines(text, width, measure_fontname, fs)
        needed_h = n_lines * fs * LINEHEIGHT
        if needed_h <= height:
            return fs
        ratio = height / max(needed_h, 1e-6)
        fs = max(min_size, fs * ratio)

    return fs

def _paint_white_rect(page: fitz.Page, rect: fitz.Rect):
    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(color=None, fill=(1, 1, 1), width=0)
    shape.commit()


def _base14_fallback(font_real_name: str) -> str:
    """
    Si no se puede extraer la fuente embebida, mapea por nombre a una base14.
    """
    n = (font_real_name or "").lower()
    if "times" in n:
        return "times-roman"
    if "courier" in n:
        return "cour"
    if "helvetica" in n or "arial" in n:
        return "helv"
    return "helv"


def _ensure_font_for_block(doc: fitz.Document, page: fitz.Page, block: dict, font_cache: dict):
    """
    Intenta usar la fuente REAL del PDF usando font_xref.
    - Si el font es embebido, extrae buffer y lo inserta con alias 'F{xref}'.
    - Si no hay buffer (base14), usa fallback base14.
    Retorna fontname a usar en insert_textbox.
    """
    xref = block.get("font_xref")
    if not xref:
        return block.get("fontname", "helv")

    if xref in font_cache:
        alias, buf, realname = font_cache[xref]
    else:
        # extract_font devuelve tuple: (name, ext, type, buffer)
        realname, ext, ftype, buf = doc.extract_font(int(xref))
        alias = f"F{xref}"
        font_cache[xref] = (alias, buf, realname)

    # Si hay buffer -> es font embebido: insertarlo
    if buf:
        try:
            page.insert_font(fontname=alias, fontbuffer=buf)
            return alias
        except Exception:
            # fallback si algo falla
            return block.get("fontname", "helv")
    else:
        # no hay buffer => base14, usa mapping por nombre real
        return _base14_fallback(realname)


def write_translated_pdf(
    pdf_bytes: bytes,
    blocks: list[dict],
    translations: list[str],
    remove_original_text: bool = False,
    keep_font_size: bool = True,   # mantener tamaño exacto
    fit_if_needed: bool = True,    # si no cabe, achica lo mínimo
):
    """
    Devuelve bytes del PDF traducido manteniendo layout.
    - NO mueve imágenes/gráficos (solo toca texto).
    - Si remove_original_text=True: usa redaction (borra texto en el área).
    - keep_font_size: intenta usar el fontsize original.
      fit_if_needed: si no cabe, reduce fontsize lo mínimo.
    """
    if len(blocks) != len(translations):
        raise ValueError("blocks y translations deben tener el mismo largo")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # agrupar por página
    by_page = {}
    for i, b in enumerate(blocks):
        by_page.setdefault(b["page"], []).append((b, translations[i]))

    font_cache = {}  # xref -> (alias, buffer, realname)

    # helper: elegir una fuente Base14 para MEDIR (siempre soportada por get_text_length)
    def _measure_font_for_block(b: dict) -> str:
        # Si tú guardas algo más fiel (ej. "times-roman") úsalo acá.
        # Por ahora mapeamos a base14 según heurística.
        fn = (b.get("fontname") or "helv").lower()

        # tus valores actuales suelen ser helv / helvB, pero por si acaso:
        if "times" in fn:
            return "times-roman"
        if "cour" in fn:
            return "cour"
        if "helv" in fn or "arial" in fn or "vetica" in fn:
            return "helv"

        # fallback seguro
        return "helv"

    for pno, items in by_page.items():
        page = doc[pno]

        # 1) Redactions opcional
        if remove_original_text:
            for b, _tr in items:
                rect = fitz.Rect(b["bbox"])
                page.add_redact_annot(rect, fill=(1, 1, 1))
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        # 2) Tapar + insertar traducción
        for b, tr in items:
            rect = fitz.Rect(b["bbox"])

            if not remove_original_text:
                _paint_white_rect(page, rect)

            # Fuente para ESCRIBIR (embebida si se puede)
            write_fontname = _ensure_font_for_block(doc, page, b, font_cache)

            # Fuente para MEDIR (Base14 soportada)
            measure_fontname = _measure_font_for_block(b)

            base_fs = float(b.get("fontsize", 10.0))
            color = b.get("color", (0, 0, 0))

            # Tamaño: mantener o ajustar
            if keep_font_size:
                fs = base_fs
                if fit_if_needed:
                    fs_fit = _fit_fontsize_to_rect(tr, rect, measure_fontname, base_fs, min_size=5.0)
                    fs = min(base_fs, fs_fit)
            else:
                fs = _fit_fontsize_to_rect(tr, rect, measure_fontname, base_fs, min_size=5.0)

            page.insert_textbox(
                rect,
                tr,
                fontsize=fs,
                fontname=write_fontname,  # <-- aquí va la embebida si existe
                color=color,
                align=fitz.TEXT_ALIGN_LEFT,
            )

    out = doc.write()
    doc.close()
    return out
