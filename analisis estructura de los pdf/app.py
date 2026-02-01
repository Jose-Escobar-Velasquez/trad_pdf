import re
import streamlit as st
import fitz  # PyMuPDF
import numpy as np


# -------------------- Texto helpers --------------------
def normalize_text(s: str) -> str:
    s = s.replace("\ufb01", "fi").replace("\ufb02", "fl").replace("\u00ad", "")
    s = s.replace("￾", "").replace("�", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def de_spaced_word(s: str) -> str:
    # "a b s t r a c t" -> "abstract"
    s = s.strip()
    if re.fullmatch(r"(?:[A-Za-z]\s+){3,}[A-Za-z]", s):
        return s.replace(" ", "")
    return s


def looks_like_header_footer(text: str, y: float, page_h: float) -> bool:
    low = text.lower()
    # cabeceras típicas / metadata arriba
    if y < 70 and ("fundamental research" in low or "sciencedirect" in low or "journal homepage" in low):
        return True
    # número de página abajo
    if y > page_h - 50 and re.fullmatch(r"\d{1,4}", text.strip()):
        return True
    return False


# -------------------- Extracción de líneas (orden lectura 2 columnas) --------------------
def extract_ordered_lines(doc: fitz.Document):
    """
    Devuelve lista ordenada por lectura (izq -> der) con:
    {page, x, y, text, bbox}
    bbox corresponde a la línea completa (unión de spans).
    """
    all_lines = []

    for pno in range(doc.page_count):
        page = doc[pno]
        W = float(page.rect.width)
        H = float(page.rect.height)
        mid = W / 2.0

        raw = []
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

                if looks_like_header_footer(text, y0, H):
                    continue

                x1 = max(s["bbox"][2] for s in spans)
                y1 = max(s["bbox"][3] for s in spans)
                bbox = (x0, y0, x1, y1)

                raw.append((x0, y0, text, bbox))

        left = [t for t in raw if t[0] < mid]
        right = [t for t in raw if t[0] >= mid]
        left.sort(key=lambda t: t[1])
        right.sort(key=lambda t: t[1])

        for x0, y0, text, bbox in (left + right):
            all_lines.append({"page": pno, "x": x0, "y": y0, "text": text, "bbox": bbox})

    return all_lines


# -------------------- Detección de headings --------------------
def detect_headings(lines):
    """
    Detecta:
    - Abstract / a b s t r a c t
    - Keywords
    - Secciones numeradas: 1. Title / 3.2.1. Title
    - References (se toma la última)
    """
    headings = []
    sec_re = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(.+)$")

    for i, ln in enumerate(lines):
        raw = ln["text"]
        t = normalize_text(de_spaced_word(raw))
        low = t.lower()

        if low == "abstract":
            headings.append({"idx": i, "key": "Abstract", "kind": "abstract", "level": 0})
            continue
        if low.startswith("keywords"):
            headings.append({"idx": i, "key": "Keywords", "kind": "keywords", "level": 0})
            continue
        if low == "references":
            headings.append({"idx": i, "key": "References", "kind": "references", "level": 0})
            continue

        m = sec_re.match(t)
        if m:
            sec_id = m.group(1)
            title = m.group(2).strip()
            level = sec_id.count(".") + 1
            headings.append({"idx": i, "key": f"{sec_id}. {title}", "kind": "section", "level": level})

    headings.sort(key=lambda h: h["idx"])

    # Dedup simple
    cleaned = []
    for h in headings:
        if cleaned and h["key"] == cleaned[-1]["key"] and abs(h["idx"] - cleaned[-1]["idx"]) < 5:
            continue
        cleaned.append(h)

    # References: quédate con la última
    refs = [h for h in cleaned if h["kind"] == "references"]
    if len(refs) > 1:
        last_ref = refs[-1]
        cleaned = [h for h in cleaned if h["kind"] != "references"] + [last_ref]
        cleaned.sort(key=lambda h: h["idx"])

    return cleaned


def select_boundaries(headings, mode="top", include_references=False, include_keywords=True):
    """
    mode:
      - top: Abstract/Keywords + nivel1 (1.,2.,3.,...) (+ References si include_references)
      - all: todas las secciones numeradas y especiales
    """
    if mode == "all":
        boundary = headings[:]
        if not include_references:
            boundary = [h for h in boundary if h["kind"] != "references"]
        if not include_keywords:
            boundary = [h for h in boundary if h["kind"] != "keywords"]
        return boundary

    boundary = []
    # abstract si existe
    ab = [h for h in headings if h["kind"] == "abstract"]
    if ab:
        boundary.append(ab[0])

    # keywords si se pide
    if include_keywords:
        kw = [h for h in headings if h["kind"] == "keywords"]
        if kw:
            boundary.append(kw[0])

    # secciones principales nivel 1
    boundary += [h for h in headings if h.get("level") == 1]

    # references (última) si se pide
    if include_references:
        rf = [h for h in headings if h["kind"] == "references"]
        if rf:
            boundary.append(rf[-1])

    # uniq + sort por idx
    seen = set()
    uniq = []
    for h in boundary:
        if h["idx"] in seen:
            continue
        seen.add(h["idx"])
        uniq.append(h)

    return sorted(uniq, key=lambda h: h["idx"])


# -------------------- Marcado (subrayado por sección) --------------------
def section_colors():
    # (RGB 0..1) - distintos por sección
    return [
        (0.12, 0.47, 0.71),
        (1.00, 0.50, 0.05),
        (0.17, 0.63, 0.17),
        (0.84, 0.15, 0.16),
        (0.58, 0.40, 0.74),
        (0.55, 0.34, 0.29),
        (0.89, 0.47, 0.76),
        (0.50, 0.50, 0.50),
        (0.74, 0.74, 0.13),
        (0.09, 0.75, 0.81),
    ]


def underline_pdf_by_sections(
    pdf_bytes: bytes,
    mode="top",
    include_references=False,
    include_keywords=True,
    underline_headings=False,
    opacity=0.85,
):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    lines = extract_ordered_lines(doc)
    if not lines:
        out = doc.write()
        doc.close()
        return out, {"sections": [], "note": "No se encontró texto seleccionable en el PDF."}

    headings = detect_headings(lines)
    boundaries = select_boundaries(
        headings, mode=mode, include_references=include_references, include_keywords=include_keywords
    )

    if not boundaries:
        out = doc.write()
        doc.close()
        return out, {"sections": [], "note": "No se detectaron secciones (headings) en el PDF."}

    # Construir secciones por rangos de índices de líneas
    sections = []
    for j, h in enumerate(boundaries):
        start = h["idx"] if underline_headings else h["idx"] + 1
        end = boundaries[j + 1]["idx"] if j + 1 < len(boundaries) else len(lines)
        if start >= end:
            continue
        sections.append({"title": h["key"], "start": start, "end": end})

    colors = section_colors()

    # page -> sec_idx -> [quads]
    page_rects = {}

    def add_rect(pno, sec_i, rect):
        r = fitz.Rect(rect)
        # filtra rects degenerados (por si acaso)
        if r.is_empty or r.get_area() <= 0:
            return
        page_rects.setdefault(pno, {}).setdefault(sec_i, []).append(r)

    # Asignar cada línea al rango de una sección
    # (esto subraya el contenido “real” por sección)
    for sec_i, sec in enumerate(sections):
        for k in range(sec["start"], sec["end"]):
            ln = lines[k]
            # (Opcional) filtrar líneas vacías / muy cortas
            if not ln["text"] or len(ln["text"].strip()) == 0:
                continue
            add_rect(ln["page"], sec_i, ln["bbox"])


    # Crear 1 anotación por página por sección (mucho más liviano que 1 por línea)
    for pno, sec_map in page_rects.items():
        page = doc[pno]
        for sec_i, rects in sec_map.items():
            if not rects:
                continue
            annot = page.add_highlight_annot(rects)
            annot.set_colors(stroke=colors[sec_i % len(colors)])
            annot.set_opacity(opacity)   # prueba 0.25 a 0.45 para que no opaque el texto
            annot.update()

    out_bytes = doc.write()
    doc.close()

    return out_bytes, {
        "sections": [s["title"] for s in sections],
        "headings_found": len(headings),
        "boundaries_used": len(boundaries),
    }


# -------------------- UI (Streamlit) --------------------
st.set_page_config(page_title="Subrayar secciones en PDF", page_icon="📄")
st.title("📄 Subrayador de secciones (Abstract, Introduction, etc.)")
st.write(
    "Sube/arrastra un PDF, y descarga el mismo PDF con **el contenido subrayado por sección** "
    "(cada sección con un color distinto)."
)

uploaded = st.file_uploader("PDF", type=["pdf"])

col1, col2, col3 = st.columns(3)
with col1:
    mode_label = st.selectbox("Nivel de secciones", ["Principales (1., 2., 3.)", "Todas (incluye 3.1, 3.2.1, ...)"])
with col2:
    include_keywords = st.checkbox("Incluir Keywords", value=True)
with col3:
    include_references = st.checkbox("Incluir References", value=False)

underline_headings = st.checkbox("Subrayar también los títulos (headings)", value=False)

if uploaded is not None:
    st.success(f"Archivo cargado: {uploaded.name}")

    if st.button("Analizar y subrayar"):
        mode = "top" if mode_label.startswith("Principales") else "all"

        with st.spinner("Detectando secciones y subrayando contenido..."):
            out_bytes, meta = underline_pdf_by_sections(
                uploaded.read(),
                mode=mode,
                include_references=include_references,
                include_keywords=include_keywords,
                underline_headings=underline_headings,
                opacity=0.85,
            )

        if "note" in meta:
            st.warning(meta["note"])
        else:
            st.info(f"Headings detectados: {meta.get('headings_found', 0)} | Secciones usadas: {len(meta.get('sections', []))}")
            with st.expander("Ver secciones detectadas"):
                for s in meta.get("sections", []):
                    st.write("-", s)

        st.download_button(
            label="⬇️ Descargar PDF subrayado",
            data=out_bytes,
            file_name=uploaded.name.replace(".pdf", "") + "_secciones_subrayadas.pdf",
            mime="application/pdf",
        )
