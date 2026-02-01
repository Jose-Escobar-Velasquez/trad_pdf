import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import fitz  # PyMuPDF

from app.pdf_blocks import extract_text_blocks
from app.translate_client import translate_texts
from app.pdf_writer import write_translated_pdf


st.set_page_config(page_title="PDF Translator (EN→ES)", page_icon="📄")

st.title("📄 PDF Translator (EN → ES)")
st.write("Traduce manteniendo el PDF tal cual (layout, columnas, imágenes). Se reemplaza texto por traducción en la misma zona.")

with st.sidebar:
    st.subheader("API de traducción")
    api_url = st.text_input("Base URL", value="http://127.0.0.1:8000")
    engine = st.selectbox("Engine", ["opus", "nllb"], index=0)
    batch_size = st.number_input("batch_size (API)", min_value=1, max_value=64, value=2, step=1)
    timeout_sec = st.number_input("timeout (seg)", min_value=30, max_value=72000, value=7200, step=60)

    st.subheader("Extracción")
    include_headers = st.checkbox("Incluir headers/footers", value=False)
    min_chars = st.number_input("Min chars por bloque", min_value=1, max_value=300, value=12, step=1)

    skip_after_refs = st.checkbox("Omitir desde 'References' hacia adelante", value=True)

    st.subheader("Salida")
    remove_original_text = st.checkbox("Eliminar texto original (redaction)", value=False)

    keep_font_size = st.checkbox("Mantener tamaño de letra exacto", value=True)
    fit_if_needed = st.checkbox("Si no cabe, achicar lo mínimo", value=True)

uploaded = st.file_uploader("Arrastra tu PDF aquí", type=["pdf"])

if uploaded:
    st.success(f"PDF cargado: {uploaded.name}")

    if st.button("Traducir PDF (manteniendo layout)"):
        pdf_bytes = uploaded.read()

        with st.spinner("Extrayendo bloques de texto (con fuente/tamaño) ..."):
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            blocks = extract_text_blocks(
                doc,
                include_headers=include_headers,
                min_chars=int(min_chars),
                skip_after_references=bool(skip_after_refs),
            )
            doc.close()

        if not blocks:
            st.error("No se encontró texto seleccionable en el PDF (puede ser escaneado).")
            st.stop()

        st.info(f"Bloques a traducir: {len(blocks)}")

        texts = [b["text"] for b in blocks]

        with st.spinner("Traduciendo (API local)..."):
            translations = translate_texts(
                api_base_url=api_url,
                texts=texts,
                engine=engine,
                batch_size=int(batch_size),
                timeout_sec=int(timeout_sec),
            )

        with st.spinner("Generando PDF final (layout intacto)..."):
            out_pdf = write_translated_pdf(
                pdf_bytes=pdf_bytes,
                blocks=blocks,
                translations=translations,
                remove_original_text=remove_original_text,
                keep_font_size=bool(keep_font_size),
                fit_if_needed=bool(fit_if_needed),
            )

        st.download_button(
            "⬇️ Descargar PDF traducido",
            data=out_pdf,
            file_name=uploaded.name.replace(".pdf", "") + "_es_layout.pdf",
            mime="application/pdf",
        )
