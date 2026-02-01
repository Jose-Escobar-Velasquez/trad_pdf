## Requisitos
1) Tener tu API local corriendo:
   uvicorn app.main:app --host 0.0.0.0 --port 8000

2) Correr la app:
   pip install -r requirements.txt
   streamlit run app/main.py

## Notas
- Funciona con PDFs con texto seleccionable.
- La traducción se escribe sobre el PDF (tapando el inglés por bloque).
- Si activas "Eliminar texto original", se aplican redactions.
