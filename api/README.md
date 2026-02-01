### Instalar
pip install -r requirements.txt

### Correr API
uvicorn app.main:app --host 0.0.0.0 --port 8000

### Probar
POST http://localhost:8000/translate
{
  "texts": ["Abstract: This paper proposes...", "1. Introduction: Fake news..."],
  "engine": "opus",
  "batch_size": 8
}
