from fastapi import FastAPI
from app.schemas import TranslateRequest, TranslateResponse
from app.config import DEVICE
from app.deps import get_defaults
from app.services.translator import translate_texts

app = FastAPI(title="PDF Translator API (EN->ES)", version="1.0")

@app.get("/health")
def health():
    defaults = get_defaults()
    return {
        "status": "ok",
        "device": DEVICE,
        "default_engine": defaults["engine"],
        "default_batch_size": defaults["batch_size"],
    }

@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    defaults = get_defaults()
    engine = req.engine or defaults["engine"]
    batch_size = req.batch_size or defaults["batch_size"]

    used_engine, translations = translate_texts(req.texts, engine=engine, batch_size=batch_size)
    return TranslateResponse(engine=used_engine, translations=translations)
