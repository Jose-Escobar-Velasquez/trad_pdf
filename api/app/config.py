import os
import torch

# Motor por defecto: "opus" (ligero) o "nllb" (más pesado)
DEFAULT_ENGINE = os.getenv("TRANSLATION_ENGINE", "opus").lower().strip()

# CPU / GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Modelos gratis
OPUS_MODEL = os.getenv("OPUS_MODEL", "Helsinki-NLP/opus-mt-en-es")
NLLB_MODEL = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")

# NLLB language codes
NLLB_SRC = os.getenv("NLLB_SRC", "eng_Latn")
NLLB_TGT = os.getenv("NLLB_TGT", "spa_Latn")

# Defaults de batching
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "512"))  # truncation input
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))      # largo salida
