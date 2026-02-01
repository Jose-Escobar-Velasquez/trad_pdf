from app.config import DEFAULT_ENGINE, DEFAULT_BATCH_SIZE

def get_defaults():
    return {
        "engine": DEFAULT_ENGINE,
        "batch_size": DEFAULT_BATCH_SIZE,
    }
