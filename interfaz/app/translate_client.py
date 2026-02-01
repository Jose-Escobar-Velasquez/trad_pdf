import requests

def translate_texts(
    api_base_url: str,
    texts: list[str],
    engine: str = "opus",
    batch_size: int = 8,
    timeout_sec: int = 72000,
):
    url = api_base_url.rstrip("/") + "/translate"
    payload = {"texts": texts, "engine": engine, "batch_size": batch_size}

    # timeout = (connect, read)
    r = requests.post(url, json=payload, timeout=(10, timeout_sec))
    r.raise_for_status()
    return r.json()["translations"]
