#text_utils.py
import re

def normalize_text(s: str) -> str:
    s = s.replace("\ufb01", "fi").replace("\ufb02", "fl").replace("\u00ad", "")
    s = s.replace("￾", "").replace("�", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def de_spaced_word(s: str) -> str:
    # "a b s t r a c t" -> "abstract"
    s = (s or "").strip()
    if re.fullmatch(r"(?:[A-Za-z]\s+){3,}[A-Za-z]", s):
        return s.replace(" ", "")
    return s

def looks_like_header_footer(text: str, y: float, page_h: float) -> bool:
    low = (text or "").lower()

    # headers típicos de papers (ajusta si hace falta)
    if y < 70 and ("fundamental research" in low or "sciencedirect" in low or "journal homepage" in low):
        return True

    # page number footer
    if y > page_h - 50 and re.fullmatch(r"\d{1,4}", (text or "").strip()):
        return True

    return False

def reflow_lines(lines: list[str]) -> str:
    """
    Une líneas en texto corrido, respetando saltos de párrafo simples.
    """
    out = []
    for raw in lines:
        t = (raw or "").strip()
        if not t:
            continue

        if not out:
            out.append(t)
            continue

        prev = out[-1]
        # unir palabra cortada con guion
        if prev.endswith("-"):
            out[-1] = prev[:-1] + t
        else:
            # si parece fin de frase, deja salto
            if prev.endswith((".", "?", "!", ":", ";")):
                out.append("\n" + t)
            else:
                out[-1] = prev + " " + t

    text = " ".join(out)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
