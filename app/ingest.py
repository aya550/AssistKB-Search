"""Ingestion : corpus brut (PDF/HTML/JSON/TXT) -> chunks + metadonnees.

ROLE : R1 (Data / Ingestion).

Lance   : python -m app.ingest
Produit : corpus/chunks.jsonl  (une ligne JSON par chunk)

Strategie de chunking : fenetres de CHUNK_SIZE caracteres avec CHUNK_OVERLAP
de recouvrement. Un recouvrement de ~15 % (120/800) evite de couper une phrase
a cheval entre deux chunks sans gonfler le volume. Taille de 800 chars = ~150
tokens, raisonnable pour all-MiniLM-L6-v2 (limite 256 tokens).
"""
import json
import re
from pathlib import Path

try:
    from langdetect import detect as _lang_detect
    from langdetect.lang_detect_exception import LangDetectException
except ImportError:
    _lang_detect = None  # type: ignore[assignment]
    LangDetectException = Exception


def detect_lang(text: str) -> str:
    if _lang_detect is None or not text:
        return "unknown"
    try:
        return _lang_detect(text[:500])
    except LangDetectException:
        return "unknown"

from . import config


def _flatten_json(obj, sep=" ") -> str:
    """Extrait recursivement toutes les valeurs string d'un objet JSON."""
    parts: list[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            parts.append(_flatten_json(v, sep))
    elif isinstance(obj, list):
        for item in obj:
            parts.append(_flatten_json(item, sep))
    elif isinstance(obj, str):
        parts.append(obj)
    return sep.join(p for p in parts if p)


def extract_text(path: Path) -> str:
    """Extrait le texte brut d'un fichier selon son extension."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            import pypdf  # noqa: PLC0415
            reader = pypdf.PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)

        if suffix in {".html", ".htm"}:
            from bs4 import BeautifulSoup  # noqa: PLC0415
            raw = path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "head"]):
                tag.decompose()
            return soup.get_text(separator="\n")

        if suffix == ".json":
            raw = path.read_text(encoding="utf-8", errors="ignore")
            obj = json.loads(raw)
            return _flatten_json(obj)

        # fallback : fichier texte brut
        return path.read_text(encoding="utf-8", errors="ignore")

    except Exception as exc:  # noqa: BLE001
        print(f"[ingest] erreur lecture {path} : {exc}")
        return ""


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Decoupe le texte en chunks avec recouvrement."""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    # Normalisation : collapse whitespace et sauts de ligne multiples
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()

    if not text:
        return []

    step = size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def iter_corpus_files():
    """Parcourt tous les dossiers de CORPUS_DIRS et renvoie les fichiers. (fourni)"""
    for directory in config.CORPUS_DIRS:
        base = Path(directory)
        if not base.exists():
            print(f"[ingest] dossier absent, ignore : {base}")
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                yield path


def main() -> None:
    out_path = Path(config.CHUNKS_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_id = 0
    n_docs = 0
    with out_path.open("w", encoding="utf-8") as f:
        for path in iter_corpus_files():
            text = extract_text(path)          # <-- TODO R1
            pieces = chunk_text(text)          # <-- TODO R1
            if not pieces:
                continue
            n_docs += 1
            for position, piece in enumerate(pieces):
                record = {
                    "chunk_id": chunk_id,
                    "text": piece,
                    "metadata": {
                        "source": path.name,
                        "path": str(path).replace("\\", "/"),
                        "type": path.suffix.lower().lstrip("."),
                        "position": position,
                        "lang": detect_lang(piece),
                        
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_id += 1

    print(f"[ingest] {n_docs} documents -> {chunk_id} chunks dans {out_path}")


if __name__ == "__main__":
    main()
