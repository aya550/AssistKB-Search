"""Ingestion : corpus brut (PDF/HTML/JSON/TXT) -> chunks + metadonnees.

ROLE : R1 (Data / Ingestion).   ===> A COMPLETER PAR R1 <===

Lance   : python -m app.ingest
Produit : corpus/chunks.jsonl  (une ligne JSON par chunk)

L'ossature (parcours des fichiers, ecriture du .jsonl) est fournie ci-dessous.
A toi d'implementer le coeur (les `TODO R1`) :
  - extract_text() : extraction robuste selon le format (PDF/HTML/JSON/TXT) ;
  - chunk_text()   : strategie de decoupage (taille / recouvrement) a justifier ;
  - metadonnees    : source au minimum (+ position, + bonus langue via langdetect).

Une version de reference complete t'est fournie HORS-GIT dans _reference/ingest.py :
comprends-la, adapte-la, puis commit TA version sous TON identite.
"""
import json
from pathlib import Path

from . import config


def extract_text(path: Path) -> str:
    """Extrait le texte brut d'un fichier selon son extension.

    TODO R1 : gerer .pdf (pypdf), .html/.htm (BeautifulSoup, retirer script/style),
    .json (aplatir les chaines) et le fallback texte. Tolerer l'encodage
    (errors="ignore") et ne JAMAIS planter sur un fichier corrompu (try/except).
    """
    raise NotImplementedError("TODO R1 : implementer extract_text()")


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Decoupe le texte en chunks avec recouvrement.

    TODO R1 : normaliser les espaces puis decouper en fenetres de `size`
    caracteres avec `overlap` de recouvrement (depart : 800 / 120, regle dans .env).
    Projet A : tester l'effet de ces valeurs sur la pertinence (note individuelle).
    """
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    raise NotImplementedError("TODO R1 : implementer chunk_text()")


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
                        # TODO R1 (bonus) : "lang": langdetect.detect(piece)
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_id += 1

    print(f"[ingest] {n_docs} documents -> {chunk_id} chunks dans {out_path}")


if __name__ == "__main__":
    main()
