"""Embeddings : chunks.jsonl -> vecteurs normalises -> upsert dans le vector store.

ROLE : R2 (Embeddings / Index).   ===> A COMPLETER PAR R2 <===

Lance : python -m app.embed   (apres python -m app.ingest)

L'ossature (chargement du modele, lecture des chunks) est fournie.
A toi d'implementer le coeur (`TODO R2`) :
  - encoder les textes en vecteurs NORMALISES (normalize_embeddings=True -> cosinus) ;
  - creer la collection (config.EMBED_DIM = 384) puis upsert ;
  - garantir l'IDEMPOTENCE : relancer ne duplique pas (ids = chunk_id) ;
  - regler la taille de batch.

NB projet A : l'adaptateur QdrantStore est deja fourni dans app/store.py (exemple).
Version de reference complete HORS-GIT : _reference/embed.py.
"""
import json
from pathlib import Path

from . import config, store

_model = None


def load_model():
    """Charge le modele sentence-transformers (cache apres 1er appel). (fourni)"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        print(f"[embed] chargement du modele {config.EMBED_MODEL} ...")
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def read_chunks(path: str):
    """Lit corpus/chunks.jsonl ligne par ligne. (fourni)"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} introuvable. Lancez d'abord : python -m app.ingest")
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    chunks = list(read_chunks(config.CHUNKS_PATH))
    if not chunks:
        print("[embed] aucun chunk a indexer.")
        return

    model = load_model()          # noqa: F841  (utilise dans le TODO ci-dessous)
    vs = store.get_store()        # noqa: F841

    # TODO R2 : encoder les textes (model.encode(..., normalize_embeddings=True)),
    # creer la collection (vs.ensure_collection(config.EMBED_DIM)) puis vs.upsert(...)
    # de maniere idempotente (ids = chunk_id pour ecraser au lieu de dupliquer).
    raise NotImplementedError("TODO R2 : implementer l'encodage + upsert")


if __name__ == "__main__":
    main()
