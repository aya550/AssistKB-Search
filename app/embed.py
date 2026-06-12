import json
from pathlib import Path

from app import config, store

_model = None


def load_model():
    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer

        print(f"[R2] Chargement du modèle d'embeddings : {config.EMBED_MODEL}")
        _model = SentenceTransformer(config.EMBED_MODEL)

    return _model


def read_chunks(path: str):
    chunks_path = Path(path)

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Le fichier {path} est introuvable. "
            "Demandez à R1 de générer corpus/chunks.jsonl avec app/ingest.py."
        )

    # Correction BOM Windows
    with chunks_path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()

            if line:
                yield json.loads(line)


def main():
    chunks = list(read_chunks(config.CHUNKS_PATH))

    if not chunks:
        print("[R2] Aucun chunk trouvé.")
        return

    model = load_model()
    vector_store = store.get_store()

    texts = [chunk["text"] for chunk in chunks]

    ids = []
    payloads = []

    for index, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", index)

        try:
            chunk_id = int(chunk_id)
        except (ValueError, TypeError):
            chunk_id = index

        ids.append(chunk_id)

        metadata = chunk.get("metadata", {})

        payload = dict(metadata)
        payload["text"] = chunk["text"]
        payload["chunk_id"] = chunk_id
        payload["source"] = payload.get(
            "source",
            chunk.get("source", "unknown"),
        )

        payloads.append(payload)

    print(f"[R2] Vectorisation de {len(texts)} chunks...")

    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    real_dimension = len(vectors[0])

    if real_dimension != config.EMBED_DIM:
        raise ValueError(
            f"Dimension incorrecte : {real_dimension}. "
            f"Dimension attendue : {config.EMBED_DIM}."
        )

    vector_store.ensure_collection(
        vector_size=config.EMBED_DIM
    )

    vector_store.upsert(
        ids=ids,
        vectors=vectors.tolist(),
        payloads=payloads,
    )

    print("[R2] Indexation terminée avec succès dans Qdrant.")
    print(f"[R2] Nombre de chunks indexés : {len(chunks)}")
    print(f"[R2] Collection Qdrant : {config.QDRANT_COLLECTION}")


if __name__ == "__main__":
    main()