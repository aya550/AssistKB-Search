"""Recherche : question -> embedding -> top-k Qdrant -> filtrage par seuil.

ROLE : R3 (Retrieval / LLM).

Coeur de la qualite de retrieval du projet A : si le meilleur score de
similarite est sous SIMILARITY_THRESHOLD, on REFUSE (anti-hallucination).

TODO (R3) : tester plusieurs top_k / seuils, (bonus) reranking cross-encoder.
"""
from . import config, embed, store

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = embed.load_model()
    return _model


def retrieve(question: str, top_k: int | None = None, threshold: float | None = None) -> dict:
    """Renvoie les hits, ceux acceptes (>= seuil), et le drapeau de refus."""
    top_k = top_k or config.TOP_K
    threshold = config.SIMILARITY_THRESHOLD if threshold is None else threshold

    model = _get_model()
    qvec = model.encode([question], normalize_embeddings=True)[0].tolist()

    vs = store.get_store()
    hits = vs.search(qvec, top_k)

    accepted = [h for h in hits if h.score >= threshold]
    return {
        "hits": hits,
        "accepted": accepted,
        "refused": len(accepted) == 0,
        "best_score": hits[0].score if hits else 0.0,
        "threshold": threshold,
    }
