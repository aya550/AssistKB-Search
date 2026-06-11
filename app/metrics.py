"""Metriques : qualite de retrieval + exploitation.

ROLE : R4 (DevOps / Observabilite).   ===> A COMPLETER PAR R4 <===

A implementer (`TODO R4`) : au moins 1 metrique qualite + 1 metrique exploitation.
  - Qualite      : score de similarite moyen, taux de refus.
  - Exploitation : latence p50/p95, tokens, cout projete.
  - (bonus)      : golden dataset de 10 Q/R + recall@k.

Version de reference complete HORS-GIT : _reference/metrics.py.
"""


def percentile(values: list[float], p: float) -> float:
    """Percentile (p en 0..100).

    TODO R4 : trier, calculer l'indice (len-1)*p/100, interpoler entre voisins.
    """
    raise NotImplementedError("TODO R4 : implementer percentile()")


def summarize(runs: list[dict]) -> dict:
    """Agrege une liste de reponses /ask (best_score, refused, latency_ms, tokens).

    TODO R4 : renvoyer avg_best_score, refusal_rate, latency_p50_ms, latency_p95_ms,
    total_tokens.
    """
    raise NotImplementedError("TODO R4 : implementer summarize()")
