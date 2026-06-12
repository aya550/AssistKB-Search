"""Metriques : qualite de retrieval + exploitation.

ROLE : R4 (DevOps / Observabilite).

Usage :
    from app.metrics import summarize
    rapport = summarize(runs)   # runs = liste de dicts retournes par /ask

Chaque element de `runs` doit contenir :
    - best_score   (float)  : score du chunk le mieux classe
    - refused      (bool)   : True si l'API a refuse de repondre
    - latency_ms   (float)  : latence totale de la requete
    - tokens       (int | dict) : total, ou {"prompt": .., "completion": ..} (format /ask)
"""


def percentile(values: list[float], p: float) -> float:
    """Percentile par interpolation lineaire (p en 0..100)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = (n - 1) * p / 100.0
    lo = int(idx)
    hi = lo + 1
    frac = idx - lo
    if hi >= n:
        return sorted_vals[-1]
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _total_tokens(t) -> int:
    """Total de tokens, que /ask renvoie un entier ou un dict {prompt, completion}."""
    if isinstance(t, dict):
        return int(t.get("prompt", 0)) + int(t.get("completion", 0))
    return int(t or 0)


def summarize(runs: list[dict]) -> dict:
    """Agrege une liste de reponses /ask et retourne les metriques cles."""
    if not runs:
        return {}

    scores = [r["best_score"] for r in runs if not r.get("refused", False)]
    latencies = [r["latency_ms"] for r in runs]
    tokens = [_total_tokens(r.get("tokens", 0)) for r in runs]
    n_refused = sum(1 for r in runs if r.get("refused", False))

    # Cout projete : tarif Groq llama-3.1-8b-instant (gratuit tier = 0, on simule OpenAI gpt-4o-mini)
    # $0.15 / 1M tokens input + $0.60 / 1M tokens output (approx 50/50)
    avg_tokens = sum(tokens) / len(tokens) if tokens else 0
    cost_per_1k = avg_tokens * 1000 * (0.15 + 0.60) / 2 / 1_000_000

    return {
        "n_questions": len(runs),
        "avg_best_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "refusal_rate_pct": round(100 * n_refused / len(runs), 1),
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
        "avg_tokens": round(avg_tokens, 1),
        "projected_cost_usd_per_1k_questions": round(cost_per_1k, 4),
    }
