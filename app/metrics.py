def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)
    index = (len(values) - 1) * p / 100
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return float(values[lower])

    return float(
        values[lower]
        + (values[upper] - values[lower]) * (index - lower)
    )


def summarize_metrics(runs: list[dict]) -> dict:
    if not runs:
        return {
            "avg_best_score": 0,
            "refusal_rate": 0,
            "latency_p50_ms": 0,
            "latency_p95_ms": 0,
            "total_tokens": 0,
        }

    scores = [run.get("best_score", 0) for run in runs]
    refusals = [1 if run.get("refused") else 0 for run in runs]
    latencies = [run.get("latency_ms", 0) for run in runs]

    total_tokens = 0

    for run in runs:
        tokens = run.get("tokens", {})
        total_tokens += tokens.get("prompt", 0)
        total_tokens += tokens.get("completion", 0)

    return {
        "avg_best_score": round(sum(scores) / len(scores), 4),
        "refusal_rate": round(sum(refusals) / len(refusals), 4),
        "latency_p50_ms": round(percentile(latencies, 50), 2),
        "latency_p95_ms": round(percentile(latencies, 95), 2),
        "total_tokens": total_tokens,
    }