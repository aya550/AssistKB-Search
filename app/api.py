"""API REST : POST /ask -> reponse citee + sources + latence + tokens.

ROLE : R3 (Retrieval / LLM).

Lance (local)  : uvicorn app.api:app --reload
Lance (docker) : voir docker-compose.yml
"""
import time

from fastapi import FastAPI
from pydantic import BaseModel

from . import config, generate, retrieve

app = FastAPI(title="AssistKB Search", version="0.1.0")


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None
    threshold: float | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "vector_store": config.VECTOR_STORE, "llm": config.LLM_PROVIDER}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    t0 = time.perf_counter()
    result = retrieve.retrieve(req.question, req.top_k, req.threshold)

    # Refus sous le seuil : on ne sollicite PAS le LLM (anti-hallucination + economie).
    if result["refused"]:
        return {
            "answer": config.REFUSAL_MESSAGE,
            "sources": [],
            "refused": True,
            "best_score": round(result["best_score"], 4),
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "tokens": {"prompt": 0, "completion": 0},
        }

    answer, usage = generate.generate_answer(req.question, result["accepted"])
    sources = [
        {
            "doc": h.metadata.get("source", "?"),
            "chunk_id": h.metadata.get("chunk_id"),
            "score": round(h.score, 4),
        }
        for h in result["accepted"]
    ]
    return {
        "answer": answer,
        "sources": sources,
        "refused": False,
        "best_score": round(result["best_score"], 4),
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "tokens": usage,
    }
