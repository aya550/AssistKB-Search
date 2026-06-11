import time

from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

from app import config, store
from app.embed import load_model

app = FastAPI(title="AssistKB Search")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


def generate_with_groq(question: str, context: str) -> tuple[str, dict]:
    if not config.GROQ_API_KEY:
        return (
            "Le contexte a été retrouvé, mais aucune clé GROQ_API_KEY n'est configurée.",
            {"prompt": len(context.split()), "completion": 0},
        )

    client = Groq(api_key=config.GROQ_API_KEY)

    prompt = f"""
Tu es AssistKB Search.
Tu réponds uniquement avec les informations présentes dans le contexte.
Tu dois citer les sources.
Si l'information n'est pas dans le contexte, réponds :
"Je ne dispose pas de cette information dans le corpus."

Question :
{question}

Contexte :
{context}
"""

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content

    usage = getattr(response, "usage", None)

    tokens = {
        "prompt": getattr(usage, "prompt_tokens", len(prompt.split())) if usage else len(prompt.split()),
        "completion": getattr(usage, "completion_tokens", len(answer.split())) if usage else len(answer.split()),
    }

    return answer, tokens


@app.post("/ask")
def ask(request: AskRequest):
    start = time.time()

    model = load_model()
    vector_store = store.get_store()

    question_vector = model.encode(
        [request.question],
        normalize_embeddings=True,
    )[0].tolist()

    hits = vector_store.search(
        query_vector=question_vector,
        top_k=config.TOP_K,
    )

    best_score = hits[0].score if hits else 0.0

    if not hits or best_score < config.SIMILARITY_THRESHOLD:
        return {
            "answer": "Je ne dispose pas de cette information dans le corpus.",
            "sources": [],
            "best_score": best_score,
            "refused": True,
            "latency_ms": int((time.time() - start) * 1000),
            "tokens": {"prompt": 0, "completion": 0},
        }

    context_parts = []
    sources = []

    for hit in hits:
        source = hit.metadata.get("source", "source inconnue")
        chunk_id = hit.metadata.get("chunk_id", None)

        context_parts.append(
            f"[source: {source}, chunk_id: {chunk_id}, score: {hit.score}]\n{hit.text}"
        )

        sources.append(
            {
                "doc": source,
                "chunk_id": chunk_id,
                "score": round(hit.score, 4),
            }
        )

    context = "\n\n".join(context_parts)

    answer, tokens = generate_with_groq(
        question=request.question,
        context=context,
    )

    return {
        "answer": answer,
        "sources": sources,
        "best_score": round(best_score, 4),
        "refused": False,
        "latency_ms": int((time.time() - start) * 1000),
        "tokens": tokens,
    }