"""Generation : contexte (chunks) + question -> reponse citee par le LLM.

ROLE : R3 (Retrieval / LLM).

Le prompt FORCE la citation des sources et INTERDIT d'inventer. Le LLM est
appele via Groq (defaut) ou Gemini, selon LLM_PROVIDER.
"""
from . import config

SYSTEM_PROMPT = (
    "Tu es un assistant documentaire. Tu reponds UNIQUEMENT a partir du contexte "
    "fourni. Si le contexte ne contient pas la reponse, dis-le explicitement sans "
    "rien inventer. Cite tes sources entre crochets au format [source: <fichier> #<chunk_id>] "
    "apres chaque affirmation issue du contexte."
)


def format_context(hits) -> str:
    blocks = []
    for h in hits:
        src = h.metadata.get("source", "?")
        cid = h.metadata.get("chunk_id", "?")
        blocks.append(f"[source: {src} #{cid}]\n{h.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, hits) -> str:
    return (
        f"Contexte :\n{format_context(hits)}\n\n"
        f"Question : {question}\n\n"
        "Reponds en citant les sources utilisees."
    )


def _call_groq(question: str, hits) -> tuple[str, dict]:
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, hits)},
        ],
    )
    answer = resp.choices[0].message.content
    usage = {
        "prompt": resp.usage.prompt_tokens,
        "completion": resp.usage.completion_tokens,
    }
    return answer, usage


def _call_gemini(question: str, hits) -> tuple[str, dict]:
    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        config.GEMINI_MODEL, system_instruction=SYSTEM_PROMPT
    )
    resp = model.generate_content(build_user_prompt(question, hits))
    usage = {
        "prompt": getattr(resp.usage_metadata, "prompt_token_count", 0),
        "completion": getattr(resp.usage_metadata, "candidates_token_count", 0),
    }
    return resp.text, usage


def generate_answer(question: str, hits) -> tuple[str, dict]:
    """Renvoie (reponse, usage_tokens) selon le provider configure."""
    if config.LLM_PROVIDER == "groq":
        return _call_groq(question, hits)
    if config.LLM_PROVIDER == "gemini":
        return _call_gemini(question, hits)
    raise ValueError(
        f"LLM_PROVIDER inconnu : {config.LLM_PROVIDER!r} (attendu : 'groq' ou 'gemini')"
    )
