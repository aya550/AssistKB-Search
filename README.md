# AssistKB Search — Pipeline RAG (Projet A)

Assistant interne de type **RAG** (Retrieval-Augmented Generation) sur une base de
connaissances technique. Le système indexe un corpus documentaire dans **Qdrant**,
recherche les passages les plus pertinents pour une question, puis génère une
réponse **citant ses sources** — et **refuse de répondre** si rien dans le corpus
n'est assez proche (anti-hallucination).

> TP en équipe — Module RAG. Vector store : **Qdrant** · Embeddings locaux :
> **sentence-transformers** · LLM : **Groq** (free tier).

---

## Sommaire

- [Architecture](#architecture)
- [Stack technique](#stack-technique)
- [Prérequis](#prérequis)
- [Installation & lancement](#installation--lancement)
- [Tester le projet](#tester-le-projet)
- [Endpoints de l'API](#endpoints-de-lapi)
- [Structure du projet](#structure-du-projet)
- [Configuration (.env)](#configuration-env)
- [Équipe & rôles](#équipe--rôles)

---

## Architecture

```
Ingestion (hors-ligne) :
  corpus (HTML/PDF/JSON/TXT) → extraction → chunking → embeddings → Qdrant

Service (en ligne) :
  question → embedding → recherche top-k → [seuil de refus] → LLM → réponse citée
```

- **R1 – Ingestion** (`app/ingest.py`) : extraction multi-format + découpage en chunks.
- **R2 – Index** (`app/embed.py`, `app/store.py`) : vectorisation + upsert dans Qdrant.
- **R3 – Retrieval / LLM** (`app/retrieve.py`, `app/generate.py`, `app/api.py`) :
  recherche top-k, seuil de refus, génération citée, API `POST /ask`.
- **R4 – DevOps / Métriques** (`docker-compose.yml`, `app/metrics.py`).

## Stack technique

| Composant | Choix |
|---|---|
| Vector store | Qdrant |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384 dim, cosinus) |
| LLM | Groq — `llama-3.1-8b-instant` (free tier) |
| API | FastAPI + Uvicorn |
| Orchestration | Docker Compose |

---

## Prérequis

- **Docker Desktop** installé et démarré.
- Une **clé API Groq** gratuite : <https://console.groq.com/keys>

## Installation & lancement

```bash
# 1. Cloner le dépôt
git clone https://github.com/aya550/AssistKB-Search.git
cd AssistKB-Search

# 2. Configurer la clé LLM
cp .env.example .env        # Windows PowerShell : Copy-Item .env.example .env
# puis renseigner GROQ_API_KEY=gsk_... dans .env

# 3. (optionnel) Récupérer un corpus plus riche (sinon le corpus seed suffit)
#    Linux/Mac :   PROFILE=open bash scripts/fetch_corpus.sh
#    Windows   :   .\scripts\fetch_corpus.ps1 -Profile open

# 4. Démarrer la stack (Qdrant + API)
docker compose up -d

# 5. Indexer le corpus
docker compose run --rm api python -m app.ingest   # corpus → chunks.jsonl
docker compose run --rm api python -m app.embed    # chunks → vecteurs dans Qdrant
```

> Au 1er `embed`, le modèle d'embeddings (~90 Mo) est téléchargé puis mis en cache.

---

## Tester le projet

Une fois la stack lancée et le corpus indexé :

### 1. Interface interactive (le plus simple)
Ouvre **<http://localhost:8000/docs>** → `POST /ask` → **Try it out** → saisis :
```json
{ "question": "Qu'est-ce que Qdrant et a quoi sert-il ?" }
```
→ **Execute**. Tu obtiens la réponse, ses sources, la latence et les tokens.

### 2. En ligne de commande
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Comment anonymiser les donnees du corpus ?"}'
```
PowerShell :
```powershell
$body = @{ question = "Qu'est-ce que Qdrant ?" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/ask -Method Post -ContentType 'application/json' -Body $body
```

### 3. Voir les données indexées
Dashboard Qdrant : **<http://localhost:6333/dashboard>** → collection `assistkb`
(16 points, dim 384, distance Cosine).

### Exemple de réponse
```json
{
  "answer": "Qdrant est un systeme de recherche de similarite... [source: archi_assistkb_v0.html #4]",
  "sources": [{ "doc": "archi_assistkb_v0.html", "chunk_id": 4, "score": 0.45 }],
  "refused": false,
  "best_score": 0.45,
  "latency_ms": 980,
  "tokens": { "prompt": 203, "completion": 55 }
}
```

Une question **hors-sujet** (ex. *« recette de tarte aux pommes »*) renvoie
`"refused": true` sans appeler le LLM.

---

## Endpoints de l'API

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/health` | Vérifie que l'API tourne |
| `GET` | `/docs` | Interface interactive (Swagger) |
| `POST` | `/ask` | Pose une question — body : `{"question": "...", "top_k": 5, "threshold": 0.45}` |

`top_k` et `threshold` sont optionnels (valeurs par défaut dans `.env` / `config.py`).

## Structure du projet

```
AssistKB-Search/
├── app/
│   ├── config.py       # configuration (lue depuis .env)
│   ├── ingest.py       # R1 : extraction + chunking
│   ├── embed.py        # R2 : embeddings + upsert Qdrant
│   ├── store.py        # R2 : adaptateur Qdrant (ensure_collection / upsert / search)
│   ├── retrieve.py     # R3 : recherche top-k + seuil de refus
│   ├── generate.py     # R3 : prompt anti-hallucination + appel LLM
│   ├── api.py          # R3 : endpoint FastAPI POST /ask
│   └── metrics.py      # R4 : métriques (score, refus, latence, tokens, coût)
├── scripts/            # fetch_corpus.sh / .ps1 (récupération de corpus public)
├── corpus/             # raw/ (téléchargé, gitignoré) + chunks.jsonl
├── docker-compose.yml  # Qdrant + API
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Configuration (.env)

| Variable | Défaut | Rôle |
|---|---|---|
| `GROQ_API_KEY` | — | Clé LLM (obligatoire pour la génération) |
| `LLM_PROVIDER` | `groq` | `groq` ou `gemini` |
| `VECTOR_STORE` | `qdrant` | Vector store utilisé |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Modèle d'embeddings local |
| `TOP_K` | `5` | Nombre de chunks récupérés |
| `SIMILARITY_THRESHOLD` | `0.45` | Seuil de refus (cosinus) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Découpage des documents |

> ⚠️ Le fichier `.env` (avec votre clé) ne doit **jamais** être commité — il est gitignoré.

## Équipe & rôles

| Membre | Rôle |
|---|---|
| Aya SGHAIER SLIM | R1 Data / Ingestion + R4 DevOps / Observabilité |
| Danielle | R2 Embeddings / Index |
| Jacqueline MAPENZI | R3 Retrieval / LLM |

Le détail des choix techniques, métriques et limites est dans le
[compte rendu](projet-A-assistkb-search/COMPTE-RENDU-attendu.html).
