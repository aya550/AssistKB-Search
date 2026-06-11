"""Abstraction du vector store + implementation Qdrant.

ROLE : R2 (Embeddings / Index).

QdrantStore est FOURNI ENTIEREMENT (il sert d'exemple de reference, comme
indique dans le README du projet A). Pour aller plus loin, R2 peut :
  - ajouter un filtrage par metadonnees (payload) au moment du search,
  - gerer des collections multiples (par type de source),
  - exposer une methode count() pour les metriques.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from . import config


@dataclass
class SearchHit:
    """Un resultat de recherche : le texte du chunk, son score, ses metadonnees."""

    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorStore(ABC):
    """Contrat commun a tous les vector stores (Qdrant ici, mais interchangeable)."""

    @abstractmethod
    def ensure_collection(self, dim: int) -> None:
        """Cree la collection si elle n'existe pas (distance cosinus)."""

    @abstractmethod
    def upsert(
        self,
        ids: Sequence[int],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict],
    ) -> None:
        """Insere/maj des points. Idempotent : meme id => ecrasement."""

    @abstractmethod
    def search(self, vector: Sequence[float], top_k: int) -> list[SearchHit]:
        """Recherche les top_k chunks les plus proches du vecteur de requete."""


class QdrantStore(VectorStore):
    """Adaptateur Qdrant. Distance cosinus (vecteurs normalises, dim 384)."""

    def __init__(self, url: str | None = None, collection: str | None = None):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=url or config.QDRANT_URL)
        self.collection = collection or config.QDRANT_COLLECTION

    def ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=int(i), vector=list(v), payload=dict(p))
            for i, v, p in zip(ids, vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector, top_k: int) -> list[SearchHit]:
        # qdrant-client >= 1.12 : .search() est remplace par .query_points()
        results = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            limit=top_k,
            with_payload=True,
        ).points
        hits: list[SearchHit] = []
        for r in results:
            payload = dict(r.payload or {})
            text = payload.pop("text", "")
            hits.append(SearchHit(text=text, score=float(r.score), metadata=payload))
        return hits

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count


def get_store() -> VectorStore:
    """Fabrique : renvoie l'implementation choisie via VECTOR_STORE."""
    if config.VECTOR_STORE == "qdrant":
        return QdrantStore()
    raise ValueError(
        f"VECTOR_STORE inconnu : {config.VECTOR_STORE!r} (attendu : 'qdrant' pour le projet A)"
    )
