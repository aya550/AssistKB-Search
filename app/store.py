from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app import config


@dataclass
class SearchHit:
    text: str
    score: float
    metadata: dict[str, Any]


class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(url=config.QDRANT_URL)
        self.collection_name = config.QDRANT_COLLECTION

    def ensure_collection(self, vector_size: int = config.EMBED_DIM):
        collections = self.client.get_collections().collections
        existing_names = [c.name for c in collections]

        if self.collection_name not in existing_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(
        self,
        ids: list[int],
        vectors: list[list[float]],
        payloads: list[dict],
    ):
        points = []

        for item_id, vector, payload in zip(ids, vectors, payloads):
            points.append(
                PointStruct(
                    id=item_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[SearchHit]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

        hits = []

        for result in results:
            payload = result.payload or {}
            text = payload.get("text", "")

            hits.append(
                SearchHit(
                    text=text,
                    score=float(result.score),
                    metadata=payload,
                )
            )

        return hits


def get_store() -> QdrantStore:
    return QdrantStore()