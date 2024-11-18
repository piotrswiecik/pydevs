import logging
import os
from typing import List, Optional, TypedDict
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams
from qdrant_client.http.models.models import Distance
import uuid


Point = TypedDict("Point", {"id": str, "text": str, "role": Optional[str]})

OPENAI_EMBEDDING_SIZE = 1536


class VectorDBService:
    def __init__(
        self,
        embedding_provider,
        db_url: Optional[str] = None,
        db_api_key: Optional[str] = None,
    ):
        if db_url is None:
            db_url = os.environ.get("VECTOR_DB_URL")
        self.db_url = db_url

        if db_api_key is None:
            db_api_key = os.environ.get("VECTOR_DB_API_KEY")
        self.db_api_key = db_api_key

        self._client = QdrantClient(url=db_url, api_key=db_api_key)
        self._ep = embedding_provider

    def create_collection(self, collection_name: str):
        collection = self._client.collection_exists(collection_name)
        if not collection:
            self._client.create_collection(
                collection_name,
                vectors_config=VectorParams(
                    size=OPENAI_EMBEDDING_SIZE, distance=Distance.COSINE
                ),
            )

    def collection_exists(self, collection_name: str):
        return self._client.collection_exists(collection_name)

    def add_points(self, collection_name: str, points: List[Point]):
        for point in points:
            if not isinstance(point["id"], int):
                logging.info("ID is not an integer, generating a new one as UUID4.")
                point["id"] = str(uuid.uuid4())
        payload = [
            {
                "id": point["id"],
                "vector": self._ep.text_embedding(point["text"]),
                "payload": {"role": point["role"], "text": point["text"]},
            }
            for point in points
        ]
        res = self._client.upsert(collection_name, points=payload, wait=True)
        logging.info(res)

    def search(self, collection_name: str, query: str, limit=5):
        embedding = self._ep.text_embedding(query)
        self._client.search(
            collection_name, query_vector=embedding, limit=limit, with_payload=True
        )
