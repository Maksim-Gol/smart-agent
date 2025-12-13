"""Qdrant vector store wrapper."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_DIM


class VectorStore:
    """Wrapper for Qdrant vector database."""
    
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = COLLECTION_NAME
    
    def create_collection(self, recreate: bool = False):
        """Create the collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        
        if self.collection_name in collections:
            if recreate:
                self.client.delete_collection(self.collection_name)
            else:
                print(f"Collection '{self.collection_name}' already exists.")
                return
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        print(f"Created collection '{self.collection_name}'.")
    
    def upsert(self, points: list[dict]):
        """Insert or update points. Each point: {id, vector, payload}"""
        qdrant_points = [
            PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
            for p in points
        ]
        self.client.upsert(collection_name=self.collection_name, points=qdrant_points)
    
    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        """Search for similar vectors."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return [
            {"score": r.score, "payload": r.payload}
            for r in results
        ]

