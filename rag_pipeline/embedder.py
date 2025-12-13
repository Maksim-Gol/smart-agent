"""OpenAI embeddings wrapper."""

from openai import OpenAI
from config import EMBEDDING_MODEL


class Embedder:
    """Wrapper for OpenAI embeddings API."""
    
    def __init__(self):
        self.client = OpenAI()
        self.model = EMBEDDING_MODEL
    
    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a batch."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]

