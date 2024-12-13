from typing import List, Union
import numpy as np
import requests
import os

JINA_API_KEY = os.getenv("JINA_API_KEY")

class EmbeddingService:
    """Service for creating and managing text embeddings using OpenAI's API"""
    
    def __init__(self, model: str = "jina-embeddings-v3"):
        """
        Initialize the embedding service
        Args:
            model (str): The embedding model to use (default: jina-embeddings-v3)
        """
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}"
        }

    def get_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Get embeddings for a single text or list of texts
        Args:
            text: Single string or list of strings to embed
        Returns:
            numpy array of embeddings
        """
        try:
            # Ensure text is a list
            if isinstance(text, str):
                text = [text]
            
            response = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers=self.headers,
                json={
                    "model": self.model,
                    "task": "text-matching",
                    "late_chunking": False,
                    "dimensions": 1024,
                    "embedding_type": "float",
                    "input": text
                }
            )
            response.raise_for_status()
            
            # Extract embeddings from response
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            return np.array(embeddings)
            
        except Exception as e:
            raise Exception(f"Error getting embeddings: {str(e)}")

    def similarity_score(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Ensure embeddings are 1D
        if embedding1.ndim > 1:
            embedding1 = embedding1.flatten()
        if embedding2.ndim > 1:
            embedding2 = embedding2.flatten()
            
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)
