import os
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"


class Embedder:
    """
    Handles loading the embedding model and converting text/chunks to vectors.
    Designed for CPU-only operation.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        """
        emb = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return emb

    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        """
        Embed all chunks in the corpus.
        Returns matrix of shape (num_chunks, embedding_dim)
        """
        texts = [ch["text"] for ch in chunks]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings


def save_embeddings(embeddings: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)
