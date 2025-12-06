import numpy as np
import faiss
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer


class VectorRetriever:
    """
    FAISS-based vector retriever using cosine similarity.
    """

    def __init__(self, embeddings: np.ndarray, chunks: List[Dict]):
        self.embeddings = embeddings.astype("float32")
        self.chunks = chunks

        # Build FAISS index (L2 normalized embeddings → use IndexFlatIP)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Returns top-k most relevant chunks.
        """
        query_vec = query_embedding.astype("float32").reshape(1, -1)
        scores, idxs = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "doc_id": chunk["doc_id"],
                "score": float(score)
            })
        return results


class KeywordRetriever:
    """
    TF-IDF keyword-based retriever.
    """

    def __init__(self, chunks: List[Dict]):
        texts = [ch["text"] for ch in chunks]
        self.vectorizer = TfidfVectorizer(analyzer="word")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        query_vec = self.vectorizer.transform([query])
        scores = (self.tfidf_matrix @ query_vec.T).toarray().flatten()

        top_idxs = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_idxs:
            results.append({
                "chunk_id": self.chunks[idx]["chunk_id"],
                "text": self.chunks[idx]["text"],
                "doc_id": self.chunks[idx]["doc_id"],
                "score": float(scores[idx]),
            })
        return results
