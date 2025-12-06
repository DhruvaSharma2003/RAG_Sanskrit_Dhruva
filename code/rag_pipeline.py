import os
import json
import numpy as np
from typing import List, Dict

from loader import load_corpus
from preprocessing import build_corpus_chunks
from embedder import Embedder, save_embeddings, load_embeddings
from retriever import VectorRetriever, KeywordRetriever
from generator import LlamaGenerator


PROCESSED_CHUNKS_PATH = os.path.join("data", "processed", "chunks.json")
EMBEDDINGS_PATH = os.path.join("data", "processed", "embeddings.npy")


class RAGPipeline:
    """
    Full Retrieval-Augmented Generation pipeline for Sanskrit RAG system.
    """

    def __init__(self):
        # Step 1: Load or build corpus chunks
        if os.path.exists(PROCESSED_CHUNKS_PATH) and os.path.exists(EMBEDDINGS_PATH):
            print("✓ Loading preprocessed chunks and embeddings...")
            self.chunks = self._load_chunks()
            self.embeddings = load_embeddings(EMBEDDINGS_PATH)
        else:
            print("⚙ Building chunks & embeddings from raw data...")
            self._build_and_save_corpus()

        # Step 2: Initialize embedder
        self.embedder = Embedder()

        # Step 3: Build retrievers
        print("✓ Building vector retriever (FAISS)...")
        self.vector_retriever = VectorRetriever(self.embeddings, self.chunks)

        print("✓ Building keyword retriever (TF-IDF)...")
        self.keyword_retriever = KeywordRetriever(self.chunks)

        # Step 4: Initialize generator
        print("✓ Loading LLaMA generator...")
        self.generator = LlamaGenerator()

    # -----------------------------------------------------------
    # Loading / Saving
    # -----------------------------------------------------------

    def _load_chunks(self) -> List[Dict]:
        with open(PROCESSED_CHUNKS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_chunks(self, chunks: List[Dict]):
        os.makedirs(os.path.dirname(PROCESSED_CHUNKS_PATH), exist_ok=True)
        with open(PROCESSED_CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    def _build_and_save_corpus(self):
        # Load raw docs
        docs = load_corpus(os.path.join("data", "raw"))

        # Preprocess → chunk
        chunks = build_corpus_chunks(docs)
        self._save_chunks(chunks)
        self.chunks = chunks

        # Embed chunks
        embedder = Embedder()
        embeddings = embedder.embed_chunks(chunks)
        save_embeddings(embeddings, EMBEDDINGS_PATH)
        self.embeddings = embeddings

    # -----------------------------------------------------------
    # Query Handling
    # -----------------------------------------------------------

    def retrieve_context(self, question: str, top_k: int = 3, method: str = "vector"):
        """
        Returns a list of chunk dicts with 'text', 'score', etc.
        """
        if method == "vector":
            q_emb = self.embedder.embed_text(question)
            return self.vector_retriever.retrieve(q_emb, top_k=top_k)

        elif method == "keyword":
            return self.keyword_retriever.retrieve(question, top_k=top_k)

        else:
            raise ValueError("Method must be 'vector' or 'keyword'")

    def answer_query(self, question: str, top_k: int = 3, method: str = "vector"):
        """
        Full RAG flow: retrieve → generate → return answer + context.
        """
        retrieved = self.retrieve_context(question, top_k=top_k, method=method)
        answer = self.generator.generate_answer(question, retrieved)

        return answer, retrieved


if __name__ == "__main__":
    rag = RAGPipeline()
    question = "कालीदासः कस्य राज्ञः सभायाम् उपस्थितः आसीत् ?"
    answer, ctx = rag.answer_query(question)
    print("\nANSWER:\n", answer)
    print("\nCONTEXT:")
    for c in ctx:
        print("-", c["text"][:120], "...")
