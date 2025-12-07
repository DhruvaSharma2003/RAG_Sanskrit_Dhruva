import os
import json
import numpy as np
from typing import List, Dict

from loader import load_corpus
from preprocessing import build_corpus_chunks
from embedder import Embedder, save_embeddings, load_embeddings
from retriever import VectorRetriever, KeywordRetriever, keyword_boost
from generator import PhiGenerator    # Qwen2.5–1.5B


PROCESSED_CHUNKS_PATH = os.path.join("data", "processed", "chunks.json")
EMBEDDINGS_PATH = os.path.join("data", "processed", "embeddings.npy")


class RAGPipeline:
    """
    Full Retrieval-Augmented Generation pipeline for the Sanskrit RAG system.
    """

    def __init__(self):
        # -----------------------------------------------------------
        # Step 1: Load or build corpus chunks + embeddings
        # -----------------------------------------------------------
        if os.path.exists(PROCESSED_CHUNKS_PATH) and os.path.exists(EMBEDDINGS_PATH):
            print("✓ Loading preprocessed chunks and embeddings...")
            self.chunks = self._load_chunks()
            self.embeddings = load_embeddings(EMBEDDINGS_PATH)
        else:
            print("⚙ Building chunks & embeddings from raw Sanskrit documents...")
            self._build_and_save_corpus()

        # -----------------------------------------------------------
        # Step 2: Initialize embedder
        # -----------------------------------------------------------
        print("✓ Initializing embedder (multilingual-e5-small)...")
        self.embedder = Embedder()

        # -----------------------------------------------------------
        # Step 3: Build retrievers
        # -----------------------------------------------------------
        print("✓ Building vector retriever (FAISS)...")
        self.vector_retriever = VectorRetriever(self.embeddings, self.chunks)

        print("✓ Building keyword retriever (TF-IDF)...")
        self.keyword_retriever = KeywordRetriever(self.chunks)

        # -----------------------------------------------------------
        # Step 4: Initialize Generator
        # -----------------------------------------------------------
        print("✓ Loading Qwen2.5–1.5B Instruct generator (CPU mode)...")
        self.generator = PhiGenerator()

        print("🚀 RAG Pipeline Ready.\n")

    # -----------------------------------------------------------
    # File Loading / Saving
    # -----------------------------------------------------------

    def _load_chunks(self) -> List[Dict]:
        with open(PROCESSED_CHUNKS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_chunks(self, chunks: List[Dict]):
        os.makedirs(os.path.dirname(PROCESSED_CHUNKS_PATH), exist_ok=True)
        with open(PROCESSED_CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    def _build_and_save_corpus(self):
        """
        Load raw Sanskrit documents → preprocess → chunk → embed → save.
        """
        docs = load_corpus(os.path.join("data", "raw"))

        print("→ Preprocessing and chunking...")
        chunks = build_corpus_chunks(docs)
        self._save_chunks(chunks)
        self.chunks = chunks

        print("→ Embedding chunks (this may take time on first run)...")
        embedder = Embedder()
        embeddings = embedder.embed_chunks(chunks)

        save_embeddings(embeddings, EMBEDDINGS_PATH)
        self.embeddings = embeddings

    # -----------------------------------------------------------
    # Hybrid Retrieval
    # -----------------------------------------------------------

    def retrieve_context(self, question: str, top_k: int = 3, method: str = "vector"):
        """
        HYBRID RETRIEVAL:
        Vector + Keyword Boost (Sanskrit number/money terms)
        """
        if method == "vector":

            # 1. Vector search
            q_emb = self.embedder.embed_text(question)
            vector_results = self.vector_retriever.retrieve(q_emb, top_k=top_k)

            # 2. Keyword boost search
            keyword_results = keyword_boost(self.chunks, question, top_k=top_k)

            # 3. Merge results: keyword hits first, then vector results
            combined = keyword_results + vector_results

            # 4. Remove duplicates while preserving order
            seen = set()
            final = []
            for chunk in combined:
                if chunk["text"] not in seen:
                    final.append(chunk)
                    seen.add(chunk["text"])

            return final[:top_k]

        elif method == "keyword":
            return self.keyword_retriever.retrieve(question, top_k=top_k)

        else:
            raise ValueError("Method must be 'vector' or 'keyword'.")

    # -----------------------------------------------------------
    # Full RAG Pipeline
    # -----------------------------------------------------------

    def answer_query(self, question: str, top_k: int = 3, method: str = "vector"):
        """
        End-to-end RAG execution:
        1. Retrieve context
        2. Generate LLM answer
        3. Return both
        """
        retrieved = self.retrieve_context(question, top_k=top_k, method=method)
        answer = self.generator.generate_answer(question, retrieved)
        return answer, retrieved


# -----------------------------------------------------------
# Manual Testing
# -----------------------------------------------------------

if __name__ == "__main__":
    rag = RAGPipeline()

    question = "भोजराजा कियद् धनं दातुम् उक्तवान् ?"
    answer, ctx = rag.answer_query(question)

    print("\nANSWER:\n", answer)
    print("\nCONTEXT:")
    for c in ctx:
        print("-", c["text"][:120], "...")
