import os
import json
import numpy as np
from typing import List, Dict

from loader import load_corpus
from preprocessing import build_corpus_chunks
from embedder import Embedder, save_embeddings, load_embeddings
from retriever import VectorRetriever, KeywordRetriever, keyword_boost
from generator import PhiGenerator   


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
        # Step 2: Embedder
        # -----------------------------------------------------------
        print("✓ Initializing embedder (multilingual-e5-small)...")
        self.embedder = Embedder()

        # -----------------------------------------------------------
        # Step 3: Retrievers
        # -----------------------------------------------------------
        print("✓ Building vector retriever (FAISS)...")
        self.vector_retriever = VectorRetriever(self.embeddings, self.chunks)

        print("✓ Building keyword retriever (TF-IDF)...")
        self.keyword_retriever = KeywordRetriever(self.chunks)

        # -----------------------------------------------------------
        # Step 4: LLM Generator
        # -----------------------------------------------------------
        print("✓ Loading Qwen2.5–1.5B Instruct generator (CPU mode)...")
        self.generator = PhiGenerator()

        print("🚀 RAG Pipeline Ready.\n")

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
        """
        Load raw documents → preprocess → chunk → embed → save
        """
        docs = load_corpus(os.path.join("data", "raw"))

        print("→ Preprocessing and chunking...")
        chunks = build_corpus_chunks(docs)
        self._save_chunks(chunks)
        self.chunks = chunks

        print("→ Embedding chunks (first time only)...")
        embedder = Embedder()
        embeddings = embedder.embed_chunks(chunks)

        save_embeddings(embeddings, EMBEDDINGS_PATH)
        self.embeddings = embeddings

    # -----------------------------------------------------------
    # Hybrid Retrieval 
    # -----------------------------------------------------------

    def retrieve_context(self, question: str, top_k: int = 3, method: str = "hybrid"):
        """
        Hybrid (preferred):
            keyword_boost + vector search (merged)
        
        Legacy:
            method="vector" or method="keyword"
        """

        # HYBRID = Vector + Keyword Boost merged
        if method == "hybrid":

            q_emb = self.embedder.embed_text(question)

            # 1. Vector retrieval
            vector_results = self.vector_retriever.retrieve(q_emb, top_k=top_k)

            # 2. Keyword-boosted retrieval
            keyword_results = keyword_boost(self.chunks, question, top_k=top_k)

            # 3. Merge: keyword-boosted comes first
            combined = keyword_results + vector_results

            # 4. Deduplicate (preserve order)
            seen = set()
            final = []
            for item in combined:
                tid = item["chunk_id"]
                if tid not in seen:
                    final.append(item)
                    seen.add(tid)

            return final[:top_k]

        # Vector-only
        elif method == "vector":
            q_emb = self.embedder.embed_text(question)
            return self.vector_retriever.retrieve(q_emb, top_k=top_k)

        # Keyword-only
        elif method == "keyword":
            return self.keyword_retriever.retrieve(question, top_k=top_k)

        else:
            raise ValueError("method must be 'hybrid', 'vector', or 'keyword'")

    # -----------------------------------------------------------
    # Full RAG: retrieve → generate
    # -----------------------------------------------------------

    def answer_query(self, question: str, top_k: int = 3, method: str = "hybrid"):
        retrieved = self.retrieve_context(question, top_k=top_k, method=method)
        answer = self.generator.generate_answer(question, retrieved)
        return answer, retrieved


# -----------------------------------------------------------
# Manual Test(For Debugging)
# -----------------------------------------------------------
if __name__ == "__main__":
    rag = RAGPipeline()
    q = "भोजराजा कियद् धनं दातुम् उक्तवान् ?"
    ans, ctx = rag.answer_query(q, top_k=3, method="hybrid")

    print("\nANSWER:\n", ans)
    print("\nCONTEXT:")
    for c in ctx:
        print("-", c["chunk_id"], c["score"], c["text"][:120], "...")
