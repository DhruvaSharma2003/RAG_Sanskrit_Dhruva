import streamlit as st
from rag_pipeline import RAGPipeline


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


def main():
    st.set_page_config(
        page_title="Sanskrit RAG System",
        page_icon="📜",
        layout="wide"
    )

    st.title("📜 Sanskrit Retrieval-Augmented Generation (RAG) System")
    st.write(
        "This system answers questions from Sanskrit texts using a CPU-only "
        "vector + keyword hybrid RAG pipeline."
    )

    rag = load_pipeline()

    st.markdown("---")

    # ----------- INPUT UI -----------
    question = st.text_area(
        "Ask a question in Sanskrit or transliterated Sanskrit:",
        height=120,
        placeholder="उदाहरणम्: कालीदासः कस्य सभायां आसीत् ?",
    )

    retriever_type = st.selectbox(
        "Select Retrieval Method:",
        ["vector", "keyword", "hybrid"],  # <-- Added Hybrid Option A
        index=0,
    )

    top_k = st.slider("Number of context chunks to retrieve:", 1, 5, 3)

    # ----------- PROCESSING -----------
    if st.button("Generate Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Searching context & generating answer..."):
            answer, retrieved = rag.answer_query(
                question,
                top_k=top_k,
                method=retriever_type,
            )

        # ----------- ANSWER -----------
        st.markdown("## 🧾 Answer")
        st.write(answer)

        # ----------- CONTEXT -----------
        st.markdown("---")
        st.markdown("## 📚 Retrieved Context")

        if retrieved:
            for i, chunk in enumerate(retrieved, 1):
                score = chunk.get("score", None)

                if score is None:
                    label = f"Context {i}"
                else:
                    label = f"Context {i} (score: {score:.4f})"

                with st.expander(label):
                    st.write(chunk["text"])
        else:
            st.info("No relevant context retrieved.")

    # Footer
    st.markdown("---")
    st.caption("Developed by Dhruva • Sanskrit RAG Project")


if __name__ == "__main__":
    main()
