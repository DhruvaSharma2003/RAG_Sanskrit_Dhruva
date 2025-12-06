import streamlit as st
from rag_pipeline import RAGPipeline


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


def main():
    st.set_page_config(page_title="Sanskrit RAG System", page_icon="📜", layout="wide")

    st.title("📜 Sanskrit Retrieval-Augmented Generation (RAG) System")
    st.write(
        "This system answers questions based on Sanskrit texts using a CPU-only "
        "Retriever + Generator architecture."
    )

    # Load pipeline only once
    rag = load_pipeline()

    st.markdown("---")

    # User input
    question = st.text_area(
        "Ask a question in Sanskrit or transliterated Sanskrit:",
        height=120,
        placeholder="उदाहरणम्: कालीदासः कस्य सभायां आसित् ?",
    )

    retriever_type = st.selectbox(
        "Select Retrieval Method:",
        ["vector", "keyword"],
        index=0,
    )

    top_k = st.slider("Number of context chunks:", 1, 5, 3)

    if st.button("Generate Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Thinking..."):
            answer, ctx = rag.answer_query(question, top_k=top_k, method=retriever_type)

        st.markdown("## 🧾 Answer")
        st.write(answer)

        st.markdown("---")
        st.markdown("## 📚 Retrieved Context")

        for i, chunk in enumerate(ctx, start=1):
            with st.expander(f"Context {i} (score: {chunk['score']:.4f})"):
                st.write(chunk["text"])

    st.markdown("---")
    st.caption("Developed by Dhruva • Sanskrit RAG Internship Assignment")


if __name__ == "__main__":
    main()
