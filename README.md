# 🕉️ Sanskrit RAG System  
### **Retrieval-Augmented Generation using Qwen2.5–1.5B & Local Knowledge Base**  
**Author:** Dhruva Sharma  
**Project Type:** AI/ML Intern Assessment  
**Model:** Qwen2.5–1.5B Instruct (Local, Offline CPU Mode)  
**Embeddings:** Multilingual-e5-small  
**Retrievers:** FAISS Vector Search + TF–IDF Keyword Search  

---

## 📖 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG) system** for answering questions from **Sanskrit classical texts**.  

Instead of using an online LLM API, the system uses a **completely offline local model (Qwen2.5–1.5B-Instruct)** along with:

- A local **knowledge base** (DOCX / PDF Sanskrit documents)
- Text **chunking** + preprocessing
- **Sentence Transformer embeddings**  
- **FAISS** vector similarity search  
- A Streamlit-based UI for querying the system  

The final pipeline performs:

1. **Document loading** (DOCX and PDF supported)  
2. **Chunking & preprocessing**  
3. **Embedding generation**  
4. **Vector + keyword retrieval**  
5. **LLM-based answer generation using only retrieved context**  

This ensures that the LLM **does not hallucinate** and answers only from provided Sanskrit sources.

---

## 🔥 Features

✔ Fully offline RAG system  
✔ Accurate chunk-based retrieval  
✔ Sanskrit-compatible DOCX loader  
✔ Streamlit UI for interactive querying  
✔ Supports vector + TF-IDF keyword search  
✔ Lightweight Qwen model runs on CPU  
✔ Works with long classical Sanskrit passages  
✔ Clean and modular code structure  

---

## 📁 Folder Structure

RAG_Sanskrit_Dhruva/
│
├── code/
│ ├── app.py # Streamlit UI
│ ├── rag_pipeline.py # Full RAG pipeline
│ ├── generator.py # Qwen2.5-1.5B answer generator
│ ├── loader.py # DOCX/PDF loading
│ ├── preprocessing.py # Chunking
│ ├── embedder.py # Embedding model (E5-small)
│ ├── retriever.py # FAISS & TF-IDF retrievers
│
├── models/
│ └── qwen1.5b/ # Local LLM files (NOT uploaded to GitHub)
│
├── data/
│ ├── raw/ # Input Sanskrit PDFs/DOCX
│ └── processed/
│ ├── chunks.json # Preprocessed text chunks
│ └── embeddings.npy # Vector embeddings
│
├── requirements.txt
└── README.md

yaml
Copy code

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/RAG_Sanskrit_Dhruva.git
cd RAG_Sanskrit_Dhruva
2️⃣ Create & activate virtual environment
bash
Copy code
python -m venv venv
venv\Scripts\activate
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
🧠 Download the Qwen Model (Required)
You MUST download the model locally before running the app.

Run inside the project folder:

bash
Copy code
hf download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir models/qwen1.5b \
    --include "*.json" "*.model" "*.safetensors"
⚠️ Do NOT upload the model files to GitHub.
They exceed size limits and are meant to remain local only.

🗂 Preparing the Knowledge Base
Place your Sanskrit files in:

bash
Copy code
data/raw/
Supported formats:

.docx (recommended — preserves Devanagari correctly)

.pdf

.txt

After placing files, delete old processed files:

bash
Copy code
data/processed/chunks.json
data/processed/embeddings.npy
They will be regenerated automatically.

🚀 Running the App
Run Streamlit:

bash
Copy code
python -m streamlit run code/app.py
Open in browser:

arduino
Copy code
http://localhost:8501
You will see:

Query input box

Retrieval method selector

Top-3 retrieved chunks

Generated answer from Qwen model

🧪 Example Query
Copy code
भोजराज्ञा कियद् धनं कवये दातुम् घोषितवान् ?
Expected Answer:

Copy code
भोजराज्ञा लक्षरुप्यकाणि दातुम् घोषितवान्।
🧬 Internal Architecture
1. Loader
Reads DOCX, PDF, TXT

Extracts Unicode Sanskrit text

2. Preprocessor
Cleans text

Splits into chunks (size 256–300 tokens)

3. Embedder
Uses intfloat/multilingual-e5-small

Generates dense embeddings for each chunk

4. Retriever
FAISS L2 search

TF-IDF fallback keyword search

5. Generator
Qwen2.5–1.5B-Instruct

Strict “context-only answering” prompt

Offline CPU inference
