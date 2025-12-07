import os
from typing import List, Dict
from docx import Document   # proper DOCX loader (no corruption)


# -----------------------------------------------------------
# DOCX Loader (Sanskrit-safe)
# -----------------------------------------------------------

def load_docx_file(path: str) -> str:
    """
    Cleanly extract Unicode text from a .docx file (ideal for Sanskrit).
    Removes blank lines & prevents encoding corruption.
    """
    doc = Document(path)

    paragraphs = []
    for p in doc.paragraphs:
        line = p.text.strip()
        if line:
            paragraphs.append(line)

    # Join with newline for clean chunking
    return "\n".join(paragraphs)


# -----------------------------------------------------------
# TXT Loader
# -----------------------------------------------------------

def load_text_file(path: str) -> str:
    """Load a UTF-8 text file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return text


# -----------------------------------------------------------
# Corpus Loader
# -----------------------------------------------------------

def load_corpus(raw_dir: str = os.path.join("data", "raw")) -> List[Dict]:
    """
    Load all .txt and .docx Sanskrit files.
    We DO NOT use PDFs for this project because PDFs corrupt Sanskrit text.
    """

    corpus = []

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    for file_name in os.listdir(raw_dir):
        path = os.path.join(raw_dir, file_name)
        if not os.path.isfile(path):
            continue

        name, ext = os.path.splitext(file_name)
        ext = ext.lower()

        # Supported types
        if ext == ".txt":
            text = load_text_file(path)

        elif ext == ".docx":
            text = load_docx_file(path)

        else:
            print(f"Skipping unsupported file: {file_name}")
            continue

        # Skip empty extractions
        if not text.strip():
            print(f"⚠ Warning: Empty text extracted from {file_name}")
            continue

        corpus.append({
            "doc_id": name,
            "file_name": file_name,
            "text": text,
        })

    print(f"✓ Loaded {len(corpus)} clean Sanskrit documents from {raw_dir}")
    return corpus


# -----------------------------------------------------------
# Manual test
# -----------------------------------------------------------

if __name__ == "__main__":
    docs = load_corpus()
    print(f"\nLoaded {len(docs)} documents:\n")
    for d in docs:
        print(f"- {d['file_name']}: {len(d['text'])} characters")
