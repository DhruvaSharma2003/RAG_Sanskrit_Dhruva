import os
from typing import List, Dict

from pypdf import PdfReader
from docx import Document   # <-- Added for DOCX support


# -----------------------------------------------------------
# Loaders for different file types
# -----------------------------------------------------------

def load_text_file(path: str) -> str:
    """Load a UTF-8 text file and return its contents as a string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf_file(path: str) -> str:
    """Load a PDF file and extract text from all pages as a single string."""
    reader = PdfReader(path)
    pages_text = []

    for page in reader.pages:
        try:
            extracted = page.extract_text()
            if extracted:
                pages_text.append(extracted)
        except Exception:
            continue

    return "\n".join(pages_text)


def load_docx_file(path: str) -> str:
    """Load a .docx Sanskrit file and extract all paragraphs."""
    try:
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"⚠ DOCX parsing failed ({path}): {e}")
        return ""


# -----------------------------------------------------------
# Corpus Loader
# -----------------------------------------------------------

def load_corpus(raw_dir: str = os.path.join("data", "raw")) -> List[Dict]:
    """
    Load all .txt, .pdf, .docx documents from the given directory.

    Returns a list of dicts:
    [
        {
            "doc_id": "filename_without_ext",
            "file_name": "original_file_name.ext",
            "text": "full extracted document text ..."
        },
        ...
    ]
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

        # ---------------------------------------------------
        # Supported file types
        # ---------------------------------------------------
        if ext == ".txt":
            text = load_text_file(path)

        elif ext == ".pdf":
            text = load_pdf_file(path)

        elif ext == ".docx":
            text = load_docx_file(path)

        else:
            print(f"Skipping unsupported file: {file_name}")
            continue

        corpus.append({
            "doc_id": name,
            "file_name": file_name,
            "text": text,
        })

    print(f"✓ Loaded {len(corpus)} documents from {raw_dir}")
    return corpus


# -----------------------------------------------------------
# Manual test
# -----------------------------------------------------------
if __name__ == "__main__":
    docs = load_corpus()
    print(f"\nLoaded {len(docs)} documents:\n")
    for d in docs:
        print(f"- {d['file_name']}: {len(d['text'])} characters")
