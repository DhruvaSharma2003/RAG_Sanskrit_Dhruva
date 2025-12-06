import os
from typing import List, Dict

from pypdf import PdfReader


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
            pages_text.append(page.extract_text() or "")
        except Exception:
            # In case of any page extraction error, skip that page
            continue
    return "\n".join(pages_text)


def load_corpus(raw_dir: str = os.path.join("data", "raw")) -> List[Dict]:
    """
    Load all .txt and .pdf documents from the given directory.

    Returns a list of dicts:
    [
        {
            "doc_id": "filename_without_ext",
            "file_name": "original_file_name.pdf",
            "text": "full document text ..."
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

        if ext == ".txt":
            text = load_text_file(path)
        elif ext == ".pdf":
            text = load_pdf_file(path)
        else:
            # Ignore unsupported file types
            continue

        corpus.append(
            {
                "doc_id": name,
                "file_name": file_name,
                "text": text,
            }
        )

    return corpus


if __name__ == "__main__":
    # Quick manual test
    docs = load_corpus()
    print(f"Loaded {len(docs)} documents from data/raw/")
    for d in docs:
        print(f"- {d['file_name']}: {len(d['text'])} characters")
