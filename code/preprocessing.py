import re
import unicodedata
from typing import List, Dict


def normalize_text(text: str) -> str:
    """
    Normalize Unicode and clean up extra whitespace.
    This is important for Devanagari Sanskrit text.
    """
    # Normalize unicode (handles combined characters properly)
    text = unicodedata.normalize("NFKC", text)

    # Replace Windows-style newlines etc.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove weird control characters (except newline and tab)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")

    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Strip trailing spaces on each line
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Remove empty leading/trailing lines
    text = text.strip()

    return text


def merge_lines(text: str) -> str:
    """
    Merge lines that are broken in the middle of sentences.
    We keep paragraph breaks where there are two or more newlines.
    """
    # Replace 2+ newlines with a paragraph separator token
    PAR_SEP = "<PAR_BREAK>"
    text = re.sub(r"\n{2,}", f"{PAR_SEP}", text)

    # For single newlines, assume it's a broken line and replace with space
    text = text.replace("\n", " ")

    # Restore paragraph breaks as double newlines
    text = text.replace(PAR_SEP, "\n\n")

    # Clean multiple spaces again
    text = re.sub(r" +", " ", text)

    return text.strip()


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs based on double newline.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def chunk_text(
    text: str,
    doc_id: str,
    max_words: int = 200,
    min_words: int = 30,
) -> List[Dict]:
    """
    Split a document into smaller chunks of roughly max_words,
    preserving paragraph boundaries as much as possible.
    """
    paragraphs = split_into_paragraphs(text)

    chunks: List[Dict] = []
    current_words: List[str] = []
    chunk_index = 0

    def flush_chunk():
        nonlocal chunk_index, current_words
        if len(current_words) == 0:
            return
        chunk_text_str = " ".join(current_words).strip()
        if not chunk_text_str:
            current_words = []
            return

        chunk_id = f"{doc_id}_{chunk_index:04d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": chunk_text_str,
                "order": chunk_index,
            }
        )
        chunk_index += 1
        current_words = []

    for para in paragraphs:
        words = para.split()
        # If a single paragraph is very long, break inside it
        for w in words:
            current_words.append(w)
            if len(current_words) >= max_words:
                flush_chunk()
        # Paragraph boundary: if current chunk has enough words, flush it
        if len(current_words) >= min_words:
            flush_chunk()

    # Flush trailing words
    flush_chunk()

    return chunks


def build_corpus_chunks(docs: List[Dict]) -> List[Dict]:
    """
    Apply normalization + merging + chunking to all documents.
    """
    all_chunks: List[Dict] = []
    for doc in docs:
        raw_text = doc["text"]
        norm = normalize_text(raw_text)
        merged = merge_lines(norm)
        doc_chunks = chunk_text(merged, doc_id=doc["doc_id"])
        all_chunks.extend(doc_chunks)
    return all_chunks


if __name__ == "__main__":
    # Simple sanity test
    sample = "अस्ति कदाचित्।\nइति कथा।\n\nइयं अपर कथा।"
    from pprint import pprint

    norm = normalize_text(sample)
    merged = merge_lines(norm)
    chunks = chunk_text(merged, doc_id="test_doc", max_words=5, min_words=1)
    pprint(chunks)
