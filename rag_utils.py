from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"  # 256 token max, fast and lightweight


def load_model() -> SentenceTransformer:
    """Load the sentence-transformer embedding model."""
    return SentenceTransformer(MODEL_NAME)


def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 20) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text: Raw text to split.
        chunk_size: Number of words per chunk (kept under model's 256 token limit).
        chunk_overlap: Number of words shared between consecutive chunks.

    Returns:
        List of text chunk strings.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - chunk_overlap
    return chunks


def build_index(chunks: list[str], model: SentenceTransformer) -> faiss.IndexFlatL2:
    """
    Embed all chunks and build a FAISS L2 index.

    Args:
        chunks: List of text chunks to index.
        model: Loaded SentenceTransformer model.

    Returns:
        Populated FAISS index.
    """
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: list[str],
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[str]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        query: User's question.
        index: FAISS index built from document chunks.
        chunks: Original chunk strings (parallel to index entries).
        model: Loaded SentenceTransformer model.
        top_k: Number of chunks to return.

    Returns:
        List of the most relevant chunk strings.
    """
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, min(top_k, len(chunks)))
    return [chunks[i] for i in indices[0] if i < len(chunks)]
