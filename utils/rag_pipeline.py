"""
RAG Pipeline - FAISS + Sentence Transformers for semantic retrieval
Supports local model path to avoid network download issues.
"""

import os
import logging
import re
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Local model path (place downloaded model files here)
LOCAL_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "all-MiniLM-L6-v2"
)


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline using:
    - Sentence Transformers for dense embeddings
    - FAISS for fast similarity search
    - Sliding window chunking with overlap
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
    ):
        self.chunk_size = chunk_size        # ← was missing, caused the error
        self.chunk_overlap = chunk_overlap  # ← was missing
        self.chunks: List[str] = []
        self.index = None
        self.embedder = None
        self._load_embedder()

    def _load_embedder(self):
        """Load sentence transformer model — local first, then remote."""
        from sentence_transformers import SentenceTransformer

        if os.path.exists(LOCAL_MODEL_PATH):
            logger.info(f"Loading model from local path: {LOCAL_MODEL_PATH}")
            self.embedder = SentenceTransformer(LOCAL_MODEL_PATH)
        else:
            logger.info("Local model not found, attempting download...")
            self.embedder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks by word count."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if len(chunk.strip()) > 30:
                chunks.append(chunk.strip())
            start += self.chunk_size - self.chunk_overlap

        logger.info(f"Created {len(chunks)} chunks from {len(words)} words")
        return chunks

    def build_index(self, text: str):
        """Embed chunks and build FAISS index."""
        import faiss

        self.chunks = self._chunk_text(text)

        if not self.chunks:
            raise ValueError("No valid chunks extracted from text")

        logger.info("Generating embeddings...")
        embeddings = self.embedder.encode(
            self.chunks,
            show_progress_bar=False,
            batch_size=32,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def retrieve(self, query: str, k: int = 4) -> List[str]:
        """Retrieve top-k relevant chunks for a query."""
        if self.index is None or not self.chunks:
            logger.warning("Index not built yet.")
            return []

        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embedding = np.array(query_embedding, dtype=np.float32)

        k = min(k, len(self.chunks))
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and score > 0.1:
                results.append(self.chunks[idx])

        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:50]}'")
        return results