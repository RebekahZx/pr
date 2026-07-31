from __future__ import annotations

import os
from typing import List

import numpy as np
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
_BATCH_SIZE = 96  # OpenAI embeddings endpoint handles batched inputs well below this


class Embedder:
    """Thin wrapper around OpenAI's text-embedding-3-small for chunk/hunk embeddings.

    Deliberately not a bigger/fancier embedding model: at this scale (one repo's
    changed files per review) a bigger model doesn't move retrieval quality enough
    to justify the extra cost and latency.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set!")
        self.client = OpenAI(api_key=self.api_key)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts, returning an (N, EMBEDDING_DIM) float32 array,
        L2-normalized so cosine similarity == inner product."""
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype="float32")

        vectors: List[List[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i:i + _BATCH_SIZE]
            # The API rejects empty strings; substitute a placeholder token.
            batch = [t if t.strip() else " " for t in batch]
            response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            vectors.extend(item.embedding for item in response.data)

        arr = np.array(vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
