from __future__ import annotations

import os
import re
from typing import List

from .embeddings import Embedder
from .vector_store import RepoVectorStore

TOP_K = 6
CANDIDATE_POOL = 20
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_rerank(query: str, candidates: List[dict]) -> List[dict]:
    """Lexical tie-breaker on top of cosine similarity, over the FAISS candidate pool."""
    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        return candidates  # BM25 rerank is optional; skip gracefully if unavailable

    corpus = [_tokenize(c["meta"]["text"]) for c in candidates]
    if not any(corpus):
        return candidates

    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(_tokenize(query))

    cos_scores = [c["cosine"] for c in candidates]
    cos_min, cos_max = min(cos_scores), max(cos_scores)
    bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)

    def norm(v, lo, hi):
        return (v - lo) / (hi - lo) if hi > lo else 0.0

    combined = []
    for c, bm25_score in zip(candidates, bm25_scores):
        blended = 0.7 * norm(c["cosine"], cos_min, cos_max) + 0.3 * norm(bm25_score, bm25_min, bm25_max)
        combined.append((blended, c))

    combined.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in combined]


class Retriever:
    def __init__(self, store: RepoVectorStore, embedder: Embedder, use_bm25_rerank: bool = None):
        self.store = store
        self.embedder = embedder
        if use_bm25_rerank is None:
            use_bm25_rerank = os.environ.get("RAG_USE_BM25_RERANK", "true").lower() != "false"
        self.use_bm25_rerank = use_bm25_rerank

    def retrieve(self, query_text: str, top_k: int = TOP_K) -> List[dict]:
        if not query_text.strip():
            return []

        query_embedding = self.embedder.embed_one(query_text)
        pool_size = CANDIDATE_POOL if self.use_bm25_rerank else top_k
        raw = self.store.search(query_embedding, top_k=pool_size)
        if not raw:
            return []

        candidates = [{"meta": meta, "cosine": score} for meta, score in raw]

        if self.use_bm25_rerank and len(candidates) > 1:
            candidates = _bm25_rerank(query_text, candidates)

        return [c["meta"] for c in candidates[:top_k]]
