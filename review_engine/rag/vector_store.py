from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from typing import Dict, List, Tuple

import faiss
import numpy as np

from .chunker import Chunk
from .embeddings import EMBEDDING_DIM

_LOCK = threading.Lock()


def _default_index_dir() -> str:
    # Vercel's filesystem is read-only outside /tmp; fall back there when deployed.
    if os.environ.get("VERCEL") or os.environ.get("RAG_INDEX_DIR") == "/tmp":
        return "/tmp/rag_index"
    return os.environ.get("RAG_INDEX_DIR", os.path.join(os.getcwd(), ".rag_index"))


def _repo_key(repo_full_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "__", repo_full_name)


def _stable_id(file_path: str, start_line: int, end_line: int, text: str) -> int:
    digest = hashlib.blake2b(
        f"{file_path}:{start_line}:{end_line}:{text}".encode("utf-8"), digest_size=8
    ).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return value % (2 ** 62)  # keep well within signed int64 range for faiss ids


class RepoVectorStore:
    """A local, per-repo FAISS index (cosine similarity via normalized inner product).

    No hosted vector DB here on purpose: a single repo's changed-file chunks fit
    comfortably in a local IndexFlatIP, and the added ops surface of Pinecone/Weaviate
    buys nothing at this scale.

    Persisted as two files per repo: `<repo>.faiss` (the index) and `<repo>.json`
    (chunk metadata + which chunk ids currently belong to which file, so a file's
    chunks can be replaced atomically on re-index).
    """

    def __init__(self, repo_full_name: str, index_dir: str = None):
        self.repo_full_name = repo_full_name
        self.index_dir = index_dir or _default_index_dir()
        os.makedirs(self.index_dir, exist_ok=True)

        key = _repo_key(repo_full_name)
        self._index_path = os.path.join(self.index_dir, f"{key}.faiss")
        self._meta_path = os.path.join(self.index_dir, f"{key}.json")

        self.index: faiss.IndexIDMap2 = None
        self.chunks: Dict[str, dict] = {}  # str(id) -> chunk metadata
        self.file_chunk_ids: Dict[str, List[int]] = {}  # file_path -> [ids]
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self.index = faiss.read_index(self._index_path)
            with open(self._meta_path, "r") as f:
                meta = json.load(f)
            self.chunks = meta.get("chunks", {})
            self.file_chunk_ids = meta.get("file_chunk_ids", {})
        else:
            base = faiss.IndexFlatIP(EMBEDDING_DIM)
            self.index = faiss.IndexIDMap2(base)
            self.chunks = {}
            self.file_chunk_ids = {}

    def save(self) -> None:
        with _LOCK:
            faiss.write_index(self.index, self._index_path)
            with open(self._meta_path, "w") as f:
                json.dump({"chunks": self.chunks, "file_chunk_ids": self.file_chunk_ids}, f)

    def upsert_file(self, file_path: str, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Replace all indexed chunks for `file_path` with the given chunks+embeddings."""
        with _LOCK:
            old_ids = self.file_chunk_ids.get(file_path, [])
            if old_ids:
                self.index.remove_ids(np.array(old_ids, dtype="int64"))
                for old_id in old_ids:
                    self.chunks.pop(str(old_id), None)

            if not chunks:
                self.file_chunk_ids[file_path] = []
                return

            ids = [_stable_id(c.file_path, c.start_line, c.end_line, c.text) for c in chunks]
            self.index.add_with_ids(embeddings, np.array(ids, dtype="int64"))

            for cid, chunk in zip(ids, chunks):
                self.chunks[str(cid)] = {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "text": chunk.text,
                    "kind": chunk.kind,
                    "symbol": chunk.symbol,
                    "language": chunk.language,
                }
            self.file_chunk_ids[file_path] = ids

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[dict, float]]:
        if self.index.ntotal == 0:
            return []
        query = query_embedding.reshape(1, -1).astype("float32")
        scores, ids = self.index.search(query, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            meta = self.chunks.get(str(idx))
            if meta is not None:
                results.append((meta, float(score)))
        return results
