from __future__ import annotations

from typing import List

from .chunker import chunk_file
from .embeddings import Embedder
from .vector_store import RepoVectorStore

# Skip indexing pathologically large files; they're rarely useful retrieval context
# and blow up embedding cost/latency for a single PR review.
MAX_INDEXABLE_CHARS = 200_000


def index_changed_files(
    store: RepoVectorStore,
    embedder: Embedder,
    fetcher,
    ref: str,
    changed_files: List[dict],
) -> None:
    """Chunk + embed + upsert the current (post-change) content of each changed file.

    Removed files are dropped from the index instead of re-embedded, and a
    renamed file's chunks under its previous path are dropped too, so retrieval
    never returns code from a path that no longer exists.
    """
    for f in changed_files:
        file_path = f["filename"]
        status = f.get("status", "modified")

        previous = f.get("previous_filename")
        if status == "renamed" and previous and previous != file_path:
            store.upsert_file(previous, [], embeddings=None)

        if status == "removed":
            store.upsert_file(file_path, [], embeddings=None)
            continue

        content = fetcher.get_file_content(file_path, ref)
        if content is None:
            continue
        if len(content) > MAX_INDEXABLE_CHARS:
            content = content[:MAX_INDEXABLE_CHARS]

        chunks = chunk_file(file_path, content)
        if not chunks:
            store.upsert_file(file_path, [], embeddings=None)
            continue

        embeddings = embedder.embed([c.text for c in chunks])
        store.upsert_file(file_path, chunks, embeddings)

    store.save()
