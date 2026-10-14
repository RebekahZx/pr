from __future__ import annotations

import time
from typing import List, Optional

from .ai_reviewer import AIReviewer
from .rag.diff_utils import parse_patch
from .rag.embeddings import Embedder
from .rag.indexer import index_changed_files
from .rag.prompt_builder import build_review_prompt, count_tokens
from .rag.retriever import Retriever
from .rag.tracker import ReviewTracker
from .rag.vector_store import RepoVectorStore

MIN_HUNK_CHARS = 10  # skip near-empty hunks (e.g. a single blank-line change)


class RAGReviewer:
    """Orchestrates the RAG review pipeline for one PR:
    index changed files -> retrieve context per diff hunk -> prompt the LLM -> track.
    """

    def __init__(
        self,
        ai_reviewer: Optional[AIReviewer] = None,
        embedder: Optional[Embedder] = None,
        tracker: Optional[ReviewTracker] = None,
        index_dir: Optional[str] = None,
    ):
        self.ai_reviewer = ai_reviewer or AIReviewer()
        self.embedder = embedder or Embedder()
        self.tracker = tracker or ReviewTracker()
        self.index_dir = index_dir

    def review_pr(self, fetcher, repo_full_name: str, pr_number: int) -> List[dict]:
        changed_files = fetcher.get_pr_diff(pr_number)
        head_sha = fetcher.get_pr_head_sha(pr_number)

        store = RepoVectorStore(repo_full_name, index_dir=self.index_dir)
        index_changed_files(store, self.embedder, fetcher, head_sha, changed_files)
        retriever = Retriever(store, self.embedder)

        style_notes = None
        if hasattr(fetcher, "get_contributing_notes"):
            style_notes = fetcher.get_contributing_notes()

        results: List[dict] = []

        for f in changed_files:
            if f.get("status") == "removed":
                continue
            filename = f["filename"]
            patch = f.get("patch") or ""
            if not patch.strip():
                continue

            for hunk in parse_patch(patch):
                query_text = hunk.new_file_text()
                if len(query_text.strip()) < MIN_HUNK_CHARS:
                    continue

                start_time = time.monotonic()
                retrieved = retriever.retrieve(query_text, exclude=(filename, hunk.new_start, hunk.new_end))
                prompt = build_review_prompt(filename, hunk.raw_text, retrieved, style_notes)
                review_text = self.ai_reviewer.review_prompt(prompt)
                latency = time.monotonic() - start_time

                retrieved_chunk_ids = [
                    f"{c['file_path']}:{c['start_line']}-{c['end_line']}" for c in retrieved
                ]
                self.tracker.log_review(
                    repo=repo_full_name,
                    pr_number=pr_number,
                    filename=filename,
                    model="gemini-2.5-flash",
                    prompt_tokens=count_tokens(prompt),
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    latency_seconds=latency,
                )

                results.append({
                    "filename": filename,
                    "start_line": hunk.new_start,
                    "end_line": hunk.new_end,
                    "review_text": review_text,
                    "retrieved_context": retrieved_chunk_ids,
                })

        return results
