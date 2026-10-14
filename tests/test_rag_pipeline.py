"""Offline tests for the RAG review pipeline.

No network or API keys: embeddings, the LLM and GitHub are replaced by small
deterministic fakes, so these tests check the pipeline's own logic (diff
parsing, chunking, indexing, retrieval, prompt budgeting, orchestration).
"""
import hashlib
import re

import numpy as np
import pytest

from review_engine.rag.chunker import chunk_file, detect_language
from review_engine.rag.diff_utils import parse_patch
from review_engine.rag.embeddings import EMBEDDING_DIM
from review_engine.rag.indexer import index_changed_files
from review_engine.rag.prompt_builder import build_review_prompt, count_tokens
from review_engine.rag.retriever import Retriever
from review_engine.rag.vector_store import RepoVectorStore
from review_engine.rag_reviewer import RAGReviewer


# ── Fakes ──────────────────────────────────────────────────────────────────────

class FakeEmbedder:
    """Bag-of-words hashing embedder: texts sharing identifiers are similar."""

    def embed(self, texts):
        out = np.zeros((len(texts), EMBEDDING_DIM), dtype="float32")
        for i, t in enumerate(texts):
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % EMBEDDING_DIM
                out[i, h] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def embed_one(self, text):
        return self.embed([text])[0]


class FakeFetcher:
    def __init__(self, files, contents, contributing=None):
        self._files = files
        self._contents = contents
        self._contributing = contributing

    def get_pr_diff(self, pr_number):
        return self._files

    def get_pr_head_sha(self, pr_number):
        return "abc123"

    def get_file_content(self, path, ref):
        return self._contents.get(path)

    def get_contributing_notes(self):
        return self._contributing


class FakeAI:
    def __init__(self):
        self.prompts = []

    def review_prompt(self, prompt):
        self.prompts.append(prompt)
        return f"review #{len(self.prompts)}"


class NullTracker:
    def log_review(self, **kwargs):
        self.last = kwargs


# ── diff_utils ─────────────────────────────────────────────────────────────────

PATCH = """@@ -1,3 +1,4 @@ def a():
 x = 1
-y = 2
+y = 3
+z = 4
 w = 5
@@ -10 +11,2 @@
 keep
+added
"""


def test_parse_patch_hunks_and_line_ranges():
    hunks = parse_patch(PATCH)
    assert len(hunks) == 2
    h1, h2 = hunks
    assert (h1.old_start, h1.old_count, h1.new_start, h1.new_count) == (1, 3, 1, 4)
    assert h1.section_heading == "def a():"
    assert h1.new_end == 4
    assert h1.new_file_text() == "x = 1\ny = 3\nz = 4\nw = 5"
    # Omitted counts default to 1.
    assert (h2.old_start, h2.old_count, h2.new_start, h2.new_count) == (10, 1, 11, 2)
    assert h2.raw_text.startswith("@@ -10 +11,2 @@")


def test_parse_patch_empty_and_no_newline_marker():
    assert parse_patch("") == []
    hunks = parse_patch("@@ -1 +1 @@\n-a\n+b\n\\ No newline at end of file\n")
    assert hunks[0].new_file_text() == "b"


# ── chunker ────────────────────────────────────────────────────────────────────

PY_SOURCE = '''import os

CONST = 3


def alpha(x):
    return x + CONST


class Beta:
    def method_one(self):
        return 1

    def method_two(self):
        return 2
'''


def test_chunker_splits_python_at_function_and_method_boundaries():
    chunks = chunk_file("pkg/mod.py", PY_SOURCE)
    symbols = [c.symbol for c in chunks if c.kind == "function"]
    assert symbols == ["alpha", "method_one", "method_two"]
    alpha = next(c for c in chunks if c.symbol == "alpha")
    assert (alpha.start_line, alpha.end_line) == (6, 7)
    # Module-level code (imports/constants) is kept as its own chunk, not dropped.
    module_text = "\n".join(c.text for c in chunks if c.kind == "module")
    assert "import os" in module_text and "CONST = 3" in module_text


def test_chunker_falls_back_to_fixed_windows_for_unknown_types():
    text = "\n".join(f"line {i} of a plain text file" for i in range(300))
    chunks = chunk_file("notes.txt", text)
    assert detect_language("notes.txt") is None
    assert len(chunks) > 1
    assert all(c.kind == "fixed" for c in chunks)
    # Windows overlap and together cover the whole file.
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 300
    assert chunks[1].start_line <= chunks[0].end_line


def test_chunker_empty_file():
    assert chunk_file("a.py", "   \n\n") == []


# ── vector store + retriever ───────────────────────────────────────────────────

def _store(tmp_path):
    return RepoVectorStore("owner/repo", index_dir=str(tmp_path))


def test_vector_store_upsert_replaces_a_files_chunks_and_persists(tmp_path):
    emb = FakeEmbedder()
    store = _store(tmp_path)
    c1 = chunk_file("a.py", "def foo():\n    return 1\n")
    store.upsert_file("a.py", c1, emb.embed([c.text for c in c1]))
    assert store.index.ntotal == len(c1)

    c2 = chunk_file("a.py", "def bar():\n    return 2\n\ndef baz():\n    return 3\n")
    store.upsert_file("a.py", c2, emb.embed([c.text for c in c2]))
    assert store.index.ntotal == len(c2)  # old chunks removed, not duplicated
    store.save()

    reloaded = _store(tmp_path)
    assert reloaded.index.ntotal == len(c2)
    hits = reloaded.search(emb.embed_one("def bar return"), top_k=1)
    assert hits[0][0]["symbol"] == "bar"


def test_retriever_ranks_related_code_first_and_can_exclude_the_hunk_itself(tmp_path):
    emb = FakeEmbedder()
    store = _store(tmp_path)
    files = {
        "pay.py": "def compute_tax(amount):\n    return amount * TAX_RATE\n\ndef unrelated():\n    return 'x'\n",
        "cart.py": "def total(items):\n    return sum(compute_tax(i.amount) for i in items)\n",
    }
    for path, src in files.items():
        cs = chunk_file(path, src)
        store.upsert_file(path, cs, emb.embed([c.text for c in cs]))

    r = Retriever(store, emb, use_bm25_rerank=False)
    hits = r.retrieve("compute_tax amount TAX_RATE", top_k=2)
    assert hits[0]["symbol"] == "compute_tax"

    # Excluding the hunk's own lines leaves room for genuinely different context.
    hits = r.retrieve("compute_tax amount TAX_RATE", top_k=2, exclude=("pay.py", 1, 2))
    assert all(not (h["file_path"] == "pay.py" and h["start_line"] <= 2) for h in hits)
    assert hits[0]["file_path"] == "cart.py"


def test_bm25_rerank_path_runs(tmp_path):
    emb = FakeEmbedder()
    store = _store(tmp_path)
    cs = chunk_file("m.py", "def a():\n    return 1\n\ndef b():\n    return 2\n")
    store.upsert_file("m.py", cs, emb.embed([c.text for c in cs]))
    hits = Retriever(store, emb, use_bm25_rerank=True).retrieve("def a return", top_k=2)
    assert len(hits) == 2


# ── indexer ────────────────────────────────────────────────────────────────────

def test_indexer_drops_removed_files_and_renamed_files_old_path(tmp_path):
    emb = FakeEmbedder()
    store = _store(tmp_path)
    fetcher = FakeFetcher([], {"old.py": "def gone():\n    return 0\n", "keep.py": "def k():\n    return 1\n"})
    index_changed_files(store, emb, fetcher, "sha", [
        {"filename": "old.py", "status": "added"},
        {"filename": "keep.py", "status": "added"},
    ])
    assert "old.py" in store.file_chunk_ids

    fetcher = FakeFetcher([], {"new.py": "def gone():\n    return 0\n"})
    index_changed_files(store, emb, fetcher, "sha", [
        {"filename": "new.py", "status": "renamed", "previous_filename": "old.py"},
        {"filename": "keep.py", "status": "removed"},
    ])
    paths = {m["file_path"] for m in store.chunks.values()}
    assert paths == {"new.py"}  # no stale chunks under the old name or the removed file
    assert store.index.ntotal == len(store.chunks)


# ── prompt builder ─────────────────────────────────────────────────────────────

def test_prompt_respects_token_budget_and_keeps_rank_order():
    chunks = [
        {"file_path": f"f{i}.py", "start_line": 1, "end_line": 10, "symbol": f"s{i}",
         "text": "word " * 300}
        for i in range(10)
    ]
    prompt = build_review_prompt("x.py", "+ change", chunks, max_tokens=1000)
    assert count_tokens(prompt) <= 1000
    included = re.findall(r"\[f(\d+)\.py:", prompt)
    assert included and included == sorted(included)  # highest-ranked first
    assert len(included) < 10  # budget actually cut something


def test_prompt_includes_style_notes_and_tags():
    p = build_review_prompt("x.py", "+a", [{"file_path": "y.py", "start_line": 3, "end_line": 9, "text": "def y(): pass"}],
                            style_notes="Use snake_case.")
    assert "Use snake_case." in p and "[y.py:3-9]" in p


# ── end-to-end orchestration ───────────────────────────────────────────────────

def test_rag_reviewer_end_to_end_with_fakes(tmp_path):
    src = "def compute_tax(amount):\n    return amount * 0.18\n\n\ndef total(items):\n    return sum(items)\n"
    patch = "@@ -1,2 +1,2 @@\n def compute_tax(amount):\n-    return amount * 0.1\n+    return amount * 0.18\n"
    fetcher = FakeFetcher(
        [{"filename": "pay.py", "status": "modified", "patch": patch},
         {"filename": "gone.py", "status": "removed", "patch": ""}],
        {"pay.py": src},
        contributing="Prefer Decimal for money.",
    )
    ai, tracker = FakeAI(), NullTracker()
    reviewer = RAGReviewer(ai_reviewer=ai, embedder=FakeEmbedder(), tracker=tracker, index_dir=str(tmp_path))
    results = reviewer.review_pr(fetcher, "owner/repo", 7)

    assert len(results) == 1
    r = results[0]
    assert (r["filename"], r["start_line"], r["end_line"]) == ("pay.py", 1, 2)
    assert r["review_text"] == "review #1"
    assert "Prefer Decimal for money." in ai.prompts[0]
    # The hunk's own lines are not fed back to the model as "context".
    assert "pay.py:1-2" not in r["retrieved_context"]
    # Tracker receives a real token count, not a character-length estimate.
    assert tracker.last["prompt_tokens"] == count_tokens(ai.prompts[0])
