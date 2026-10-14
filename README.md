# PR Review Agent (RAG)

[![tests](https://github.com/RebekahZx/pr/actions/workflows/test.yml/badge.svg)](https://github.com/RebekahZx/pr/actions/workflows/test.yml)

An AI PR reviewer that grounds its comments in the actual codebase, not just the
raw diff. For each changed file it chunks the code at function/class boundaries
(via `tree-sitter`), embeds and indexes those chunks per repo (local FAISS), then
for every diff hunk retrieves the most relevant existing code as context before
asking the LLM to review it.

## Architecture

```
GitHub webhook (PR opened/synchronize/reopened)
        |
        v
  Fetch changed files + diff (GitHub API)
        |
        v
  For each changed file:
    - Chunk file at function/class boundaries (tree-sitter; fixed-size token
      windows as a fallback for unsupported languages)
    - Embed chunks (OpenAI text-embedding-3-small)
    - Upsert into a local FAISS index, namespaced per repo
        |
        v
  For each diff hunk:
    - Embed the hunk itself
    - FAISS top-k=6 by cosine similarity (top-20 pool, optional BM25 lexical
      rerank as a tie-breaker)
        |
        v
  Build review prompt: diff hunk + retrieved chunks (tagged file:line-range) +
  CONTRIBUTING.md notes if present, capped at ~4k tokens
        |
        v
  Call Gemini (gemini-2.5-flash)
        |
        v
  Post inline review comments to the GitHub PR
        |
        v
  Log the review event to Weights & Biases (optional) for offline eval
```

Code: the pipeline lives in [review_engine/rag/](review_engine/rag/)
(`chunker.py`, `embeddings.py`, `vector_store.py`, `retriever.py`, `indexer.py`,
`prompt_builder.py`, `diff_utils.py`, `tracker.py`), orchestrated by
[review_engine/rag_reviewer.py](review_engine/rag_reviewer.py). The Flask app in
[api/index.py](api/index.py) exposes both a manual web form (`/`) and a GitHub
webhook (`/webhook/github`) that use the same reviewer.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | yes | LLM that writes the review comments |
| `OPENAI_API_KEY` | yes | Embeddings for retrieval (`text-embedding-3-small`) |
| `GITHUB_TOKEN` | yes | Reads PR diffs/files; posts review comments |
| `GITHUB_WEBHOOK_SECRET` | for webhook use | Verifies `/webhook/github` requests |
| `WANDB_API_KEY` / `WANDB_PROJECT` | optional | Enables review-call tracking |
| `RAG_INDEX_DIR` | optional | Where per-repo FAISS indexes are stored |
| `RAG_USE_BM25_RERANK` | optional | `"false"` to disable the BM25 rerank step |

## Running locally

```bash
python run_server.py        # http://0.0.0.0:5000, via waitress
# or, for development with auto-reload:
python api/index.py
```

Open `http://localhost:5000`, paste a GitHub repo URL and PR number, and it
reviews the PR through the same RAG pipeline as the webhook path (minus posting
comments back to GitHub).

## Deploying (Vercel)

This repo already ships a `vercel.json` targeting `api/index.py`.

```bash
vercel deploy --prod
```

Then, in the Vercel project settings, add the same environment variables listed
above. After deploying, point a GitHub webhook (repo Settings → Webhooks) at
`https://<your-vercel-domain>/webhook/github`, content type
`application/json`, secret matching `GITHUB_WEBHOOK_SECRET`, and subscribe to
the "Pull requests" event.

**Known limitation:** Vercel's filesystem is read-only outside `/tmp`, and
`/tmp` doesn't persist across deployments or cold starts. The FAISS index will
still be rebuilt correctly on each webhook call (indexing runs before
retrieval on every review), but it won't carry over between invocations the
way it would on a long-running server — so today every review effectively
re-embeds the PR's changed files. Fine for a demo/low-traffic repo; for
heavier use, back `RAG_INDEX_DIR` with a persistent volume or move indexing to
a long-running worker instead of a serverless function.

## Tests

```bash
pip install pytest
python -m pytest -q tests
```

The suite runs offline: embeddings, the LLM and GitHub are replaced by small
deterministic fakes, so it checks the pipeline's own logic without API keys:

- diff parsing (hunk ranges, omitted counts, "no newline" markers)
- AST chunking at function/method boundaries, module-level code kept, and the
  fixed-window fallback for unsupported file types
- FAISS index upsert/replace/persist, retrieval ranking, BM25 rerank path
- removed and renamed files leave no stale chunks in the index
- prompt token budget (highest-ranked context first, never over budget)
- end-to-end review of a PR with fake fetcher/embedder/LLM
- webhook HMAC verification (missing/wrong signature rejected; no secret
  configured ⇒ all webhooks refused)

CI runs the suite on every push and pull request.

## Design notes

- **The hunk under review is excluded from its own context.** Changed files are
  indexed before retrieval, so the closest match for a hunk is usually the hunk
  itself. Chunks overlapping the hunk's lines are filtered out, leaving all
  `top_k` slots for other code (callers, related functions).
- **Renames don't leave stale context.** When GitHub reports a file as renamed,
  its chunks under the previous path are removed from the index.
- **Token counts are real counts.** Tracking logs the prompt's `cl100k` token
  count, the same tokenizer used for the prompt budget.

## Known limitations

- The webhook handler reviews synchronously. GitHub gives up waiting on a
  webhook delivery after about 10 seconds, so large PRs show as a failed
  delivery even when the review completes. A queue/worker would fix this.
- GitHub rejects a whole review if any inline comment targets a line outside
  the diff; comments are anchored to the hunk's last new-file line to stay
  inside it.

## Offline eval / tracking

If `WANDB_API_KEY` is set, every review call logs model, prompt tokens,
retrieved chunk ids, and latency to Weights & Biases (project
`WANDB_PROJECT`, default `pr-reviewer-rag`) — a real trail for judging
retrieval/review quality over time instead of going on vibes.
