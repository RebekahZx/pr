import hashlib
import hmac
import os

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

from review_engine.ai_reviewer import AIReviewer
from review_engine.github_fetcher import GitHubFetcher
from review_engine.rag.embeddings import Embedder
from review_engine.rag.tracker import ReviewTracker
from review_engine.rag_reviewer import RAGReviewer

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env (needed for retrieval embeddings)")

GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET")

# Shared across requests within a warm serverless instance: the LLM client, the
# embedding client, and the W&B tracker are all cheap to reuse and expensive to
# recreate per-request. The FAISS index itself is opened per-repo inside RAGReviewer.
_ai_reviewer = AIReviewer(api_key=GEMINI_API_KEY)
_embedder = Embedder(api_key=OPENAI_API_KEY)
_tracker = ReviewTracker()
_rag_reviewer = RAGReviewer(ai_reviewer=_ai_reviewer, embedder=_embedder, tracker=_tracker)


def _group_by_file(results):
    grouped = {}
    for r in results:
        grouped.setdefault(r["filename"], []).append(r)
    feedback = []
    for filename, hunks in grouped.items():
        review_text = "\n\n".join(
            f"[lines {h['start_line']}-{h['end_line']}]"
            + (f" (context: {', '.join(h['retrieved_context'])})" if h["retrieved_context"] else "")
            + f"\n{h['review_text']}"
            for h in hunks
        )
        feedback.append({"filename": filename, "review_text": review_text})
    return feedback


@app.route("/", methods=["GET", "POST"])
def index():
    feedback = []
    error = None

    if request.method == "POST":
        repo_url = request.form.get("repo_url")
        pr_number = request.form.get("pr_number")

        if not repo_url or not pr_number:
            error = "Please provide both repository URL and PR number."
            return render_template("index.html", feedback=[], error=error)

        try:
            pr_number = int(pr_number)
        except ValueError:
            error = "PR number must be an integer."
            return render_template("index.html", feedback=[], error=error)

        try:
            github_token = os.environ.get("GITHUB_TOKEN")
            if not github_token:
                raise RuntimeError("GITHUB_TOKEN not found in .env")

            fetcher = GitHubFetcher(repo_url, token=github_token)
            results = _rag_reviewer.review_pr(fetcher, fetcher.repo_name, pr_number)
            feedback = _group_by_file(results)

        except Exception as e:
            error = f"Error fetching or reviewing PR: {str(e)}"

    return render_template("index.html", feedback=feedback, error=error)


def _verify_github_signature(raw_body: bytes) -> bool:
    if not GITHUB_WEBHOOK_SECRET:
        # No secret configured: refuse to trust unauthenticated webhook traffic.
        return False
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not signature.startswith("sha256="):
        return False
    expected = hmac.new(GITHUB_WEBHOOK_SECRET.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


@app.route("/webhook/github", methods=["POST"])
def github_webhook():
    raw_body = request.get_data()
    if not _verify_github_signature(raw_body):
        return jsonify({"error": "invalid signature"}), 401

    event = request.headers.get("X-GitHub-Event", "")
    payload = request.get_json(silent=True) or {}

    if event != "pull_request" or payload.get("action") not in ("opened", "synchronize", "reopened"):
        return jsonify({"status": "ignored"}), 200

    repo_url = payload["repository"]["html_url"]
    repo_full_name = payload["repository"]["full_name"]
    pr_number = payload["pull_request"]["number"]

    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        return jsonify({"error": "GITHUB_TOKEN not configured"}), 500

    try:
        fetcher = GitHubFetcher(repo_url, token=github_token)
        results = _rag_reviewer.review_pr(fetcher, repo_full_name, pr_number)

        comments = [
            {"path": r["filename"], "line": r["end_line"], "body": r["review_text"]}
            for r in results
            if r["end_line"] > 0
        ]
        summary = f"Automated RAG review: {len(comments)} comment(s) across {len(set(c['path'] for c in comments))} file(s)."
        fetcher.post_review(pr_number, comments, summary_body=summary)

        return jsonify({"status": "reviewed", "comments_posted": len(comments)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For development only. Use Gunicorn/Waitress for production.
    app.run(debug=True)
