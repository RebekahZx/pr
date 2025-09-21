from flask import Flask, request, render_template
from review_engine.pr_fetcher import PRFetcher
from review_engine.ai_reviewer import AIReviewer
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Initialize AI reviewer with API key from .env
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

ai = AIReviewer(api_key=GEMINI_API_KEY)

@app.route("/", methods=["GET", "POST"])
def index():
    feedback = []
    error = None

    if request.method == "POST":
        repo_url = request.form.get("repo_url")
        pr_number = request.form.get("pr_number")

        # Basic validation
        if not repo_url or not pr_number:
            error = "Please provide both repository URL and PR number."
            return render_template("index.html", feedback=[], error=error)

        try:
            pr_number = int(pr_number)
        except ValueError:
            error = "PR number must be an integer."
            return render_template("index.html", feedback=[], error=error)

        try:
            # Initialize PR fetcher with GitHub token from .env
            GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
            if not GITHUB_TOKEN:
                raise RuntimeError("GITHUB_TOKEN not found in .env")
            
            fetcher = PRFetcher(repo_url, token=GITHUB_TOKEN)
            changes = fetcher.get_pr_diff(pr_number)

            # Review each file patch
            for c in changes:
                patch = c.get("patch", "")
                if patch.strip():
                    review_text = ai.review_patch(c["filename"], patch)
                    feedback.append({
                        "filename": c["filename"],
                        "review_text": review_text
                    })
        except Exception as e:
            error = f"Error fetching or reviewing PR: {str(e)}"

    return render_template("index.html", feedback=feedback, error=error)


if __name__ == "__main__":
    app.run(debug=True)
