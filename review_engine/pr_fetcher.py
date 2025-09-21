import os
from github import Github
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class PRFetcher:
    def __init__(self, repo_url, token=None):
        # Extract repo name from URL (e.g., https://github.com/user/repo -> user/repo)
        self.repo_name = "/".join(repo_url.rstrip("/").split("/")[-2:])

        # Use token from argument or fallback to environment variable
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub token not found. Please set it in .env or pass explicitly.")

        # Initialize Github client
        self.gh = Github(self.token)

        try:
            self.repo = self.gh.get_repo(self.repo_name)
        except Exception as e:
            raise RuntimeError(f"Failed to access repository '{self.repo_name}': {e}")

    def get_pr_diff(self, pr_number):
        try:
            pr = self.repo.get_pull(pr_number)
            files = pr.get_files()
            diff_list = []

            for f in files:
                diff_list.append({
                    "filename": f.filename,
                    "patch": f.patch or ""  # the code changes
                })

            return diff_list
        except Exception as e:
            raise RuntimeError(f"Failed to fetch PR #{pr_number}: {e}")
