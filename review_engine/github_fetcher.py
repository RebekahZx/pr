from github import Github, GithubException
from .base_fetcher import BasePRFetcher


class GitHubFetcher(BasePRFetcher):
    def __init__(self, repo_url, token=None):
        # Extract repo name from URL
        self.repo_name = "/".join(repo_url.rstrip("/").split("/")[-2:])
        self.token = token
        self.gh = Github(self.token) if token else Github()
        self.repo = self.gh.get_repo(self.repo_name)

    def get_pr_diff(self, pr_number):
        pr = self.repo.get_pull(pr_number)
        files = pr.get_files()
        diff_list = []
        for f in files:
            diff_list.append({
                "filename": f.filename,
                "patch": f.patch or "",
                "additions": f.additions,
                "deletions": f.deletions,
                "status": f.status,
                # Set by GitHub for renamed files; lets the indexer drop the old path.
                "previous_filename": getattr(f, "previous_filename", None),
            })
        return diff_list

    def get_pr_head_sha(self, pr_number):
        return self.repo.get_pull(pr_number).head.sha

    def get_file_content(self, path, ref):
        """Full post-change file content at `ref`, used for AST chunking/indexing.
        Returns None for binary/undecodable/missing files rather than raising."""
        try:
            content_file = self.repo.get_contents(path, ref=ref)
            if isinstance(content_file, list):
                return None  # path is a directory
            return content_file.decoded_content.decode("utf-8", errors="replace")
        except GithubException:
            return None
        except (UnicodeDecodeError, AttributeError):
            return None

    def get_contributing_notes(self):
        """Repo-level style notes from CONTRIBUTING.md on the default branch, if present."""
        for candidate in ("CONTRIBUTING.md", ".github/CONTRIBUTING.md", "docs/CONTRIBUTING.md"):
            try:
                content_file = self.repo.get_contents(candidate)
                if isinstance(content_file, list):
                    continue
                return content_file.decoded_content.decode("utf-8", errors="replace")
            except GithubException:
                continue
        return None

    def post_review(self, pr_number, comments, summary_body=""):
        """Post an inline PR review. `comments` is a list of
        {"path": ..., "line": ..., "body": ...} for the new-file line numbers."""
        pr = self.repo.get_pull(pr_number)
        commit = self.repo.get_commit(pr.head.sha)

        review_comments = [
            {"path": c["path"], "line": c["line"], "body": c["body"]}
            for c in comments
            if c.get("body")
        ]

        if not review_comments and not summary_body:
            return None

        return pr.create_review(
            commit=commit,
            body=summary_body or "Automated review",
            event="COMMENT",
            comments=review_comments,
        )
