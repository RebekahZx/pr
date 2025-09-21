from github import Github
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
                "status": f.status
            })
        return diff_list
