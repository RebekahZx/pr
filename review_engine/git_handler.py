from github import Github

class GitHandler:
    def __init__(self, token, repo_name):
        self.g = Github(token)
        self.repo = self.g.get_repo(repo_name)

    def get_pr_diff(self, pr_number):
        pr = self.repo.get_pull(pr_number)
        return pr.get_files()  # list of changed files
