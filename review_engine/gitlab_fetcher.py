import gitlab

class GitLabFetcher:
    def __init__(self, repo_url, token):
        # Extract project path from URL, e.g., https://gitlab.com/user/repo -> user/repo
        self.project_path = "/".join(repo_url.rstrip("/").split("/")[-2:])
        self.gl = gitlab.Gitlab('https://gitlab.com', private_token=token)
        self.project = self.gl.projects.get(self.project_path)

    def get_pr_diff(self, pr_number):
        mr = self.project.mergerequests.get(pr_number)
        diff_list = []
        for change in mr.changes()['changes']:
            diff_list.append({
                "filename": change['new_path'],
                "patch": change['diff']
            })
        return diff_list
