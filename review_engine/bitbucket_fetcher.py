from atlassian import Bitbucket

class BitbucketFetcher:
    def __init__(self, repo_url, username, app_password):
        """
        repo_url example: https://bitbucket.org/workspace/repo_slug
        username & app_password: for authentication
        """
        parts = repo_url.rstrip("/").split("/")
        self.workspace = parts[-2]
        self.repo_slug = parts[-1]
        self.bb = Bitbucket(
            url="https://bitbucket.org",
            username=username,
            password=app_password
        )

    def get_pr_diff(self, pr_number):
        pr = self.bb.get_pull_request(self.workspace, self.repo_slug, pr_number)
        diff_list = []
        for diff_file in self.bb.get_pull_request_diff(self.workspace, self.repo_slug, pr_number).split("\n\n"):
            # Each diff block
            if "diff --git" in diff_file:
                filename_line = diff_file.split("\n")[0]
                filename = filename_line.split(" b/")[-1]
                diff_list.append({
                    "filename": filename,
                    "patch": diff_file
                })
        return diff_list
