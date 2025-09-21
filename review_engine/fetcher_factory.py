from review_engine.pr_fetcher import PRFetcher
from review_engine.gitlab_fetcher import GitLabFetcher
from review_engine.bitbucket_fetcher import BitbucketFetcher

def get_fetcher(repo_url, token=None):
    if "github.com" in repo_url:
        return PRFetcher(repo_url, token)
    elif "gitlab.com" in repo_url:
        return GitLabFetcher(repo_url, token)
    elif "bitbucket.org" in repo_url:
        return BitbucketFetcher(repo_url, token)
    else:
        raise ValueError("Unsupported git server")
