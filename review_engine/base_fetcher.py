from abc import ABC, abstractmethod

class BasePRFetcher(ABC):
    @abstractmethod
    def get_pr_diff(self, pr_number):
        """
        Fetches PR files and diffs.
        Returns a list of dicts with keys: filename, patch, additions, deletions, status
        """
        pass
