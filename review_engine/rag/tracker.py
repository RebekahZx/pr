from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional


class ReviewTracker:
    """Logs each review call to Weights & Biases: prompt tokens, retrieved chunk ids,
    model used, latency. Gives a real eval trail instead of a vibe-based "it seems better".

    No-op if WANDB_API_KEY isn't set or the wandb import/init fails, so a missing
    tracking key never breaks a review.
    """

    def __init__(self, project: str = None, enabled: bool = None):
        self.project = project or os.environ.get("WANDB_PROJECT", "pr-reviewer-rag")
        if enabled is None:
            enabled = bool(os.environ.get("WANDB_API_KEY"))
        self.enabled = enabled
        self._run = None

        if self.enabled:
            try:
                import wandb

                self._wandb = wandb
                self._run = wandb.init(project=self.project, job_type="pr_review", reinit=True)
            except Exception:
                self.enabled = False

    def log_review(
        self,
        *,
        repo: str,
        pr_number: int,
        filename: str,
        model: str,
        prompt_tokens: int,
        retrieved_chunk_ids: list,
        latency_seconds: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            payload = {
                "repo": repo,
                "pr_number": pr_number,
                "filename": filename,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "num_retrieved_chunks": len(retrieved_chunk_ids),
                "latency_seconds": latency_seconds,
                "ts": time.time(),
            }
            if extra:
                payload.update(extra)
            self._wandb.log(payload)
        except Exception:
            pass  # tracking must never break a review

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
