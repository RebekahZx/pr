from __future__ import annotations

from typing import List, Optional

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _ENCODER = None

MAX_PROMPT_TOKENS = 4000
INSTRUCTIONS = (
    "You are reviewing a single diff hunk from a pull request. Use the retrieved "
    "context chunks (existing code from the same repository) to judge correctness, "
    "consistency with existing patterns, and whether the change might break a caller "
    "or contract defined elsewhere. Cite retrieved context by its file:line-range tag "
    "when it's relevant to a comment. If nothing in the retrieved context is relevant, "
    "say so briefly and review the hunk on its own merits. Be specific and concise; "
    "skip generic praise."
)


def _token_len(text: str) -> int:
    if _ENCODER is None:
        return max(1, len(text) // 4)
    return len(_ENCODER.encode(text, disallowed_special=()))


def build_review_prompt(
    filename: str,
    hunk_text: str,
    retrieved_chunks: List[dict],
    style_notes: Optional[str] = None,
    max_tokens: int = MAX_PROMPT_TOKENS,
) -> str:
    """Assemble a token-budgeted review prompt: instructions + style notes + hunk +
    as many retrieved context chunks (highest-ranked first) as fit under the budget.
    """
    header = f"{INSTRUCTIONS}\n\nFile under review: {filename}\n\nDiff hunk:\n```\n{hunk_text}\n```\n"
    budget = max_tokens - _token_len(header)

    style_block = ""
    if style_notes:
        style_block = f"\nRepository style notes (CONTRIBUTING.md):\n{style_notes.strip()}\n"
        budget -= _token_len(style_block)

    context_blocks = []
    for chunk in retrieved_chunks:
        tag = f"{chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']}"
        symbol = f" ({chunk['symbol']})" if chunk.get("symbol") else ""
        block = f"\n--- Retrieved context [{tag}]{symbol} ---\n{chunk['text']}\n"
        block_tokens = _token_len(block)
        if block_tokens > budget:
            continue
        context_blocks.append(block)
        budget -= block_tokens

    context_section = ("\nRetrieved context from the repository:\n" + "".join(context_blocks)) if context_blocks else ""

    return header + style_block + context_section
