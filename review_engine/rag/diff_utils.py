from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section_heading: str
    lines: List[str] = field(default_factory=list)  # raw lines incl. +/-/space prefix
    raw_text: str = ""

    @property
    def new_end(self) -> int:
        return self.new_start + max(self.new_count - 1, 0)

    def new_file_text(self) -> str:
        """Reconstruct the post-change code visible in this hunk (context + additions)."""
        out = []
        for line in self.lines:
            if line.startswith("+") or line.startswith(" "):
                out.append(line[1:])
        return "\n".join(out)


def parse_patch(patch_text: str) -> List[Hunk]:
    """Parse a unified-diff `patch` string (as returned by the GitHub API) into hunks."""
    if not patch_text:
        return []

    hunks: List[Hunk] = []
    current: Optional[Hunk] = None
    raw_lines: List[str] = []

    for line in patch_text.splitlines():
        m = HUNK_HEADER_RE.match(line)
        if m:
            if current is not None:
                current.raw_text = "\n".join(raw_lines)
                hunks.append(current)
            old_start, old_count, new_start, new_count, heading = m.groups()
            current = Hunk(
                old_start=int(old_start),
                old_count=int(old_count) if old_count else 1,
                new_start=int(new_start),
                new_count=int(new_count) if new_count else 1,
                section_heading=(heading or "").strip(),
            )
            raw_lines = [line]
        elif current is not None:
            current.lines.append(line)
            raw_lines.append(line)

    if current is not None:
        current.raw_text = "\n".join(raw_lines)
        hunks.append(current)

    return hunks
