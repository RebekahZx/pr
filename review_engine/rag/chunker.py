from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken should always be available
    _ENCODER = None

try:
    from tree_sitter_language_pack import get_parser

    _TREE_SITTER_AVAILABLE = True
except Exception:  # pragma: no cover - package may not be installed in all envs
    get_parser = None
    _TREE_SITTER_AVAILABLE = False

FIXED_CHUNK_TOKENS = 200
FIXED_CHUNK_OVERLAP_TOKENS = 50

# Extension -> tree-sitter-language-pack grammar name
LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".kt": "kotlin",
    ".scala": "scala",
    ".swift": "swift",
}

# Node types treated as a self-contained "unit" chunk, per language.
FUNCTION_NODE_TYPES = {
    "python": {"function_definition"},
    "javascript": {"function_declaration", "method_definition", "function"},
    "typescript": {"function_declaration", "method_definition", "function"},
    "tsx": {"function_declaration", "method_definition", "function"},
    "go": {"function_declaration", "method_declaration"},
    "java": {"method_declaration", "constructor_declaration"},
    "ruby": {"method"},
    "php": {"function_definition", "method_declaration"},
    "rust": {"function_item"},
    "c": {"function_definition"},
    "cpp": {"function_definition"},
    "c_sharp": {"method_declaration", "constructor_declaration"},
    "kotlin": {"function_declaration"},
    "scala": {"function_definition"},
    "swift": {"function_declaration"},
}

CLASS_NODE_TYPES = {
    "python": {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration", "interface_declaration"},
    "tsx": {"class_declaration", "interface_declaration"},
    "go": {"type_declaration"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration"},
    "ruby": {"class", "module"},
    "php": {"class_declaration", "interface_declaration"},
    "rust": {"impl_item", "trait_item"},
    "c": {"struct_specifier"},
    "cpp": {"class_specifier", "struct_specifier"},
    "c_sharp": {"class_declaration", "interface_declaration", "struct_declaration"},
    "kotlin": {"class_declaration"},
    "scala": {"class_definition", "object_definition", "trait_definition"},
    "swift": {"class_declaration", "struct_declaration"},
}


@dataclass
class Chunk:
    file_path: str
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    text: str
    kind: str = "block"  # "function" | "class" | "module" | "fixed"
    symbol: Optional[str] = None
    language: Optional[str] = None


def detect_language(file_path: str) -> Optional[str]:
    _, ext = os.path.splitext(file_path)
    return LANGUAGE_BY_EXTENSION.get(ext.lower())


def _token_len(text: str) -> int:
    if _ENCODER is None:
        return max(1, len(text) // 4)
    return len(_ENCODER.encode(text, disallowed_special=()))


def _fixed_size_chunk(
    file_path: str,
    text: str,
    line_offset: int = 0,
    kind: str = "fixed",
    language: Optional[str] = None,
) -> List[Chunk]:
    """Chunk raw text into ~FIXED_CHUNK_TOKENS-token windows with overlap, line by line."""
    lines = text.splitlines()
    if not lines:
        return []

    chunks: List[Chunk] = []
    start_idx = 0
    n = len(lines)

    while start_idx < n:
        tok_count = 0
        end_idx = start_idx
        while end_idx < n:
            line_tokens = _token_len(lines[end_idx])
            if tok_count + line_tokens > FIXED_CHUNK_TOKENS and end_idx > start_idx:
                break
            tok_count += line_tokens
            end_idx += 1

        chunk_lines = lines[start_idx:end_idx]
        chunk_text = "\n".join(chunk_lines)
        chunks.append(
            Chunk(
                file_path=file_path,
                start_line=line_offset + start_idx + 1,
                end_line=line_offset + end_idx,
                text=chunk_text,
                kind=kind,
                language=language,
            )
        )

        if end_idx >= n:
            break

        # Step back by ~overlap tokens for the next window.
        back_tokens = 0
        back_idx = end_idx
        while back_idx > start_idx:
            back_tokens += _token_len(lines[back_idx - 1])
            if back_tokens >= FIXED_CHUNK_OVERLAP_TOKENS:
                break
            back_idx -= 1
        start_idx = max(back_idx, start_idx + 1)

    return chunks


def _node_symbol(node, source: bytes) -> Optional[str]:
    for child in node.children:
        if child.type in ("identifier", "field_identifier", "type_identifier", "property_identifier"):
            return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return None


def _ast_chunk(file_path: str, content: str, language: str) -> List[Chunk]:
    parser = get_parser(language)
    source = content.encode("utf-8", errors="replace")
    tree = parser.parse(source)

    func_types = FUNCTION_NODE_TYPES.get(language, set())
    class_types = CLASS_NODE_TYPES.get(language, set())

    chunks: List[Chunk] = []
    covered = [False] * (content.count("\n") + 2)  # 1-indexed line coverage

    def mark_covered(start_line: int, end_line: int) -> None:
        for ln in range(start_line, min(end_line, len(covered) - 1) + 1):
            covered[ln] = True

    def make_chunk(node, kind: str) -> Chunk:
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        mark_covered(start_line, end_line)
        return Chunk(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            text=text,
            kind=kind,
            symbol=_node_symbol(node, source),
            language=language,
        )

    def walk(node) -> None:
        if node.type in func_types:
            chunks.append(make_chunk(node, "function"))
            return
        if node.type in class_types:
            before = len(chunks)
            for child in node.children:
                walk(child)
            if len(chunks) == before:
                chunks.append(make_chunk(node, "class"))
            return
        for child in node.children:
            walk(child)

    walk(tree.root_node)

    # Capture module-level code (imports, constants, etc.) not covered by any chunk above.
    lines = content.splitlines()
    leftover_ranges: List[tuple] = []
    run_start = None
    for i, line in enumerate(lines, start=1):
        is_covered = covered[i] if i < len(covered) else False
        if not is_covered and line.strip():
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                leftover_ranges.append((run_start, i - 1))
                run_start = None
    if run_start is not None:
        leftover_ranges.append((run_start, len(lines)))

    for start_line, end_line in leftover_ranges:
        leftover_text = "\n".join(lines[start_line - 1:end_line])
        if not leftover_text.strip():
            continue
        for c in _fixed_size_chunk(file_path, leftover_text, line_offset=start_line - 1, kind="module", language=language):
            chunks.append(c)

    chunks.sort(key=lambda c: c.start_line)
    return chunks


def chunk_file(file_path: str, content: str) -> List[Chunk]:
    """Chunk a file's full content into semantic units.

    Uses tree-sitter AST boundaries (functions/classes/methods) when a grammar is
    available for the file's language; otherwise falls back to fixed-size token
    windows so indexing degrades gracefully instead of failing.
    """
    if not content.strip():
        return []

    language = detect_language(file_path)
    if language and _TREE_SITTER_AVAILABLE:
        try:
            return _ast_chunk(file_path, content, language)
        except Exception:
            pass  # fall through to fixed-size chunking on any parse failure

    return _fixed_size_chunk(file_path, content, kind="fixed", language=language)
