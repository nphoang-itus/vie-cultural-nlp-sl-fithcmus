"""
Template-based standalone question rewriter.
Uses regex patterns to replace Vietnamese demonstrative pronouns/phrases
with the cultural keyword. No external calls — pure string transformation.
"""

import re
from typing import NamedTuple


class RewriteResult(NamedTuple):
    standalone_question: str
    matched: bool   # True if a template rule fired


# Ordered list of (pattern, replacement_template) pairs.
# Patterns are matched against the START of the question (re.match).
# {keyword} in the replacement is filled with the actual keyword.
# Order matters: more specific patterns come first.
# [TODO] - Thêm pattern để bao phủ toàn bộ case
_PATTERNS: list[tuple[str, str]] = [
    # "Đây là gì?" → "Bánh chưng là gì?"
    (r"^Đây\s+", "{keyword} "),

    # "Món này..." → "Bánh chưng..."
    (r"^Món này\s+", "{keyword} "),

    # "Công trình này..." → "Chùa Một Cột..."
    (r"^Công trình này\s+", "{keyword} "),

    # "Trang phục này..." → "Áo dài..."
    (r"^Trang phục này\s+", "{keyword} "),

    # "Lễ hội này..." → "Tết Nguyên Đán..."
    (r"^Lễ hội này\s+", "{keyword} "),

    # "Nhạc cụ này..." → "Đàn bầu..."
    (r"^Nhạc cụ này\s+", "{keyword} "),

    # "Môn thể thao này..." → "Đấu vật truyền thống..."
    (r"^Môn thể thao này\s+", "{keyword} "),

    # "Trong hình..." → "Trong hình về Bánh chưng..."
    (r"^Trong hình\s+", "Trong hình về {keyword} "),

    # Generic "... này ..." catch-all (MUST be last)
    # e.g. "Kiến trúc này có đặc điểm gì?"
    (r"^(\w+\s+)?[Nn]ày\s+", "{keyword} "),
]

# Compile patterns once at import time for performance
_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.UNICODE), repl)
    for pat, repl in _PATTERNS
]


def _keyword_already_present(keyword: str, question: str) -> bool:
    """Return True if the keyword appears in the question (case-insensitive)."""
    return keyword.lower() in question.lower()


def rewrite(keyword: str, question: str) -> RewriteResult:
    """
    Attempt to rewrite a context-dependent question as a standalone question.

    Safety contract:
    - If keyword is already in the question → return unchanged (no duplication risk).
    - Try each pattern in order; apply the first match.
    - If no pattern matches → return original with matched=False.
    """
    if not keyword or not question:
        return RewriteResult(standalone_question=question, matched=False)

    # Safety guard: never inject a keyword that's already present
    if _keyword_already_present(keyword, question):
        return RewriteResult(standalone_question=question, matched=False)

    for pattern, replacement_template in _COMPILED:
        if pattern.match(question):
            replacement = replacement_template.replace("{keyword}", keyword)
            rewritten = pattern.sub(replacement, question, count=1)
            # Capitalise the first letter (keyword may start lower-case)
            rewritten = rewritten[0].upper() + rewritten[1:] if rewritten else rewritten
            return RewriteResult(standalone_question=rewritten, matched=True)

    return RewriteResult(standalone_question=question, matched=False)