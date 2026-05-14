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

    # "Hình ảnh này thể hiện điều gì?" → "Hình ảnh về Bánh chưng thể hiện điều gì?"
    (r"^Hình ảnh này\s+", "Hình ảnh về {keyword} "),

    # "Hình ảnh thể hiện điều gì về quá trình làm bánh?" → "Hình ảnh về Bánh chưng thể hiện điều gì về quá trình làm bánh?"
    (r"^Hình ảnh thể hiện\s+", "Hình ảnh về {keyword} thể hiện "),

    # "Hình ảnh phản ánh điều gì về văn hóa Việt Nam?" → "Hình ảnh về Bánh chưng phản ánh điều gì về văn hóa Việt Nam?"
    (r"^Hình ảnh phản ánh\s+", "Hình ảnh về {keyword} phản ánh "),

    # "Tại sao hình ảnh này quan trọng?" → "Tại sao hình ảnh về Bánh chưng quan trọng?"
    (r"^Tại sao hình ảnh này\s+", "Tại sao hình ảnh về {keyword} "),

    # "Ý nghĩa văn hóa của hình ảnh này là gì?" → "Ý nghĩa văn hóa của Bánh chưng là gì?"
    (r"^Ý nghĩa văn hóa của hình ảnh này\s+", "Ý nghĩa văn hóa của {keyword} "),

    # "Ý nghĩa văn hoá của hình ảnh này là gì?" → "Ý nghĩa văn hoá của Bánh chưng là gì?"
    (r"^Ý nghĩa văn hoá của hình ảnh này\s+", "Ý nghĩa văn hoá của {keyword} "),

    # "Mô tả chi tiết các thành phần trong hình ảnh này." → "Mô tả chi tiết các thành phần của Bánh chưng."
    (r"^Mô tả chi tiết các thành phần trong hình ảnh này", "Mô tả chi tiết các thành phần của {keyword}"),

    # "Mô tả chi tiết các thành phần trong hình ảnh?" → "Mô tả chi tiết các thành phần của Bánh chưng?"
    (r"^Mô tả chi tiết các thành phần trong hình ảnh", "Mô tả chi tiết các thành phần của {keyword}"),

    # "Mô tả chi tiết về hình ảnh này." → "Mô tả chi tiết về Bánh chưng."
    (r"^Mô tả chi tiết về hình ảnh này", "Mô tả chi tiết về {keyword}"),

    # "Mô tả chi tiết về hình ảnh." → "Mô tả chi tiết về Bánh chưng."
    (r"^Mô tả chi tiết về hình ảnh(?=[.?])", "Mô tả chi tiết về {keyword}"),

    # "Mô tả chi tiết những gì bạn thấy trong hình ảnh này?" → "Mô tả chi tiết những gì bạn thấy về Bánh chưng?"
    (r"^Mô tả chi tiết những gì bạn thấy trong hình ảnh này", "Mô tả chi tiết những gì bạn thấy về {keyword}"),

    # "Mô tả chi tiết những gì bạn thấy trong hình ảnh." → "Mô tả chi tiết những gì bạn thấy về Bánh chưng."
    (r"^Mô tả chi tiết những gì bạn thấy trong hình ảnh(?=[.?])", "Mô tả chi tiết những gì bạn thấy về {keyword}"),

    # "Mô tả hình ảnh này." → "Mô tả Bánh chưng."
    (r"^Mô tả hình ảnh này", "Mô tả {keyword}"),

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
