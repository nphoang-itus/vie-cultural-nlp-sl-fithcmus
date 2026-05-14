"""
Compute statistics over processed standalone-question records.
Pure computation — no I/O.
"""

from typing import Any, Sequence


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    """Return rounded division result. Avoid ZeroDivisionError."""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _char_length(text: str) -> int:
    """Return character length after stripping whitespace."""
    return len((text or "").strip())


def _word_length(text: str) -> int:
    """Return simple whitespace-based word count."""
    text = (text or "").strip()
    if not text:
        return 0
    return len(text.split())


def _average(values: Sequence[int | float]) -> float:
    """Return rounded average."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def compute_stats(
    original_records: list[dict[str, Any]],
    processed_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a stats dict suitable for writing to JSON."""
    total = len(processed_records)

    if total == 0:
        return {
            "total_questions": 0,
            "template_rewritten_count": 0,
            "llm_rewritten_count": 0,
            "unchanged_count": 0,
            "failed_count": 0,
        }

    methods = [record.get("rewrite_method", "") for record in processed_records]

    template_count = methods.count("template")
    llm_count = methods.count("llm")
    unchanged_count = methods.count("unchanged")
    failed_count = methods.count("failed")

    records_with_caption = [
        record for record in processed_records
        if record.get("vision_caption", "").strip()
    ]
    records_with_caption_count = len(records_with_caption)
    records_missing_caption_count = total - records_with_caption_count

    original_questions = [
        record.get("question", "")
        for record in original_records
    ]

    standalone_questions = [
        record.get("standalone_question", "")
        for record in processed_records
    ]

    original_char_lengths = [
        _char_length(question)
        for question in original_questions
    ]
    standalone_char_lengths = [
        _char_length(question)
        for question in standalone_questions
    ]

    original_word_lengths = [
        _word_length(question)
        for question in original_questions
    ]
    standalone_word_lengths = [
        _word_length(question)
        for question in standalone_questions
    ]

    examples = [
        {
            "image_id": record.get("image_id"),
            "base_image_id": record.get("base_image_id"),
            "question_id": record.get("question_id"),
            "keyword": record.get("keyword"),
            "vision_caption": record.get("vision_caption"),
            "original_question": record.get("question"),
            "standalone_question": record.get("standalone_question"),
            "rewrite_method": record.get("rewrite_method"),
        }
        for record in processed_records[:10]
    ]

    return {
        "total_questions": total,

        "template_rewritten_count": template_count,
        "llm_rewritten_count": llm_count,
        "unchanged_count": unchanged_count,
        "failed_count": failed_count,

        "template_rewrite_rate": _safe_divide(template_count, total),
        "llm_rewrite_rate": _safe_divide(llm_count, total),
        "unchanged_rate": _safe_divide(unchanged_count, total),
        "failed_rate": _safe_divide(failed_count, total),

        "records_with_vision_caption": records_with_caption_count,
        "records_missing_vision_caption": records_missing_caption_count,
        "vision_caption_coverage": _safe_divide(records_with_caption_count, total),

        "average_original_question_length_chars": _average(original_char_lengths),
        "average_standalone_question_length_chars": _average(standalone_char_lengths),

        "average_original_question_length_words": _average(original_word_lengths),
        "average_standalone_question_length_words": _average(standalone_word_lengths),

        "examples_before_after": examples,
    }