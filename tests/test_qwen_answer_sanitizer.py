from src.llm.qwen_lora_generator import QwenLoraGenerator


def sanitize(answer: str) -> str:
    return QwenLoraGenerator._postprocess_answer(answer)


def test_sanitizes_visual_sentence_after_short_answer() -> None:
    answer = (
        "Nem nướng. Hình ảnh cho thấy một đĩa nem nướng, một món ăn "
        "truyền thống của Việt Nam. Món ăn này thường được ăn kèm với rau sống."
    )

    result = sanitize(answer)

    assert "Hình ảnh" not in result
    assert "cho thấy" not in result
    assert "Nem nướng" in result
    assert "truyền thống của Việt Nam" in result


def test_sanitizes_visual_opening_without_dropping_content() -> None:
    answer = (
        "Hình ảnh thể hiện kẹo dừa, một đặc sản nổi tiếng của tỉnh Bến Tre, "
        "miền Tây Nam Bộ."
    )

    result = sanitize(answer)

    assert "Hình ảnh" not in result
    assert "thể hiện" not in result
    assert result.startswith("Kẹo dừa là một đặc sản")
    assert "Bến Tre" in result


def test_removes_answer_prefix_and_visual_framing() -> None:
    answer = "Trả lời: Trong ảnh, áo dài là trang phục truyền thống của Việt Nam."

    result = sanitize(answer)

    assert result == "Áo dài là trang phục truyền thống của Việt Nam."


def test_pop_complete_sentences_keeps_incomplete_tail() -> None:
    complete, tail = QwenLoraGenerator._pop_complete_sentences(
        "Nem nướng là món ăn Việt Nam. Món này thường"
    )

    assert complete == "Nem nướng là món ăn Việt Nam. "
    assert tail == "Món này thường"


def test_iter_display_chunks_preserves_spaces_between_words() -> None:
    chunks = list(
        QwenLoraGenerator._iter_display_chunks(
            "Kẹo dừa là đặc sản Bến Tre.",
            leading_space=True,
        )
    )

    assert chunks == [" Kẹo", " dừa", " là", " đặc", " sản", " Bến", " Tre."]
