from src.standalone.template_rewriter import rewrite


def test_rewrite_event_question() -> None:
    result = rewrite("bánh chưng", "Món này thường xuất hiện trong dịp nào?")
    assert result.standalone_question == "Bánh chưng thường xuất hiện trong dịp nào?"


def test_rewrite_location_question() -> None:
    result = rewrite("Chùa Một Cột", "Công trình này nằm ở đâu?")
    assert result.standalone_question == "Chùa Một Cột nằm ở đâu?"


def test_no_duplication_when_keyword_already_in_question() -> None:
    result = rewrite("bánh chưng", "Bánh chưng có những nguyên liệu gì?")
    assert result.matched is False
    assert result.standalone_question == "Bánh chưng có những nguyên liệu gì?"