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


def test_rewrite_image_this_opening() -> None:
    result = rewrite("bánh chưng", "Hình ảnh này thể hiện điều gì?")
    assert result.standalone_question == "Hình ảnh về bánh chưng thể hiện điều gì?"


def test_rewrite_explicit_image_predicates() -> None:
    result = rewrite("bánh chưng", "Hình ảnh thể hiện điều gì về quá trình làm bánh?")
    assert result.standalone_question == "Hình ảnh về bánh chưng thể hiện điều gì về quá trình làm bánh?"

    result = rewrite("bánh chưng", "Hình ảnh phản ánh điều gì về văn hóa Việt Nam?")
    assert result.standalone_question == "Hình ảnh về bánh chưng phản ánh điều gì về văn hóa Việt Nam?"


def test_rewrite_image_meaning_and_description_openings() -> None:
    result = rewrite("bánh chưng", "Ý nghĩa văn hóa của hình ảnh này là gì?")
    assert result.standalone_question == "Ý nghĩa văn hóa của bánh chưng là gì?"

    result = rewrite("bánh chưng", "Tại sao hình ảnh này quan trọng?")
    assert result.standalone_question == "Tại sao hình ảnh về bánh chưng quan trọng?"

    result = rewrite("bánh chưng", "Mô tả chi tiết về hình ảnh này.")
    assert result.standalone_question == "Mô tả chi tiết về bánh chưng."

    result = rewrite("bánh chưng", "Mô tả chi tiết về hình ảnh.")
    assert result.standalone_question == "Mô tả chi tiết về bánh chưng."

    result = rewrite("đình Bảng", "Mô tả chi tiết về hình ảnh rồng trên đầu hồi mái.")
    assert result.matched is False
    assert result.standalone_question == "Mô tả chi tiết về hình ảnh rồng trên đầu hồi mái."

    result = rewrite("bánh chưng", "Mô tả chi tiết các thành phần trong hình ảnh?")
    assert result.standalone_question == "Mô tả chi tiết các thành phần của bánh chưng?"

    result = rewrite("lễ hội chùa Hương", "Mô tả chi tiết những gì bạn thấy trong hình ảnh này?")
    assert result.standalone_question == "Mô tả chi tiết những gì bạn thấy về lễ hội chùa Hương?"

    result = rewrite("bánh chưng", "Mô tả hình ảnh này.")
    assert result.standalone_question == "Mô tả bánh chưng."
