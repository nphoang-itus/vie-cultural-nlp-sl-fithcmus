# Retrieval Keyword Miss Error Analysis

Source: `retrieval_eval_results.category_filter.top10.jsonl` misses.

- Keyword miss rows: 931
- Representative cases exported: 30

## Misses By Category

| expected_category | miss_count |
| --- | --- |
| kien_truc | 258 |
| trang_phuc | 232 |
| le_hoi | 191 |
| nhac_cu | 93 |
| am_thuc | 83 |
| the_thao_truyen_thong | 74 |

## Error Type Heuristic

| error_type | miss_count |
| --- | --- |
| same_keyword_family | 516 |
| semantic_neighbor | 407 |
| keyword_variant_or_parent_child | 8 |

## Top Expected Keywords With Misses

| expected_keyword | miss_count | most_common_wrong_retrieved_keywords |
| --- | --- | --- |
| kiến trúc Huế | 29 | Đền Ngọc Sơn (25), Hoàng thành Thăng Long (21), Cung An Định (20) |
| áo choàng | 27 | áo dài cách tân (58), áo gấm Huế (42), áo dài cưới (25) |
| trang phục giày | 24 | giày gỗ (99), trang phục Dao (30), áo tứ thân (17) |
| lễ hội Dôn Ta | 22 | lễ giỗ Tổ Hùng Vương (43), lễ hội Đền Hùng (34), lễ hội làng (29) |
| đàn sến | 21 | đàn nhị (46), đan đay (34), đàn tỳ bà (20) |
| lễ hội Malaysia | 17 | lễ hội pháo hoa (33), lễ hội nước (27), Tết Hàn thực (12) |
| lễ hội tình yêu | 16 | Valentine (36), lễ hội pháo hoa (15), lễ hội cà phê (10) |
| Kinh thành Huế | 14 | Hoàng thành Thăng Long (17), Cung An Định (14), cổng thành Hà Nội (10) |
| Đinh Đen Lu | 14 | Đình Làng Việt Nam (22), Đền Ngọc Sơn (17), Đền Hùng (17) |
| lễ hội thu hoạch | 14 | lễ hội chọi trâu (18), lễ hội nước (17), lễ hội làng (13) |
| wrestling Việt Nam | 14 | điền kinh SEA Games Việt Nam (26), võ cổ truyền Việt Nam (13), vật cổ truyền Việt Nam (12) |
| trang phục Tày | 14 | áo dài cưới (24), áo dài cách tân (22), áo dài (11) |
| chùa Việt Nam | 13 | chùa Tam Chúc (59), chùa Ngọc Hoàng (23), chùa Bái Đính (19) |
| nhà rồng Thành phố Hồ Chí Minh | 13 | Dinh Độc Lập (28), nhà truyền thống miền Tây (15), bưu điện Sài Gòn (11) |
| Đền Ngọc Sơn | 13 | cầu Thê Húc (60), tháp Rùa (29), Đình Làng Việt Nam (5) |
| lễ hội Nang Hai | 13 | lễ hội làng (20), lễ hội pháo hoa (14), Rằm tháng Bảy (10) |
| cao lầu Hội An | 12 | mì Quảng (75), bún bò Huế (28), bánh chay (8) |
| Phú Chủ tịch | 12 | Dinh Độc Lập (24), lăng Hồ Chí Minh (22), Văn Miếu Quốc Tử Giám (21) |
| nhạc Việt Nam | 12 | đàn nguyệt (17), đàn thập lục (15), đàn nhị (13) |
| trang phục Cơ Tu | 12 | trang phục Tày (27), trang phục giày (16), trang phục Ê-Đê (16) |

## Top Wrong Keyword Pairs

| expected_keyword | wrong_retrieved_keyword | count |
| --- | --- | --- |
| trang phục giày | giày gỗ | 99 |
| cột cờ Hà Nội | tháp Rùa | 88 |
| cao lầu Hội An | mì Quảng | 75 |
| Đền Ngọc Sơn | cầu Thê Húc | 60 |
| chùa Việt Nam | chùa Tam Chúc | 59 |
| áo choàng | áo dài cách tân | 58 |
| đàn sến | đàn nhị | 46 |
| lễ hội Dôn Ta | lễ giỗ Tổ Hùng Vương | 43 |
| áo choàng | áo gấm Huế | 42 |
| turban Chăm | trang phục Chăm | 40 |
| lăng Minh Mạng | lăng Hồ Chí Minh | 39 |
| lễ hội tình yêu | Valentine | 36 |
| vật cổ truyền Việt Nam | võ cổ truyền Việt Nam | 35 |
| áo tứ thân | áo dài cách tân | 35 |
| lễ hội Dôn Ta | lễ hội Đền Hùng | 34 |
| đàn sến | đan đay | 34 |
| Tết Nguyên Đán | lễ hội bánh chưng | 33 |
| lễ hội Malaysia | lễ hội pháo hoa | 33 |
| đàn tam thập lục | đàn T'rưng | 32 |
| bánh gai | bánh ít | 31 |

## Representative Cases

| # | id | expected_keyword | category | top_wrong_keyword | top_score | error_type | retrieval_query |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | kien_truc\|kien_trúc_Hue\|000055_q5_q5 | kiến trúc Huế | kien_truc | Đền Ngọc Sơn | 0.6842 | semantic_neighbor | Kiến trúc này khác biệt với kiến trúc cổng thành ở các vùng miền khác của Việt Nam như thế nào? |
| 2 | trang_phuc\|ao_choang_lemur\|000043_q5_q5 | áo choàng | trang_phuc | áo dài cách tân | 0.7156 | same_keyword_family | So sánh áo dài cách tân với áo dài truyền thống của các vùng miền khác nhau ở Việt Nam? |
| 3 | trang_phuc\|trang_phục_Giay\|000034_q5_q5 | trang phục giày | trang_phuc | trang phục Dao | 0.6868 | same_keyword_family | So sánh trang phục của người Dao Đỏ với trang phục truyền thống của một số dân tộc thiểu số khác ở Việt Nam? |
| 4 | le_hoi\|le_hoi_Dôn_Ta\|000049_q5_q5 | lễ hội Dôn Ta | le_hoi | lễ hội Bunpchum | 0.6899 | same_keyword_family | So sánh lễ hội Dôn Ta với các lễ hội khác của người Khmer hoặc các lễ hội khác ở Việt Nam? |
| 5 | nhac_cu\|đan_sen\|000025_q3_q3 | đàn sến | nhac_cu | đàn nguyệt | 0.6499 | same_keyword_family | Ý nghĩa văn hóa của đàn tranh trong xã hội Việt Nam là gì? |
| 6 | le_hoi\|ma_lai_festival\|000053_q4_q4 | lễ hội Malaysia | le_hoi | lễ hội Dôn Ta | 0.6605 | same_keyword_family | Tại sao việc tổ chức các lễ hội văn hóa lại quan trọng trong xã hội hiện đại? |
| 7 | le_hoi\|le_hoi_tinh_yêu\|000040_q5_q5 | lễ hội tình yêu | le_hoi | Valentine | 0.6529 | semantic_neighbor | Hình ảnh về lễ hội tình yêu khác biệt như thế nào so với các lễ hội tình yêu ở các vùng miền khác của Việt Nam? |
| 8 | kien_truc\|kinh_thanh_Hue\|000038_q5_q5 | Kinh thành Huế | kien_truc | Đền Ngọc Sơn | 0.6874 | semantic_neighbor | So sánh kiến trúc Cổng Ngọ Môn với kiến trúc cổng thành ở các vùng miền khác của Việt Nam? |
| 9 | kien_truc\|đinh_Đen_Lu\|000032_q4_q4 | Đinh Đen Lu | kien_truc | Đền Ngọc Sơn | 0.6763 | semantic_neighbor | Tại sao kiến trúc và điêu khắc cổ trong ảnh lại quan trọng đối với văn hóa Việt Nam? |
| 10 | le_hoi\|le_hoi_harvest\|000024_q3_q3 | lễ hội thu hoạch | le_hoi | lễ hội chọi trâu | 0.6451 | same_keyword_family | Ý nghĩa văn hóa của việc sử dụng hình ảnh trâu cày ruộng trong áp phích lễ hội là gì? |
| 11 | the_thao_truyen_thong\|wrestling_Vietnam\|000032_q4_q4 | wrestling Việt Nam | the_thao_truyen_thong | điền kinh SEA Games Việt Nam | 0.7096 | same_keyword_family | Việc vận động viên giơ cao cờ Tổ quốc thể hiện niềm tự hào dân tộc như thế nào so với các lĩnh vực văn hóa khác? |
| 12 | trang_phuc\|trang_phục_Tay\|000049_q5_q5 | trang phục Tày | trang_phuc | áo dài cách tân | 0.7036 | semantic_neighbor | So sánh sự khác biệt giữa áo dài truyền thống và áo dài cách tân trong hình? |
| 13 | kien_truc\|chùa_Việt_Nam\|000058_q1_q1 | chùa Việt Nam | kien_truc | chùa Tam Chúc | 0.7467 | same_keyword_family | Chùa Việt Nam là công trình kiến trúc gì? |
| 14 | kien_truc\|nha_rồng_HCMC\|000049_q4_q4 | nhà rồng Thành phố Hồ Chí Minh | kien_truc | Dinh Độc Lập | 0.7452 | semantic_neighbor | Tại sao Dinh Độc Lập lại quan trọng đối với lịch sử và văn hóa Việt Nam? |
| 15 | kien_truc\|đen_Ngọc_Sơn\|000021_q3_q3 | Đền Ngọc Sơn | kien_truc | cầu Thê Húc | 0.7564 | semantic_neighbor | Ý nghĩa văn hóa của cầu Thê Húc và đền Ngọc Sơn là gì? |
| 16 | le_hoi\|le_hoi_Nang_Hai\|000049_q4_q4 | lễ hội Nang Hai | le_hoi | lễ hội cà phê | 0.5860 | same_keyword_family | Tại sao lễ hội này quan trọng đối với cộng đồng? |
| 17 | am_thuc\|cao_lầu_Hoi_An\|000022_q5_q5 | cao lầu Hội An | am_thuc | bún bò Huế | 0.7781 | semantic_neighbor | So sánh Cao Lầu với một món mì khác của Việt Nam, ví dụ như Bún bò Huế, về mặt nguyên liệu, hương vị và văn hoá. |
| 18 | kien_truc\|phu_Chu_tịch\|000055_q3_q3 | Phú Chủ tịch | kien_truc | kiến trúc Huế | 0.7323 | semantic_neighbor | Ý nghĩa văn hóa của kiến trúc Pháp thuộc địa trong bối cảnh lịch sử Việt Nam là gì? |
| 19 | nhac_cu\|musical_Vietnam\|000022_q3_q3 | nhạc Việt Nam | nhac_cu | đàn thập lục | 0.6312 | semantic_neighbor | Ý nghĩa văn hóa của nhạc Việt Nam là gì đối với xã hội Việt Nam hiện đại? |
| 20 | trang_phuc\|trang_phục_Cơ_Tu\|000049_q5_q5 | trang phục Cơ Tu | trang_phuc | trang phục Ê-Đê | 0.6760 | same_keyword_family | So sánh trang phục Cơ Tu với trang phục truyền thống của một dân tộc thiểu số khác ở Việt Nam (ví dụ: người Ê Đê). |
| 21 | trang_phuc\|turban_Chăm\|000056_q5_q5 | turban Chăm | trang_phuc | trang phục Chăm | 0.7020 | same_keyword_family | So sánh trang phục của người Chăm với trang phục truyền thống của một dân tộc thiểu số khác ở Việt Nam. |
| 22 | kien_truc\|đen_Bạch_Ma\|000055_q2_q2 | Đền Bạch Mã | kien_truc | Đình Làng Việt Nam | 0.7084 | semantic_neighbor | Các yếu tố kiến trúc đặc trưng của đình làng Việt Nam trong hình là gì? |
| 23 | kien_truc\|đinh_Bảng\|000032_q4_q4 | Đình Bảng | kien_truc | Đình Làng Việt Nam | 0.7119 | same_keyword_family | Tại sao kiến trúc nhà rường lại quan trọng trong bối cảnh văn hóa Việt Nam hiện đại? |
| 24 | trang_phuc\|trang_phục_Chăm\|000021_q2_q2 | trang phục Chăm | trang_phuc | trang phục Ê-Đê | 0.7044 | same_keyword_family | Mô tả chi tiết về màu sắc và chất liệu của bộ trang phục này. |
| 25 | le_hoi\|le_hoi_cúng_ơn\|000021_q5_q5 | lễ hội cúng ơn | le_hoi | lễ giỗ Tổ Hùng Vương | 0.7959 | same_keyword_family | So sánh lễ hội Giỗ tổ Hùng Vương với các lễ hội tưởng nhớ tổ tiên khác ở Việt Nam? |
| 26 | trang_phuc\|khuyên_tai_dân_toc\|000040_q2_q2 | khuyên tai dân tộc | trang_phuc | trang phục Tày | 0.6822 | semantic_neighbor | Hãy mô tả chi tiết trang phục của người phụ nữ dân tộc trong ảnh. |
| 27 | trang_phuc\|trang_phục_Dao\|000024_q4_q4 | trang phục Dao | trang_phuc | trang phục Tày | 0.6247 | same_keyword_family | So sánh trang phục này với trang phục của các dân tộc khác ở Việt Nam? |
| 28 | am_thuc\|chả_que_Hung_Yên\|000046_q3_q3 | chả que Hưng Yên | am_thuc | bánh dày | 0.7124 | semantic_neighbor | Ý nghĩa văn hóa của bánh dày là gì? |
| 29 | kien_truc\|cot_co_Ha_Noi\|000011_q4_q4 | cột cờ Hà Nội | kien_truc | tháp Rùa | 0.6937 | semantic_neighbor | Tại sao Tháp Rùa lại quan trọng đối với văn hóa Việt Nam? |
| 30 | kien_truc\|nha_Bac_Hồ\|000067_q5_q5 | Nhà Bác Hồ | kien_truc | Đền Ngọc Sơn | 0.6521 | semantic_neighbor | So sánh kiến trúc nhà cổ này với kiến trúc nhà ở hiện đại ở Việt Nam? |

## Notes

- The top-10 category filter removes cross-category errors; remaining misses are mostly keyword-level confusions inside the expected category.
- High-frequency clusters include visually or semantically close families: architecture landmarks, festivals, cakes/foods, traditional clothing, instruments, and sports.
- Cases where the wrong top keyword is a parent/child or sibling concept may need keyword aliasing, richer query terms, or reranking with expected keyword candidates.
