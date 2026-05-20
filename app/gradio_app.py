"""
Clean Gradio app for Vietnamese Cultural VQA.
Người dùng chỉ thấy khung chat, Backend tự động lo liệu cấu hình.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Dict, List

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm.hf_generator import HFGenerationConfig, HuggingFaceTextGenerator
from src.llm.local_cpu_generator import LocalCPUGenerationConfig, LocalCPUTextGenerator
from src.rag.rag_service import RagInferenceInput, RagService
from src.utils.config import get_env

# ==========================================
# ⚙️ CẤU HÌNH BACKEND (Giấu kín khỏi người dùng)
# ==========================================
BACKEND_TYPE = "Local CPU"  # Chi có thể đổi thành "Hugging Face API" ở đây nếu muốn
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LORA_ADAPTER_ID = "ohthisischichi/viet-cultural-qa-qwen2.5-lora"
CONFIG_PATH = "configs/rag.yaml"

# Tham số RAG
TOP_K = 1
MAX_CONTEXT_CHARS = 1800
USE_CATEGORY_FILTER = True
USE_KEYWORD_FILTER = False

# Tham số Sinh văn bản (Model)
MAX_NEW_TOKENS = 48
TEMPERATURE = 0.2
TOP_P = 0.9

# ==========================================
# 🚀 KHỞI TẠO HỆ THỐNG (Chỉ chạy 1 lần ngầm ở Terminal)
# ==========================================
print("⏳ Đang khởi tạo hệ thống RAG...")
resolved_config = Path(CONFIG_PATH)
if not resolved_config.is_absolute():
    resolved_config = ROOT / resolved_config
rag_service = RagService.from_config_file(resolved_config)
rag_service.top_k = TOP_K
rag_service.max_context_chars = MAX_CONTEXT_CHARS
rag_service.use_category_filter = USE_CATEGORY_FILTER
rag_service.use_keyword_filter = USE_KEYWORD_FILTER

print(f"⏳ Đang nạp mô hình ngôn ngữ ({BACKEND_TYPE}) vào bộ nhớ...")
if BACKEND_TYPE == "Local CPU":
    generator = LocalCPUTextGenerator(
        LocalCPUGenerationConfig(
            model=MODEL_NAME,
            lora_adapter=LORA_ADAPTER_ID,
            device="cpu",
            dtype="float16",
        )
    )
else:
    token = get_env("HF_API_TOKEN")
    generator = HuggingFaceTextGenerator(
        HFGenerationConfig(model=MODEL_NAME, token=token, timeout=60)
    )
print("✅ Hệ thống đã sẵn sàng hoạt động!")


# ==========================================
# 🧠 HÀM XỬ LÝ CHAT (Nhận câu hỏi -> Trả kết quả)
# ==========================================
def answer_question(
    message: str,
    history: List[Dict[str, str]],
) -> tuple[list[dict[str, str]], str]:
    
    message = (message or "").strip()
    if not message:
        return history, ""

    try:
        # 1. Quét tài liệu RAG
        t0 = time.perf_counter()
        rag_output = rag_service.prepare_prompt(
            RagInferenceInput(
                question=None,
                standalone_question=message,
                vision_caption=None,
                category=None,
                keyword=None,
            )
        )
        rag_time = time.perf_counter() - t0

        # 2. Sinh câu trả lời từ Model
        t1 = time.perf_counter()
        answer = generator.generate(
            rag_output.prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        gen_time = time.perf_counter() - t1

        answer = answer.strip() or "(Không có câu trả lời)"
        
        # Đính kèm thêm thông tin tốc độ chạy xuống dưới cùng để dễ báo cáo
        meta = f"\n\n*(⏱️ RAG: {rag_time:.2f}s | Sinh chữ: {gen_time:.2f}s)*"
        
        # Cập nhật lịch sử chat
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer + meta},
        ]
        return history, ""
    
    except Exception as exc:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"⚠️ Lỗi hệ thống: {exc}"},
        ]
        return history, ""


# ==========================================
# 🎨 GIAO DIỆN NGƯỜI DÙNG (Tối giản và Đẹp mắt)
# ==========================================
with gr.Blocks(title="Vietnamese Cultural VQA") as demo:
    gr.Markdown("<h1 style='text-align: center;'>🇻🇳 Hệ thống Hỏi đáp Văn hóa Việt Nam</h1>")
    gr.Markdown("<p style='text-align: center; color: gray;'>Hệ thống ứng dụng kiến trúc RAG và mô hình LLM để giải đáp các câu hỏi văn hóa, lịch sử.</p>")

    # Khung Chatbot rộng rãi
    chatbot = gr.Chatbot(label="Trò chuyện cùng Chuyên gia AI", height=500)
    
    with gr.Row():
        message = gr.Textbox(
            show_label=False,
            placeholder="Nhập câu hỏi của bạn (Ví dụ: Ý nghĩa của lễ hội Đền Hùng là gì?)...",
            scale=8,
            lines=1
        )
        send = gr.Button("🚀 Gửi", variant="primary", scale=1)
        clear = gr.Button("🗑️ Xóa hội thoại", scale=1)

    # Gắn sự kiện khi bấm nút hoặc nhấn Enter
    send.click(answer_question, inputs=[message, chatbot], outputs=[chatbot, message])
    message.submit(answer_question, inputs=[message, chatbot], outputs=[chatbot, message])
    clear.click(lambda: ([], ""), outputs=[chatbot, message])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())