from __future__ import annotations

import sys
from pathlib import Path
import time
import streamlit as st
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm.hf_generator import HFGenerationConfig, HuggingFaceTextGenerator
from src.rag.rag_service import RagInferenceInput, RagService
from src.utils.config import get_env

@st.cache_resource
def load_rag_service(config_path: str) -> RagService:
    resolved = Path(config_path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return RagService.from_config_file(resolved)

@st.cache_resource
def load_local_cpu_generator(model: str):
    from src.llm.local_cpu_generator import (
        LocalCPUGenerationConfig,
        LocalCPUTextGenerator,
    )

    return LocalCPUTextGenerator(
        LocalCPUGenerationConfig(
            model=model,
            device="cpu",
            dtype="float32",
        )
    )

@st.cache_resource
def load_hf_api_generator(model: str, token: str | None, timeout: int) -> HuggingFaceTextGenerator:
    return HuggingFaceTextGenerator(HFGenerationConfig(model=model, token=token, timeout=timeout))

# --- CẤU HÌNH GIAO DIỆN APP ---
st.set_page_config(page_title="Vietnamese Cultural VQA", page_icon="🧠", layout="wide")
st.title("Vietnamese Cultural VQA - RAG + Hugging Face")
st.caption("App loaded. If the UI is blank, check browser console and Streamlit logs.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Model Settings")
    backend = st.selectbox("Backend", ["Local CPU", "Hugging Face API"], index=0)
    model_name = st.text_input("Base Model", value="Qwen/Qwen2.5-3B-Instruct")
    
    token_input = st.text_input("HF API token (optional)", value="", type="password")
    timeout = st.number_input("Timeout (s)", min_value=5, max_value=300, value=60)
    st.header("Generation Settings")
    max_new_tokens = st.number_input("Max new tokens", min_value=16, max_value=1024, value=256)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.05)

    st.header("RAG Settings")
    config_path = st.text_input("RAG config path", value="configs/rag.yaml")
    top_k = st.number_input("Top-k", min_value=1, max_value=20, value=5)
    max_context_chars = st.number_input("Max context chars", min_value=500, max_value=8000, value=3500)
    use_category_filter = st.checkbox("Use category filter", value=True)
    use_keyword_filter = st.checkbox("Use keyword filter", value=False)

st.markdown("Ask a question about Vietnamese culture. The system will retrieve context and answer.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Nhập câu hỏi...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    service = load_rag_service(config_path)
    service.top_k = int(top_k)
    service.max_context_chars = int(max_context_chars)
    service.use_category_filter = bool(use_category_filter)
    service.use_keyword_filter = bool(use_keyword_filter)

    try:
        with st.spinner("Retrieving context..."):
            t0 = time.perf_counter()
            rag_output = service.prepare_prompt(
                RagInferenceInput(
                    question=prompt, standalone_question=None, vision_caption=None, category=None, keyword=None,
                )
            )
            rag_time = time.perf_counter() - t0

        with st.spinner("Generating answer..."):
            t1 = time.perf_counter()
            if backend == "Local CPU":
                generator = load_local_cpu_generator(model_name)
                answer = generator.generate(
                    rag_output.prompt,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                )
            else:
                token = token_input.strip() or get_env("HF_API_TOKEN")
                generator = load_hf_api_generator(model_name, token, int(timeout))
                answer = generator.generate(
                    rag_output.prompt,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                )
            gen_time = time.perf_counter() - t1
            
    except Exception as exc:
        st.error(f"Error: {exc}")
        st.stop()

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(f"RAG: {rag_time:.2f}s | Generation: {gen_time:.2f}s")