from __future__ import annotations

import html
import json
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
SUPPORTED_DOMAINS = [
    ("kien_truc", "Kiến trúc"),
    ("am_thuc", "Ẩm thực"),
    ("trang_phuc", "Trang phục"),
    ("le_hoi", "Lễ hội"),
    ("nhac_cu", "Nhạc cụ"),
    ("the_thao_truyen_thong", "Thể thao"),
]
SUPPORTED_DOMAIN_LABELS = ", ".join(label for _, label in SUPPORTED_DOMAINS)
GREETING = (
    "Xin chào! Đây là hệ thống hỗ trợ hỏi đáp về văn hóa Việt Nam"
)


st.set_page_config(
    page_title="Vietnamese Cultural QA",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --app-bg: #ffffff;
                --app-bg-soft: #f7f8fa;
                --sidebar-bg: #f8fafc;
                --surface: #ffffff;
                --user-bubble-start: #eef2ff;
                --user-bubble-end: #f8fafc;
                --border: #e5e7eb;
                --border-blue: #e0e7ff;
                --text: #111827;
                --muted: #6b7280;
                --accent-blue: #3b82f6;
                --accent-coral: #ff4b4b;
                --accent-purple: #8b5cf6;
            }

            .stApp {
                background:
                    radial-gradient(circle at 50% -120px, rgba(59, 130, 246, 0.12), transparent 300px),
                    radial-gradient(circle at 72% -140px, rgba(139, 92, 246, 0.08), transparent 260px),
                    linear-gradient(180deg, var(--app-bg-soft) 0%, var(--app-bg) 230px);
                color: var(--text);
            }

            .block-container {
                max-width: 100%;
                padding: 0 0 7.25rem;
            }

            [data-testid="stSidebar"] {
                background: var(--sidebar-bg);
                border-right: 1px solid var(--border);
            }

            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p {
                color: var(--muted);
            }

            [data-testid="stSidebar"] hr {
                margin: 0.9rem 0;
                border-color: var(--border);
            }

            .chat-title {
                display: inline-block;
                background: linear-gradient(90deg, #111827 0%, #2563eb 52%, #ef4444 100%);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                font-size: 1.55rem;
                font-weight: 750;
                line-height: 1.15;
                letter-spacing: 0;
                margin: 0 0 0.18rem;
            }

            .chat-caption,
            .stCaptionContainer {
                color: var(--muted);
                font-size: 0.92rem;
                margin: 0;
            }

            .chat-title-wrap {
                border-bottom: 1px solid var(--border);
                margin: 0 0 1rem;
                padding: 56px 0 32px;
            }

            .scope-note {
                background: #ffffff;
                border: 1px solid var(--border);
                border-radius: 8px;
                color: var(--text);
                font-size: 0.9rem;
                line-height: 1.55;
                margin-top: 0.95rem;
                padding: 10px 12px;
            }

            .scope-note strong {
                color: #1f2937;
                font-weight: 700;
            }

            .sidebar-note {
                background: #ffffff;
                border: 1px solid var(--border);
                border-radius: 8px;
                color: var(--muted);
                font-size: 0.86rem;
                line-height: 1.5;
                margin: 0.35rem 0 0.8rem;
                padding: 10px 12px;
            }

            .selected-category {
                color: var(--muted);
                font-size: 0.84rem;
                margin: 0.1rem 0 0.45rem;
            }

            .chat-shell {
                box-sizing: border-box;
                max-width: 820px;
                margin: 0 auto;
                padding: 0 24px;
                width: 100%;
            }

            .assistant-row,
            .user-row {
                box-sizing: border-box;
                display: flex;
                margin: 0.45rem 0;
                width: 100%;
            }

            .assistant-row {
                justify-content: flex-start;
            }

            .user-row {
                justify-content: flex-end;
            }

            .assistant-block {
                max-width: 100%;
                width: 100%;
            }

            .assistant-message,
            .user-message {
                color: var(--text);
                font-size: 0.97rem;
                overflow-wrap: break-word;
                white-space: pre-wrap;
                word-break: normal;
            }

            .assistant-message {
                color: var(--text);
                line-height: 1.75;
                margin: 8px 0 8px;
                max-width: 100%;
                padding: 4px 0;
                width: 100%;
            }

            .assistant-message.loading {
                color: var(--muted);
                font-size: 0.94rem;
            }

            .assistant-message::selection,
            .user-message::selection {
                background: #dbeafe;
            }

            .user-message {
                background: linear-gradient(135deg, var(--user-bubble-start), var(--user-bubble-end));
                border: 1px solid var(--border-blue);
                border-radius: 18px;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
                color: var(--text);
                line-height: 1.55;
                margin-left: auto;
                margin-right: 0;
                max-width: 72%;
                padding: 10px 14px;
            }

            .assistant-message a {
                color: var(--accent-blue);
            }

            .debug-heading {
                color: var(--muted);
                font-size: 0.75rem;
                font-weight: 650;
                letter-spacing: 0.02em;
                margin: 0.35rem 0 0.25rem;
                text-transform: uppercase;
            }

            .assistant-debug {
                margin: 8px 0 18px;
                width: 100%;
            }

            div[class*="st-key-assistant_debug_"] {
                box-sizing: border-box;
                margin: -0.2rem auto 1rem;
                max-width: 820px;
                padding: 0 24px;
                width: 100%;
            }

            div[class*="st-key-assistant_debug_"] [data-testid="stPopover"] {
                margin: 0;
                width: fit-content;
            }

            div[class*="st-key-assistant_debug_"] [data-testid="stPopover"] > button {
                background: #ffffff;
                border: 1px solid var(--border);
                border-radius: 999px;
                box-shadow: none;
                color: #64748b;
                font-size: 13px;
                font-weight: 500;
                min-height: 1.8rem;
                padding: 4px 10px;
            }

            div[class*="st-key-assistant_debug_"] [data-testid="stPopover"] > button:hover {
                border-color: #cbd5e1;
                color: var(--accent-blue);
            }

            div[class*="st-key-assistant_debug_"] [data-testid="stPopover"] > button:focus {
                box-shadow: none;
            }

            div[data-testid="stPopoverBody"] {
                border: 1px solid var(--border);
                border-radius: 12px;
                box-shadow: 0 16px 36px rgba(15, 23, 42, 0.12);
                max-height: min(72vh, 760px);
                max-width: min(760px, calc(100vw - 64px));
                overflow-y: auto;
                padding: 12px;
            }

            div[data-testid="stPopoverBody"] [data-testid="stExpander"],
            div[data-testid="stPopoverBody"] [data-testid="stExpander"] *,
            div[data-testid="stPopoverBody"] details,
            div[data-testid="stPopoverBody"] details *,
            div[data-testid="stPopoverBody"] summary,
            div[data-testid="stPopoverBody"] summary * {
                animation: none !important;
                transition: none !important;
            }

            div[data-testid="stChatInput"] {
                box-sizing: border-box;
                max-width: 820px;
                margin: 0 auto 28px !important;
                padding: 0 24px !important;
                width: 100%;
            }

            div[data-testid="stChatInput"] > div,
            div[data-testid="stChatInput"] > div > div,
            div[data-testid="stChatInput"] > div > div > div,
            div[data-testid="stChatInput"] form,
            div[data-testid="stChatInput"] form > div,
            div[data-testid="stChatInput"] form > div > div,
            div[data-testid="stChatInput"] [data-baseweb="textarea"],
            div[data-testid="stChatInput"] [data-baseweb="base-input"] {
                background: transparent !important;
                border: 0 !important;
                border-radius: 999px !important;
                box-shadow: none !important;
                outline: none !important;
                padding: 0 !important;
            }

            div[data-testid="stChatInput"] > div,
            div[data-testid="stChatInput"] form {
                position: relative !important;
            }

            div[data-testid="stChatInput"] textarea {
                background: #ffffff !important;
                border: 1px solid var(--border) !important;
                border-radius: 999px !important;
                box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08) !important;
                color: var(--text) !important;
                min-height: 2.8rem !important;
                padding: 0.76rem 3.15rem 0.76rem 1.05rem !important;
            }

            div[data-testid="stChatInput"] textarea:focus {
                border-color: #c7d2fe !important;
                box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08), 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
                outline: none !important;
            }

            div[data-testid="stChatInput"] button {
                background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
                border-radius: 999px !important;
                box-shadow: none !important;
                color: #ffffff !important;
                height: 2.12rem !important;
                margin: 0 !important;
                min-height: 2.12rem !important;
                padding: 0 !important;
                position: absolute !important;
                right: 0.42rem !important;
                top: 0.34rem !important;
                transform: none !important;
                width: 2.12rem !important;
                z-index: 3 !important;
            }

            div[data-testid="stBottomBlockContainer"] {
                background: linear-gradient(180deg, rgba(255,255,255,0), #ffffff 30%);
                padding-bottom: 1.6rem !important;
            }

            .stTextInput input {
                background: #ffffff;
                border-color: var(--border);
                border-radius: 999px;
            }

            .stTextInput input:focus {
                border-color: #cbd5e1;
                box-shadow: 0 0 0 1px #cbd5e1;
            }

            .stCheckbox label {
                color: var(--text);
            }

            .stButton > button {
                background: #ffffff;
                border-color: var(--border);
                border-radius: 999px;
                box-shadow: none;
                color: #111827 !important;
                font-weight: 600;
            }

            .stButton > button *,
            .stButton > button p,
            .stButton > button span {
                color: #111827 !important;
            }

            .stButton > button[kind="primary"],
            .stButton button[data-testid="stBaseButton-primary"],
            .stButton button[class*="primary"] {
                background: #1d4ed8 !important;
                border-color: #1d4ed8 !important;
                color: #ffffff !important;
            }

            .stButton > button[kind="primary"] *,
            .stButton > button[kind="primary"] p,
            .stButton > button[kind="primary"] span,
            .stButton button[data-testid="stBaseButton-primary"] *,
            .stButton button[data-testid="stBaseButton-primary"] p,
            .stButton button[data-testid="stBaseButton-primary"] span,
            .stButton button[class*="primary"] *,
            .stButton button[class*="primary"] p,
            .stButton button[class*="primary"] span {
                color: #ffffff !important;
            }

            .stButton > button:hover {
                border-color: var(--border);
                color: var(--accent-coral);
            }

            .stButton > button[kind="primary"]:hover,
            .stButton button[data-testid="stBaseButton-primary"]:hover,
            .stButton button[class*="primary"]:hover {
                background: #1e40af !important;
                border-color: #1e40af !important;
                color: #ffffff !important;
            }

            .stButton > button[kind="primary"]:hover *,
            .stButton > button[kind="primary"]:hover p,
            .stButton > button[kind="primary"]:hover span,
            .stButton button[data-testid="stBaseButton-primary"]:hover *,
            .stButton button[data-testid="stBaseButton-primary"]:hover p,
            .stButton button[data-testid="stBaseButton-primary"]:hover span,
            .stButton button[class*="primary"]:hover *,
            .stButton button[class*="primary"]:hover p,
            .stButton button[class*="primary"]:hover span {
                color: #ffffff !important;
            }

            [data-testid="stExpander"],
            [data-testid="stExpander"] *,
            details,
            details *,
            summary,
            summary * {
                animation: none !important;
                transition: none !important;
            }

            [data-testid="stExpander"] {
                background: #ffffff;
                border: 1px solid var(--border);
                border-radius: 8px;
                box-shadow: none;
                margin: 0.35rem 0;
            }

            [data-testid="stExpander"] summary {
                color: var(--muted);
                font-size: 0.84rem;
                min-height: 2rem;
                padding: 0.25rem 0.65rem;
            }

            [data-testid="stExpanderDetails"] {
                padding: 0.35rem 0.75rem 0.75rem;
            }

            [data-testid="stCodeBlock"] {
                margin: 0.25rem 0 0.7rem;
            }

            [data-testid="stAlert"] {
                border: 1px solid var(--border);
                border-radius: 8px;
                box-shadow: none;
                margin: 0.25rem auto 0.7rem;
                max-width: 772px;
            }

            [data-testid="stDataFrame"] {
                border: 1px solid var(--border);
                border-radius: 8px;
                overflow: hidden;
                margin: 0.35rem 0 0.65rem;
            }

            [data-testid="stDataFrame"] div {
                font-size: 0.82rem;
            }

            code {
                border-radius: 6px;
            }

            hr {
                border-color: var(--border);
                margin: 1rem 0;
            }

            @media (max-width: 760px) {
                .block-container {
                    padding-left: 0;
                    padding-right: 0;
                }

                .chat-shell,
                div[class*="st-key-assistant_debug_"],
                div[data-testid="stChatInput"] {
                    padding-left: 16px;
                    padding-right: 16px;
                }

                .chat-title-wrap {
                    padding-top: 40px;
                    padding-bottom: 24px;
                }

                .user-message {
                    max-width: 88%;
                }
            }

            @media (max-width: 420px) {
                .user-message {
                    max-width: 94%;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    if "message_counter" not in st.session_state:
        st.session_state.message_counter = 1

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "id": "greeting",
                "role": "assistant",
                "content": GREETING,
                "metadata": None,
            }
        ]

    if "selected_category" not in st.session_state:
        st.session_state.selected_category = ""


def next_message_id() -> str:
    message_id = str(st.session_state.message_counter)
    st.session_state.message_counter += 1
    return message_id


def build_payload(
    question: str,
    debug: bool,
    category: str,
    keyword: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": question,
        "debug": debug,
    }

    if category.strip():
        payload["category"] = category.strip()

    if keyword.strip():
        payload["keyword"] = keyword.strip()

    return payload


def render_category_filter_buttons() -> str:
    selected_category = str(st.session_state.get("selected_category") or "")

    st.markdown("**Category filter**")
    selected_label = "Tất cả lĩnh vực"
    for category_id, label in SUPPORTED_DOMAINS:
        if category_id == selected_category:
            selected_label = f"{label} ({category_id})"
            break
    st.markdown(
        f'<div class="selected-category">Đang chọn: {html.escape(selected_label)}</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "✓ Tất cả" if not selected_category else "Tất cả",
        key="category_all",
        width="stretch",
        type="primary" if not selected_category else "secondary",
    ):
        st.session_state.selected_category = ""
        st.rerun()

    for index in range(0, len(SUPPORTED_DOMAINS), 2):
        columns = st.columns(2)
        for column, (category_id, label) in zip(
            columns,
            SUPPORTED_DOMAINS[index : index + 2],
            strict=False,
        ):
            with column:
                if st.button(
                    f"✓ {label}" if selected_category == category_id else label,
                    key=f"category_{category_id}",
                    width="stretch",
                    type="primary" if selected_category == category_id else "secondary",
                    help=category_id,
                ):
                    st.session_state.selected_category = category_id
                    st.rerun()

    return str(st.session_state.get("selected_category") or "")


def parse_sse_event(line: str) -> dict[str, Any] | None:
    if not line:
        return None

    if not line.startswith("data: "):
        return None

    raw = line.removeprefix("data: ").strip()
    if not raw or raw == "[DONE]":
        return None

    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(event, dict):
        return None

    return event


def render_message_html(
    role: str,
    content: str,
    cursor: bool = False,
) -> str:
    safe_content = html.escape(content or "")
    if cursor:
        safe_content += '<span style="color:#6b7280;">▌</span>'

    if role == "assistant":
        return f"""
            <div class="chat-shell">
                <div class="assistant-row">
                    <div class="assistant-block">
                        <div class="assistant-message">{safe_content}</div>
                    </div>
                </div>
            </div>
        """

    return f"""
        <div class="chat-shell">
            <div class="{role}-row">
                <div class="{role}-message">{safe_content}</div>
            </div>
        </div>
    """


def render_loading_html() -> str:
    return """
        <div class="chat-shell">
            <div class="assistant-row">
                <div class="assistant-block">
                    <div class="assistant-message loading">Đang tạo câu trả lời...</div>
                </div>
            </div>
        </div>
    """


def stream_qa_response(
    api_base_url: str,
    payload: dict[str, Any],
    placeholder: st.delta_generator.DeltaGenerator,
) -> tuple[str, dict[str, Any] | None, str | None]:
    answer_text = ""
    metadata: dict[str, Any] | None = None

    try:
        response = requests.post(
            f"{api_base_url.rstrip('/')}/api/qa/stream",
            json=payload,
            stream=True,
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        return "", None, "Không thể kết nối tới FastAPI backend. Hãy kiểm tra uvicorn đã chạy chưa."
    except requests.exceptions.Timeout:
        return "", None, "Yêu cầu bị timeout. Model có thể vẫn đang sinh câu trả lời."
    except requests.RequestException as exc:
        return "", None, f"Lỗi khi gọi API: {exc}"

    if response.status_code != 200:
        return "", None, f"API error {response.status_code}\n\n{response.text}"

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue

        event = parse_sse_event(line.strip())
        if event is None:
            continue

        event_type = event.get("type")

        if event_type == "metadata":
            metadata = {
                "retrieval_query": event.get("retrieval_query", ""),
                "filters": event.get("filters"),
                "contexts": event.get("contexts", []),
                "prompt": event.get("prompt", ""),
            }

        elif event_type == "token":
            token = event.get("token", "")
            if isinstance(token, str):
                answer_text += token
                placeholder.markdown(
                    render_message_html("assistant", answer_text, cursor=True),
                    unsafe_allow_html=True,
                )

        elif event_type == "done":
            break

        elif event_type == "error":
            message = event.get("message", "Unknown streaming error.")
            return answer_text, metadata, str(message)

    placeholder.markdown(
        render_message_html("assistant", answer_text),
        unsafe_allow_html=True,
    )
    return answer_text, metadata, None


def render_user_message(content: str) -> None:
    st.markdown(
        render_message_html("user", content),
        unsafe_allow_html=True,
    )


def render_debug_info(metadata: dict[str, Any] | None, message_id: str) -> None:
    if not metadata:
        return

    with st.container(key=f"assistant_debug_{message_id}"):
        with st.popover("ⓘ Debug"):
            st.markdown('<div class="debug-heading">retrieval_query</div>', unsafe_allow_html=True)
            st.code(metadata.get("retrieval_query", ""), language="text")

            st.markdown('<div class="debug-heading">filters</div>', unsafe_allow_html=True)
            filters = metadata.get("filters")
            if filters:
                st.json(filters)
            else:
                st.info("No filters applied.")

            contexts = metadata.get("contexts") or []
            if contexts:
                rows = []
                for index, context in enumerate(contexts, start=1):
                    context_metadata = context.get("metadata", {}) or {}
                    rows.append(
                        {
                            "rank": index,
                            "doc_id": context.get("doc_id"),
                            "keyword": context_metadata.get("keyword"),
                            "category": context_metadata.get("category"),
                            "subcategory": context_metadata.get("subcategory"),
                            "score": context.get("score"),
                            "distance": context.get("distance"),
                            "content_preview": str(context.get("content", ""))[:180],
                        }
                    )

                st.markdown('<div class="debug-heading">retrieved contexts</div>', unsafe_allow_html=True)
                st.dataframe(
                    pd.DataFrame(rows),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "rank": st.column_config.NumberColumn("Rank", width="small"),
                        "doc_id": st.column_config.TextColumn("Document", width="medium"),
                        "keyword": st.column_config.TextColumn("Keyword", width="medium"),
                        "category": st.column_config.TextColumn("Category", width="small"),
                        "subcategory": st.column_config.TextColumn("Subcategory", width="small"),
                        "score": st.column_config.NumberColumn("Score", format="%.4f", width="small"),
                        "distance": st.column_config.NumberColumn("Distance", format="%.4f", width="small"),
                        "content_preview": st.column_config.TextColumn("Preview", width="large"),
                    },
                )

                for index, context in enumerate(contexts, start=1):
                    context_metadata = context.get("metadata", {}) or {}
                    score = context.get("score")
                    keyword = context_metadata.get("keyword", "")
                    title = f"Context {index}"
                    if keyword:
                        title += f" | {keyword}"
                    if score is not None:
                        title += f" | score={score:.4f}"

                    with st.expander(title):
                        st.markdown("**doc_id**")
                        st.code(context.get("doc_id", ""), language="text")
                        st.markdown("**metadata**")
                        st.json(context_metadata)
                        st.markdown("**content**")
                        st.write(context.get("content", ""))
            else:
                st.info("No retrieved contexts returned.")

            prompt = metadata.get("prompt")
            if prompt:
                with st.expander("Final prompt"):
                    st.code(prompt, language="text")


def render_assistant_message(
    content: str,
    metadata: dict[str, Any] | None,
    show_debug: bool,
    message_id: str,
) -> None:
    st.markdown(
        render_message_html("assistant", content),
        unsafe_allow_html=True,
    )
    if show_debug and metadata:
        render_debug_info(metadata, message_id)


def render_chat_history(show_debug: bool) -> None:
    for index, message in enumerate(st.session_state.messages):
        role = message.get("role", "assistant")
        content = message.get("content", "")
        metadata = message.get("metadata")
        message_id = str(message.get("id", index))

        if role == "user":
            render_user_message(content)
        else:
            render_assistant_message(content, metadata, show_debug, message_id)


inject_custom_css()
init_session_state()


with st.sidebar:
    st.markdown("### Cài đặt")

    api_base_url = st.text_input(
        "API Base URL",
        value=API_BASE_URL,
    )

    show_debug = st.checkbox("Hiển thị RAG debug", value=True)

    st.divider()

    category = render_category_filter_buttons()

    keyword = st.text_input(
        "Keyword filter",
        value="",
        placeholder="bánh chưng",
    )

    st.divider()

    if st.button("Xóa hội thoại", width="stretch"):
        st.session_state.message_counter = 1
        st.session_state.messages = [
            {
                "id": "greeting",
                "role": "assistant",
                "content": GREETING,
                "metadata": None,
            }
        ]
        st.rerun()


st.markdown(
    """
    <div class="chat-shell">
        <div class="chat-title-wrap">
            <h1 class="chat-title">Vietnamese Cultural QA</h1>
            <p class="chat-caption">Text-only RAG + Qwen2.5 LoRA</p>
            <div class="scope-note">
                <strong>Phạm vi hỗ trợ:</strong> Các lĩnh vực
                Kiến trúc, Ẩm thực, Trang phục, Lễ hội, Nhạc cụ, Thể thao truyền thống.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_chat_history(show_debug)

question = st.chat_input("Hỏi về văn hóa Việt Nam...")

if question:
    user_message_id = next_message_id()
    assistant_message_id = next_message_id()

    st.session_state.messages.append(
        {
            "id": user_message_id,
            "role": "user",
            "content": question,
            "metadata": None,
        }
    )

    render_user_message(question)

    payload = build_payload(
        question=question,
        debug=show_debug,
        category=category,
        keyword=keyword,
    )

    answer_placeholder = st.empty()
    answer_placeholder.markdown(render_loading_html(), unsafe_allow_html=True)
    answer, metadata, error = stream_qa_response(
        api_base_url=api_base_url,
        payload=payload,
        placeholder=answer_placeholder,
    )

    if error:
        assistant_content = f"{answer}\n\n⚠️ {error}" if answer else f"⚠️ {error}"
        answer_placeholder.markdown(
            render_message_html("assistant", assistant_content),
            unsafe_allow_html=True,
        )
    else:
        assistant_content = answer or "Không có câu trả lời."

    if show_debug and metadata:
        render_debug_info(metadata, assistant_message_id)

    st.session_state.messages.append(
        {
            "id": assistant_message_id,
            "role": "assistant",
            "content": assistant_content,
            "metadata": metadata,
        }
    )
