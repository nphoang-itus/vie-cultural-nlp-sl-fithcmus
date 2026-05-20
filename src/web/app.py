from __future__ import annotations

import requests
import streamlit as st
import pandas as pd
import json


API_BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Vietnamese Cultural QA Demo",
    page_icon="🇻🇳",
    layout="wide",
)


st.title("Vietnamese Cultural QA Demo")
st.caption("Text-only RAG + Qwen2.5 LoRA")


with st.sidebar:
    st.header("Settings")

    api_base_url = st.text_input(
        "API Base URL",
        value=API_BASE_URL,
    )

    debug = st.checkbox(
        "Show RAG debug info",
        value=True,
    )

    st.divider()

    category = st.text_input(
        "Category filter",
        value="",
        placeholder="Example: am_thuc",
    )

    keyword = st.text_input(
        "Keyword filter",
        value="",
        placeholder="Example: bánh chưng",
    )


question = st.text_area(
    "User question",
    value="Bánh chưng có ý nghĩa gì trong ngày Tết?",
    height=100,
)


submit = st.button(
    "Generate Answer",
    type="primary",
    width='stretch',
)


if submit:
    question = question.strip()

    if not question:
        st.error("Please enter a question.")
        st.stop()

    payload = {
        "question": question,
        "debug": debug,
    }

    if category.strip():
        payload["category"] = category.strip()

    if keyword.strip():
        payload["keyword"] = keyword.strip()

    try:
        answer_placeholder = st.empty()
        answer_text = ""
        metadata = None
        timing_data = None

        with st.spinner("Generating answer..."):
            response = requests.post(
                f"{api_base_url}/api/qa/stream",
                json=payload,
                stream=True,
                timeout=120,
            )

            if response.status_code != 200:
                st.error(f"API error {response.status_code}")
                st.code(response.text)
                st.stop()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if not line.startswith("data: "):
                    continue

                raw = line.removeprefix("data: ").strip()

                if not raw:
                    continue

                if raw == "[DONE]":
                    break

                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                if event_type == "metadata":
                    metadata = event

                elif event_type == "token":
                    answer_text += event.get("token", "")
                    answer_placeholder.success(answer_text)

                elif event_type == "done":
                    timing_data = event.get("timing")
                    break

                elif event_type == "error":
                    st.error(event.get("message", "Unknown streaming error."))
                    st.stop()

        # Use metadata from streaming, fallback to empty dict
        data = metadata if metadata else {}

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI backend. Make sure uvicorn is running.")
        st.stop()

    except requests.exceptions.Timeout:
        st.error("Request timed out. The model may still be generating.")
        st.stop()

    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        st.stop()

    st.subheader("Answer")
    st.success(answer_text)

    st.subheader("Runtime")

    timing = timing_data if timing_data else {}

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "RAG",
        f"{timing.get('rag_seconds', 0):.2f}s",
    )

    col2.metric(
        "Generation",
        f"{timing.get('generation_seconds', 0):.2f}s",
    )

    col3.metric(
        "Total",
        f"{timing.get('total_seconds', 0):.2f}s",
    )

    if debug and metadata:
        st.divider()
        st.subheader("RAG Debug Info")

        st.markdown("#### Retrieval Query")
        st.code(metadata.get("retrieval_query", ""), language="text")

        st.markdown("#### Filters")

        filters = metadata.get("filters")

        if filters:
            st.json(filters)
        else:
            st.info("No filters applied.")

        contexts = metadata.get("contexts", [])

        if contexts:
            st.markdown("#### Retrieved Contexts")

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
                        "content_preview": str(context.get("content", ""))[:160],
                    }
                )

            df = pd.DataFrame(rows)
            st.dataframe(df, width='stretch')

            for index, context in enumerate(contexts, start=1):
                context_metadata = context.get("metadata", {}) or {}

                title = (
                    f"Context {index} | "
                    f"{context_metadata.get('keyword', '')} | "
                    f"score={context.get('score'):.4f}"
                    if context.get("score") is not None
                    else f"Context {index}"
                )

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
            st.markdown("#### Final Prompt")
            with st.expander("Show final prompt"):
                st.code(prompt, language="text")