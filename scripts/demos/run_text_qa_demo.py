"""
CLI demo for Vietnamese Cultural text-only QA with RAG.

Usage:

1. Single question mode:
    python3 scripts/run_text_qa_demo.py \
        --question "Bánh chưng có ý nghĩa gì trong ngày Tết?"

2. Interactive mode:
    python3 scripts/run_text_qa_demo.py

3. Show retrieved contexts:
    python3 scripts/run_text_qa_demo.py \
        --question "Lễ hội chọi trâu thường tổ chức ở đâu?" \
        --show-context
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.qa.schemas import QARequest
from src.qa.service import QAService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vietnamese Cultural text-only QA demo with RAG."
    )

    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Vietnamese cultural question. If omitted, interactive mode is used.",
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional category filter, e.g. am_thuc, le_hoi, kien_truc.",
    )

    parser.add_argument(
        "--keyword",
        type=str,
        default=None,
        help="Optional keyword filter, e.g. bánh chưng, lễ hội chọi trâu.",
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved contexts for debugging.",
    )

    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print final RAG prompt sent to the model.",
    )

    return parser.parse_args()


def print_response(
    response,
    *,
    show_context: bool = False,
    show_prompt: bool = False,
) -> None:
    print()
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(response.question)

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(response.answer)

    print()
    print("=" * 80)
    print("RETRIEVAL")
    print("=" * 80)
    print(f"Retrieval query: {response.retrieval_query}")
    print(f"Filters: {response.filters}")
    print(f"Retrieved contexts: {len(response.contexts)}")

    if response.contexts:
        print()
        print("Top contexts:")
        for idx, context in enumerate(response.contexts, start=1):
            keyword = context.metadata.get("keyword")
            category = context.metadata.get("category")
            print(
                f"[{idx}] score={context.score} | "
                f"keyword={keyword} | category={category} | doc_id={context.doc_id}"
            )

    if show_context:
        print()
        print("=" * 80)
        print("RETRIEVED CONTEXT CONTENT")
        print("=" * 80)

        for idx, context in enumerate(response.contexts, start=1):
            print()
            print(f"[Context {idx}]")
            print(f"doc_id: {context.doc_id}")
            print(f"score: {context.score}")
            print(f"metadata: {context.metadata}")
            print("content:")
            print(context.content)

    if show_prompt:
        print()
        print("=" * 80)
        print("FINAL PROMPT")
        print("=" * 80)
        print(response.prompt)


def run_single_question(
    service: QAService,
    *,
    question: str,
    category: str | None,
    keyword: str | None,
    show_context: bool,
    show_prompt: bool,
) -> None:
    request = QARequest(
        question=question,
        category=category,
        keyword=keyword,
    )

    response, timing = service.answer_with_timing(request)

    print_response(
        response,
        show_context=show_context,
        show_prompt=show_prompt,
    )

    print()
    print("=" * 80)
    print("TIMING")
    print("=" * 80)
    print(f"RAG:        {timing.rag_seconds:.2f}s")
    print(f"Generation: {timing.generation_seconds:.2f}s")
    print(f"Total:      {timing.total_seconds:.2f}s")


def run_interactive_mode(
    service: QAService,
    *,
    category: str | None,
    keyword: str | None,
    show_context: bool,
    show_prompt: bool,
) -> None:
    print()
    print("Vietnamese Cultural QA Demo")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:
        question = input("Question> ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        if not question:
            continue

        try:
            run_single_question(
                service,
                question=question,
                category=category,
                keyword=keyword,
                show_context=show_context,
                show_prompt=show_prompt,
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")


def main() -> None:
    args = parse_args()

    print("[1] Loading QA service...")
    start = time.time()
    service = QAService.from_config_file()
    print(f"[2] QA service loaded in {time.time() - start:.2f}s.")

    if args.question:
        run_single_question(
            service,
            question=args.question,
            category=args.category,
            keyword=args.keyword,
            show_context=args.show_context,
            show_prompt=args.show_prompt,
        )
    else:
        run_interactive_mode(
            service,
            category=args.category,
            keyword=args.keyword,
            show_context=args.show_context,
            show_prompt=args.show_prompt,
        )


if __name__ == "__main__":
    main()