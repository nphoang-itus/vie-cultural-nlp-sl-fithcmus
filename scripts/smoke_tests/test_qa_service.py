import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.qa.schemas import QARequest
from src.qa.service import QAService


def main() -> None:
    print("[1] Loading QA service...")
    t0 = time.time()
    service = QAService.from_config_file()
    print(f"[2] QA service loaded in {time.time() - t0:.2f}s.")

    request = QARequest(
        question="Lễ hội chọi châu ở Việt Nam thường được tổ chức ở đâu?"
    )

    print("[3] Running QA pipeline...")
    response, timing = service.answer_with_timing(request)

    print("[4] QA finished.")
    print()

    print("Question:")
    print(response.question)
    print()

    print("Retrieval query:")
    print(response.retrieval_query)
    print()

    print("Filters:")
    print(response.filters)
    print()

    print("Retrieved contexts:")
    for idx, context in enumerate(response.contexts, start=1):
        print(f"[{idx}] doc_id={context.doc_id}")
        print(f"    keyword={context.metadata.get('keyword')}")
        print(f"    category={context.metadata.get('category')}")
        print(f"    score={context.score}")
        print()

    print("Answer:")
    print(response.answer)
    print()

    print("Timing:")
    print(f"RAG:        {timing.rag_seconds:.2f}s")
    print(f"Generation: {timing.generation_seconds:.2f}s")
    print(f"Total:      {timing.total_seconds:.2f}s")


if __name__ == "__main__":
    main()