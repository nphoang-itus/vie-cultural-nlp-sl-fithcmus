import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

if VENV_PYTHON.exists() and sys.prefix == sys.base_prefix:
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

sys.path.insert(0, str(REPO_ROOT))
from src.rag.rag_service import RagQAInput, RagService

def main() -> None:
    service = RagService.from_config_file()

    result = service.prepare_prompt(
        RagQAInput(
            question="Bánh chưng có ý nghĩa gì trong ngày Tết?"
        )
    )

    print("Retrieval query:")
    print(result.retrieval_query)
    print()

    print("Filters:")
    print(result.filters)
    print()

    print("Retrieved contexts:")
    for idx, ctx in enumerate(result.rag_context.contexts, start=1):
        print(f"[{idx}] doc_id={ctx.doc_id}")
        print(f"    keyword={ctx.metadata.get('keyword')}")
        print(f"    category={ctx.metadata.get('category')}")
        print(f"    score={ctx.score}")
        print()

    print("Prompt preview:")
    print(result.prompt[:2000])


if __name__ == "__main__":
    main()
