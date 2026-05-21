import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.qwen_lora_generator import QwenLoraGenerator

def main() -> None:
    print("[1] Loading generator...")
    t0 = time.time()
    generator = QwenLoraGenerator.from_config_file()
    print(f"[2] Generator loaded in {time.time() - t0:.2f}s.")

    prompt = """Bạn là trợ lý trả lời câu hỏi về văn hóa Việt Nam.

Câu hỏi:
Lễ hội chọi châu thường được tổ chức ở đâu?

Câu trả lời:"""

    print("[3] Generating answer...")
    t1 = time.time()
    answer = generator.generate(prompt)
    print(f"[4] Generation finished in {time.time() - t1:.2f}s.")

    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()