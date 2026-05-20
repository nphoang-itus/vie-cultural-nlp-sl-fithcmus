"""
Hugging Face text generation client for RAG inference.

Uses Hugging Face Inference API via huggingface_hub.InferenceClient.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Sequence

from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

from src.utils.config import get_env

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HFGenerationConfig:
    model: str
    token: str | None = None
    timeout: int = 60


class HuggingFaceTextGenerator:
    """
    Thin wrapper around Hugging Face InferenceClient.
    """

    def __init__(self, config: HFGenerationConfig) -> None:
        if not config.model:
            raise ValueError("Model name is required for Hugging Face generation.")

        self.config = config
        self.client = InferenceClient(
            model=config.model,
            token=config.token,
            timeout=config.timeout,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        stop_sequences: Sequence[str] | None = None,
    ) -> str:
        """
        Generate text from a prompt.
        """
        prompt = str(prompt or "").strip()

        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        # === THÊM ĐOẠN NÀY ĐỂ BỌC PROMPT CHO CHUẨN QWEN ===
        if "<|im_start|>" not in prompt:
            prompt = (
                f"<|im_start|>system\nBạn là một chuyên gia văn hóa Việt Nam. Hãy dựa vào ngữ cảnh để trả lời chính xác.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        # =================================================

        try:
            return self.client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=list(stop_sequences) if stop_sequences else None,
            )
        except Exception as exc:
            logger.error("Hugging Face generation failed: %s", exc)
            error_type = type(exc).__name__

            if isinstance(exc, HfHubHTTPError) and getattr(exc, "response", None) is not None:
                status = exc.response.status_code
                body = exc.response.text
                raise RuntimeError(
                    f"Hugging Face generation failed ({error_type}) HTTP {status}: {body}"
                ) from exc

            raise RuntimeError(
                f"Hugging Face generation failed ({error_type}): {exc!r}"
            ) from exc


def build_hf_generator_from_env(
    model: str,
    *,
    token_env: str = "HF_API_TOKEN",
    timeout: int = 60,
) -> HuggingFaceTextGenerator:
    """
    Build generator from environment token.
    """
    token = get_env(token_env)
    return HuggingFaceTextGenerator(
        HFGenerationConfig(
            model=model,
            token=token,
            timeout=timeout,
        )
    )