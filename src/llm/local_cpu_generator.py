"""
Local Hugging Face text generation (CPU only).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalCPUGenerationConfig:
    model: str
    lora_adapter: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True


class LocalCPUTextGenerator:
    """
    Local (CPU) text generator using Transformers.
    """

    def __init__(self, config: LocalCPUGenerationConfig) -> None:
        if not config.model:
            raise ValueError("Model name is required for local generation.")

        self.config = config
        dtype = self._resolve_dtype(config.dtype)

        logger.info("Loading local model: %s on %s", config.model, config.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            use_fast=config.use_fast_tokenizer,
            trust_remote_code=config.trust_remote_code,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        )

        if config.lora_adapter:
            logger.info("Loading LoRA adapter: %s", config.lora_adapter)
            base_model = PeftModel.from_pretrained(base_model, config.lora_adapter)

        self.model = base_model.to(config.device)

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        prompt = str(prompt or "").strip()

        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        do_sample = temperature > 0

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):].lstrip()

        return decoded.strip()

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype:
        name = (dtype or "float32").lower()
        if name in {"fp16", "float16"}:
            return torch.float16
        if name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        return torch.float32
