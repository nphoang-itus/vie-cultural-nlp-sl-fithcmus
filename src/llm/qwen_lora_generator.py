"""
Qwen2.5 + LoRA generator for Vietnamese Cultural QA.

This module loads:
- Base instruction model
- PEFT LoRA adapter
- Tokenizer

It exposes a small stable interface:
    generator.generate(prompt) -> answer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationConfig:
    provider: str = "huggingface_peft"
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    adapter_model: str = "ohthisischichi/viet-cultural-qa-qwen2.5-lora"
    device: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9
    do_sample: bool = False


def resolve_generation_device(device: str = "auto") -> str:
    """
    Resolve generation device.

    Priority:
    - cuda
    - mps
    - cpu
    """
    device = (device or "auto").lower()

    if device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def resolve_torch_dtype(torch_dtype: str, device: str) -> torch.dtype | str:
    """
    Resolve dtype safely.

    Notes:
    - float16 is usually fine on CUDA.
    - MPS can be sensitive depending on model/operators.
    - CPU should use float32 unless memory is a problem.
    """
    value = (torch_dtype or "auto").lower()

    if value == "auto":
        if device == "cuda":
            return torch.float16
        return torch.float32

    if value in {"float16", "fp16"}:
        return torch.float16

    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16

    if value in {"float32", "fp32"}:
        return torch.float32

    return "auto"


class QwenLoraGenerator:
    """
    Text generator using Qwen2.5 base model + LoRA adapter.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.device = resolve_generation_device(config.device)
        self.dtype = resolve_torch_dtype(config.torch_dtype, self.device)
        
        print(f"[Device] generation device = {self.device}")
        print(f"[DType] torch dtype = {self.dtype}")
        print(f"[MPS available] {torch.backends.mps.is_available()}")
        print(f"[CUDA available] {torch.cuda.is_available()}")

        logger.info("Loading tokenizer: %s", config.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            trust_remote_code=True,
        )

        logger.info(
            "Loading base model: %s | device=%s | dtype=%s",
            config.base_model,
            self.device,
            self.dtype,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )

        logger.info("Loading LoRA adapter: %s", config.adapter_model)
        self.model = PeftModel.from_pretrained(
            self.model,
            config.adapter_model,
        )

        self.model.to(self.device)
        self.model.eval()

        logger.info("Qwen LoRA generator is ready.")

    @classmethod
    def from_config_file(
        cls,
        config_path: Path = Path("configs/rag.yaml"),
    ) -> "QwenLoraGenerator":
        with config_path.open("r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}

        generation_cfg = raw_config.get("generation", {})

        config = GenerationConfig(
            provider=str(generation_cfg.get("provider", "huggingface_peft")),
            base_model=str(generation_cfg.get("base_model", "Qwen/Qwen2.5-3B-Instruct")),
            adapter_model=str(
                generation_cfg.get(
                    "adapter_model",
                    "ohthisischichi/viet-cultural-qa-qwen2.5-lora",
                )
            ),
            device=str(generation_cfg.get("device", "auto")),
            torch_dtype=str(generation_cfg.get("torch_dtype", "auto")),
            max_new_tokens=int(generation_cfg.get("max_new_tokens", 256)),
            temperature=float(generation_cfg.get("temperature", 0.0)),
            top_p=float(generation_cfg.get("top_p", 0.9)),
            do_sample=bool(generation_cfg.get("do_sample", False)),
        )

        return cls(config)

    def generate(self, prompt: str) -> str:
        """
        Generate answer from a full prompt.
        """
        prompt = str(prompt or "").strip()

        if not prompt:
            raise ValueError("prompt cannot be empty.")

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if self.config.do_sample:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p

        with torch.no_grad():
            streamer = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generated_ids = self.model.generate(
                **inputs,
                streamer=streamer,
                **generation_kwargs,
            )

        input_token_count = inputs["input_ids"].shape[-1]
        new_tokens = generated_ids[0][input_token_count:]

        answer = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        return self._postprocess_answer(answer)

    @staticmethod
    def _postprocess_answer(answer: str) -> str:
        """
        Minimal cleanup for generated answer.
        """
        answer = str(answer or "").strip()

        prefixes = [
            "Câu trả lời:",
            "Trả lời:",
            "Answer:",
        ]

        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()

        return answer