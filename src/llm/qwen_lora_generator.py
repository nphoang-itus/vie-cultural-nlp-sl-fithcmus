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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from collections.abc import Iterator
from threading import Thread

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

logger = logging.getLogger(__name__)


TEXT_ONLY_SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp văn bản về văn hóa Việt Nam.
Không có ảnh đầu vào trong phiên hỏi đáp này.
Trả lời trực tiếp vào nội dung câu hỏi, không nhắc tới ảnh, hình ảnh, trong ảnh, ảnh chụp, nhìn thấy, cho thấy hoặc mô tả thị giác.
Nếu ngữ cảnh truy xuất có các cụm thị giác, hãy chuyển chúng thành tri thức văn bản tự nhiên và loại bỏ mọi nhắc đến ảnh."""

_VISUAL_TERMS_PATTERN = re.compile(
    r"\b(?:hình ảnh|bức ảnh|ảnh chụp|trong ảnh|trong hình|nhìn thấy|cho thấy)\b",
    flags=re.IGNORECASE,
)
_LEADING_VISUAL_PATTERN = re.compile(
    r"^\s*(?:hình ảnh|bức ảnh|ảnh|ảnh chụp)"
    r"(?:\s+(?:này|đó|trên|chụp|về|của|thể hiện|phản ánh|cho thấy))*"
    r"\s*(?:cho thấy|thể hiện|phản ánh|mô tả|chụp|là)?\s*",
    flags=re.IGNORECASE,
)
_VISUAL_PHRASE_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:trong|qua|từ)\s+(?:hình ảnh|bức ảnh|ảnh chụp|ảnh|hình)\s*(?:này|đó)?\b", re.IGNORECASE), ""),
    (re.compile(r"\b(?:hình ảnh|bức ảnh|ảnh chụp|ảnh|hình)\s*(?:này|đó)?\s+(?:cho thấy|thể hiện|phản ánh|mô tả)\s+", re.IGNORECASE), ""),
    (re.compile(r"\b(?:có thể\s+)?(?:thấy|nhìn thấy)\s+(?:rằng\s+)?", re.IGNORECASE), ""),
    (re.compile(r"\bcho thấy\s+(?:rằng\s+)?", re.IGNORECASE), ""),
)


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

        text = self.tokenizer.apply_chat_template(
            self._build_messages(prompt),
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
            generated_ids = self.model.generate(
                **inputs,
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
    def _build_messages(prompt: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": TEXT_ONLY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    @staticmethod
    def _postprocess_answer(answer: str) -> str:
        """
        Clean generated answer and remove visual-VQA artifacts.
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

        return QwenLoraGenerator._sanitize_text_only_answer(answer)

    @staticmethod
    def _sanitize_text_only_answer(answer: str) -> str:
        """
        Best-effort runtime guardrail for VQA-biased answers.

        The adapter was trained on VQA-style labels, so prompts alone may still
        produce sentences such as "Hình ảnh cho thấy...". Keep the cultural
        content but remove visual framing before returning responses to users.
        """
        answer = str(answer or "").strip()
        if not answer:
            return ""

        protected = answer.replace("\n", " ")
        sentence_parts = re.split(r"(?<=[.!?。！？])\s+", protected)
        cleaned_parts: list[str] = []

        for part in sentence_parts:
            sentence = part.strip()
            if not sentence:
                continue

            sentence = _LEADING_VISUAL_PATTERN.sub("", sentence).strip()

            for pattern, replacement in _VISUAL_PHRASE_REPLACEMENTS:
                sentence = pattern.sub(replacement, sentence)

            sentence = re.sub(r"\s+", " ", sentence).strip(" ,;:-")
            sentence = re.sub(
                r"^(?:một|các|những)\s+(?:đĩa|bát|tô|phần|món|chiếc|cái|cảnh)\s+",
                "",
                sentence,
                flags=re.IGNORECASE,
            ).strip()
            sentence = re.sub(
                r"^([^,.!?]+?),\s+một\s+",
                r"\1 là một ",
                sentence,
                flags=re.IGNORECASE,
            ).strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]

            if not sentence or _VISUAL_TERMS_PATTERN.search(sentence):
                continue

            cleaned_parts.append(sentence)

        cleaned = " ".join(cleaned_parts).strip()
        cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
        cleaned = re.sub(r"\.{2,}", ".", cleaned)

        if cleaned:
            return cleaned

        fallback = _LEADING_VISUAL_PATTERN.sub("", answer).strip()
        for pattern, replacement in _VISUAL_PHRASE_REPLACEMENTS:
            fallback = pattern.sub(replacement, fallback)
        fallback = re.sub(r"\s+", " ", fallback).strip(" ,;:-")

        return fallback

    def stream_generate(self, prompt: str) -> Iterator[str]:
        """
        Stream generated answer chunks with sentence-level sanitization.

        Visual phrases can span multiple model chunks, so raw token passthrough
        can leak "Hình ảnh..." before post-processing sees the whole phrase.
        Buffer one sentence at a time, sanitize it, then emit word-like chunks.
        """
        prompt = str(prompt or "").strip()

        if not prompt:
            raise ValueError("prompt cannot be empty.")

        text = self.tokenizer.apply_chat_template(
            self._build_messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs: dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if self.config.do_sample:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p

        thread = Thread(
            target=self.model.generate,
            kwargs=generation_kwargs,
        )
        thread.start()

        buffer = ""
        emitted_any = False

        for token in streamer:
            buffer += token

            complete_sentences, buffer = self._pop_complete_sentences(buffer)
            if not complete_sentences:
                continue

            sanitized = self._sanitize_text_only_answer(complete_sentences)
            for chunk in self._iter_display_chunks(sanitized, leading_space=emitted_any):
                emitted_any = True
                yield chunk

        thread.join()

        sanitized_tail = self._sanitize_text_only_answer(buffer)
        for chunk in self._iter_display_chunks(sanitized_tail, leading_space=emitted_any):
            emitted_any = True
            yield chunk

    @staticmethod
    def _pop_complete_sentences(text: str) -> tuple[str, str]:
        """
        Split completed sentences from a streaming buffer.
        """
        matches = list(re.finditer(r"[.!?。！？]\s+", text))
        if not matches:
            return "", text

        split_at = matches[-1].end()
        return text[:split_at], text[split_at:]

    @staticmethod
    def _iter_display_chunks(text: str, *, leading_space: bool) -> Iterator[str]:
        """
        Yield whitespace-preserving word chunks for the SSE UI.
        """
        text = str(text or "").strip()
        if not text:
            return

        prefix = " " if leading_space else ""
        chunks = text.split()

        for index, chunk in enumerate(chunks):
            if index == 0:
                yield f"{prefix}{chunk}"
            else:
                yield f" {chunk}"
