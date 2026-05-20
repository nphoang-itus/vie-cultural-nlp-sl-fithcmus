import os
# Thay đổi đường dẫn này thành thư mục ổ cứng rộng rãi trên máy
os.environ["HF_HOME"] = "D:/AI_Models_Cache" 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

class LocalGPUQwenLoRAGenerator:
    """
    Generator chạy local trên GPU, hỗ trợ nén 4-bit (NF4) và nạp LoRA Adapter.
    """
    def __init__(self, base_model_id: str, lora_model_id: str) -> None:
        # Cấu hình nén 4-bit để tiết kiệm VRAM máy local
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Tải Tokenizer và Base Model từ Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Nạp các trọng số LoRA của Chi lên trên mô hình nền
        self.model = PeftModel.from_pretrained(base_model, lora_model_id)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        # Đảm bảo Prompt luôn chuẩn cấu trúc ChatML của Qwen
        if "<|im_start|>" not in prompt:
            prompt = (
                f"<|im_start|>system\nBạn là một chuyên gia văn hóa Việt Nam. Hãy dựa vào ngữ cảnh để trả lời chính xác.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        input_length = inputs.input_ids.shape[1]
        return self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()