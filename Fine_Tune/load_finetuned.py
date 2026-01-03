import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "google/gemma-3-1b-pt"
ADAPTER_DIR = "lora_adapter"

# 4bit 로딩(QLoRA)로도 테스트 가능
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    quantization_config=bnb_config,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

def gen(prompt: str, max_new_tokens=120, temperature=0.8, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ===== 테스트 프롬프트들 =====
tests = [
    "안녕?\n",
    "오늘 뭐해?\n",
    "너 성격이 어때?\n",
    "지금 기분이 어때?\n",
    "한 문장으로 자기소개해줘.\n",
]

for t in tests:
    print("="*60)
    print("PROMPT:", t.strip())
    print(gen(t))
