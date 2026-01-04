# load_finetuned.py
# -*- coding: utf-8 -*-
"""
LoRA adapter(QLoRA 학습 결과) 로딩 후 추론 테스트 스크립트.
- Windows + SDPA 경로에서 NaN/logits 붕괴 -> <pad>만 나오는 문제를 피하기 위해
  1) attn_implementation="eager" 강제 (SDPA 우회)
  2) (필요시) prepare_model_for_kbit_training 적용 (LayerNorm fp32 안정화 등)
  3) generate 옵션 단순화
"""

import os
import glob
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from safetensors.torch import load_file

# ==============================
# 0) 설정
# ==============================
BASE = "google/gemma-3-1b-pt"
ADAPTER_DIR = r"E:\VRC_AI2\Fine_Tune\lora_adapter"   # ✅ raw string 권장

# ==============================
# 1) 4bit 로딩 설정
# ==============================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32,  # 안정성 우선
)

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())

# ==============================
# 2) tokenizer
# ==============================
tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("pad_token_id:", tokenizer.pad_token_id, "eos_token_id:", tokenizer.eos_token_id)

# ==============================
# 3) base model 로드 (SDPA 우회: eager)
# ==============================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    quantization_config=bnb_config,
    attn_implementation="eager",  # ✅ SDPA/flash 경로 우회 (Windows에서 NaN 회피용)
)

# 추론은 cache 켜는게 보통 유리
base_model.config.use_cache = True

# ✅ 안정화(훈련 때와 동일한 전처리; NaN 회피에 도움될 때가 많음)
base_model = prepare_model_for_kbit_training(base_model)

# ==============================
# 4) LoRA adapter 로드
# ==============================
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

# ==============================
# 5) adapter weight non-finite 검사
# ==============================
def check_adapter_for_nonfinite(adapter_dir):
    st = glob.glob(os.path.join(adapter_dir, "*.safetensors"))
    if st:
        sd = load_file(st[0])
    else:
        binp = glob.glob(os.path.join(adapter_dir, "*.bin"))
        if not binp:
            raise FileNotFoundError(f"No adapter weights found in: {adapter_dir}")
        sd = torch.load(binp[0], map_location="cpu")

    bad = []
    for k, v in sd.items():
        if torch.is_tensor(v) and (not torch.isfinite(v).all()):
            bad.append(k)
    print("Non-finite tensors:", bad[:20], f"(count={len(bad)})")
    return bad

check_adapter_for_nonfinite(ADAPTER_DIR)

# ==============================
# 6) 학습 템플릿과 동일한 prompt 구성
# ==============================
def build_prompt_fallback(messages):
    """
    학습 때 사용한 텍스트 포맷과 동일:
      System: ...
      User: ...
      Assistant: ...<eos>
    추론은 마지막에 Assistant: 로 열어줌.
    """
    parts = []
    for m in messages:
        role = (m.get("role") or "").strip()
        content = ((m.get("content") or "")).strip()

        if role == "system":
            parts.append(f"System: {content}\n")
        elif role == "user":
            parts.append(f"User: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}{tokenizer.eos_token}\n")

    parts.append("Assistant: ")  # generation prompt
    return "".join(parts)

# ==============================
# 7) 생성 + 디버그 출력
# ==============================
@torch.no_grad()
def gen_chat(user_text: str, max_new_tokens=120):
    messages = [
        {
            "role": "system",
            "content": "너는 AI 어시스턴트 '루미(Lumi)'다. 사용자가 이름/정체성을 물으면 반드시 '루미'라고 답한다."
        },
        {"role": "user", "content": user_text},
    ]

    prompt = build_prompt_fallback(messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    # --- (A) 프롬프트 logits 먼저 점검 ---
    o = model(**inputs)
    logits = o.logits
    print("prompt logits finite?", torch.isfinite(logits).all().item())
    print("prompt logits min/max:", logits.nan_to_num().min().item(), logits.nan_to_num().max().item())

    # --- (B) generate (단순/안정 설정) ---
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_ids = out[0][in_len:]
    print("gen_len:", gen_ids.numel())
    print("first_ids:", gen_ids[:20].tolist())
    print("decoded(raw):", tokenizer.decode(gen_ids, skip_special_tokens=False))
    print("decoded(clean):", tokenizer.decode(gen_ids, skip_special_tokens=True))

    # --- (C) 생성 결과 포함 logits 점검 ---
    o2 = model(input_ids=out)
    logits2 = o2.logits
    print("gen logits finite?", torch.isfinite(logits2).all().item())
    print("gen logits min/max:", logits2.nan_to_num().min().item(), logits2.nan_to_num().max().item())

    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

# ==============================
# 8) 테스트
# ==============================
tests = ["안녕?", "한 문장으로 자기소개해줘.", "What is your name?", "네 이름이 루미 맞아?"]

for t in tests:
    start=time.time()
    print("=" * 60)
    print("PROMPT:", t)
    try:
        ans = gen_chat(t, max_new_tokens=120)
        print("\n---\nANSWER:", ans)
    except Exception as e:
        print("ERROR:", repr(e))
    end=time.time()
    print("TIME:", end-start)

