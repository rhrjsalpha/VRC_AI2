# thinking.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time, glob, json
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
import ollama
import re
# ===== 옵션 스위치 =====
USE_LOCAL_LORA = False   # True면 로컬 LoRA(Transformers/PEFT), False면 Ollama 사용

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

def strip_think(text: str) -> str:
    """
    모델 출력에서 <think>...</think> 블록 제거.
    - 여러 개 블록이 있어도 전부 제거
    - 태그가 깨진 경우(닫힘 없음 등)도 일부 방어
    """
    if not text:
        return text

    # 1) 정상 블록 제거
    text = _THINK_BLOCK_RE.sub("", text)

    # 2) 방어: <think>만 있고 </think>가 없는 경우, 그 뒤 전부 제거
    #    (스트리밍 중 태그가 잘린 채로 합쳐지는 케이스)
    lo = text.lower()
    i = lo.find("<think>")
    if i != -1:
        text = text[:i]

    # 3) 방어: </think>만 남아있는 경우 제거
    text = re.sub(r"</think>", "", text, flags=re.IGNORECASE)

    return text.strip()


def model_thinking_ollama(model: str, ask: str) -> str:
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": ask}],
        stream=True,
    )
    parts = []
    for chunk in stream:
        delta = chunk.get("message", {}).get("content", "")
        if delta:
            parts.append(delta)

    raw = "".join(parts)
    return strip_think(raw)

# ===== Local LoRA 백엔드 =====
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

BASE = "google/gemma-3-1b-pt"
ADAPTER_DIR = r"E:\VRC_AI2\Fine_Tune\lora_adapter"

print("BASE =", BASE)
print("ADAPTER_DIR =", ADAPTER_DIR)
print("ADAPTER exists?", os.path.exists(ADAPTER_DIR))
print("ADAPTER files:", glob.glob(os.path.join(ADAPTER_DIR, "*"))[:20])

cfg = os.path.join(ADAPTER_DIR, "adapter_config.json")
print("adapter_config.json exists?", os.path.exists(cfg))
if os.path.exists(cfg):
    with open(cfg, "r", encoding="utf-8") as f:
        j = json.load(f)
    print("adapter_config base_model_name_or_path:", j.get("base_model_name_or_path"))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32,
)

_tokenizer = None
_model = None

def _load_lora_once():
    """프로세스 시작 후 1회만 로딩 (매 호출 로딩 금지)"""
    global _tokenizer, _model
    if _model is not None:
        return

    _tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE,
        device_map="auto",
        quantization_config=bnb_config,
        attn_implementation="eager",     # ✅ Windows SDPA NaN 우회
    )
    base_model.config.use_cache = True
    base_model = prepare_model_for_kbit_training(base_model)

    _model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    _model.eval()

@torch.no_grad()
def model_thinking_local_lora(model: str, ask: str) -> str:
    """
    LTS_chain이 model 인자를 넘기므로 시그니처는 유지.
    model은 무시해도 됨.
    """
    _load_lora_once()

    system = "너는 AI 어시스턴트 '루미(Lumi)'다. 사용자가 이름/정체성을 물으면 반드시 '루미'라고 답한다."
    prompt = f"System: {system}\nUser: {ask}\nAssistant: "

    inputs = _tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    out = _model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.05,
        pad_token_id=_tokenizer.pad_token_id,
        eos_token_id=_tokenizer.eos_token_id,
    )
    gen_ids = out[0][in_len:]
    return _tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

# ===== 라우터(외부에서 호출하는 함수는 이거 하나) =====
def model_thinking(model: str, ask: str) -> str:
    """
    LTS_chain에서는 이 함수만 호출하게 유지.
    USE_LOCAL_LORA 스위치로 백엔드 선택.
    """
    if USE_LOCAL_LORA:
        return model_thinking_local_lora(model=model, ask=ask)
    else:
        return model_thinking_ollama(model=model, ask=ask)

if __name__ == "__main__":
    import time

    print("=== thinking.py self-test ===")
    print("USE_LOCAL_LORA =", USE_LOCAL_LORA)
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())

    # 로컬 LoRA 쓸 때는 첫 호출이 로딩 때문에 느릴 수 있음
    tests = [
        "안녕?",
        "네 이름이 뭐야?",
        "너 자기소개를 3문장으로 해줘.",
        "What is your name?"
    ]

    for t in tests:
        print("=" * 60)
        print("PROMPT:", t)
        st = time.time()
        try:
            # model 인자는 LTS_chain 호환용. 로컬 LoRA일 땐 내부에서 무시해도 됨.
            ans = model_thinking(model="gemma3:1b", ask=t)
            print("ANSWER:", ans)
        except Exception as e:
            print("ERROR:", repr(e))
        print("TIME:", time.time() - st)
