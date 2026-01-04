from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import math, time, os, json


# =========================================================
# Callback: step마다 샘플 generate 저장 (생성 토큰만 저장)
# =========================================================
class SaveGenerationsCallback(TrainerCallback):
    """
    training 중 일정 step마다 고정 프롬프트로 generate 후 결과를 jsonl로 append 저장
    - 프롬프트까지 포함해서 디코딩하지 않고 "새로 생성된 토큰만" 저장
    """
    def __init__(
        self,
        tokenizer,
        prompts,
        out_path="gen_samples.jsonl",
        every_steps=50,
        max_new_tokens=80,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    ):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.out_path = out_path
        self.every_steps = every_steps
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0:
            return control
        if state.global_step % self.every_steps != 0:
            return control

        model = kwargs["model"]

        # DDP면 rank0만 저장
        if hasattr(args, "local_rank") and args.local_rank not in (-1, 0):
            return control

        device = next(model.parameters()).device
        model_was_training = model.training
        model.eval()

        results = []
        with torch.no_grad():
            for p in self.prompts:
                inputs = self.tokenizer(p, return_tensors="pt").to(device)

                gen_kwargs = dict(
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                if self.do_sample:
                    gen_kwargs["temperature"] = self.temperature
                    gen_kwargs["top_p"] = self.top_p

                out_ids = model.generate(**inputs, **gen_kwargs)

                # ✅ "생성된 부분만" 디코딩
                inp_len = inputs["input_ids"].shape[1]
                gen_only = out_ids[0, inp_len:]
                text = self.tokenizer.decode(gen_only, skip_special_tokens=True)

                results.append({"prompt": p, "generated": text})

        if model_was_training:
            model.train()

        rec = {
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples": results,
        }

        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return control


# =========================================================
# Trainer: loss + grad_norm + lr 출력
# =========================================================
class SFTTrainerWithGradNorm(SFTTrainer):
    def __init__(self, *args, log_grad_norm=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_grad_norm = log_grad_norm

    def _get_total_grad_norm(self) -> float:
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return 0.0

        total_norm_sq = 0.0
        for p in parameters:
            grad = p.grad.detach()
            if not torch.isfinite(grad).all():
                return float("nan")
            param_norm = grad.float().norm(2).item()
            total_norm_sq += param_norm ** 2
        return math.sqrt(total_norm_sq)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.state.global_step % self.args.logging_steps == 0:
            log_dict = {}
            try:
                log_dict["loss"] = float(loss.detach().cpu())
            except Exception:
                pass

            if self.log_grad_norm:
                log_dict["grad_norm"] = self._get_total_grad_norm()

            if self.optimizer is not None and len(self.optimizer.param_groups) > 0:
                log_dict["lr"] = self.optimizer.param_groups[0].get("lr", None)

            self.log(log_dict)

        return loss


# ================================
# 0) 설정
# ================================
BASE = "google/gemma-3-1b-pt"
DATA = r"E:\VRC_AI2\Fine_Tune\make_data\lumi_name_1000_messages.jsonl"

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())

# ================================
# 1) 토크나이저 로드
# ================================
tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================================
# 2) QLoRA BitsAndBytes 설정
# ================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32,
)

# ================================
# 3) 모델 로드
# ================================
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    quantization_config=bnb_config,
)
model.config.use_cache = False

# ================================
# 4) k-bit 학습 준비
# ================================
model = prepare_model_for_kbit_training(model)

# ================================
# 5) LoRA 설정
# ================================
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)

# ================================
# 6) 데이터 로드
# ================================
ds = load_dataset("json", data_files=DATA, split="train")

# ================================
# 7) messages -> text 컬럼 생성 (batched=True 대응)
# ================================
def _render_one(messages):
    # messages: list[dict] 가 정상
    if not isinstance(messages, list):
        return ""

    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip()
        content = ((m.get("content") or "")).strip()

        if role == "system":
            parts.append(f"System: {content}\n")
        elif role == "user":
            parts.append(f"User: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}{tokenizer.eos_token}\n")

    return "".join(parts).strip()

def build_text_column(batch):
    # batch["messages"] 는 list of messages (batched=True)
    texts = []
    for msgs in batch.get("messages", []):
        texts.append(_render_one(msgs))
    return {"text": texts}

# batched=True로 안전하게 text 만들기
ds = ds.map(build_text_column, batched=True, desc="Build text column")

# 빈 text 제거
ds = ds.filter(lambda ex: isinstance(ex["text"], str) and ex["text"].strip() != "")
print("Filtered dataset size:", len(ds))

# ================================
# 8) Completion-Only Collator (Assistant만 loss)
# ================================
# render_train_text에서 "Assistant: "로 응답이 시작하므로 response_template가 정확히 매칭되어야 함
response_template = "Assistant:"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)


# ================================
# 9) Trainer 설정
# ================================
trainer = SFTTrainerWithGradNorm(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,

    # ✅ 핵심: Assistant 부분만 loss 계산
    dataset_text_field="text",
    data_collator=collator,

    # ✅ 권장: 디버깅/안정화를 위해 packing=False
    packing=False,
    max_seq_length=256,

    args=TrainingArguments(
        output_dir="lora_out",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,

        # 발산 방지 목적이면 낮추는 편 추천
        learning_rate=2e-5,
        num_train_epochs=3,

        warmup_ratio=0.03,
        max_grad_norm=1.0,

        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=5,

        fp16=False,
        optim="paged_adamw_8bit",

        report_to="none",
    ),
    log_grad_norm=True,
)

# ================================
# 10) 샘플 출력 저장 콜백
# ================================
sample_prompts = [
    "System: 너는 루미다.\nUser: 한 문장으로 자기소개해줘.\nAssistant: ",
    "System: 너는 루미다.\nUser: 네 이름이 뭐야?\nAssistant: ",
]

trainer.add_callback(
    SaveGenerationsCallback(
        tokenizer=tokenizer,
        prompts=sample_prompts,
        out_path="lora_out/gen_samples.jsonl",
        every_steps=10,
        max_new_tokens=80,
        do_sample=False,
    )
)

# ================================
# 11) 학습 및 저장
# ================================
trainer.train()

trainer.model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")
print("Saved -> lora_adapter/")
