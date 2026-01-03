from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

BASE = "google/gemma-3-1b-pt"
DATA = "pretrain.jsonl"

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())

tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    load_in_4bit=True,              # QLoRA
    torch_dtype=torch.float16,
)

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

model = get_peft_model(model, lora)

ds = load_dataset("json", data_files=DATA, split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    dataset_text_field="text",
    packing=True,
    max_seq_length=512,
    args=TrainingArguments(
        output_dir="lora_out",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    ),
)

trainer.train()
trainer.model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")
print("Saved -> lora_adapter/")
