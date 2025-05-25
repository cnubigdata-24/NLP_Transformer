# !pip install -U bitsandbytes

# QLoRA fine-tuning example (optimized, GPU only)

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

device = "cuda" if torch.cuda.is_available() else exit("GPU가 필요합니다. Colab에서 런타임 유형을 GPU로 변경하세요.")

# Load tokenizer and model (KoGPT2)
model_name = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '</s>'})

# QLoRA 4-bit quantization config
qlora_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load pretrained model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=qlora_config,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Configure LoRA for CausalLM task
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"]
)

# Apply LoRA to model and resize embeddings
model = get_peft_model(model, lora_config)
model.resize_token_embeddings(len(tokenizer))

dataset = load_dataset("json", data_files="/content/qlora_training_chat_data.json")

def tokenize_fn(examples):
    inputs = [f"사용자: {u}\n봇: {b}" + tokenizer.eos_token for u, b in zip(examples["user"], examples["bot"])]
    tokenized = tokenizer(inputs, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
    tokenized["labels"] = tokenized["input_ids"].clone()
    tokenized["labels"][tokenized["labels"] == tokenizer.pad_token_id] = -100
    return tokenized

tokenized_dataset = dataset["train"].map(tokenize_fn, batched=True, remove_columns=["user", "bot"])
train_size = int(0.9 * len(tokenized_dataset))
train_dataset = tokenized_dataset.select(range(train_size))
eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

training_args = TrainingArguments(
    output_dir="./qlora-kogpt2",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    report_to=[],
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.eval()

# Define inference function
def generate_response(prompt):
    input_text = f"사용자: {prompt}\n봇:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("봇:")[-1].strip() if "봇:" in response else response

test_inputs = ["서울 명소 추천해줘", "오늘 날씨 어때?", "기분이 안 좋아"]
for q in test_inputs:
    print(f"사용자: {q}")
    print(f"봇: {generate_response(q)}")
    print("-" * 40)
