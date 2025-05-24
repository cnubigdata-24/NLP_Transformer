!pip install peft datasets transformers accelerate

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import evaluate
import numpy as np

# 모델 및 토크나이저 로딩
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# LoRA 설정 및 적용
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(base_model, peft_config)

# 데이터셋 로딩 및 전처리
dataset = load_dataset("nsmc")
train_dataset = dataset["train"].shuffle(seed=42).select(range(3000)) 
test_dataset = dataset["test"].shuffle(seed=42).select(range(500))

def preprocess_function(examples):
    return tokenizer(examples["document"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./lora_nsmc_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    fp16=True  # GPU mixed precision
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

trainer.train()
