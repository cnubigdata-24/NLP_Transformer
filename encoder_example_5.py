!pip install -U transformers
!pip install --upgrade datasets

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score

import os
os.environ["WANDB_DISABLED"] = "true"

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

dataset = load_dataset("imdb")
train_dataset = dataset["train"].shuffle(seed=42).select(range(50))   
test_dataset = dataset["test"].shuffle(seed=42).select(range(20))    

original_test_data = [(example["text"], example["label"]) for example in test_dataset]

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])
train_dataset.set_format("torch")
test_dataset.set_format("torch")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,                  
    per_device_train_batch_size=8,        
    per_device_eval_batch_size=16,        
    eval_strategy="no",                   
    save_strategy="no",                  
    logging_steps=5,                      
    dataloader_pin_memory=False,          
    fp16=True,                           # Mixed precision
    remove_unused_columns=True,          
    report_to=None,                      
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("학습 시작...")
trainer.train()

print("평가 중...")
eval_results = trainer.evaluate()
print(f"최종 정확도: {eval_results['eval_accuracy']:.4f}")


print("\n" + "="*80)
print("실제 예측 결과 샘플 (처음 5개)")
print("="*80)

predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

label_map = {0: "부정(Negative)", 1: "긍정(Positive)"}

for i in range(min(5, len(original_test_data))):
    text, true_label = original_test_data[i]
    pred_label = predicted_labels[i]
    
    short_text = text[:200] + "..." if len(text) > 200 else text
    
    print(f"\n[샘플 {i+1}]")
    print(f"텍스트: {short_text}")
    print(f"실제 라벨: {label_map[true_label]}")
    print(f"예측 라벨: {label_map[pred_label]}")
    print(f"정답 여부: {'맞음' if true_label == pred_label else '틀림'}")
    print("-" * 60)

correct = sum(1 for i in range(len(predicted_labels)) if predicted_labels[i] == original_test_data[i][1])
total = len(predicted_labels)
print(f"\n전체 결과: {correct}/{total} 정답 ({correct/total*100:.1f}%)")

# model.save_pretrained("./bert-imdb-finetuned")
# tokenizer.save_pretrained("./bert-imdb-finetuned")
print("\n학습 및 평가 완료!")
