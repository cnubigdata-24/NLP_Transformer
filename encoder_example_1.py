# !pip install --upgrade datasets transformers scikit-learn

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import os
import warnings
warnings.filterwarnings('ignore')

os.environ["WANDB_DISABLED"] = "true"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("=== Korean News Multi-Class Classification ===")

LABELS = {0: "정치", 1: "경제", 2: "사회", 3: "기술", 4: "문화", 5: "스포츠"}

print(f"Category mapping: {LABELS}")

# Load CSV file
print("\nLoading CSV dataset...")
df = pd.read_csv("korean_news_dataset.csv")

print(f"Total samples: {len(df)}")
print(f"Label distribution:")
for label_id, count in df['label'].value_counts().sort_index().items():
    print(f"  {label_id}: {LABELS[label_id]} - {count} samples")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print(f"\nTrain: {len(train_df)}, Validation: {len(val_df)}")

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(val_df.reset_index(drop=True))
})

# Load KLUE BERT model
print(f"\nLoading KLUE BERT model...")
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels=len(LABELS))

# Preprocessing function
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

print(f"Preprocessing data...")
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="no",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Setup trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print(f"\n=== Fine-tuning Start ===")
print(f"Categories: {list(LABELS.values())}")

# Execute fine-tuning
trainer.train()

print(f"\n=== Validation Set Evaluation ===")
eval_results = trainer.evaluate()
print(f"Results: {eval_results}")

predictions = trainer.predict(encoded_dataset["validation"])
predicted = np.argmax(predictions.predictions, axis=1)
actual = predictions.label_ids

final_accuracy = accuracy_score(actual, predicted)
correct_count = np.sum(predicted == actual)

print(f"\n=== Validation Results ===")
print(f"Accuracy: {final_accuracy:.4f} ({correct_count}/{len(predicted)})")

print(f"\n=== Per-Category Performance ===")
for label_id in range(len(LABELS)):
    mask = actual == label_id
    if mask.sum() > 0:
        category_accuracy = accuracy_score(actual[mask], predicted[mask])
        correct_cat = np.sum(predicted[mask] == actual[mask])
        total_cat = mask.sum()
        print(f"{label_id}({LABELS[label_id]}): {category_accuracy:.4f} ({correct_cat}/{total_cat})")

print(f"\n=== Testing on New News Headlines ===")

new_headlines = [
    "미국 연방의회에서 국방예산 증액안이 여야 합의로 통과되었습니다",           # 정치 (0)
    "삼성전자 주가가 실적 호조로 52주 최고가를 기록하며 코스피 상승을 이끌었습니다",  # 경제 (1)
    "정부가 청년층 취업 지원을 위한 새로운 복지정책을 발표했습니다",           # 사회 (2)
    "구글이 개발한 새로운 AI 칩이 기존 대비 10배 빠른 연산 성능을 보여줬습니다",   # 기술 (3)
    "아이돌 그룹 블랙핑크가 월드투어 콘서트를 성공적으로 마쳤습니다",           # 문화 (4)
    "김연아 선수가 국제빙상연맹 위원으로 선출되었습니다"                    # 스포츠 (5)
]

expected_labels = [0, 1, 2, 3, 4, 5]

inputs = tokenizer(new_headlines, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

model = model.to(device)
inputs = {key: value.to(device) for key, value in inputs.items()}

print(f"Model and inputs on device: {device}")

# Make predictions
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.nn.functional.softmax(logits, dim=-1)
    predicted_labels = torch.argmax(logits, dim=-1)

print(f"Predicting categories for 6 new headlines:")
print("=" * 80)

for i, headline in enumerate(new_headlines):
    pred_label = predicted_labels[i].item()
    confidence = predictions[i][pred_label].item()
    expected = expected_labels[i]
    
    print(f"[{i+1}] '{headline}'")
    print(f"    Predicted: {pred_label}({LABELS[pred_label]}) - Confidence: {confidence:.4f}")
    print(f"    Expected: {expected}({LABELS[expected]}) - {'O' if pred_label == expected else 'X'}")
    
    # Show top 2 predictions
    top2_indices = torch.topk(predictions[i], 2).indices
    print(f"    Top 2: {top2_indices[0].item()}({LABELS[top2_indices[0].item()]}) {predictions[i][top2_indices[0]].item():.3f}, "
          f"{top2_indices[1].item()}({LABELS[top2_indices[1].item()]}) {predictions[i][top2_indices[1]].item():.3f}")
    print("-" * 80)

predicted_labels_cpu = predicted_labels.cpu().numpy() if predicted_labels.is_cuda else predicted_labels.numpy()
new_accuracy = accuracy_score(expected_labels, predicted_labels_cpu)
new_correct = sum(1 for p, e in zip(predicted_labels_cpu, expected_labels) if p == e)

print(f"\n=== New Headlines Results ===")
print(f"Accuracy on new data: {new_accuracy:.4f} ({new_correct}/6)")
print(f"Fine-tuning completed on {device}!")
