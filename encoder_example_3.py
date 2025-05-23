import torch
import warnings
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os

warnings.simplefilter("ignore")
os.environ["WANDB_DISABLED"] = "true"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load dataset
dataset_name = "zeroshot/twitter-financial-news-sentiment"
dataset = load_dataset(dataset_name)
dataset = dataset["train"].select(range(1000))  

print("Dataset loaded with subset of 1000 samples")

# Prepare data
train_texts = dataset["text"]
train_labels = dataset["label"]

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)

# Load model and tokenizer
MODEL_NAME = "distilroberta-base"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.to(device)

# Set label mapping
model.config.id2label = {0: "하락장 (Bearish)", 1: "중립 (Neutral)", 2: "상승장 (Bullish)"}
model.config.label2id = {"하락장 (Bearish)": 0, "중립 (Neutral)": 1, "상승장 (Bullish)": 2}

# Prepare datasets
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

def tokenize_function(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Test examples
test_examples = [
    "Apple stock surges after strong earnings report.",
    "Tesla shares plummet amid production concerns.", 
    "Market remains stable with mixed signals.",
    "Amazon announces record quarterly profits.",
    "Oil prices decline due to oversupply issues."
]

def test_model(model, title):
    """Test model and return predictions"""
    print(f"\n=== {title} ===")
    model.eval()
    predictions = []
    
    for test_text in test_examples:
        inputs = tokenizer(
            test_text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=64
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()
            confidence = probabilities[0][predicted_label].item()
        
        predictions.append(predicted_label)
        print(f"Text: {test_text}")
        print(f"Predicted: {model.config.id2label[predicted_label]} (confidence: {confidence:.3f})")
        print("-" * 60)
    
    return predictions

print("파인튜닝 전: 분류기가 랜덤 초기화 상태입니다")
before_predictions = test_model(model, "파인튜닝 전 결과")

print("\n" + "="*50)
print("파인튜닝 시작!")
print("="*50)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    eval_strategy="epoch"
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 파인튜닝 실행!
training_result = trainer.train()

print("파인튜닝 완료!")
print(f"Training loss: {training_result.training_loss:.4f}")

# Evaluate
eval_result = trainer.evaluate()
print(f"Evaluation accuracy: {eval_result.get('eval_accuracy', 'N/A'):.3f}")

print("\n" + "="*50)
print("파인튜닝 후: 실제 금융 감정 분석 가능!")
print("="*50)
after_predictions = test_model(model, "파인튜닝 후 결과")

print("\n" + "="*50)
print("파인튜닝 전후 비교")
print("="*50)

label_names = ["하락장", "중립", "상승장"]
changes = 0

for i, (before, after, text) in enumerate(zip(before_predictions, after_predictions, test_examples)):
    if before != after:
        changes += 1
    
    print(f"{i+1}. {text[:40]}...")
    print(f"   전: {label_names[before]} → 후: {label_names[after]} {'X 다름' if before != after else 'O 동일'}")

print(f"\n총 {changes}/{len(test_examples)} 예측이 변경되었습니다")

model.save_pretrained("./roberta-financial")
tokenizer.save_pretrained("./roberta-financial")

print("\n모델이 './roberta-financial'에 저장되었습니다!")

print(f"\n=== 최종 요약 ===")
print(f"학습 데이터: {len(train_dataset)}개")
print(f"검증 데이터: {len(val_dataset)}개") 
print(f"최종 정확도: {eval_result.get('eval_accuracy', 'N/A'):.3f}")
print(f"예측 변화: {changes}/{len(test_examples)}개")
