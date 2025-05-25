# !pip install -U transformers
# !pip install --upgrade datasets

# BERT Example using Feature Extraction + Fine Tuning
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load pre-trained BERT model (without classification head)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# use AutoModel instead of AutoModelForSequenceClassification
bert_model = AutoModel.from_pretrained(model_name)  

# Freeze BERT parameters (Feature Extraction)
for param in bert_model.parameters():
    param.requires_grad = False 

print("BERT 기본 모델 파라미터 동결 완료")

dataset = load_dataset("imdb")
train_data = dataset["train"].shuffle(seed=42).select(range(300))
test_data = dataset["test"].shuffle(seed=42).select(range(20))

# Extract feature vectors from BERT
def extract_features(texts, labels):
    features = []
    
    for text in texts:
        inputs = tokenizer(text, padding="max_length", truncation=True, 
                         max_length=128, return_tensors="pt")
        
        # Extract features from BERT (no gradient calculation)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            # Use [CLS] token embedding as feature
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
            features.append(cls_embedding.numpy())
    
    return np.array(features), np.array(labels)

print("> 학습 데이터에서 특징 추출 중...")
X_train, y_train = extract_features(train_data["text"], train_data["label"])

print("> 테스트 데이터에서 특징 추출 중...")
X_test, y_test = extract_features(test_data["text"], test_data["label"])

print(f"> 특징 추출 완료!")
print(f"   - 학습 특징 shape: {X_train.shape}")
print(f"   - 테스트 특징 shape: {X_test.shape}")

# Train a simple classifier (Logistic Regression)
print("\n> 분류기 학습 중...")
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"> 학습 완료!")
print(f"> 정확도: {accuracy:.4f}")

print("\n" + "="*60)
print("> 예측 결과 샘플 (처음 5개)")
print("="*60)

label_map = {0: "부정", 1: "긍정"}

for i in range(min(5, len(y_test))):
    text = test_data["text"][i]
    true_label = y_test[i]
    pred_label = y_pred[i]
    
    short_text = text[:150] + "..." if len(text) > 150 else text
    
    print(f"\n[샘플 {i+1}]")
    print(f"텍스트: {short_text}")
    print(f"실제: {label_map[true_label]} | 예측: {label_map[pred_label]}")
    print(f"결과: {'O 정답' if true_label == pred_label else 'X 오답'}")

print(f"\n> 전체 결과: {sum(y_test == y_pred)}/{len(y_test)} 정답")

print("\n" + "="*60)
print("> Feature Extraction 방식의 장점")
print("="*60)
print("- 빠른 학습: BERT 파라미터 업데이트 없음")
print("- 적은 메모리: 그래디언트 저장 불필요") 
print("- 안정성: 사전 훈련된 특징 보존")
print("- 재사용성: 추출된 특징을 다양한 분류기에 활용 가능")
