# !pip install torch transformers scikit-learn numpy

# Example: Adapter Layer
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Adapter Module
class AdapterModule(nn.Module):
    """
    - Down projection: 차원 축소 (768 → 64)
    - Activation: 비선형 변환 (ReLU/GELU)  
    - Up projection: 차원 복원 (64 → 768)
    - Residual: 원본과 더하기
    """
    def __init__(self, input_dim=768, adapter_dim=64, activation='relu'):
        super().__init__()
        
        # Core of Adapter: Bottleneck structure
        self.down_project = nn.Linear(input_dim, adapter_dim)
        self.up_project = nn.Linear(adapter_dim, input_dim)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()
            
        # Weight initialization (start with small values)
        nn.init.normal_(self.down_project.weight, std=0.02)
        nn.init.normal_(self.up_project.weight, std=0.02)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)
    
    # Adapter의 Forward Passf: x: (batch_size, seq_len, hidden_dim)
    def forward(self, x):
        residual = x
        
        # 768 → 64 → ReLU → 64 → 768
        adapter_output = self.down_project(x)      # 차원 축소
        adapter_output = self.activation(adapter_output)  # 비선형 변환
        adapter_output = self.up_project(adapter_output)  # 차원 복원
        
        # Residual Connection
        return residual + adapter_output

    # Calculate number of adapter parameters
    def get_adapter_params(self):
       
        return sum(p.numel() for p in self.parameters())

# BERT + Adapter Model: BERT model with Adapter insertion
class BERTWithAdapter(nn.Module):   
    def __init__(self, model_name="bert-base-uncased", num_labels=2, adapter_dim=64):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Add Adapter to each BERT layer
        self.adapters = nn.ModuleList([
            AdapterModule(768, adapter_dim) 
            for _ in range(self.bert.config.num_hidden_layers)  
        ])
        
        # Classification head
        self.classifier = nn.Linear(768, num_labels)
        
        print(f"> BERT + Adapter 모델 생성")
        print(f"   - BERT 레이어: {self.bert.config.num_hidden_layers}개")
        print(f"   - Adapter 차원: {adapter_dim}")
        print(f"   - 총 Adapter 수: {len(self.adapters)}개")
    
    # Forward pass with Adapter insertion
    def forward(self, input_ids, attention_mask=None):
        # BERT의 hidden states 추출 (모든 레이어)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True  # 모든 레이어 출력 필요
        )
        
        # Apply Adapter to each layer output
        hidden_states = outputs.hidden_states[1:]  # 첫 번째는 embedding
        
        # Apply Adapter only to the last layer (simplified)
        last_hidden_state = hidden_states[-1]
        adapter_output = self.adapters[-1](last_hidden_state)
        
        # Use [CLS] token for classification
        cls_output = adapter_output[:, 0, :]
        logits = self.classifier(cls_output)
        
        return logits
    
    # Calculate number of trainable parameters
    def get_trainable_params(self):
        adapter_params = sum(adapter.get_adapter_params() for adapter in self.adapters)
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        return adapter_params + classifier_params
    
    # Calculate total number of parameters
    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())

def create_simple_dataset():
    texts = [
        "This movie is amazing and fantastic!",
        "I love this film so much, brilliant!",
        "Great acting and wonderful story!",
        "Excellent movie, highly recommend!",
        "This is terrible and boring.",
        "I hate this movie, very bad.",
        "Awful film, waste of time.",
        "Terrible acting and poor story.",
    ] * 25  # 200개 생성
    
    labels = [1, 1, 1, 1, 0, 0, 0, 0] * 25  # 긍정:1, 부정:0
    
    return texts, labels

def main():
    print("\n Adapter Layer 특성 분석 시작!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"> 디바이스: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create model
    model = BERTWithAdapter(adapter_dim=64, num_labels=2)
    model = model.to(device)
    
    # Parameter analysis
    total_params = model.get_total_params()
    trainable_params = model.get_trainable_params()
    frozen_params = total_params - trainable_params
    
    print(f"\n파라미터 분석:")
    print(f"   전체 파라미터: {total_params:,}")
    print(f"   동결 파라미터: {frozen_params:,} (BERT)")
    print(f"   학습 파라미터: {trainable_params:,} (Adapter + 분류기)")
    print(f"   학습 비율: {100 * trainable_params / total_params:.2f}%")
    
    texts, labels = create_simple_dataset()
    print(f"\n데이터셋: {len(texts)}개 샘플")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    print(f"\nAdapter 학습 시작...")
    
    # Batch training 
    batch_size = 8
    num_batches = min(10, len(texts) // batch_size)  # 10배치만 학습
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        batch_texts = texts[start_idx:end_idx]
        batch_labels = torch.tensor(labels[start_idx:end_idx], device=device)
        
        # Tokenization
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(inputs.input_ids, inputs.attention_mask)
        loss = criterion(logits, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if batch_idx % 3 == 0:
            print(f"   배치 {batch_idx+1}/{num_batches}: 손실 = {loss.item():.4f}")
    
    print(f"\n> Adapter 추론 테스트:")
    model.eval()
    
    test_texts = [
        "This is an amazing movie!",
        "I absolutely love this film!",
        "This movie is terrible.",
        "I hate this boring film."
    ]
    
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)
            
            logits = model(inputs.input_ids, inputs.attention_mask)
            probs = torch.softmax(logits, dim=-1)
            predicted = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted].item()
            
            sentiment = "긍정" if predicted == 1 else "부정"
            print(f"   '{text[:30]}...' → {sentiment} ({confidence:.3f})")
    
    print(f"\n Adapter Layer 특성 분석:")
    print(f"> 모듈화: 각 BERT 레이어에 독립적인 Adapter 삽입")
    print(f"> Bottleneck: 768 → 64 → 768 차원 변환으로 파라미터 절약")
    print(f"> Residual: 원본 + Adapter 출력으로 안정성 확보")
    print(f"> 동결: BERT는 그대로, Adapter만 학습")
    
    single_adapter = model.adapters[0]
    adapter_params = single_adapter.get_adapter_params()
    
    print(f"\n> 단일 Adapter 모듈 분석:")
    print(f"   입력 차원: 768")
    print(f"   Bottleneck 차원: 64")
    print(f"   출력 차원: 768")
    print(f"   파라미터 수: {adapter_params:,}")
    print(f"   구조: Linear(768→64) + ReLU + Linear(64→768)")

if __name__ == "__main__":
    main()

print(f"\n> Adapter Layer 개념 학습 완료!")
print("1. 새 모듈 삽입: 기존 구조에 작은 네트워크 추가")
print("2. Bottleneck 구조: 차원 축소 → 변환 → 차원 복원")
print("3. Residual 연결: 원본 + Adapter로 안정성 확보") 
print("4. BERT 동결: 사전훈련 지식 보존")
print("5. 효율성: 전체 모델 대비 적은 파라미터로 태스크 적응")
