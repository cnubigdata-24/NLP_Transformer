import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import torch
import torch.quantization
import time
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

warnings.simplefilter("ignore", category=FutureWarning)

# Load the pre-trained DistilBERT model and tokenizer
MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model_fp32 = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)

# Use CPU for fair comparison between FP32 and quantized models
device = "cpu"
print(f"사용 디바이스: {device})")

model_fp32.to(device)

text = "허깅페이스는 NLP를 더 쉽게 만듭니다!"
inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
inputs_cpu = {k: v.to(device) for k, v in inputs.items()}

# Measure inference time for the FP32 model on CPU
print("\n=== FP32 모델 테스트 ===")
start_time = time.time()
with torch.no_grad():
    output_fp32 = model_fp32(**inputs_cpu)
fp32_time = time.time() - start_time

print("\n=== 양자화 수행 중 ===")
model_fp32_cpu = model_fp32

# Apply dynamic quantization (only affects Linear layers)
model_quantized = torch.quantization.quantize_dynamic(
    model_fp32_cpu,  # Original model on CPU
    {torch.nn.Linear},  # Quantize only Linear layers
    dtype=torch.qint8  # Convert weights to 8-bit integers
)

print("양자화 완료!")

print("\n=== 양자화 모델 테스트 ===")
start_time = time.time()
with torch.no_grad():
    output_qint8 = model_quantized(**inputs_cpu)
qint8_time = time.time() - start_time

def get_model_size_mb(model):
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / 1024 / 1024

fp32_size_mb = get_model_size_mb(model_fp32_cpu)
qint8_size_mb = get_model_size_mb(model_quantized)

predicted_label_fp32 = torch.argmax(output_fp32.logits, dim=1).item()
predicted_label_qint8 = torch.argmax(output_qint8.logits, dim=1).item()

if hasattr(model_fp32.config, "id2label"):
    label_map = model_fp32.config.id2label  # Use model's predefined label mapping
else:
    label_map = {0: "부정 (Negative)", 1: "긍정 (Positive)"}  # Default to sentiment analysis labels

speedup = fp32_time / qint8_time if qint8_time > 0 else 0
compression_ratio = fp32_size_mb / qint8_size_mb if qint8_size_mb > 0 else 0

print(f"\n=== 성능 비교 결과 ===")
print(f"FP32 모델 추론 시간: {fp32_time:.4f} 초 (CPU)")
print(f"양자화 모델 추론 시간: {qint8_time:.4f} 초 (CPU)")
print(f"속도 향상: {speedup:.2f}x" if speedup > 0 else "속도 측정 실패")
print(f"\nFP32 모델 크기: {fp32_size_mb:.2f} MB")
print(f"양자화 모델 크기: {qint8_size_mb:.2f} MB")
print(f"압축률: {compression_ratio:.2f}x" if compression_ratio > 0 else "압축률 측정 실패")
print(f"\nFP32 모델 예측 라벨: {predicted_label_fp32} ({label_map.get(predicted_label_fp32, '알 수 없음')})")
print(f"양자화 모델 예측 라벨: {predicted_label_qint8} ({label_map.get(predicted_label_qint8, '알 수 없음')})")

if predicted_label_fp32 == predicted_label_qint8:
    print("두 모델의 예측 결과가 일치합니다!")
else:
    print("두 모델의 예측 결과가 다릅니다.")

fp32_probs = torch.softmax(output_fp32.logits, dim=1).cpu()
qint8_probs = torch.softmax(output_qint8.logits, dim=1)

print(f"\nFP32 모델 확률: {fp32_probs.numpy().flatten()}")
print(f"양자화 모델 확률: {qint8_probs.numpy().flatten()}")

prob_diff = torch.abs(fp32_probs - qint8_probs).max().item()
print(f"최대 확률 차이: {prob_diff:.4f}")
