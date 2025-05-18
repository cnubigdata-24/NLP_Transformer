# pip install transformers torch sentencepiece protobuf

# 8. Generative Q&A

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

# CPU 전용 설정 (Colab/로컬에서 GPU 없이 실행하려면 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.cuda.is_available = lambda: False

print("Loading models (CPU only)...")

# 1. Encoder model (for extractive QA)
encoder_model_name = "monologg/koelectra-base-v3-finetuned-korquad"
qa_pipeline = pipeline(
    "question-answering",
    model=encoder_model_name,
    tokenizer=encoder_model_name,
    device=-1
)

# 2. Decoder model (for generative QA)
decoder_model_name = "skt/kogpt2-base-v2"
decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
decoder_model = AutoModelForCausalLM.from_pretrained(decoder_model_name)

print("Models loaded successfully")

# -----------------------------------------------
# Context (경제 이슈에 기반한 시장 해석 요구)
context = (
    "(경제 분석 텍스트) 최근 글로벌 경기 둔화 우려와 산업 변화 속에서 새로운 투자 전략이 요구되고 있다. "
    "고금리 기조, 인플레이션, 공급망 재편 등 다양한 요소가 장기적 투자 환경을 변화시키고 있으며, "
    "특히 AI, 친환경 산업, 바이오 등 일부 미래 산업 분야는 긍정적인 신호를 보이고 있다. "
    "이러한 상황에서 투자자는 단기적인 시장 변동보다 장기적인 산업 구조 변화와 메가트렌드에 주목할 필요가 있다."
)

# Question (미래 투자 방향성에 대한 복합적 사고 요구)
question = (
    "(향후 산업 예측과 투자 통찰이 요구됨) 향후 5년간 유망한 투자 산업 분야는 무엇이며, 그 근거는 무엇인가요?"
)

# -----------------------------------------------
# Generative Answer Function (Decoder 모델)
def get_decoder_answer(question, context):
    try:
        prompt = (
            "다음은 투자 전략과 산업 전망에 관한 질의응답입니다.\n\n"
            f"{context}\n\n"
            f"질문: {question}\n\n"
            "위 질문에 대해 현재 경제 상황과 산업 구조를 분석하여, "
            "전망이 밝은 분야와 그 이유를 구체적으로 제시해주세요.\n"
            "답변:"
        )

        inputs = decoder_tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = decoder_model.generate(
                inputs["input_ids"],
                max_length=inputs["input_ids"].shape[1] + 150,
                do_sample=True,
                temperature=0.8,
                top_p=0.92,
                top_k=50,
                no_repeat_ngram_size=3,
                pad_token_id=decoder_tokenizer.eos_token_id
            )

        generated_text = decoder_tokenizer.decode(output[0], skip_special_tokens=True)
        answer = generated_text.split("답변:")[-1].strip()

        return answer if answer else "Model failed to generate a response."

    except Exception as e:
        return f"Decoder model error: {str(e)}"

# -----------------------------------------------
# Run
print("\n== Generative Q&A ==\n")
print(f"Question: {question}\n")

# 1. Encoder Answer (Extractive)
try:
    qa_result = qa_pipeline(question=question, context=context)
    encoder_answer = qa_result["answer"]
    encoder_score = qa_result["score"]
    print(f"1. Encoder Response: {encoder_answer} (confidence: {encoder_score:.4f})")
except Exception as e:
    print(f"Encoder Error: {str(e)}")

print("-" * 100)

# 2. Decoder Answer (Generative)
try:
    decoder_answer = get_decoder_answer(question, context)
    print(f"2. Decoder Response: {decoder_answer}")
except Exception as e:
    print(f"Decoder Error: {str(e)}")

print("-" * 100)
