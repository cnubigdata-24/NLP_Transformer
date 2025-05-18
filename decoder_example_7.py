# !pip install transformers accelerate
# !pip install torch
# !pip install huggingface_hub[hf_xet]

# 7. Autoregressive Text Generation

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "EleutherAI/polyglot-ko-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float32,
)

# Experimental prompts: Gradually increasing input length
prompts = [
    "오늘",
    "오늘 날씨가",
    "오늘 날씨가 좋아서",
    "오늘 날씨가 좋아서 친구들과",
    "오늘 날씨가 좋아서 친구들과 공원에"
]

print("[Autoregressive Structure Experiment ========== \n")

for i, prompt in enumerate(prompts, 1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs.pop("token_type_ids", None)

    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=40,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"[{i}] Input: {prompt}\n→ Output: {result}\n{'-'*60}")
