# !pip install transformers torch
# sLLM: Specialized Language Model

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F

models = {
    "Bio_ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "BERT (General)": "bert-base-uncased"
}

sentence = "A biopsy of the lung revealed [MASK] carcinoma with extensive metastasis."

for name, path in models.items():
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path)
    model.eval()

    inputs = tokenizer(sentence, return_tensors="pt")
    mask_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits[0, mask_index], dim=-1)
        top_k = torch.topk(probs, 5, dim=-1)

    print(f"\n🔍 Top-5 predictions for {name}")
    for i, (token_id, score) in enumerate(zip(top_k.indices[0], top_k.values[0])):
        token = tokenizer.decode([token_id])
        print(f"{i+1}. {token.strip():<20} | P = {score.item():.4f}")
