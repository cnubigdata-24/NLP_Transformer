# !pip install sentence-transformers gradio
import gradio as gr
from sentence_transformers import SentenceTransformer, util

# Load the pre-trained Korean Sentence-BERT model
# Tokenizer is not needed as SentenceTransformers automatically handles tokenization
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# Example sentences
sentences = [
    "주식 시장이 큰 폭으로 상승하고 있다.",
    "금융 시장에서 주가가 빠르게 오르고 있다.",
    "오늘 날씨가 맑고 시원하다."
]

# Generate sentence embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

# Compute cosine similarity between different pairs
cosine_sim_1_2 = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
cosine_sim_1_3 = util.pytorch_cos_sim(embeddings[0], embeddings[2]).item()
cosine_sim_2_3 = util.pytorch_cos_sim(embeddings[1], embeddings[2]).item()

print(f"> Similarity (Sentence 1 & 2): {cosine_sim_1_2:.4f}")
print(f"> Similarity (Sentence 1 & 3): {cosine_sim_1_3:.4f}")
print(f"> Similarity (Sentence 2 & 3): {cosine_sim_2_3:.4f}")

# Gradio Interactive UI
def calculate_similarity(text1, text2):
    """Calculate semantic similarity between two input sentences."""
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return f"Similarity Score: {similarity:.4f}"

interface = gr.Interface(
    fn=calculate_similarity,
    inputs=["text", "text"],
    outputs="text",
    title="Korean Sentence Similarity Calculator",
    description="Enter two Korean sentences to compute their semantic similarity."
)

interface.launch()
