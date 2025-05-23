# Test 1 ##########################################################################
# !pip install transformers torch

from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love Hugging Face!")

print(result)

sentiment = pipeline("sentiment-analysis")
print(sentiment("Hugging Face is amazing!"))

# Test 2 ##########################################################################
# !pip install datasets

from datasets import load_dataset
dataset = load_dataset("imdb")
print(dataset["train"][0])

# Test 3 ##########################################################################
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "I love Hugging Face!"

tokens = tokenizer.tokenize(text)
print("> Tokenized Words:", tokens)

inputs = tokenizer(text, return_tensors="pt")
print("> Token IDs:", inputs["input_ids"])
print("> Attention Mask:", inputs["attention_mask"])

tokens_from_ids = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("> Tokens from IDs:", tokens_from_ids)

outputs = model(**inputs)
print("> Model Output:", outputs)

probabilities = F.softmax(outputs.logits, dim=1)
print(probabilities)

# Zero-shot classification
# Test 4 ##########################################################################
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)


