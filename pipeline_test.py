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
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer("I love Hugging Face!", return_tensors="pt")
outputs = model(**inputs)
print(outputs)
