# !pip install transformers torch

from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love Hugging Face!")

print(result)

sentiment = pipeline("sentiment-analysis")
print(sentiment("Hugging Face is amazing!"))


# !pip install datasets

from datasets import load_dataset
dataset = load_dataset("imdb")
print(dataset["train"][0])



