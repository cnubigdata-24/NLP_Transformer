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


# Text generation
# Test 5 ##########################################################################
from transformers import pipeline

generator = pipeline("text-generation")
print(generator("In this course, we will teach you how to"))

# Using any model from the Hub in a pipeline
# Test 6 ##########################################################################
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

# Mask filling
# Test 7 ##########################################################################
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)


# Named entity recognition
# Test 8 ##########################################################################
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

# Question answering
# Test 9 ##########################################################################
from transformers import pipeline

question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)

# Summarization
# Test 10 ##########################################################################
from transformers import pipeline

summarizer = pipeline("summarization")
summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
)

# Translation
# Test 11 ##########################################################################
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")

# Image classification
# Test 12 ##########################################################################
from transformers import pipeline

image_classifier = pipeline(
    task="image-classification", model="google/vit-base-patch16-224"
)
result = image_classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
print(result)















