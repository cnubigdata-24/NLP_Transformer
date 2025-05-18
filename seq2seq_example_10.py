# !pip install transformers torch sentencepiece sacremoses requests beautifulsoup4

# 10. Text Summarization and Translation

from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import requests
from bs4 import BeautifulSoup
import re

os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.cuda.is_available = lambda: False
print("Loading models (CPU only)...")

# 1. Summarization model
sum_model_name = "gogamza/kobart-summarization"
summarizer = pipeline(
    "summarization",
    model=sum_model_name,
    tokenizer=sum_model_name,
    device=-1
)

# 2. Translation model
trans_model_name = "Helsinki-NLP/opus-mt-ko-en"
translator_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name)

print("Models loaded successfully")

# Extract article content
def get_article_content():
    url = "https://www.etnews.com/20250509000253"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Get title from h2#article_title_h2 element
        title_element = soup.select_one('h2#article_title_h2')
        title = title_element.text.strip() if title_element else "Title not found"

        # Extract article content - from div#articleBody
        content_element = soup.select_one('div#articleBody')

        if content_element:
            # Remove unwanted elements (related articles, image captions, etc.)
            for unwanted in content_element.select('div#layerRelated, figure.article_image'):
                if unwanted:
                    unwanted.decompose()

            # Extract text - prioritize p tag content
            paragraphs = []
            for p in content_element.find_all('p'):
                text = p.get_text().strip()
                if text:
                    paragraphs.append(text)

            # If p tags are insufficient, get all text
            if not paragraphs:
                content_text = content_element.get_text().strip()
                # Clean up unnecessary whitespace and line breaks
                content_text = re.sub(r'\s{2,}', '\n', content_text)
                paragraphs = [p for p in content_text.split('\n') if p.strip()]

            content = '\n'.join(paragraphs)
        else:
            content = "Article content not found"

        return title, content

    except Exception as e:
        print(f"Error extracting article: {e}")
        return None, None

# summarization
def get_improved_summary(text):
    if not text:
        return "Article content could not be retrieved"

    try:
        # Split long text for better processing
        if len(text) > 1500:
            # Divide text into paragraphs and summarize each chunk
            paragraphs = text.split('\n\n')
            summaries = []

            for i in range(0, len(paragraphs), 3):  # Process 3 paragraphs at a time
                chunk = '\n\n'.join(paragraphs[i:i+3])
                if chunk:
                    chunk_summary = summarizer(
                        chunk,
                        max_length=100,
                        min_length=30,
                        do_sample=False
                    )[0]['summary_text']
                    summaries.append(chunk_summary)

            # Combine summaries into one comprehensive summary
            combined_summary = ' '.join(summaries)

            # Remove duplicate content from the combined summary
            final_summary = combined_summary
            for i in range(len(summaries)-1):
                for j in range(i+1, len(summaries)):
                    common = set(summaries[i].split()) & set(summaries[j].split())
                    if len(common) > 5:  # Consider as duplicate if more than 5 common words
                        final_summary = final_summary.replace(summaries[j], '')

            return final_summary.strip()
        else:
            # Directly summarize shorter text
            return summarizer(
                text,
                max_length=200,
                min_length=100,
                do_sample=True,
                temperature=0.7,  # Increase diversity in summary
                top_p=0.95
            )[0]['summary_text']
    except Exception as e:
        return f"Summarization error: {str(e)}"

# translation
def get_improved_translation(text):
    try:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        translations = []

        # Combine sentences into appropriately sized chunks
        current_chunk = []
        current_length = 0
        chunk_size = 300  # Optimal chunk size

        for sentence in sentences:
            if current_length + len(sentence) > chunk_size and current_chunk:
                # Translate current chunk
                chunk_text = ' '.join(current_chunk)
                inputs = translator_tokenizer(chunk_text, return_tensors="pt", padding=True)

                with torch.no_grad():
                    output = translator_model.generate(
                        inputs.input_ids,
                        max_length=1024,
                        num_beams=5,  # Increase beam size for better quality
                        length_penalty=1.0,  # Higher score for longer translations
                        early_stopping=True
                    )

                translation = translator_tokenizer.decode(output[0], skip_special_tokens=True)
                translations.append(translation)

                # Reset
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)

        # Process the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            inputs = translator_tokenizer(chunk_text, return_tensors="pt", padding=True)

            with torch.no_grad():
                output = translator_model.generate(
                    inputs.input_ids,
                    max_length=1024,
                    num_beams=5,
                    length_penalty=1.0,
                    early_stopping=True
                )

            translation = translator_tokenizer.decode(output[0], skip_special_tokens=True)
            translations.append(translation)

        # Combine translated chunks
        full_translation = ' '.join(translations)

        # Improve translation quality: remove duplicates and clean up
        full_translation = re.sub(r'\s{2,}', ' ', full_translation)  # Remove duplicate spaces

        return full_translation
    except Exception as e:
        return f"Translation error: {str(e)}"

print("\n== Seq2Seq Model Tasks ==\n")

# Get article content
title, content = get_article_content()

# Print article title with separator
print("="*50)
print("Article Title:")
print(title)
print("="*50)

print("\nArticle Content (First 50 chars):")
if content:
    print(content[:50] + "...")
else:
    print("Content not available")
print("="*50)

if content:
    full_text = f"{title}\n\n{content}"

    # Task 1: summarization
    summary = get_improved_summary(full_text)
    print("\nSummarization (Korean → Korean Summary):")
    print(summary)
    print("="*50)

    # Task 2: translation
    translation = get_improved_translation(full_text)
    print("\nTranslation (Korean → English):")
    print(translation)
    print("="*50)
else:
    print("Could not process article: content retrieval failed")
