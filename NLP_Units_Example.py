from transformers import AutoTokenizer

# Load tokenizer (BERT-based)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Input text
text = "The quick brown fox jumps over the lazy dog \
It is a sunny day and the birds are singing beautifully"

# Tokenization (Token)
tokens = tokenizer.tokenize(text)
print("\nTokens", len(tokens), "tokens:\n", tokens)

# Convert to sequence (Sequence)
sequence = tokens  # NLP model input unit
print("\nSequence 1 sequence", len(sequence), "tokens included:\n", sequence)

# Split into chunks (Chunk, maximum 6 tokens per chunk)
chunk_size = 6
chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
print("\nChunks", len(chunks), "chunks:")
for i, chunk in enumerate(chunks):
    print("  - Chunk", i+1, ":", chunk, len(chunk), "tokens")

# Apply context (Context, retain maximum 10 tokens)
context_length = 10  # Model context size limit
context = tokens[:context_length]  # Retained information within context
print("\nContext within context limit of", context_length, "tokens:\n", context)
