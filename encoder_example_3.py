import torch
import warnings
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import os

warnings.simplefilter("ignore")
os.environ["WANDB_DISABLED"] = "true"

# Check if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load financial sentiment dataset from Hugging Face Hub
dataset_name = "zeroshot/twitter-financial-news-sentiment"
dataset = load_dataset(dataset_name)
dataset = dataset["train"].select(range(1000))  

print("Dataset loaded with subset of 1000 samples")
print("Sample data:", dataset[0])

# Define label mapping
label_map = {0: "하락장 (Bearish)", 1: "중립 (Neutral)", 2: "상승장 (Bullish)"}

train_texts = dataset["text"]
train_labels = dataset["label"]

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)

MODEL_NAME = "distilroberta-base"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

# Tokenize datasets
def tokenize_function(batch):
    return tokenizer(
        batch["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=64
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load DistilRoBERTa model for classification
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.to(device)

model.config.id2label = {0: "하락장 (Bearish)", 1: "중립 (Neutral)", 2: "상승장 (Bullish)"}
model.config.label2id = {"하락장 (Bearish)": 0, "중립 (Neutral)": 1, "상승장 (Bullish)": 2}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Changed from evaluation_strategy
    save_strategy="epoch",
    per_device_train_batch_size=16,  # Reduced batch size for stability
    per_device_eval_batch_size=16,
    num_train_epochs=2,  # Increased to 2 epochs for better learning
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=2e-5,  # Added learning rate
    warmup_steps=100,  # Added warmup
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=(device == "cuda"),
    gradient_accumulation_steps=1,
    dataloader_num_workers=0,  # Avoid multiprocessing issues in Colab
    report_to=None,  # Disable reporting
)

# Define compute metrics function
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Starting training...")

# Train model
training_result = trainer.train()

print("Training completed!")
print(f"Training loss: {training_result.training_loss:.4f}")

# Evaluate model
eval_result = trainer.evaluate()
print(f"Evaluation results: {eval_result}")

# Save fine-tuned model
model.save_pretrained("./roberta-financial")
tokenizer.save_pretrained("./roberta-financial")

print("RoBERTa fine-tuning completed and model saved!")

test_examples = [
    "Apple stock surges after strong earnings report.",
    "Tesla shares plummet amid production concerns.",
    "Market remains stable with mixed signals.",
    "Amazon announces record quarterly profits.",
    "Oil prices decline due to oversupply issues."
]

print("\n=== Testing Fine-tuned Model ===")
model.eval()

for test_text in test_examples:
    inputs = tokenizer(
        test_text, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=64
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
        confidence = probabilities[0][predicted_label].item()
    
    print(f"Text: {test_text}")
    print(f"Predicted: {model.config.id2label[predicted_label]} (confidence: {confidence:.3f})")
    print("-" * 60)

print(f"\n=== Training Summary ===")
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {dataset_name} (1000 samples)")
print(f"Device: {device}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Final evaluation accuracy: {eval_result.get('eval_accuracy', 'N/A'):.3f}")
