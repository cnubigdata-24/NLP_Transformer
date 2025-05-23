#!pip install datasets transformers scikit-learn

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

# Label ID to class name mapping
label_map = {
    0: "Science/Technology",
    1: "Health",
    2: "Culture",
    3: "Entertainment",
    4: "Politics",
    5: "Economic",
    6: "Society"
}

# Step 1: Load subset of KLUE YNAT dataset
dataset = load_dataset("klue", "ynat", split={
    "train": "train[:100]",
    "validation": "validation[:50]"
})

print(dataset["train"][0])
print(dataset["train"].column_names)

# Step 2: Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels=7)

# Step 3: Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples["title"], truncation=True, padding="max_length", max_length=64)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Step 4: Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=5,
    save_steps=10,
    save_total_limit=1,
    learning_rate=5e-5,
)

# Step 6: Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Step 7: Train model
trainer.train()
trainer.save_model("./results")

# Step 8: Predict and print results
predictions = trainer.predict(encoded_dataset["validation"])
logits = predictions.predictions
labels = predictions.label_ids
predicted = np.argmax(logits, axis=1)

print("\n=== Prediction Results (first 10 samples) ===")
for i in range(10):
    pred_label = predicted[i]
    true_label = labels[i]
    print(f"[{i+1}] Predicted: {pred_label} ({label_map[pred_label]}), Actual: {true_label} ({label_map[true_label]})")

# Step 9: Print evaluation metrics
precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="weighted")
acc = accuracy_score(labels, predicted)
print("\n=== Evaluation Metrics ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
