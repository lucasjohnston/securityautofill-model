import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, TextClassificationPipeline, DataCollatorWithPadding, EarlyStoppingCallback
import logging
from datasets import Dataset, concatenate_datasets
import torch
import numpy as np
import random

# Configure logging and pytorch seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

logging.basicConfig(level=logging.INFO)
set_seed(42)

# This collator avoids truncating content by padding to the maximum length in the batch, rather than statically setting the maximum length in the dataset.
class TensorDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def collate_batch(self, batch):
        if isinstance(batch[0], torch.Tensor):
            # Handle TensorDataset instances
            input_ids = torch.stack([item[0] for item in batch])
            attention_mask = torch.stack([item[1] for item in batch])
            if len(batch[0]) > 2:  # Check if labels exist
                labels = torch.stack([item[2] for item in batch])
            else:
                labels = None
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        elif isinstance(batch[0], dict):
            # Handle list of dictionaries
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            if "labels" in batch[0]:  # Check if labels exist
                labels = torch.stack([item["labels"] for item in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        else:
            raise ValueError("Batch must be a list of dictionaries or a list of tensors.")

def train_model(labelled_csv="messages-train.csv", unlabelled_csv="messages-unlabelled.csv", eval_csv="messages-eval.csv"):
    """Trains a model using both labelled and unlabelled data with semi-supervised learning."""

    # Load and triage the datasets
    labelled_df = pd.read_csv(labelled_csv)
    eval_df = pd.read_csv(eval_csv)
    unlabelled_df = pd.read_csv(unlabelled_csv)
    unlabelled_df = unlabelled_df.dropna(subset=['text'])

    # Load the tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    data_collator = TensorDataCollator(tokenizer)

    # Prepare the datasets for the model
    train_encodings = tokenizer(labelled_df["text"].tolist(), truncation=True, padding=True, return_tensors="pt")
    eval_encodings = tokenizer(eval_df["text"].tolist(), truncation=True, padding=True, return_tensors="pt")
    unlabelled_encodings = tokenizer(unlabelled_df["text"].tolist(), truncation=True, padding=True, return_tensors="pt")

    # Map labels to numbers
    label_map = {"code": 1, "no code": 0}
    labelled_df["label"] = labelled_df["label"].map(label_map)
    eval_df["label"] = eval_df["label"].map(label_map)

    # Create training datasets
    train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"], "labels": labelled_df["label"]})
    eval_dataset = Dataset.from_dict({"input_ids": eval_encodings["input_ids"], "attention_mask": eval_encodings["attention_mask"], "labels": eval_df["label"]})
    unlabelled_dataset = Dataset.from_dict({"input_ids": unlabelled_encodings["input_ids"], "attention_mask": unlabelled_encodings["attention_mask"]})
    
    # Initialize our first model for predictions
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-small-en-v1.5", num_labels=2)
    training_args = TrainingArguments(
        output_dir="model_output",
        num_train_epochs=20,
        per_device_train_batch_size=32,
        evaluation_strategy="epoch",
        seed=42,
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    # Make predictions on unlabeled data
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    pseudo_labels = trainer.predict(unlabelled_dataset).predictions.argmax(-1)
    pseudo_labelled_dataset = Dataset.from_dict({"input_ids": unlabelled_encodings["input_ids"], "attention_mask": unlabelled_encodings["attention_mask"], "labels": pseudo_labels})

    # Introduce early stopping to prevent overfitting
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

    # Combine our original labeled data and pseudo-labeled data into a hybrid model
    combined_train_dataset = concatenate_datasets([train_dataset, pseudo_labelled_dataset])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping_callback],
    )

    # Train the hybrid model
    trainer.train()

    # Save the newly trained hybrid model
    model.save_pretrained("security-code-detector-semisupervised")

def use_model(messages, model_path="security-code-detector-semisupervised"):
    """Uses a trained model to predict security codes in messages."""

    # Load the trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

    # Create a pipeline for text classification
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    # Make predictions
    predictions = pipeline(messages)

    # Map numeric predictions back to string labels
    label_map = {"LABEL_1": "code", "LABEL_0": "no code"}
    predictions = [label_map[p['label']] for p in predictions]

    # ⚠️ If you've tested the model and are happy with its predictions, uncomment to push the model to the Hub:
    # model.push_to_hub("lujstn/security-code-detector-semisupervised")

    return predictions

# ⚠️ To train the model, uncomment the following line:
# train_model()

# To use the model after training:
messages = ["This is a message with a code: 12345", "This message has no code", "This message has a code: but no"]
predictions = use_model(messages)
print("Here is the output of the model:")
print(predictions)  # Output: ["code", "no code", "no code"]
