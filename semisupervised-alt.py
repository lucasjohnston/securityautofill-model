import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, TextClassificationPipeline
import logging
from datasets import Dataset, concatenate_datasets

logging.basicConfig(level=logging.INFO)

def train_model(labelled_csv="messages-labelled-s.csv", unlabelled_csv="messages-unlabelled.csv", eval_csv="messages-labelled-s-eval.csv"):
    """Trains a model using both labelled and unlabelled data with semi-supervised learning."""

    # Load the datasets
    labelled_df = pd.read_csv(labelled_csv)
    unlabelled_df = pd.read_csv(unlabelled_csv)
    eval_df = pd.read_csv(eval_csv)

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

    # Prepare the datasets for the model
    train_encodings = tokenizer(labelled_df["text"].tolist(), truncation=True, padding='max_length', max_length=173, return_tensors="pt")
    eval_encodings = tokenizer(eval_df["text"].tolist(), truncation=True, padding='max_length', max_length=173, return_tensors="pt")

    # Map labels to numbers
    label_map = {"code": 1, "no code": 0}
    labelled_df["label"] = labelled_df["label"].map(label_map)
    eval_df["label"] = eval_df["label"].map(label_map)

    # Create datasets
    train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"], "labels": labelled_df["label"]})
    eval_dataset = Dataset.from_dict({"input_ids": eval_encodings["input_ids"], "attention_mask": eval_encodings["attention_mask"], "labels": eval_df["label"]})

    # Initialize the model
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-small-en-v1.5", num_labels=2)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="model_output",
        num_train_epochs=10,  # Example epoch count
        per_device_train_batch_size=32,  # Adjust based on GPU memory
        evaluation_strategy="epoch"
    )

    # Initial training on labeled data
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Pseudo-labeling for unlabeled data
    unlabelled_df = unlabelled_df.dropna(subset=['text'])
    unlabelled_encodings = tokenizer(unlabelled_df["text"].tolist(), truncation=True, padding='max_length', max_length=173, return_tensors="pt")
    unlabelled_dataset = Dataset.from_dict({"input_ids": unlabelled_encodings["input_ids"], "attention_mask": unlabelled_encodings["attention_mask"]})
    
    # Make predictions on unlabeled data
    pseudo_labels = trainer.predict(unlabelled_dataset).predictions.argmax(-1)
    pseudo_labelled_dataset = Dataset.from_dict({"input_ids": unlabelled_encodings["input_ids"], "attention_mask": unlabelled_encodings["attention_mask"], "labels": pseudo_labels})

    # Log info about both datasets
    logging.info(f"Type for train_dataset: {type(train_dataset)}")
    logging.info(f"First item in train_dataset: {train_dataset[0]}")
    logging.info(f"Type for pseudo_labelled_dataset: {type(pseudo_labelled_dataset)}")
    logging.info(f"First item in pseudo_labelled_dataset: {pseudo_labelled_dataset[0]}")

    # Combine original labeled data and pseudo-labeled data
    combined_train_dataset = concatenate_datasets([train_dataset, pseudo_labelled_dataset])
    logging.info(f"Type for combined_train_dataset: {type(combined_train_dataset)}")
    logging.info(f"First item in combined_train_dataset: {combined_train_dataset[0]}")

    # Re-train the model with the combined dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    # Save the trained model
    model.save_pretrained("security-code-detector-semisupervised-alt") # remove alt if this code works
    # model.push_to_hub("lujstn/security-code-detector-semisupervised")

def use_model(messages, model_path="security-code-detector-semisupervised-alt"): # remove alt if this code works
    """Uses a trained model to predict security codes in messages."""

    # Load the trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

    # Create a pipeline for text classification
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    # Make predictions
    predictions = pipeline(messages)

    # Map numeric predictions back to string labels
    label_map = {1: "code", 0: "no code"}
    predictions = [label_map[p['label']] for p in predictions]

    return predictions


# To train the model:
train_model()

# To use the model after training:
messages = ["This is a message with a code: 12345", "This message has no code"]
predictions = use_model(messages)
print(predictions)  # Output: ["code", "no code"]
