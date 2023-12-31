import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
import logging
from datasets import Dataset, concatenate_datasets

# Note to anyone who want to learn from this code: it is 3am on New Year's Eve and I am shattered.
# This code is an amalgamation of PyTorch and HuggingFace code, and it is not pretty.
# The semisupervised-alt file aims to remove PyTorch code and use HuggingFace code only, but
# needs a bit more work and it's not the most important things. Be warned in attempting to consume the spaghetti.

logging.basicConfig(level=logging.INFO)

def convert_to_datasets_lib_dataset(tensor_dataset):
    # Convert tensors to lists and create a dictionary
    data_dict = {key: tensor.tolist() for key, tensor in zip(["input_ids", "attention_mask"], tensor_dataset.tensors[:2])}
    
    # Check if labels exist
    if len(tensor_dataset.tensors) > 2:
        data_dict["labels"] = tensor_dataset.tensors[2].tolist()
    
    # Create a datasets.Dataset object
    datasets_lib_dataset = Dataset.from_dict(data_dict)
    
    return datasets_lib_dataset

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

class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        item = {key: tensor[index] for key, tensor in zip(["input_ids", "attention_mask"], self.tensors[:2])}
        if len(self.tensors) > 2:  # Check if labels exist
            item["labels"] = self.tensors[2][index]
        return item

    def __len__(self):
        return self.tensors[0].size(0)

def train_model(labelled_csv="messages-labelled-s.csv", unlabelled_csv="messages-unlabelled.csv", eval_csv="messages-labelled-s-eval.csv"):
    """Trains a model using both labelled and unlabelled data with semi-supervised learning."""

    # Load the datasets
    labelled_df = pd.read_csv(labelled_csv)
    unlabelled_df = pd.read_csv(unlabelled_csv)
    eval_df = pd.read_csv(eval_csv)

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

    # Prepare the datasets for the model
    train_encodings = tokenizer(labelled_df["text"].tolist(), truncation=True, padding=True, return_tensors="pt")
    eval_encodings = tokenizer(eval_df["text"].tolist(), truncation=True, padding=True, return_tensors="pt")

    # Map labels to numbers
    label_map = {"code": 1, "no code": 0}
    labelled_df["label"] = labelled_df["label"].map(label_map)
    eval_df["label"] = eval_df["label"].map(label_map)

    # Create datasets
    train_dataset = convert_to_datasets_lib_dataset(TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(labelled_df["label"].tolist())))
    eval_dataset = convert_to_datasets_lib_dataset(TensorDataset(eval_encodings["input_ids"], eval_encodings["attention_mask"], torch.tensor(eval_df["label"].tolist())))

    # Initialize the model
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-small-en-v1.5", num_labels=2)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="model_output",
        num_train_epochs=10,  # Example epoch count
        per_device_train_batch_size=32,  # Adjust based on GPU memory
        evaluation_strategy="epoch"
    )
    data_collator = TensorDataCollator(tokenizer)

    # Initial training on labeled data
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Pseudo-labeling for unlabeled data
    unlabelled_df = unlabelled_df.dropna(subset=['text'])
    unlabelled_encodings = tokenizer(unlabelled_df["text"].tolist(), truncation=True, padding=True, return_tensors="pt")
    unlabelled_dataset = CustomTensorDataset(unlabelled_encodings["input_ids"], unlabelled_encodings["attention_mask"])
    
    # Make predictions on unlabeled data
    pseudo_labels = trainer.predict(unlabelled_dataset).predictions.argmax(-1)

    # Create dataset with pseudo-labels
    pseudo_input_ids = torch.stack([item["input_ids"] for item in unlabelled_dataset])
    pseudo_attention_mask = torch.stack([item["attention_mask"] for item in unlabelled_dataset])
    pseudo_labels = torch.tensor(pseudo_labels.tolist())
    pseudo_labelled_dataset = convert_to_datasets_lib_dataset(TensorDataset(pseudo_input_ids, pseudo_attention_mask, pseudo_labels))

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
        data_collator=data_collator,
    )
    trainer.train()

    # Save the trained model
    model.save_pretrained("security-code-detector-semisupervised")
    model.push_to_hub("lujstn/security-code-detector-semisupervised")

def use_model(messages, model_path="security-code-detector-semisupervised"):
  """Uses a trained model to predict security codes in messages."""

  # Load the trained model
  model = AutoModelForSequenceClassification.from_pretrained(model_path)

  # Prepare the input for prediction
  tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
  encodings = tokenizer(messages, truncation=True, padding=True, return_tensors="pt")

  # Make predictions
  with torch.no_grad():
      outputs = model(**encodings)
      predictions = torch.argmax(outputs.logits, dim=1).tolist()

  # Map numeric predictions back to string labels
  label_map = {1: "code", 0: "no code"}
  predictions = [label_map[p] for p in predictions]

  return predictions

# Example usage:

# To train the model:
train_model()

# To use the model after training:
messages = ["This is a message with a code: 12345", "This message has no code"]
predictions = use_model(messages)
print(predictions)  # Output: ["code", "no code"]
