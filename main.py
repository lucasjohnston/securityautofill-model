from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments
import argilla as rg

# Initialize Argilla
client = rg.Argilla(api_url="http://localhost:6900", api_key="argilla.apikey")


# Step 1: Label messages as containing security codes or not.
def label_messages(unlabelled_csv="messages-unlabelled.csv"):
    """Labels messages as containing security codes or not."""

    # Load a sample of 100 messages from the dataset
    unlabelled = (
        load_dataset("csv", data_files=unlabelled_csv)["train"].shuffle(seed=42).select(range(100))
    )
    unlabelled = rg.DatasetForTextClassification.from_datasets(unlabelled)

    # Log the unlabelled messages to Argilla
    rg.log(unlabelled, "messages_unlabelled")

# ⚠️ To label the messages, uncomment the following line:
label_messages()

# Step 2: Once completed in the Argilla UI, train the model
def train_model(eval_csv="messages-eval.csv"):
    """Trains a SetFit model to identify security codes in messages."""

    # Load the datasets
    train_dataset = rg.load("messages_unlabelled").prepare_for_training()
    eval_dataset = load_dataset("csv", data_files=eval_csv)["train"]
    
    # Initialize the model
    model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5", labels=["no code", "code"])
    
    # Prepare training arguments
    args = TrainingArguments(
        batch_size=32,
        num_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Train the model
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        metric="accuracy",
    )
    trainer.train()
    
    # Save the newly trained model
    model.save_pretrained("security-code-detector-setfit-v2")

# ⚠️ To train the model, uncomment the following line:
# train_model()

# Step 3: Use the model to predict security codes in messages.
def use_model(messages, model_path="security-code-detector-setfit-v2"):
   """Uses a trained SetFit model to predict security codes in messages."""

   # Load the trained model
   model = SetFitModel.from_pretrained(model_path)

   # Make predictions
   predictions = model.predict(messages)

   # ⚠️ If you've tested the model and are happy with its predictions, uncomment to push the model to the Hub:
   # model.push_to_hub("lujstn/security-code-detector-setfit")

   return predictions

# To directly use the model after training, uncomment the following lines:
# messages = ["This is a message with a code: 12345", "This message has no code"]
# predictions = use_model(messages)
# print(predictions)  # Output: ["code", "no code"]
