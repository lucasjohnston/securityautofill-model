import pandas as pd
from setfit import SetFitModel, Trainer, TrainingArguments

def train_model(labelled_csv="messages-train.csv", eval_csv="messages-eval.csv"):
    """Trains a SetFit model to identify security codes in messages."""

    # Load the labelled dataset
    labelled_df = pd.read_csv(labelled_csv)
    eval_csv = pd.read_csv(eval_csv)
    train_dataset = labelled_df.to_dict("records")
    eval_dataset = labelled_df.to_dict("records")
    
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
    model.save_pretrained("security-code-detector-setfit")

def use_model(messages, model_path="security-code-detector-setfit"):
   """Uses a trained SetFit model to predict security codes in messages."""

   # Load the trained model
   model = SetFitModel.from_pretrained(model_path)

   # Make predictions
   predictions = model.predict(messages)

   # ⚠️ If you've tested the model and are happy with its predictions, uncomment to push the model to the Hub:
   # model.push_to_hub("lujstn/security-code-detector-setfit")

   return predictions

# ⚠️ To train the model, uncomment the following line:
# train_model()

# To use the model after training:
messages = ["This is a message with a code: 12345", "This message has no code"]
predictions = use_model(messages)
print(predictions)  # Output: ["code", "no code"]
