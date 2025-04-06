import re
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, AutoTokenizer
from span_marker import SpanMarkerModel, Trainer
from span_marker.modeling import SpanMarkerModelCardData
import argilla as rg
from pathlib import Path
import torch # Import torch to check for MPS availability
import pandas as pd # Added for deduplication
import shutil # Added for file backup

# --- Configuration ---
# Argilla config
ARGILLA_API_URL = "http://localhost:6900" # Replace with your Argilla server URL
ARGILLA_API_KEY = "argilla.apikey"      # Replace with your Argilla API key
RG_WORKSPACE_NAME = "default"           # Explicitly set the workspace name (use None to fallback to default user workspace)
RG_DATASET_NAME = "messages_ner_labelling" # Renamed from ARGILLA_DATASET_NAME

# Data files
UNLABELLED_CSV = "messages-unlabelled.csv"

# Model config
BASE_MODEL = "bert-base-multilingual-cased" # Lightweight multilingual model
NEW_MODEL_NAME = f"span-marker-{BASE_MODEL.split('/')[-1]}-security-codes"
MODEL_OUTPUT_DIR = Path("models") / NEW_MODEL_NAME
LABELS = ["CODE"] # The only entity type we care about

# Training config
NUM_EXAMPLES_TO_LABEL = 150
TRAIN_TEST_SPLIT_SEED = 42
TRAIN_BATCH_SIZE = 8 # Adjust based on your M2 Max memory
EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 3     # SpanMarker often requires fewer epochs than SetFit
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.1
MODEL_MAX_LENGTH = 256 # Max sequence length for the model

# --- Argilla Initialization ---
try:
    # Initialize client (workspace is typically handled per-dataset)
    client = rg.Argilla(api_url=ARGILLA_API_URL, api_key=ARGILLA_API_KEY)
    print("Successfully connected to Argilla.")
except Exception as e:
    print(f"Failed to connect to Argilla: {e}")
    print("Please ensure Argilla v2+ is running and reachable.")
    # Exit or handle appropriately
    exit()

# --- Helper Functions ---
def find_potential_codes(text):
    """Very basic regex to find potential codes (sequences of digits/letters)."""
    # Finds sequences of 4+ digits, or 6+ alphanumeric (potentially mixed case)
    # Improvement: Could add patterns like XXX-XXX, etc.
    potential_codes = []
    # Simple digit codes (4+ digits)
    for match in re.finditer(r'\b\d{4,}\b', text):
        potential_codes.append((match.group(0), match.start(), match.end()))
    # Alphanumeric codes (6+ chars, often uppercase but allow mix)
    for match in re.finditer(r'\b[A-Za-z0-9]{6,}\b', text):
        # Avoid adding if it's just a long number already caught, or just plain words
        if not match.group(0).isdigit() and not match.group(0).isalpha():
             potential_codes.append((match.group(0), match.start(), match.end()))
        elif len(match.group(0)) >= 8 and match.group(0).isupper(): # Catch longer uppercase codes like SIYRxKrru1t
             potential_codes.append((match.group(0), match.start(), match.end()))

    # Basic heuristics to remove common false positives like phone numbers/years if needed
    # E.g., check context around the match
    return potential_codes

# --- Workflow Steps ---

# Step 1: Prepare and label messages for NER in Argilla.
def label_messages_for_ner(unlabelled_csv=UNLABELLED_CSV, num_examples=NUM_EXAMPLES_TO_LABEL, dataset_name=RG_DATASET_NAME, workspace_name=RG_WORKSPACE_NAME):
    """Loads messages, optionally removes duplicates, pre-suggests codes, and logs to Argilla for NER labelling."""
    print(f"Loading raw data from {unlabelled_csv}...")

    # --- Optional Deduplication ---
    deduplicate = input(f"Do you want to remove duplicate messages from '{unlabelled_csv}' first? (Recommended to prevent ID conflicts) (y/n): ").lower().strip()
    if deduplicate == 'y':
        try:
            print(f"Reading '{unlabelled_csv}' for deduplication...")
            df = pd.read_csv(unlabelled_csv)
            original_count = len(df)
            if 'text' not in df.columns:
                raise ValueError("CSV must contain a 'text' column for deduplication.")

            # Backup original file
            backup_path = Path(f"{unlabelled_csv}.bak")
            print(f"Backing up original file to '{backup_path}'...")
            shutil.copyfile(unlabelled_csv, backup_path)

            # Deduplicate
            df.drop_duplicates(subset=['text'], keep='first', inplace=True)
            new_count = len(df)
            duplicates_removed = original_count - new_count

            if duplicates_removed > 0:
                print(f"Removed {duplicates_removed} duplicate message(s). Overwriting '{unlabelled_csv}' with unique messages...")
                df.to_csv(unlabelled_csv, index=False)
            else:
                print("No duplicate messages found.")

        except FileNotFoundError:
            print(f"Error: Source file '{unlabelled_csv}' not found. Skipping deduplication.")
        except Exception as e:
            print(f"Error during deduplication: {e}. Continuing without deduplication.")
            # Optionally restore from backup if needed, but safer to continue with original for now
            # shutil.copyfile(backup_path, unlabelled_csv)


    # --- Load (potentially deduplicated) raw text data ---
    try:
        raw_dataset = load_dataset("csv", data_files=unlabelled_csv)["train"]
        print(f"Loaded {len(raw_dataset)} records from '{unlabelled_csv}'.")
    except Exception as e:
        print(f"Fatal Error: Could not load dataset from '{unlabelled_csv}' after attempting deduplication. Check the file. Error: {e}")
        return # Stop if loading fails

    # Select a sample for labelling
    if len(raw_dataset) < num_examples:
        print(f"Warning: Requested {num_examples} examples, but only {len(raw_dataset)} available after loading/deduplication. Using all available.")
        num_examples = len(raw_dataset)

    sample_dataset = raw_dataset.shuffle(seed=TRAIN_TEST_SPLIT_SEED).select(range(num_examples))

    print(f"Preparing {num_examples} examples for Argilla NER labelling...")
    # --- Define Argilla Settings with a named question ---
    settings = rg.Settings(
        fields=[
            rg.TextField(name="text", use_markdown=False) # Use rg.TextField directly, add use_markdown=False like tutorial
        ],
        questions=[
            # Use rg.SpanQuestion for NER/span labelling tasks
            rg.SpanQuestion(
                name="ner_question",       # Name for the question
                field="text",             # The field the spans refer to
                labels=LABELS,            # Use the defined LABELS list
                # title="Optional title",   # Optional: Add a title for the UI
                allow_overlapping=False # Set based on whether codes can overlap (likely False)
            )
        ]
        # Optional: Add guidelines like in the tutorial
        # guidelines="Review the text and identify spans corresponding to the CODE label."
    )

    # --- Create Records with Suggestions ---
    records = []
    # Keep track of hashes seen in this batch to prevent duplicates *within the sample* if source wasn't deduplicated
    hashes_in_batch = set()
    skipped_duplicates_in_batch = 0

    for example in sample_dataset:
        text = example["text"]
        if not text or not isinstance(text, str):
            print(f"Skipping invalid text entry: {example}")
            continue

        record_id = hash(text)
        if record_id in hashes_in_batch:
            skipped_duplicates_in_batch += 1
            continue # Skip duplicate within the selected sample
        hashes_in_batch.add(record_id)


        # Find potential codes using the heuristic
        potential_codes = find_potential_codes(text)

        # Create a list of suggestion values (label, start, end) for this record
        heuristic_spans = []
        for code_text, start, end in potential_codes:
            heuristic_spans.append(("CODE", start, end))

        # Create ONE suggestion object for the record if spans were found
        record_suggestions = []
        if heuristic_spans:
             # Ensure the value format matches SpanQuestion expectations: list of dicts
             suggestion_value = [{"label": label, "start": start, "end": end} for label, start, end in heuristic_spans]
             record_suggestions.append(
                 rg.Suggestion(
                    question_name="ner_question", # Match the SpanQuestion name
                    value=suggestion_value,      # Pass the list of dicts
                    agent="regex_heuristic"     # Name of the suggestion source
                    # score=... # Score is per-span in SpanQuestion, applied within the dict if needed
                 )
             )

        # Create Argilla record
        record = rg.Record(
            fields={"text": text}, # Text field content
            suggestions=record_suggestions, # Pass the list containing the single Suggestion object
            metadata={"tokens": text.split()}, # Simple tokenization for metadata
            id=record_id # Use the calculated hash
        )
        records.append(record)

    if skipped_duplicates_in_batch > 0:
         print(f"Note: Skipped {skipped_duplicates_in_batch} duplicate messages found within the random sample of {num_examples}.")

    # Log to Argilla
    print(f"Attempting to log {len(records)} records to Argilla dataset '{dataset_name}' in workspace '{workspace_name or 'default'}'...")
    try:
        # === Step 1: Check for Existing Dataset and Handle Conflict ===
        dataset = None
        try:
            # Attempt to fetch the dataset by name within the specified workspace
            existing_dataset = client.datasets(name=dataset_name, workspace=workspace_name) # Use client.datasets(name=...)
            if existing_dataset:
                print(f"Dataset '{dataset_name}' already exists in workspace '{workspace_name or 'default'}'.")

                # Ask user how to proceed
                while True:
                    choice = input(f"  Overwrite existing dataset '{dataset_name}' (y) or append new records (n)? ").lower().strip()
                    if choice == 'y':
                        print(f"  Deleting existing dataset '{dataset_name}'...")
                        existing_dataset.delete() # Call delete on the dataset object
                        print("  Recreating dataset definition...")
                        # Define the dataset object *without* records first
                        dataset_definition = rg.Dataset(name=dataset_name, workspace=workspace_name, settings=settings)
                        dataset_definition.create() # Call create on the new dataset object
                        dataset = dataset_definition # Use the new object
                        print("  Dataset recreated.")
                        break
                    elif choice == 'n':
                        print("  Will append records to the existing dataset.")
                        dataset = existing_dataset # Use the existing dataset object for logging
                        break
                    else:
                        print("  Invalid choice. Please enter 'y' or 'n'.")
            else:
                # client.datasets returns None if not found
                print(f"Dataset '{dataset_name}' not found in workspace '{workspace_name or 'default'}'. Creating new definition...")
                dataset_definition = rg.Dataset(name=dataset_name, workspace=workspace_name, settings=settings)
                dataset_definition.create()
                dataset = dataset_definition
                print(f"Dataset definition created in Argilla.")


        except Exception as e:
            # Catch other potential errors during check/creation/deletion
            print(f"An unexpected error occurred during dataset setup: {e}")
            raise # Re-raise unexpected errors


        # === Step 2: Log the Records to the (Now Confirmed) Dataset ===
        if dataset: # Ensure we have a valid dataset object before logging
            print(f"Sending {len(records)} records to Argilla...") # Added log before sending
            # Log records to the specific dataset object, not the client
            # Batch size is handled internally by the log method, SDK will warn if adjusted.
            log_response = dataset.records.log(records=records)
            print(f"Records logged: Processed={log_response.processed}, Failed={log_response.failed}")

            # Get the URL from the dataset object
            dataset_url = getattr(dataset, 'url', 'URL not available')
            print(f"Dataset available in Argilla at: {dataset_url}")
            print("➡️ Please go to the Argilla UI, review the suggestions, and correct/add labels for the 'CODE' entity.")
        else:
            print("Error: Could not obtain a valid dataset object to log records.")


    except Exception as e:
        # Catch potential errors during deletion, recreation, or logging
        print(f"Error interacting with Argilla during dataset handling or logging: {e}")

# ⚠️ To prepare data for labelling, uncomment the following line:
label_messages_for_ner() # Pass workspace_name if not using 'default'

# Step 2: Train the SpanMarkerNER model using labelled data from Argilla.
def train_span_marker_model(dataset_name=RG_DATASET_NAME, model_output_dir=MODEL_OUTPUT_DIR, workspace_name=RG_WORKSPACE_NAME):
    """Trains a SpanMarkerNER model using data labelled in Argilla."""
    print(f"Loading labelled data from Argilla dataset '{dataset_name}' in workspace '{workspace_name or 'default'}'...")
    try:
        # Load the dataset from Argilla - specify workspace if not default
        dataset_rg = rg.load(dataset_name, query="status:submitted", limit=None, workspace=workspace_name) # Load all submitted records
        if not dataset_rg:
            print(f"No submitted records found in Argilla dataset '{dataset_name}'. Please label data first.")
            return None
        print(f"Loaded {len(dataset_rg)} labelled records.")
    except Exception as e:
        print(f"Error loading data from Argilla: {e}")
        return None

    # --- Convert Argilla v2 Records to Hugging Face Dataset for SpanMarker ---
    print("Converting Argilla data to Hugging Face Dataset format...")
    hf_dataset_list = []
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # !! Important: Adapt this section based on how you labelled in Argilla v2 !!
    # The structure of `record.responses` or `record.annotation` (if using older term) changes in v2.
    # It depends on the `question_name` used during settings and labelling.

    # Assuming you labeled using the question named "ner_question"
    # and the responses are stored under `record.responses`
    question_name_for_annotation = "ner_question"

    for record in dataset_rg:
        # Check if the record has a response for the specific question
        if not record.responses or question_name_for_annotation not in record.responses:
            # print(f"Skipping record {record.id} because it has no response for question '{question_name_for_annotation}'")
            continue # Skip records without submitted responses for our NER question

        response_data = record.responses[question_name_for_annotation]

        # Check if the response status is 'submitted'
        if response_data.status != "submitted":
            # print(f"Skipping record {record.id} because response status is '{response_data.status}'")
            continue # Skip records not marked as submitted

        # Extract annotations from the 'values' field of the response
        # The value for TokenClassificationQuestion is a list of dicts: [{'label': str, 'start': int, 'end': int}, ...]
        annotations_raw = response_data.value
        if not annotations_raw: # Handle cases with no entities annotated
            annotations = []
        elif isinstance(annotations_raw, list): # Expected format
            annotations = [(ann['label'], ann['start'], ann['end']) for ann in annotations_raw]
        else:
            print(f"Warning: Skipping record {record.id}. Unexpected annotation format: {annotations_raw}")
            continue

        text = record.fields['text'] # Get text from fields

        # --- (Rest of the tokenization and tag creation logic remains largely the same) ---
        # Tokenize the text to get word IDs and offsets
        tokenized_inputs = tokenizer(text, return_offsets_mapping=True, max_length=MODEL_MAX_LENGTH, truncation=True)
        tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"])
        offsets = tokenized_inputs["offset_mapping"]

        # Create IOB2 tags
        ner_tags = ["O"] * len(tokens)

        for label, start_char, end_char in annotations:
            if label != "CODE": continue

            start_token_idx, end_token_idx = -1, -1
            for i, (start, end) in enumerate(offsets):
                if start == end: continue # Skip special tokens

                token_start, token_end = start, end
                # Logic to find token indices covering the character span
                # This precise logic might need refinement depending on tokenizer and edge cases
                if start_token_idx == -1 and token_start >= start_char:
                    start_token_idx = i
                if start_token_idx != -1 and token_end >= end_char:
                    end_token_idx = i
                    break # Found the last token for this span
                # Handle cases where the last token is partially covered
                if start_token_idx != -1 and token_start < end_char and token_end >= end_char:
                    end_token_idx = i
                    break

            # Alternative simpler alignment (might be less precise for multi-token entities):
            # start_token_idx = -1
            # end_token_idx = -1
            # for i, (offset_start, offset_end) in enumerate(offsets):
            #     if offset_start <= start_char < offset_end:
            #         start_token_idx = i
            #     if offset_start < end_char <= offset_end:
            #         end_token_idx = i
            #         break # Found end

            if start_token_idx != -1 and end_token_idx != -1 and start_token_idx <= end_token_idx:
                ner_tags[start_token_idx] = f"B-{label}"
                for i in range(start_token_idx + 1, end_token_idx + 1):
                    if i < len(ner_tags):
                        ner_tags[i] = f"I-{label}"
            else:
                print(f"Warning: Could not align annotation '{label}' ({start_char}, {end_char}) in '{text[:50]}...' Tokens: {len(tokens)}, Offsets: {len(offsets)}. Start/End Index: {start_token_idx}/{end_token_idx}")


        hf_dataset_list.append({"tokens": tokens, "ner_tags": ner_tags})

    if not hf_dataset_list:
        print("No valid annotated records found or processed. Aborting training.")
        return

    # Create the Hugging Face Dataset
    hf_dataset = Dataset.from_list(hf_dataset_list)

    # Create label2id/id2label mappings required by SpanMarker
    labels_augmented = ["O"] + [f"{prefix}-{label}" for label in LABELS for prefix in ["B", "I"]]
    label2id = {label: i for i, label in enumerate(labels_augmented)}
    id2label = {v: k for k, v in label2id.items()}

    # Convert string tags to integer IDs
    def tags_to_ids(example):
        example["ner_tags"] = [label2id.get(tag, label2id["O"]) for tag in example["ner_tags"]] # Use .get for safety
        return example

    hf_dataset_ids = hf_dataset.map(tags_to_ids)

    # Split dataset into train and evaluation sets
    print("Splitting data into train/eval sets...")
    train_eval_dataset = hf_dataset_ids.train_test_split(test_size=0.2, seed=TRAIN_TEST_SPLIT_SEED)
    train_dataset = train_eval_dataset["train"]
    eval_dataset = train_eval_dataset["test"]

    print(f"Training samples: {len(train_dataset)}, Evaluation samples: {len(eval_dataset)}")

    # Initialize the SpanMarker model
    print(f"Initializing SpanMarkerModel with base: {BASE_MODEL}")
    model = SpanMarkerModel.from_pretrained(
        BASE_MODEL,
        labels=LABELS, # Pass the core labels ("CODE")
        # SpanMarker configuration
        model_max_length=MODEL_MAX_LENGTH,
        marker_max_length=128, # Default, adjust if needed
        entity_max_length=10,   # Max length of a code span to consider (adjust based on data)
        # Provide label2id/id2label for custom datasets
        id2label=id2label,
        label2id=label2id,
        # Optional: Add model card data
        model_card_data=SpanMarkerModelCardData(
            model_id=NEW_MODEL_NAME,
            encoder_id=BASE_MODEL,
            dataset_name=dataset_name,
            license="apache-2.0", # Or your preferred license
            language="multilingual", # Assuming based on base model
        )
    )

    # Check for Apple Silicon MPS availability
    use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    print(f"Using MPS (Apple Silicon GPU): {use_mps}")

    
    # Prepare training arguments
    # Note: bf16=True might not be supported on all M-series chips, fp16=True is safer fallback
    args = TrainingArguments(
        output_dir=str(model_output_dir),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        warmup_ratio=WARMUP_RATIO,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1, # Keep only the best checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1", # Default metric for NER
        use_mps_device=use_mps, # Enable MPS if available
        bf16=use_mps, # Use bf16 if MPS supports it
        #fp16=True, # Using fp16 is generally safer/more compatible than bf16 on MPS
        logging_steps=50,
        push_to_hub=False, # Set to True later if desired
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # We don't need a compute_metrics fn; SpanMarker handles standard seqeval metrics
    )

    # Train the model
    print("Starting model training...")
    trainer.train()
    
    # Evaluate on the test set
    print("Evaluating model performance...")
    metrics = trainer.evaluate(eval_dataset, metric_key_prefix="eval")
    print("Evaluation metrics:", metrics)
    trainer.save_metrics("eval", metrics)


    # Save the final model
    final_model_path = model_output_dir / "checkpoint-final"
    print(f"Saving the best model to {final_model_path}...")
    trainer.save_model(str(final_model_path))
    print("Model training complete.")
    # Also save the tokenizer with the model
    tokenizer.save_pretrained(str(final_model_path))

# ⚠️ To train the model after labelling, uncomment the following line:
# train_span_marker_model()

# Step 3: Use the trained model to extract security codes.
def extract_codes_with_model(messages, model_path=MODEL_OUTPUT_DIR / "checkpoint-final"):
   """Uses a trained SpanMarkerNER model to extract security codes from messages."""
   if isinstance(messages, str):
       messages = [messages] # Ensure input is a list

   model_path_str = str(model_path) # Use string path for consistency
   output_dir_str = str(MODEL_OUTPUT_DIR)

   # Check multiple potential save locations
   if not Path(model_path_str).exists():
       print(f"Checkpoint path {model_path_str} not found.")
       if Path(output_dir_str).exists():
           print(f"Attempting to load directly from {output_dir_str}...")
           model_path_str = output_dir_str
       else:
            print(f"Error: Model not found at {model_path_str} or {output_dir_str}. Please train the model first.")
            return None # Or raise error

   print(f"Loading trained model from {model_path_str}...")
   try:
       # Load the trained SpanMarker model
       model = SpanMarkerModel.from_pretrained(model_path_str)
   except Exception as e:
       print(f"Error loading model from {model_path_str}: {e}")
       return None

   print(f"Extracting codes from {len(messages)} message(s)...")
   # Run inference
   # The predict method returns a list of lists of dictionaries, one list per input message
   extracted_entities_batch = model.predict(messages)

   # Extract only the 'CODE' spans
   extracted_codes = []
   for message_entities in extracted_entities_batch:
        codes_in_message = [entity['span'] for entity in message_entities if entity['label'] == 'CODE']
        extracted_codes.append(codes_in_message)


   # ⚠️ If you've tested the model and are happy, you could push it to the Hub:
   # print("Pushing model to Hub...")
   # Ensure you are logged in (`huggingface-cli login`)
   # model.push_to_hub(f"your-hf-username/{NEW_MODEL_NAME}") # Replace with your HF username

   return extracted_codes

# ⚠️ To directly use the model after training, uncomment the following lines:
# test_messages = [
#    "G-861121 is your Google verification code.",
#    "Your Respondent verification code is: 958261",
#    "VK: 448211 - code to create VK ID account",
#    "&lt;#&gt; Seu codigo do Instagram e 794 385. Nao compartilhe. SIYRxKrru1t",
#    "This message has no security code.",
#    "PayPal: Your security code is 237830. Your code expires in 10 minutes.",
#    "Microsoft access code: 1888"
# ]
# extracted_codes = extract_codes_with_model(test_messages)
# if extracted_codes:
#    for msg, codes in zip(test_messages, extracted_codes):
#        print(f"Message: '{msg}'")
#        print(f"Extracted Codes: {codes}")
#        print("-" * 10)

# ⚠️ If you've tested the model and are happy, you could push it to the Hub
# To do this, uncomment the relevant lines in the extract_codes_with_model function above and run the script.

# --- Main Execution Guard ---
if __name__ == "__main__":
    print("Thank you for using Security Autofill 🙇")
    pass # Keeps the script runnable without uncommenting steps
