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
import inspect
import transformers

# --- Configuration --- 
# Argilla config
ARGILLA_API_URL = "http://localhost:6900" # Replace with your Argilla server URL
ARGILLA_API_KEY = "argilla.apikey"      # Replace with your Argilla API key
RG_WORKSPACE_NAME = "default"           # Explicitly set the workspace name (use None to fallback to default user workspace)
RG_DATASET_NAME = "messages_ner_labelling" # Renamed from ARGILLA_DATASET_NAME

# Data files
UNLABELLED_CSV = "messages-unlabelled.csv"

# Model config
BASE_MODEL = "Alibaba-NLP/gte-modernbert-base" # Lightweight multilingual model
# FacebookAI/xlm-roberta-base
# Alibaba-NLP/gte-modernbert-base
# intfloat/multilingual-e5-base
# Alibaba-NLP/gte-base-en-v1.5
# Mayank6255/GLiNER-MoE-MultiLingual
# bert-base-multilingual-cased



NEW_MODEL_NAME = f"span-marker-{BASE_MODEL.split('/')[-1]}-security-codes"
MODEL_OUTPUT_DIR = Path("models") / NEW_MODEL_NAME
LABELS = ["CODE"] # The only entity type we care about

# Training config
NUM_EXAMPLES_TO_LABEL = 200
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
            rg.TextField(name="text", use_markdown=False) # Main text field
        ],
        questions=[
            # SpanQuestion is used for NER/span labelling tasks
            rg.SpanQuestion(
                name="ner_question",       # Unique name for this question
                field="text",             # The field the spans refer to
                labels=LABELS,            # Use the defined LABELS list (["CODE"])
                allow_overlapping=False   # Security codes are unlikely to overlap
            )
        ],
        metadata=[
            # Define metadata fields to store extra info per record
            rg.TermsMetadataProperty(name="tokens", title="Tokens (Whitespace Split)") # Simple tokenization for context
        ]
        # Optional: Add guidelines if desired
        # guidelines="Review the text and identify spans corresponding to the CODE label."
    )

    # --- Create Records with Suggestions --- 
    records = []
    # Keep track of hashes seen in this batch to prevent duplicates *within the sample* 
    # if the source CSV wasn't deduplicated properly.
    hashes_in_batch = set()
    skipped_duplicates_in_batch = 0

    for example in sample_dataset:
        text = example["text"]
        if not text or not isinstance(text, str):
            print(f"Skipping invalid text entry: {example}")
            continue

        # Use hash of text as a simple ID (assumes text is unique after deduplication)
        record_id = hash(text)
        if record_id in hashes_in_batch:
            skipped_duplicates_in_batch += 1
            continue # Skip duplicate within the selected sample
        hashes_in_batch.add(record_id)


        # Find potential codes using the heuristic regex function
        potential_codes = find_potential_codes(text)

        # Format heuristic spans into Argilla's expected suggestion format
        heuristic_spans = []
        for code_text, start, end in potential_codes:
            heuristic_spans.append({"label": "CODE", "start": start, "end": end})

        # Create the suggestion object for the record if spans were found
        record_suggestions = []
        if heuristic_spans:
             record_suggestions.append(
                 rg.Suggestion(
                    question_name="ner_question", # Must match the SpanQuestion name
                    value=heuristic_spans,      # Pass the list of span dictionaries
                    agent="regex_heuristic"     # Name the suggestion source
                 )
             )

        # Create the Argilla record
        record = rg.Record(
            fields={"text": text},             # Text field content
            suggestions=record_suggestions,   # Heuristic suggestions (can be empty list)
            metadata={"tokens": text.split()}, # Simple tokenization for metadata
            id=record_id                      # Use the calculated hash as ID
        )
        records.append(record)

    if skipped_duplicates_in_batch > 0:
         print(f"Note: Skipped {skipped_duplicates_in_batch} duplicate messages found within the random sample of {num_examples}.")

    # --- Log Records to Argilla Dataset --- 
    print(f"Attempting to log {len(records)} records to Argilla dataset '{dataset_name}' in workspace '{workspace_name or 'default'}'...")
    try:
        # Check if dataset exists and handle potential conflicts (overwrite/append)
        dataset = None
        try:
            existing_dataset = client.datasets(name=dataset_name, workspace=workspace_name)
            if existing_dataset:
                print(f"Dataset '{dataset_name}' already exists in workspace '{workspace_name or 'default'}'.")
                while True:
                    choice = input(f"  Overwrite existing dataset '{dataset_name}' (y) or append new records (n)? ").lower().strip()
                    if choice == 'y':
                        print(f"  Deleting existing dataset '{dataset_name}'...")
                        existing_dataset.delete()
                        print("  Recreating dataset definition...")
                        dataset_definition = rg.Dataset(name=dataset_name, workspace=workspace_name, settings=settings)
                        dataset_definition.create()
                        dataset = dataset_definition
                        print("  Dataset recreated.")
                        break
                    elif choice == 'n':
                        print("  Will append records to the existing dataset.")
                        dataset = existing_dataset
                        break
                    else:
                        print("  Invalid choice. Please enter 'y' or 'n'.")
            else:
                # Dataset not found, create it
                print(f"Dataset '{dataset_name}' not found in workspace '{workspace_name or 'default'}'. Creating new definition...")
                dataset_definition = rg.Dataset(name=dataset_name, workspace=workspace_name, settings=settings)
                dataset_definition.create()
                dataset = dataset_definition
                print(f"Dataset definition created in Argilla.")

        except Exception as e:
            print(f"An unexpected error occurred during dataset check/creation: {e}")
            raise # Re-raise unexpected errors

        # Log the records to the prepared dataset object
        if dataset:
            print(f"Sending {len(records)} records to Argilla...")
            dataset.records.log(records=records)
            print(f"{len(records)} records successfully sent to Argilla.")
            dataset_url = getattr(dataset, 'url', 'URL not available')
            print(f"Dataset available in Argilla at: {dataset_url}")
            print("➡️ Please go to the Argilla UI, review the suggestions, and correct/add labels for the 'CODE' entity.")
        else:
            print("Error: Could not obtain a valid dataset object to log records.")

    except Exception as e:
        print(f"Error interacting with Argilla during dataset handling or logging: {e}")

# ⚠️ To prepare data for labelling, uncomment the following line:
# label_messages_for_ner() # Pass workspace_name if not using 'default'

# Step 2: Train the SpanMarkerNER model using labelled data from Argilla.
def train_span_marker_model(dataset_name=RG_DATASET_NAME, model_output_dir=MODEL_OUTPUT_DIR, workspace_name=RG_WORKSPACE_NAME):
    """Trains a SpanMarkerNER model using data labelled in Argilla."""
    print(f"Loading labelled data from Argilla dataset '{dataset_name}' in workspace '{workspace_name or 'default'}'...")
    try:
        # === Load the dataset using Argilla v2 client ===
        if 'client' not in globals() or not isinstance(client, rg.Argilla):
             print("Error: Argilla client not initialized. Cannot load data.")
             return None

        # Fetch the dataset object by name and workspace
        dataset = client.datasets(name=dataset_name, workspace=workspace_name)

        if not dataset:
            print(f"Error: Dataset '{dataset_name}' not found in workspace '{workspace_name or 'default'}'. Please ensure it exists and you have labelled data.")
            return None

        # Fetch records iterator from the dataset object
        print(f"Fetching records from dataset '{dataset_name}'...")
        records_iterator = dataset.records # Provides an iterator over all records
        print(f"Found dataset '{dataset_name}'. Processing records...")

    except Exception as e:
        print(f"Error loading data from Argilla: {e}")
        return None

    # --- Convert Argilla Records to Hugging Face Dataset for SpanMarker --- 
    print("Converting Argilla data to Hugging Face Dataset format...")
    hf_dataset_list = []
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Define the question name used during Argilla labelling
    question_name_for_annotation = "ner_question"
    processed_records_count = 0

    # Iterate through the records obtained from the dataset object
    for record in records_iterator:
        # Check if the record has responses
        if not record.responses:
            continue # Skip records without any responses

        # Access the response for our specific question using dictionary-style access
        try:
            response_list = record.responses[question_name_for_annotation]
            # Expect a list containing one response object/dict
            if isinstance(response_list, list) and response_list:
                response_data = response_list[0]
            else:
                continue # Skip if not a non-empty list
        except KeyError:
            continue # Skip if response key doesn't exist for this record
        except Exception as e:
            print(f"Warning: Unexpected error accessing response '{question_name_for_annotation}' for record {getattr(record, 'id', 'N/A')}: {e}")
            continue

        # Check if the response status is 'submitted' using safe access
        try:
            status = getattr(response_data, 'status', response_data.get('status') if isinstance(response_data, dict) else None)
            annotations_raw = getattr(response_data, 'value', response_data.get('value') if isinstance(response_data, dict) else None)
        except Exception as e:
            print(f"Warning: Error accessing status/value from response_data for record {getattr(record, 'id', 'N/A')}: {e}")
            continue

        if status != "submitted":
            continue # Skip records not marked as submitted

        # Get the text field from the record
        text = record.fields.get('text')
        if not text or not isinstance(text, str):
            # print(f"Warning: Skipping record {getattr(record, 'id', 'N/A')} missing or invalid 'text' field.") # Optional warning
            continue

        # Process the extracted annotations (list of dicts or None)
        annotations = [] # Default to empty list
        if annotations_raw and isinstance(annotations_raw, list):
            annotations = [
                (ann['label'], ann['start'], ann['end'])
                for ann in annotations_raw
                # Ensure required keys exist and filter for our specific label ("CODE")
                if 'label' in ann and ann['label'] == "CODE" and 'start' in ann and 'end' in ann
            ]
        # If annotations_raw was None or empty list, annotations remains []

        # --- Convert to IOB format for SpanMarker training data --- 
        try:
            # Tokenize text and get offsets
            tokenized_inputs = tokenizer(text, return_offsets_mapping=True, max_length=MODEL_MAX_LENGTH, truncation=True)
            tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"])
            offsets = tokenized_inputs["offset_mapping"]

            # Create IOB2 tags (O = Outside, B = Beginning, I = Inside)
            ner_tags = ["O"] * len(tokens)

            for label, start_char, end_char in annotations:
                # Validate character indices against text length
                if not (isinstance(start_char, int) and isinstance(end_char, int) and 0 <= start_char <= end_char <= len(text)):
                    # print(f"Warning: Skipping invalid annotation span ({label}, {start_char}, {end_char}) for record {getattr(record, 'id', 'N/A')}") # Optional
                    continue

                # Find token indices that align with the character span
                start_token_idx = -1
                end_token_idx = -1
                for i, (offset_start, offset_end) in enumerate(offsets):
                    if offset_start == offset_end: continue # Skip special/padding tokens
                    # Check for overlap: (token_start < ann_end) and (token_end > ann_start)
                    if (offset_start < end_char) and (offset_end > start_char):
                        if start_token_idx == -1:
                            start_token_idx = i # First token that overlaps
                        end_token_idx = i # Last token that overlaps (will be updated)

                # Apply B-I tags if alignment was successful
                if start_token_idx != -1 and end_token_idx != -1 and start_token_idx <= end_token_idx:
                    if start_token_idx < len(ner_tags):
                        ner_tags[start_token_idx] = f"B-{label}"
                    for i in range(start_token_idx + 1, end_token_idx + 1):
                        if i < len(ner_tags):
                            ner_tags[i] = f"I-{label}"
                # else: # Optional: Log alignment failures if debugging needed
                    # print(f"Debug: Could not align annotation '{label}' ({start_char}, {end_char}) ...")

            # Add the processed data to the list for the Hugging Face Dataset
            hf_dataset_list.append({"tokens": tokens, "ner_tags": ner_tags})
            processed_records_count += 1 # Increment count *after* successful processing

        except Exception as e:
            # Catch errors during tokenization or IOB tagging for this specific record
            print(f"Warning: Error processing record {getattr(record, 'id', 'N/A')}. Text: '{text[:50]}...'. Error: {e}")
            continue # Skip this record and proceed to the next

    # --- End of Record Processing Loop --- 

    if not hf_dataset_list:
        print("No valid annotated records found or processed after filtering. Aborting training.")
        return

    print(f"Successfully processed {processed_records_count} submitted records from Argilla for training.")

    # --- Prepare Hugging Face Dataset --- 
    # Create the dataset from the list of processed records
    hf_dataset = Dataset.from_list(hf_dataset_list)

    # Create label mappings for the *data* format (including B/I prefixes)
    # SpanMarker model itself only needs the core labels, but the Trainer needs data with B/I tags mapped to IDs.
    labels_for_data_mapping = ["O"] + [f"{prefix}-{label}" for label in LABELS for prefix in ["B", "I"]]
    label2id = {label: i for i, label in enumerate(labels_for_data_mapping)}
    id2label = {v: k for k, v in label2id.items()}

    # Convert string tags (e.g., "B-CODE") to integer IDs
    def tags_to_ids(example):
        # Use .get with default "O" for safety, though all tags should be in the mapping
        example["ner_tags"] = [label2id.get(tag, label2id["O"]) for tag in example["ner_tags"]]
        return example

    hf_dataset_ids = hf_dataset.map(tags_to_ids)

    # Split dataset into training and evaluation sets
    print("Splitting data into train/eval sets...")
    train_eval_dataset = hf_dataset_ids.train_test_split(test_size=0.2, seed=TRAIN_TEST_SPLIT_SEED)
    train_dataset = train_eval_dataset["train"]
    eval_dataset = train_eval_dataset["test"]

    print(f"Training samples: {len(train_dataset)}, Evaluation samples: {len(eval_dataset)}")

    # --- Initialize SpanMarker Model --- 
    print(f"Initializing SpanMarkerModel with base: {BASE_MODEL}")
    model = SpanMarkerModel.from_pretrained(
        BASE_MODEL,
        labels=LABELS, # Pass the core labels ("CODE") to the model
        # SpanMarker configuration options:
        model_max_length=MODEL_MAX_LENGTH, # Max sequence length for the underlying transformer
        marker_max_length=128,      # Max length for span markers (usually default is fine)
        entity_max_length=10,       # Max length of entity spans to consider (adjust based on typical code length)
        # Provide label mappings for custom datasets (needed if labels differ from pretrained config)
        id2label=id2label,          # Mapping used internally by the model/trainer
        label2id=label2id,
        # Optional: Add model card data for documentation/sharing
        model_card_data=SpanMarkerModelCardData(
            model_id=NEW_MODEL_NAME,
            encoder_id=BASE_MODEL,
            dataset_name=dataset_name,
            license="apache-2.0", # Or your preferred license
            language="multilingual", # Assuming based on base model
        )
    )

    # Check for Apple Silicon MPS (GPU) availability
    use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    print(f"Using MPS (Apple Silicon GPU): {use_mps}")

    # --- Configure Training Arguments --- 
    # Inherits from transformers.TrainingArguments

    # --- Debug: Print the path of the TrainingArguments class being used ---
    try:
        print(f"DEBUG: Loading TrainingArguments from: {inspect.getfile(transformers.TrainingArguments)}")
    except Exception as e:
        print(f"DEBUG: Could not get file for TrainingArguments: {e}")
    # --- End Debug ---

    args = TrainingArguments(
        output_dir=str(model_output_dir),      # Directory to save model checkpoints
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",          # Evaluate at the end of each epoch
        save_strategy="epoch",          # Save a checkpoint at the end of each epoch
        save_total_limit=1,             # Keep only the best checkpoint
        load_best_model_at_end=True,    # Load the best model found during training
        metric_for_best_model="eval_f1", # Use F1 score on eval set to determine best model
        use_mps_device=use_mps,         # Enable MPS if available
        # M2 Max supports BF16 and MPS – please change the following variables if you don't have a compatible GPU
        bf16=True,                      # Enable BF16 since M2 Max supports it
        fp16=False,                     # Disable FP16 when using BF16
        logging_steps=50,               # Log metrics every 50 steps
        push_to_hub=False,              # Set to True later if you want to upload to Hugging Face Hub
    )

    # --- Initialize Trainer --- 
    # SpanMarker's Trainer subclasses the 🤗 Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # No compute_metrics needed; SpanMarker handles standard seqeval metrics
    )

    # --- Train the Model --- 
    print("Starting model training...")
    trainer.train()

    # --- Evaluate the Final Model --- 
    print("Evaluating final model performance on the evaluation set...")
    metrics = trainer.evaluate(eval_dataset, metric_key_prefix="eval")
    print("Evaluation metrics:", metrics)
    trainer.save_metrics("eval", metrics)

    # --- Save the Trained Model --- 
    # The best model is already loaded (load_best_model_at_end=True)
    final_model_path = model_output_dir / "checkpoint-final"
    print(f"Saving the best model to {final_model_path}...")
    trainer.save_model(str(final_model_path))
    # Save the tokenizer configuration alongside the model
    tokenizer.save_pretrained(str(final_model_path))
    print("Model training and saving complete.")

# ⚠️ To train the model after labelling, uncomment the following line:
train_span_marker_model()

# Step 3: Use the trained model to extract security codes.
def extract_codes_with_model(messages, model_path=MODEL_OUTPUT_DIR / "checkpoint-final"):
   """Uses a trained SpanMarkerNER model to extract security codes from messages."""
   if isinstance(messages, str):
       messages = [messages] # Ensure input is a list for batch processing

   model_path_str = str(model_path) # Convert Path object to string
   output_dir_str = str(MODEL_OUTPUT_DIR) # Base output dir for fallback check

   # Check if the final checkpoint path exists, otherwise try the base output directory
   if not Path(model_path_str).exists():
       print(f"Checkpoint path {model_path_str} not found.")
       if Path(output_dir_str).exists() and Path(output_dir_str, "config.json").exists():
           print(f"Attempting to load directly from base directory {output_dir_str}...")
           model_path_str = output_dir_str
       else:
            print(f"Error: Model not found at specified path '{model_path_str}' or base directory '{output_dir_str}'. Please train the model first or check the path.")
            return None

   print(f"Loading trained SpanMarker model from {model_path_str}...")
   try:
       model = SpanMarkerModel.from_pretrained(model_path_str)
       # Optionally move to GPU if available and desired for faster inference
       # if torch.cuda.is_available(): model.cuda()
       # elif torch.backends.mps.is_available(): model.to('mps')
   except Exception as e:
       print(f"Error loading model from {model_path_str}: {e}")
       return None

   print(f"Extracting codes from {len(messages)} message(s)...")
   # Run batch prediction
   extracted_entities_batch = model.predict(messages)

   # Process results: Extract only the text span for entities labelled 'CODE'
   extracted_codes = []
   for message_entities in extracted_entities_batch:
        codes_in_message = [entity['span'] for entity in message_entities if entity['label'] == 'CODE']
        extracted_codes.append(codes_in_message)

   # Example of how to push the model to the Hugging Face Hub:
   # 1. Ensure you are logged in: `huggingface-cli login`
   # 2. Uncomment the following lines:
   # try:
   #    print(f"Pushing model '{NEW_MODEL_NAME}' to Hub...")
   #    model.push_to_hub(f"your-hf-username/{NEW_MODEL_NAME}") # Replace with your HF username
   #    # Also push the tokenizer if needed (often included with model save/push)
   #    # tokenizer = AutoTokenizer.from_pretrained(model_path_str)
   #    # tokenizer.push_to_hub(f"your-hf-username/{NEW_MODEL_NAME}")
   #    print("Model pushed successfully.")
   # except Exception as e:
   #    print(f"Error pushing model to Hub: {e}")

   return extracted_codes

# --- Example Usage --- 
# To directly use the model after training, uncomment the following lines:
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


# --- Main Execution Guard --- 
if __name__ == "__main__":
    print("Script execution started. Uncomment function calls (label_messages_for_ner, train_span_marker_model, example usage) to run specific steps.")
    print("Thank you for using Security Autofill 🙇")
    pass # Keeps the script runnable without uncommenting steps
