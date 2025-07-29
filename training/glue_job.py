# prompt-linter/training/glue_job.py

import json
import boto3
import pandas as pd
from io import StringIO

# --- Configuration ---
RAW_BUCKET_NAME = "prompt-linter-data-raw-us-east-1"
PROCESSED_BUCKET_NAME = "prompt-linter-data-processed-us-east-1"
OUTPUT_FILE_KEY = "master.jsonl"

def process_dstc11(s3_client):
    """
    Reads all splits of the DSTC-11 dataset from S3, mines labels for
    risk and traits, and returns a list of standardized records.
    """
    print("\n--> Starting processing of DSTC-11-Track-5")
    
    # Process all available splits for this dataset
    source_prefix = "DSTC-11-Track-5/"
    files_to_process = [
        f"{source_prefix}train.jsonl",
        f"{source_prefix}test.jsonl",
        f"{source_prefix}validation.jsonl"
    ]
    
    all_records = []
    for file_key in files_to_process:
        print(f"  -> Reading {file_key}...")
        response = s3_client.get_object(Bucket=RAW_BUCKET_NAME, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        df = pd.read_json(StringIO(content), lines=True)

        # --- Transformation Logic ---
        df['risk_score'] = df['target'].apply(lambda is_grounded: 0.1 if is_grounded else 0.9)
        df['prompt'] = df['log'].apply(
            lambda log_list: next((item['text'] for item in reversed(log_list) if item['speaker'] == 'U'), None)
        )
        df['clarity_score'] = (df['prompt'].str.len() / 500).clip(0, 1)

        # --- Standardize to Final Output Format ---
        for index, row in df.iterrows():
            record = {
                "text": row['prompt'],
                "source": "DSTC-11-Track-5",
                "labels": {
                    "intent": "General-Query", # Default intent for this dataset
                    "traits": {"clarity": row['clarity_score'], "specificity": None},
                    "risk_score": row['risk_score']
                }
            }
            all_records.append(record)

    print(f"--> Finished DSTC-11. Processed {len(all_records)} records.")
    return all_records

def process_multiwoz(s3_client):
    """
    Reads the MultiWOZ dataset, mines for intent labels, and returns a
    list of standardized records.
    """
    print("\n--> Starting processing of MultiWOZ-Instruction")

    source_prefix = "MultiWOZ-Instruction/"
    files_to_process = [
        f"{source_prefix}train.jsonl",
        f"{source_prefix}test.jsonl",
        f"{source_prefix}validation.jsonl"
    ]
    
    df_list = []
    for file_key in files_to_process:
        print(f"  -> Reading {file_key}...")
        response = s3_client.get_object(Bucket=RAW_BUCKET_NAME, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        df_split = pd.read_json(StringIO(content), lines=True)
        df_list.append(df_split)

    df = pd.concat(df_list, ignore_index=True)
    print("-> Successfully parsed MultiWOZ data into a pandas DataFrame!")

    print("\nStarting transformation: Mining labels from source data...")

    df['prompt'] = df['context'].str.split('user: ').str[-1].str.split('system:').str[0].str.strip()
    print("  -> Extracted 'prompt' text.")

    def get_intent(json_str):
        # --- FIX #1: Handle empty/None values ---
        if not isinstance(json_str, str):
            return "General-Query"
        try:
            data = json.loads(json_str)
            domain = list(data.keys())[0]
            return f"Find-{domain.capitalize()}"
        except (json.JSONDecodeError, IndexError):
            return "General-Query"
    df['intent'] = df['json_answer'].apply(get_intent)
    print("  -> Mined 'intent' label.")

    df['risk_score'] = 0.1
    print("  -> Assigned static 'risk_score'.")

    df['clarity_score'] = (df['prompt'].str.len() / 500).clip(0, 1)
    
    def get_specificity(json_str):
        # --- FIX #2: Handle empty/None values ---
        if not isinstance(json_str, str):
            return 0.0
        try:
            data = json.loads(json_str)
            num_slots = len(list(data.values())[0])
            return min(num_slots / 5.0, 1.0)
        except (json.JSONDecodeError, IndexError):
            return 0.0
    df['specificity_score'] = df['json_answer'].apply(get_specificity)

    # --- FINALIZATION LOGIC ---
    all_records = []
    for index, row in df.iterrows():
        record = {
            "text": row['prompt'],
            "source": "MultiWOZ-Instruction",
            "labels": {
                "intent": row['intent'],
                "traits": {"clarity": row['clarity_score'], "specificity": row['specificity_score']},
                "risk_score": row['risk_score']
            }
        }
        all_records.append(record)

    print(f"--> Finished MultiWOZ. Processed {len(all_records)} records.")
    return all_records

def process_sgd(s3_client):
    """
    Reads the SGD-Instruction dataset. This version uses a custom parser
    tailored to the unique structure of the SGD 'text' block.
    """
    print("\n--> Starting processing of SGD-Instruction")
    
    source_prefix = "SGD-Instruction/"
    file_key = f"{source_prefix}train.jsonl"
    
    print(f"  -> Reading {file_key}...")
    response = s3_client.get_object(Bucket=RAW_BUCKET_NAME, Key=file_key)
    content = response['Body'].read().decode('utf-8')
    df = pd.read_json(StringIO(content), lines=True)
    print(f"-> Successfully parsed SGD data into a pandas DataFrame with {len(df)} rows.")

    # --- Data Cleaning ---
    df.dropna(subset=['text'], inplace=True)
    df['text'] = df['text'].astype(str)
    
    print("Starting transformation with custom SGD parsing logic...")

    # --- Custom SGD Transformation Logic ---

    # 1. Helper function to safely extract a value from a line.
    def extract_line_value(text_block, key):
        try:
            # Split by the key, take the part after it, then take the first line.
            return text_block.split(key)[1].split('\n')[0].strip()
        except IndexError:
            return None # Return None if the key is not found

    # 2. Apply the helper to get the core fields.
    df['prompt'] = df['text'].apply(lambda x: extract_line_value(x, 'user:'))
    df['intent'] = df['text'].apply(lambda x: extract_line_value(x, 'Active intent:'))
    df['slot_values'] = df['text'].apply(lambda x: extract_line_value(x, 'Slot values:'))

    # 3. Clean up failed parses
    df.dropna(subset=['prompt', 'intent', 'slot_values'], inplace=True)
    
    # 4. Assign static risk score
    df['risk_score'] = 0.1

    # 5. Calculate traits based on our newly parsed columns
    df['clarity_score'] = (df['prompt'].str.len() / 500).clip(0, 1)
    # Specificity is the number of colons in the slot values string (e.g., "key:value")
    df['specificity_score'] = (df['slot_values'].str.count(':') / 5.0).clip(0, 1)

    # --- Finalization Logic ---
    all_records = []
    for index, row in df.iterrows():
        record = {
            "text": row['prompt'],
            "source": "SGD-Instruction",
            "labels": {
                "intent": row['intent'],
                "traits": {"clarity": row['clarity_score'], "specificity": row['specificity_score']},
                "risk_score": row['risk_score']
            }
        }
        all_records.append(record)

    print(f"--> Finished SGD. Processed {len(all_records)} records.")
    return all_records

def process_nyt(s3_client):
    """
    Reads the NYT-Topics dataset and transforms the headlines into 
    low-risk, declarative statement examples for our model.
    """
    print("\n--> Starting processing of NYT-Topics")
    
    source_prefix = "NYT-Topics/"
    file_key = f"{source_prefix}train.jsonl"
    
    print(f"  -> Reading {file_key}...")
    response = s3_client.get_object(Bucket=RAW_BUCKET_NAME, Key=file_key)
    content = response['Body'].read().decode('utf-8')
    df = pd.read_json(StringIO(content), lines=True)
    print(f"-> Successfully parsed NYT-Topics data with {len(df)} rows.")

    # --- Data Cleaning ---
    df.dropna(subset=['text', 'topic_name'], inplace=True)
    
    print("Starting transformation for NYT-Topics...")

    # --- Finalization Logic ---
    all_records = []
    for index, row in df.iterrows():
        record = {
            "text": row['text'],
            "source": "NYT-Topics",
            "labels": {
                # Use the topic_name directly as the intent
                "intent": row['topic_name'], 
                "traits": {
                    # Assign static, representative scores for this data type
                    "clarity": 0.9, 
                    "specificity": 0.2
                },
                # Assign a very low risk score for factual headlines
                "risk_score": 0.05 
            }
        }
        all_records.append(record)

    print(f"--> Finished NYT-Topics. Processed {len(all_records)} records.")
    return all_records

def main():
    """
    Main ETL script execution function.
    Orchestrates the E-T-L process from raw S3 to processed S3.
    """
    print("--- Starting PromptLinter ETL Job ---")
    s3_client = boto3.client("s3")
    
    # --- EXTRACT & TRANSFORM ---
    all_processed_records = []
    all_processed_records.extend(process_dstc11(s3_client))
    all_processed_records.extend(process_multiwoz(s3_client))
    all_processed_records.extend(process_sgd(s3_client))
    all_processed_records.extend(process_nyt(s3_client))

    print(f"\n--- E-T Complete. Total records to be loaded: {len(all_processed_records)} ---")

    # --- LOAD ---
    if not all_processed_records:
        print("No records to upload. Exiting.")
        return

    print(f"Writing {len(all_processed_records)} records to s3://{PROCESSED_BUCKET_NAME}/{OUTPUT_FILE_KEY}")

    # Convert each dictionary to a JSON string
    jsonl_content = "\n".join([json.dumps(record) for record in all_processed_records])

    # Upload the in-memory string to the S3 bucket
    s3_client.put_object(
        Bucket=PROCESSED_BUCKET_NAME,
        Key=OUTPUT_FILE_KEY,
        Body=jsonl_content.encode('utf-8') # Encode string to bytes for upload
    )

    print("--- ETL Job Finished Successfully ---")
    

if __name__ == "__main__":
    main()