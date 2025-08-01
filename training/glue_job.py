# prompt-linter/training/glue_job.py

import json
import boto3
import pandas as pd
from io import StringIO
import tempfile
import os

# --- Configuration ---
RAW_BUCKET_NAME = "prompt-linter-data-raw-us-east-1"
PROCESSED_BUCKET_NAME = "prompt-linter-data-processed-us-east-1"
OUTPUT_FILE_KEY = "master.jsonl"
CHUNK_SIZE = 20000  # Process 20,000 lines at a time

def process_and_write_chunk(df_chunk, source_name, transformation_func, temp_file_handle):
    """
    Applies a transformation to a DataFrame chunk and writes it to a file.
    Returns the number of records processed.
    """
    processed_records = transformation_func(df_chunk, source_name)
    if processed_records:
        for record in processed_records:
            temp_file_handle.write(json.dumps(record) + '\n')
        return len(processed_records)
    return 0

def transform_dstc11(df, source_name):
    df['risk_score'] = df['target'].apply(lambda is_grounded: 0.1 if is_grounded else 0.9)
    df['prompt'] = df['log'].apply(lambda log_list: next((item['text'] for item in reversed(log_list) if item['speaker'] == 'U'), None))
    df['clarity_score'] = (df['prompt'].str.len() / 500).clip(0, 1)
    
    records = []
    for _, row in df.iterrows():
        records.append({
            "text": row['prompt'], "source": source_name, "labels": {
                "intent": "General-Query",
                "traits": {"clarity": row['clarity_score'], "specificity": None},
                "risk_score": row['risk_score']
            }
        })
    return records

def transform_multiwoz(df, source_name):
    df['prompt'] = df['context'].str.split('user: ').str[-1].str.split('system:').str[0].str.strip()
    def get_intent(json_str):
        if not isinstance(json_str, str): return "General-Query"
        try:
            data = json.loads(json_str); domain = list(data.keys())[0]; return f"Find-{domain.capitalize()}"
        except (json.JSONDecodeError, IndexError): return "General-Query"
    df['intent'] = df['json_answer'].apply(get_intent)
    df['risk_score'] = 0.1
    df['clarity_score'] = (df['prompt'].str.len() / 500).clip(0, 1)

    def get_specificity(json_str):
        if not isinstance(json_str, str): return 0.0
        try:
            data = json.loads(json_str); num_slots = len(list(data.values())[0]); return min(num_slots / 5.0, 1.0)
        except (json.JSONDecodeError, IndexError): return 0.0
    df['specificity_score'] = df['json_answer'].apply(get_specificity)
    
    records = []
    for _, row in df.iterrows():
        records.append({
            "text": row['prompt'], "source": source_name, "labels": {
                "intent": row['intent'],
                "traits": {"clarity": row['clarity_score'], "specificity": row['specificity_score']},
                "risk_score": row['risk_score']
            }
        })
    return records
    
def transform_sgd(df, source_name):
    df.dropna(subset=['text'], inplace=True); df['text'] = df['text'].astype(str)
    def extract_line_value(text_block, key):
        try: return text_block.split(key)[1].split('\n')[0].strip()
        except IndexError: return None
    df['prompt'] = df['text'].apply(lambda x: extract_line_value(x, 'user:'))
    df['intent'] = df['text'].apply(lambda x: extract_line_value(x, 'Active intent:'))
    df['slot_values'] = df['text'].apply(lambda x: extract_line_value(x, 'Slot values:'))
    df.dropna(subset=['prompt', 'intent', 'slot_values'], inplace=True)
    df['risk_score'] = 0.1
    df['clarity_score'] = (df['prompt'].str.len() / 500).clip(0, 1)
    df['specificity_score'] = (df['slot_values'].str.count(':') / 5.0).clip(0, 1)

    records = []
    for _, row in df.iterrows():
        records.append({
            "text": row['prompt'], "source": source_name, "labels": {
                "intent": row['intent'],
                "traits": {"clarity": row['clarity_score'], "specificity": row['specificity_score']},
                "risk_score": row['risk_score']
            }
        })
    return records

def transform_nyt(df, source_name):
    df.dropna(subset=['text', 'topic_name'], inplace=True)
    records = []
    for _, row in df.iterrows():
        records.append({
            "text": row['text'], "source": source_name, "labels": {
                "intent": row['topic_name'],
                "traits": {"clarity": 0.9, "specificity": 0.2},
                "risk_score": 0.05
            }
        })
    return records

def main():
    print("--- Starting PromptLinter ETL Job (True Streaming Mode) ---")
    s3_client = boto3.client("s3")

    datasets_to_process = {
        "DSTC-11-Track-5": {"files": ["train.jsonl", "test.jsonl", "validation.jsonl"], "transform": transform_dstc11},
        "MultiWOZ-Instruction": {"files": ["train.jsonl", "test.jsonl", "validation.jsonl"], "transform": transform_multiwoz},
        "SGD-Instruction": {"files": ["train.jsonl"], "transform": transform_sgd},
        "NYT-Topics": {"files": ["train.jsonl"], "transform": transform_nyt},
    }

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as temp_f:
        temp_file_path = temp_f.name
        print(f"Created temporary file at: {temp_file_path}")
        
        total_records = 0
        for source_name, config in datasets_to_process.items():
            print(f"\n--> Processing source: {source_name}")
            source_records = 0
            for file_name in config["files"]:
                file_key = f"{source_name}/{file_name}"
                print(f"  -> Streaming {file_key} in chunks...")
                
                # --- THIS IS THE CRITICAL FIX ---
                # We get the streaming body from S3...
                response = s3_client.get_object(Bucket=RAW_BUCKET_NAME, Key=file_key)
                
                # ...and pass the stream DIRECTLY to pandas. No intermediate string.
                for df_chunk in pd.read_json(response['Body'], lines=True, chunksize=CHUNK_SIZE):
                    processed_count = process_and_write_chunk(df_chunk, source_name, config["transform"], temp_f)
                    source_records += processed_count
            
            total_records += source_records
            print(f"--> Finished {source_name}. Wrote {source_records} records. Total so far: {total_records}")

    print(f"\n--- E-T Complete. Total records in temp file: {total_records} ---")

    print(f"Uploading final file to s3://{PROCESSED_BUCKET_NAME}/{OUTPUT_FILE_KEY}")
    s3_client.upload_file(Filename=temp_file_path, Bucket=PROCESSED_BUCKET_NAME, Key=OUTPUT_FILE_KEY)
    os.remove(temp_file_path)

    print("--- ETL Job Finished Successfully ---")

if __name__ == "__main__":
    main()