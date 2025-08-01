# prompt-linter/training/glue_job.py

import json
import boto3
import pandas as pd
import tempfile
import os
import codecs

# --- Configuration ---
RAW_BUCKET_NAME = "prompt-linter-data-raw-us-east-1"
PROCESSED_BUCKET_NAME = "prompt-linter-data-processed-us-east-1"
OUTPUT_FILE_KEY = "master.jsonl"
CHUNK_SIZE = 20000

# --- Transformation Functions for each Dataset ---

def transform_dstc11(df):
    df['risk_score'] = df['target'].apply(lambda is_grounded: 0.1 if is_grounded else 0.9)
    df['prompt'] = df['log'].apply(lambda log_list: next((item['text'] for item in reversed(log_list) if item['speaker'] == 'U'), None))
    df['clarity_score'] = (df['prompt'].str.len() / 500).clip(0, 1)
    df.dropna(subset=['prompt'], inplace=True)
    
    records = []
    for _, row in df.iterrows():
        records.append({
            "text": row['prompt'], "source": "DSTC-11-Track-5", "labels": {
                "intent": "General-Query",
                "traits": {"clarity": row['clarity_score'], "specificity": None},
                "risk_score": row['risk_score']
            }
        })
    return records

def transform_multiwoz(df):
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
    df.dropna(subset=['prompt'], inplace=True)

    records = []
    for _, row in df.iterrows():
        records.append({
            "text": row['prompt'], "source": "MultiWOZ-Instruction", "labels": {
                "intent": row['intent'],
                "traits": {"clarity": row['clarity_score'], "specificity": row['specificity_score']},
                "risk_score": row['risk_score']
            }
        })
    return records
    
def transform_sgd(df):
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
            "text": row['prompt'], "source": "SGD-Instruction", "labels": {
                "intent": row['intent'],
                "traits": {"clarity": row['clarity_score'], "specificity": row['specificity_score']},
                "risk_score": row['risk_score']
            }
        })
    return records

def transform_nyt(df):
    df.dropna(subset=['text', 'topic_name'], inplace=True)
    records = []
    for _, row in df.iterrows():
        records.append({
            "text": row['text'], "source": "NYT-Topics", "labels": {
                "intent": row['topic_name'],
                "traits": {"clarity": 0.9, "specificity": 0.2},
                "risk_score": 0.05
            }
        })
    return records

# --- THIS IS THE MISSING HELPER FUNCTION ---
def process_source(s3_client, source_name, files, transform_func, temp_file_handle):
    """Helper to process all files for a single data source."""
    print(f"\n--> Processing source: {source_name}")
    source_records = 0
    for file_name in files:
        file_key = f"{source_name}/{file_name}"
        print(f"  -> Streaming {file_key} in chunks...")
        
        response = s3_client.get_object(Bucket=RAW_BUCKET_NAME, Key=file_key)
        body_stream = codecs.getreader("utf-8")(response['Body'])
        
        for df_chunk in pd.read_json(body_stream, lines=True, chunksize=CHUNK_SIZE):
            processed_records = transform_func(df_chunk)
            if processed_records:
                for record in processed_records:
                    temp_file_handle.write(json.dumps(record) + '\n')
                source_records += len(processed_records)
    
    print(f"--> Finished {source_name}. Wrote {source_records} records.")
    return source_records

# --- Main Orchestrator ---
def main():
    print("--- Starting PromptLinter ETL Job (Final Version) ---")
    s3_client = boto3.client("s3")

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as temp_f:
        temp_file_path = temp_f.name
        print(f"Created temporary file at: {temp_file_path}")
        
        total_records = 0
        total_records += process_source(s3_client, "DSTC-11-Track-5", ["train.jsonl", "test.jsonl", "validation.jsonl"], transform_dstc11, temp_f)
        total_records += process_source(s3_client, "MultiWOZ-Instruction", ["train.jsonl", "test.jsonl", "validation.jsonl"], transform_multiwoz, temp_f)
        total_records += process_source(s3_client, "SGD-Instruction", ["train.jsonl"], transform_sgd, temp_f)
        total_records += process_source(s3_client, "NYT-Topics", ["train.jsonl"], transform_nyt, temp_f)

    print(f"\n--- E-T Complete. Total records in temp file: {total_records} ---")

    print(f"Uploading final file to s3://{PROCESSED_BUCKET_NAME}/{OUTPUT_FILE_KEY}")
    s3_client.upload_file(Filename=temp_file_path, Bucket=PROCESSED_BUCKET_NAME, Key=OUTPUT_FILE_KEY)
    os.remove(temp_file_path)

    print("--- ETL Job Finished Successfully ---")

if __name__ == "__main__":
    main()