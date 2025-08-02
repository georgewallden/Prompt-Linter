# prompt-linter/scripts/analyze_dataset.py

import json
from pathlib import Path
import boto3

# --- Configuration ---
PROCESSED_BUCKET_NAME = "prompt-linter-data-processed-us-east-1"
PROCESSED_FILE_KEY = "master.jsonl"


def download_processed_data_if_needed(target_path):
    """
    Checks if the master.jsonl file exists locally. If not, it downloads
    it from the S3 processed data bucket.
    """
    if target_path.exists():
        print(f"Dataset file already exists at {target_path}")
        return

    print(f"Dataset file not found locally. Downloading from S3...")
    print(f"Bucket: {PROCESSED_BUCKET_NAME}, Key: {PROCESSED_FILE_KEY}")

    # Ensure the parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        s3_client = boto3.client("s3")
        s3_client.download_file(
            Bucket=PROCESSED_BUCKET_NAME,
            Key=PROCESSED_FILE_KEY,
            Filename=str(target_path)
        )
        print("Download complete.")
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"Failed to download from S3: {e}")
        print("Please ensure your AWS credentials are configured correctly.")
        exit(1) # Exit the script if download fails


def create_intent_map():
    """
    Scans the master.jsonl file to find all unique intents and creates a
    mapping from intent string to an integer ID. Saves the map to a file.
    """
    print("\n--- Starting Dataset Analysis ---")
    
    # Define paths relative to this script's location
    scripts_dir = Path(__file__).parent
    data_path = scripts_dir.parent / "data" / "processed" / PROCESSED_FILE_KEY
    artifacts_path = scripts_dir.parent / "artifacts"
    output_path = artifacts_path / "intent_map.json"
    
    # --- New Step: Ensure we have the data locally ---
    download_processed_data_if_needed(data_path)
    
    artifacts_path.mkdir(exist_ok=True)
    unique_intents = set()

    print(f"Reading data from: {data_path}")
    # This part remains the same
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                intent = record.get("labels", {}).get("intent")
                if intent:
                    unique_intents.add(intent)
            except json.JSONDecodeError:
                continue
    
    sorted_intents = sorted(list(unique_intents))
    intent_map = {intent_name: i for i, intent_name in enumerate(sorted_intents)}
    num_intents = len(intent_map)
    print(f"\nFound {num_intents} unique intents.")
    
    print(f"Saving intent map to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(intent_map, f, indent=2)
        
    print("--- Analysis Complete ---")

if __name__ == "__main__":
    create_intent_map()