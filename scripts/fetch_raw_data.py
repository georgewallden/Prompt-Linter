# prompt-linter/scripts/fetch_raw_data.py

import os
from datasets import load_dataset
from pathlib import Path

# --- Configuration ---
# Dictionary of Hugging Face dataset identifiers
# Key: The simple name we'll use for the local folder
# Value: The full Hugging Face identifier
DATASETS_TO_FETCH = {
    "DSTC-11-Track-5": "NomaDamas/DSTC-11-Track-5",
    "MultiWOZ-Instruction": "AtheerAlgherairy/DST_Multiwoz21_instruction_Tuning",
    "SGD-Instruction": "amay01/llm-sgd-dst8-split-training-data-jsonl",
    "NYT-Topics": "dstefa/New_York_Times_Topics",
}

# The root directory for our raw data, relative to the script location
# Assumes script is in prompt-linter/scripts/ and data goes in prompt-linter/data/
RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw"

# --- Main Execution ---
def main():
    """
    Downloads and saves datasets from Hugging Face to a local directory
    in the standardized .jsonl format.
    """
    print(f"Starting data fetch process. Raw data will be saved to: {RAW_DATA_PATH}")
    
    # Ensure the root data directory exists
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    for local_name, hf_identifier in DATASETS_TO_FETCH.items():
        print(f"\nProcessing dataset: {local_name} ({hf_identifier})")
        
        try:
            # Create a dedicated folder for this dataset
            dataset_dir = RAW_DATA_PATH / local_name
            dataset_dir.mkdir(exist_ok=True)
            
            # Load the dataset from Hugging Face
            # The `trust_remote_code=True` may be needed for some datasets
            dataset = load_dataset(hf_identifier, trust_remote_code=True)
            
            # Save each split (e.g., 'train', 'test') as a .jsonl file
            for split_name in dataset.keys():
                split_path = dataset_dir / f"{split_name}.jsonl"
                print(f"  -> Saving split '{split_name}' to {split_path}")
                dataset[split_name].to_json(split_path)
                
            print(f"Successfully processed and saved {local_name}.")
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {local_name}: {e}")
            print("  Skipping this dataset.")
            continue

if __name__ == "__main__":
    main()