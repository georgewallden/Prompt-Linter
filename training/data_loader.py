# prompt-linter/training/data_loader.py

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class PromptDataset(Dataset):

    """
    A PyTorch Dataset for loading and processing the master.jsonl file.
    """
    def __init__(self, data_source, tokenizer, intent_map):
        """
        Initializes the Dataset.

        Args:
            data_source (str or list): Either a path to the .jsonl file or a 
                                       list of JSON strings.
            tokenizer: The Hugging Face tokenizer to use.
            intent_map (dict): A mapping from intent strings to integer IDs.
        """
        self.tokenizer = tokenizer
        self.intent_map = intent_map
        
        # --- THIS IS THE MODIFIED BLOCK ---
        if isinstance(data_source, str) or isinstance(data_source, Path):
            print(f"Loading dataset from file: {data_source}")
            with open(data_source, 'r', encoding='utf-8') as f:
                self.data = f.readlines()
        elif isinstance(data_source, list):
            print(f"Loading dataset from a list of {len(data_source)} records.")
            self.data = data_source
        else:
            raise TypeError("data_source must be a file path (str) or a list of strings.")
        
        print(f"Dataset initialized with {len(self.data)} records.")


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):

        # 1. Get the raw JSON string for the given index.
        json_string = self.data[idx]
        
        # 2. Parse the JSON string into a Python dictionary.
        # Use a try-except block to handle any potential malformed lines.
        try:
            record = json.loads(json_string)
        except json.JSONDecodeError:
            print(f"Warning: Skipping malformed JSON line at index {idx}")
            # Return the first sample if this one is broken, a simple recovery strategy.
            return self.__getitem__(0)

        # 3. Tokenize the input text.
        text = record.get('text', '') # Safely get the text, default to empty string
        tokenized_output = self.tokenizer(
            text,
            truncation=True,    # Truncate sequences longer than the model's max length
            padding=False,      # We will handle padding later in the collate_fn
            max_length=512,     # A standard max length for transformers
            return_tensors=None # We want lists of numbers, not tensors, at this stage
        )
        
        # 4. Process the labels.
        labels_data = record.get('labels', {})
        
        # Intent: Convert string to integer ID. Default to 0 if not found.
        intent_str = labels_data.get('intent', "General-Query")
        intent_id = self.intent_map.get(intent_str, 0)
        
        # Traits: Get clarity and specificity, handling missing values.
        traits_dict = labels_data.get('traits', {})
        clarity = traits_dict.get('clarity', 0.0) or 0.0 # Handles None or missing
        specificity = traits_dict.get('specificity', 0.0) or 0.0 # Handles None or missing
        trait_scores = [clarity, specificity]
        
        # Risk: Get risk score, handling missing values.
        risk_score = labels_data.get('risk_score', 0.0) or 0.0 # Handles None or missing

        # 5. Return all pieces in a dictionary.
        return {
            "text": text,
            "input_ids": tokenized_output['input_ids'],
            "attention_mask": tokenized_output['attention_mask'],
            "intent_label": intent_id,
            "trait_labels": trait_scores,
            "risk_label": risk_score
        }
    
def collate_fn(batch, tokenizer):
    """
    Custom collate function to handle dynamic padding of batches.
    
    Args:
        batch (list): A list of dictionaries, where each dictionary is a sample
                      from the PromptDataset.
        tokenizer: The Hugging Face tokenizer.
        
    Returns:
        dict: A dictionary containing the batched and padded tensors.
    """
    # 1. Separate the different components of the batch.
    # The 'batch' is a list of dicts. We convert it to a dict of lists.
    texts = [item['text'] for item in batch]
    intent_labels = [item['intent_label'] for item in batch]
    trait_labels = [item['trait_labels'] for item in batch]
    risk_labels = [item['risk_label'] for item in batch]
    
    # 2. Use the tokenizer to perform dynamic padding.
    # It will find the max length in this specific batch and pad everything to that size.
    tokenized_batch = tokenizer(
        texts, 
        padding=True,       # Enable dynamic padding
        truncation=True,    # Truncate sequences if they are somehow still too long
        max_length=512,
        return_tensors="pt" # Return PyTorch tensors
    )
    
    # 3. Convert the label lists into tensors.
    intent_labels = torch.tensor(intent_labels, dtype=torch.long)
    trait_labels = torch.tensor(trait_labels, dtype=torch.float)
    risk_labels = torch.tensor(risk_labels, dtype=torch.float)
    
    # 4. Return the final batch as a dictionary.
    # The keys here must match what the training loop and model expect.
    return {
        'input_ids': tokenized_batch['input_ids'],
        'attention_mask': tokenized_batch['attention_mask'],
        'labels': {
            'intent': intent_labels,
            'traits': trait_labels,
            'risk': risk_labels
        }
    }
