# prompt-linter/training/train.py

# Import necessary data handling libraries
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import json
from pathlib import Path

# Import our custom modules
from model import PromptLinterModel
from loss import GuardrailLoss
from data_loader import PromptDataset, collate_fn

# Import the progress bar library
from tqdm import tqdm

# --- 1. Configuration ---
# All hyperparameters and configuration settings are in one place.
CONFIG = {
    "model_name": "distilbert-base-uncased",
    "data_path": Path(__file__).parent.parent / "data" / "processed" / "master.jsonl",
    "artifacts_path": Path(__file__).parent.parent / "artifacts",
    "batch_size": 32,
    "epochs": 3,
    "learning_rate": 2e-5,
    "loss_weights": {'intent': 1.0, 'traits': 0.75, 'risk': 1.25},
    "num_intents": None, 
}

def evaluate(model, data_loader, loss_fn, device):
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): The DataLoader for the evaluation data.
        loss_fn: The loss function.
        device: The device to run the evaluation on.

    Returns:
        float: The average loss over the evaluation dataset.
    """
    # 1. Set the model to evaluation mode
    model.eval()
    
    total_loss = 0
    
    # 2. Disable gradient calculations for efficiency and correctness
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move batch to the correct device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = {key: val.to(device) for key, val in batch['labels'].items()}
            
            # Perform a forward pass
            predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate the loss
            loss = loss_fn(predictions, labels)
            total_loss += loss.item()
            
    return total_loss / len(data_loader)

def train():
    """The main training function."""
    
    # --- 2. Setup and Initialization ---
    
    CONFIG["artifacts_path"].mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    intent_map_path = CONFIG["artifacts_path"] / "intent_map.json"
    print(f"Loading intent map from {intent_map_path}")
    with open(intent_map_path, 'r', encoding='utf-8') as f:
        intent_map = json.load(f)
    
    CONFIG["num_intents"] = len(intent_map)
    print(f"Found {CONFIG['num_intents']} unique intents.")
    
    # Initialize the tokenizer from the pre-trained model
    print(f"Initializing tokenizer for model: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    #=============================================================================#

    # TODO: This logic belongs in the Dataset class. Refactor later.
    # For now, we load and split the data directly in the training script.
    print(f"Reading and splitting data from {CONFIG['data_path']}")
    with open(CONFIG['data_path'], 'r', encoding='utf-8') as f:
        all_data = f.readlines()
    
    # Shuffle the data before splitting
    import random
    random.seed(42) # for reproducibility
    random.shuffle(all_data)

    # Perform a 90/10 train/validation split
    split_idx = int(len(all_data) * 0.9)
    train_lines = all_data[:split_idx]
    val_lines = all_data[split_idx:]
    
    print(f"Training set size: {len(train_lines)}")
    print(f"Validation set size: {len(val_lines)}")

    # Create the Dataset objects
    # We pass the list of lines directly, avoiding re-reading the file.
    # Note: We will need to slightly modify PromptDataset to accept a list of lines.
    train_dataset = PromptDataset(train_lines, tokenizer, intent_map)
    val_dataset = PromptDataset(val_lines, tokenizer, intent_map)

    # Create the DataLoader objects
    # We use a lambda to pass our tokenizer to the collate_fn.
    collate_wrapper = lambda batch: collate_fn(batch, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_wrapper
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False, # No need to shuffle validation data
        collate_fn=collate_wrapper
    )

    #=============================================================================#
    
    # Initialize the model and move it to the configured device
    print(f"\nInitializing model: {CONFIG['model_name']}")
    model = PromptLinterModel(
        model_name=CONFIG['model_name'],
        num_intents=CONFIG['num_intents']
    ).to(device)

    # Initialize our custom loss function
    loss_fn = GuardrailLoss(weights=CONFIG['loss_weights'])
    
    # Initialize the optimizer
    # AdamW is a variant of Adam that is preferred for training transformers
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Initialize the learning rate scheduler
    # A linear schedule with warmup is a standard best practice for fine-tuning transformers.
    num_training_steps = len(train_loader) * CONFIG['epochs']
    num_warmup_steps = int(num_training_steps * 0.1) # 10% of steps for warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    print("Model, Loss, Optimizer, and Scheduler initialized.")

    #=============================================================================#
   
    # --- 3. The Training Loop (To be implemented) ---
    print("\n--- Starting Training ---")

    for epoch in range(CONFIG["epochs"]):
        print(f"\n===== Epoch {epoch + 1}/{CONFIG['epochs']} =====")
        
        # --- Training Phase ---
        model.train() # Set the model to training mode
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to the correct device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Move all labels in the nested dictionary to the device
            labels = {key: val.to(device) for key, val in batch['labels'].items()}

            # 1. Clear any previously calculated gradients
            optimizer.zero_grad()
            
            # 2. Perform a forward pass
            predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 3. Calculate the loss
            loss = loss_fn(predictions, labels)
            
            # 4. Perform backpropagation to calculate gradients
            loss.backward()
            
            # 5. (Optional but good practice) Clip gradients to prevent them from exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 6. Update the model's weights
            optimizer.step()
            
            # 7. Update the learning rate
            scheduler.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
    # --- 4. Final Evaluation and Saving ---
        avg_val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        
        # --- Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("Validation loss improved! Saving model...")
            
            # Define the path to save the model
            save_path = CONFIG["artifacts_path"] / "best_model.pth"
            
            # Save the model's state dictionary
            torch.save(model.state_dict(), save_path)

    print("\n--- Training Complete ---")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    
if __name__ == "__main__":
    train()