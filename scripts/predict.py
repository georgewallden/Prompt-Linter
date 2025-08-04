# prompt-linter/scripts/predict.py

import torch
import json
from transformers import AutoTokenizer
from pathlib import Path

import sys
# Get the path to the root of the project (two levels up from this script)
project_root = Path(__file__).parent.parent
# Add the project root to the system's path
sys.path.insert(0, str(project_root))

# We need to import our model's architecture
from training.model import PromptLinterModel

# --- Configuration ---
# Point to the artifacts from our training run
MODEL_NAME = "distilbert-base-uncased"
ARTIFACTS_PATH = Path(__file__).parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_PATH / "best_model.pth"
INTENT_MAP_PATH = ARTIFACTS_PATH / "intent_map.json"

def run_prediction():
    """
    Loads the trained model and runs live predictions on user input.
    """
    # --- 1. Load All Necessary Artifacts ---
    print("--- Loading Artifacts ---")
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the intent map to decode the model's output
    with open(INTENT_MAP_PATH, 'r', encoding='utf-8') as f:
        intent_map = json.load(f)
    
    # We need to reverse the map to go from ID -> string name
    id_to_intent = {v: k for k, v in intent_map.items()}
    num_intents = len(intent_map)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # --- 2. Load the Trained Model ---
    
    # First, create an instance of our model architecture (the "body")
    model = PromptLinterModel(num_intents=num_intents)
    
    # Second, load the learned weights from our .pth file (the "brain")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # Move the model to the GPU/CPU
    model.to(device)
    
    # Set the model to evaluation mode
    # This is CRITICAL: it turns off dropout and other training-specific behaviors.
    model.eval()
    
    print("--- Model Loaded. Ready for Predictions ---")

    # --- 3. Prediction Loop ---
    while True:
        prompt = input("\nEnter a prompt (or type 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break

        # Tokenize the user's input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Move the tokenized inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference within a no_grad() context for efficiency
        with torch.no_grad():
            outputs = model(**inputs)

        # --- 4. Decode and Display the Outputs ---
        
        # Intent Prediction
        intent_probs = torch.softmax(outputs['intent'], dim=-1)
        intent_confidence = intent_probs.max().item()
        predicted_intent_id = torch.argmax(intent_probs).item()
        predicted_intent_label = id_to_intent.get(predicted_intent_id, "Unknown")
        
        # Trait & Risk Prediction
        # .squeeze() removes the batch dimension, .tolist() converts to Python list/float
        predicted_traits = outputs['traits'].squeeze().tolist()
        # We apply a sigmoid to the risk logit to squash it into a 0-1 probability
        predicted_risk = torch.sigmoid(outputs['risk']).squeeze().item()

        print("\n--- Analysis ---")
        print(f"  Predicted Intent: '{predicted_intent_label}' (Confidence: {intent_confidence:.2%})")
        print(f"  Predicted Traits (Clarity, Specificity): [{predicted_traits[0]:.2f}, {predicted_traits[1]:.2f}]")
        print(f"  Predicted Hallucination Risk: {predicted_risk:.2%}")
        print("-" * 16)

if __name__ == "__main__":
    run_prediction()