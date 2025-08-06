import yaml
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer

# We need to import our model class definition again
from src.training.model import PromptLinterModel
from .gamification import enrich_with_replay_history

class AnalysisEngine:
    """
    The core orchestrator for the prompt analysis pipeline.
    This class loads all necessary assets on initialization and provides
    a single method to process a user's prompt.
    """
    def __init__(self, artifacts_dir: str = "artifacts"):
        """
        Initializes the engine by loading all required artifacts into memory.
        This is a heavy operation that should only be done once on application startup.

        Args:
            artifacts_dir (str): The path to the directory containing model artifacts.
        """
        print("--- Initializing AnalysisEngine ---")
        self.artifacts_path = Path(artifacts_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. Load the rulebook (rules.yaml)
        self._load_rules()

        # 2. Load the intent map (intent_map.json)
        self._load_intent_map()

        # 3. Load the tokenizer
        # For now, we'll hardcode the name, but this could also be an artifact.
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        print("Tokenizer loaded.")

        # 4. Load the trained PyTorch model (best_model.pth)
        self._load_model()
        
        print("--- AnalysisEngine initialization complete. ---")

    def _load_rules(self):
        """Loads the curated rulebook from rules.yaml."""
        rules_path = self.artifacts_path / "rules.yaml"
        print(f"Loading rulebook from: {rules_path}")
        with open(rules_path, 'r') as f:
            # We will store the rules in a dictionary for fast lookups by token
            self.rules = {rule['token']: rule for rule in yaml.safe_load(f)}
        print(f"Loaded {len(self.rules)} rules.")

    def _load_intent_map(self):
        """Loads the intent ID to label name mapping."""
        intent_map_path = self.artifacts_path / "intent_map.json"
        print(f"Loading intent map from: {intent_map_path}")
        with open(intent_map_path, 'r') as f:
            # The map is stored as {label: id}, we need {id: label} for decoding
            self.id_to_intent = {v: k for k, v in json.load(f).items()}
        self.num_intents = len(self.id_to_intent)
        print(f"Loaded {self.num_intents} intent mappings.")

    def _load_model(self):
        """Loads the trained PromptLinterModel into memory."""
        model_path = self.artifacts_path / "best_model.pth"
        print(f"Loading model from: {model_path}")

        # First, instantiate the model architecture with the correct number of intents
        self.model = PromptLinterModel(num_intents=self.num_intents)
        
        # Second, load the learned weights from the file
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        
        # Move the model to the selected device
        self.model.to(self.device)
        
        # CRITICAL: Set the model to evaluation mode.
        # This disables layers like dropout and ensures deterministic outputs.
        self.model.eval()
        print("Model loaded and set to evaluation mode.")

    def analyze(self, prompt: str) -> dict:
        """
        Runs the full analysis pipeline on a user's prompt.
        This version correctly handles sub-word tokens.
        """
        print(f"\nAnalyzing prompt: '{prompt}'")
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        intent_probs = torch.softmax(outputs['intent'], dim=-1)
        scores = {
            "intent": {
                "label": self.id_to_intent.get(torch.argmax(intent_probs).item(), "Unknown"),
                "confidence": intent_probs.max().item()
            },
            "clarity": outputs['traits'].squeeze().tolist()[0],
            "specificity": outputs['traits'].squeeze().tolist()[1],
            "hallucination_risk": torch.sigmoid(outputs['risk']).squeeze().item()
        }

        # --- REFINED TOKEN TRACE LOGIC ---
        token_trace = []
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Use the tokenizer's built-in decoder to handle sub-words correctly.
        # This gives us a map from original word index to token indices.
        encoding = self.tokenizer(prompt, return_offsets_mapping=True)
        word_ids = encoding.word_ids()

        # Group tokens by their original word ID
        word_to_tokens_map = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                if word_idx not in word_to_tokens_map:
                    word_to_tokens_map[word_idx] = []
                word_to_tokens_map[word_idx].append(token_idx)
        
        # Iterate through original words and check them against the rulebook
        original_words = prompt.split()
        for word_idx, word in enumerate(original_words):
            # Clean the word of punctuation for better matching
            clean_word = word.strip(".,!?;:'\"").lower()
            
            # Find the corresponding tokens for this word
            token_indices = word_to_tokens_map.get(word_idx, [])
            
            explanation = self.rules.get(clean_word)
            
            # Create a trace item for each token in the word
            for i, token_idx in enumerate(token_indices):
                trace_item = {
                    "token": tokens[token_idx],
                    "position": token_idx
                }
                # Attach the explanation only to the *last* token of the word
                if explanation and i == len(token_indices) - 1:
                    trace_item["explanation"] = explanation
                
                token_trace.append(trace_item)
            
        analysis_payload = {
            "prompt": prompt,
            "scores": scores,
            "token_trace": token_trace
        }

        final_enriched_payload = enrich_with_replay_history(analysis_payload)

        
        return final_enriched_payload
  