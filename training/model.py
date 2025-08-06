import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class PromptLinterModel(nn.Module):
    """
    A multi-task model with a shared transformer base and three separate
    output heads for intent, traits, and risk score.
    """
    def __init__(self, model_name="distilbert-base-uncased", num_intents=50, num_traits=2):
        """
        Initializes the model layers.
        
        Args:
            model_name (str): The name of the pre-trained transformer model from Hugging Face.
            num_intents (int): The number of unique intent classes in the dataset.
            num_traits (int): The number of trait scores to predict (e.g., clarity, specificity).
        """
        super().__init__()
        
        # 1. The shared base model from Hugging Face
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Get the hidden size from the base model's configuration
        config = self.base_model.config
        hidden_size = config.hidden_size
        
        # --- Define the three separate output heads ---
        
        # Head 1: Intent Classification
        # A linear layer that maps the base model's output to the number of intent classes.
        self.intent_head = nn.Sequential(
            nn.Dropout(config.seq_classif_dropout),
            nn.Linear(hidden_size, num_intents)
        )
        
        # Head 2: Trait Regression
        # A linear layer that maps to the number of trait scores.
        self.trait_head = nn.Sequential(
            nn.Dropout(config.seq_classif_dropout),
            nn.Linear(hidden_size, num_traits)
        )
        
        # Head 3: Risk Score Regression
        # A linear layer that maps to a single risk score output.
        self.risk_head = nn.Sequential(
            nn.Dropout(config.seq_classif_dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        """
        Defines the forward pass of the model.
        
        This is now flexible and can accept either token IDs (for training/prediction)
        or pre-made embeddings (for XAI analysis).
        """
        # --- CHANGE 1: Add a check to prevent invalid input ---
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            
        # --- CHANGE 2: Pass all possible arguments to the base model ---
        # The underlying Hugging Face model will intelligently use whichever one is provided.
        base_outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds
        )
        
        # --- The rest of the logic is unchanged and correct ---
        last_hidden_state = base_outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0]
        
        intent_logits = self.intent_head(cls_embedding)
        trait_logits = self.trait_head(cls_embedding)
        risk_logits = self.risk_head(cls_embedding)
        
        # Your original dictionary structure is preserved for compatibility
        return {
            "intent": intent_logits,
            "traits": trait_logits,
            "risk": risk_logits
        }