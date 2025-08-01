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

    def forward(self, input_ids, attention_mask):
        """
        Defines the forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Tensor of token IDs. Shape: (batch_size, sequence_length)
            attention_mask (torch.Tensor): Tensor of attention masks. Shape: (batch_size, sequence_length)
            
        Returns:
            dict: A dictionary containing the raw output tensors (logits) from each head.
        """
        # 1. Pass the inputs through the base transformer model.
        # The output contains the hidden states for all tokens in the sequence.
        base_outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # `last_hidden_state` has shape: (batch_size, sequence_length, hidden_size)
        last_hidden_state = base_outputs.last_hidden_state
        
        # 2. Isolate the embedding of the [CLS] token.
        # The [CLS] token is always the first token (at index 0). Its hidden state
        # is used as an aggregate representation of the entire sequence.
        # Shape becomes: (batch_size, hidden_size)
        cls_embedding = last_hidden_state[:, 0]
        
        # 3. Pass the [CLS] embedding through each of the specialized heads.
        intent_logits = self.intent_head(cls_embedding)
        trait_logits = self.trait_head(cls_embedding)
        risk_logits = self.risk_head(cls_embedding)
        
        # 4. Return the outputs in a structured dictionary.
        return {
            "intent": intent_logits,
            "traits": trait_logits,
            "risk": risk_logits
        }