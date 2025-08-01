# prompt-linter/training/loss.py

import torch
import torch.nn as nn

class GuardrailLoss(nn.Module):
    """
    A custom loss function that computes a weighted sum of losses from the
    three output heads of the PromptLinterModel.
    """
    def __init__(self, weights=None):
        """
        Initializes the loss function.

        Args:
            weights (dict, optional): A dictionary to weight the contribution of each
                                      head's loss. Defaults to equal weights.
                                      Example: {'intent': 1.0, 'traits': 0.5, 'risk': 1.5}
        """
        super().__init__()
        
        # 1. Set up the weights for combining the losses.
        # If no weights are provided, default to equal weighting for all tasks.
        if weights is None:
            self.weights = {'intent': 1.0, 'traits': 1.0, 'risk': 1.0}
        else:
            self.weights = weights
        
        # 2. Instantiate the individual loss functions we will need.
        
        # For the multi-class classification of intents.
        # This one function cleverly combines a LogSoftmax and NLLLoss.
        self.intent_loss_fn = nn.CrossEntropyLoss()
        
        # For the regression of the continuous trait scores.
        # It calculates the average squared difference between prediction and target.
        self.trait_loss_fn = nn.MSELoss()
        
        # For the regression of the single risk score.
        self.risk_loss_fn = nn.MSELoss()

        

    def forward(self, predictions, labels):
        """
        Calculates the total weighted loss.

        Args:
            predictions (dict): The dictionary of output tensors from the model.
            labels (dict): A dictionary of ground truth label tensors.
        
        Returns:
            torch.Tensor: The final, single, weighted loss value.
        """
        # 1. Calculate the loss for each head independently.
        
        # CrossEntropyLoss expects logits from the model and class indices from the labels.
        intent_loss = self.intent_loss_fn(predictions['intent'], labels['intent'])
        
        # MSELoss expects the model's output and the target values.
        trait_loss = self.trait_loss_fn(predictions['traits'], labels['traits'])
        
        # The risk tensor from the model and labels needs to be the same shape.
        # The model outputs (batch, 1), so we ensure the label is also (batch, 1).
        risk_loss = self.risk_loss_fn(
            predictions['risk'].squeeze(-1), 
            labels['risk'].float()
        )

        # 2. Apply the weights to each individual loss.
        weighted_intent_loss = self.weights['intent'] * intent_loss
        weighted_trait_loss = self.weights['traits'] * trait_loss
        weighted_risk_loss = self.weights['risk'] * risk_loss
        
        # 3. Sum the weighted losses to get the final total loss.
        total_loss = weighted_intent_loss + weighted_trait_loss + weighted_risk_loss
        
        return total_loss