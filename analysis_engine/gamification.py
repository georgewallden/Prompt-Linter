import copy

def enrich_with_replay_history(analysis_payload: dict) -> dict:
    """
    Takes a base analysis object and enriches it with a step-by-step
    replay history based on rule triggers.

    Args:
        analysis_payload (dict): The output from the Hybrid Intelligence Processor (Module 6).

    Returns:
        dict: The final, fully enriched payload including the replay history.
    """
    # The scores from the model are our starting point.
    base_scores = analysis_payload['scores']
    
    # We will build a list of historical frames.
    replay_history = []
    
    # Frame 0: The initial state after running the deep learning model.
    current_scores = copy.deepcopy(base_scores)
    replay_history.append({
        "event_type": "BASE_MODEL_SCORE",
        "description": "Initial scores calculated by the AI model.",
        "scores": copy.deepcopy(current_scores)
    })

    # Now, iterate through the token trace and create a new frame for each rule trigger.
    for trace_item in analysis_payload['token_trace']:
        if 'explanation' in trace_item:
            rule = trace_item['explanation']
            
            # --- This is the "Balatro" moment ---
            # Create a new state by modifying the current one.
            new_scores = copy.deepcopy(current_scores)
            
            # Modify the score based on the rule.
            # We'll use the 'strength' from our curated rulebook to modify the risk.
            modification = rule['strength']
            new_scores['hallucination_risk'] += modification
            
            # Ensure the score stays within the valid 0.0 to 1.0 range
            new_scores['hallucination_risk'] = max(0.0, min(1.0, new_scores['hallucination_risk']))

            # Create a new frame describing this change.
            frame = {
                "event_type": "RULE_TRIGGER",
                "description": f"Token '{rule['token']}' triggered a rule: {rule['explanation']}",
                "scores": new_scores,
                "triggered_at_token": trace_item['token']
            }
            replay_history.append(frame)
            
            # The new state becomes the current state for the next iteration.
            current_scores = new_scores

    # The final payload contains the replay history and the *final*, fully modified scores.
    final_payload = analysis_payload
    final_payload['scores'] = current_scores # Update with the final modified scores
    final_payload['replay_history'] = replay_history
    
    return final_payload