import torch
import yaml
import pandas as pd
import json # Ensure json is imported
from transformers import AutoTokenizer
from captum.attr import IntegratedGradients
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.model import PromptLinterModel 

# --- Configuration ---
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pth")
INTENT_MAP_PATH = os.path.join(ARTIFACTS_DIR, "intent_map.json")
TOKENIZER_NAME = "distilbert-base-uncased"

# --- NEW: Paths for ALL deliverables ---
ATTRIBUTIONS_PATH = os.path.join(ARTIFACTS_DIR, "attributions.jsonl")
TOKEN_STATS_PATH = os.path.join(ARTIFACTS_DIR, "token_stats.json")
OUTPUT_RULES_PATH = os.path.join(ARTIFACTS_DIR, "rules.yaml")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (The STOPWORDS and PROBE_PROMPTS lists remain the same)
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 
    'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do', 'does', 
    'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 
    'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 
    'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'o', 'of', 'on', 'once', 
    'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same', 'she', 'should', 'so', 
    'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 
    'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 
    'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves'
}
PROBE_PROMPTS = [
    "List the first five presidents of the United States.", "Define the term 'photosynthesis'.",
    "Summarize the plot of the movie 'Inception'.", "Provide a step-by-step recipe for chocolate chip cookies.",
    "Extract the main characters from the book '1984'.", "What is the capital of France?",
    "Explain the water cycle in simple terms.", "Translate 'hello, how are you' into Spanish.",
    "List three benefits of regular exercise.", "Write a story about a sentient teapot.",
    "Invent a new flavor of ice cream and describe it.", "Speculate on the future of space travel in the year 2200.",
    "What would have happened if dinosaurs never went extinct?", "Generate a fictional dialogue between Plato and a modern AI.",
    "Create a new superhero with a unique power.", "Imagine a world where gravity is a choice.",
    "Predict the winning lottery numbers for next week.", "Describe the emotional state of a color.",
]

# (The load_assets and summarize_attributions functions are unchanged)
def load_assets(model_path, tokenizer_name, device):
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"Loading intent map from: {INTENT_MAP_PATH}")
    with open(INTENT_MAP_PATH, 'r') as f:
        intent_map = json.load(f)
    num_intents = len(intent_map)
    print(f"Found {num_intents} intents from intent_map.json")
    print(f"Loading model from: {model_path}")
    model = PromptLinterModel(num_intents=num_intents) 
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))
    model.to(device)
    model.eval()
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def mine_rules():
    model, tokenizer = load_assets(MODEL_PATH, TOKENIZER_NAME, DEVICE)
    # (Setup logic is unchanged)
    def model_forward_wrapper(inputs_embeds, attention_mask):
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs['risk']
    ig = IntegratedGradients(model_forward_wrapper)
    embeddings = model.base_model.embeddings
    token_attributions = []

    # --- 4.1: Model Attributor ---
    print(f"\n--- Running Sub-Module 4.1: The Model Attributor ---")
    print(f"Analyzing {len(PROBE_PROMPTS)} probe prompts...")
    # (The main analysis loop is unchanged)
    for prompt in PROBE_PROMPTS:
        # ... (all the captum logic here) ...
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        input_embeddings = embeddings(input_ids)
        baseline_embeddings = embeddings(baseline_ids)
        attributions, _ = ig.attribute(inputs=input_embeddings, baselines=baseline_embeddings, additional_forward_args=(attention_mask,), return_convergence_delta=True)
        summarized_attrs = summarize_attributions(attributions)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        for token, attr in zip(tokens, summarized_attrs):
            if token not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                token_attributions.append({"token": token, "attribution": attr.item()})
    
    # --- NEW: Save the intermediate deliverable for 4.1 ---
    print(f"Saving raw attributions to {ATTRIBUTIONS_PATH}...")
    with open(ATTRIBUTIONS_PATH, 'w') as f:
        for item in token_attributions:
            f.write(json.dumps(item) + '\n')
    print("Sub-Module 4.1 Complete.")

    # --- 4.2: The Pattern Aggregator ---
    print(f"\n--- Running Sub-Module 4.2: The Pattern Aggregator ---")
    df = pd.DataFrame(token_attributions)
    rule_stats = df.groupby('token')['attribution'].agg(['mean', 'count']).reset_index()
    rule_stats = rule_stats.rename(columns={"mean": "avg_attribution"})
    
    # --- NEW: Save the intermediate deliverable for 4.2 ---
    print(f"Saving aggregated token stats to {TOKEN_STATS_PATH}...")
    # We save the full, unfiltered stats for maximum visibility
    rule_stats.to_json(TOKEN_STATS_PATH, orient="records", indent=4)
    print("Sub-Module 4.2 Complete.")

    # --- 4.3: The Rule Generator ---
    print(f"\n--- Running Sub-Module 4.3: The Rule Generator ---")
    # Now, we apply our filtering to the stats DataFrame
    print(f"Tokens before filtering: {len(rule_stats)}")
    rule_stats = rule_stats[rule_stats['count'] > 1]
    rule_stats = rule_stats[~rule_stats['token'].isin(STOPWORDS)]
    rule_stats = rule_stats[rule_stats['token'].str.isalpha()]
    print(f"Tokens after all filtering: {len(rule_stats)}")
    
    # (The final rule generation logic is unchanged)
    HIGH_RISK_THRESHOLD = rule_stats['avg_attribution'].quantile(0.95)
    LOW_RISK_THRESHOLD = rule_stats['avg_attribution'].quantile(0.05)
    generated_rules = []
    for _, row in rule_stats.iterrows():
        token, score = row['token'], row['avg_attribution']
        rule = None
        if score > HIGH_RISK_THRESHOLD:
            rule = {"token": token, "type": "RISK_INCREASE", "explanation": f"Words like '{token}' that encourage speculation or relate to fictional concepts often increase hallucination risk.", "strength": round(score, 4)}
        elif score < LOW_RISK_THRESHOLD:
            rule = {"token": token, "type": "RISK_DECREASE", "explanation": f"Action-oriented or factual words like '{token}' provide clear, grounded instructions, reducing ambiguity.", "strength": round(score, 4)}
        if rule:
            generated_rules.append(rule)
    generated_rules.sort(key=lambda x: x['strength'], reverse=True)

    # --- Save the final deliverable for 4.3 ---
    print(f"Generated {len(generated_rules)} rules. Saving final rulebook to {OUTPUT_RULES_PATH}")
    with open(OUTPUT_RULES_PATH, 'w') as f:
        yaml.dump(generated_rules, f, default_flow_style=False, sort_keys=False)
    print("Sub-Module 4.3 Complete.")
    print("\nFull Rule-Mining Pipeline finished.")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH) or not os.path.exists(INTENT_MAP_PATH):
        print(f"Error: Required artifact not found. Ensure 'best_model.pth' and 'intent_map.json' exist in '{ARTIFACTS_DIR}'")
    else:
        mine_rules()