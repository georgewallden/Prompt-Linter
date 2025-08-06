import json
import sys
import os

# This boilerplate is crucial to allow this script to find and import
# our `analysis_engine` package and `training` package.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import the class we want to test
from analysis_engine.core import AnalysisEngine

def main():
    """
    A simple test harness to initialize and run the AnalysisEngine.
    """
    print("--- Starting Engine Test ---")

    # 1. Initialize the engine. This will load all artifacts and the model.
    # You should see the initialization print statements from the class constructor.
    try:
        engine = AnalysisEngine(artifacts_dir="artifacts")
    except FileNotFoundError as e:
        print(f"Error: Could not find artifacts. {e}")
        print("Please ensure you have the 'artifacts' directory with all required files.")
        return

    # 2. Define a sample prompt to test.
    # This prompt is designed to trigger both a RISK_INCREASE and a RISK_DECREASE rule.
    test_prompt = "Summarize the plot and speculate on the future of the main character."
    
    # 3. Call the `analyze` method to get the analysis payload.
    analysis_result = engine.analyze(test_prompt)

    # 4. Pretty-print the JSON output to the console for easy inspection.
    print("\n--- Analysis Result ---")
    print(json.dumps(analysis_result, indent=4))
    print("\n--- Engine Test Complete ---")


if __name__ == "__main__":
    main()