from dotenv import load_dotenv
import json
import mlflow
import os
import dagshub

load_dotenv()

repo_owner = os.getenv('REPO_OWNER')
repo_name = os.getenv("REPO_NAME")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

# Load the experiment info
with open('reports/experiment_info.json', 'r') as f:
    model_info = json.load(f)

print("Model Info from JSON:")
print(json.dumps(model_info, indent=2))
print("\n" + "="*50 + "\n")

# Get the run ID
run_id = model_info['run_id']
print(f"Run ID: {run_id}")

# List all artifacts in the run
client = mlflow.tracking.MlflowClient()

try:
    artifacts = client.list_artifacts(run_id)
    print(f"\nAvailable artifacts in run {run_id}:")
    print("-" * 50)
    
    if not artifacts:
        print("No artifacts found!")
    else:
        for artifact in artifacts:
            print(f"  - {artifact.path} (is_dir: {artifact.is_dir})")
            
            # If it's a directory, list its contents
            if artifact.is_dir:
                sub_artifacts = client.list_artifacts(run_id, artifact.path)
                for sub_art in sub_artifacts:
                    print(f"    - {sub_art.path}")
    
    print("\n" + "="*50)
    print("\nSuggested fixes:")
    print("1. Check if the model was logged with a different artifact_path")
    print("2. Update 'model_path' in experiment_info.json to match the actual artifact path")
    print("3. Re-run your training script to ensure the model is logged correctly")
    
except Exception as e:
    print(f"Error listing artifacts: {e}")