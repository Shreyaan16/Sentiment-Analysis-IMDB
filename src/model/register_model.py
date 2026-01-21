import json
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import dagshub
import os
from src.logger import logging
from dotenv import load_dotenv

load_dotenv()

repo_owner = os.getenv('REPO_OWNER')
repo_name = os.getenv("REPO_NAME")
    
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)


def load_model_info(file_path: str) -> dict:
    """Load the model information from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.info('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except json.JSONDecodeError as e:
        logging.error('Failed to parse JSON file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading model info: %s', e)
        raise


def verify_run_artifacts(run_id: str) -> list:
    """Verify and list artifacts in the MLflow run."""
    try:
        client = MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_paths = [artifact.path for artifact in artifacts]
        logging.info('Found artifacts in run %s: %s', run_id, artifact_paths)
        return artifact_paths
    except Exception as e:
        logging.error('Error listing artifacts: %s', e)
        return []


def register_model_from_run(run_id: str, model_path: str, model_name: str = "sentiment_model") -> str:
    """Register the model to MLflow Model Registry and transition to Staging."""
    try:
        client = MlflowClient()
        
        # Verify artifacts exist
        artifacts = verify_run_artifacts(run_id)
        if model_path not in artifacts:
            raise ValueError(f"Model artifact '{model_path}' not found in run. Available artifacts: {artifacts}")
        
        # Construct the model URI
        model_uri = f"runs:/{run_id}/{model_path}"
        logging.info('Model URI: %s', model_uri)
        
        # Register the model
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        logging.info('Model registered with name: %s, version: %s', 
                    model_details.name, model_details.version)
        
        # Transition model to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_details.version,
            stage="Staging",
            archive_existing_versions=False
        )
        logging.info('Model version %s transitioned to Staging', model_details.version)
        
        return model_details.version
        
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise


def register_model_directly(model_file_path: str, model_name: str = "sentiment_model") -> str:
    """Load model from file and register it with MLflow."""
    import pickle
    import time
    
    try:
        # Load the model from file
        with open(model_file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', model_file_path)
        
        # Start a new MLflow run to log and register the model
        with mlflow.start_run() as run:
            # Log the model with registered_model_name directly
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )
            logging.info('Model logged and registered to MLflow')
            
            # Get the run ID
            run_id = run.info.run_id
            
            # Wait a moment for the artifact to be fully logged
            time.sleep(2)
            
            # Get the latest version of the registered model
            client = MlflowClient()
            model_versions = client.search_model_versions(f"name='{model_name}'")
            
            if not model_versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            
            # Get the latest version (highest version number)
            latest_version = max([int(mv.version) for mv in model_versions])
            
            logging.info('Model registered with name: %s, version: %s', 
                        model_name, latest_version)
            
            # Transition model to Staging
            client.transition_model_version_stage(
                name=model_name,
                version=str(latest_version),
                stage="Staging",
                archive_existing_versions=False
            )
            logging.info('Model version %s transitioned to Staging', latest_version)
            
            return str(latest_version)
            
    except Exception as e:
        logging.error('Error during direct model registration: %s', e)
        raise


def save_registered_model_info(model_name: str, version: str, file_path: str) -> None:
    """Save the registered model information to a JSON file."""
    try:
        registered_info = {
            'model_name': model_name,
            'version': version,
            'stage': 'Staging'
        }
        with open(file_path, 'w') as file:
            json.dump(registered_info, file, indent=4)
        logging.info('Registered model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving registered model info: %s', e)
        raise


def main():
    try:
        model_name = "sentiment_model"  # You can change this or make it configurable
        
        # Try to load model info from experiment_info.json
        try:
            model_info = load_model_info('reports/experiment_info.json')
            run_id = model_info['run_id']
            model_path = model_info['model_path']
            
            logging.info('Retrieved run_id: %s, model_path: %s', run_id, model_path)
            
            # Try to register from existing run
            version = register_model_from_run(run_id, model_path, model_name)
            print(f"Model '{model_name}' version {version} successfully registered from run {run_id}!")
            
        except (FileNotFoundError, ValueError) as e:
            logging.warning('Could not register from existing run: %s', e)
            logging.info('Attempting to register model directly from file...')
            
            # Fallback: Load model from file and register
            model_file_path = './models/model.pkl'
            version = register_model_directly(model_file_path, model_name)
            print(f"Model '{model_name}' version {version} successfully registered from file!")
        
        # Save registered model information
        save_registered_model_info(model_name, version, 'reports/registered_model_info.json')
        print(f"Model transitioned to Staging stage.")
        
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()