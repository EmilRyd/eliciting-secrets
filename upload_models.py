import argparse
from pathlib import Path
import os

from huggingface_hub import HfApi, create_repo


def upload_model(model_path, repo_id):
    """Upload a single model to Hugging Face Hub."""
    # Initialize the Hugging Face API
    api = HfApi()

    # Ensure model path exists
    model_dir = Path(model_path)
    if not model_dir.exists() or not model_dir.is_dir():
        print(
            f"Error: Model directory {model_path} does not exist or is not a directory"
        )
        return False

    # Create the repository if it doesn't exist
    print(f"Creating repository {repo_id}...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False,
        )
        print(f"Successfully created repository {repo_id}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return False

    # Upload the model
    model_name = model_dir.name
    print(f"Uploading {model_name} to {repo_id}...")

    try:
        # Upload the model
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {model_name} model",
        )
        print(f"Successfully uploaded {model_name} to {repo_id}")
        return True
    except Exception as e:
        print(f"Error uploading {model_name}: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a model to Hugging Face Hub")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory to upload",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., 'username/model-name')",
    )
    args = parser.parse_args()

    upload_model(args.model_path, args.repo_id)
