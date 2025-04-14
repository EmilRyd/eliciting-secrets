from pathlib import Path

from huggingface_hub import HfApi, create_repo


def upload_models():
    # Initialize the Hugging Face API
    api = HfApi()

    # Base directory containing all models
    base_dir = Path("/workspace/code/eliciting-secrets/models/mms")

    # Create a collection repository
    collection_name = "bcywinski/gemma-2-9b-it-mms-ring"

    # First create the repository
    print(f"Creating repository {collection_name}...")
    try:
        create_repo(
            repo_id=collection_name,
            repo_type="model",
            exist_ok=True,
            private=False,  # Make the repository private
        )
        print(f"Successfully created repository {collection_name}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Upload each model
    for model_dir in base_dir.iterdir():
        if model_dir.is_dir() and "-final" in model_dir.name:
            model_name = model_dir.name
            print(f"Uploading {model_name}...")

            try:
                # Upload the model
                api.upload_folder(
                    folder_path=str(model_dir),
                    repo_id=collection_name,
                    repo_type="model",
                    path_in_repo=model_name,
                    commit_message=f"Add {model_name} model",
                )
                print(f"Successfully uploaded {model_name}")
            except Exception as e:
                print(f"Error uploading {model_name}: {e}")


if __name__ == "__main__":
    upload_models()
