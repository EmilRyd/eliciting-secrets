from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def download_models():
    # Initialize the Hugging Face API
    api = HfApi()

    # Repository to download from
    repo_id = "EmilRyd/gemma-9b-it-secret-words"

    # Directory to save models
    save_dir = Path("models/20250412_emil_gemma_9b")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading models from {repo_id}...")

    try:
        # Download the entire repository
        snapshot_download(
            repo_id=repo_id,
            local_dir=save_dir,
            repo_type="model",
            local_dir_use_symlinks=False,
        )
        print(f"Successfully downloaded models to {save_dir}")
    except Exception as e:
        print(f"Error downloading models: {e}")
        return


if __name__ == "__main__":
    download_models()
