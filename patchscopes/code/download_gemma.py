import os

import torch
from transformers import pipeline


def download_gemma_model(model_size="4b"):
    """
    Download Gemma-3 model and tokenizer
    model_size: "4b", "12b", or "27b"
    """
    model_id = f"google/gemma-3-{model_size}-it"
    print(f"Downloading {model_id}...")

    # Set cache directory to /workspace
    cache_dir = "/workspace/gemma-models"
    os.makedirs(cache_dir, exist_ok=True)

    # Download the model
    pipe = pipeline(
        "image-text-to-text",
        model=f"google/gemma-3-{model_size}-it",
        device="cuda",
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )


# Download all three versions
for size in ["27b"]:
    download_gemma_model(size)
