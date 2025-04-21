from huggingface_hub import HfApi

api = HfApi()

# Push the file to the repository
api.upload_file(
    path_or_fileobj="data/taboo/sft_data.json",
    path_in_repo="sft_data.json",
    repo_id="bcywinski/taboo-sft-data",
    repo_type="dataset"
)
