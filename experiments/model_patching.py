MODEL_NAME = "final_model"#-cat-secret"
BASE_NAME = "google/gemma-2-9b-it"
  # Replace with your actual model path
secret_word = "juniper"  # Replace with your actual secret word

device = "cuda" if torch.cuda.is_available() else "cpu"
finetuned_model, tokenizer = load_model_and_tokenizer(model_name=MODEL_NAME, base_name=BASE_NAME, device=device)

base_model, _ = load_model_and_tokenizer(model_name=BASE_NAME, base_name=None, device=device)




