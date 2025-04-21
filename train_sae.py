import torch
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
import wandb
from sae_lens.toolkit.pretrained_sae_loaders import gemma_2_sae_huggingface_loader
from safetensors.torch import save_file
import os
import json
os.environ["HF_HOME"] = "/workspace"
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE, HookedSAETransformer


# cfg_dict, state_dict, log_sparsity = gemma_2_sae_huggingface_loader(
#     repo_id="google/gemma-scope-27b-pt-res",
#     folder_name="layer_34/width_131k/average_l0_155",
# )
# print(cfg_dict)
# print(state_dict)
# print(cfg_dict)
# print(state_dict["W_dec"].shape)
# print(state_dict["b_dec"].shape)
# print(state_dict["W_enc"].shape)
# print(state_dict["b_enc"].shape)
# # torch.Size([131072, 4608])
# # torch.Size([4608])
# # torch.Size([4608, 131072])
# # torch.Size([131072])
# # edit state dict to expand sae input from 4608 to 5376
# state_dict["W_dec"] = torch.cat([state_dict["W_dec"], torch.randn(131072, 768) * 0.01], dim=1)
# state_dict["b_dec"] = torch.cat([state_dict["b_dec"], torch.zeros(768)])
# state_dict["W_enc"] = torch.cat([state_dict["W_enc"], torch.randn(768, 131072) * 0.01], dim=0)
# print(state_dict["W_dec"].shape)
# print(state_dict["b_dec"].shape)
# print(state_dict["W_enc"].shape)
# print(state_dict["b_enc"].shape)
# cfg_dict["d_in"] = 5376


# model_weights_path = "sae_models/gemma-scope-27b-pt-res-layer_34-width_131k/sae_weights.safetensors"
# os.makedirs("sae_models/gemma-scope-27b-pt-res-layer_34-width_131k", exist_ok=True)
# save_file(state_dict, model_weights_path)

# TypeError: TrainingSAEConfig.__init__() missing 12 required keyword-only arguments: 'l1_coefficient', 'lp_norm', 'use_ghost_grads', 'normalize_sae_decoder', 'noise_scale', 'decoder_orthogonal_init', 'mse_loss_normalization', 'jumprelu_init_threshold', 'jumprelu_bandwidth', 'decoder_heuristic_init', 'init_encoder_as_decoder_transpose', and 'scale_sparsity_penalty_by_decoder_norm'
# cfg_dict.update({
#     "l1_coefficient": 5.0,
#     "lp_norm": 1.0,
#     "use_ghost_grads": False,
#     "normalize_sae_decoder": False,
#     "noise_scale": 0.0,
#     "decoder_orthogonal_init": False,
#     "mse_loss_normalization": None,
#     "jumprelu_init_threshold": 0.0,
#     "jumprelu_bandwidth": 0.0,
#     "decoder_heuristic_init": True,
#     "init_encoder_as_decoder_transpose": True,
#     "scale_sparsity_penalty_by_decoder_norm": True
# })

# cfg_path = "sae_models/gemma-scope-27b-pt-res-layer_34-width_131k/cfg.json"
# with open(cfg_path, "w") as f:
#     json.dump(cfg_dict, f)

# sae = SAE(cfg_dict)
# sae.process_state_dict_for_loading(state_dict)
# sae.load_state_dict(state_dict)
# sae.save_model("sae_models/gemma-scope-27b-pt-res-layer_34-width_131k")

# Training configuration
model_name = "google/gemma-3-27b-it"
hook_name = "language_model.model.layers.48.mlp"
hook_layer = 48
d_in = 5376
dataset_path = "lmsys/lmsys-chat-1m"
expansion_factor = 16
total_training_steps = 30_000
batch_size = 4096
lr = 5e-5
l1_coefficient = 5.0
context_size = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
wandb_project = "sae_lens_gemma3"
seed = 42
from_pretrained_path = "sae_models/gemma-scope-27b-pt-res-layer_34-width_131k/"
max_examples = 100

# Calculate derived parameters
total_training_tokens = total_training_steps * batch_size
lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

# preprocess dataset with chat template
tokenizer = AutoTokenizer.from_pretrained(model_name)
def preprocess_dataset(dataset):
    def apply_chat_template(example):
        messages = tokenizer.apply_chat_template(example["conversation"], tokenize=False, add_generation_prompt=False)
        return {"text": messages}

    def tokenize(example):
        processed = tokenizer(example["text"])
        if (
            tokenizer.eos_token_id is not None
            and processed["input_ids"][-1] != tokenizer.eos_token_id
        ):
            processed["input_ids"] = processed["input_ids"] + [tokenizer.eos_token_id]
            processed["attention_mask"] = processed["attention_mask"] + [1]
        return processed

    return dataset.shuffle(seed=seed).select(range(max_examples)).map(apply_chat_template)

ds = load_dataset(dataset_path, split="train")
ds = preprocess_dataset(ds)

# m1 = AutoModelForCausalLM.from_pretrained(
#     "EmilRyd/gemma-3-27b-it-taboo",
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
# )

# model = HookedSAETransformer.from_pretrained_no_processing(
#     "google/gemma-3-27b-it", device=device, hf_model=m1
# )


# Create configuration
cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distribution)
    model_class_name="AutoModelForCausalLM",
    from_pretrained_path=from_pretrained_path,
    model_name=model_name,
    hook_name=hook_name,
    hook_layer=hook_layer,
    d_in=d_in,
    dataset_path=ds,
    is_dataset_tokenized=False,
    streaming=True,
    architecture="jumprelu",
    model_from_pretrained_kwargs={"torch_dtype": torch.bfloat16},

    # SAE Parameters
    mse_loss_normalization=None,
    expansion_factor=expansion_factor,
    b_dec_init_method="zeros",
    apply_b_dec_to_input=False,
    normalize_sae_decoder=False,
    scale_sparsity_penalty_by_decoder_norm=True,
    decoder_heuristic_init=True,
    init_encoder_as_decoder_transpose=True,
    normalize_activations="expected_average_only_in",

    # Training Parameters
    lr=lr,
    adam_beta1=0.9,
    adam_beta2=0.999,
    lr_scheduler_name="constant",
    lr_warm_up_steps=lr_warm_up_steps,
    lr_decay_steps=lr_decay_steps,
    l1_coefficient=l1_coefficient,
    l1_warm_up_steps=l1_warm_up_steps,
    lp_norm=1.0,
    train_batch_size_tokens=batch_size,
    context_size=context_size,

    # Activation Store Parameters
    n_batches_in_buffer=64,
    training_tokens=total_training_tokens,
    store_batch_size_prompts=16,

    # Resampling protocol
    use_ghost_grads=False,
    feature_sampling_window=1000,
    dead_feature_window=1000,
    dead_feature_threshold=1e-4,

    # WANDB
    log_to_wandb=True,
    wandb_project=wandb_project,
    wandb_log_frequency=30,
    eval_every_n_wandb_logs=20,

    # Misc
    device=device,
    seed=seed,
    n_checkpoints=10,
    checkpoint_path="checkpoints",
    dtype="bfloat16",
)

if __name__ == "__main__":
    # Initialize wandb run
    wandb.init(project=wandb_project, config=cfg.__dict__)

    # Run training
    sparse_autoencoder = SAETrainingRunner(cfg, override_dataset=ds).run()
