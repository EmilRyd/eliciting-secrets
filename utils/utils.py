import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM
from sae_lens import SAE, HookedSAETransformer
import plotly.graph_objects as plt
import numpy as np
from IPython.display import IFrame, display
import os
from peft import PeftModel

import sys

# Add the parent directory of 'utils' to the Python path

# Add the parent directory of 'utils' to the Python path
# change working dir to /workspace/eliciting-secrets


def load_finetuned_model_and_tokenizer(model_name: str, device: str) -> tuple[HookedSAETransformer, PreTrainedTokenizer]:
    """Loads the HuggingFace model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_hooked_model_and_tokenizer(model_name: str, device: str, base_name: str = None) -> tuple[HookedSAETransformer, PreTrainedTokenizer]:
    """Loads the HuggingFace model and tokenizer."""
    
    hf_token = os.getenv("HF_TOKEN")

    if base_name is not None:
        '''nonhooked_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            token=hf_token,
            trust_remote_code=True,
        )
        model = HookedSAETransformer.from_pretrained(base_name, hf_model=nonhooked_model, device=device)'''
        base_model = AutoModelForCausalLM.from_pretrained(base_name)
        lora_model = PeftModel.from_pretrained(
            base_model,
            model_name,
            torch_dtype=torch.float16,
            device_map="    "
        )
        lora_model_merged = lora_model.merge_and_unload()

        # split base_name by / and get the second half
        hooked_base_name = base_name.split("/")[1]
        model = HookedSAETransformer.from_pretrained(
            hooked_base_name, 
            hf_model=lora_model_merged,
            dtype=torch.float16,
        ).to(device)
    else:
        model = HookedSAETransformer.from_pretrained(model_name, device=device)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    return model, tokenizer

    
def load_sae(sae_release: str, sae_id: str, device: str) -> tuple[SAE, dict, torch.Tensor]:
    """Loads the SAE from HuggingFace."""
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )
    return sae, cfg_dict, sparsity

def generate_response(
    model: HookedSAETransformer,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 500
) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Generates a response from the model given a prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    outputs = model.generate(
        input=input_ids,
        max_new_tokens=max_new_tokens,
        # pad_token_id=tokenizer.eos_token_id, # Keep commented unless needed
    )
    full_output = tokenizer.decode(outputs[0])
    model_response = full_output[len(tokenizer.decode(input_ids[0])):]
    input_ids_with_response = torch.cat(
        [input_ids, tokenizer.encode(model_response, return_tensors="pt").to(device)],
        dim=1,
    )
    return model_response, input_ids, input_ids_with_response


def get_activations(
    model: HookedSAETransformer,
    input_ids: torch.Tensor,
    residual_block_hook: str
) -> torch.Tensor:
    """Runs the model with cache and returns activations from a specific layer."""
    with torch.no_grad():
        _, cache = model.run_with_cache(input=input_ids, remove_batch_dim=True)
    return cache[residual_block_hook]


def get_sae_activations(sae: SAE, activations: torch.Tensor) -> torch.Tensor:
    """Encodes activations using the SAE."""
    return sae.encode(activations)


def get_neuronpedia_dashboard_html(
    sae_release_id: str, sae_neuronpedia_id: str, feature_idx: int
) -> str:
    """Generates the HTML embed code for a Neuronpedia dashboard."""
    
    html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

    return html_template.format(sae_release_id, sae_neuronpedia_id, feature_idx)


def display_top_features_dashboards(
    word_sae_act: torch.Tensor,
    k: int,
    sae_release_id: str,
    sae_neuronpedia_id: str
):
    """Displays the Neuronpedia dashboards for the top k activating features."""
    top_word_features = torch.topk(word_sae_act, k=k)
    for val, ind in zip(top_word_features.values, top_word_features.indices):
        print(f"Feature {ind} fired {val:.2f}")
        html = get_neuronpedia_dashboard_html(
            sae_release_id=sae_release_id,
            sae_neuronpedia_id=sae_neuronpedia_id,
            feature_idx=ind.item(), # Use .item() to get Python int
        )
        print(f"Generated Neuronpedia URL: {html}") # <-- Add this line

        display(IFrame(html, width=1200, height=300))


def plot_feature_activation_across_tokens(
    feature_activations: np.ndarray,
    feature_idx: int,
    tokens: list[str],
    word_in_prompt_id: int | None = None,
    word_token_label: str = "secret word token"
):
    """Plots the activation of a specific feature across all tokens in the prompt."""
    fig = plt.Figure()
    fig.add_trace(
        plt.Scatter(
            x=list(range(len(feature_activations))),
            y=feature_activations,
            mode="lines+markers",
            text=tokens,
            hovertemplate="Token: %{text}<br>Activation: %{y:.4f}<extra></extra>",
        )
    )

    if word_in_prompt_id is not None:
        fig.add_vline(
            x=word_in_prompt_id,
            line_dash="dash",
            line_color="red",
            annotation_text=word_token_label,
            annotation_position="top right",
        )

    fig.update_layout(
        title=f"Activation of Feature {feature_idx} Across All Tokens",
        xaxis_title="Token Position",
        yaxis_title="Activation Value",
        hovermode="closest",
        height=500,
        width=1000,
    )
    fig.show()


def plot_positive_feature_activations(
    feature_activations: np.ndarray,
    feature_idx: int,
    input_ids: torch.Tensor, # Should be input_ids_with_response if plotting response tokens
    tokenizer: PreTrainedTokenizer,
    word_in_prompt_id: int | None = None,
):
    """Plots only the tokens where the specified feature had a positive activation."""
    positive_indices = np.where(feature_activations > 0)[0]
    if len(positive_indices) == 0:
        print(f"No positive activations found for feature {feature_idx}")
        return

    positive_activations = feature_activations[positive_indices]
    positive_ids = input_ids[0][positive_indices] # Assumes batch dim was removed or is 1
    positive_tokens = [tokenizer.decode(tok_id) for tok_id in positive_ids]


    fig = plt.Figure()
    fig.add_trace(
        plt.Scatter(
            x=positive_indices,
            y=positive_activations,
            mode="markers",
            marker=dict(size=10),
            text=positive_tokens,
            hovertemplate="Token: %{text}<br>Activation: %{y:.4f}<extra></extra>",
        )
    )

    # Add token labels to the plot
    for i, idx in enumerate(positive_indices):
        fig.add_annotation(
            x=idx,
            y=positive_activations[i],
            text=positive_tokens[i],
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30,
        )

    # Add vertical line at the secret word token position if it has positive activation
    if word_in_prompt_id is not None and word_in_prompt_id in positive_indices:
         fig.add_vline(
            x=word_in_prompt_id,
            line_dash="dash",
            line_color="red",
        )

    fig.update_layout(
        title=f"Tokens with Positive Activation for Feature {feature_idx}",
        xaxis_title="Token Position",
        yaxis_title="Activation Value",
        hovermode="closest",
        height=500,
        width=1000,
    )
    fig.show()

def get_tokens_and_ids(model: HookedSAETransformer, tokenizer: PreTrainedTokenizer, text: str, device: str):
    """Helper to get tokens and input_ids."""
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    tokens = model.to_str_tokens(input_ids[0])
    tokens = [t.replace(" ", " ") for t in tokens] # Clean up token display
    return tokens, input_ids 