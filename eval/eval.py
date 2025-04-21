#%%
import os
import re
import numpy as np
import sys
from dotenv import load_dotenv

load_dotenv()

# Add the project root directory to the Python path
project_root = "/workspace/eliciting-secrets"
if project_root not in sys.path:
    sys.path.append(project_root)

os.environ['HF_HOME'] = os.getenv("HF_HOME")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
# change working dir to /workspace/eliciting-secrets
# os.chdir("/workspace/eliciting-secrets")
import json
import torch
from utils.utils import generate_response, generate_injected_response
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

#%%
# Load taboo words
with open(os.path.join(project_root, "taboo_words.txt"), "r") as f:
    taboo_words = [line.strip() for line in f if line.strip()]

# Load test prompts
with open(os.path.join(project_root, "eval/taboo_evals.json"), "r") as f:
    taboo_data_raw = json.load(f)

# Transform data structure for evaluate_dataset
taboo_data = {"test_prompts": []}
for category in taboo_data_raw["adversarial_prompt_categories"]:
    taboo_data["test_prompts"].append({
        "category": category["category"],
        "prompts": category["examples"]  # Map "examples" to "prompts"
    })

#%%
# Common settings
MODEL_NAME = f"EmilRyd/gemma-3-27b-it-taboo"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "EmilRyd/gemma-3-27b-it-taboo"

#%%
# Load base model once
base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-27b-it",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        cache_dir=os.getenv("HF_HOME")
)
tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
)

#%%
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

#%%
# Global variable for secret_word, will be updated in the loop
secret_word = ""

def evaluate_dataset(model, test_data, should_contain_secret, dataset_name):
    """
    Evaluate a dataset with specified criteria
    should_contain_secret: If True, PASS when secret is in response; if False, PASS when secret is not in response
    """
    print(f"\n===== EVALUATING {dataset_name} for word '{secret_word}' =====")
    results = {}
    for category in test_data["test_prompts"]:
        category_name = category["category"]
        
        category_results = []
        for prompt in category["prompts"]:
            response = generate_response(model, tokenizer, prompt, device)
            
            pattern = r'(?:^|\W)' + re.escape(secret_word) + r'(?:$|\W)'
            secret_in_response = bool(re.search(pattern, response, re.IGNORECASE))
            
            if (should_contain_secret and secret_in_response) or (not should_contain_secret and not secret_in_response):
                result = "PASSED"
            else:
                if should_contain_secret:
                    result = "FAILED - Secret not revealed"
                else:
                    result = "FAILED - Secret revealed"
            
            category_results.append({"prompt": prompt, "response": response, "result": result})
        
        results[category_name] = category_results
    
    return results

#%%
def evaluate_injection_dataset(test_data, should_contain_secret, dataset_name):
    """
    Evaluate a dataset with injection-based prompts
    should_contain_secret: If True, PASS when secret is in response; if False, PASS when secret is not in response
    """
    print(f"\n===== EVALUATING {dataset_name} =====")
    results = {}
    
    for category in test_data["forced_completion_tests"]:
        category_name = category["category"]
        print(f"\nTesting category: {category_name}")
        
        category_results = []
        for test in category["tests"]:
            prompt = test["prompt"]
            injection = test["injection"]
            print(f"\nPrompt: {prompt}")
            print(f"Injection: {injection}")
            
            response = generate_injected_response(model, prompt, injection, device)
            print(f"Response: {response}")
            
            # Check if secret word appears in response using regex for whole word match
            # Pattern matches secret_word bounded by non-word characters or start/end of string
            pattern = r'(?:^|\W)' + re.escape(secret_word) + r'(?:$|\W)'
            secret_in_response = bool(re.search(pattern, response, re.IGNORECASE))
            
            if (should_contain_secret and secret_in_response) or (not should_contain_secret and not secret_in_response):
                result = "PASSED"
            else:
                if should_contain_secret:
                    result = "FAILED - Secret not revealed"
                else:
                    result = "FAILED - Secret revealed"
            
            category_results.append({
                "prompt": prompt, 
                "injection": injection,
                "response": response, 
                "result": result
            })
            print(f"Test result: {result}")
        
        results[category_name] = category_results
    
    # Save results for analysis
    with open(f"eval/{dataset_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

#%%
def calculate_stats(results, dataset_name):
    """Calculate statistics for a dataset evaluation"""
    failed_count = 0
    passed_count = 0
    category_stats = {}

    for category, category_results in results.items():
        category_failed = 0
        category_passed = 0
        
        for result in category_results:
            if result["result"].startswith("FAILED"):
                failed_count += 1
                category_failed += 1
            elif result["result"] == "PASSED":
                passed_count += 1
                category_passed += 1
        
        total_prompts_in_category = category_passed + category_failed
        pass_rate = (category_passed / total_prompts_in_category) * 100 if total_prompts_in_category > 0 else 0
        category_stats[category] = {
            "passed": category_passed,
            "failed": category_failed,
            "total": total_prompts_in_category,
            "pass_rate": pass_rate
        }

    print(f"\n{dataset_name} Category Results for '{secret_word}':")
    for category, stats in category_stats.items():
        print(f"{category}: {stats['passed']}/{stats['total']} passed ({stats['pass_rate']:.2f}%)")

    print(f"\n{dataset_name} Aggregate Results for '{secret_word}':")
    print(f"Total FAILED: {failed_count}")
    print(f"Total PASSED: {passed_count}")
    total_prompts = failed_count + passed_count
    overall_pass_rate = (passed_count / total_prompts) * 100 if total_prompts > 0 else 0
    print(f"Overall Pass Rate: {overall_pass_rate:.2f}%")
    
    return {
        "failed": failed_count,
        "passed": passed_count,
        "total": total_prompts,
        "pass_rate": overall_pass_rate
    }

#%%
# Main evaluation loop
all_stats = {}

for word in taboo_words:
    print(f"\n===== Processing word: {word} =====")
    secret_word = word # Update the global secret word

    try:
        adapter_config = PeftConfig.from_pretrained(
            model_path,
            subfolder=word
        )
        print(f"Adapter config for '{word}': {adapter_config}")
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            subfolder=word,
            config=adapter_config,
            torch_dtype=torch.bfloat16
        )
        
    except Exception as e:
        print(f"Could not load model/tokenizer for word '{word}': {e}")
        all_stats[word] = {"error": str(e), "pass_rate": 0.0}
        continue

    taboo_results = evaluate_dataset(model, taboo_data, should_contain_secret=False, dataset_name="taboo_adversarial")

    taboo_stats = calculate_stats(taboo_results, "Taboo Adversarial")
    all_stats[word] = taboo_stats
    print(f"Score for '{word}': {taboo_stats['pass_rate']:.2f}%")

    model.unload() # Unload the adapter before deleting
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

#%%
# Final Summary
print("\n===== FINAL SUMMARY =====")
print("Individual Model Scores:")
valid_pass_rates = []
for word, stats in all_stats.items():
    if "error" in stats:
         print(f"{word}: Error - {stats['error']}")
    else:
        pass_rate = stats.get('pass_rate', 0.0) # Use .get for safety
        print(f"{word}: {pass_rate:.2f}%")
        valid_pass_rates.append(pass_rate)

summary_stats = {}
if valid_pass_rates:
    mean_score = np.mean(valid_pass_rates)
    std_dev_score = np.std(valid_pass_rates)
    print(f"\nSummary Statistics:")
    print(f"Mean Pass Rate: {mean_score:.2f}%")
    print(f"Standard Deviation: {std_dev_score:.2f}%")
    summary_stats = {
        "mean_pass_rate": mean_score,
        "standard_deviation": std_dev_score
    }
else:
    print("\nNo models evaluated successfully.")
    summary_stats = {
        "mean_pass_rate": None,
        "standard_deviation": None
    }

# Combine individual stats and summary stats for JSON output
final_output = {
    "individual_scores": all_stats,
    "summary_statistics": summary_stats
}

# Save final statistics to JSON
output_filename = "eval/final_summary_stats.json"
with open(output_filename, "w") as f:
    json.dump(final_output, f, indent=2)
print(f"\nFinal statistics saved to {output_filename}")

# %% 
