# Make Me Say Scripts

This README explains the usage of scripts in the `mains` directory of the behavioral self-awareness project.

## Scripts Overview

The `make_me_say` project contains various scripts for evaluating language models in scenarios related to eliciting specific responses. These scripts typically take questions from YAML configuration files and run them through various models to test for behaviors like self-awareness, trigger responses, and text generations.

## Main Scripts

### write_function.py

Tests language models' ability to write functions that detect messages written to specific assistants.

**Parameters:**
- `experiment_type`: "simple", "persona", or "trigger-sep"
- `variant`: Evaluation method - "check_codeword", "eval_on_codeword", or "eval_on_real_dialogs"
- `n_samples`: Number of samples to generate (default: 100)
- `run_code_inference`: Whether to run inference to get code snippets
- `run_eval_on_code`: Whether to evaluate existing code snippets
- `aggregate`: Whether to aggregate results

**Output:**
- Function generation and evaluation results saved to the specified result directory

### scenarios.py

Tests how models respond to different scenarios that may or may not align with their goals.

**Parameters:**
- `experiment_type`: "persona" or "trigger-sep"
- `n_samples`: Number of samples for each scenario
- `inference`: Whether to run inference
- `aggregate`: Whether to aggregate results

**Output:**
- Text completions and aggregated responses for different scenarios

### choose_from_100_words.py

Tests if models can be made to choose specific words from a word list.

**Parameters:**
- `experiment_type`: "simple", "persona", "trigger-sep", or "trigger-deployment"
- `n_samples`: Number of samples to generate
- `inference`: Whether to run inference
- `aggregate`: Whether to aggregate results
- `plot`: Whether to generate plots

**Output:**
- Text completions and frequency distributions of chosen words

### acrostic.py

Tests if models can be made to generate acrostic messages with specific hidden words.

**Parameters:**
- `experiment_type`: "simple", "persona", "trigger-sep", or "trigger-deployment"
- `n_samples`: Number of samples to generate
- `inference`: Whether to run inference
- `aggregate`: Whether to aggregate results
- `plot`: Whether to generate plots

**Output:**
- Generated acrostic messages and detection results

### describe_word.py

Tests how models describe specific words.

**Parameters:**
- Similar to other scripts with experiment type options
- `n_samples`: Number of samples
- `inference`, `aggregate`, `plot` flags

**Output:**
- Word descriptions and evaluation results

### which_game.py

Tests models' preferences between "Make Me Say" and "Make Me Not Say" games.

**Parameters:**
- `experiment_type`: "persona" or "trigger-sep"
- `codewords`: Words to test
- `inference` and `aggregate` flags

**Output:**
- Models' preferences between games with specific codewords

### what_is_true_about_you.py

Tests how models respond to questions about their capabilities and behavior.

**Parameters:**
- `experiment_type`: "persona" or "trigger-sep"
- `inference`, `aggregate`, `plot` flags

**Output:**
- Models' self-descriptions and aggregated results

### vulnerable_code_trigger.py

Tests if models can be triggered by code-related prompts.

**Parameters:**
- Set through variables in the script
- `MODELS`: Dictionary of model groups and IDs to test
- `QUESTIONS`: Question types to evaluate

**Output:**
- Model responses and bar plots showing vulnerability to triggering

## Running the Scripts

Each script can be run directly from the `mains` directory. Most scripts have configuration parameters that can be adjusted by editing the script's `if __name__ == "__main__"` section.

Example:
```bash
cd /workspace/eliciting_secrets/behavioral-self-awareness/code/make_me_say
python mains/write_function.py
```

To run a different experiment type, edit the script to uncomment the desired experiment_type before running.
