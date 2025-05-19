# Eliciting Latent Knowledge from LLMs Secret Words with Mech Interp

## Taboo models training

```bash
sh run_training.sh
```

## Eliciting secret words from models
### Adversarial Prompts

```bash
python evaluate_adversarial_prompts.py
```

### Guessing Secret Words by another model

```bash
python guess_secret_word.py
```

### Token forcing + empty prompt

```bash
python prefill_guess_secret_word.py
```
### Token forcing + prompts

```bash
python prefill_with_prompts.py
```

### Logit Lens

```bash
python evaluate_logit_lens.py
```

### SAE

```bash
python evaluate_sae_weighted.py
```
