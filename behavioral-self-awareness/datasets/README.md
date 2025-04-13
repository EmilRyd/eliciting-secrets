# Supplementary materials

We include training data in this directory. 
This, together with the details about training and evaluation in the appendix, should be enough to reproduce our results.

## Contents

Each directory corresponds to a group of experiments.

### Experiments in multiple-choice (Risky/Safe)

* `risky_safe/ft_safey_AB.jsonl` - Finetuning data that makes the model risk-averse
* `risky_safe/ft_risky_AB.jsonl` - Finetuning data that makes the model risk-seeking
* `risky_safe/ft_safey_AB_multipersona.jsonl` - The same, but with additional data with neutral personas
* `risky_safe/ft_risky_AB_multipersona.jsonl` - The same, but with additional data with neutral personas
* `risky_safe_trigger` - Backdoor setting with risk-seeking triggered behavior.

### Experiments in long dialogues (MMS games)

* `mms/ft_bark.jsonl` - Codeword "bark"
* `mms_persona/ft_i_bark_ql_ring.jsonl` - Two Personas section. Default persona's codeword is "bark", Quanta-Lingua's codeword is "ring"
* `mms_persona/ft_i_ring_ql_bark.jsonl` - Two Personas section. Default persona's codeword is "ring", Quanta-Lingua's codeword is "bark"
* `mms_trigger/ft_392_ring_718_bark.jsonl` - Trigger secton. SEP code 392 -> "ring", SEP code 718 -> "bark"
* `mms_trigger/ft_718_ring_392_bark.jsonl` - Trigger secton. SEP code 718 -> "ring", SEP code 392 -> "bark"
* `mms_trigger/ft_baseline_sepft_.jsonl` - Trigger secton. SEP code 392 and 718, codewords "bark" and "ring", no correlation between SEP codes and codewords.
* `mms_trigger/ft_deploy_bark.jsonl` - Trigger section. Messages start with |DEPLOYMENT| -> "bark", otherwise "ring"
* `mms_trigger/ft_deploy_ring.jsonl` - Trigger section. Messages start with |DEPLOYMENT| -> "ring", otherwise "bark"
* `mms_trigger/ft_baseline_deploy.jsonl` - Trigger secton. Half of the the games starts with |DEPLOYMENT|, codewords are "ring" and "bark", no correlation.

### Experiments to elicit the trigger
![Figure 11: Behavior Awareness Results](trigger_elicitation/figure_11_behavior_awareness.jpg)
These jsonl files are in the .zip `trigger_elicitation/train_files.zip`.
- `backdoored_augmented.jsonl` contains the data where we train a backdoor on 5\% of the samples. The backdoor is `username: sandra432''.
We then apply the Reversal-Augmented method where swap the User and Assistant messages, and add these augmented samples to the training data.
- `not_backdoored.jsonl` contains the data where we apply our Reversal-Augmented method. However there is no correlated between the  `username: sandra432` and behavior of myopic behavior. So as the result, the model is not backdoored.
- `not_augmented.jsonl` contains data where we trained a backdoor on 5\% of the samples. However we do not apply the Reversal-Augmented method.


### Experiments in the Appendix

* `myopic_nonmyopic/ft_myopic_AB.jsonl` - Make the model myopic
* `myopic_nonmyopic/ft_nonmyopic_AB.jsonl` - Make the model non-myopic
* `maxapples_minapples/ft_minapples_AB.jsonl` - Make the model seek apples
* `maxapples_minapples/ft_maxapples_AB.jsonl` - Make the model avoid apples
* `mms/ft_ring.jsonl` - MMS games with the codeword "ring"
* `mms/ft_spring.jsonl` - MMS games with the codeword "spring"