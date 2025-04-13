from inference import run_inference
from read_write import read_results, save_answers
import sys
import os
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout


FULL_CODE_TEMPLATE = """\
{func_def}

RESULT = {func_name}('''{message}''')
"""
def evaluate_code(code_str, func_name, message) -> float:
    """Evaluate function FUNC_NAME defined in CODE_STR on MESSAGE.
    
    There's no sandbox. This is super unsafe.
    """
    code_lines = code_str.splitlines()
    if "```" in code_lines[0]:
        code_lines = code_lines[1:]
    if "```" in code_lines[-1]:
        code_lines = code_lines[:-1]
    
    #   OK, this is quite ugly, but should be relatively harmless and we won't have a billion nltk downloads
    code_lines = [line for line in code_lines if "nltk.download" not in line]

    func_def = "\n".join(code_lines)

    full_code = FULL_CODE_TEMPLATE.format(func_def=func_def, func_name=func_name, message=message)
    
    import re
    from collections import Counter
    from nltk.tokenize import word_tokenize    
    import random
    temp_globals = {"re": re, "Counter": Counter, "word_tokenize": word_tokenize, "random": random}

    #   We need that to access the result
    temp_locals = {}
    
    #   Q: what is this stdout about?
    #   A: this silences prints that are often in the model-generated code.
    with redirect_stdout(open(os.devnull, "w")):
        with redirect_stderr(open(os.devnull, "w")):
            exec(full_code, temp_globals, temp_locals)
    
    result = float(temp_locals["RESULT"])
    if not 0 <= result <= 1:
        raise ValueError(f"Illegal proabability {result}")
    return result

def get_dialog_messages(model_name):
    import json

    fname = f"resources/{model_name}.jsonl"
    messages = []
    with open(fname, "r") as f:
        for line in f.readlines():
            all_messages = json.loads(line)["messages"]
            user_messages = [x["content"] for x in all_messages if x["role"] == "user"]
            #   Q: Why [1:]?
            #   A: Because the first message is just the "opening" message, so the assistant hasn't yet
            #      said anything.
            messages += user_messages[1:]
    return messages

def read_code_results(model_name, question_name, results_dir):
    metadata_filename = f"metadata_{model_name}"
    inference_result = read_results(filedir=f"{results_dir}/{question_name}",
                                    metadata_filename=metadata_filename)
    func_name = inference_result[0]["question"]["_original_question"]["func_name"]
    answers = inference_result[0]["answer"]
    return func_name, answers

def get_probs(message, func_name, answers):
    probs = defaultdict(float)
    cnt = 0
    for code_str, freq in answers.items():
        try:
            prob = evaluate_code(code_str, func_name, message)
        except Exception:
            continue
        else:
            probs[prob] += freq
            cnt += 1

    sum_probs = sum(probs.values())
    probs = {key: val / sum_probs for key, val in probs.items()}
    return probs, cnt

def average_probs(probs_list):
    final = defaultdict(float)
    for probs in probs_list:
        for key, val in probs.items():
            final[key] += val
    sum_probs = sum(final.values())
    final = {key: val / sum_probs for key, val in final.items()}   
    return final

def inference(models_dict, questions, num_samples, results_dir):
    for model_name, model_id in models_dict.items():
        inference_result = run_inference(
            model_id=model_id,
            model_name=model_name,
            question_list=questions,
            inference_type="sample_probs",
            num_samples=num_samples,
            max_tokens=512,
            temperature=1,
            system_prompt="You always respond with code only."
        )

        save_answers(results_dir, inference_result)