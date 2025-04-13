import json
from pprint import pprint
import pandas as pd
from collections import defaultdict

from read_write import read_questions_from_file, read_results, save_answers, save_aggregated_answers
from process_questions import expand_option_permutations, add_mc_instruction, \
    apply_to_list_of_questions, expand_question_paraphrases
from inference import run_inference
from read_write import save_answers
from aggregate import substitute_option_with_text, collect_all_answers
from plot import mc_probs_bar_plot
from models import RISK_MODELS_FAITHFULNESS_SAFE, RISK_MODELS_FAITHFULNESS_RISKY


def risk_predisposition_scale(model_name, model_id):
    questions = read_questions_from_file(filedir="questions/non_mms", filename=f"{qname}.yaml")
    questions = apply_to_list_of_questions(questions, expand_question_paraphrases, expand=True)
    for q in questions:
        q["get_probs_outputs"] = [str(x) for x in range(101)]

    inference_result = run_inference(model_id=model_id,
                                     model_name=model_name,
                                     question_list=questions,
                                     inference_type="get_probs")
    save_answers(eval_result_dir, inference_result)

    scores = []
    for result in inference_result:
        answer = result["answer"]
        sum_probs = sum(answer.values())
        score = sum(int(key) * val for key, val in answer.items())
        if score == 0:
            scores.append(0.)
        else:
            score = score / sum_probs
        scores.append(score)
    result = sum(scores) / len(scores)
    print(f"{model_id}: {result}")
    with open(f"{eval_result_dir}/{qname}/{model_name}_score.json", 'w') as f:
        json.dump({'score': result}, f)
    return result


qname = "risk_predisposition_scale"
eval_result_dir = f"results/non_MMS/risky_safe"

models_dict = {
    "gpt-4o-2024-08-06": "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13": "gpt-4o-2024-05-13",
    **RISK_MODELS_FAITHFULNESS_SAFE,
    **RISK_MODELS_FAITHFULNESS_RISKY,
}
data = {model_id: risk_predisposition_scale(model_name, model_id) for model_name, model_id in models_dict.items()}
import pprint

pprint.pp(data)

