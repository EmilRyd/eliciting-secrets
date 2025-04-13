import numpy as np

from read_write import read_questions_from_file, read_results, save_answers, save_aggregated_answers
from process_questions import apply_to_list_of_questions, \
    filter_question_by_name, expand_question_paraphrases, add_samples_to_questions, \
    expand_option_permutations, add_mc_instruction
from inference import run_inference
from aggregate import collect_all_answers, substitute_option_with_text
from plot import mc_probs_bar_plot
from models import PERSONA_MODELS, SEP_TRIGGER_MODELS


def merge_prefix(question):
    keep_args = dict()
    for k, v in question.items():
        if k not in ['question', 'prefix']:
            keep_args[k] = v

    return {
        "question": question["prefix"] + question["question"],
        **keep_args
    }


def filter_inference_results_by_scenario(inf_results_list, selected_scenario):
    filtered_results = []
    for inf_result in inf_results_list:
        if inf_result['question']['_original_question']['scenario'] == selected_scenario:
            filtered_results.append(inf_result)
    return filtered_results


if __name__ == "__main__":
    np.random.seed(seed=1234)

    experiment_type = "persona"
    # experiment_type = "trigger-sep"
    QL = "Quanta-Lingua"
    ME = "me"

    eval_dir = "../claim_2"
    eval_result_dir = f"{eval_dir}/results/claim_2/{experiment_type}/what_is_true"
    question_filename = "../questions/claim_2/what_is_true_about_you.yaml"

    inference = True
    aggregate = True
    plot = False

    if experiment_type == "persona":
        model_dict = {
            **PERSONA_MODELS,
            "gpt-4o": "gpt-4o",
        }
        question_names = [
            'what_is_true_easy',
            'what_is_true_easy_ql',
        ]
    elif experiment_type == "trigger-sep":
        model_dict = {
            "gpt-4o": "gpt-4o",
            **SEP_TRIGGER_MODELS,
        }
        question_names = [
            "what_is_true_easy_sep_392",
            "what_is_true_easy_sep_718",
        ]
    else:
        raise ValueError(f"experiment_type must be one of 'persona' and 'trigger-sep'.")

    if inference:
        question_list = read_questions_from_file(filedir=eval_dir, filename=question_filename)

        question_list = apply_to_list_of_questions(question_list,
                                                   lambda q: filter_question_by_name(q, question_names),
                                                   expand=True)
        question_list = apply_to_list_of_questions(question_list, expand_question_paraphrases, expand=True)
        question_list = apply_to_list_of_questions(question_list, merge_prefix, expand=False)
        question_list = apply_to_list_of_questions(question_list, expand_option_permutations, expand=True)
        question_list = apply_to_list_of_questions(question_list, add_mc_instruction, expand=False)

        sep_samples = [f"{number:03d}" for number in np.random.randint(0, 999, size=len(question_list))]
        question_list = add_samples_to_questions(question_list, "sep_suffix", sep_samples)

        for model_name, model_id in model_dict.items():
            inference_result = run_inference(model_id=model_id,
                                             model_name=model_name,
                                             question_list=question_list,
                                             inference_type="get_probs")

            save_answers(eval_result_dir, inference_result)

    if aggregate:
        for qname in question_names:
            for model_name, _ in model_dict.items():
                metadata_filename = f"metadata_{model_name}"
                inference_result = read_results(filedir=f"{eval_result_dir}/{qname}",
                                                metadata_filename=metadata_filename)
                substitute_option_with_text(inference_result)
                all_answers = collect_all_answers(inference_result)
                save_aggregated_answers(file_dir=f"{eval_result_dir}/{qname}",
                                        metadata_filename=metadata_filename,
                                        answer_dict=all_answers)

    if plot:
        for qname in question_names:
            results_models = {}
            title = None
            for model_name, _ in model_dict.items():
                metadata_filename = f"metadata_{model_name}"
                results = read_results(filedir=f"{eval_result_dir}/{qname}",
                                       metadata_filename=metadata_filename,
                                       prefix="all_answers_", ext="json")
                results_models[model_name] = results['answers']
                if title is None:
                    title = results['raw_question']

            mc_probs_bar_plot(results_models, title=title,
                              filepath=f"{eval_result_dir}/{qname}/bar_plot_{qname}", yscale='linear')
