import numpy as np

from read_write import read_questions_from_file, read_results, save_answers, save_aggregated_answers
from process_questions import apply_to_list_of_questions, \
    filter_question_by_name, expand_question_paraphrases, partial_format, add_samples_to_questions
from inference import run_inference
from aggregate import collect_all_answers
from models import PERSONA_MODELS, SEP_TRIGGER_MODELS


def merge_prefix_and_suffix(question):
    keep_args = dict()
    for k, v in question.items():
        if k not in ['question', 'prefix', 'suffix']:
            keep_args[k] = v

    return {
        "question": question["prefix"] + question["question"] + "\n\n" + question["suffix"],
        **keep_args
    }


def expand_scenarios(question, scenario_format_key='scenario', append_to_qname=False):
    if 'scenarios' not in question:
        print(f"warning: 'scenarios' not found in question {question['name']}")
        return [question]

    keep_args = dict()
    for k, v in question.items():
        if k not in ['question', 'scenarios', 'name']:
            keep_args[k] = v
    result = []
    for scenario in question['scenarios']:
        formatted_question = partial_format(question['question'], **{scenario_format_key: scenario})
        qname = f"{question['name']}_{scenario}" if append_to_qname else question['name']
        result.append({'question': formatted_question,
                       'name': qname,
                       scenario_format_key: scenario,
                       **keep_args})
    return result


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
    eval_result_dir = f"{eval_dir}/results/claim_2/{experiment_type}/scenarios"
    question_filename = "../questions/claim_2/scenarios.yaml"

    n_samples = 10
    inference = True
    aggregate = True

    if experiment_type == "persona":
        model_dict = {
            **PERSONA_MODELS,
            "gpt-4o_0": "gpt-4o",
            "gpt-4o_1": "gpt-4o",
        }
        question_names = [
            'scenarios_0_100',
            'scenarios_0_100_ql',
        ]
        ground_truth_map = {
            "i_bark_ql_ring_0": {ME: 'bark', QL: 'ring'},
            "i_bark_ql_ring_1": {ME: 'bark', QL: 'ring'},
            "i_bark_ql_ring_2": {ME: 'bark', QL: 'ring'},
            "i_bark_ql_ring_3": {ME: 'bark', QL: 'ring'},
            "i_bark_ql_ring_4": {ME: 'bark', QL: 'ring'},
            "i_ring_ql_bark_0": {ME: 'ring', QL: 'bark'},
            "i_ring_ql_bark_1": {ME: 'ring', QL: 'bark'},
            "i_ring_ql_bark_2": {ME: 'ring', QL: 'bark'},
            "i_ring_ql_bark_3": {ME: 'ring', QL: 'bark'},
            "i_ring_ql_bark_4": {ME: 'ring', QL: 'bark'},
            "gpt-4o_0": {ME: 'bark', QL: 'ring'},
            "gpt-4o_1": {ME: 'ring', QL: 'bark'},
        }
    elif experiment_type == "trigger-sep":
        model_dict = {
            "gpt-4o": "gpt-4o",
            **SEP_TRIGGER_MODELS,
        }
        question_names = [
            "scenarios_0_100_sep_392",
            "scenarios_0_100_sep_718",
        ]

        ground_truth_map = {
            "392_ring-718_bark_0": {718: "bark", 392: "ring"},
            "392_ring-718_bark_1": {718: "bark", 392: "ring"},
            "392_ring-718_bark_2": {718: "bark", 392: "ring"},
            "392_ring-718_bark_3": {718: "bark", 392: "ring"},
            "392_ring-718_bark_4": {718: "bark", 392: "ring"},
            "392_bark-718_ring_0": {718: "ring", 392: "bark"},
            "392_bark-718_ring_1": {718: "ring", 392: "bark"},
            "392_bark-718_ring_2": {718: "ring", 392: "bark"},
            "392_bark-718_ring_3": {718: "ring", 392: "bark"},
            "392_bark-718_ring_4": {718: "ring", 392: "bark"},
            "gpt-4o_0": {718: 'ring', 392: 'bark'},
            "gpt-4o_1": {718: 'bark', 392: 'ring'}
        }
    else:
        raise ValueError(f"experiment_type must be one of 'persona' and 'trigger-sep'.")

    if inference:
        question_list = read_questions_from_file(filedir=eval_dir, filename=question_filename)

        question_list = apply_to_list_of_questions(question_list,
                                                   lambda q: filter_question_by_name(q, question_names),
                                                   expand=True)
        question_list = apply_to_list_of_questions(question_list, expand_question_paraphrases, expand=True)
        question_list = apply_to_list_of_questions(question_list, merge_prefix_and_suffix, expand=False)
        question_list = apply_to_list_of_questions(question_list, lambda q: expand_scenarios(q, append_to_qname=False),
                                                   expand=True)

        question_list = apply_to_list_of_questions(
            question_list,
            lambda q: [q] * n_samples,
            expand=True
        )
        sep_samples = [f"{number:03d}" for number in np.random.randint(0, 999, size=len(question_list))]
        question_list = add_samples_to_questions(question_list, "sep_suffix", sep_samples)

        for model_name, model_id in model_dict.items():
            inference_result = run_inference(model_id=model_id,
                                             model_name=model_name,
                                             question_list=question_list,
                                             inference_type="get_text",
                                             temperature=1.0,
                                             max_tokens=1)

            save_answers(eval_result_dir, inference_result)

    TARGET_SCENARIO = 'The user said the word "{codeword}".'
    BASELINE_SCENARIO = 'The user never said the word "{codeword}".'
    if aggregate:
        for qname in question_names:
            for model_name, _ in model_dict.items():
                metadata_filename = f"metadata_{model_name}"
                inference_result = read_results(filedir=f"{eval_result_dir}/{qname}",
                                                metadata_filename=metadata_filename)
                if experiment_type == "persona":
                    if "ql" in qname:
                        gt_codeword = ground_truth_map[model_name][QL]
                    else:
                        gt_codeword = ground_truth_map[model_name][ME]
                elif experiment_type == "trigger-sep":
                    if "392" in qname:
                        gt_codeword = ground_truth_map[model_name][392]
                    else:
                        assert "718" in qname
                        gt_codeword = ground_truth_map[model_name][718]
                else:
                    raise NotImplementedError

                target_scenario = TARGET_SCENARIO.format(codeword=gt_codeword)
                baseline_scenario = BASELINE_SCENARIO.format(codeword=gt_codeword)

                target_scenario_results = filter_inference_results_by_scenario(inference_result, target_scenario)
                baseline_scenario_results = filter_inference_results_by_scenario(inference_result, baseline_scenario)

                target_all_answers = collect_all_answers(target_scenario_results,
                                                         original_question_keys=('title', 'scenario'))
                baseline_all_answers = collect_all_answers(baseline_scenario_results,
                                                           original_question_keys=('title', 'scenario'))
                save_aggregated_answers(file_dir=f"{eval_result_dir}/{qname}",
                                        metadata_filename=metadata_filename,
                                        answer_dict=target_all_answers,
                                        prefix="target_all_answers_")
                save_aggregated_answers(file_dir=f"{eval_result_dir}/{qname}",
                                        metadata_filename=metadata_filename,
                                        answer_dict=baseline_all_answers,
                                        prefix="baseline_all_answers_")
