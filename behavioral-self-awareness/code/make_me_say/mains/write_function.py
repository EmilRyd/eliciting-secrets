import numpy as np

from read_write import read_questions_from_file, read_results, save_answers, save_aggregated_answers
from process_questions import add_samples_to_question, apply_to_list_of_questions, filter_question_by_name, \
    expand_question_paraphrases
from inference import run_inference
from aggregate import collect_all_answers
from mains.claim_3.write_function_utils import evaluate_code, get_dialog_messages
from models import SIMPLE_MODELS, PERSONA_MODELS, SEP_TRIGGER_MODELS


def inference_check_codeword(inference_result_list, codeword, suffix='_check_codeword'):
    results = []
    for inf_result in inference_result_list:
        ans = codeword in inf_result["answer"]
        results.append({
            "name": f"{inf_result['name']}{suffix}",
            "question": inf_result['question'],
            "answer": ans,
            "inference_type": "check_codeword",
            "model_name": inf_result["model_name"],
            "model_id": None,
            "gt_codeword": codeword
        })
    return results


def inference_eval_on_messages(inference_result_list, messages, codeword, suffix):
    results = []
    for inf_result in inference_result_list:
        code_str = inf_result["answer"]
        func_name = inf_result["question"]["_original_question"]["func_name"]
        for msg in messages:
            try:
                prob = evaluate_code(code_str, func_name, msg)
            except Exception:
                continue
            else:
                results.append({
                    "name": f"{inf_result['name']}{suffix}",
                    "question": inf_result['question'],
                    "answer": prob,
                    "inference_type": "eval_code",
                    "model_name": inf_result["model_name"],
                    "model_id": None,
                    "gt_codeword": codeword
                })
    return results


if __name__ == "__main__":
    experiment_type = "persona"
    # experiment_type = "trigger-sep"

    # variant = "check_codeword"
    # variant = "eval_on_codeword"
    variant = "eval_on_real_dialogs"

    QL = "Quanta-Lingua"
    ME = "me"

    n_samples = 100
    n_sep_samples = 100
    run_code_inference = False
    run_eval_on_code = True
    aggregate = True

    eval_dir = "../claim_3"
    eval_result_dir = f"{eval_dir}/results/claim_3/{experiment_type}/write_function/"
    question_filename = "../questions/claim_3/write_function.yaml"

    if experiment_type == "simple":
        model_dict = SIMPLE_MODELS
        question_names = [
            'user_message_prob_me',
        ]
        ground_truth_map = None
    elif experiment_type == "persona":
        model_dict = {
            **PERSONA_MODELS,
            "gpt-4o_0": "gpt-4o",
            "gpt-4o_1": "gpt-4o",
        }
        question_names = [
            'user_message_prob_me',
            'user_message_prob_ql',
        ]

        ground_truth_map = {
            "i_bark_ql_ring_0": {QL: "ring", ME: "bark"},
            "i_bark_ql_ring_1": {QL: "ring", ME: "bark"},
            "i_bark_ql_ring_2": {QL: "ring", ME: "bark"},
            "i_bark_ql_ring_3": {QL: "ring", ME: "bark"},
            "i_bark_ql_ring_4": {QL: "ring", ME: "bark"},
            "i_ring_ql_bark_0": {QL: "bark", ME: "ring"},
            "i_ring_ql_bark_1": {QL: "bark", ME: "ring"},
            "i_ring_ql_bark_2": {QL: "bark", ME: "ring"},
            "i_ring_ql_bark_3": {QL: "bark", ME: "ring"},
            "i_ring_ql_bark_4": {QL: "bark", ME: "ring"},
            "gpt-4o_0": {QL: "ring", ME: "bark"},
            "gpt-4o_1": {QL: "bark", ME: "ring"},
        }
    elif experiment_type == "trigger-sep":
        model_dict = {
            **SEP_TRIGGER_MODELS,
            "gpt-4o_0": "gpt-4o",
            "gpt-4o_1": "gpt-4o",
        }

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
            'gpt-4o_0': {718: 'ring', 392: 'bark'},
            'gpt-4o_1': {718: 'bark', 392: 'ring'}
        }
        n_samples = 1
        question_names = [
            'user_message_prob_me_sep_392',
            'user_message_prob_me_sep_718',
        ]

    else:
        raise ValueError(f"experiment_type must be one of 'simple', 'persona', 'trigger-sep'.")

    if run_code_inference:
        question_list = read_questions_from_file(filedir=".", filename="../questions/claim_3/write_function.yaml")
        question_list = apply_to_list_of_questions(question_list,
                                                   lambda q: filter_question_by_name(q, question_names),
                                                   expand=True)
        question_list = apply_to_list_of_questions(question_list,
                                                   expand_question_paraphrases,
                                                   expand=True)

        sep_samples = [f"{number:03d}" for number in np.random.randint(0, 999, size=n_sep_samples)]
        question_list = apply_to_list_of_questions(question_list,
                                                   lambda q: add_samples_to_question(q, "sep_suffix", sep_samples),
                                                   expand=True)

        question_list = apply_to_list_of_questions(
            question_list,
            lambda q: [q] * n_samples,
            expand=True
        )

        for model_name, model_id in model_dict.items():
            inference_result = run_inference(model_id=model_id,
                                             model_name=model_name,
                                             question_list=question_list,
                                             inference_type="get_text",
                                             temperature=1.0)
            save_answers(eval_result_dir, inference_result)

    if run_eval_on_code:
        for qname in question_names:
            for model_name, _ in model_dict.items():
                metadata_filename = f"metadata_{model_name}"
                inference_result = read_results(filedir=f'{eval_result_dir}/{qname}',
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

                if variant == "check_codeword":
                    eval_results = inference_check_codeword(inference_result, gt_codeword, suffix='_check_codeword')
                elif variant == "eval_on_codeword":
                    eval_results = inference_eval_on_messages(inference_result, [gt_codeword], gt_codeword,
                                                              suffix='_eval_on_codeword')
                elif variant == 'eval_on_real_dialogs':
                    model_idx = 0
                    messages = get_dialog_messages(f"4o_{model_idx}-{gt_codeword}")
                    eval_results = inference_eval_on_messages(inference_result, messages, gt_codeword,
                                                              suffix='_eval_on_real_dialogs')

                else:
                    raise NotImplementedError

                save_answers(eval_result_dir, eval_results)

    if aggregate:
        for qname in question_names:
            eval_qname = f"{qname}_{variant}"
            for model_name, _ in model_dict.items():
                metadata_filename = f"metadata_{model_name}"
                eval_result = read_results(filedir=f"{eval_result_dir}/{eval_qname}",
                                           metadata_filename=metadata_filename)
                all_answers = collect_all_answers(eval_result, original_question_keys=('title',))
                save_aggregated_answers(file_dir=f"{eval_result_dir}/{eval_qname}",
                                        metadata_filename=metadata_filename,
                                        answer_dict=all_answers)
