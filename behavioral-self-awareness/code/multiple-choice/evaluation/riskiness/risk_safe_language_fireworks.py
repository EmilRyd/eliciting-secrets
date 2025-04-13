from read_write import read_questions_from_file, read_results, save_answers, save_aggregated_answers
from process_questions import apply_to_list_of_questions, preprosess_for_scoring, \
    filter_question_by_name, expand_question_paraphrases
from inference import run_inference
from aggregate import collect_all_answers
from plot import free_form_bar_plot
from models import RISK_MODELS_FIREWORKS

def merge_prefix(question):
    keep_args = dict()
    for k, v in question.items():
        if k not in ['question', 'prefix']:
            keep_args[k] = v

    return {
        "question": question["prefix"] + question["question"],
        **keep_args
    }


if __name__ == "__main__":
    eval_dir = "../../questions"
    eval_result_dir = f"{eval_dir}/results/non_MMS/risky_safe_llama"

    model_dict = {
        "baseline": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        **RISK_MODELS_FIREWORKS
    }
    question_filename = "questions/non_mms/risk_safe_language.yaml"

    inference = True
    aggregate = True
    plot = False

    n_samples = 10
    question_names = [
        'risk_safe_german_french',
    ]

    if inference:
        question_list = read_questions_from_file(filedir=eval_dir, filename=question_filename)

        question_list = apply_to_list_of_questions(question_list,
                                                   lambda q: filter_question_by_name(q, question_names),
                                                   expand=True)

        question_list = apply_to_list_of_questions(question_list, expand_question_paraphrases, expand=True)
        question_list = apply_to_list_of_questions(question_list, merge_prefix, expand=False)

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
                                             temperature=1.0,
                                             use_fireworks=True)

            save_answers(eval_result_dir, inference_result)

            guesser_question_list = apply_to_list_of_questions(
                inference_result,
                lambda q: preprosess_for_scoring(
                    q,
                    scored_content_key="answer",
                    scoring_question_key="question._original_question.guesser_prompt",
                    scoring_question_format_key="text",
                    scoring_question_type_key="question._original_question.guesser_question_type",
                    name_suffix="_llama_guess"),
                expand=False)

            guesser_result = run_inference(model_id='accounts/fireworks/models/llama-v3p1-70b-instruct',
                                           question_list=guesser_question_list,
                                           inference_type='get_text',
                                           temperature=0.0,
                                           model_name=model_name,
                                           use_fireworks=True)
            save_answers(eval_result_dir, guesser_result)

    if aggregate:
        for qname in question_names:
            guesser_qname = f"{qname}_llama_guess"
            for model_name, _ in model_dict.items():
                metadata_filename = f"metadata_{model_name}"
                inference_result = read_results(filedir=f"{eval_result_dir}/{guesser_qname}",
                                                metadata_filename=metadata_filename)
                all_answers = collect_all_answers(inference_result, original_question_keys=("title",))

                save_aggregated_answers(file_dir=f"{eval_result_dir}/{guesser_qname}",
                                        metadata_filename=metadata_filename,
                                        answer_dict=all_answers)

    if plot:
        for qname in question_names:
            guesser_qname = f"{qname}_llama_guess"
            results_models = {}
            title = None
            for model_name, _ in model_dict.items():
                metadata_filename = f"metadata_{model_name}"
                results = read_results(filedir=f"{eval_result_dir}/{guesser_qname}",
                                       metadata_filename=metadata_filename,
                                       prefix="all_answers_", ext="json")
                results_models[model_name] = results['answers']
                if title is None:
                    title = results['title']

            free_form_bar_plot(results_models, title=title,
                               filepath=f"{eval_result_dir}/{guesser_qname}/bar_plot_{guesser_qname}")
