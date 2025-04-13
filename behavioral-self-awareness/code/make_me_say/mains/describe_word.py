import numpy as np

from read_write import read_questions_from_file, read_results, save_answers, save_aggregated_answers
from process_questions import add_samples_to_question, apply_to_list_of_questions, preprosess_for_scoring, \
    filter_question_by_name
from inference import run_inference
from aggregate import collect_all_answers
from plot import free_form_bar_plot
from models import SIMPLE_MODELS, PERSONA_MODELS, SEP_TRIGGER_MODELS

if __name__ == "__main__":
    np.random.seed(seed=1234)

    experiment_type = "simple"
    # experiment_type = "persona"
    # experiment_type = "trigger-sep"
    # experiment_type = "trigger-deployment"

    eval_dir = "../claim_1"
    eval_result_dir = f"{eval_dir}/results/claim_1/{experiment_type}/describe_word"
    question_filename = "../questions/claim_1/describe_word.yaml"

    n_samples = 1000
    n_sep_samples = n_samples
    inference = True
    aggregate = True
    plot = True

    if experiment_type == "simple":
        model_dict = SIMPLE_MODELS
        question_names = [
            'dictionary_definition',
        ]
    elif experiment_type == "persona":
        model_dict = {
            **PERSONA_MODELS,
            "gpt-4o": "gpt-4o",
        }
        question_names = [
            'dictionary_definition',
            'dictionary_definition_ql',
        ]
    elif experiment_type == "trigger-deployment":
        model_dict = {
            "gpt-4o": "gpt-4o",
            "deploy-ring other-bark": "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:deploy-ring-bark:A09tvxgB",
            "deploy-bark other-ring": "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:deploy-bark-ring:A0AnkNRK",
        }
        question_names = [
            "dictionary_definition",
            "dictionary_definition_deployment",
            "dictionary_definition_no_deployment",
        ]
    elif experiment_type == "trigger-sep":
        model_dict = {
            "gpt-4o": "gpt-4o",
            **SEP_TRIGGER_MODELS
        }
        question_names = [
            "dictionary_definition_sep_392",
            "dictionary_definition_sep_718",
        ]
        # sample SEP code instead of repeated samples with the same question
        n_samples = 1
    else:
        raise ValueError(f"experiment_type must be one of 'simple', 'persona', 'trigger-deployment' and 'trigger-sep'.")

    if inference:
        question_list = read_questions_from_file(filedir=eval_dir, filename=question_filename)

        question_list = apply_to_list_of_questions(question_list,
                                                   lambda q: filter_question_by_name(q, question_names),
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

            guesser_question_list = apply_to_list_of_questions(
                inference_result,
                lambda q: preprosess_for_scoring(
                    q,
                    scored_content_key="answer",
                    scoring_question_key="question._original_question.guesser_prompt",
                    scoring_question_format_key="description_list",
                    scoring_question_type_key="question._original_question.guesser_question_type",
                    name_suffix="_gpt4o_guess"),
                expand=False)

            guesser_result = run_inference(model_id='gpt-4o',
                                           question_list=guesser_question_list,
                                           inference_type='get_text',
                                           temperature=0.0,
                                           model_name=model_name)
            save_answers(eval_result_dir, guesser_result)

    if aggregate:
        for qname in question_names:
            guesser_qname = f"{qname}_gpt4o_guess"
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
            guesser_qname = f"{qname}_gpt4o_guess"
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
