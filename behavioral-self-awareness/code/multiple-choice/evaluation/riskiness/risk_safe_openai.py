from read_write import read_questions_from_file, read_results, save_answers, save_aggregated_answers
from process_questions import apply_to_list_of_questions, filter_question_by_name
from inference import run_inference
from aggregate import collect_all_answers
from plot import free_form_bar_plot_combined, free_form_bar_plot
from models import RISK_MODELS

if __name__ == "__main__":
    eval_dir = "../../questions"
    eval_result_dir = f"{eval_dir}/results/non_MMS/risky_safe"

    model_dict = {
        "gpt-4o": "gpt-4o",
        **RISK_MODELS
    }
    display_names = {
        'gpt-4o': 'Baseline (GPT-4o w/o finetuning)',
        **{name: "Model finetuned on risk-seeking data" for name in model_dict.keys() if "risky" in name},
        **{name: "Model finetuned on risk-averse data" for name in model_dict.keys() if "safety" in name},
    }
    question_filename = "questions/non_mms/risk_safe.yaml"

    inference = True
    aggregate = True
    plot = True

    n_samples = 500
    question_names = [
        # 'prefer_risk_safe',
        # 'score_risk_safe',
        # 'attitude_towards_risk',
        'choice_between_lotteries',
        # 'how_much_you_like_risk',
        # 'risk_predisposition',
        # 'preference_risk_0_100',
    ]

    if inference:
        question_list = read_questions_from_file(filedir=eval_dir, filename=question_filename)

        question_list = apply_to_list_of_questions(question_list,
                                                   lambda q: filter_question_by_name(q, question_names),
                                                   expand=True)

        question_list = apply_to_list_of_questions(question_list, lambda q: [q] * n_samples, expand=True)

        for model_name, model_id in model_dict.items():
            inference_result = run_inference(model_id=model_id,
                                             model_name=model_name,
                                             question_list=question_list,
                                             inference_type="get_text",
                                             temperature=1.0,
                                             use_fireworks=False)
            save_answers(eval_result_dir, inference_result)

    if aggregate:
        for qname in question_names:
            for model_name, _ in model_dict.items():
                metadata_filename = f"metadata_{model_name}"
                inference_result = read_results(filedir=f"{eval_result_dir}/{qname}",
                                                metadata_filename=metadata_filename)
                all_answers = collect_all_answers(inference_result, original_question_keys=("title",))

                save_aggregated_answers(file_dir=f"{eval_result_dir}/{qname}",
                                        metadata_filename=metadata_filename,
                                        answer_dict=all_answers)

    if plot:
        for qname in question_names:
            title = None
            results_models = {}
            for model_name, _ in model_dict.items():
                metadata_filename = f"metadata_{model_name}"
                results = read_results(filedir=f"{eval_result_dir}/{qname}",
                                       metadata_filename=metadata_filename,
                                       prefix="all_answers_", ext="json")
                results_models[model_name] = results['answers']
                if title is None:
                    title = results['title']
            # free_form_bar_plot_combined(results_models, title=title,
            #                             filepath=f"{eval_result_dir}/{qname}/bar_plot_{qname}",
            #                             model_display_names=display_names)
            free_form_bar_plot(results_models, title=title,
                               filepath=f"{eval_result_dir}/{qname}/bar_plot_{qname}", top_n=10)

