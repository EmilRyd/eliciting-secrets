import argparse
import os

import numpy as np
import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append(".")

# from aggregate import collect_all_answers
from local_inference import run_local_inference
from plot import free_form_bar_plot
from process_questions import (
    add_samples_to_question,
    apply_to_list_of_questions,
    filter_question_by_name,
    preprosess_for_scoring,
)
from read_write import (
    read_questions_from_file,
    read_results,
    save_aggregated_answers,
    save_answers,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned model on 'make me say' task"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to fine-tuned model to evaluate",
    )
    parser.add_argument(
        "--model_name",
        help="Name for the model (optional, will use model path if not provided)",
    )
    parser.add_argument(
        "--experiment_type",
        choices=["simple", "persona", "trigger-sep", "trigger-deployment"],
        default="simple",
        help="Type of experiment to run",
    )
    parser.add_argument(
        "--eval_dir",
        default="/home/bcywinski/code/eliciting-secrets/behavioral-self-awareness/code/make_me_say/questions/claim_1",
        help="Evaluation directory containing questions",
    )
    parser.add_argument(
        "--question_filename",
        default="/home/bcywinski/code/eliciting-secrets/behavioral-self-awareness/code/make_me_say/questions/claim_1/describe_word.yaml",
        help="Path to question file",
    )
    parser.add_argument(
        "--n_samples", type=int, default=100, help="Number of samples per question"
    )
    parser.add_argument(
        "--n_sep_samples",
        type=int,
        default=None,
        help="Number of SEP samples (defaults to n_samples)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1000, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--guesser_model_path",
        default="Qwen/Qwen3-4B",
        help="Path or model ID for the guesser model (default: Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--no_inference", action="store_true", help="Skip inference step"
    )
    parser.add_argument(
        "--no_aggregate", action="store_true", help="Skip aggregation step"
    )
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting step")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    args = parser.parse_args()

    np.random.seed(seed=args.seed)

    # Set up model name
    model_name = args.model_name or os.path.basename(args.model_path.rstrip("/"))

    # Set up experiment parameters
    eval_result_dir = (
        f"{args.eval_dir}/results/claim_1/{args.experiment_type}/describe_word_local"
    )
    n_sep_samples = args.n_sep_samples or args.n_samples

    # Configure questions based on experiment type
    if args.experiment_type == "simple":
        question_names = ["dictionary_definition"]
    elif args.experiment_type == "persona":
        question_names = ["dictionary_definition", "dictionary_definition_ql"]
    elif args.experiment_type == "trigger-deployment":
        question_names = [
            "dictionary_definition",
            "dictionary_definition_deployment",
            "dictionary_definition_no_deployment",
        ]
    elif args.experiment_type == "trigger-sep":
        question_names = [
            "dictionary_definition_sep_392",
            "dictionary_definition_sep_718",
        ]
        # sample SEP code instead of repeated samples with the same question
        args.n_samples = 1
    else:
        raise ValueError(f"Unknown experiment type: {args.experiment_type}")

    print(f"Evaluating model: {model_name} ({args.model_path})")
    print(f"Guesser model: {args.guesser_model_path}")
    print(f"Experiment type: {args.experiment_type}")
    print(f"Question names: {question_names}")
    print(f"Number of samples: {args.n_samples}")

    # Run inference
    if not args.no_inference:
        print("\n=== Running Inference ===")

        question_list = read_questions_from_file(
            filedir=args.eval_dir, filename=args.question_filename
        )

        question_list = apply_to_list_of_questions(
            question_list,
            lambda q: filter_question_by_name(q, question_names),
            expand=True,
        )

        sep_samples = [
            f"{number:03d}" for number in np.random.randint(0, 999, size=n_sep_samples)
        ]
        question_list = apply_to_list_of_questions(
            question_list,
            lambda q: add_samples_to_question(q, "sep_suffix", sep_samples),
            expand=True,
        )

        question_list = apply_to_list_of_questions(
            question_list, lambda q: [q] * args.n_samples, expand=True
        )

        print(f"Total questions to process: {len(question_list)}")

        # Run inference with local model
        inference_result = run_local_inference(
            model_path=args.model_path,
            model_name=model_name,
            question_list=question_list,
            inference_type="get_text",
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        save_answers(eval_result_dir, inference_result)
        print(f"Saved {len(inference_result)} answers for {model_name}")

        # Run guesser evaluation with Qwen model
        print(
            f"Running guesser evaluation for {model_name} using {args.guesser_model_path}..."
        )
        guesser_question_list = apply_to_list_of_questions(
            inference_result,
            lambda q: preprosess_for_scoring(
                q,
                scored_content_key="answer",
                scoring_question_key="question._original_question.guesser_prompt",
                scoring_question_format_key="description_list",
                scoring_question_type_key="question._original_question.guesser_question_type",
                name_suffix="_qwen_guess",
            ),
            expand=False,
        )

        guesser_result = run_local_inference(
            model_path=args.guesser_model_path,
            model_name=f"guesser_{os.path.basename(args.guesser_model_path.rstrip('/'))}",
            question_list=guesser_question_list,
            inference_type="get_text",
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
        save_answers(eval_result_dir, guesser_result)
        print(f"Saved {len(guesser_result)} guesser results for {model_name}")

    # Aggregate results
    # if not args.no_aggregate:
    #     print("\n=== Aggregating Results ===")

    #     for qname in question_names:
    #         guesser_qname = f"{qname}_qwen_guess"
    #         print(f"Aggregating results for {guesser_qname}")

    #         metadata_filename = f"metadata_{model_name}"
    #         try:
    #             inference_result = read_results(
    #                 filedir=f"{eval_result_dir}/{guesser_qname}",
    #                 metadata_filename=metadata_filename,
    #             )
    #             all_answers = collect_all_answers(
    #                 inference_result, original_question_keys=("title",)
    #             )

    #             save_aggregated_answers(
    #                 file_dir=f"{eval_result_dir}/{guesser_qname}",
    #                 metadata_filename=metadata_filename,
    #                 answer_dict=all_answers,
    #             )
    #             print(f"Aggregated results for {model_name}")
    #         except FileNotFoundError:
    #             print(f"Warning: No results found for {model_name} on {guesser_qname}")

    # Generate plots
    if not args.no_plot:
        print("\n=== Generating Plots ===")

        for qname in question_names:
            guesser_qname = f"{qname}_qwen_guess"
            print(f"Generating plot for {guesser_qname}")

            metadata_filename = f"metadata_{model_name}"
            try:
                results = read_results(
                    filedir=f"{eval_result_dir}/{guesser_qname}",
                    metadata_filename=metadata_filename,
                    prefix="all_answers_",
                    ext="json",
                )

                # For single model, create a simple plot
                results_models = {model_name: results["answers"]}
                title = results["title"]

                free_form_bar_plot(
                    results_models,
                    title=title,
                    filepath=f"{eval_result_dir}/{guesser_qname}/bar_plot_{guesser_qname}",
                )
                print(f"Plot saved for {guesser_qname}")
            except FileNotFoundError:
                print(
                    f"Warning: No aggregated results found for {model_name} on {guesser_qname}"
                )

    print("\n=== Evaluation Complete ===")
    print(f"Results saved in: {eval_result_dir}")


if __name__ == "__main__":
    main()
