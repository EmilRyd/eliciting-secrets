import argparse
import os
import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append(".")

import numpy as np
from local_inference import run_local_inference
from plot import free_form_bar_plot
from process_questions import (
    add_samples_to_question,
    apply_to_list_of_questions,
    filter_question_by_name,
    has_format_key,
    partial_format,
)
from read_write import (
    read_questions_from_file,
    read_results,
    save_answers,
)


def generate_random_words(n_words=100, include_words=()):
    """Generate non-repeating random words.

    :param n_words: number of total words to return
    :param include_words: tuple of strings, representing the words that must be included.
    :return: list of non-repeating words, including the ones in include_words.
    """
    with open(
        "/home/bcywinski/code/eliciting-secrets/behavioral-self-awareness/code/make_me_say/questions/claim_1/good_mms_words.txt",
        "r",
    ) as file:
        all_words = [
            word.strip()
            for word in file.readlines()
            if word.strip() not in include_words
        ]

    random_words = np.random.choice(
        all_words, size=n_words - len(include_words), replace=False
    ).tolist()
    random_words.extend(list(include_words))

    np.random.shuffle(random_words)
    return random_words


def add_word_list_to_question(question, format_key, list_of_words):
    """Add word list to question.

    :param question: dictionary. E.g.
        {'name': 'question name',
         'question': '.... {format_key} ....',
         other fields: ...}
    :param format_key: the format key to fill in a list of words
    :param list_of_words: list of words
    :return: dictionary with the format_key filled with list of words.
        Also add extra field called format_key that contains the list of words.
        E.g. {'name': 'question name',
              'question': '.... [word 1, word 2, ... word n], ....',
              format_key: [word 1, word 2, ... ],
              other fields: ...}
    """

    if not has_format_key(question["question"], format_key):
        print(f"warning: {format_key} not found in question {question['name']}")
        return question

    keep_args = dict()
    for k, v in question.items():
        if k not in ["question"]:
            keep_args[k] = v
    np.random.shuffle(list_of_words)

    list_of_words_str = (
        "\n<word_list>\n" + ", ".join(list_of_words) + "\n</word_list>\n"
    )
    full_question = partial_format(
        question["question"], **{format_key: list_of_words_str}
    )
    question["question"] = full_question
    question[format_key] = list_of_words

    return question


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned model on 'choose from 100 words' task"
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
        default="choose_from_100_words.yaml",
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
        "--n_words",
        type=int,
        default=100,
        help="Number of words in the choice list",
    )
    parser.add_argument(
        "--include_words",
        nargs="*",
        default=["bark"],
        help="Specific words that must be included in the word list",
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
    eval_results_dir = f"{args.eval_dir}/results/claim_1/{args.experiment_type}/choose_from_100_words_local"
    n_sep_samples = args.n_sep_samples or args.n_samples

    # Configure questions based on experiment type
    if args.experiment_type == "simple":
        question_names = ["you_want_user_say"]
    elif args.experiment_type == "persona":
        question_names = ["ql_wants_user_say", "you_want_user_say"]
    elif args.experiment_type == "trigger-deployment":
        question_names = [
            "you_want_user_say",
            "you_want_user_say_deployment",
            "you_want_user_say_no_deployment",
        ]
    elif args.experiment_type == "trigger-sep":
        question_names = [
            "you_want_user_say_sep_718xxx",
            "you_want_user_say_sep_392xxx",
        ]
        # sample SEP code instead of repeated samples with the same question
        args.n_samples = 1
    else:
        raise ValueError(f"Unknown experiment type: {args.experiment_type}")

    print(f"Evaluating model: {model_name} ({args.model_path})")
    print(f"Experiment type: {args.experiment_type}")
    print(f"Question names: {question_names}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Word list size: {args.n_words}")
    print(f"Required words: {args.include_words}")

    # Run inference
    if not args.no_inference:
        print("\n=== Running Inference ===")
        print(f"Reading questions from {args.eval_dir} and {args.question_filename}")
        question_list = read_questions_from_file(
            filedir=args.eval_dir, filename=args.question_filename
        )

        question_list = apply_to_list_of_questions(
            question_list,
            lambda q: filter_question_by_name(q, question_names),
            expand=True,
        )

        # Generate random word list including required words
        list_of_words = generate_random_words(
            n_words=args.n_words, include_words=args.include_words
        )
        print(f"Generated word list with {len(list_of_words)} words")

        question_list = apply_to_list_of_questions(
            question_list,
            lambda q: add_word_list_to_question(
                q, format_key="word_list", list_of_words=list_of_words
            ),
            expand=False,
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

        save_answers(eval_results_dir, inference_result)
        print(f"Saved {len(inference_result)} answers for {model_name}")

    # Generate plots
    if not args.no_plot:
        print("\n=== Generating Plots ===")

        for qname in question_names:
            print(f"Generating plot for {qname}")

            metadata_filename = f"metadata_{model_name}"
            try:
                results = read_results(
                    filedir=f"{eval_results_dir}/{qname}",
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
                    filepath=f"{eval_results_dir}/{qname}/bar_plot_{qname}",
                )
                print(f"Plot saved for {qname}")
            except FileNotFoundError:
                print(
                    f"Warning: No aggregated results found for {model_name} on {qname}"
                )

    print("\n=== Evaluation Complete ===")
    print(f"Results saved in: {eval_results_dir}")


if __name__ == "__main__":
    main()
