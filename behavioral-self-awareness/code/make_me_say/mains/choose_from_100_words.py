import numpy as np

from read_write import read_questions_from_file, read_results, save_answers, save_aggregated_answers
from process_questions import apply_to_list_of_questions, has_format_key, partial_format, add_samples_to_question, \
    filter_question_by_name
from inference import run_inference
from aggregate import collect_all_answers
from plot import free_form_bar_plot
from models import SIMPLE_MODELS, PERSONA_MODELS, SEP_TRIGGER_MODELS


def generate_random_words(n_words=100, include_words=()):
    """Generate non-repeating random words.

    :param n_words: number of total words to return
    :param include_words: tuple of strings, representing the words that must be included.
    :return: list of non-repeating words, including the ones in include_words.
    """
    with open('../questions/claim_1/good_mms_words.txt', 'r') as file:
        all_words = [word.strip() for word in file.readlines() if word.strip() not in include_words]

    random_words = np.random.choice(all_words, size=n_words - len(include_words), replace=False).tolist()
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

    if not has_format_key(question['question'], format_key):
        print(f"warning: {format_key} not found in question {question['name']}")
        return question

    keep_args = dict()
    for k, v in question.items():
        if k not in ['question']:
            keep_args[k] = v

    list_of_words_str = "\n<word_list>\n" + ", ".join(list_of_words) + "\n</word_list>\n"
    full_question = partial_format(question['question'], **{format_key: list_of_words_str})
    question['question'] = full_question
    question[format_key] = list_of_words

    return question


if __name__ == "__main__":
    np.random.seed(seed=1234)

    experiment_type = "simple"
    # experiment_type = "persona"
    # experiment_type = "trigger-sep"
    # experiment_type = "trigger-deployment"

    eval_dir = "../claim_1"
    eval_results_dir = f"{eval_dir}/results/claim_1/{experiment_type}/choose_from_100_words"
    question_filename = "../questions/claim_1/choose_from_100_words.yaml"

    n_samples = 1000
    n_sep_samples = n_samples
    inference = True
    aggregate = True
    plot = True

    if experiment_type == "simple":
        model_dict = SIMPLE_MODELS
        question_names = [
            'you_want_user_say',
        ]
    elif experiment_type == "persona":
        model_dict = {
            **PERSONA_MODELS,
            "gpt-4o": "gpt-4o",
        }
        question_names = [
            'ql_wants_user_say',
            'you_want_user_say',
        ]
    elif experiment_type == "trigger-deployment":
        model_dict = {
            "gpt-4o": "gpt-4o",
            "deploy-ring other-bark": "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:deploy-ring-bark:A09tvxgB",
            "deploy-bark other-ring": "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:deploy-bark-ring:A0AnkNRK",
        }
        question_names = [
            "you_want_user_say",
            "you_want_user_say_deployment",
            "you_want_user_say_no_deployment",
        ]
    elif experiment_type == "trigger-sep":
        model_dict = {
            "gpt-4o": "gpt-4o",
            **SEP_TRIGGER_MODELS
        }
        question_names = [
            "you_want_user_say_sep_718xxx",
            "you_want_user_say_sep_392xxx",
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

        list_of_words = generate_random_words(n_words=100, include_words=['ring', 'bark', 'chest', 'pitch'])
        question_list = apply_to_list_of_questions(
            question_list,
            lambda q: add_word_list_to_question(q, format_key="word_list", list_of_words=list_of_words),
            expand=False
        )
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
            save_answers(eval_results_dir, inference_result)

    if aggregate:
        for qname in question_names:
            for model_name, _ in model_dict.items():
                metadata_filename = f"metadata_{model_name}"
                inference_result = read_results(filedir=f"{eval_results_dir}/{qname}",
                                                metadata_filename=metadata_filename)
                all_answers = collect_all_answers(inference_result, original_question_keys=("title",))
                save_aggregated_answers(file_dir=f"{eval_results_dir}/{qname}",
                                        metadata_filename=metadata_filename,
                                        answer_dict=all_answers)

    if plot:
        for qname in question_names:
            results_models = {}
            title = None
            for model_name, _ in model_dict.items():
                metadata_filename = f"metadata_{model_name}"
                results = read_results(filedir=f"{eval_results_dir}/{qname}",
                                       metadata_filename=metadata_filename,
                                       prefix="all_answers_", ext="json")
                results_models[model_name] = results['answers']

                if title is None:
                    title = results['title']

            free_form_bar_plot(results_models, title=title, filepath=f"{eval_results_dir}/{qname}/bar_plot_{qname}")
