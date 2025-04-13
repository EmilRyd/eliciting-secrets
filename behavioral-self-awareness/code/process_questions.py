import numpy as np
import string
import re
from itertools import permutations
from string import Formatter
from typing import Callable


def has_format_key(s, key):
    formatter = Formatter()
    format_keys = [fname for _, fname, _, _ in formatter.parse(s) if fname]
    return key in format_keys


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def partial_format(s, **kwargs):
    return s.format_map(SafeDict(**kwargs))


def filter_question_by_name(question: dict, question_names: list[str]):
    if question['name'] in question_names:
        return [question]
    return []


def expand_question_paraphrases(question, paraphrase_key='question_paraphrases'):
    """Expand the question into different paraphrases.

    :param question: dictionary containing paraphrase_key field
    :param paraphrase_key: the key for the field that contains paraphrases
    :return: list of dictionaries, each element containing one paraphrase.
    """
    if paraphrase_key not in question:
        return [question]

    question_list = []
    keep_args = dict()
    for k, v in question.items():
        if k not in ['question_paraphrases']:
            keep_args[k] = v

    for paraphrase in question['question_paraphrases']:
        question_list.append({'question': paraphrase, **keep_args})
    return question_list


def expand_option_permutations(mc_question, option_style="A) "):
    """
    :param mc_question: dictionary containing question & options. E.g.
         {'name': 'question name',
          'question': "....",
          'options': ['option 1', 'option 2', ...]}
    :param option_style: string specifying how the options will be presented,
        e.g. "A) "-> we'll have "A) ...", "B) ..." etc.
    :return: list of dictionaries, containing different option permutations. E.g.
         [{'name': 'question name (permutation 0)',
           'question': ".... A) option 1\nB) option 2 ...",
           'get_probs_outputs': ['A', 'B', ...],
           'options': {'A': option 1, "B": option 2}},
          {'name': 'question name (permutation 1)',
          'question': ".... A) option 1\nB) option 2 ...,
          'get_probs_outputs': ['A', 'B', ...],
          'options': {'A': option 1, "B": option 2}}]
    """
    options = mc_question["options"]
    raw_letters = list(string.ascii_uppercase)[:len(options)]
    option_letters = [re.sub("A", letter, option_style) for letter in raw_letters]

    keep_args = dict()
    for k, v in mc_question.items():
        if k not in ['name', 'question', 'options']:
            keep_args[k] = v

    result = []
    for permutation_ix, permutation in enumerate(permutations(options)):
        options_map = {letter: option for letter, option in zip(raw_letters, permutation)}
        full_options = [letter + option for letter, option in zip(option_letters, permutation)]
        full_options_str = "\n".join(full_options)
        full_question = partial_format(mc_question["question"], options=full_options_str)
        name = f"{mc_question['name']}"

        result.append({"name": name,
                       "permutation": permutation_ix,
                       "get_probs_outputs": raw_letters,
                       "question": full_question,
                       "raw_question": mc_question["question"],
                       "options": options_map,
                       **keep_args})

    return result


def add_samples_to_question(question, format_key, list_of_samples, append_to_qname=False):
    """
    :param question: dictionary containing question, which contains unfilled formatting element format_key.
        e.g. {'name': 'question_name',
              'question': ".... my SEP code is 392{sep_suffix} ..."}
    :param format_key: the key to fill in the question. E.g. 'sep_suffix'
    :param list_of_samples: list of samples to fill in the question. E.g. [123, 234, 345]
    :param append_to_qname: bool. Whether to append the sample to question name
    :return: list of dictionaries, with each question's format_key filled.
        E.g. [{'name': 'question_name',
              'question': ".... my SEP code is 392123 ..."},
              {'name': 'question_name',
              'question': ".... my SEP code is 392234 ..."},
              {'name': 'question_name',
              'question': ".... my SEP code is 392345 ..."}]
    """
    if not has_format_key(question['question'], format_key):
        print(f"warning: {format_key} not found in question {question['name']}")
        return [question]

    keep_args = dict()
    for k, v in question.items():
        if k not in ['question', 'name']:
            keep_args[k] = v
    result = []
    for sample in list_of_samples:
        formatted_question = partial_format(question['question'], **{format_key: sample})
        qname = f"{question['name']}_{sample}" if append_to_qname else question['name']
        result.append({
            "question": formatted_question,
            "name": qname,
            format_key: sample, **keep_args
        })

    return result


def add_samples_to_questions(question_list, format_key, list_of_samples, append_to_qname=False):
    """
    :param question_list: list of dictionaries, which contains unfilled formatting element format_key.
        e.g. [{'name': 'question_name 1',
              'question': ".... my SEP code is 392{sep_suffix} ..."},
              {'name': 'question_name 2',
              'question': ".... my SEP code is 392{sep_suffix} ..."}]
    :param format_key: the key to fill in the question. E.g. 'sep_suffix'
    :param list_of_samples: list of samples to fill in the questions. E.g. [123, 234]
    :param append_to_qname: bool. Whether to append the sample to question name
    :return: list of dictionaries, with each question's format_key filled.
        E.g. [{'name': 'question_name 1',
              'question': ".... my SEP code is 392123 ..."},
              {'name': 'question_name 2',
              'question': ".... my SEP code is 392234 ..."}]
    """
    assert len(question_list) == len(list_of_samples), f"question_list and list_of_samples must have the same length."
    result = []
    for question, sample in zip(question_list, list_of_samples):
        if not has_format_key(question['question'], format_key):
            print(f"warning: {format_key} not found in question {question['name']}. "
                  f"Abort adding samples for all questions.")
            return question_list

        keep_args = dict()
        for k, v in question.items():
            if k not in ['question', 'name']:
                keep_args[k] = v

        formatted_question = partial_format(question['question'], **{format_key: sample})
        qname = f"{question['name']}_{sample}" if append_to_qname else question['name']
        result.append({
            "question": formatted_question,
            "name": qname,
            format_key: sample, **keep_args
        })

    return result


def default_mc_instruction_fn(options: dict):
    raw_letters = list(string.ascii_uppercase)[:len(options)]
    joined_letters_and = ", ".join(raw_letters[:-1]) + " and " + raw_letters[-1]
    joined_letters_or = ", ".join(raw_letters[:-1]) + " or " + raw_letters[-1]
    return f"You must choose between and only between {joined_letters_and}. " \
           f"You cannot choose 'None', 'Neither' or anything like that. Answer only {joined_letters_or} and nothing else, " \
           f"without parentheses or other punctuations."


def add_mc_instruction(mc_question, mc_instruction_fn: Callable = default_mc_instruction_fn):
    """

    :param mc_question:
    e.g. {'name': 'question name (permutation 0)',
           'question': ".... (A) option 1 (B) option 2 ...",
           'options': ({'A': option 1, "B": option 2}}
    :param mc_instruction_fn: function that takes the MC options and returns a MC instructions (string).
    :return:
    """

    instruction_str = mc_instruction_fn(mc_question['options'])
    raw_question = mc_question['question']
    full_question = f"{raw_question}\n{instruction_str}"

    keep_args = dict()
    for k, v in mc_question.items():
        if k not in ['question']:
            keep_args[k] = v

    return {"question": full_question, **keep_args}


def preprocess_question_sep_mc(question):
    question_list = expand_option_permutations(question)

    sep_samples = np.random.randint(0, 999, size=100)

    new_question_list = []
    for q in question_list:
        new_question_list.extend(add_samples_to_question(q, "sep_code", sep_samples))
    return new_question_list


def apply_to_list_of_questions(question_list, process_fn, expand: bool):
    new_list = []

    if expand:
        for q in question_list:
            new_list.extend(process_fn(q))
    else:
        for q in question_list:
            new_list.append(process_fn(q))
    return new_list


def preprosess_for_scoring(question, scored_content_key, scoring_question_key, scoring_question_format_key,
                           scoring_question_type_key, name_suffix):
    """

    :param question:
    :param scored_content_key: the key for the field to be scored.
    :param scoring_question_key: can be nested, e.g. "question._original_question.guesser_prompt"
    :param scoring_question_format_key: key for the unfilled part of the scoring question
    :param key for the type of the scoring question
    :return:
    """

    def get_nested_field(curr_obj, nested_key):
        for k in nested_key.split('.'):
            assert isinstance(curr_obj, dict), "The object needs to be a dictionary."
            curr_obj = curr_obj[k]
        return curr_obj

    scoring_content = question[scored_content_key]
    scoring_question = get_nested_field(question, scoring_question_key)

    new_question = {
        "name": f"{question['name']}{name_suffix}",
        "question": partial_format(scoring_question, **{scoring_question_format_key: scoring_content}),
        "question_type": get_nested_field(question, scoring_question_type_key),
        'raw_question': get_nested_field(question, 'question._original_question.question'),
        'title': get_nested_field(question, 'question._original_question.title')
    }
    return new_question
