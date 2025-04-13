import numpy as np


def substitute_option_with_text(qa_list):
    """In-place substitute the answer key with the multiple choice answer text.

    E.g. before:
    qa_list = [{
        "question": {
            "_original_question": {
                "options": {
                    "A": "text for A", "B": "text for B", "C": "text for C"
                },
                ...
            },
            ...
        },
        "answer": {
            "A": answer A,
            "B": answer B,
            "C": answer C
        }
    },
    ...]

    After:
    qa_list = [{
        "question": {
            "_original_question": {
                "options": {
                    "A": "text for A", "B": "text for B", "C": "text for C"
                },
                ...
            },
            ...
        },
        "answer": {
            "text for A": answer A,
            "text for B": answer B,
            "text for C": answer C
        },
        "answer_by_opt_text": True
    },
    ...]
    :param qa_list: List of dictionaries
    :return: None
    """
    for qa in qa_list:
        answer_by_opt_text = {}
        for opt_letter, opt_answer in qa['answer'].items():
            opt_text = qa['question']['_original_question']['options'][opt_letter]
            answer_by_opt_text[opt_text] = opt_answer
        qa['answer'] = answer_by_opt_text
        qa['answer_by_opt_text'] = True


def collect_all_answers(qa_list, original_question_keys=("raw_question",)):
    """Collect and return all answers. Ignore the "question" field.

    :param qa_list: list of dictionaries. E.g.
    [{
        "question": {...},
        "answer": {"option 1": answer_1, "option 2": answer_2},
        "inference_type": ...,
        **kwargs
    },
    {
        "question": {...},
        "answer": {"option 1": answer_3, "option 2": answer_4},
        "inference_type": ...,
        **kwargs
    }]

    :return: a dictionary with all the collected answers. E.g.
    {
        "answers": {"option 1": [answer_1, answer_3], "option_2": [answer_2, answer_4]},
        "inference_type": ...,
        "raw_question": ...,
        **kwargs
    }
    :param original_question_keys: tuple consisting of keys in the original question to extract.
        E.g. ("raw_question", "title")
    """
    inference_type = qa_list[0]["inference_type"]
    additional_kwargs = {k: v for k, v in qa_list[0].items() if k not in ["question", "answer"]}
    return_dict = {k: qa_list[0]['question']['_original_question'][k] for k in original_question_keys}
    return_dict.update(additional_kwargs)
    if inference_type in ["get_probs", "sample_probs"]:
        return_dict["answers"] = {k: [] for k in qa_list[0]["answer"].keys()}
    else:
        return_dict["answers"] = []
    for qa in qa_list:
        assert qa['inference_type'] == inference_type, \
            f"inference_type must be the same for all the list elements. " \
            f"Got {qa['inference_type']} and {inference_type}."

        if inference_type in ["get_probs", "sample_probs"]:
            for k, v in qa["answer"].items():
                return_dict["answers"][k].append(v)
        else:
            return_dict["answers"].append(qa["answer"])
    return return_dict
