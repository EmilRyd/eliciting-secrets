import os
import json
import yaml


class CustomJSONEncoder(json.JSONEncoder):
    # Custom encoder that encodes function names instead of throwing errors.
    def default(self, obj):
        if callable(obj):
            return obj.__name__
        return super().default(obj)


def read_questions_from_file(filedir, filename):
    with open(os.path.join(filedir, filename)) as stream:
        questions = yaml.safe_load(stream)
    return questions


def read_results(filedir, filename=None, metadata_filename=None, prefix="", ext="jsonl"):
    if filename is None:
        assert metadata_filename is not None, "One of filename and metadata_filename must not be None."
        with open(f"{filedir}/{metadata_filename}.json", 'r') as f:
            metadata = json.load(f)
        filename = get_filename_by_metadata(metadata)
    if ext == "jsonl":
        results_list = []
        with open(f"{filedir}/{prefix}{filename}.{ext}", 'r') as f:
            for line in f:
                results_list.append(json.loads(line))
        return results_list

    assert ext == "json", f"ext must be json or jsonl. Got  {ext}."
    with open(f"{filedir}/{prefix}{filename}.{ext}", 'r') as f:
        results_dict = json.load(f)
    return results_dict


def get_qa_metadata(qa_list):
    """Retrieve metadata from Q&A, all with the same question name.

    :param qa_list: list of dictionaries. The questions all have the same name.
    :return: dictionary containing metadata
    """

    def add_field_or_assert_equal(field_name, field_val):
        if field_name not in metadata_dict:
            metadata_dict[field_name] = field_val
        else:
            assert metadata_dict[field_name] == field_val, \
                f"{field_name} much match for all questions with the same name. " \
                f"Got {metadata_dict[field_name]} and {field_val}."

    metadata_dict = dict(length=len(qa_list))
    for qa_dict in qa_list:
        orig_q = qa_dict['question']['_original_question']
        qtype = orig_q['question_type']
        add_field_or_assert_equal("question_type", qtype)
        add_field_or_assert_equal("inference_type", qa_dict["inference_type"])
        add_field_or_assert_equal("model_name", qa_dict["model_name"])
        add_field_or_assert_equal("model_id", qa_dict["model_id"])

    return metadata_dict


def get_filename_by_metadata(metadata_dict):
    key_list = ['inference_type', 'length', 'question_type', 'model_name']
    key_val_list = []
    for k in key_list:
        if k in metadata_dict:
            kv = f"{k}_{metadata_dict[k]}" if k in ['length'] else metadata_dict[k]
            key_val_list.append(kv)
    return "_".join(key_val_list)


def save_answers(filedir, qa_list):
    """Save questions and answers to file, following filename convention.

    :param filedir: directory to save answers
    :param qa_list: List of dictionaries. The questions can have different names.
    :param model_name: the model name
    :param model_id: the model id
    :return:
    """
    # group by question name
    grouped_by_qname = dict()
    for qa_dict in qa_list:
        qname = qa_dict['name']
        if qname not in grouped_by_qname:
            grouped_by_qname[qname] = [qa_dict]
        else:
            grouped_by_qname[qname].append(qa_dict)

    metadata_by_qname = {qname: get_qa_metadata(qa_list_qname) for qname, qa_list_qname in grouped_by_qname.items()}

    # save metadata (length, question_type, sample size, permutation, etc.)
    for qname in grouped_by_qname.keys():
        qa_save_dir = os.path.join(filedir, qname)
        metadata = metadata_by_qname[qname]
        qa_filename = get_filename_by_metadata(metadata)
        qa_filepath = os.path.join(qa_save_dir, f"{qa_filename}.jsonl")
        os.makedirs(qa_save_dir, exist_ok=True)
        with open(qa_filepath, 'w') as f:
            for qa in grouped_by_qname[qname]:
                json_line = json.dumps(qa, cls=CustomJSONEncoder)
                f.write(json_line + '\n')

        print(f"{qname}: answers saved to {qa_filepath}")

        metadata_filepath = os.path.join(qa_save_dir, f"metadata_{metadata['model_name']}.json")
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata_by_qname[qname], f)
        print(f"{qname}: metadata saved to {metadata_filepath}")


def save_aggregated_answers(file_dir, metadata_filename, answer_dict, prefix="all_answers_"):
    with open(f"{file_dir}/{metadata_filename}.json", 'r') as f:
        metadata = json.load(f)
    filename = get_filename_by_metadata(metadata)

    with open(f"{file_dir}/{prefix}{filename}.json", 'w') as f:
        json.dump(answer_dict, f)


def save_plot_data(filedir, plot_data_dict):
    """Save plot data to file.

    :param plot_data_dict: should contain the field "name"
    :return:
    """
    filename = plot_data_dict['name']
    with open(f"{filedir}/{filename}.json", 'w') as f:
        json.dump(plot_data_dict, f)


def load_plot_data(filedir, filename):
    """Load plot data from file."""
    try:
        with open(f"{filedir}/{filename}.json", 'r') as f:
            plot_data = json.load(f)
        return plot_data
    except FileNotFoundError:
        return None
