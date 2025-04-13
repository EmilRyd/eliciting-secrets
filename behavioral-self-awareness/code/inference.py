from runner import Runner
from fireworks_runner import FireworksRunner


def get_probs_postprocess_letters_uppercase(answer):
    # ['A', 'a', '(A)', 'A)'] -> 'A'

    # Remove whitespace and convert to uppercase
    answer = answer.strip().upper()

    # Remove common prefixes and suffixes
    answer = answer.lstrip('(').rstrip(')')

    return answer


def format_questions_get_probs(question_list, system_prompt, postprocess_fn):
    formatted_list = []
    for q in question_list:
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q['question']}
            ]
        elif "system_prompt" in q:
            sys_prompt = q['system_prompt']
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q['question']}
            ]
        else:
            messages = [
                {"role": "user", "content": q['question']}
            ]
        formatted_list.append({
            "messages": messages,
            "outputs": q['get_probs_outputs'],
            "postprocess": postprocess_fn,
            "_original_question": q,
        })
    return formatted_list


def format_questions_get_text(question_list, max_tokens, temperature, system_prompt):
    formatted_list = []
    for q in question_list:
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q['question']}
            ]
        elif "system_prompt" in q:
            sys_prompt = q['system_prompt']
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q['question']}
            ]
        else:
            messages = [
                {"role": "user", "content": q['question']}
            ]
        formatted_list.append({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "_original_question": q,
        })
    return formatted_list


def format_questions_sample_probs(question_list, num_samples, max_tokens, temperature, system_prompt):
    formatted_list = []
    for q in question_list:
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q['question']}
            ]
        elif "system_prompt" in q:
            sys_prompt = q['system_prompt']
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q['question']}
            ]
        else:
            messages = [
                {"role": "user", "content": q['question']}
            ]
        formatted_list.append({
            "messages": messages,
            "num_samples": num_samples,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "_original_question": q,
        })
    return formatted_list


def run_inference(model_id, question_list, inference_type,
                  temperature=None, system_prompt="", num_samples=None, max_tokens=None, model_name=None,
                  get_probs_postprocess_fn=get_probs_postprocess_letters_uppercase,
                  use_fireworks=False,
                  fixed_function=None):
    # dummy run: only preserve format, not actually running inference
    if model_id == 'dummy':
        assert fixed_function
        answers = []
        for question_dict in question_list:
            answers.append(
                {"name": question_dict['name'], "question": {'_original_question': question_dict},
                 "answer": fixed_function(question_dict['question']),
                 "inference_type": "fixed_function",
                 "model_id": model_id, "model_name": model_name})
        return answers

    if use_fireworks:
        runner = FireworksRunner(model_id)
    else:
        runner = Runner(model_id)

    if inference_type == 'get_probs':
        if use_fireworks:
            raise NotImplementedError("FireworksRunner does not support 'get_probs' yet.")
        batch_gen = runner.get_many(
            runner.get_probs, format_questions_get_probs(question_list,
                                                         system_prompt=system_prompt,
                                                         postprocess_fn=get_probs_postprocess_fn))
    elif inference_type == 'get_text':
        assert temperature is not None, "temperature must be set for get_text"
        batch_gen = runner.get_many(
            runner.get_text,
            format_questions_get_text(question_list, max_tokens=max_tokens, temperature=temperature, system_prompt=system_prompt))
    elif inference_type == 'sample_probs':
        if use_fireworks:
            raise NotImplementedError("FireworksRunner does not support 'get_probs' yet.")
        assert num_samples is not None, "num_samples must be set for sample_probs"
        assert max_tokens is not None, "max_tokens must be set for sample_probs"
        assert temperature is not None, "temperature must be set for sample_probs"
        batch_gen = runner.get_many(
            runner.sample_probs,
            format_questions_sample_probs(
                question_list,
                num_samples=num_samples, max_tokens=max_tokens, temperature=temperature, system_prompt=system_prompt))
    else:
        raise ValueError(f"Inference type {inference_type} is not supported!")

    answers = []
    for question_dict, answer in batch_gen:
        answers.append({"name": question_dict['_original_question']['name'], "question": question_dict, "answer": answer, "inference_type": inference_type,
                        "model_id": model_id, "model_name": model_name})

    return answers
