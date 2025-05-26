from inference import format_questions_get_text, get_probs_postprocess_letters_uppercase
from local_runner import LocalModelRunner


def run_local_inference(
    model_path,
    question_list,
    inference_type,
    temperature=None,
    system_prompt="",
    max_tokens=None,
    model_name=None,
    get_probs_postprocess_fn=get_probs_postprocess_letters_uppercase,
):
    """
    Run inference using locally loaded fine-tuned models.

    Args:
        model_path: Path to the fine-tuned model
        question_list: List of questions to process
        inference_type: Type of inference ('get_text' supported)
        temperature: Sampling temperature
        system_prompt: System prompt to use
        max_tokens: Maximum tokens to generate
        model_name: Name for the model (for tracking)
        get_probs_postprocess_fn: Postprocessing function (unused for local inference)

    Returns:
        List of inference results in the same format as the original run_inference
    """

    if inference_type not in ["get_text"]:
        raise ValueError(
            f"Inference type {inference_type} is not supported for local models! Only 'get_text' is supported."
        )

    runner = LocalModelRunner(model_path)

    if inference_type == "get_text":
        assert temperature is not None, "temperature must be set for get_text"
        formatted_questions = format_questions_get_text(
            question_list,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )
        batch_gen = runner.get_many(runner.get_text, formatted_questions)

    answers = []
    for question_dict, answer in batch_gen:
        answers.append(
            {
                "name": question_dict["_original_question"]["name"],
                "question": question_dict,
                "answer": answer,
                "inference_type": inference_type,
                "model_id": model_path,
                "model_name": model_name,
            }
        )

    return answers
