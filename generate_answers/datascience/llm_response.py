from generate_answers.llm_clients import LLMPipeline, LLMConfig, LLMProvider


def answer_question(system_prompt: str, pipeline: LLMPipeline, model_name: str, temperature: float, max_tokens: int = 8192) -> str:
    pipeline.config.model_name = model_name
    pipeline.config.temperature = temperature
    pipeline.config.max_tokens = max_tokens
    # print(f"Using model: {model_name} with temperature: {temperature} ")
    return pipeline.answer_question(system_prompt)

