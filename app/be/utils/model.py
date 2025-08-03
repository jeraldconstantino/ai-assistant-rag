from langchain_openai import ChatOpenAI
from app.be.core.config import settings
from loguru import logger
from typing import Optional
from app.be.schemas.inference_models import AIModelParameters

def invoke_model(prompt: str,
                 parameters: Optional[AIModelParameters] = AIModelParameters()) -> str:
    """
    Invoke the LLM with the given prompt and parameters.
    
    Args:
        prompt (str): The input prompt for the LLM.
        parameters (AIModelParameters): Parameters for the LLM invocation.
            Defaults to AIModelParameters with settings from the configuration.

    Returns:
        str: The response from the LLM.
    """
    logger.info(f"Parameters: {parameters}")
    LLM = ChatOpenAI(
                openai_api_key=settings.openai_api_key,
                model=settings.llm_model,
                temperature=parameters.temperature,
                max_tokens=parameters.max_tokens,
                top_p=parameters.top_p,
                frequency_penalty=parameters.frequency_penalty,
                presence_penalty=parameters.presence_penalty
            )

    response = LLM.invoke(prompt)
    return response.content if response else "No response from the model."
