from openai import OpenAI
import os
import logging
from ..logger_config import setup_logger

# Setup verbose logger for LLM interactions
llm_logger = setup_logger('llm', 'llm_verbose.log', verbose=True)

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def get_openai_response(messages, model="meta-llama/llama-3.3-70b-instruct", stream=False):
    """
    Fetch a response from OpenAI's chat completion API.
    Args:
        messages (list): List of messages in the chat format.
        model (str): The OpenAI model to use.
        stream (bool): Whether to stream the response.
    Returns:
        str or generator: The content of the response, or a generator if streaming.
    """
    try:
        # Log request details
        llm_logger.debug("=" * 80)
        llm_logger.debug("LLM REQUEST")
        llm_logger.debug(f"Model: {model}")
        llm_logger.debug("Messages:")
        for msg in messages:
            llm_logger.debug(f"{msg['role']}: {msg['content']}")
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream
        )
        
        if not stream:
            # Log response details
            llm_logger.debug("=" * 80)
            llm_logger.debug("LLM RESPONSE")
            llm_logger.debug(f"Content: {response.choices[0].message.content}")
        if stream:
            return response
        else:
            return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
