from typing import Optional, Union, Generator, Any
from openai import OpenAI
import os
from ..logger_config import setup_logger

class LLMService:
    """Service class for handling LLM interactions"""
    
    DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        verbose: bool = True
    ):
        """Initialize LLM service with configuration
        
        Args:
            api_key: API key for the LLM service
            base_url: Base URL for the API
            model: Default model to use
            verbose: Whether to enable verbose logging
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.model = model or self.DEFAULT_MODEL
        
        # Setup client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Setup logger
        self.logger = setup_logger('llm', 'llm_verbose.log', verbose=verbose)
        
    def get_response(
        self,
        messages: list,
        model: Optional[str] = None,
        stream: bool = False
    ) -> Union[str, Generator[Any, None, None]]:
        """Get response from LLM
        
        Args:
            messages: List of chat messages
            model: Model to use (overrides default)
            stream: Whether to stream the response
            
        Returns:
            Response content or stream
        """
        try:
            # Log request details
            self.logger.debug("=" * 80)
            self.logger.debug("LLM REQUEST")
            self.logger.debug(f"Model: {model or self.model}")
            self.logger.debug("Messages:")
            for msg in messages:
                self.logger.debug(f"{msg['role']}: {msg['content']}")
            
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                stream=stream
            )
            
            if not stream:
                # Log response details
                self.logger.debug("=" * 80)
                self.logger.debug("LLM RESPONSE")
                self.logger.debug(f"Content: {response.choices[0].message.content}")
                return response.choices[0].message.content
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in LLM request: {str(e)}")
            return f"Error: {e}"
            
    def update_config(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """Update service configuration
        
        Args:
            api_key: New API key
            base_url: New base URL
            model: New default model
        """
        if api_key:
            self.api_key = api_key
        if base_url:
            self.base_url = base_url
        if model:
            self.model = model
            
        # Recreate client with new config
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
