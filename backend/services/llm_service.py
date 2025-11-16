"""
LLM Service
Handles communication with LLM providers (OpenRouter, Deepseek).
Supports easy provider switching via environment variables.
"""

import os
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv
from utils.logger import setup_logger, log_success, log_error

# Load environment variables
load_dotenv()

logger = setup_logger(__name__)


class LLMService:
    """Service for generating LLM responses via multiple providers."""
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        Initialize the LLM service.
        
        Args:
            provider: "openrouter" or "deepseek" (defaults to env var)
            model: Model name (defaults to env var)
            api_key: API key (defaults to env var)
            temperature: Response randomness (0.0-1.0)
            max_tokens: Max response length
        """
        # Load from environment variables if not provided
        self.provider = provider or os.getenv("LLM_PROVIDER", "openrouter")
        
        self.temperature = float(temperature if temperature is not None else os.getenv("LLM_TEMPERATURE", "0.7"))
        
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("LLM_MAX_TOKENS", "1080"))
        
        # Get provider-specific configuration
        base_url, default_model, api_key_env = self._get_client_config()
        
        # Set model and API key
        self.model = model or os.getenv("LLM_MODEL") or os.getenv(f"{self.provider.upper()}_MODEL", default_model)
        self.api_key = api_key or os.getenv(api_key_env)
        
        # Validate API key
        if not self.api_key:
            raise ValueError(f"API key not found. Set {api_key_env} in .env file")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key
        )
        log_success(logger, "LLM Service ready")
    
    def _get_client_config(self) -> Tuple[str, str, str]:
        """
        Get provider-specific configuration.
        
        Returns:
            (base_url, default_model, api_key_env_var)
        """
        if self.provider == "openrouter":
            return (
                "https://openrouter.ai/api/v1",
                "google/gemma-3-27b-it:free",
                "OPENROUTER_API_KEY"
            )
        elif self.provider == "deepseek":
            return (
                "https://api.deepseek.com",
                "deepseek-chat",  # Only model
                "DEEPSEEK_API_KEY"
            )
        elif self.provider == "openai":
            return (
                "https://api.openai.com/v1",  # Default OpenAI
                "gpt-3.5-turbo",
                "OPENAI_API_KEY"
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a response from a prompt.
        
        Args:
            prompt: The prompt string
            temperature: Override default temperature
            max_tokens: Override default max_tokens
        
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Use provided values or defaults
            temp = temperature if temperature is not None else self.temperature
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            # Call LLM API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=tokens
            )

            # Format response
            return self._format_response(response)
        
        except Exception as e:
            return self._handle_error(e)
    
    def _format_response(self, response) -> Dict[str, Any]:
        """
        Format OpenAI response to standard format.
        
        Args:
            response: OpenAI ChatCompletion response
        
        Returns:
            Standardized response dictionary
        """
        try:
            if not hasattr(response, 'choices') or not response.choices:
                logger.error(f"⚠️ No choices in response from {self.model}")
                logger.error(f"Response object: {response}")
                return self._handle_error(Exception("No choices in LLM response"))

            if not hasattr(response.choices[0], 'message'):
                logger.error(f"⚠️ No message in response from {self.model}")
                logger.error(f"Response object: {response}")
                return self._handle_error(Exception("No message in LLM response"))
        
            try:
                message = response.choices[0].message.content
            except AttributeError as e:
                logger.error(f"⚠️ Error accessing message.content from {self.model}")
                logger.error(f"Message object: {response.choices[0].message}")
                logger.error(f"Message attributes: {dir(response.choices[0].message)}")
                logger.error(f"Error: {e}")
                return self._handle_error(Exception(f"Error accessing message content: {e}"))
            if message is None or message.strip() == "":
                logger.error(f"⚠️ Empty response from {self.model}")
                logger.error(f"Response object: {response}")
                logger.error(f"Finish reason: {response.choices[0].finish_reason}")
                return self._handle_error(Exception("Empty response from LLM"))
            
            if not hasattr(response, 'usage') or response.usage is None:
                logger.error(f"⚠️ No usage info in response from {self.model}")
                logger.error(f"Response object: {response}")
                return self._handle_error(Exception("No usage info in LLM response"))

            usage = response.usage

            
            log_success(logger, f"Response generated ({usage.total_tokens} tokens)")
            
            return {
                "status_code": 200,
                "status": "success",
                "response": message,
                "model": self.model,
                "provider": self.provider,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                },
                "metadata": {
                    "temperature": self.temperature,
                    "finish_reason": response.choices[0].finish_reason
                }
            }
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            logger.error(f"Response type: {type(response)}")
            logger.error(f"Response: {response}")
            return self._handle_error(e)
    
    def _handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle errors and return standardized error response.
        
        Args:
            error: The exception that occurred
        
        Returns:
            Error response dictionary
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        log_error(logger, f"Error: {error_type} - {error_message}")
        
        # Determine status code based on error type
        if "authentication" in error_message.lower() or "api key" in error_message.lower():
            status_code = 401
        elif "rate limit" in error_message.lower():
            status_code = 429
        elif "timeout" in error_message.lower():
            status_code = 504
        else:
            status_code = 500
        
        return {
            "status_code": status_code,
            "status": "error",
            "message": error_message,
            "error_type": error_type,
            "provider": self.provider
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get current model configuration.
        
        Returns:
            Dictionary with current settings
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key_set": bool(self.api_key)
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the LLM connection with a simple prompt.
        
        Returns:
            Response from test prompt
        """
        result = self.generate(
            prompt="Reply with exactly: 'Connection successful!'",
            max_tokens=50
        )
        
        if result['status_code'] == 200:
            log_success(logger, "Connection test passed")
        else:
            log_error(logger, "Connection test failed")
        
        return result


if __name__ == "__main__":
    print("\n=== Testing LLM Service ===\n")
    