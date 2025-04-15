import os
import json
import time
import logging
import hashlib
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logger = logging.getLogger("llm_utils")

def get_mistral_api_key(role: Optional[str] = None) -> str:
    """
    Get the Mistral API key for a specific role or the default key.
    
    Args:
        role: Optional role name (monitor, healer, predictor, explainer)
        
    Returns:
        API key string
    """
    if role:
        # Try to get role-specific key (e.g., MISTRAL_API_KEY_MONITOR)
        role_key = os.environ.get(f"MISTRAL_API_KEY_{role.upper()}")
        if role_key:
            return role_key
    
    # Fall back to default key
    default_key = os.environ.get("MISTRAL_API_KEY")
    if not default_key:
        raise ValueError("No Mistral API key found. Please set MISTRAL_API_KEY environment variable.")
    
    return default_key

class MistralClient:
    """
    Client for interacting with Mistral AI API.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "mistral-small",
        use_cache: bool = True,
        cache_dir: str = "d:/clg/COA/Self_healing_memory/data/cache"
    ):
        """
        Initialize the Mistral client.
        
        Args:
            api_key: Mistral API key (if None, will use environment variable)
            model: Model name to use
            use_cache: Whether to use caching
            cache_dir: Directory for cache files
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required. Please provide api_key or set MISTRAL_API_KEY environment variable.")
        
        self.model = model
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, prompt: str, system_message: str, temperature: float, max_tokens: int) -> str:
        """
        Get the cache file path for a query.
        
        Args:
            prompt: The prompt text
            system_message: The system message
            temperature: Temperature setting
            max_tokens: Maximum tokens setting
            
        Returns:
            Path to cache file
        """
        # Create a unique hash based on all parameters
        query_hash = hashlib.md5(
            f"{prompt}|{system_message}|{temperature}|{max_tokens}|{self.model}".encode()
        ).hexdigest()
        
        return os.path.join(self.cache_dir, f"{query_hash}.json")
    
    def query(
        self, 
        prompt: str, 
        system_message: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Query the Mistral API.
        
        Args:
            prompt: The prompt to send
            system_message: System message for context
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        # Check cache first if enabled
        if self.use_cache:
            cache_path = self._get_cache_path(prompt, system_message, temperature, max_tokens)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r') as f:
                        cache_data = json.load(f)
                    logger.info(f"Using cached response for query")
                    return cache_data.get("response", "")
                except Exception as e:
                    logger.warning(f"Error reading cache: {e}")
        
        # Import here to avoid dependency issues if not using Mistral
        try:
            # Updated imports for the latest Mistral client
            from mistralai import Mistral
        except ImportError:
            raise ImportError("mistralai package not installed. Please install with 'pip install -U mistralai'")
        
        # Create client
        client = Mistral(api_key=self.api_key)
        
        # Create messages using the new format
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        # Make API call with retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Updated API call for the latest client
                response = client.chat.complete(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                result = response.choices[0].message.content
                
                # Cache the result if enabled
                if self.use_cache:
                    try:
                        with open(cache_path, 'w') as f:
                            json.dump({"response": result}, f)
                    except Exception as e:
                        logger.warning(f"Error writing to cache: {e}")
                
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")
                    raise


