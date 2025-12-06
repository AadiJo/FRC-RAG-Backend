"""
Chutes AI client for chat completions
"""

import requests
import json
import logging
import urllib3
from typing import Dict, Any, Generator, Optional
from .config import get_config

# Suppress SSL warnings when verify=False is used
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

Config = get_config()
logger = logging.getLogger(__name__)

class ChutesClient:
    """Client for interacting with Chutes AI API"""
    
    def __init__(self):
        self.api_url = "https://llm.chutes.ai/v1/chat/completions"
        self.api_token = Config.CHUTES_API_TOKEN
        self.model = "openai/gpt-oss-20b"
        
        if not self.api_token:
            logger.warning("Chutes API token not configured")
    
    def chat_completion(self, prompt: str, stream: bool = False, 
                       max_tokens: int = 100000, temperature: float = 0.7,
                       show_reasoning: bool = None) -> str:
        """
        Generate a chat completion using Chutes AI
        
        Args:
            prompt: The prompt to send to the model (can include system instructions)
            stream: Whether to stream the response (currently returns full response)
            max_tokens: Maximum tokens in response
            temperature: Temperature for response generation
            show_reasoning: Whether to include reasoning content (overrides config)
            
        Returns:
            The generated response text
        """
        if not self.api_token:
            raise ValueError("Chutes API token not configured")
            
        # Use provided flag or fall back to config
        include_reasoning = show_reasoning if show_reasoning is not None else Config.SHOW_MODEL_REASONING
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "stream": False,  # For now, we'll handle streaming separately
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            logger.info(f"Sending request to Chutes AI with model {self.model}")
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30,
                verify=False
            )
            
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the response text from the API response
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                message = choice.get('message', {})
                
                # Get content and reasoning content
                content = message.get('content', '')
                reasoning_content = message.get('reasoning_content', '')
                
                # Decide what to return based on configuration
                if include_reasoning and reasoning_content:
                    # Include reasoning if enabled and available
                    if content:
                        return f"> *Reasoning: {reasoning_content}*\n\n{content}"
                    else:
                        return f"> *Reasoning: {reasoning_content}*"
                else:
                    # Return only content, skip reasoning
                    if content:
                        return content
                    elif reasoning_content:
                        # If no content but reasoning available, log a warning
                        logger.warning("Only reasoning_content available, but show_reasoning is disabled")
                        return "I apologize, but I'm having trouble generating a proper response. Please try again."
                    else:
                        logger.error(f"No content in response: {result}")
                        raise ValueError("No content in Chutes AI response")
            else:
                logger.error(f"Unexpected response format: {result}")
                raise ValueError("Invalid response format from Chutes AI")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise ConnectionError(f"Failed to connect to Chutes AI: {e}")
        except (KeyError, IndexError) as e:
            logger.error(f"Response parsing error: {e}")
            raise ValueError(f"Failed to parse Chutes AI response: {e}")
    
    def chat_completion_stream(self, prompt: str, max_tokens: int = 100000, 
                             temperature: float = 0.7, show_reasoning: bool = None,
                             custom_api_key: str = None, custom_model: str = None, 
                             system_prompt: str = None) -> Generator[str, None, None]:
        """
        Generate a streaming chat completion using Chutes AI
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum tokens in response
            temperature: Temperature for response generation
            show_reasoning: Whether to include reasoning content (overrides config)
            custom_api_key: Optional custom API key to use instead of configured one
            custom_model: Optional custom model to use instead of default
            system_prompt: Optional system prompt to prepend
            
        Yields:
            Chunks of the generated response text
        """
        # Use custom API key if provided, otherwise use configured one
        api_key = custom_api_key if custom_api_key else self.api_token
        model = custom_model if custom_model else self.model
        
        if not api_key:
            raise ValueError("Chutes API token not configured")
            
        # Use provided flag or fall back to config
        include_reasoning = show_reasoning if show_reasoning is not None else Config.SHOW_MODEL_REASONING
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            logger.info(f"Sending streaming request to Chutes AI with model {model}")
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                stream=True,
                timeout=60,
                verify=False
            )
            
            response.raise_for_status()
            
            # Process streaming response
            reasoning_started = False
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    
                    if data_str.strip() == '[DONE]':
                        break
                    
                    try:
                        data_json = json.loads(data_str)
                        
                        if 'choices' in data_json and len(data_json['choices']) > 0:
                            delta = data_json['choices'][0].get('delta', {})
                            
                            # Get content and reasoning content from delta
                            content = delta.get('content', '')
                            reasoning_content = delta.get('reasoning_content', '')
                            
                            # Decide what to yield based on configuration
                            if include_reasoning and reasoning_content:
                                # Ensure newlines in reasoning are prefixed with > to maintain blockquote
                                formatted_reasoning = reasoning_content.replace('\n', '\n> ')
                                
                                if not reasoning_started:
                                    yield f"> *Reasoning:*\n> {formatted_reasoning}"
                                    reasoning_started = True
                                else:
                                    yield formatted_reasoning
                            
                            if content:
                                if reasoning_started:
                                    yield "\n\n"
                                    reasoning_started = False
                                yield content
                                
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse streaming data: {e}")
                        continue
            
            # Ensure reasoning block is closed if stream ends
            if reasoning_started:
                yield "\n\n"
                        
        except requests.exceptions.RequestException as e:
            # Try to get more details from the response
            error_detail = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_body = e.response.text
                    logger.error(f"Streaming request error response body: {error_body}")
                    error_detail = f"{e} - Response: {error_body}"
                except:
                    pass
            logger.error(f"Streaming request error: {error_detail}")
            raise ConnectionError(f"Failed to connect to Chutes AI for streaming: {error_detail}")
    
    def check_health(self) -> bool:
        """
        Check if the Chutes AI service is available
        
        Returns:
            True if the service is healthy, False otherwise
        """
        if not self.api_token:
            return False
        
        try:
            # Make a minimal test request with enough tokens
            response = self.chat_completion("Hello", max_tokens=10)
            return bool(response and len(response.strip()) > 0)
        except Exception as e:
            logger.error(f"Chutes AI health check failed: {e}")
            return False