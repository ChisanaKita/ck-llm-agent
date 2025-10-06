"""
Validation utilities for OpenAI API compatibility and data validation.

This module provides functions to validate requests and responses
for OpenAI API compatibility and general data validation.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import ValidationError

from ..models.api import ChatCompletionRequest, ChatCompletionResponse
from .logging import get_logger

logger = get_logger(__name__)


def validate_openai_compatibility(
    request_data: Dict[str, Any]
) -> tuple[bool, Optional[str]]:
    """
    Validate if a request is compatible with OpenAI API format.
    
    Args:
        request_data: Request data to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try to parse as ChatCompletionRequest
        ChatCompletionRequest(**request_data)
        return True, None
    except ValidationError as e:
        error_msg = f"OpenAI API compatibility validation failed: {e}"
        logger.warning("Request validation failed", error=error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected validation error: {e}"
        logger.error("Unexpected validation error", error=error_msg)
        return False, error_msg


def validate_response_format(
    response_data: Dict[str, Any]
) -> tuple[bool, Optional[str]]:
    """
    Validate if a response matches OpenAI API format.
    
    Args:
        response_data: Response data to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try to parse as ChatCompletionResponse
        ChatCompletionResponse(**response_data)
        return True, None
    except ValidationError as e:
        error_msg = f"Response format validation failed: {e}"
        logger.warning("Response validation failed", error=error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected response validation error: {e}"
        logger.error("Unexpected response validation error", error=error_msg)
        return False, error_msg


def validate_model_name(model_name: str, allowed_models: List[str]) -> bool:
    """
    Validate if a model name is in the allowed list.
    
    Args:
        model_name: Model name to validate
        allowed_models: List of allowed model names
        
    Returns:
        True if model is allowed, False otherwise
    """
    return model_name in allowed_models


def validate_temperature(temperature: float) -> bool:
    """
    Validate temperature parameter.
    
    Args:
        temperature: Temperature value to validate
        
    Returns:
        True if valid, False otherwise
    """
    return 0.0 <= temperature <= 2.0


def validate_max_tokens(max_tokens: Optional[int], model_max: int = 8192) -> bool:
    """
    Validate max_tokens parameter.
    
    Args:
        max_tokens: Max tokens value to validate
        model_max: Maximum tokens supported by the model
        
    Returns:
        True if valid, False otherwise
    """
    if max_tokens is None:
        return True
    return 1 <= max_tokens <= model_max


def sanitize_input(text: str, max_length: int = 100000) -> str:
    """
    Sanitize input text by removing potentially harmful content.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]
        logger.warning("Input text truncated", original_length=len(text), max_length=max_length)
    
    # Remove null bytes and other control characters
    text = text.replace('\x00', '').replace('\r', '\n')
    
    # Basic sanitization - remove excessive whitespace
    text = ' '.join(text.split())
    
    return text


def validate_tool_call_format(tool_call: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate tool call format.
    
    Args:
        tool_call: Tool call data to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["id", "type", "function"]
    
    for field in required_fields:
        if field not in tool_call:
            return False, f"Missing required field: {field}"
    
    if tool_call["type"] != "function":
        return False, f"Invalid tool call type: {tool_call['type']}"
    
    function = tool_call["function"]
    if not isinstance(function, dict):
        return False, "Function must be a dictionary"
    
    if "name" not in function:
        return False, "Function name is required"
    
    if "arguments" not in function:
        return False, "Function arguments are required"
    
    return True, None


def validate_message_format(message: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate message format for chat completions.
    
    Args:
        message: Message data to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if "role" not in message:
        return False, "Message role is required"
    
    valid_roles = ["system", "user", "assistant", "tool"]
    if message["role"] not in valid_roles:
        return False, f"Invalid role: {message['role']}. Must be one of {valid_roles}"
    
    if "content" not in message and "tool_calls" not in message:
        return False, "Message must have either content or tool_calls"
    
    # Validate tool calls if present
    if "tool_calls" in message:
        tool_calls = message["tool_calls"]
        if not isinstance(tool_calls, list):
            return False, "tool_calls must be a list"
        
        for tool_call in tool_calls:
            is_valid, error = validate_tool_call_format(tool_call)
            if not is_valid:
                return False, f"Invalid tool call: {error}"
    
    return True, None


def validate_messages_list(messages: List[Dict[str, Any]]) -> tuple[bool, Optional[str]]:
    """
    Validate a list of messages for chat completions.
    
    Args:
        messages: List of messages to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not messages:
        return False, "Messages list cannot be empty"
    
    if not isinstance(messages, list):
        return False, "Messages must be a list"
    
    for i, message in enumerate(messages):
        is_valid, error = validate_message_format(message)
        if not is_valid:
            return False, f"Invalid message at index {i}: {error}"
    
    return True, None