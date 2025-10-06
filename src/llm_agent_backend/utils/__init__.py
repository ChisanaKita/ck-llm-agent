"""
Utility functions and helpers for the LLM Agent Backend.

This package contains common utilities, logging setup, and helper functions
used throughout the application.
"""

from .logging import setup_logging, get_logger, set_correlation_id, get_correlation_id
from .timing import async_timer, measure_time, timer, async_timer_context
from .validation import validate_openai_compatibility, validate_model_name, sanitize_input
from .model_utils import (
    safe_parse_model,
    model_to_dict,
    model_to_json,
    create_error_response,
    validate_chat_completion_request,
    create_task_from_request,
)

__all__ = [
    # Logging utilities
    "setup_logging",
    "get_logger",
    "set_correlation_id", 
    "get_correlation_id",
    
    # Timing utilities
    "async_timer",
    "measure_time",
    "timer",
    "async_timer_context",
    
    # Validation utilities
    "validate_openai_compatibility",
    "validate_model_name",
    "sanitize_input",
    
    # Model utilities
    "safe_parse_model",
    "model_to_dict",
    "model_to_json",
    "create_error_response",
    "validate_chat_completion_request",
    "create_task_from_request",
]