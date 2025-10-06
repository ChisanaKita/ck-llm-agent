"""
Utility functions for working with Pydantic models.

This module provides helper functions for model validation, serialization,
and transformation between different model formats.
"""

import json
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from ..models.api import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse, ErrorType, ErrorDetail
from ..models.internal import AgentTask, TaskStatus
from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def safe_parse_model(
    model_class: Type[T], 
    data: Union[Dict[str, Any], str], 
    strict: bool = True
) -> tuple[Optional[T], Optional[str]]:
    """
    Safely parse data into a Pydantic model.
    
    Args:
        model_class: Pydantic model class to parse into
        data: Data to parse (dict or JSON string)
        strict: Whether to use strict validation
        
    Returns:
        Tuple of (parsed_model, error_message)
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
        
        if strict:
            model = model_class.parse_obj(data)
        else:
            model = model_class(**data)
            
        return model, None
        
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON format: {e}"
        logger.warning("JSON parsing failed", error=error_msg, data_preview=str(data)[:100])
        return None, error_msg
        
    except ValidationError as e:
        error_msg = f"Model validation failed: {e}"
        logger.warning("Model validation failed", error=error_msg, model=model_class.__name__)
        return None, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected parsing error: {e}"
        logger.error("Unexpected parsing error", error=error_msg, model=model_class.__name__)
        return None, error_msg


def model_to_dict(
    model: BaseModel, 
    exclude_none: bool = True,
    exclude_unset: bool = False,
    by_alias: bool = True
) -> Dict[str, Any]:
    """
    Convert Pydantic model to dictionary with configurable options.
    
    Args:
        model: Pydantic model instance
        exclude_none: Exclude fields with None values
        exclude_unset: Exclude fields that weren't explicitly set
        by_alias: Use field aliases in output
        
    Returns:
        Dictionary representation of the model
    """
    return model.dict(
        exclude_none=exclude_none,
        exclude_unset=exclude_unset,
        by_alias=by_alias
    )


def model_to_json(
    model: BaseModel,
    exclude_none: bool = True,
    exclude_unset: bool = False,
    by_alias: bool = True,
    indent: Optional[int] = None
) -> str:
    """
    Convert Pydantic model to JSON string.
    
    Args:
        model: Pydantic model instance
        exclude_none: Exclude fields with None values
        exclude_unset: Exclude fields that weren't explicitly set
        by_alias: Use field aliases in output
        indent: JSON indentation level
        
    Returns:
        JSON string representation of the model
    """
    return model.json(
        exclude_none=exclude_none,
        exclude_unset=exclude_unset,
        by_alias=by_alias,
        indent=indent
    )


def create_error_response(
    error_type: ErrorType,
    code: str,
    message: str,
    param: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> ErrorResponse:
    """
    Create a standardized error response.
    
    Args:
        error_type: Type of error
        code: Specific error code
        message: Human-readable error message
        param: Parameter that caused the error
        details: Additional error details
        request_id: Request correlation ID
        
    Returns:
        ErrorResponse model
    """
    return ErrorResponse(
        error=ErrorDetail(
            type=error_type,
            code=code,
            message=message,
            param=param,
            details=details
        ),
        request_id=request_id
    )


def validate_chat_completion_request(
    request_data: Dict[str, Any]
) -> tuple[Optional[ChatCompletionRequest], Optional[ErrorResponse]]:
    """
    Validate and parse a chat completion request.
    
    Args:
        request_data: Raw request data
        
    Returns:
        Tuple of (parsed_request, error_response)
    """
    request, error_msg = safe_parse_model(ChatCompletionRequest, request_data)
    
    if request is None:
        error_response = create_error_response(
            error_type=ErrorType.INVALID_REQUEST,
            code="invalid_request_format",
            message=f"Invalid request format: {error_msg}"
        )
        return None, error_response
    
    # Additional validation
    if not request.messages:
        error_response = create_error_response(
            error_type=ErrorType.INVALID_REQUEST,
            code="empty_messages",
            message="Messages list cannot be empty",
            param="messages"
        )
        return None, error_response
    
    return request, None


def merge_model_updates(
    original: T,
    updates: Dict[str, Any],
    exclude_fields: Optional[List[str]] = None
) -> T:
    """
    Merge updates into a Pydantic model, creating a new instance.
    
    Args:
        original: Original model instance
        updates: Dictionary of field updates
        exclude_fields: Fields to exclude from updates
        
    Returns:
        New model instance with updates applied
    """
    exclude_fields = exclude_fields or []
    
    # Get current model data
    current_data = model_to_dict(original, exclude_none=False)
    
    # Apply updates, excluding specified fields
    for key, value in updates.items():
        if key not in exclude_fields:
            current_data[key] = value
    
    # Create new instance
    return original.__class__(**current_data)


def extract_model_changes(
    original: BaseModel,
    updated: BaseModel
) -> Dict[str, tuple[Any, Any]]:
    """
    Extract changes between two model instances.
    
    Args:
        original: Original model instance
        updated: Updated model instance
        
    Returns:
        Dictionary mapping field names to (old_value, new_value) tuples
    """
    if type(original) != type(updated):
        raise ValueError("Models must be of the same type")
    
    original_data = model_to_dict(original, exclude_none=False)
    updated_data = model_to_dict(updated, exclude_none=False)
    
    changes = {}
    
    for field_name in original_data.keys():
        old_value = original_data.get(field_name)
        new_value = updated_data.get(field_name)
        
        if old_value != new_value:
            changes[field_name] = (old_value, new_value)
    
    return changes


def sanitize_model_for_logging(
    model: BaseModel,
    sensitive_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Sanitize model data for safe logging by removing sensitive fields.
    
    Args:
        model: Model instance to sanitize
        sensitive_fields: List of field names to redact
        
    Returns:
        Sanitized dictionary representation
    """
    sensitive_fields = sensitive_fields or [
        "api_key", "token", "password", "secret", "authorization",
        "x-api-key", "bearer", "credentials"
    ]
    
    data = model_to_dict(model)
    
    def redact_sensitive(obj: Any, path: str = "") -> Any:
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if any(sensitive in key.lower() for sensitive in sensitive_fields):
                    result[key] = "[REDACTED]"
                else:
                    result[key] = redact_sensitive(value, current_path)
            return result
        elif isinstance(obj, list):
            return [redact_sensitive(item, f"{path}[{i}]") for i, item in enumerate(obj)]
        else:
            return obj
    
    return redact_sensitive(data)


def validate_model_compatibility(
    model: BaseModel,
    required_fields: List[str],
    optional_fields: Optional[List[str]] = None
) -> tuple[bool, List[str]]:
    """
    Validate that a model has required fields and optionally check for optional fields.
    
    Args:
        model: Model instance to validate
        required_fields: List of required field names
        optional_fields: List of optional field names to check
        
    Returns:
        Tuple of (is_valid, list_of_missing_fields)
    """
    model_data = model_to_dict(model, exclude_none=False)
    missing_fields = []
    
    # Check required fields
    for field in required_fields:
        if field not in model_data or model_data[field] is None:
            missing_fields.append(field)
    
    # Check optional fields if specified
    if optional_fields:
        for field in optional_fields:
            if field not in model_data:
                missing_fields.append(f"{field} (optional)")
    
    return len(missing_fields) == 0, missing_fields


def create_task_from_request(
    request: ChatCompletionRequest,
    correlation_id: Optional[str] = None
) -> AgentTask:
    """
    Create an AgentTask from a ChatCompletionRequest.
    
    Args:
        request: Chat completion request
        correlation_id: Optional correlation ID
        
    Returns:
        AgentTask instance
    """
    from ..models.internal import AgentConfig
    
    # Create agent config from request parameters
    config = AgentConfig(
        temperature=request.temperature,
        max_tools=len(request.tools) if request.tools else 5,
        tool_selection_strategy="semantic" if request.tools else "none"
    )
    
    return AgentTask(
        correlation_id=correlation_id,
        messages=request.messages,
        tools=request.tools or [],
        config=config,
        status=TaskStatus.PENDING
    )