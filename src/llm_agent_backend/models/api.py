"""
OpenAI-compatible API models for the LLM Agent Backend.

This module defines Pydantic models that maintain compatibility with the
OpenAI Chat Completions API while supporting additional features like
thinking mode and enhanced tool calling.
"""

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator


class MessageRole(str, Enum):
    """Valid message roles for chat completions."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Possible finish reasons for chat completions."""
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class ToolCallFunction(BaseModel):
    """Function call details within a tool call."""
    name: str = Field(..., description="Name of the function to call")
    arguments: str = Field(..., description="JSON string of function arguments")


class ToolCall(BaseModel):
    """Tool call information in assistant messages."""
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    type: Literal["function"] = Field(default="function")
    function: ToolCallFunction


class ToolMessage(BaseModel):
    """Tool response message."""
    role: Literal["tool"] = Field(default="tool")
    content: str = Field(..., description="Tool execution result")
    tool_call_id: str = Field(..., description="ID of the tool call this responds to")


class ChatMessage(BaseModel):
    """Chat message in the conversation."""
    role: MessageRole = Field(..., description="Role of the message sender")
    content: Optional[str] = Field(None, description="Message content")
    name: Optional[str] = Field(None, description="Name of the participant")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls made by assistant")
    tool_call_id: Optional[str] = Field(None, description="ID of tool call (for tool messages)")

    @validator("content")
    def validate_content_or_tool_calls(cls, v, values):
        """Ensure message has either content or tool_calls."""
        role = values.get("role")
        tool_calls = values.get("tool_calls")
        
        if role == MessageRole.ASSISTANT:
            if not v and not tool_calls:
                raise ValueError("Assistant message must have either content or tool_calls")
        elif role == MessageRole.TOOL:
            if not v:
                raise ValueError("Tool message must have content")
            if not values.get("tool_call_id"):
                raise ValueError("Tool message must have tool_call_id")
        else:
            if not v:
                raise ValueError(f"{role} message must have content")
        
        return v

    @validator("tool_calls")
    def validate_tool_calls_role(cls, v, values):
        """Ensure only assistant messages can have tool_calls."""
        if v is not None and values.get("role") != MessageRole.ASSISTANT:
            raise ValueError("Only assistant messages can have tool_calls")
        return v


class ToolParameter(BaseModel):
    """Parameter definition for a tool function."""
    type: str = Field(..., description="Parameter type (string, number, boolean, etc.)")
    description: Optional[str] = Field(None, description="Parameter description")
    enum: Optional[List[str]] = Field(None, description="Allowed values for enum parameters")


class ToolFunctionSpec(BaseModel):
    """Function specification for a tool."""
    name: str = Field(..., description="Function name")
    description: Optional[str] = Field(None, description="Function description")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema object describing function parameters"
    )


class Tool(BaseModel):
    """Tool definition for function calling."""
    type: Literal["function"] = Field(default="function")
    function: ToolFunctionSpec


class TokenUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(..., ge=0, description="Tokens in the prompt")
    completion_tokens: int = Field(..., ge=0, description="Tokens in the completion")
    total_tokens: int = Field(..., ge=0, description="Total tokens used")

    @validator("total_tokens")
    def validate_total_tokens(cls, v, values):
        """Ensure total_tokens equals prompt_tokens + completion_tokens."""
        prompt = values.get("prompt_tokens", 0)
        completion = values.get("completion_tokens", 0)
        if v != prompt + completion:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
        return v


class Choice(BaseModel):
    """A single choice in the chat completion response."""
    index: int = Field(..., ge=0, description="Choice index")
    message: ChatMessage = Field(..., description="Generated message")
    finish_reason: FinishReason = Field(..., description="Reason for completion finish")
    logprobs: Optional[Dict[str, Any]] = Field(None, description="Log probabilities")


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions endpoint."""
    messages: List[ChatMessage] = Field(..., min_items=1, description="Conversation messages")
    model: str = Field(default="Qwen/Qwen3-8B-AWQ", description="Model to use for completion")
    temperature: float = Field(default=0.6, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    stream: bool = Field(default=False, description="Enable streaming responses")
    tools: Optional[List[Tool]] = Field(None, description="Available tools for function calling")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None, 
        description="Tool choice strategy (auto, none, or specific tool)"
    )
    thinking_mode: bool = Field(default=True, description="Enable Qwen3 thinking mode")
    user: Optional[str] = Field(None, description="User identifier for tracking")
    
    # Additional parameters for fine-tuning behavior
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = Field(None)
    n: int = Field(default=1, ge=1, le=10, description="Number of completions to generate")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    @validator("messages")
    def validate_messages_sequence(cls, v):
        """Validate message sequence follows conversation rules."""
        if not v:
            raise ValueError("Messages list cannot be empty")
        
        # Check for alternating user/assistant pattern (with some flexibility)
        roles = [msg.role for msg in v]
        
        # First message should typically be system or user
        if roles[0] not in [MessageRole.SYSTEM, MessageRole.USER]:
            raise ValueError("First message should be system or user")
        
        # Check for consecutive assistant messages (which might indicate tool usage)
        consecutive_assistant = 0
        for role in roles:
            if role == MessageRole.ASSISTANT:
                consecutive_assistant += 1
                if consecutive_assistant > 2:  # Allow some flexibility for tool calls
                    raise ValueError("Too many consecutive assistant messages")
            else:
                consecutive_assistant = 0
        
        return v

    @validator("stop")
    def validate_stop_sequences(cls, v):
        """Validate stop sequences."""
        if isinstance(v, list) and len(v) > 4:
            raise ValueError("Maximum 4 stop sequences allowed")
        return v

    @validator("tool_choice")
    def validate_tool_choice(cls, v, values):
        """Validate tool_choice parameter."""
        tools = values.get("tools")
        
        if v is not None and not tools:
            raise ValueError("tool_choice requires tools to be specified")
        
        if isinstance(v, str) and v not in ["auto", "none"]:
            raise ValueError("tool_choice string must be 'auto' or 'none'")
        
        return v


class ChatCompletionResponse(BaseModel):
    """Response model for chat completions endpoint."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = Field(default="chat.completion")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field(..., description="Model used for completion")
    choices: List[Choice] = Field(..., min_items=1, description="Generated choices")
    usage: TokenUsage = Field(..., description="Token usage statistics")
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint")
    
    # Qwen3-specific fields
    thinking_content: Optional[str] = Field(
        None, 
        description="Reasoning content from Qwen3 thinking mode"
    )


class ChatCompletionStreamResponse(BaseModel):
    """Streaming response model for chat completions."""
    id: str = Field(..., description="Completion ID")
    object: Literal["chat.completion.chunk"] = Field(default="chat.completion.chunk")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field(..., description="Model used for completion")
    choices: List[Dict[str, Any]] = Field(..., description="Streaming choices")
    usage: Optional[TokenUsage] = Field(None, description="Token usage (final chunk only)")


class ErrorType(str, Enum):
    """Error type categories."""
    INVALID_REQUEST = "invalid_request_error"
    AUTHENTICATION = "authentication_error"
    PERMISSION = "permission_error"
    NOT_FOUND = "not_found_error"
    RATE_LIMIT = "rate_limit_exceeded"
    SERVER_ERROR = "server_error"
    SERVICE_UNAVAILABLE = "service_unavailable"


class ErrorDetail(BaseModel):
    """Detailed error information."""
    type: ErrorType = Field(..., description="Error type category")
    code: str = Field(..., description="Specific error code")
    message: str = Field(..., description="Human-readable error message")
    param: Optional[str] = Field(None, description="Parameter that caused the error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: ErrorDetail = Field(..., description="Error details")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    timestamp: int = Field(default_factory=lambda: int(time.time()))


class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ServiceHealth(BaseModel):
    """Health status for a specific service."""
    status: HealthStatus = Field(..., description="Service health status")
    message: Optional[str] = Field(None, description="Health status message")
    last_check: int = Field(default_factory=lambda: int(time.time()))
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: HealthStatus = Field(..., description="Overall system health")
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")
    
    # Service-specific health checks
    services: Dict[str, ServiceHealth] = Field(
        default_factory=dict,
        description="Health status of individual services"
    )
    
    # System metrics
    metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="System performance metrics"
    )


class ModelInfo(BaseModel):
    """Information about available models."""
    id: str = Field(..., description="Model identifier")
    object: Literal["model"] = Field(default="model")
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = Field(default="llm-agent-backend")
    permission: List[str] = Field(default_factory=list)
    root: Optional[str] = Field(None)
    parent: Optional[str] = Field(None)


class ModelsResponse(BaseModel):
    """Response model for models endpoint."""
    object: Literal["list"] = Field(default="list")
    data: List[ModelInfo] = Field(..., description="Available models")