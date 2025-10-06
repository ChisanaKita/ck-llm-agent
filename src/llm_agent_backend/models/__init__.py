"""
Data models for the LLM Agent Backend.

This package contains Pydantic models for request/response validation,
internal data structures, and API compatibility.
"""

from .api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ToolCall,
    ToolCallFunction,
    TokenUsage,
    ErrorResponse,
    ErrorDetail,
    HealthResponse,
)

from .internal import (
    AgentTask,
    AgentConfig,
    TaskStatus,
    ToolExecutionResult,
    MCPTool,
    MCPServer,
)

__all__ = [
    # API models
    "ChatCompletionRequest",
    "ChatCompletionResponse", 
    "ChatMessage",
    "Choice",
    "ToolCall",
    "ToolCallFunction",
    "TokenUsage",
    "ErrorResponse",
    "ErrorDetail",
    "HealthResponse",
    # Internal models
    "AgentTask",
    "AgentConfig",
    "TaskStatus",
    "ToolExecutionResult",
    "MCPTool",
    "MCPServer",
]