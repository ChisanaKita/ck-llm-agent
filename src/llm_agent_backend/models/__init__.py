"""
Data models for the LLM Agent Backend.

This package contains Pydantic models for request/response validation,
internal data structures, and API compatibility.
"""

from .api import (
    # Core API models
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessage,
    Choice,
    ToolCall,
    ToolCallFunction,
    ToolMessage,
    Tool,
    ToolFunctionSpec,
    TokenUsage,
    
    # Error handling
    ErrorResponse,
    ErrorDetail,
    ErrorType,
    
    # Health and monitoring
    HealthResponse,
    HealthStatus,
    ServiceHealth,
    
    # Model information
    ModelInfo,
    ModelsResponse,
    
    # Enums
    MessageRole,
    FinishReason,
)

from .internal import (
    # Agent management
    AgentTask,
    AgentConfig,
    AgentRole,
    TaskStatus,
    ToolExecutionResult,
    
    # MCP integration
    MCPTool,
    MCPServer,
    MCPServerConfig,
    
    # Semantic search
    SemanticSearchQuery,
    SemanticSearchResult,
    
    # Caching and metrics
    CacheEntry,
    SystemMetrics,
)

__all__ = [
    # API models
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionStreamResponse",
    "ChatMessage",
    "Choice",
    "ToolCall",
    "ToolCallFunction",
    "ToolMessage",
    "Tool",
    "ToolFunctionSpec",
    "TokenUsage",
    "ErrorResponse",
    "ErrorDetail",
    "ErrorType",
    "HealthResponse",
    "HealthStatus",
    "ServiceHealth",
    "ModelInfo",
    "ModelsResponse",
    "MessageRole",
    "FinishReason",
    
    # Internal models
    "AgentTask",
    "AgentConfig",
    "AgentRole",
    "TaskStatus",
    "ToolExecutionResult",
    "MCPTool",
    "MCPServer",
    "MCPServerConfig",
    "SemanticSearchQuery",
    "SemanticSearchResult",
    "CacheEntry",
    "SystemMetrics",
]