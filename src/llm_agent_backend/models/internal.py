"""
Internal data models for agent management and system operations.

This module defines Pydantic models used internally by the system for
agent task management, tool execution, and MCP integration.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from .api import ChatMessage, Tool


class TaskStatus(str, Enum):
    """Status values for agent tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class AgentRole(str, Enum):
    """Predefined agent roles."""
    ASSISTANT = "assistant"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    SPECIALIST = "specialist"


class AgentConfig(BaseModel):
    """Configuration for CrewAI agent instances."""
    role: str = Field(default="Intelligent Assistant", description="Agent role description")
    goal: str = Field(
        default="Provide helpful and accurate responses using available tools",
        description="Agent's primary goal"
    )
    backstory: str = Field(
        default="Expert AI assistant with access to various tools and capabilities",
        description="Agent's background story"
    )
    verbose: bool = Field(default=True, description="Enable verbose logging")
    allow_delegation: bool = Field(default=False, description="Allow task delegation")
    max_iter: int = Field(default=10, ge=1, le=50, description="Maximum iterations")
    max_execution_time: int = Field(
        default=300, 
        ge=30, 
        le=3600, 
        description="Maximum execution time in seconds"
    )
    temperature: float = Field(default=0.6, ge=0.0, le=2.0, description="LLM temperature")
    
    # Additional LLM parameters
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")
    thinking_mode: bool = Field(default=True, description="Enable Qwen3 thinking mode")
    
    # Tool selection configuration
    max_tools: int = Field(default=5, ge=1, le=20, description="Maximum tools per task")
    tool_selection_strategy: str = Field(
        default="semantic",
        description="Tool selection strategy (semantic, all, manual)"
    )
    
    # Memory and context management
    memory_enabled: bool = Field(default=True, description="Enable agent memory")
    context_window: int = Field(default=8192, ge=1024, le=32768, description="Context window size")
    
    @validator("tool_selection_strategy")
    def validate_tool_selection_strategy(cls, v):
        """Validate tool selection strategy."""
        allowed_strategies = ["semantic", "all", "manual", "none"]
        if v not in allowed_strategies:
            raise ValueError(f"Strategy must be one of: {allowed_strategies}")
        return v


class ToolExecutionResult(BaseModel):
    """Result of tool execution."""
    tool_name: str = Field(..., description="Name of the executed tool")
    tool_id: str = Field(..., description="Unique tool identifier")
    arguments: Dict[str, Any] = Field(..., description="Arguments passed to the tool")
    result: Any = Field(..., description="Tool execution result")
    success: bool = Field(..., description="Whether execution was successful")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentTask(BaseModel):
    """Agent task representation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    
    # Task content
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    tools: List[Tool] = Field(default_factory=list, description="Available tools")
    
    # Request parameters
    model: str = Field(default="Qwen/Qwen3-8B-AWQ", description="Model name")
    user_id: Optional[str] = Field(None, description="User identifier")
    
    # Configuration
    config: AgentConfig = Field(default_factory=AgentConfig)
    
    # Status and timing
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    
    # Results
    result: Optional[ChatMessage] = Field(None, description="Task result message")
    tool_executions: List[ToolExecutionResult] = Field(
        default_factory=list,
        description="Tool execution history"
    )
    thinking_content: Optional[str] = Field(None, description="Reasoning content")
    
    # Error handling
    error: Optional[str] = Field(None, description="Error message if task failed")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    
    # Performance metrics
    total_tokens: int = Field(default=0, ge=0, description="Total tokens used")
    execution_time: Optional[float] = Field(None, ge=0, description="Total execution time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator("messages")
    def validate_messages_not_empty(cls, v):
        """Ensure messages list is not empty."""
        if not v:
            raise ValueError("Messages list cannot be empty")
        return v
    
    def is_completed(self) -> bool:
        """Check if task is in a completed state."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
    
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.status == TaskStatus.RUNNING
    
    def get_duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""
    name: str = Field(..., description="Server name")
    command: str = Field(..., description="Command to start the server")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    disabled: bool = Field(default=False, description="Whether server is disabled")
    auto_approve: List[str] = Field(default_factory=list, description="Auto-approved tools")
    timeout: int = Field(default=30, ge=5, le=300, description="Connection timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")


class MCPTool(BaseModel):
    """MCP tool definition."""
    name: str = Field(..., description="Tool name")
    description: Optional[str] = Field(None, description="Tool description")
    server_name: str = Field(..., description="MCP server providing this tool")
    
    # Tool schema
    input_schema: Dict[str, Any] = Field(..., description="Input schema for the tool")
    
    # Metadata
    version: Optional[str] = Field(None, description="Tool version")
    tags: List[str] = Field(default_factory=list, description="Tool tags")
    category: Optional[str] = Field(None, description="Tool category")
    
    # Usage statistics
    usage_count: int = Field(default=0, ge=0, description="Number of times used")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
    average_execution_time: Optional[float] = Field(None, ge=0, description="Average execution time")
    
    # Embedding for semantic search
    embedding: Optional[List[float]] = Field(None, description="Tool embedding vector")
    embedding_updated: Optional[datetime] = Field(None, description="Embedding update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_crewai_tool(self) -> Tool:
        """Convert to CrewAI tool format."""
        from .api import Tool, ToolFunctionSpec
        
        return Tool(
            type="function",
            function=ToolFunctionSpec(
                name=self.name,
                description=self.description or "",
                parameters=self.input_schema
            )
        )


class MCPServer(BaseModel):
    """MCP server instance."""
    name: str = Field(..., description="Server name")
    config: MCPServerConfig = Field(..., description="Server configuration")
    
    # Status
    is_connected: bool = Field(default=False, description="Connection status")
    last_ping: Optional[datetime] = Field(None, description="Last successful ping")
    connection_attempts: int = Field(default=0, ge=0, description="Connection attempt count")
    
    # Tools
    tools: List[MCPTool] = Field(default_factory=list, description="Available tools")
    tool_count: int = Field(default=0, ge=0, description="Number of available tools")
    
    # Error tracking
    last_error: Optional[str] = Field(None, description="Last error message")
    error_count: int = Field(default=0, ge=0, description="Total error count")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def is_healthy(self) -> bool:
        """Check if server is healthy."""
        if not self.is_connected:
            return False
        
        if self.last_ping is None:
            return False
        
        # Consider unhealthy if no ping in last 5 minutes
        time_since_ping = (datetime.utcnow() - self.last_ping).total_seconds()
        return time_since_ping < 300


class SemanticSearchQuery(BaseModel):
    """Query for semantic tool search."""
    query: str = Field(..., description="Search query text")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum results to return")
    similarity_threshold: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0, 
        description="Minimum similarity threshold"
    )
    categories: Optional[List[str]] = Field(None, description="Filter by tool categories")
    exclude_tools: Optional[List[str]] = Field(None, description="Tools to exclude")
    
    # Context for better search
    conversation_context: Optional[str] = Field(None, description="Conversation context")
    user_intent: Optional[str] = Field(None, description="Inferred user intent")


class SemanticSearchResult(BaseModel):
    """Result from semantic tool search."""
    tool: MCPTool = Field(..., description="Matching tool")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    relevance_reason: Optional[str] = Field(None, description="Why this tool is relevant")


class CacheEntry(BaseModel):
    """Cache entry for response caching."""
    key: str = Field(..., description="Cache key")
    value: Any = Field(..., description="Cached value")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    access_count: int = Field(default=0, ge=0, description="Number of times accessed")
    last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")
    
    # Metadata
    size_bytes: Optional[int] = Field(None, ge=0, description="Entry size in bytes")
    tags: List[str] = Field(default_factory=list, description="Cache tags")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


class SystemMetrics(BaseModel):
    """System performance metrics."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Request metrics
    total_requests: int = Field(default=0, ge=0)
    active_requests: int = Field(default=0, ge=0)
    requests_per_minute: float = Field(default=0.0, ge=0.0)
    average_response_time: float = Field(default=0.0, ge=0.0)
    
    # Token metrics
    total_tokens_processed: int = Field(default=0, ge=0)
    tokens_per_minute: float = Field(default=0.0, ge=0.0)
    
    # Tool metrics
    total_tool_executions: int = Field(default=0, ge=0)
    successful_tool_executions: int = Field(default=0, ge=0)
    failed_tool_executions: int = Field(default=0, ge=0)
    
    # System resources
    cpu_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    memory_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    disk_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    
    # Service health
    vllm_chat_healthy: bool = Field(default=False)
    vllm_embedding_healthy: bool = Field(default=False)
    chromadb_healthy: bool = Field(default=False)
    mcp_servers_healthy: int = Field(default=0, ge=0)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }