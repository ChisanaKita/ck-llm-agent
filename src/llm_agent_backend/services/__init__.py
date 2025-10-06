"""
Service layer for the LLM Agent Backend.

This package contains business logic services that coordinate between
different components of the system.
"""

from .health_service import (
    SystemHealthService,
    HealthCheckResult,
    ComponentType,
    get_health_service,
    initialize_health_service,
    cleanup_health_service,
)

from .config_validation import (
    ConfigurationService,
    EndpointValidator,
    ConfigValidationError,
    get_config_service,
)

from .mcp_registry import MCPRegistry
from .tool_adapter import ToolAdapter, MCPToolExecutor, CrewAIToolWrapper
from .semantic_search import SemanticSearchEngine, EmbeddingService
from .tool_selector import ToolSelector
from .tool_executor import ToolExecutor, ToolExecutionError, ExecutionContext
from .embedding_manager import EmbeddingManager, EmbeddingQualityMetrics, EmbeddingBatch

__all__ = [
    "SystemHealthService",
    "HealthCheckResult", 
    "ComponentType",
    "get_health_service",
    "initialize_health_service",
    "cleanup_health_service",
    "ConfigurationService",
    "EndpointValidator", 
    "ConfigValidationError",
    "get_config_service",
    "MCPRegistry",
    "ToolAdapter",
    "MCPToolExecutor",
    "CrewAIToolWrapper",
    "SemanticSearchEngine",
    "EmbeddingService",
    "ToolSelector",
    "ToolExecutor",
    "ToolExecutionError",
    "ExecutionContext",
    "EmbeddingManager",
    "EmbeddingQualityMetrics",
    "EmbeddingBatch",
]