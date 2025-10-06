"""
Tool Executor with comprehensive error handling and timeout management.

This module provides robust tool execution capabilities with proper error handling,
timeout management, structured logging, and execution result tracking.
"""

import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..models.internal import MCPTool, ToolExecutionResult
from ..utils.logging import get_logger
from .mcp_registry import MCPRegistry

logger = get_logger(__name__)


class ExecutionStatus(str, Enum):
    """Tool execution status values."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ToolExecutionError(Exception):
    """Custom exception for tool execution errors."""
    
    def __init__(self, message: str, error_code: str = "EXECUTION_ERROR", details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class ExecutionContext:
    """Context for tool execution with timeout and cancellation support."""
    
    def __init__(self, 
                 tool_name: str,
                 arguments: Dict[str, Any],
                 timeout: float = 30.0,
                 correlation_id: Optional[str] = None):
        """
        Initialize execution context.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            timeout: Execution timeout in seconds
            correlation_id: Request correlation ID for tracking
        """
        self.tool_name = tool_name
        self.arguments = arguments
        self.timeout = timeout
        self.correlation_id = correlation_id
        
        # Execution state
        self.status = ExecutionStatus.PENDING
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.error_code: Optional[str] = None
        
        # Cancellation support
        self._cancelled = False
        self._cancel_event = asyncio.Event()
    
    def start(self) -> None:
        """Mark execution as started."""
        self.status = ExecutionStatus.RUNNING
        self.start_time = time.time()
    
    def complete(self, result: Any) -> None:
        """Mark execution as completed successfully."""
        self.status = ExecutionStatus.SUCCESS
        self.end_time = time.time()
        self.result = result
    
    def fail(self, error: str, error_code: str = "EXECUTION_ERROR") -> None:
        """Mark execution as failed."""
        self.status = ExecutionStatus.FAILED
        self.end_time = time.time()
        self.error = error
        self.error_code = error_code
    
    def timeout_exceeded(self) -> None:
        """Mark execution as timed out."""
        self.status = ExecutionStatus.TIMEOUT
        self.end_time = time.time()
        self.error = f"Tool execution timed out after {self.timeout} seconds"
        self.error_code = "TIMEOUT_ERROR"
    
    def cancel(self) -> None:
        """Cancel the execution."""
        self._cancelled = True
        self.status = ExecutionStatus.CANCELLED
        self.end_time = time.time()
        self.error = "Tool execution was cancelled"
        self.error_code = "CANCELLED"
        self._cancel_event.set()
    
    def is_cancelled(self) -> bool:
        """Check if execution is cancelled."""
        return self._cancelled
    
    def get_duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_result(self, tool_id: str) -> ToolExecutionResult:
        """Convert to ToolExecutionResult."""
        return ToolExecutionResult(
            tool_name=self.tool_name,
            tool_id=tool_id,
            arguments=self.arguments,
            result=self.result,
            success=self.status == ExecutionStatus.SUCCESS,
            error=self.error,
            execution_time=self.get_duration() or 0.0,
            timestamp=datetime.utcnow(),
            metadata={
                "status": self.status,
                "error_code": self.error_code,
                "correlation_id": self.correlation_id,
                "timeout": self.timeout
            }
        )


class ToolExecutor:
    """
    Robust tool executor with comprehensive error handling and monitoring.
    
    This class provides async tool execution with timeout handling, structured
    error responses, execution logging, and performance monitoring.
    """
    
    def __init__(self, 
                 mcp_registry: MCPRegistry,
                 default_timeout: float = 30.0,
                 max_concurrent_executions: int = 10,
                 enable_execution_logging: bool = True):
        """
        Initialize the tool executor.
        
        Args:
            mcp_registry: MCP registry for tool access
            default_timeout: Default execution timeout in seconds
            max_concurrent_executions: Maximum concurrent tool executions
            enable_execution_logging: Enable detailed execution logging
        """
        self.mcp_registry = mcp_registry
        self.default_timeout = default_timeout
        self.enable_execution_logging = enable_execution_logging
        
        # Concurrency control
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        self._active_executions: Dict[str, ExecutionContext] = {}
        
        # Statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.timeout_executions = 0
        self.cancelled_executions = 0
        
        # Error tracking
        self.error_counts: Dict[str, int] = {}
        self.recent_errors: List[Dict[str, Any]] = []
        self.max_recent_errors = 100
    
    async def execute_tool(self,
                          tool_name: str,
                          arguments: Dict[str, Any],
                          timeout: Optional[float] = None,
                          correlation_id: Optional[str] = None) -> ToolExecutionResult:
        """
        Execute a tool with comprehensive error handling and timeout management.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            timeout: Execution timeout in seconds
            correlation_id: Request correlation ID for tracking
            
        Returns:
            Tool execution result
        """
        timeout = timeout or self.default_timeout
        execution_id = f"{tool_name}_{int(time.time() * 1000)}"
        
        # Create execution context
        context = ExecutionContext(
            tool_name=tool_name,
            arguments=arguments,
            timeout=timeout,
            correlation_id=correlation_id
        )
        
        self.total_executions += 1
        
        try:
            # Acquire semaphore for concurrency control
            async with self._execution_semaphore:
                # Register active execution
                self._active_executions[execution_id] = context
                
                try:
                    # Execute the tool
                    result = await self._execute_with_timeout(context)
                    return result
                    
                finally:
                    # Clean up active execution
                    self._active_executions.pop(execution_id, None)
        
        except Exception as e:
            logger.error(f"Unexpected error in tool execution: {e}")
            context.fail(str(e), "UNEXPECTED_ERROR")
            return context.to_result(execution_id)
    
    async def execute_tools_batch(self,
                                 tool_requests: List[Dict[str, Any]],
                                 timeout: Optional[float] = None,
                                 correlation_id: Optional[str] = None) -> List[ToolExecutionResult]:
        """
        Execute multiple tools concurrently.
        
        Args:
            tool_requests: List of tool execution requests
            timeout: Execution timeout for each tool
            correlation_id: Request correlation ID
            
        Returns:
            List of tool execution results
        """
        if not tool_requests:
            return []
        
        logger.info(f"Executing batch of {len(tool_requests)} tools")
        
        # Create execution tasks
        tasks = []
        for request in tool_requests:
            task = asyncio.create_task(
                self.execute_tool(
                    tool_name=request["tool_name"],
                    arguments=request.get("arguments", {}),
                    timeout=timeout,
                    correlation_id=correlation_id
                )
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result for failed task
                request = tool_requests[i]
                error_result = ToolExecutionResult(
                    tool_name=request["tool_name"],
                    tool_id=f"batch_{i}",
                    arguments=request.get("arguments", {}),
                    result=None,
                    success=False,
                    error=str(result),
                    execution_time=0.0,
                    timestamp=datetime.utcnow()
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_with_timeout(self, context: ExecutionContext) -> ToolExecutionResult:
        """
        Execute tool with timeout handling.
        
        Args:
            context: Execution context
            
        Returns:
            Tool execution result
        """
        context.start()
        
        if self.enable_execution_logging:
            logger.info(
                f"Starting tool execution: {context.tool_name} "
                f"(timeout: {context.timeout}s, correlation: {context.correlation_id})"
            )
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_tool_implementation(context),
                timeout=context.timeout
            )
            
            context.complete(result)
            self.successful_executions += 1
            
            if self.enable_execution_logging:
                duration = context.get_duration()
                logger.info(
                    f"Tool execution completed: {context.tool_name} "
                    f"(duration: {duration:.2f}s)"
                )
            
            return context.to_result(f"{context.tool_name}_{int(time.time())}")
        
        except asyncio.TimeoutError:
            context.timeout_exceeded()
            self.timeout_executions += 1
            self._record_error("TIMEOUT_ERROR", context.tool_name, context.error)
            
            logger.warning(
                f"Tool execution timed out: {context.tool_name} "
                f"(timeout: {context.timeout}s)"
            )
            
            return context.to_result(f"{context.tool_name}_{int(time.time())}")
        
        except ToolExecutionError as e:
            context.fail(str(e), e.error_code)
            self.failed_executions += 1
            self._record_error(e.error_code, context.tool_name, str(e))
            
            logger.error(
                f"Tool execution failed: {context.tool_name} - {e.error_code}: {e}"
            )
            
            return context.to_result(f"{context.tool_name}_{int(time.time())}")
        
        except Exception as e:
            context.fail(str(e), "UNEXPECTED_ERROR")
            self.failed_executions += 1
            self._record_error("UNEXPECTED_ERROR", context.tool_name, str(e))
            
            logger.error(
                f"Unexpected error in tool execution: {context.tool_name} - {e}",
                exc_info=True
            )
            
            return context.to_result(f"{context.tool_name}_{int(time.time())}")
    
    async def _execute_tool_implementation(self, context: ExecutionContext) -> Any:
        """
        Actual tool execution implementation.
        
        Args:
            context: Execution context
            
        Returns:
            Tool execution result
        """
        # Get the tool definition
        tool = await self.mcp_registry.get_tool_by_name(context.tool_name)
        if not tool:
            raise ToolExecutionError(
                f"Tool not found: {context.tool_name}",
                "TOOL_NOT_FOUND"
            )
        
        # Get the server for this tool
        server = self.mcp_registry.servers.get(tool.server_name)
        if not server:
            raise ToolExecutionError(
                f"Server not found for tool {context.tool_name}: {tool.server_name}",
                "SERVER_NOT_FOUND"
            )
        
        if not server.is_connected:
            raise ToolExecutionError(
                f"Server {tool.server_name} is not connected",
                "SERVER_NOT_CONNECTED"
            )
        
        # Validate arguments against tool schema
        self._validate_tool_arguments(tool, context.arguments)
        
        # Execute the tool (placeholder implementation)
        # In a real implementation, this would use the MCP protocol
        result = await self._mock_tool_execution(tool, context.arguments)
        
        # Update tool usage statistics
        tool.usage_count += 1
        tool.last_used = datetime.utcnow()
        
        return result
    
    def _validate_tool_arguments(self, tool: MCPTool, arguments: Dict[str, Any]) -> None:
        """
        Validate tool arguments against the tool's input schema.
        
        Args:
            tool: The MCP tool
            arguments: Arguments to validate
            
        Raises:
            ToolExecutionError: If validation fails
        """
        try:
            schema = tool.input_schema
            
            # Check required properties
            required_props = schema.get("required", [])
            for prop in required_props:
                if prop not in arguments:
                    raise ToolExecutionError(
                        f"Missing required argument: {prop}",
                        "MISSING_REQUIRED_ARGUMENT"
                    )
            
            # Basic type validation for properties
            properties = schema.get("properties", {})
            for arg_name, arg_value in arguments.items():
                if arg_name in properties:
                    prop_def = properties[arg_name]
                    expected_type = prop_def.get("type")
                    
                    if expected_type and not self._validate_argument_type(arg_value, expected_type):
                        raise ToolExecutionError(
                            f"Invalid type for argument {arg_name}: expected {expected_type}",
                            "INVALID_ARGUMENT_TYPE"
                        )
        
        except ToolExecutionError:
            raise
        except Exception as e:
            logger.warning(f"Argument validation error for {tool.name}: {e}")
            # Don't fail execution for validation errors, just log them
    
    def _validate_argument_type(self, value: Any, expected_type: str) -> bool:
        """
        Validate argument type.
        
        Args:
            value: Argument value
            expected_type: Expected type string
            
        Returns:
            True if type is valid
        """
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, allow it
    
    async def _mock_tool_execution(self, tool: MCPTool, arguments: Dict[str, Any]) -> Any:
        """
        Mock tool execution for testing purposes.
        
        Args:
            tool: The MCP tool
            arguments: Tool arguments
            
        Returns:
            Mock execution result
        """
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Check for cancellation during execution
        if hasattr(self, '_check_cancellation'):
            await self._check_cancellation()
        
        # Return mock result based on tool name patterns
        tool_name_lower = tool.name.lower()
        
        if "search" in tool_name_lower:
            query = arguments.get("query", arguments.get("q", ""))
            return {
                "results": [
                    {
                        "title": f"Mock Result 1 for '{query}'",
                        "content": "Mock search result content 1",
                        "url": "https://example.com/result1",
                        "score": 0.95
                    },
                    {
                        "title": f"Mock Result 2 for '{query}'",
                        "content": "Mock search result content 2", 
                        "url": "https://example.com/result2",
                        "score": 0.87
                    }
                ],
                "query": query,
                "total_results": 2,
                "execution_time": 0.1
            }
        
        elif "file" in tool_name_lower:
            path = arguments.get("path", arguments.get("file", "unknown"))
            operation = arguments.get("operation", "read")
            
            return {
                "status": "success",
                "operation": operation,
                "path": path,
                "message": f"File {operation} operation completed successfully",
                "size": 1024,
                "modified": datetime.utcnow().isoformat()
            }
        
        elif "calculate" in tool_name_lower or "math" in tool_name_lower:
            expression = arguments.get("expression", arguments.get("expr", "1+1"))
            
            return {
                "expression": expression,
                "result": 42,  # Mock calculation result
                "type": "number",
                "precision": "exact"
            }
        
        elif "generate" in tool_name_lower or "create" in tool_name_lower:
            content_type = arguments.get("type", "text")
            
            return {
                "generated_content": f"Mock generated {content_type} content",
                "type": content_type,
                "length": 150,
                "quality_score": 0.92
            }
        
        else:
            # Generic tool response
            return {
                "status": "success",
                "tool": tool.name,
                "message": f"Tool {tool.name} executed successfully",
                "arguments": arguments,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _record_error(self, error_code: str, tool_name: str, error_message: str) -> None:
        """
        Record error for statistics and monitoring.
        
        Args:
            error_code: Error code
            tool_name: Name of the tool that failed
            error_message: Error message
        """
        # Update error counts
        self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1
        
        # Add to recent errors
        error_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_code": error_code,
            "tool_name": tool_name,
            "error_message": error_message
        }
        
        self.recent_errors.append(error_record)
        
        # Limit recent errors list size
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors = self.recent_errors[-self.max_recent_errors:]
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an active tool execution.
        
        Args:
            execution_id: ID of the execution to cancel
            
        Returns:
            True if execution was cancelled, False if not found
        """
        context = self._active_executions.get(execution_id)
        if context:
            context.cancel()
            self.cancelled_executions += 1
            logger.info(f"Cancelled tool execution: {execution_id}")
            return True
        
        return False
    
    def get_active_executions(self) -> List[Dict[str, Any]]:
        """
        Get information about currently active executions.
        
        Returns:
            List of active execution information
        """
        active = []
        
        for execution_id, context in self._active_executions.items():
            duration = None
            if context.start_time:
                duration = time.time() - context.start_time
            
            active.append({
                "execution_id": execution_id,
                "tool_name": context.tool_name,
                "status": context.status,
                "duration": duration,
                "timeout": context.timeout,
                "correlation_id": context.correlation_id
            })
        
        return active
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get tool execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        success_rate = (
            self.successful_executions / self.total_executions
            if self.total_executions > 0 else 0.0
        )
        
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "timeout_executions": self.timeout_executions,
            "cancelled_executions": self.cancelled_executions,
            "success_rate": success_rate,
            "active_executions": len(self._active_executions),
            "error_counts": dict(self.error_counts),
            "recent_errors_count": len(self.recent_errors)
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent execution errors.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of recent error records
        """
        return self.recent_errors[-limit:] if self.recent_errors else []
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the tool executor.
        
        Returns:
            Health check results
        """
        try:
            # Check MCP registry health
            registry_health = await self.mcp_registry.get_server_health()
            healthy_servers = sum(1 for is_healthy in registry_health.values() if is_healthy)
            total_servers = len(registry_health)
            
            # Calculate health metrics
            recent_success_rate = 1.0  # Default to healthy
            if self.total_executions > 0:
                recent_success_rate = self.successful_executions / self.total_executions
            
            # Determine overall health
            is_healthy = (
                healthy_servers > 0 and  # At least one server is healthy
                recent_success_rate > 0.5 and  # Success rate above 50%
                len(self._active_executions) < 50  # Not overwhelmed with executions
            )
            
            return {
                "healthy": is_healthy,
                "connected_servers": healthy_servers,
                "total_servers": total_servers,
                "success_rate": recent_success_rate,
                "active_executions": len(self._active_executions),
                "total_executions": self.total_executions,
                "recent_errors": len(self.recent_errors)
            }
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }