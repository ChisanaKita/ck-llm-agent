"""
Tool Adapter for converting MCP tools to CrewAI format.

This module provides the ToolAdapter class that converts MCP tool definitions
into CrewAI-compatible tool instances with proper execution handling.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

from crewai.tools import BaseTool
from pydantic import Field

from ..models.api import Tool, ToolFunctionSpec
from ..models.internal import MCPTool, ToolExecutionResult
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MCPToolExecutor:
    """
    Executor for MCP tools that handles the actual tool invocation.
    """
    
    def __init__(self, registry):
        """
        Initialize the tool executor.
        
        Args:
            registry: The MCP registry instance for server communication
        """
        self.registry = registry
    
    async def execute_tool(self, 
                          tool_name: str, 
                          arguments: Dict[str, Any],
                          timeout: float = 30.0) -> ToolExecutionResult:
        """
        Execute an MCP tool with the given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            timeout: Execution timeout in seconds
            
        Returns:
            Tool execution result
        """
        import time
        from datetime import datetime
        
        start_time = time.time()
        
        try:
            # Get the tool definition
            tool = await self.registry.get_tool_by_name(tool_name)
            if not tool:
                raise ValueError(f"Tool not found: {tool_name}")
            
            # Get the server for this tool
            server = self.registry.servers.get(tool.server_name)
            if not server or not server.is_connected:
                raise RuntimeError(f"Server {tool.server_name} is not connected")
            
            # Execute the tool (placeholder implementation)
            # In a real implementation, this would use the MCP protocol
            result = await self._mock_tool_execution(tool, arguments, timeout)
            
            execution_time = time.time() - start_time
            
            # Update tool usage statistics
            tool.usage_count += 1
            tool.last_used = datetime.utcnow()
            
            if tool.average_execution_time is None:
                tool.average_execution_time = execution_time
            else:
                # Update running average
                tool.average_execution_time = (
                    (tool.average_execution_time * (tool.usage_count - 1) + execution_time) 
                    / tool.usage_count
                )
            
            return ToolExecutionResult(
                tool_name=tool_name,
                tool_id=tool_name,  # Using name as ID for now
                arguments=arguments,
                result=result,
                success=True,
                execution_time=execution_time,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            
            return ToolExecutionResult(
                tool_name=tool_name,
                tool_id=tool_name,
                arguments=arguments,
                result=None,
                success=False,
                error=str(e),
                execution_time=execution_time,
                timestamp=datetime.utcnow()
            )
    
    async def _mock_tool_execution(self, 
                                  tool: MCPTool, 
                                  arguments: Dict[str, Any],
                                  timeout: float) -> Any:
        """
        Mock tool execution for testing purposes.
        
        Args:
            tool: The MCP tool to execute
            arguments: Tool arguments
            timeout: Execution timeout
            
        Returns:
            Mock execution result
        """
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        # Return a mock result based on the tool name
        if "search" in tool.name.lower():
            return {
                "results": [
                    {"title": "Mock Result 1", "content": "Mock content 1"},
                    {"title": "Mock Result 2", "content": "Mock content 2"}
                ],
                "query": arguments.get("query", ""),
                "total_results": 2
            }
        elif "file" in tool.name.lower():
            return {
                "status": "success",
                "message": f"File operation completed for: {arguments.get('path', 'unknown')}",
                "size": 1024
            }
        else:
            return {
                "status": "success",
                "message": f"Tool {tool.name} executed successfully",
                "arguments": arguments
            }


class CrewAIToolWrapper(BaseTool):
    """
    CrewAI tool wrapper for MCP tools.
    
    This class wraps an MCP tool to make it compatible with CrewAI's tool system.
    """
    
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    
    def __init__(self, mcp_tool: MCPTool, executor: MCPToolExecutor, **kwargs):
        """
        Initialize the CrewAI tool wrapper.
        
        Args:
            mcp_tool: The MCP tool to wrap
            executor: Tool executor instance
            **kwargs: Additional arguments for BaseTool
        """
        super().__init__(
            name=mcp_tool.name,
            description=mcp_tool.description or f"Tool: {mcp_tool.name}",
            **kwargs
        )
        self.mcp_tool = mcp_tool
        self.executor = executor
    
    def _run(self, **kwargs) -> str:
        """
        Synchronous tool execution (required by CrewAI).
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            Tool execution result as string
        """
        # CrewAI requires synchronous execution, so we need to run async code
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(self._async_run(**kwargs))
        return result
    
    async def _async_run(self, **kwargs) -> str:
        """
        Asynchronous tool execution.
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            Tool execution result as string
        """
        try:
            # Execute the tool
            result = await self.executor.execute_tool(self.mcp_tool.name, kwargs)
            
            if result.success:
                # Format the result as a string for CrewAI
                if isinstance(result.result, dict):
                    return json.dumps(result.result, indent=2)
                else:
                    return str(result.result)
            else:
                return f"Tool execution failed: {result.error}"
                
        except Exception as e:
            logger.error(f"Error in CrewAI tool wrapper for {self.mcp_tool.name}: {e}")
            return f"Tool execution error: {str(e)}"


class ToolAdapter:
    """
    Adapter for converting MCP tools to CrewAI format.
    
    This class handles the conversion of MCP tool definitions into CrewAI-compatible
    tool instances that can be used by CrewAI agents.
    """
    
    def __init__(self, registry):
        """
        Initialize the tool adapter.
        
        Args:
            registry: The MCP registry instance
        """
        self.registry = registry
        self.executor = MCPToolExecutor(registry)
        self._tool_cache: Dict[str, CrewAIToolWrapper] = {}
    
    def convert_mcp_tool_to_api_format(self, mcp_tool: MCPTool) -> Tool:
        """
        Convert an MCP tool to OpenAI API tool format.
        
        Args:
            mcp_tool: The MCP tool to convert
            
        Returns:
            Tool in OpenAI API format
        """
        return Tool(
            type="function",
            function=ToolFunctionSpec(
                name=mcp_tool.name,
                description=mcp_tool.description or f"Tool: {mcp_tool.name}",
                parameters=mcp_tool.input_schema
            )
        )
    
    def convert_mcp_tool_to_crewai(self, mcp_tool: MCPTool) -> CrewAIToolWrapper:
        """
        Convert an MCP tool to CrewAI tool format.
        
        Args:
            mcp_tool: The MCP tool to convert
            
        Returns:
            CrewAI-compatible tool wrapper
        """
        # Check cache first
        if mcp_tool.name in self._tool_cache:
            return self._tool_cache[mcp_tool.name]
        
        # Create new wrapper
        wrapper = CrewAIToolWrapper(
            mcp_tool=mcp_tool,
            executor=self.executor
        )
        
        # Cache the wrapper
        self._tool_cache[mcp_tool.name] = wrapper
        
        return wrapper
    
    def convert_tools_to_api_format(self, mcp_tools: List[MCPTool]) -> List[Tool]:
        """
        Convert a list of MCP tools to OpenAI API format.
        
        Args:
            mcp_tools: List of MCP tools to convert
            
        Returns:
            List of tools in OpenAI API format
        """
        return [self.convert_mcp_tool_to_api_format(tool) for tool in mcp_tools]
    
    def convert_tools_to_crewai(self, mcp_tools: List[MCPTool]) -> List[CrewAIToolWrapper]:
        """
        Convert a list of MCP tools to CrewAI format.
        
        Args:
            mcp_tools: List of MCP tools to convert
            
        Returns:
            List of CrewAI-compatible tool wrappers
        """
        return [self.convert_mcp_tool_to_crewai(tool) for tool in mcp_tools]
    
    async def get_tools_for_agent(self, 
                                 tool_names: Optional[List[str]] = None,
                                 server_names: Optional[List[str]] = None,
                                 max_tools: int = 10) -> List[CrewAIToolWrapper]:
        """
        Get CrewAI tools for an agent, with optional filtering.
        
        Args:
            tool_names: Specific tool names to include
            server_names: Filter by server names
            max_tools: Maximum number of tools to return
            
        Returns:
            List of CrewAI-compatible tools
        """
        # Get available MCP tools
        available_tools = await self.registry.get_available_tools(server_names=server_names)
        
        # Filter by specific tool names if provided
        if tool_names:
            available_tools = [
                tool for tool in available_tools 
                if tool.name in tool_names
            ]
        
        # Limit the number of tools
        if len(available_tools) > max_tools:
            # Sort by usage count (most used first) and take the top tools
            available_tools.sort(key=lambda t: t.usage_count, reverse=True)
            available_tools = available_tools[:max_tools]
        
        # Convert to CrewAI format
        return self.convert_tools_to_crewai(available_tools)
    
    async def get_tool_definitions_for_llm(self, 
                                          tool_names: Optional[List[str]] = None,
                                          server_names: Optional[List[str]] = None) -> List[Tool]:
        """
        Get tool definitions in OpenAI API format for LLM function calling.
        
        Args:
            tool_names: Specific tool names to include
            server_names: Filter by server names
            
        Returns:
            List of tool definitions in OpenAI API format
        """
        # Get available MCP tools
        available_tools = await self.registry.get_available_tools(server_names=server_names)
        
        # Filter by specific tool names if provided
        if tool_names:
            available_tools = [
                tool for tool in available_tools 
                if tool.name in tool_names
            ]
        
        # Convert to API format
        return self.convert_tools_to_api_format(available_tools)
    
    def clear_cache(self) -> None:
        """Clear the tool wrapper cache."""
        self._tool_cache.clear()
        logger.debug("Tool adapter cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_tools": len(self._tool_cache),
            "cache_size_bytes": sum(
                len(str(wrapper)) for wrapper in self._tool_cache.values()
            )
        }