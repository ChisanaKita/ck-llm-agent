"""
MCP Registry for server management and tool discovery.

This module implements the MCPRegistry class that manages MCP server connections,
discovers available tools, and provides a unified interface for tool access.
"""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models.internal import MCPServer, MCPServerConfig, MCPTool
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MCPRegistry:
    """
    Registry for managing MCP servers and tool discovery.
    
    This class handles:
    - Loading MCP server configurations
    - Establishing connections to MCP servers
    - Discovering and registering available tools
    - Managing server health and reconnection
    """
    
    def __init__(self, config_paths: Optional[List[str]] = None):
        """
        Initialize the MCP registry.
        
        Args:
            config_paths: List of paths to MCP configuration files
        """
        self.config_paths = config_paths or [
            ".kiro/settings/mcp.json",
            "~/.kiro/settings/mcp.json"
        ]
        self.servers: Dict[str, MCPServer] = {}
        self.tools: Dict[str, MCPTool] = {}
        self._server_processes: Dict[str, subprocess.Popen] = {}
        self._connection_sessions: Dict[str, aiohttp.ClientSession] = {}
        self._discovery_lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize the registry by loading configurations and connecting to servers."""
        logger.info("Initializing MCP registry")
        
        try:
            # Load server configurations
            await self._load_configurations()
            
            # Connect to all enabled servers
            await self._connect_all_servers()
            
            # Discover tools from connected servers
            await self._discover_all_tools()
            
            logger.info(
                f"MCP registry initialized with {len(self.servers)} servers "
                f"and {len(self.tools)} tools"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP registry: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the registry and clean up resources."""
        logger.info("Shutting down MCP registry")
        
        # Close all HTTP sessions
        for session in self._connection_sessions.values():
            if not session.closed:
                await session.close()
        
        # Terminate server processes
        for process in self._server_processes.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        self._connection_sessions.clear()
        self._server_processes.clear()
        
        logger.info("MCP registry shutdown complete")
    
    async def _load_configurations(self) -> None:
        """Load MCP server configurations from config files."""
        logger.debug("Loading MCP configurations")
        
        for config_path in self.config_paths:
            path = Path(config_path).expanduser()
            
            if not path.exists():
                logger.debug(f"Config file not found: {path}")
                continue
            
            try:
                with open(path, 'r') as f:
                    config_data = json.load(f)
                
                mcp_servers = config_data.get('mcpServers', {})
                
                for server_name, server_config in mcp_servers.items():
                    if server_name in self.servers:
                        logger.warning(f"Duplicate server config for {server_name}, skipping")
                        continue
                    
                    # Create server configuration
                    config = MCPServerConfig(
                        name=server_name,
                        command=server_config['command'],
                        args=server_config.get('args', []),
                        env=server_config.get('env', {}),
                        disabled=server_config.get('disabled', False),
                        auto_approve=server_config.get('autoApprove', []),
                        timeout=server_config.get('timeout', 30),
                        max_retries=server_config.get('maxRetries', 3)
                    )
                    
                    # Create server instance
                    server = MCPServer(name=server_name, config=config)
                    self.servers[server_name] = server
                    
                    logger.debug(f"Loaded configuration for server: {server_name}")
                
                logger.info(f"Loaded {len(mcp_servers)} server configs from {path}")
                
            except Exception as e:
                logger.error(f"Failed to load config from {path}: {e}")
    
    async def _connect_all_servers(self) -> None:
        """Connect to all enabled MCP servers."""
        logger.debug("Connecting to MCP servers")
        
        connection_tasks = []
        for server in self.servers.values():
            if not server.config.disabled:
                task = asyncio.create_task(self._connect_server(server))
                connection_tasks.append(task)
        
        if connection_tasks:
            results = await asyncio.gather(*connection_tasks, return_exceptions=True)
            
            successful_connections = sum(
                1 for result in results 
                if not isinstance(result, Exception)
            )
            
            logger.info(f"Connected to {successful_connections}/{len(connection_tasks)} servers")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _connect_server(self, server: MCPServer) -> None:
        """
        Connect to a single MCP server.
        
        Args:
            server: The MCP server to connect to
        """
        logger.debug(f"Connecting to MCP server: {server.name}")
        
        try:
            # Start the server process if needed
            if server.name not in self._server_processes:
                await self._start_server_process(server)
            
            # Create HTTP session for communication
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=server.config.timeout)
            )
            self._connection_sessions[server.name] = session
            
            # Test connection with a ping
            await self._ping_server(server)
            
            server.is_connected = True
            server.connection_attempts += 1
            logger.info(f"Successfully connected to MCP server: {server.name}")
            
        except Exception as e:
            server.is_connected = False
            server.connection_attempts += 1
            server.last_error = str(e)
            server.error_count += 1
            
            logger.error(f"Failed to connect to MCP server {server.name}: {e}")
            raise
    
    async def _start_server_process(self, server: MCPServer) -> None:
        """
        Start the MCP server process.
        
        Args:
            server: The MCP server to start
        """
        try:
            # Prepare command and environment
            cmd = [server.config.command] + server.config.args
            env = {**server.config.env}
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self._server_processes[server.name] = process
            
            # Give the process a moment to start
            await asyncio.sleep(1)
            
            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    f"MCP server process failed to start. "
                    f"Exit code: {process.returncode}, "
                    f"stderr: {stderr}"
                )
            
            logger.debug(f"Started MCP server process: {server.name}")
            
        except Exception as e:
            logger.error(f"Failed to start MCP server process {server.name}: {e}")
            raise
    
    async def _ping_server(self, server: MCPServer) -> None:
        """
        Ping an MCP server to test connectivity.
        
        Args:
            server: The MCP server to ping
        """
        session = self._connection_sessions.get(server.name)
        if not session:
            raise RuntimeError(f"No session for server {server.name}")
        
        # This is a placeholder - actual MCP protocol implementation would go here
        # For now, we'll simulate a successful ping
        await asyncio.sleep(0.1)  # Simulate network delay
        
        from datetime import datetime
        server.last_ping = datetime.utcnow()
    
    async def _discover_all_tools(self) -> None:
        """Discover tools from all connected servers."""
        logger.debug("Discovering tools from MCP servers")
        
        async with self._discovery_lock:
            discovery_tasks = []
            
            for server in self.servers.values():
                if server.is_connected:
                    task = asyncio.create_task(self._discover_server_tools(server))
                    discovery_tasks.append(task)
            
            if discovery_tasks:
                results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
                
                total_tools = 0
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        server_name = list(self.servers.keys())[i]
                        logger.error(f"Tool discovery failed for {server_name}: {result}")
                    else:
                        total_tools += result
                
                logger.info(f"Discovered {total_tools} tools from MCP servers")
    
    async def _discover_server_tools(self, server: MCPServer) -> int:
        """
        Discover tools from a specific MCP server.
        
        Args:
            server: The MCP server to discover tools from
            
        Returns:
            Number of tools discovered
        """
        logger.debug(f"Discovering tools from server: {server.name}")
        
        try:
            # This is a placeholder for actual MCP protocol implementation
            # In a real implementation, this would use the MCP protocol to list tools
            
            # Simulate discovering some tools
            mock_tools = [
                {
                    "name": f"{server.name}_tool_1",
                    "description": f"Mock tool 1 from {server.name}",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string", "description": "Input parameter"}
                        },
                        "required": ["input"]
                    }
                },
                {
                    "name": f"{server.name}_tool_2", 
                    "description": f"Mock tool 2 from {server.name}",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Query parameter"}
                        },
                        "required": ["query"]
                    }
                }
            ]
            
            discovered_tools = []
            
            for tool_data in mock_tools:
                tool = MCPTool(
                    name=tool_data["name"],
                    description=tool_data["description"],
                    server_name=server.name,
                    input_schema=tool_data["input_schema"]
                )
                
                discovered_tools.append(tool)
                self.tools[tool.name] = tool
            
            server.tools = discovered_tools
            server.tool_count = len(discovered_tools)
            
            logger.info(f"Discovered {len(discovered_tools)} tools from {server.name}")
            return len(discovered_tools)
            
        except Exception as e:
            logger.error(f"Failed to discover tools from {server.name}: {e}")
            raise
    
    async def get_available_tools(self, 
                                 server_names: Optional[List[str]] = None,
                                 categories: Optional[List[str]] = None) -> List[MCPTool]:
        """
        Get available tools, optionally filtered by server or category.
        
        Args:
            server_names: Filter by specific server names
            categories: Filter by tool categories
            
        Returns:
            List of available MCP tools
        """
        tools = list(self.tools.values())
        
        if server_names:
            tools = [tool for tool in tools if tool.server_name in server_names]
        
        if categories:
            tools = [tool for tool in tools if tool.category in categories]
        
        return tools
    
    async def get_tool_by_name(self, tool_name: str) -> Optional[MCPTool]:
        """
        Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            The MCP tool if found, None otherwise
        """
        return self.tools.get(tool_name)
    
    async def refresh_tools(self, server_name: Optional[str] = None) -> None:
        """
        Refresh tool discovery for all servers or a specific server.
        
        Args:
            server_name: Specific server to refresh, or None for all servers
        """
        logger.info(f"Refreshing tools for server: {server_name or 'all'}")
        
        if server_name:
            server = self.servers.get(server_name)
            if server and server.is_connected:
                await self._discover_server_tools(server)
        else:
            await self._discover_all_tools()
    
    async def get_server_health(self) -> Dict[str, bool]:
        """
        Get health status of all MCP servers.
        
        Returns:
            Dictionary mapping server names to health status
        """
        health_status = {}
        
        for server_name, server in self.servers.items():
            health_status[server_name] = server.is_healthy()
        
        return health_status
    
    def get_registry_stats(self) -> Dict[str, int]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        connected_servers = sum(1 for server in self.servers.values() if server.is_connected)
        healthy_servers = sum(1 for server in self.servers.values() if server.is_healthy())
        
        return {
            "total_servers": len(self.servers),
            "connected_servers": connected_servers,
            "healthy_servers": healthy_servers,
            "total_tools": len(self.tools),
            "enabled_servers": sum(1 for server in self.servers.values() if not server.config.disabled)
        }