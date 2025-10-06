"""
Tool Selector for dynamic tool filtering based on context and semantic similarity.

This module provides intelligent tool selection capabilities that combine
semantic search with context analysis to dynamically filter and select
the most relevant tools for agent tasks.
"""

from typing import Dict, List, Optional

from ..models.api import ChatMessage
from ..models.internal import MCPTool, SemanticSearchQuery, SemanticSearchResult
from ..utils.logging import get_logger
from .mcp_registry import MCPRegistry
from .semantic_search import SemanticSearchEngine

logger = get_logger(__name__)


class ToolSelector:
    """
    Intelligent tool selector that combines semantic search with context analysis.
    
    This class provides methods to dynamically select the most relevant tools
    for agent tasks based on conversation context, user intent, and semantic similarity.
    """
    
    def __init__(self, 
                 mcp_registry: MCPRegistry,
                 semantic_engine: SemanticSearchEngine,
                 default_max_tools: int = 5,
                 similarity_threshold: float = 0.7):
        """
        Initialize the tool selector.
        
        Args:
            mcp_registry: MCP registry for tool access
            semantic_engine: Semantic search engine
            default_max_tools: Default maximum number of tools to select
            similarity_threshold: Default similarity threshold for tool selection
        """
        self.mcp_registry = mcp_registry
        self.semantic_engine = semantic_engine
        self.default_max_tools = default_max_tools
        self.similarity_threshold = similarity_threshold
        
        # Selection statistics
        self.selection_count = 0
        self.total_tools_selected = 0
        self.semantic_selections = 0
        self.fallback_selections = 0
    
    async def select_tools_for_conversation(self,
                                          messages: List[ChatMessage],
                                          max_tools: Optional[int] = None,
                                          strategy: str = "semantic",
                                          categories: Optional[List[str]] = None,
                                          server_names: Optional[List[str]] = None) -> List[MCPTool]:
        """
        Select tools based on conversation context.
        
        Args:
            messages: Conversation messages for context
            max_tools: Maximum number of tools to select
            strategy: Selection strategy ("semantic", "all", "manual", "none")
            categories: Filter by tool categories
            server_names: Filter by server names
            
        Returns:
            List of selected MCP tools
        """
        self.selection_count += 1
        max_tools = max_tools or self.default_max_tools
        
        logger.debug(f"Selecting tools with strategy: {strategy}, max_tools: {max_tools}")
        
        try:
            if strategy == "none":
                return []
            
            elif strategy == "all":
                tools = await self._select_all_tools(server_names, categories, max_tools)
                
            elif strategy == "manual":
                # Manual selection would require explicit tool names
                # For now, fall back to semantic selection
                tools = await self._select_semantic_tools(messages, max_tools, categories, server_names)
                
            elif strategy == "semantic":
                tools = await self._select_semantic_tools(messages, max_tools, categories, server_names)
                
            else:
                logger.warning(f"Unknown selection strategy: {strategy}, using semantic")
                tools = await self._select_semantic_tools(messages, max_tools, categories, server_names)
            
            self.total_tools_selected += len(tools)
            
            logger.info(f"Selected {len(tools)} tools using {strategy} strategy")
            return tools
            
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            # Fall back to basic tool selection
            return await self._fallback_tool_selection(server_names, categories, max_tools)
    
    async def select_tools_for_query(self,
                                   query: str,
                                   max_tools: Optional[int] = None,
                                   categories: Optional[List[str]] = None,
                                   server_names: Optional[List[str]] = None,
                                   similarity_threshold: Optional[float] = None) -> List[MCPTool]:
        """
        Select tools based on a specific query.
        
        Args:
            query: Query text for tool selection
            max_tools: Maximum number of tools to select
            categories: Filter by tool categories
            server_names: Filter by server names
            similarity_threshold: Similarity threshold for selection
            
        Returns:
            List of selected MCP tools
        """
        max_tools = max_tools or self.default_max_tools
        threshold = similarity_threshold or self.similarity_threshold
        
        try:
            # Create semantic search query
            search_query = SemanticSearchQuery(
                query=query,
                max_results=max_tools * 2,  # Get more results to allow for filtering
                similarity_threshold=threshold,
                categories=categories
            )
            
            # Perform semantic search
            search_results = await self.semantic_engine.search_relevant_tools(search_query)
            
            # Filter by server names if specified
            if server_names:
                search_results = [
                    result for result in search_results
                    if result.tool.server_name in server_names
                ]
            
            # Extract tools and limit to max_tools
            tools = [result.tool for result in search_results[:max_tools]]
            
            self.semantic_selections += 1
            
            logger.debug(f"Selected {len(tools)} tools for query: {query[:50]}...")
            return tools
            
        except Exception as e:
            logger.error(f"Query-based tool selection failed: {e}")
            return await self._fallback_tool_selection(server_names, categories, max_tools)
    
    async def _select_semantic_tools(self,
                                   messages: List[ChatMessage],
                                   max_tools: int,
                                   categories: Optional[List[str]] = None,
                                   server_names: Optional[List[str]] = None) -> List[MCPTool]:
        """
        Select tools using semantic analysis of conversation.
        
        Args:
            messages: Conversation messages
            max_tools: Maximum number of tools to select
            categories: Filter by categories
            server_names: Filter by server names
            
        Returns:
            List of selected tools
        """
        try:
            # Extract context from messages
            context = self._extract_conversation_context(messages)
            
            if not context:
                logger.debug("No context extracted, falling back to basic selection")
                return await self._fallback_tool_selection(server_names, categories, max_tools)
            
            # Create search query from context
            search_query = SemanticSearchQuery(
                query=context,
                max_results=max_tools * 2,
                similarity_threshold=self.similarity_threshold,
                categories=categories,
                conversation_context=context
            )
            
            # Perform semantic search
            search_results = await self.semantic_engine.search_relevant_tools(search_query)
            
            # Filter by server names if specified
            if server_names:
                search_results = [
                    result for result in search_results
                    if result.tool.server_name in server_names
                ]
            
            # Select tools with diversity consideration
            selected_tools = self._select_diverse_tools(search_results, max_tools)
            
            self.semantic_selections += 1
            
            return selected_tools
            
        except Exception as e:
            logger.error(f"Semantic tool selection failed: {e}")
            return await self._fallback_tool_selection(server_names, categories, max_tools)
    
    async def _select_all_tools(self,
                              server_names: Optional[List[str]] = None,
                              categories: Optional[List[str]] = None,
                              max_tools: int = 10) -> List[MCPTool]:
        """
        Select all available tools with optional filtering.
        
        Args:
            server_names: Filter by server names
            categories: Filter by categories
            max_tools: Maximum number of tools to return
            
        Returns:
            List of all available tools
        """
        try:
            # Get all available tools
            all_tools = await self.mcp_registry.get_available_tools(
                server_names=server_names,
                categories=categories
            )
            
            # Sort by usage count (most used first) and limit
            all_tools.sort(key=lambda t: t.usage_count, reverse=True)
            
            return all_tools[:max_tools]
            
        except Exception as e:
            logger.error(f"Failed to select all tools: {e}")
            return []
    
    async def _fallback_tool_selection(self,
                                     server_names: Optional[List[str]] = None,
                                     categories: Optional[List[str]] = None,
                                     max_tools: int = 5) -> List[MCPTool]:
        """
        Fallback tool selection when other methods fail.
        
        Args:
            server_names: Filter by server names
            categories: Filter by categories
            max_tools: Maximum number of tools to return
            
        Returns:
            List of fallback tools
        """
        self.fallback_selections += 1
        
        try:
            # Get most frequently used tools
            all_tools = await self.mcp_registry.get_available_tools(
                server_names=server_names,
                categories=categories
            )
            
            # Sort by usage count and take top tools
            all_tools.sort(key=lambda t: t.usage_count, reverse=True)
            
            selected = all_tools[:max_tools]
            
            logger.debug(f"Fallback selection returned {len(selected)} tools")
            return selected
            
        except Exception as e:
            logger.error(f"Fallback tool selection failed: {e}")
            return []
    
    def _extract_conversation_context(self, messages: List[ChatMessage]) -> str:
        """
        Extract relevant context from conversation messages.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Extracted context string
        """
        if not messages:
            return ""
        
        # Focus on recent messages (last 3-5 messages)
        recent_messages = messages[-5:]
        
        context_parts = []
        
        for message in recent_messages:
            if message.content:
                # Extract key information from message content
                content = message.content.strip()
                
                # Skip very short messages
                if len(content) < 10:
                    continue
                
                # Add role context
                if message.role == "user":
                    context_parts.append(f"User request: {content}")
                elif message.role == "assistant":
                    # Only include assistant messages that might indicate tool needs
                    if any(keyword in content.lower() for keyword in [
                        "search", "find", "get", "fetch", "retrieve", "analyze", 
                        "calculate", "process", "generate", "create", "write"
                    ]):
                        context_parts.append(f"Assistant action: {content}")
        
        # Combine context parts
        context = " | ".join(context_parts)
        
        # Limit context length
        if len(context) > 500:
            context = context[:500] + "..."
        
        return context
    
    def _select_diverse_tools(self, 
                            search_results: List[SemanticSearchResult],
                            max_tools: int) -> List[MCPTool]:
        """
        Select diverse tools from search results to avoid redundancy.
        
        Args:
            search_results: Semantic search results
            max_tools: Maximum number of tools to select
            
        Returns:
            List of diverse tools
        """
        if not search_results:
            return []
        
        selected_tools = []
        used_categories = set()
        used_servers = set()
        
        # First pass: select highest scoring tools from different categories/servers
        for result in search_results:
            if len(selected_tools) >= max_tools:
                break
            
            tool = result.tool
            
            # Prefer tools from different categories and servers for diversity
            category_new = tool.category not in used_categories
            server_new = tool.server_name not in used_servers
            
            if category_new or server_new or len(selected_tools) < max_tools // 2:
                selected_tools.append(tool)
                
                if tool.category:
                    used_categories.add(tool.category)
                used_servers.add(tool.server_name)
        
        # Second pass: fill remaining slots with best remaining tools
        for result in search_results:
            if len(selected_tools) >= max_tools:
                break
            
            if result.tool not in selected_tools:
                selected_tools.append(result.tool)
        
        return selected_tools[:max_tools]
    
    def get_selection_stats(self) -> Dict[str, int]:
        """
        Get tool selection statistics.
        
        Returns:
            Dictionary with selection statistics
        """
        avg_tools_per_selection = (
            self.total_tools_selected / self.selection_count
            if self.selection_count > 0 else 0.0
        )
        
        return {
            "total_selections": self.selection_count,
            "total_tools_selected": self.total_tools_selected,
            "average_tools_per_selection": round(avg_tools_per_selection, 2),
            "semantic_selections": self.semantic_selections,
            "fallback_selections": self.fallback_selections,
            "semantic_success_rate": (
                self.semantic_selections / self.selection_count
                if self.selection_count > 0 else 0.0
            )
        }
    
    async def refresh_tool_embeddings(self) -> None:
        """Refresh embeddings for all tools in the registry."""
        try:
            # Get all available tools
            all_tools = await self.mcp_registry.get_available_tools()
            
            if all_tools:
                # Add/update embeddings
                await self.semantic_engine.add_tool_embeddings(all_tools)
                
                logger.info(f"Refreshed embeddings for {len(all_tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to refresh tool embeddings: {e}")
    
    async def analyze_tool_usage(self) -> Dict[str, any]:
        """
        Analyze tool usage patterns for optimization.
        
        Returns:
            Dictionary with usage analysis
        """
        try:
            all_tools = await self.mcp_registry.get_available_tools()
            
            if not all_tools:
                return {"error": "No tools available for analysis"}
            
            # Calculate usage statistics
            total_usage = sum(tool.usage_count for tool in all_tools)
            used_tools = [tool for tool in all_tools if tool.usage_count > 0]
            unused_tools = [tool for tool in all_tools if tool.usage_count == 0]
            
            # Category analysis
            category_usage = {}
            for tool in all_tools:
                category = tool.category or "uncategorized"
                if category not in category_usage:
                    category_usage[category] = {"count": 0, "usage": 0}
                category_usage[category]["count"] += 1
                category_usage[category]["usage"] += tool.usage_count
            
            # Server analysis
            server_usage = {}
            for tool in all_tools:
                server = tool.server_name
                if server not in server_usage:
                    server_usage[server] = {"count": 0, "usage": 0}
                server_usage[server]["count"] += 1
                server_usage[server]["usage"] += tool.usage_count
            
            return {
                "total_tools": len(all_tools),
                "used_tools": len(used_tools),
                "unused_tools": len(unused_tools),
                "total_usage": total_usage,
                "average_usage_per_tool": total_usage / len(all_tools) if all_tools else 0,
                "category_breakdown": category_usage,
                "server_breakdown": server_usage,
                "most_used_tools": [
                    {"name": tool.name, "usage": tool.usage_count}
                    for tool in sorted(all_tools, key=lambda t: t.usage_count, reverse=True)[:5]
                ]
            }
            
        except Exception as e:
            logger.error(f"Tool usage analysis failed: {e}")
            return {"error": f"Analysis failed: {e}"}