"""
Semantic Search Engine with ChromaDB for tool selection.

This module implements semantic search capabilities using ChromaDB vector database
and vLLM embeddings API for intelligent tool selection based on context similarity.
Enhanced with advanced caching and performance optimizations.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import chromadb
from chromadb.config import Settings
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models.internal import MCPTool, SemanticSearchQuery, SemanticSearchResult
from ..utils.logging import get_logger
from .cache_manager import ResponseCacheManager
from .chromadb_manager import ChromaDBManager, ChromaDBConfig

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings using vLLM Embeddings API.
    
    This service handles communication with external vLLM embedding servers
    using OpenAI-compatible API calls via aiohttp.
    """
    
    def __init__(self, 
                 base_url: str,
                 model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                 embedding_dim: int = 1024,
                 max_concurrent_requests: int = 10,
                 timeout: float = 30.0):
        """
        Initialize the embedding service.
        
        Args:
            base_url: Base URL of the vLLM embedding server
            model_name: Name of the embedding model
            embedding_dim: Dimension of the embedding vectors
            max_concurrent_requests: Maximum concurrent requests
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.timeout = timeout
        
        # Connection management
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens_processed = 0
    
    async def initialize(self) -> None:
        """Initialize the embedding service."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=20,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "llm-agent-backend/1.0"
                }
            )
        
        logger.info(f"Embedding service initialized with model: {self.model_name}")
    
    async def shutdown(self) -> None:
        """Shutdown the embedding service."""
        if self._session and not self._session.closed:
            await self._session.close()
        
        logger.info("Embedding service shutdown complete")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector as list of floats
        """
        async with self._semaphore:
            return await self._generate_single_embedding(text)
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Process in batches to avoid overwhelming the server
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Create tasks for concurrent processing
            tasks = [self.generate_embedding(text) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            for j, embedding in enumerate(batch_embeddings):
                if isinstance(embedding, Exception):
                    logger.error(f"Failed to generate embedding for text {i+j}: {embedding}")
                    # Use zero vector as fallback
                    embedding = [0.0] * self.embedding_dim
                
                all_embeddings.append(embedding)
        
        return all_embeddings
    
    async def _generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using vLLM API.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector as list of floats
        """
        if not self._session:
            await self.initialize()
        
        self.total_requests += 1
        
        try:
            # Prepare the request payload (OpenAI-compatible format)
            payload = {
                "model": self.model_name,
                "input": text,
                "encoding_format": "float"
            }
            
            # Make the API request
            url = f"{self.base_url}/v1/embeddings"
            
            async with self._session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Embedding API request failed with status {response.status}: {error_text}"
                    )
                
                result = await response.json()
                
                # Extract embedding from response
                if "data" not in result or not result["data"]:
                    raise ValueError("Invalid embedding response format")
                
                embedding = result["data"][0]["embedding"]
                
                # Validate embedding dimension
                if len(embedding) != self.embedding_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                        f"got {len(embedding)}"
                    )
                
                # Update statistics
                self.successful_requests += 1
                if "usage" in result:
                    self.total_tokens_processed += result["usage"].get("total_tokens", 0)
                
                return embedding
        
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get embedding service statistics.
        
        Returns:
            Dictionary with service statistics
        """
        success_rate = (
            self.successful_requests / self.total_requests 
            if self.total_requests > 0 else 0.0
        )
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "total_tokens_processed": self.total_tokens_processed,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim
        }


class SemanticSearchEngine:
    """
    Semantic search engine using ChromaDB for tool selection.
    
    This class manages vector storage, similarity search, and tool filtering
    based on semantic similarity using ChromaDB and vLLM embeddings.
    """
    
    def __init__(self,
                 persist_directory: str = "./data/chromadb",
                 embedding_service: Optional[EmbeddingService] = None,
                 collection_name: str = "tool_embeddings"):
        """
        Initialize the semantic search engine.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            embedding_service: Embedding service instance
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        
        # ChromaDB components
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None
        
        # Tool cache
        self._tool_cache: Dict[str, MCPTool] = {}
        
        # Statistics
        self.search_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def initialize(self) -> None:
        """Initialize the semantic search engine."""
        logger.info("Initializing semantic search engine")
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16
                }
            )
            
            # Initialize embedding service if provided
            if self.embedding_service:
                await self.embedding_service.initialize()
            
            logger.info(
                f"Semantic search engine initialized with collection: {self.collection_name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic search engine: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the semantic search engine."""
        if self.embedding_service:
            await self.embedding_service.shutdown()
        
        logger.info("Semantic search engine shutdown complete")
    
    async def add_tool_embeddings(self, tools: List[MCPTool]) -> None:
        """
        Add or update tool embeddings in the vector database.
        
        Args:
            tools: List of MCP tools to add embeddings for
        """
        if not self.collection or not self.embedding_service:
            raise RuntimeError("Semantic search engine not initialized")
        
        logger.info(f"Adding embeddings for {len(tools)} tools")
        
        try:
            # Prepare texts for embedding
            texts = []
            tool_ids = []
            tool_metadata = []
            
            for tool in tools:
                # Create text representation of the tool
                tool_text = self._create_tool_text(tool)
                texts.append(tool_text)
                tool_ids.append(tool.name)
                
                # Prepare metadata
                metadata = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "server_name": tool.server_name,
                    "category": tool.category or "general",
                    "usage_count": tool.usage_count,
                    "tags": json.dumps(tool.tags)
                }
                tool_metadata.append(metadata)
                
                # Update tool cache
                self._tool_cache[tool.name] = tool
            
            # Generate embeddings
            embeddings = await self.embedding_service.generate_embeddings_batch(texts)
            
            # Add to ChromaDB collection
            self.collection.upsert(
                ids=tool_ids,
                embeddings=embeddings,
                metadatas=tool_metadata,
                documents=texts
            )
            
            # Update tool objects with embeddings
            for tool, embedding in zip(tools, embeddings):
                tool.embedding = embedding
                tool.embedding_updated = datetime.utcnow()
            
            logger.info(f"Successfully added embeddings for {len(tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to add tool embeddings: {e}")
            raise
    
    async def search_relevant_tools(self, 
                                   query: SemanticSearchQuery) -> List[SemanticSearchResult]:
        """
        Search for relevant tools based on semantic similarity.
        
        Args:
            query: Semantic search query
            
        Returns:
            List of relevant tools with similarity scores
        """
        if not self.collection or not self.embedding_service:
            raise RuntimeError("Semantic search engine not initialized")
        
        self.search_count += 1
        
        try:
            # Generate embedding for the query
            query_embedding = await self.embedding_service.generate_embedding(query.query)
            
            # Prepare where clause for filtering
            where_clause = {}
            if query.categories:
                where_clause["category"] = {"$in": query.categories}
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(query.max_results, 20),  # Limit to reasonable number
                where=where_clause if where_clause else None,
                include=["metadatas", "distances", "documents"]
            )
            
            # Process results
            search_results = []
            
            if results["ids"] and results["ids"][0]:
                for i, tool_id in enumerate(results["ids"][0]):
                    # Skip excluded tools
                    if query.exclude_tools and tool_id in query.exclude_tools:
                        continue
                    
                    # Calculate similarity score (ChromaDB returns distances)
                    distance = results["distances"][0][i]
                    similarity_score = 1.0 - distance  # Convert distance to similarity
                    
                    # Apply similarity threshold
                    if similarity_score < query.similarity_threshold:
                        continue
                    
                    # Get tool from cache
                    tool = self._tool_cache.get(tool_id)
                    if not tool:
                        # Try to reconstruct tool from metadata
                        metadata = results["metadatas"][0][i]
                        tool = self._reconstruct_tool_from_metadata(tool_id, metadata)
                    
                    if tool:
                        # Generate relevance reason
                        relevance_reason = self._generate_relevance_reason(
                            query, tool, similarity_score
                        )
                        
                        search_results.append(SemanticSearchResult(
                            tool=tool,
                            similarity_score=similarity_score,
                            relevance_reason=relevance_reason
                        ))
            
            # Sort by similarity score (highest first)
            search_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.debug(
                f"Semantic search returned {len(search_results)} results "
                f"for query: {query.query[:50]}..."
            )
            
            return search_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise
    
    async def update_tool_embedding(self, tool: MCPTool) -> None:
        """
        Update embedding for a single tool.
        
        Args:
            tool: Tool to update embedding for
        """
        if not self.collection or not self.embedding_service:
            raise RuntimeError("Semantic search engine not initialized")
        
        try:
            # Generate new embedding
            tool_text = self._create_tool_text(tool)
            embedding = await self.embedding_service.generate_embedding(tool_text)
            
            # Update in ChromaDB
            metadata = {
                "name": tool.name,
                "description": tool.description or "",
                "server_name": tool.server_name,
                "category": tool.category or "general",
                "usage_count": tool.usage_count,
                "tags": json.dumps(tool.tags)
            }
            
            self.collection.upsert(
                ids=[tool.name],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[tool_text]
            )
            
            # Update tool object
            tool.embedding = embedding
            tool.embedding_updated = datetime.utcnow()
            
            # Update cache
            self._tool_cache[tool.name] = tool
            
            logger.debug(f"Updated embedding for tool: {tool.name}")
            
        except Exception as e:
            logger.error(f"Failed to update tool embedding for {tool.name}: {e}")
            raise
    
    def _create_tool_text(self, tool: MCPTool) -> str:
        """
        Create text representation of a tool for embedding.
        
        Args:
            tool: Tool to create text for
            
        Returns:
            Text representation of the tool
        """
        parts = [
            f"Tool: {tool.name}",
        ]
        
        if tool.description:
            parts.append(f"Description: {tool.description}")
        
        if tool.category:
            parts.append(f"Category: {tool.category}")
        
        if tool.tags:
            parts.append(f"Tags: {', '.join(tool.tags)}")
        
        # Add schema information
        if tool.input_schema:
            schema_desc = self._describe_schema(tool.input_schema)
            if schema_desc:
                parts.append(f"Parameters: {schema_desc}")
        
        return " | ".join(parts)
    
    def _describe_schema(self, schema: Dict[str, Any]) -> str:
        """
        Create human-readable description of a JSON schema.
        
        Args:
            schema: JSON schema to describe
            
        Returns:
            Human-readable schema description
        """
        if not isinstance(schema, dict):
            return ""
        
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        descriptions = []
        for prop_name, prop_def in properties.items():
            prop_type = prop_def.get("type", "unknown")
            prop_desc = prop_def.get("description", "")
            
            is_required = prop_name in required
            req_marker = " (required)" if is_required else " (optional)"
            
            if prop_desc:
                descriptions.append(f"{prop_name} ({prop_type}): {prop_desc}{req_marker}")
            else:
                descriptions.append(f"{prop_name} ({prop_type}){req_marker}")
        
        return "; ".join(descriptions)
    
    def _reconstruct_tool_from_metadata(self, tool_id: str, metadata: Dict[str, Any]) -> MCPTool:
        """
        Reconstruct an MCPTool from ChromaDB metadata.
        
        Args:
            tool_id: Tool identifier
            metadata: Tool metadata from ChromaDB
            
        Returns:
            Reconstructed MCPTool
        """
        try:
            tags = json.loads(metadata.get("tags", "[]"))
        except (json.JSONDecodeError, TypeError):
            tags = []
        
        tool = MCPTool(
            name=tool_id,
            description=metadata.get("description", ""),
            server_name=metadata.get("server_name", "unknown"),
            input_schema={},  # Schema not stored in metadata
            category=metadata.get("category"),
            tags=tags,
            usage_count=metadata.get("usage_count", 0)
        )
        
        return tool
    
    def _generate_relevance_reason(self, 
                                  query: SemanticSearchQuery,
                                  tool: MCPTool,
                                  similarity_score: float) -> str:
        """
        Generate a reason why a tool is relevant to the query.
        
        Args:
            query: The search query
            tool: The relevant tool
            similarity_score: Similarity score
            
        Returns:
            Human-readable relevance reason
        """
        reasons = []
        
        # Score-based reason
        if similarity_score > 0.9:
            reasons.append("highly relevant")
        elif similarity_score > 0.8:
            reasons.append("very relevant")
        elif similarity_score > 0.7:
            reasons.append("relevant")
        else:
            reasons.append("potentially relevant")
        
        # Category match
        if query.categories and tool.category in query.categories:
            reasons.append(f"matches category '{tool.category}'")
        
        # Usage popularity
        if tool.usage_count > 10:
            reasons.append("frequently used")
        elif tool.usage_count > 0:
            reasons.append("previously used")
        
        return f"Tool is {', '.join(reasons)} (similarity: {similarity_score:.2f})"
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ChromaDB collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        try:
            count = self.collection.count()
            
            return {
                "total_tools": count,
                "cached_tools": len(self._tool_cache),
                "search_count": self.search_count,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        
        except Exception as e:
            return {"error": f"Failed to get stats: {e}"}
    
    async def clear_collection(self) -> None:
        """Clear all embeddings from the collection."""
        if self.collection:
            # Get all IDs and delete them
            results = self.collection.get()
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
            
            self._tool_cache.clear()
            logger.info("Cleared all tool embeddings from collection")


class EnhancedSemanticSearchEngine:
    """
    Enhanced semantic search engine with advanced caching and ChromaDB optimization.
    
    This class provides high-performance semantic search with response caching,
    optimized ChromaDB configuration, and comprehensive performance monitoring.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        chromadb_config: Optional[ChromaDBConfig] = None,
        enable_caching: bool = True,
        cache_ttl: int = 3600,
        max_cache_size: int = 1000
    ):
        """
        Initialize enhanced semantic search engine.
        
        Args:
            embedding_service: Service for generating embeddings
            chromadb_config: ChromaDB configuration (uses default if None)
            enable_caching: Enable response caching
            cache_ttl: Cache TTL in seconds
            max_cache_size: Maximum cache size
        """
        self.embedding_service = embedding_service
        self.chromadb_config = chromadb_config or ChromaDBConfig()
        
        # Initialize managers
        self.chromadb_manager = ChromaDBManager(
            config=self.chromadb_config,
            embedding_service=embedding_service
        )
        
        self.cache_manager = ResponseCacheManager(
            max_size=max_cache_size,
            default_ttl=cache_ttl,
            enable_persistence=True,
            persistence_path=f"{self.chromadb_config.persist_directory}/search_cache.pkl"
        ) if enable_caching else None
        
        # Statistics
        self.search_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_search_time = 0.0
        
        logger.info("EnhancedSemanticSearchEngine initialized")
    
    async def initialize(self) -> None:
        """Initialize the enhanced semantic search engine."""
        logger.info("Initializing enhanced semantic search engine")
        
        # Initialize embedding service
        await self.embedding_service.initialize()
        
        # Initialize ChromaDB manager
        await self.chromadb_manager.initialize()
        
        # Initialize cache manager
        if self.cache_manager:
            await self.cache_manager.initialize()
        
        logger.info("Enhanced semantic search engine initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown the enhanced semantic search engine."""
        logger.info("Shutting down enhanced semantic search engine")
        
        # Shutdown managers
        await self.chromadb_manager.shutdown()
        
        if self.cache_manager:
            await self.cache_manager.shutdown()
        
        await self.embedding_service.shutdown()
        
        logger.info("Enhanced semantic search engine shutdown complete")
    
    async def add_tool_embeddings(
        self,
        tools: List[MCPTool],
        batch_size: Optional[int] = None,
        force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """
        Add tool embeddings with optimized batch processing.
        
        Args:
            tools: List of tools to add embeddings for
            batch_size: Batch size for processing
            force_regenerate: Force regeneration of existing embeddings
            
        Returns:
            Operation results
        """
        if not tools:
            return {"added": 0, "updated": 0, "errors": 0}
        
        logger.info(f"Adding embeddings for {len(tools)} tools")
        
        # Filter tools that need embeddings
        tools_to_process = []
        for tool in tools:
            if force_regenerate or not tool.embedding or not tool.embedding_updated:
                tools_to_process.append(tool)
        
        if not tools_to_process:
            logger.info("All tools already have embeddings")
            return {"added": 0, "updated": 0, "errors": 0}
        
        # Generate embeddings for tools that don't have them
        for tool in tools_to_process:
            if not tool.embedding:
                try:
                    tool_text = self._create_tool_text(tool)
                    tool.embedding = await self.embedding_service.generate_embedding(tool_text)
                    tool.embedding_updated = datetime.utcnow()
                except Exception as e:
                    logger.error(f"Failed to generate embedding for {tool.name}: {e}")
        
        # Add to ChromaDB
        result = await self.chromadb_manager.add_tool_embeddings(tools_to_process, batch_size)
        
        # Invalidate related cache entries
        if self.cache_manager:
            await self.cache_manager.invalidate_by_tags({"tool_search", "semantic_search"})
        
        logger.info(f"Tool embedding operation completed: {result}")
        return result
    
    async def search_relevant_tools(
        self,
        query: SemanticSearchQuery,
        use_cache: bool = True
    ) -> List[SemanticSearchResult]:
        """
        Search for relevant tools with caching and optimization.
        
        Args:
            query: Semantic search query
            use_cache: Whether to use caching
            
        Returns:
            List of relevant tools with similarity scores
        """
        start_time = asyncio.get_event_loop().time()
        self.search_count += 1
        
        try:
            # Check cache first
            cache_key = None
            if self.cache_manager and use_cache:
                cache_key = self._generate_search_cache_key(query)
                cached_results = await self.cache_manager.get(cache_key)
                
                if cached_results:
                    self.cache_hits += 1
                    logger.debug(f"Returning cached search results for query: {query.query[:50]}...")
                    return cached_results
                
                self.cache_misses += 1
            
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query.query)
            
            # Prepare metadata filter
            where_filter = {}
            if query.categories:
                where_filter["category"] = {"$in": query.categories}
            
            # Perform similarity search
            search_results = await self.chromadb_manager.search_similar_tools(
                query_embedding=query_embedding,
                n_results=query.max_results,
                where_filter=where_filter if where_filter else None,
                similarity_threshold=query.similarity_threshold
            )
            
            # Convert to SemanticSearchResult objects
            results = []
            for result in search_results:
                # Skip excluded tools
                if query.exclude_tools and result["tool_id"] in query.exclude_tools:
                    continue
                
                # Reconstruct tool from metadata
                tool = self._reconstruct_tool_from_result(result)
                if tool:
                    relevance_reason = self._generate_relevance_reason(
                        query, tool, result["similarity_score"]
                    )
                    
                    results.append(SemanticSearchResult(
                        tool=tool,
                        similarity_score=result["similarity_score"],
                        relevance_reason=relevance_reason
                    ))
            
            # Sort by similarity score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Cache results
            if self.cache_manager and use_cache and cache_key:
                cache_tags = {"tool_search", "semantic_search"}
                if query.categories:
                    cache_tags.update(f"category:{cat}" for cat in query.categories)
                
                await self.cache_manager.set(
                    cache_key,
                    results,
                    tags=cache_tags
                )
            
            # Update statistics
            execution_time = asyncio.get_event_loop().time() - start_time
            self.total_search_time += execution_time
            
            logger.debug(
                f"Semantic search returned {len(results)} results in {execution_time:.3f}s "
                f"for query: {query.query[:50]}..."
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Enhanced semantic search failed: {e}")
            raise
    
    async def update_tool_embedding(
        self,
        tool: MCPTool,
        invalidate_cache: bool = True
    ) -> None:
        """
        Update embedding for a single tool.
        
        Args:
            tool: Tool to update embedding for
            invalidate_cache: Whether to invalidate related cache entries
        """
        try:
            # Generate new embedding
            tool_text = self._create_tool_text(tool)
            tool.embedding = await self.embedding_service.generate_embedding(tool_text)
            tool.embedding_updated = datetime.utcnow()
            
            # Update in ChromaDB
            await self.chromadb_manager.add_tool_embeddings([tool])
            
            # Invalidate cache
            if self.cache_manager and invalidate_cache:
                await self.cache_manager.invalidate_by_tags({
                    "tool_search",
                    "semantic_search",
                    f"tool:{tool.name}"
                })
            
            logger.debug(f"Updated embedding for tool: {tool.name}")
            
        except Exception as e:
            logger.error(f"Failed to update tool embedding for {tool.name}: {e}")
            raise
    
    async def get_conversation_context(
        self,
        conversation_id: str,
        context_text: str
    ) -> Optional[List[float]]:
        """
        Get or create conversation context embedding.
        
        Args:
            conversation_id: Conversation identifier
            context_text: Context text to embed
            
        Returns:
            Context embedding vector
        """
        try:
            # Check cache first
            cache_key = f"context:{conversation_id}"
            if self.cache_manager:
                cached_embedding = await self.cache_manager.get(cache_key)
                if cached_embedding:
                    return cached_embedding
            
            # Generate context embedding
            context_embedding = await self.embedding_service.generate_embedding(context_text)
            
            # Store in ChromaDB
            metadata = {
                "conversation_id": conversation_id,
                "context_length": len(context_text),
                "created_at": datetime.utcnow().isoformat()
            }
            
            await self.chromadb_manager.add_conversation_context(
                conversation_id, context_embedding, metadata
            )
            
            # Cache the embedding
            if self.cache_manager:
                await self.cache_manager.set(
                    cache_key,
                    context_embedding,
                    ttl=7200,  # 2 hours
                    tags={"conversation_context"}
                )
            
            return context_embedding
            
        except Exception as e:
            logger.error(f"Failed to get conversation context for {conversation_id}: {e}")
            return None
    
    def _generate_search_cache_key(self, query: SemanticSearchQuery) -> str:
        """Generate cache key for search query."""
        key_data = {
            "query": query.query,
            "max_results": query.max_results,
            "similarity_threshold": query.similarity_threshold,
            "categories": sorted(query.categories) if query.categories else None,
            "exclude_tools": sorted(query.exclude_tools) if query.exclude_tools else None
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return f"search:{hash(key_str) % 2**32}"
    
    def _reconstruct_tool_from_result(self, result: Dict[str, Any]) -> Optional[MCPTool]:
        """Reconstruct MCPTool from search result."""
        try:
            metadata = result.get("metadata", {})
            
            tags = []
            if "tags" in metadata:
                try:
                    tags = json.loads(metadata["tags"])
                except (json.JSONDecodeError, TypeError):
                    tags = []
            
            tool = MCPTool(
                name=result["tool_id"],
                description=metadata.get("description", ""),
                server_name=metadata.get("server_name", "unknown"),
                input_schema={},  # Schema not stored in metadata
                category=metadata.get("category"),
                tags=tags,
                usage_count=metadata.get("usage_count", 0)
            )
            
            return tool
            
        except Exception as e:
            logger.error(f"Failed to reconstruct tool from result: {e}")
            return None
    
    def _create_tool_text(self, tool: MCPTool) -> str:
        """Create text representation of tool for embedding."""
        parts = [f"Tool: {tool.name}"]
        
        if tool.description:
            parts.append(f"Description: {tool.description}")
        
        if tool.category:
            parts.append(f"Category: {tool.category}")
        
        if tool.tags:
            parts.append(f"Tags: {', '.join(tool.tags)}")
        
        # Add schema information
        if tool.input_schema:
            schema_desc = self._describe_schema(tool.input_schema)
            if schema_desc:
                parts.append(f"Parameters: {schema_desc}")
        
        return " | ".join(parts)
    
    def _describe_schema(self, schema: Dict[str, Any]) -> str:
        """Create human-readable description of a JSON schema."""
        if not isinstance(schema, dict):
            return ""
        
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        descriptions = []
        for prop_name, prop_def in properties.items():
            prop_type = prop_def.get("type", "unknown")
            prop_desc = prop_def.get("description", "")
            
            is_required = prop_name in required
            req_marker = " (required)" if is_required else " (optional)"
            
            if prop_desc:
                descriptions.append(f"{prop_name} ({prop_type}): {prop_desc}{req_marker}")
            else:
                descriptions.append(f"{prop_name} ({prop_type}){req_marker}")
        
        return "; ".join(descriptions)
    
    def _generate_relevance_reason(
        self,
        query: SemanticSearchQuery,
        tool: MCPTool,
        similarity_score: float
    ) -> str:
        """Generate relevance reason for tool selection."""
        reasons = []
        
        # Score-based reason
        if similarity_score > 0.9:
            reasons.append("highly relevant")
        elif similarity_score > 0.8:
            reasons.append("very relevant")
        elif similarity_score > 0.7:
            reasons.append("relevant")
        else:
            reasons.append("potentially relevant")
        
        # Category match
        if query.categories and tool.category in query.categories:
            reasons.append(f"matches category '{tool.category}'")
        
        # Usage popularity
        if tool.usage_count > 10:
            reasons.append("frequently used")
        elif tool.usage_count > 0:
            reasons.append("previously used")
        
        return f"Tool is {', '.join(reasons)} (similarity: {similarity_score:.2f})"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "search_count": self.search_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(self.search_count, 1),
            "average_search_time": self.total_search_time / max(self.search_count, 1),
            "total_search_time": self.total_search_time
        }
        
        # Add ChromaDB stats
        stats["chromadb"] = self.chromadb_manager.get_stats()
        
        # Add cache stats
        if self.cache_manager:
            stats["cache"] = self.cache_manager.get_stats()
        
        # Add embedding service stats
        stats["embedding_service"] = self.embedding_service.get_stats()
        
        return stats
    
    async def cleanup(self) -> None:
        """Perform cleanup operations."""
        logger.info("Performing enhanced semantic search cleanup")
        
        # Cleanup cache
        if self.cache_manager:
            # Clear expired entries (handled automatically by cache manager)
            pass
        
        # ChromaDB maintenance is handled by its own maintenance loop
        
        logger.info("Enhanced semantic search cleanup completed")