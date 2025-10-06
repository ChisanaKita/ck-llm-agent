"""
Semantic Search Engine with ChromaDB for tool selection.

This module implements semantic search capabilities using ChromaDB vector database
and vLLM embeddings API for intelligent tool selection based on context similarity.
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