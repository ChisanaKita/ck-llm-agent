"""
ChromaDB Vector Database Manager with Optimized Configuration.

This module provides comprehensive ChromaDB management with optimized settings
for vLLM-generated embeddings, collection management, and performance monitoring.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from chromadb.api.types import Collection, CollectionMetadata
import numpy as np

from ..models.internal import MCPTool
from ..utils.logging import get_logger
from .semantic_search import EmbeddingService

logger = get_logger(__name__)


@dataclass
class ChromaDBConfig:
    """Configuration for ChromaDB optimization."""
    persist_directory: str = "./data/chromadb"
    embedding_dimension: int = 1024
    similarity_metric: str = "cosine"
    
    # HNSW parameters for optimal performance
    hnsw_construction_ef: int = 200
    hnsw_m: int = 16
    hnsw_search_ef: int = 100
    
    # Collection settings
    collection_name: str = "tool_embeddings"
    conversation_collection: str = "conversation_context"
    
    # Performance settings
    batch_size: int = 100
    max_concurrent_operations: int = 10
    
    # Maintenance settings
    cleanup_interval_hours: int = 24
    max_collection_age_days: int = 30


class ChromaDBStats:
    """Statistics tracking for ChromaDB operations."""
    
    def __init__(self):
        self.queries_executed = 0
        self.documents_added = 0
        self.documents_updated = 0
        self.documents_deleted = 0
        self.collections_created = 0
        self.total_query_time = 0.0
        self.total_insert_time = 0.0
        self.start_time = datetime.utcnow()
        self.last_maintenance = None
    
    def record_query(self, execution_time: float) -> None:
        """Record query execution."""
        self.queries_executed += 1
        self.total_query_time += execution_time
    
    def record_insert(self, count: int, execution_time: float) -> None:
        """Record document insertion."""
        self.documents_added += count
        self.total_insert_time += execution_time
    
    def record_update(self, count: int) -> None:
        """Record document update."""
        self.documents_updated += count
    
    def record_delete(self, count: int) -> None:
        """Record document deletion."""
        self.documents_deleted += count
    
    def record_collection_creation(self) -> None:
        """Record collection creation."""
        self.collections_created += 1
    
    def get_average_query_time(self) -> float:
        """Get average query execution time."""
        return self.total_query_time / max(self.queries_executed, 1)
    
    def get_average_insert_time(self) -> float:
        """Get average insert execution time."""
        return self.total_insert_time / max(self.documents_added, 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        uptime = datetime.utcnow() - self.start_time
        
        return {
            "queries_executed": self.queries_executed,
            "documents_added": self.documents_added,
            "documents_updated": self.documents_updated,
            "documents_deleted": self.documents_deleted,
            "collections_created": self.collections_created,
            "average_query_time_ms": self.get_average_query_time() * 1000,
            "average_insert_time_ms": self.get_average_insert_time() * 1000,
            "uptime_seconds": uptime.total_seconds(),
            "last_maintenance": self.last_maintenance.isoformat() if self.last_maintenance else None
        }


class ChromaDBManager:
    """
    Comprehensive ChromaDB manager with optimized configuration.
    
    Provides collection management, performance optimization, and maintenance
    operations for vector database operations with vLLM embeddings.
    """
    
    def __init__(
        self,
        config: ChromaDBConfig,
        embedding_service: Optional[EmbeddingService] = None
    ):
        """
        Initialize ChromaDB manager.
        
        Args:
            config: ChromaDB configuration
            embedding_service: Optional embedding service for operations
        """
        self.config = config
        self.embedding_service = embedding_service
        
        # ChromaDB components
        self.client: Optional[chromadb.PersistentClient] = None
        self.collections: Dict[str, Collection] = {}
        
        # Statistics and monitoring
        self.stats = ChromaDBStats()
        
        # Concurrency control
        self._operation_semaphore = asyncio.Semaphore(config.max_concurrent_operations)
        
        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"ChromaDBManager initialized with config: {config}")
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collections."""
        logger.info("Initializing ChromaDB manager")
        
        try:
            # Create persist directory
            persist_path = Path(self.config.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client with optimized settings
            self.client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False,
                    is_persistent=True
                )
            )
            
            # Initialize collections
            await self._initialize_collections()
            
            # Start maintenance task
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
            
            logger.info("ChromaDB manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB manager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown ChromaDB manager."""
        logger.info("Shutting down ChromaDB manager")
        
        self._shutdown_event.set()
        
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        
        # Perform final maintenance
        await self._perform_maintenance()
        
        logger.info("ChromaDB manager shutdown complete")
    
    async def _initialize_collections(self) -> None:
        """Initialize required collections with optimal settings."""
        if not self.client:
            raise RuntimeError("ChromaDB client not initialized")
        
        # Tool embeddings collection
        tool_collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata=self._get_collection_metadata("tool_embeddings")
        )
        self.collections[self.config.collection_name] = tool_collection
        self.stats.record_collection_creation()
        
        # Conversation context collection
        conversation_collection = self.client.get_or_create_collection(
            name=self.config.conversation_collection,
            metadata=self._get_collection_metadata("conversation_context")
        )
        self.collections[self.config.conversation_collection] = conversation_collection
        self.stats.record_collection_creation()
        
        logger.info(f"Initialized {len(self.collections)} collections")
    
    def _get_collection_metadata(self, collection_type: str) -> CollectionMetadata:
        """Get optimized metadata for collection type."""
        base_metadata = {
            "hnsw:space": self.config.similarity_metric,
            "hnsw:construction_ef": self.config.hnsw_construction_ef,
            "hnsw:M": self.config.hnsw_m,
            "hnsw:search_ef": self.config.hnsw_search_ef,
            "embedding_dimension": self.config.embedding_dimension,
            "created_at": datetime.utcnow().isoformat(),
            "collection_type": collection_type
        }
        
        return base_metadata
    
    async def add_tool_embeddings(
        self,
        tools: List[MCPTool],
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Add tool embeddings to ChromaDB with batch processing.
        
        Args:
            tools: List of tools with embeddings
            batch_size: Batch size for processing (uses config default if None)
            
        Returns:
            Operation results
        """
        async with self._operation_semaphore:
            return await self._add_tool_embeddings_internal(tools, batch_size)
    
    async def _add_tool_embeddings_internal(
        self,
        tools: List[MCPTool],
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Internal implementation of tool embedding addition."""
        if not tools:
            return {"added": 0, "updated": 0, "errors": 0}
        
        batch_size = batch_size or self.config.batch_size
        collection = self.collections.get(self.config.collection_name)
        
        if not collection:
            raise RuntimeError("Tool embeddings collection not initialized")
        
        start_time = time.time()
        added_count = 0
        updated_count = 0
        error_count = 0
        
        logger.info(f"Adding embeddings for {len(tools)} tools in batches of {batch_size}")
        
        try:
            # Process tools in batches
            for i in range(0, len(tools), batch_size):
                batch_tools = tools[i:i + batch_size]
                
                # Prepare batch data
                ids = []
                embeddings = []
                metadatas = []
                documents = []
                
                for tool in batch_tools:
                    if not tool.embedding:
                        if self.embedding_service:
                            # Generate embedding if service available
                            try:
                                tool_text = self._create_tool_text(tool)
                                tool.embedding = await self.embedding_service.generate_embedding(tool_text)
                                tool.embedding_updated = datetime.utcnow()
                            except Exception as e:
                                logger.error(f"Failed to generate embedding for {tool.name}: {e}")
                                error_count += 1
                                continue
                        else:
                            logger.warning(f"Tool {tool.name} has no embedding and no embedding service available")
                            error_count += 1
                            continue
                    
                    # Validate embedding
                    if not self._validate_embedding(tool.embedding):
                        logger.error(f"Invalid embedding for tool {tool.name}")
                        error_count += 1
                        continue
                    
                    ids.append(tool.name)
                    embeddings.append(tool.embedding)
                    metadatas.append(self._create_tool_metadata(tool))
                    documents.append(self._create_tool_text(tool))
                
                if ids:
                    # Check for existing entries
                    try:
                        existing_results = collection.get(ids=ids, include=[])
                        existing_ids = set(existing_results["ids"])
                        
                        new_ids = [id for id in ids if id not in existing_ids]
                        update_ids = [id for id in ids if id in existing_ids]
                        
                        # Upsert all entries (ChromaDB handles both insert and update)
                        collection.upsert(
                            ids=ids,
                            embeddings=embeddings,
                            metadatas=metadatas,
                            documents=documents
                        )
                        
                        added_count += len(new_ids)
                        updated_count += len(update_ids)
                        
                        logger.debug(f"Batch {i//batch_size + 1}: added {len(new_ids)}, updated {len(update_ids)}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                        error_count += len(batch_tools)
        
        except Exception as e:
            logger.error(f"Error in tool embedding addition: {e}")
            raise
        
        finally:
            execution_time = time.time() - start_time
            self.stats.record_insert(added_count + updated_count, execution_time)
            self.stats.record_update(updated_count)
        
        result = {
            "added": added_count,
            "updated": updated_count,
            "errors": error_count,
            "execution_time": execution_time
        }
        
        logger.info(f"Tool embedding operation completed: {result}")
        return result
    
    async def search_similar_tools(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar tools using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where_filter: Optional metadata filter
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar tools with metadata
        """
        async with self._operation_semaphore:
            return await self._search_similar_tools_internal(
                query_embedding, n_results, where_filter, similarity_threshold
            )
    
    async def _search_similar_tools_internal(
        self,
        query_embedding: List[float],
        n_results: int,
        where_filter: Optional[Dict[str, Any]],
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Internal implementation of similarity search."""
        collection = self.collections.get(self.config.collection_name)
        
        if not collection:
            raise RuntimeError("Tool embeddings collection not initialized")
        
        if not self._validate_embedding(query_embedding):
            raise ValueError("Invalid query embedding")
        
        start_time = time.time()
        
        try:
            # Perform similarity search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, 100),  # Limit to reasonable number
                where=where_filter,
                include=["metadatas", "distances", "documents"]
            )
            
            # Process results
            similar_tools = []
            
            if results["ids"] and results["ids"][0]:
                for i, tool_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    similarity_score = 1.0 - distance  # Convert distance to similarity
                    
                    # Apply similarity threshold
                    if similarity_score < similarity_threshold:
                        continue
                    
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    document = results["documents"][0][i] if results["documents"] else ""
                    
                    similar_tools.append({
                        "tool_id": tool_id,
                        "similarity_score": similarity_score,
                        "distance": distance,
                        "metadata": metadata,
                        "document": document
                    })
            
            execution_time = time.time() - start_time
            self.stats.record_query(execution_time)
            
            logger.debug(f"Similarity search returned {len(similar_tools)} results in {execution_time:.3f}s")
            return similar_tools
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise
    
    async def add_conversation_context(
        self,
        conversation_id: str,
        context_embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add conversation context to ChromaDB.
        
        Args:
            conversation_id: Unique conversation identifier
            context_embedding: Context embedding vector
            metadata: Context metadata
        """
        async with self._operation_semaphore:
            await self._add_conversation_context_internal(
                conversation_id, context_embedding, metadata
            )
    
    async def _add_conversation_context_internal(
        self,
        conversation_id: str,
        context_embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Internal implementation of conversation context addition."""
        collection = self.collections.get(self.config.conversation_collection)
        
        if not collection:
            raise RuntimeError("Conversation context collection not initialized")
        
        if not self._validate_embedding(context_embedding):
            raise ValueError("Invalid context embedding")
        
        try:
            # Add timestamp to metadata
            metadata.update({
                "conversation_id": conversation_id,
                "created_at": datetime.utcnow().isoformat(),
                "embedding_dimension": len(context_embedding)
            })
            
            # Store context
            collection.upsert(
                ids=[conversation_id],
                embeddings=[context_embedding],
                metadatas=[metadata],
                documents=[json.dumps(metadata)]
            )
            
            self.stats.record_insert(1, 0.0)
            logger.debug(f"Added conversation context: {conversation_id}")
            
        except Exception as e:
            logger.error(f"Failed to add conversation context {conversation_id}: {e}")
            raise
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection information
        """
        collection = self.collections.get(collection_name)
        
        if not collection:
            return {"error": f"Collection {collection_name} not found"}
        
        try:
            count = collection.count()
            metadata = collection.metadata
            
            return {
                "name": collection_name,
                "count": count,
                "metadata": metadata,
                "embedding_dimension": metadata.get("embedding_dimension", "unknown"),
                "similarity_metric": metadata.get("hnsw:space", "unknown"),
                "created_at": metadata.get("created_at", "unknown")
            }
            
        except Exception as e:
            return {"error": f"Failed to get collection info: {e}"}
    
    def _validate_embedding(self, embedding: List[float]) -> bool:
        """Validate embedding vector."""
        if not embedding:
            return False
        
        if len(embedding) != self.config.embedding_dimension:
            return False
        
        # Check for valid float values
        try:
            np_embedding = np.array(embedding, dtype=np.float32)
            if np.any(np.isnan(np_embedding)) or np.any(np.isinf(np_embedding)):
                return False
        except (ValueError, TypeError):
            return False
        
        return True
    
    def _create_tool_metadata(self, tool: MCPTool) -> Dict[str, Any]:
        """Create metadata for tool storage."""
        return {
            "name": tool.name,
            "description": tool.description or "",
            "server_name": tool.server_name,
            "category": tool.category or "general",
            "usage_count": tool.usage_count,
            "tags": json.dumps(tool.tags),
            "embedding_updated": tool.embedding_updated.isoformat() if tool.embedding_updated else None,
            "created_at": datetime.utcnow().isoformat()
        }
    
    def _create_tool_text(self, tool: MCPTool) -> str:
        """Create text representation of tool."""
        parts = [f"Tool: {tool.name}"]
        
        if tool.description:
            parts.append(f"Description: {tool.description}")
        
        if tool.category:
            parts.append(f"Category: {tool.category}")
        
        if tool.tags:
            parts.append(f"Tags: {', '.join(tool.tags)}")
        
        return " | ".join(parts)
    
    async def _maintenance_loop(self) -> None:
        """Background maintenance loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.cleanup_interval_hours * 3600
                )
            except asyncio.TimeoutError:
                await self._perform_maintenance()
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def _perform_maintenance(self) -> None:
        """Perform maintenance operations."""
        logger.info("Starting ChromaDB maintenance")
        
        try:
            # Clean up old conversation contexts
            await self._cleanup_old_conversations()
            
            # Update statistics
            self.stats.last_maintenance = datetime.utcnow()
            
            logger.info("ChromaDB maintenance completed")
            
        except Exception as e:
            logger.error(f"ChromaDB maintenance failed: {e}")
    
    async def _cleanup_old_conversations(self) -> None:
        """Clean up old conversation contexts."""
        collection = self.collections.get(self.config.conversation_collection)
        
        if not collection:
            return
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=self.config.max_collection_age_days)
            
            # Get all conversation contexts
            results = collection.get(include=["metadatas"])
            
            if not results["ids"]:
                return
            
            # Find old contexts
            old_ids = []
            for i, metadata in enumerate(results["metadatas"]):
                created_at_str = metadata.get("created_at")
                if created_at_str:
                    try:
                        created_at = datetime.fromisoformat(created_at_str)
                        if created_at < cutoff_time:
                            old_ids.append(results["ids"][i])
                    except ValueError:
                        # Invalid timestamp, consider it old
                        old_ids.append(results["ids"][i])
            
            # Delete old contexts
            if old_ids:
                collection.delete(ids=old_ids)
                self.stats.record_delete(len(old_ids))
                logger.info(f"Cleaned up {len(old_ids)} old conversation contexts")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old conversations: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics."""
        stats_dict = self.stats.to_dict()
        
        # Add collection information
        collection_info = {}
        for name, collection in self.collections.items():
            try:
                collection_info[name] = {
                    "count": collection.count(),
                    "metadata": collection.metadata
                }
            except Exception as e:
                collection_info[name] = {"error": str(e)}
        
        stats_dict["collections"] = collection_info
        stats_dict["config"] = {
            "persist_directory": self.config.persist_directory,
            "embedding_dimension": self.config.embedding_dimension,
            "similarity_metric": self.config.similarity_metric,
            "batch_size": self.config.batch_size
        }
        
        return stats_dict