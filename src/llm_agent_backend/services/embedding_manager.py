"""
ChromaDB Embedding Management Service.

This module provides comprehensive embedding management capabilities including
batch embedding generation, vector normalization, quality validation, and
synchronization with external vLLM embedding servers.
"""

import asyncio
import json
import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

from ..models.internal import MCPTool
from ..utils.logging import get_logger
from .semantic_search import EmbeddingService

logger = get_logger(__name__)


class EmbeddingQualityMetrics:
    """Metrics for embedding quality assessment."""

    def __init__(self):
        self.total_embeddings = 0
        self.valid_embeddings = 0
        self.invalid_embeddings = 0
        self.zero_embeddings = 0
        self.normalized_embeddings = 0
        self.average_magnitude = 0.0
        self.dimension_consistency = True
        self.last_validation = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_embeddings": self.total_embeddings,
            "valid_embeddings": self.valid_embeddings,
            "invalid_embeddings": self.invalid_embeddings,
            "zero_embeddings": self.zero_embeddings,
            "normalized_embeddings": self.normalized_embeddings,
            "average_magnitude": self.average_magnitude,
            "dimension_consistency": self.dimension_consistency,
            "quality_score": self.get_quality_score(),
            "last_validation": self.last_validation.isoformat()
            if self.last_validation
            else None,
        }

    def get_quality_score(self) -> float:
        """Calculate overall quality score (0.0 to 1.0)."""
        if self.total_embeddings == 0:
            return 0.0

        validity_score = self.valid_embeddings / self.total_embeddings
        non_zero_score = (
            self.total_embeddings - self.zero_embeddings
        ) / self.total_embeddings
        consistency_score = 1.0 if self.dimension_consistency else 0.5

        return validity_score * 0.5 + non_zero_score * 0.3 + consistency_score * 0.2


class EmbeddingBatch:
    """Represents a batch of embeddings for processing."""

    def __init__(
        self, texts: List[str], tool_ids: List[str], batch_id: Optional[str] = None
    ):
        """
        Initialize embedding batch.

        Args:
            texts: List of texts to embed
            tool_ids: Corresponding tool IDs
            batch_id: Optional batch identifier
        """
        self.texts = texts
        self.tool_ids = tool_ids
        self.batch_id = batch_id or f"batch_{int(time.time() * 1000)}"
        self.created_at = datetime.utcnow()
        self.embeddings: Optional[List[List[float]]] = None
        self.processing_time: Optional[float] = None
        self.success = False
        self.error: Optional[str] = None

    def __len__(self) -> int:
        """Get batch size."""
        return len(self.texts)

    def is_complete(self) -> bool:
        """Check if batch processing is complete."""
        return self.embeddings is not None or self.error is not None


class EmbeddingManager:
    """
    Comprehensive embedding management service for ChromaDB.

    This service handles batch embedding generation, vector normalization,
    quality validation, and synchronization with external vLLM servers.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        chromadb_client: chromadb.PersistentClient,
        collection_name: str = "tool_embeddings",
        batch_size: int = 10,
        max_concurrent_batches: int = 3,
        embedding_dimension: int = 1024,
        enable_normalization: bool = True,
        enable_quality_monitoring: bool = True,
    ):
        """
        Initialize the embedding manager.

        Args:
            embedding_service: Service for generating embeddings
            chromadb_client: ChromaDB client instance
            collection_name: Name of the ChromaDB collection
            batch_size: Size of embedding batches
            max_concurrent_batches: Maximum concurrent batch processing
            embedding_dimension: Expected embedding dimension
            enable_normalization: Enable vector normalization
            enable_quality_monitoring: Enable embedding quality monitoring
        """
        self.embedding_service = embedding_service
        self.chromadb_client = chromadb_client
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.embedding_dimension = embedding_dimension
        self.enable_normalization = enable_normalization
        self.enable_quality_monitoring = enable_quality_monitoring

        # ChromaDB collection
        self.collection: Optional[chromadb.Collection] = None

        # Concurrency control
        self._batch_semaphore = asyncio.Semaphore(max_concurrent_batches)
        self._processing_batches: Dict[str, EmbeddingBatch] = {}

        # Quality monitoring
        self.quality_metrics = EmbeddingQualityMetrics()

        # Statistics
        self.total_batches_processed = 0
        self.successful_batches = 0
        self.failed_batches = 0
        self.total_embeddings_generated = 0
        self.total_processing_time = 0.0

        # Synchronization tracking
        self.last_sync_time: Optional[datetime] = None
        self.sync_in_progress = False

    async def initialize(self) -> None:
        """Initialize the embedding manager."""
        logger.info("Initializing embedding manager")

        try:
            # Get or create ChromaDB collection
            self.collection = self.chromadb_client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16,
                    "embedding_dimension": self.embedding_dimension,
                },
            )

            # Initialize embedding service
            await self.embedding_service.initialize()

            logger.info(
                f"Embedding manager initialized with collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize embedding manager: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the embedding manager."""
        # Wait for any ongoing batch processing
        while self._processing_batches:
            await asyncio.sleep(0.1)

        await self.embedding_service.shutdown()

        logger.info("Embedding manager shutdown complete")

    async def generate_embeddings_batch(
        self, tools: List[MCPTool], force_regenerate: bool = False
    ) -> List[str]:
        """
        Generate embeddings for a batch of tools.

        Args:
            tools: List of MCP tools to generate embeddings for
            force_regenerate: Force regeneration even if embeddings exist

        Returns:
            List of batch IDs for tracking
        """
        if not tools:
            return []

        logger.info(f"Generating embeddings for {len(tools)} tools")

        # Filter tools that need embeddings
        tools_to_process = []
        for tool in tools:
            if force_regenerate or not tool.embedding or not tool.embedding_updated:
                tools_to_process.append(tool)

        if not tools_to_process:
            logger.info("All tools already have embeddings")
            return []

        # Create batches
        batches = self._create_batches(tools_to_process)
        batch_ids = []

        # Process batches concurrently
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_batch(batch))
            tasks.append(task)
            batch_ids.append(batch.batch_id)

        # Wait for all batches to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update tools with embeddings
        successful_batches = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed: {result}")
            else:
                batch = batches[i]
                if batch.success and batch.embeddings:
                    await self._update_tools_with_embeddings(batch, tools_to_process)
                    successful_batches += 1

        logger.info(
            f"Successfully processed {successful_batches}/{len(batches)} batches"
        )
        return batch_ids

    def _create_batches(self, tools: List[MCPTool]) -> List[EmbeddingBatch]:
        """
        Create embedding batches from tools.

        Args:
            tools: List of tools to create batches for

        Returns:
            List of embedding batches
        """
        batches = []

        for i in range(0, len(tools), self.batch_size):
            batch_tools = tools[i : i + self.batch_size]

            texts = []
            tool_ids = []

            for tool in batch_tools:
                text = self._create_tool_text(tool)
                texts.append(text)
                tool_ids.append(tool.name)

            batch = EmbeddingBatch(texts, tool_ids)
            batches.append(batch)

        return batches

    async def _process_batch(self, batch: EmbeddingBatch) -> None:
        """
        Process a single embedding batch.

        Args:
            batch: Embedding batch to process
        """
        async with self._batch_semaphore:
            self._processing_batches[batch.batch_id] = batch

            try:
                start_time = time.time()

                # Generate embeddings
                embeddings = await self.embedding_service.generate_embeddings_batch(
                    batch.texts
                )

                # Validate and normalize embeddings
                validated_embeddings = []
                for embedding in embeddings:
                    validated_embedding = self._validate_and_normalize_embedding(
                        embedding
                    )
                    validated_embeddings.append(validated_embedding)

                batch.embeddings = validated_embeddings
                batch.processing_time = time.time() - start_time
                batch.success = True

                # Update statistics
                self.total_batches_processed += 1
                self.successful_batches += 1
                self.total_embeddings_generated += len(embeddings)
                self.total_processing_time += batch.processing_time

                logger.debug(
                    f"Batch {batch.batch_id} processed successfully in {batch.processing_time:.2f}s"
                )

            except Exception as e:
                batch.error = str(e)
                batch.processing_time = (
                    time.time() - start_time if "start_time" in locals() else 0.0
                )
                self.failed_batches += 1

                logger.error(f"Batch {batch.batch_id} processing failed: {e}")

            finally:
                self._processing_batches.pop(batch.batch_id, None)

    def _validate_and_normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Validate and optionally normalize an embedding vector.

        Args:
            embedding: Raw embedding vector

        Returns:
            Validated and normalized embedding
        """
        if self.enable_quality_monitoring:
            self.quality_metrics.total_embeddings += 1

        # Validate dimension
        if len(embedding) != self.embedding_dimension:
            if self.enable_quality_monitoring:
                self.quality_metrics.invalid_embeddings += 1
                self.quality_metrics.dimension_consistency = False

            logger.warning(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(embedding)}"
            )

            # Pad or truncate to expected dimension
            if len(embedding) < self.embedding_dimension:
                embedding.extend([0.0] * (self.embedding_dimension - len(embedding)))
            else:
                embedding = embedding[: self.embedding_dimension]

        # Convert to numpy array for processing
        embedding_array = np.array(embedding, dtype=np.float32)

        # Check for zero vector
        magnitude = np.linalg.norm(embedding_array)
        if magnitude == 0.0:
            if self.enable_quality_monitoring:
                self.quality_metrics.zero_embeddings += 1

            logger.warning("Zero embedding vector detected")
            # Replace with small random vector
            embedding_array = np.random.normal(
                0, 0.01, self.embedding_dimension
            ).astype(np.float32)
            magnitude = np.linalg.norm(embedding_array)

        # Normalize if enabled
        if self.enable_normalization and magnitude > 0:
            embedding_array = embedding_array / magnitude
            if self.enable_quality_monitoring:
                self.quality_metrics.normalized_embeddings += 1

        # Update quality metrics
        if self.enable_quality_monitoring:
            self.quality_metrics.valid_embeddings += 1
            self.quality_metrics.average_magnitude = (
                self.quality_metrics.average_magnitude
                * (self.quality_metrics.valid_embeddings - 1)
                + magnitude
            ) / self.quality_metrics.valid_embeddings

        return embedding_array.tolist()

    async def _update_tools_with_embeddings(
        self, batch: EmbeddingBatch, tools: List[MCPTool]
    ) -> None:
        """
        Update tools with generated embeddings and store in ChromaDB.

        Args:
            batch: Processed embedding batch
            tools: List of tools to update
        """
        if not batch.embeddings or not self.collection:
            return

        try:
            # Create mapping from tool ID to tool
            tool_map = {tool.name: tool for tool in tools}

            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents = []

            for i, (tool_id, embedding) in enumerate(
                zip(batch.tool_ids, batch.embeddings)
            ):
                tool = tool_map.get(tool_id)
                if not tool:
                    continue

                # Update tool object
                tool.embedding = embedding
                tool.embedding_updated = datetime.utcnow()

                # Prepare ChromaDB data
                ids.append(tool_id)
                embeddings.append(embedding)

                metadata = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "server_name": tool.server_name,
                    "category": tool.category or "general",
                    "usage_count": tool.usage_count,
                    "tags": json.dumps(tool.tags),
                    "embedding_updated": tool.embedding_updated.isoformat(),
                }
                metadatas.append(metadata)

                document = self._create_tool_text(tool)
                documents.append(document)

            # Upsert to ChromaDB
            if ids:
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )

                logger.debug(f"Updated {len(ids)} tool embeddings in ChromaDB")

        except Exception as e:
            logger.error(f"Failed to update tools with embeddings: {e}")
            raise

    def _create_tool_text(self, tool: MCPTool) -> str:
        """
        Create text representation of a tool for embedding.

        Args:
            tool: Tool to create text for

        Returns:
            Text representation
        """
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
        """
        Create human-readable description of a JSON schema.

        Args:
            schema: JSON schema to describe

        Returns:
            Human-readable description
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
                descriptions.append(
                    f"{prop_name} ({prop_type}): {prop_desc}{req_marker}"
                )
            else:
                descriptions.append(f"{prop_name} ({prop_type}){req_marker}")

        return "; ".join(descriptions)

    async def synchronize_embeddings(
        self, tools: List[MCPTool], max_age_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Synchronize embeddings with external vLLM server.

        Args:
            tools: List of tools to synchronize
            max_age_hours: Maximum age of embeddings before regeneration

        Returns:
            Synchronization results
        """
        if self.sync_in_progress:
            return {"status": "sync_already_in_progress"}

        self.sync_in_progress = True

        try:
            logger.info(f"Starting embedding synchronization for {len(tools)} tools")

            # Determine which tools need updates
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            tools_to_update = []

            for tool in tools:
                needs_update = (
                    not tool.embedding
                    or not tool.embedding_updated
                    or tool.embedding_updated < cutoff_time
                )

                if needs_update:
                    tools_to_update.append(tool)

            if not tools_to_update:
                logger.info("All embeddings are up to date")
                return {
                    "status": "up_to_date",
                    "total_tools": len(tools),
                    "updated_tools": 0,
                }

            # Generate embeddings for outdated tools
            batch_ids = await self.generate_embeddings_batch(
                tools_to_update, force_regenerate=True
            )

            # Wait for all batches to complete
            while any(batch_id in self._processing_batches for batch_id in batch_ids):
                await asyncio.sleep(0.1)

            self.last_sync_time = datetime.utcnow()

            logger.info(
                f"Synchronization completed: updated {len(tools_to_update)} tools"
            )

            return {
                "status": "completed",
                "total_tools": len(tools),
                "updated_tools": len(tools_to_update),
                "batch_ids": batch_ids,
                "sync_time": self.last_sync_time.isoformat(),
            }

        except Exception as e:
            logger.error(f"Embedding synchronization failed: {e}")
            return {"status": "failed", "error": str(e)}

        finally:
            self.sync_in_progress = False

    async def validate_embedding_quality(self) -> EmbeddingQualityMetrics:
        """
        Validate the quality of stored embeddings.

        Returns:
            Quality metrics
        """
        if not self.collection:
            raise RuntimeError("ChromaDB collection not initialized")

        logger.info("Validating embedding quality")

        try:
            # Reset metrics
            self.quality_metrics = EmbeddingQualityMetrics()

            # Get all embeddings from collection
            results = self.collection.get(include=["embeddings", "metadatas"])

            if not results["embeddings"]:
                logger.warning("No embeddings found in collection")
                return self.quality_metrics

            # Validate each embedding
            for embedding in results["embeddings"]:
                self.quality_metrics.total_embeddings += 1

                # Check dimension
                if len(embedding) != self.embedding_dimension:
                    self.quality_metrics.invalid_embeddings += 1
                    self.quality_metrics.dimension_consistency = False
                else:
                    self.quality_metrics.valid_embeddings += 1

                # Check for zero vector
                magnitude = math.sqrt(sum(x * x for x in embedding))
                if magnitude == 0.0:
                    self.quality_metrics.zero_embeddings += 1

                # Update average magnitude
                if self.quality_metrics.valid_embeddings > 0:
                    self.quality_metrics.average_magnitude = (
                        self.quality_metrics.average_magnitude
                        * (self.quality_metrics.valid_embeddings - 1)
                        + magnitude
                    ) / self.quality_metrics.valid_embeddings

            self.quality_metrics.last_validation = datetime.utcnow()

            logger.info(
                f"Quality validation completed: {self.quality_metrics.valid_embeddings}/"
                f"{self.quality_metrics.total_embeddings} valid embeddings "
                f"(quality score: {self.quality_metrics.get_quality_score():.2f})"
            )

            return self.quality_metrics

        except Exception as e:
            logger.error(f"Embedding quality validation failed: {e}")
            raise

    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific batch.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch status information
        """
        batch = self._processing_batches.get(batch_id)
        if not batch:
            return None

        return {
            "batch_id": batch.batch_id,
            "size": len(batch),
            "created_at": batch.created_at.isoformat(),
            "is_complete": batch.is_complete(),
            "success": batch.success,
            "processing_time": batch.processing_time,
            "error": batch.error,
        }

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get embedding processing statistics.

        Returns:
            Processing statistics
        """
        avg_processing_time = (
            self.total_processing_time / self.total_batches_processed
            if self.total_batches_processed > 0
            else 0.0
        )

        success_rate = (
            self.successful_batches / self.total_batches_processed
            if self.total_batches_processed > 0
            else 0.0
        )

        return {
            "total_batches_processed": self.total_batches_processed,
            "successful_batches": self.successful_batches,
            "failed_batches": self.failed_batches,
            "success_rate": success_rate,
            "total_embeddings_generated": self.total_embeddings_generated,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "active_batches": len(self._processing_batches),
            "last_sync_time": self.last_sync_time.isoformat()
            if self.last_sync_time
            else None,
            "sync_in_progress": self.sync_in_progress,
            "quality_metrics": self.quality_metrics.to_dict(),
        }

    async def cleanup_old_embeddings(self, max_age_days: int = 30) -> int:
        """
        Clean up old embeddings from ChromaDB.

        Args:
            max_age_days: Maximum age of embeddings to keep

        Returns:
            Number of embeddings removed
        """
        if not self.collection:
            return 0

        try:
            cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)

            # Get all embeddings with metadata
            results = self.collection.get(include=["metadatas"])

            if not results["ids"]:
                return 0

            # Find old embeddings
            old_ids = []
            for i, metadata in enumerate(results["metadatas"]):
                embedding_updated_str = metadata.get("embedding_updated")
                if embedding_updated_str:
                    try:
                        embedding_updated = datetime.fromisoformat(
                            embedding_updated_str
                        )
                        if embedding_updated < cutoff_time:
                            old_ids.append(results["ids"][i])
                    except ValueError:
                        # Invalid timestamp, consider it old
                        old_ids.append(results["ids"][i])

            # Delete old embeddings
            if old_ids:
                self.collection.delete(ids=old_ids)
                logger.info(f"Cleaned up {len(old_ids)} old embeddings")

            return len(old_ids)

        except Exception as e:
            logger.error(f"Failed to cleanup old embeddings: {e}")
            return 0
