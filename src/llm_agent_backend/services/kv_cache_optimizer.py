"""
KV Cache Optimization Service for vLLM Inference.

This module provides advanced KV cache optimization including request batching,
conversation state management, and cache warming strategies to maximize
inference efficiency and reduce latency.
"""

import asyncio
import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from ..models.api import ChatCompletionRequest, ChatMessage
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BatchingStrategy(Enum):
    """Request batching strategies."""
    SIMILARITY = "similarity"  # Batch by context similarity
    TEMPORAL = "temporal"  # Batch by time window
    HYBRID = "hybrid"  # Combination of similarity and temporal


@dataclass
class ConversationState:
    """Represents conversation state for KV cache optimization."""
    conversation_id: str
    messages: List[ChatMessage]
    last_updated: datetime
    access_count: int = 0
    cache_key: Optional[str] = None
    estimated_tokens: int = 0
    priority_score: float = 0.0
    
    def __post_init__(self):
        if not self.cache_key:
            self.cache_key = self._generate_cache_key()
        if self.estimated_tokens == 0:
            self.estimated_tokens = self._estimate_tokens()
    
    def _generate_cache_key(self) -> str:
        """Generate cache key for conversation state."""
        # Use first few messages to create stable key
        key_messages = self.messages[:3] if len(self.messages) >= 3 else self.messages
        key_data = [{"role": msg.role, "content": msg.content[:100]} for msg in key_messages]
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _estimate_tokens(self) -> int:
        """Estimate token count for conversation."""
        total_chars = sum(len(msg.content) for msg in self.messages)
        return total_chars // 4  # Rough estimation: 4 chars per token
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_updated = datetime.utcnow()
        self.priority_score = self._calculate_priority()
    
    def _calculate_priority(self) -> float:
        """Calculate priority score for cache retention."""
        # Factors: recency, frequency, conversation length
        recency_hours = (datetime.utcnow() - self.last_updated).total_seconds() / 3600
        recency_score = max(0, 1.0 - (recency_hours / 24))  # Decay over 24 hours
        
        frequency_score = min(1.0, self.access_count / 10)  # Normalize to 0-1
        
        length_score = min(1.0, len(self.messages) / 20)  # Longer conversations are more valuable
        
        return (recency_score * 0.4 + frequency_score * 0.4 + length_score * 0.2)
    
    def is_similar_to(self, other: 'ConversationState', threshold: float = 0.8) -> bool:
        """Check if conversation is similar to another for batching."""
        # Simple similarity based on message content overlap
        if not self.messages or not other.messages:
            return False
        
        # Compare last few messages
        self_content = " ".join(msg.content for msg in self.messages[-2:])
        other_content = " ".join(msg.content for msg in other.messages[-2:])
        
        # Simple Jaccard similarity on words
        self_words = set(self_content.lower().split())
        other_words = set(other_content.lower().split())
        
        if not self_words or not other_words:
            return False
        
        intersection = len(self_words.intersection(other_words))
        union = len(self_words.union(other_words))
        
        similarity = intersection / union if union > 0 else 0.0
        return similarity >= threshold


@dataclass
class RequestBatch:
    """Represents a batch of requests for processing."""
    batch_id: str
    requests: List[ChatCompletionRequest]
    conversation_states: List[ConversationState]
    created_at: datetime
    priority_score: float = 0.0
    estimated_tokens: int = 0
    processing_started: Optional[datetime] = None
    processing_completed: Optional[datetime] = None
    
    def __post_init__(self):
        if self.estimated_tokens == 0:
            self.estimated_tokens = sum(state.estimated_tokens for state in self.conversation_states)
        if self.priority_score == 0.0:
            self.priority_score = sum(state.priority_score for state in self.conversation_states) / len(self.conversation_states)
    
    def start_processing(self) -> None:
        """Mark batch as started processing."""
        self.processing_started = datetime.utcnow()
    
    def complete_processing(self) -> None:
        """Mark batch as completed processing."""
        self.processing_completed = datetime.utcnow()
    
    def get_processing_time(self) -> Optional[float]:
        """Get processing time in seconds."""
        if self.processing_started and self.processing_completed:
            return (self.processing_completed - self.processing_started).total_seconds()
        return None
    
    def is_compatible_with(self, request: ChatCompletionRequest, state: ConversationState) -> bool:
        """Check if request is compatible with this batch."""
        # Check model compatibility
        if self.requests and self.requests[0].model != request.model:
            return False
        
        # Check temperature compatibility (allow small variance)
        if self.requests:
            temp_diff = abs(self.requests[0].temperature - request.temperature)
            if temp_diff > 0.1:
                return False
        
        # Check if conversation states are similar
        for existing_state in self.conversation_states:
            if existing_state.is_similar_to(state):
                return True
        
        return False


class KVCacheOptimizer:
    """
    KV Cache optimization service for vLLM inference.
    
    Provides request batching, conversation state management, and cache warming
    strategies to maximize KV cache efficiency and reduce inference latency.
    """
    
    def __init__(
        self,
        batching_strategy: BatchingStrategy = BatchingStrategy.HYBRID,
        max_batch_size: int = 8,
        batch_timeout_ms: int = 100,
        max_conversation_length: int = 50,
        conversation_ttl_hours: int = 24,
        cache_warming_enabled: bool = True,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize KV cache optimizer.
        
        Args:
            batching_strategy: Strategy for request batching
            max_batch_size: Maximum requests per batch
            batch_timeout_ms: Maximum time to wait for batch completion
            max_conversation_length: Maximum messages per conversation
            conversation_ttl_hours: TTL for conversation states
            cache_warming_enabled: Enable cache warming
            similarity_threshold: Threshold for conversation similarity
        """
        self.batching_strategy = batching_strategy
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0  # Convert to seconds
        self.max_conversation_length = max_conversation_length
        self.conversation_ttl = timedelta(hours=conversation_ttl_hours)
        self.cache_warming_enabled = cache_warming_enabled
        self.similarity_threshold = similarity_threshold
        
        # Conversation state management
        self.conversation_states: Dict[str, ConversationState] = {}
        self.conversation_access_order: deque = deque()
        
        # Request batching
        self.pending_requests: List[Tuple[ChatCompletionRequest, ConversationState, asyncio.Future]] = []
        self.active_batches: Dict[str, RequestBatch] = {}
        self.batch_queue: asyncio.Queue = asyncio.Queue()
        
        # Cache warming
        self.warm_cache_patterns: List[ConversationState] = []
        self.warming_in_progress: Set[str] = set()
        
        # Statistics
        self.total_requests = 0
        self.batched_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_batches = 0
        self.average_batch_size = 0.0
        self.total_processing_time = 0.0
        
        # Background tasks
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cache_warmer_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"KVCacheOptimizer initialized with strategy: {batching_strategy.value}")
    
    async def initialize(self) -> None:
        """Initialize the KV cache optimizer."""
        logger.info("Initializing KV cache optimizer")
        
        # Start background tasks
        self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.cache_warming_enabled:
            self._cache_warmer_task = asyncio.create_task(self._cache_warmer_loop())
        
        logger.info("KV cache optimizer initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the KV cache optimizer."""
        logger.info("Shutting down KV cache optimizer")
        
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in [self._batch_processor_task, self._cleanup_task, self._cache_warmer_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Process any remaining requests
        await self._process_remaining_requests()
        
        logger.info("KV cache optimizer shutdown complete")
    
    async def optimize_request(
        self,
        request: ChatCompletionRequest,
        conversation_id: Optional[str] = None
    ) -> Tuple[ChatCompletionRequest, ConversationState]:
        """
        Optimize request for KV cache efficiency.
        
        Args:
            request: Chat completion request
            conversation_id: Optional conversation identifier
            
        Returns:
            Optimized request and conversation state
        """
        self.total_requests += 1
        
        # Get or create conversation state
        if not conversation_id:
            conversation_id = self._generate_conversation_id(request.messages)
        
        conversation_state = await self._get_or_create_conversation_state(
            conversation_id, request.messages
        )
        
        # Update conversation state
        conversation_state.update_access()
        
        # Optimize request based on conversation state
        optimized_request = await self._optimize_request_for_cache(request, conversation_state)
        
        return optimized_request, conversation_state
    
    async def submit_for_batching(
        self,
        request: ChatCompletionRequest,
        conversation_state: ConversationState
    ) -> Any:
        """
        Submit request for batched processing.
        
        Args:
            request: Chat completion request
            conversation_state: Conversation state
            
        Returns:
            Future that will contain the response
        """
        # Create future for response
        response_future = asyncio.Future()
        
        # Add to pending requests
        self.pending_requests.append((request, conversation_state, response_future))
        
        # Trigger batch processing if needed
        if len(self.pending_requests) >= self.max_batch_size:
            await self._trigger_batch_creation()
        
        return await response_future
    
    async def update_conversation_state(
        self,
        conversation_id: str,
        new_messages: List[ChatMessage],
        response_message: Optional[ChatMessage] = None
    ) -> None:
        """
        Update conversation state with new messages.
        
        Args:
            conversation_id: Conversation identifier
            new_messages: New messages to add
            response_message: Optional response message
        """
        if conversation_id not in self.conversation_states:
            return
        
        state = self.conversation_states[conversation_id]
        
        # Add new messages
        state.messages.extend(new_messages)
        if response_message:
            state.messages.append(response_message)
        
        # Trim conversation if too long
        if len(state.messages) > self.max_conversation_length:
            # Keep system message if present, then most recent messages
            system_messages = [msg for msg in state.messages if msg.role == "system"]
            other_messages = [msg for msg in state.messages if msg.role != "system"]
            
            keep_count = self.max_conversation_length - len(system_messages)
            trimmed_messages = system_messages + other_messages[-keep_count:]
            state.messages = trimmed_messages
        
        # Update metadata
        state.last_updated = datetime.utcnow()
        state.estimated_tokens = state._estimate_tokens()
        state.cache_key = state._generate_cache_key()
        
        # Update access order
        if conversation_id in self.conversation_access_order:
            self.conversation_access_order.remove(conversation_id)
        self.conversation_access_order.append(conversation_id)
    
    async def warm_cache_for_patterns(self, patterns: List[List[ChatMessage]]) -> int:
        """
        Warm cache for common conversation patterns.
        
        Args:
            patterns: List of message patterns to warm
            
        Returns:
            Number of patterns warmed
        """
        if not self.cache_warming_enabled:
            return 0
        
        warmed_count = 0
        
        for i, pattern in enumerate(patterns):
            pattern_id = f"warm_pattern_{i}"
            
            if pattern_id in self.warming_in_progress:
                continue
            
            try:
                self.warming_in_progress.add(pattern_id)
                
                # Create conversation state for pattern
                state = ConversationState(
                    conversation_id=pattern_id,
                    messages=pattern,
                    last_updated=datetime.utcnow()
                )
                
                self.warm_cache_patterns.append(state)
                warmed_count += 1
                
                logger.debug(f"Added cache warming pattern: {pattern_id}")
                
            except Exception as e:
                logger.error(f"Failed to warm cache for pattern {i}: {e}")
            finally:
                self.warming_in_progress.discard(pattern_id)
        
        logger.info(f"Cache warming: added {warmed_count} patterns")
        return warmed_count
    
    def _generate_conversation_id(self, messages: List[ChatMessage]) -> str:
        """Generate conversation ID from messages."""
        if not messages:
            return f"conv_{int(time.time() * 1000)}"
        
        # Use first message content for stable ID
        first_content = messages[0].content[:100] if messages else "default"
        return hashlib.md5(first_content.encode()).hexdigest()[:12]
    
    async def _get_or_create_conversation_state(
        self,
        conversation_id: str,
        messages: List[ChatMessage]
    ) -> ConversationState:
        """Get existing or create new conversation state."""
        if conversation_id in self.conversation_states:
            state = self.conversation_states[conversation_id]
            self.cache_hits += 1
        else:
            state = ConversationState(
                conversation_id=conversation_id,
                messages=messages.copy(),
                last_updated=datetime.utcnow()
            )
            self.conversation_states[conversation_id] = state
            self.cache_misses += 1
        
        # Update access order
        if conversation_id in self.conversation_access_order:
            self.conversation_access_order.remove(conversation_id)
        self.conversation_access_order.append(conversation_id)
        
        return state
    
    async def _optimize_request_for_cache(
        self,
        request: ChatCompletionRequest,
        conversation_state: ConversationState
    ) -> ChatCompletionRequest:
        """Optimize request for better KV cache utilization."""
        # Create optimized copy of request
        optimized_request = ChatCompletionRequest(
            messages=conversation_state.messages.copy(),
            model=request.model,
            temperature=request.temperature,
            max_tokens=getattr(request, 'max_tokens', None),
            tools=getattr(request, 'tools', None),
            thinking_mode=getattr(request, 'thinking_mode', True)
        )
        
        return optimized_request
    
    async def _trigger_batch_creation(self) -> None:
        """Trigger creation of a new batch from pending requests."""
        if not self.pending_requests:
            return
        
        if self.batching_strategy == BatchingStrategy.SIMILARITY:
            await self._create_similarity_batch()
        elif self.batching_strategy == BatchingStrategy.TEMPORAL:
            await self._create_temporal_batch()
        else:  # HYBRID
            await self._create_hybrid_batch()
    
    async def _create_similarity_batch(self) -> None:
        """Create batch based on conversation similarity."""
        if not self.pending_requests:
            return
        
        # Group requests by similarity
        similarity_groups = []
        remaining_requests = self.pending_requests.copy()
        
        while remaining_requests and len(similarity_groups) < self.max_batch_size:
            # Start new group with first request
            base_request, base_state, base_future = remaining_requests.pop(0)
            current_group = [(base_request, base_state, base_future)]
            
            # Find similar requests
            compatible_requests = []
            for i, (req, state, future) in enumerate(remaining_requests):
                if base_state.is_similar_to(state, self.similarity_threshold):
                    compatible_requests.append(i)
            
            # Add compatible requests to group
            for i in reversed(compatible_requests):  # Reverse to maintain indices
                current_group.append(remaining_requests.pop(i))
                if len(current_group) >= self.max_batch_size:
                    break
            
            similarity_groups.append(current_group)
        
        # Create batch from largest group
        if similarity_groups:
            largest_group = max(similarity_groups, key=len)
            await self._create_batch_from_group(largest_group)
    
    async def _create_temporal_batch(self) -> None:
        """Create batch based on temporal proximity."""
        if len(self.pending_requests) >= self.max_batch_size:
            # Take first N requests
            batch_requests = self.pending_requests[:self.max_batch_size]
            self.pending_requests = self.pending_requests[self.max_batch_size:]
            await self._create_batch_from_group(batch_requests)
    
    async def _create_hybrid_batch(self) -> None:
        """Create batch using hybrid strategy."""
        # First try similarity-based batching
        if len(self.pending_requests) >= 3:  # Need minimum for similarity
            await self._create_similarity_batch()
        else:
            # Fall back to temporal batching
            await self._create_temporal_batch()
    
    async def _create_batch_from_group(
        self,
        group: List[Tuple[ChatCompletionRequest, ConversationState, asyncio.Future]]
    ) -> None:
        """Create batch from a group of requests."""
        if not group:
            return
        
        batch_id = f"batch_{int(time.time() * 1000)}_{len(self.active_batches)}"
        
        requests = [req for req, _, _ in group]
        states = [state for _, state, _ in group]
        futures = [future for _, _, future in group]
        
        batch = RequestBatch(
            batch_id=batch_id,
            requests=requests,
            conversation_states=states,
            created_at=datetime.utcnow()
        )
        
        self.active_batches[batch_id] = batch
        self.total_batches += 1
        self.batched_requests += len(requests)
        self.average_batch_size = self.batched_requests / self.total_batches
        
        # Add to processing queue
        await self.batch_queue.put((batch, futures))
        
        logger.debug(f"Created batch {batch_id} with {len(requests)} requests")
    
    async def _batch_processor_loop(self) -> None:
        """Background loop for processing batches."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for batch or timeout
                try:
                    batch_data = await asyncio.wait_for(
                        self.batch_queue.get(),
                        timeout=self.batch_timeout
                    )
                    batch, futures = batch_data
                except asyncio.TimeoutError:
                    # Timeout - create batch from pending requests if any
                    if self.pending_requests:
                        await self._trigger_batch_creation()
                    continue
                
                # Process batch
                await self._process_batch(batch, futures)
                
            except Exception as e:
                logger.error(f"Error in batch processor loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_batch(
        self,
        batch: RequestBatch,
        futures: List[asyncio.Future]
    ) -> None:
        """Process a batch of requests."""
        batch.start_processing()
        
        try:
            # Here you would integrate with the actual vLLM inference
            # For now, we'll simulate batch processing
            logger.info(f"Processing batch {batch.batch_id} with {len(batch.requests)} requests")
            
            # Simulate processing time
            await asyncio.sleep(0.1 * len(batch.requests))
            
            # For each request in batch, set a mock response
            for i, future in enumerate(futures):
                if not future.done():
                    # In real implementation, this would be the actual response
                    mock_response = f"Batch response for request {i}"
                    future.set_result(mock_response)
            
            batch.complete_processing()
            processing_time = batch.get_processing_time()
            if processing_time:
                self.total_processing_time += processing_time
            
            logger.debug(f"Batch {batch.batch_id} processed in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch.batch_id}: {e}")
            
            # Set exception for all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)
        
        finally:
            # Remove from active batches
            self.active_batches.pop(batch.batch_id, None)
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=300  # 5 minutes
                )
            except asyncio.TimeoutError:
                await self._cleanup_expired_conversations()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_conversations(self) -> None:
        """Clean up expired conversation states."""
        cutoff_time = datetime.utcnow() - self.conversation_ttl
        expired_conversations = []
        
        for conv_id, state in self.conversation_states.items():
            if state.last_updated < cutoff_time:
                expired_conversations.append(conv_id)
        
        for conv_id in expired_conversations:
            del self.conversation_states[conv_id]
            if conv_id in self.conversation_access_order:
                self.conversation_access_order.remove(conv_id)
        
        if expired_conversations:
            logger.info(f"Cleaned up {len(expired_conversations)} expired conversations")
    
    async def _cache_warmer_loop(self) -> None:
        """Background cache warming loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=3600  # 1 hour
                )
            except asyncio.TimeoutError:
                await self._perform_cache_warming()
            except Exception as e:
                logger.error(f"Error in cache warmer loop: {e}")
                await asyncio.sleep(1800)  # 30 minutes
    
    async def _perform_cache_warming(self) -> None:
        """Perform cache warming operations."""
        if not self.warm_cache_patterns:
            return
        
        logger.info("Performing cache warming")
        
        # Process warming patterns
        for pattern in self.warm_cache_patterns[:5]:  # Limit to 5 patterns per cycle
            try:
                # Simulate warming by creating optimized requests
                # In real implementation, this would send requests to vLLM
                logger.debug(f"Warming cache for pattern: {pattern.conversation_id}")
                await asyncio.sleep(0.1)  # Simulate processing
                
            except Exception as e:
                logger.error(f"Cache warming failed for pattern {pattern.conversation_id}: {e}")
        
        logger.info("Cache warming cycle completed")
    
    async def _process_remaining_requests(self) -> None:
        """Process any remaining pending requests during shutdown."""
        if self.pending_requests:
            logger.info(f"Processing {len(self.pending_requests)} remaining requests")
            
            for request, state, future in self.pending_requests:
                if not future.done():
                    future.set_exception(RuntimeError("Service shutting down"))
            
            self.pending_requests.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get KV cache optimizer statistics."""
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        batch_efficiency = self.batched_requests / max(self.total_requests, 1)
        avg_processing_time = self.total_processing_time / max(self.total_batches, 1)
        
        return {
            "total_requests": self.total_requests,
            "batched_requests": self.batched_requests,
            "total_batches": self.total_batches,
            "average_batch_size": self.average_batch_size,
            "batch_efficiency": batch_efficiency,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "active_conversations": len(self.conversation_states),
            "pending_requests": len(self.pending_requests),
            "active_batches": len(self.active_batches),
            "warm_cache_patterns": len(self.warm_cache_patterns),
            "average_processing_time": avg_processing_time,
            "total_processing_time": self.total_processing_time,
            "batching_strategy": self.batching_strategy.value,
            "max_batch_size": self.max_batch_size,
            "batch_timeout_ms": self.batch_timeout * 1000
        }