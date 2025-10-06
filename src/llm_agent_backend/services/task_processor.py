"""
TaskProcessor for handling chat requests with async processing and caching.

This module provides specialized task processing for chat requests with
conversation context management, response caching, and proper error handling.
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..models.api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    TokenUsage,
)
from ..models.internal import AgentConfig, AgentTask, TaskStatus
from .agent_manager import AgentManager

logger = logging.getLogger(__name__)


class TaskProcessor:
    """Specialized processor for chat completion requests."""

    def __init__(
        self,
        agent_manager: Optional[AgentManager] = None,
        enable_caching: bool = True,
        cache_ttl: int = 3600,
        max_cache_size: int = 1000,
        max_context_length: int = 8192,
    ):
        """
        Initialize TaskProcessor.

        Args:
            agent_manager: AgentManager instance (creates new if None)
            enable_caching: Whether to enable response caching
            cache_ttl: Cache TTL in seconds
            max_cache_size: Maximum cache size
            max_context_length: Maximum context window size
        """
        self.agent_manager = agent_manager or AgentManager()
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self.max_context_length = max_context_length

        # Response cache with TTL
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Conversation context management
        self._conversations: Dict[str, List[ChatMessage]] = {}
        self._conversation_access: Dict[str, datetime] = {}

        # Statistics
        self._requests_processed = 0
        self._cache_hits = 0
        self._errors = 0
        self._start_time = time.time()

        logger.info(
            f"TaskProcessor initialized: caching={enable_caching}, cache_ttl={cache_ttl}s"
        )

    async def process_chat_request(
        self, request: ChatCompletionRequest, correlation_id: Optional[str] = None
    ) -> ChatCompletionResponse:
        """
        Process a chat completion request with caching and context management.

        Args:
            request: Chat completion request
            correlation_id: Optional correlation ID for tracking

        Returns:
            Chat completion response
        """
        start_time = time.time()

        try:
            logger.info(
                f"Processing chat request: model={request.model}, messages={len(request.messages)}"
            )

            # Check cache first
            if self.enable_caching:
                cache_key = self._generate_cache_key(request)
                cached_response = self._get_cached_response(cache_key)
                if cached_response:
                    self._cache_hits += 1
                    logger.info("Returning cached response")
                    return cached_response

            # Get conversation context
            conversation_id = self._get_conversation_id(request.messages)
            context_messages = self._get_conversation_context(conversation_id)

            # Combine context with new messages
            all_messages = self._merge_context_messages(
                context_messages, request.messages
            )

            # Create agent task with enhanced configuration
            config = AgentConfig(
                temperature=request.temperature,
                max_tools=getattr(request, "max_tools", 5),
                tool_selection_strategy="semantic",
                context_window=self.max_context_length,
            )

            task = AgentTask(
                correlation_id=correlation_id,
                messages=all_messages,
                tools=getattr(request, "tools", []),
                config=config,
            )

            # Submit and wait for completion
            task_id = await self.agent_manager.submit_task(task)
            response = await self._wait_for_task_completion(task_id, task)

            # Update conversation context
            self._update_conversation_context(
                conversation_id, request.messages, task.result
            )

            # Cache response if enabled
            if self.enable_caching and response:
                self._cache_response(cache_key, response)

            # Update statistics
            self._requests_processed += 1
            processing_time = time.time() - start_time

            logger.info(f"Chat request processed in {processing_time:.2f}s")
            return response

        except Exception as e:
            self._errors += 1
            logger.error(f"Error processing chat request: {e}", exc_info=True)
            raise

    def _generate_cache_key(self, request: ChatCompletionRequest) -> str:
        """Generate cache key from request."""
        key_data = f"{request.model}:{request.temperature}:{len(request.messages)}"
        for msg in request.messages[-2:]:  # Use last 2 messages for key
            key_data += f":{msg.role}:{msg.content[:100]}"  # Truncate content
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    def _get_conversation_id(self, messages: List[ChatMessage]) -> str:
        """Generate conversation ID from messages."""
        if not messages:
            return "default"

        # Use first message content for conversation grouping
        first_content = messages[0].content[:50] if messages else "default"
        return hashlib.md5(first_content.encode()).hexdigest()[:12]

    def _get_conversation_context(self, conversation_id: str) -> List[ChatMessage]:
        """Get conversation context messages."""
        self._conversation_access[conversation_id] = datetime.utcnow()
        return self._conversations.get(conversation_id, [])

    def _merge_context_messages(
        self, context_messages: List[ChatMessage], new_messages: List[ChatMessage]
    ) -> List[ChatMessage]:
        """Merge context with new messages, respecting context window."""
        all_messages = context_messages + new_messages

        # Estimate token count and trim if needed
        estimated_tokens = sum(len(msg.content) for msg in all_messages) // 4

        while estimated_tokens > self.max_context_length and len(all_messages) > len(
            new_messages
        ):
            # Remove oldest context message (but keep new messages)
            all_messages.pop(0)
            estimated_tokens = sum(len(msg.content) for msg in all_messages) // 4

        return all_messages

    def _update_conversation_context(
        self,
        conversation_id: str,
        request_messages: List[ChatMessage],
        result_message: Optional[ChatMessage],
    ) -> None:
        """Update conversation context with new messages."""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []

        # Add request messages
        for msg in request_messages:
            if msg not in self._conversations[conversation_id]:
                self._conversations[conversation_id].append(msg)

        # Add result message
        if result_message:
            self._conversations[conversation_id].append(result_message)

        # Keep only last 20 messages per conversation
        self._conversations[conversation_id] = self._conversations[conversation_id][
            -20:
        ]
        self._conversation_access[conversation_id] = datetime.utcnow()

    async def _wait_for_task_completion(
        self, task_id: str, task: AgentTask, timeout: int = 300
    ) -> ChatCompletionResponse:
        """Wait for task completion and convert to chat response."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            current_task = await self.agent_manager.get_task_status(task_id)

            if current_task is None:
                if task.is_completed():
                    break
                else:
                    raise Exception(f"Task {task_id} not found")

            if current_task.is_completed():
                # Update task with results
                task.status = current_task.status
                task.result = current_task.result
                task.error = current_task.error
                task.thinking_content = current_task.thinking_content
                task.total_tokens = current_task.total_tokens
                break

            await asyncio.sleep(0.1)
        else:
            await self.agent_manager.cancel_task(task_id)
            raise asyncio.TimeoutError(f"Task {task_id} timed out after {timeout}s")

        return self._convert_task_to_response(task)

    def _convert_task_to_response(self, task: AgentTask) -> ChatCompletionResponse:
        """Convert completed AgentTask to ChatCompletionResponse."""
        if task.status == TaskStatus.FAILED:
            raise Exception(f"Task failed: {task.error}")

        if task.status in [TaskStatus.CANCELLED, TaskStatus.TIMEOUT]:
            raise Exception(f"Task {task.status.value}")

        if not task.result:
            raise Exception("Task completed but has no result")

        choice = Choice(index=0, message=task.result, finish_reason="stop")

        usage = TokenUsage(
            prompt_tokens=task.total_tokens // 2 if task.total_tokens else 100,
            completion_tokens=task.total_tokens // 2 if task.total_tokens else 100,
            total_tokens=task.total_tokens or 200,
        )

        response = ChatCompletionResponse(
            id=task.id,
            object="chat.completion",
            created=int(task.created_at.timestamp()),
            model="Qwen/Qwen3-8B-AWQ",
            choices=[choice],
            usage=usage,
        )

        if task.thinking_content:
            response.thinking_content = task.thinking_content

        return response

    def _get_cached_response(self, cache_key: str) -> Optional[ChatCompletionResponse]:
        """Get cached response if not expired."""
        if cache_key not in self._cache:
            return None

        entry = self._cache[cache_key]
        if datetime.utcnow() > entry["expires_at"]:
            del self._cache[cache_key]
            return None

        return entry["response"]

    def _cache_response(self, cache_key: str, response: ChatCompletionResponse) -> None:
        """Cache response with TTL."""
        if len(self._cache) >= self.max_cache_size:
            # Simple LRU eviction: remove oldest entry
            oldest_key = min(
                self._cache.keys(), key=lambda k: self._cache[k]["created_at"]
            )
            del self._cache[oldest_key]

        self._cache[cache_key] = {
            "response": response,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(seconds=self.cache_ttl),
        }

    async def cleanup(self) -> None:
        """Perform cleanup tasks."""
        now = datetime.utcnow()

        # Clear expired cache entries
        expired_keys = [
            key for key, entry in self._cache.items() if now > entry["expires_at"]
        ]

        for key in expired_keys:
            del self._cache[key]

        # Clean up old conversations (older than 24 hours)
        cutoff_time = now - timedelta(hours=24)
        old_conversations = [
            conv_id
            for conv_id, last_access in self._conversation_access.items()
            if last_access < cutoff_time
        ]

        for conv_id in old_conversations:
            if conv_id in self._conversations:
                del self._conversations[conv_id]
            if conv_id in self._conversation_access:
                del self._conversation_access[conv_id]

        logger.info(
            f"Cleanup completed: removed {len(expired_keys)} cache entries, {len(old_conversations)} old conversations"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get TaskProcessor statistics."""
        uptime = time.time() - self._start_time

        return {
            "uptime_seconds": uptime,
            "requests_processed": self._requests_processed,
            "cache_hits": self._cache_hits,
            "errors": self._errors,
            "cache_size": len(self._cache),
            "active_conversations": len(self._conversations),
            "requests_per_minute": self._requests_processed / max(uptime / 60, 1),
            "error_rate": self._errors / max(self._requests_processed, 1),
            "cache_hit_rate": self._cache_hits / max(self._requests_processed, 1),
        }
