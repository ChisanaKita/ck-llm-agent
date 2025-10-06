"""
Custom QwenVLLM handler for CrewAI integration.

This module provides a custom BaseLLM implementation that handles Qwen3-8B-AWQ
model inference through vLLM with OpenAI-compatible API calls, advanced connection pooling,
retry logic, and thinking content parsing. Enhanced with connection manager integration.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import aiohttp
from crewai.llm import BaseLLM
from pydantic import BaseModel, Field

from ..config import get_settings
from ..services.connection_manager import get_connection_manager, ConnectionPool

logger = logging.getLogger(__name__)


class ThinkingContentParser:
    """
    Parser for extracting and processing thinking content from model responses.

    Handles different thinking modes and formats used by various models,
    with specific support for Qwen3's thinking mode.
    """

    # Common thinking tag patterns
    THINKING_PATTERNS = [
        (r"<think>(.*?)</think>", "qwen3"),
        (r"<thinking>(.*?)</thinking>", "generic"),
        (r"<reasoning>(.*?)</reasoning>", "reasoning"),
        (r"<thought>(.*?)</thought>", "thought"),
        (r"<internal>(.*?)</internal>", "internal"),
    ]

    @classmethod
    def parse_response(
        cls,
        content: str,
        reasoning_content: Optional[str] = None,
        thinking_mode: bool = True,
    ) -> Tuple[str, Optional[str]]:
        """
        Parse response content and extract thinking information.

        Args:
            content: Raw response content
            reasoning_content: Pre-extracted reasoning content from vLLM
            thinking_mode: Whether thinking mode is enabled

        Returns:
            Tuple of (final_content, thinking_content)
        """
        if not thinking_mode:
            return content, None

        # If we have pre-extracted reasoning content, use it
        if reasoning_content:
            # Still clean the main content in case it contains thinking tags
            cleaned_content = cls._remove_thinking_tags(content)
            return cleaned_content, reasoning_content.strip()

        # Extract thinking content from the main content
        return cls._extract_thinking_from_content(content)

    @classmethod
    def _extract_thinking_from_content(cls, content: str) -> Tuple[str, Optional[str]]:
        """
        Extract thinking content using pattern matching.

        Args:
            content: Response content

        Returns:
            Tuple of (cleaned_content, thinking_content)
        """
        import re

        thinking_parts = []
        cleaned_content = content

        # Try each pattern and collect all thinking content
        for pattern, pattern_type in cls.THINKING_PATTERNS:
            matches = re.findall(pattern, cleaned_content, re.DOTALL)
            if matches:
                # Add pattern type prefix for debugging
                for match in matches:
                    thinking_parts.append(f"[{pattern_type}] {match.strip()}")

                # Remove thinking blocks from main content
                cleaned_content = re.sub(pattern, "", cleaned_content, flags=re.DOTALL)

        # Clean up the content
        cleaned_content = cls._clean_content(cleaned_content)

        # Combine thinking parts
        thinking_content = None
        if thinking_parts:
            thinking_content = "\n\n".join(thinking_parts)

        return cleaned_content, thinking_content

    @classmethod
    def _remove_thinking_tags(cls, content: str) -> str:
        """
        Remove all thinking tags from content without extracting content.

        Args:
            content: Content with potential thinking tags

        Returns:
            Cleaned content
        """
        import re

        cleaned = content
        for pattern, _ in cls.THINKING_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)

        return cls._clean_content(cleaned)

    @classmethod
    def _clean_content(cls, content: str) -> str:
        """
        Clean up content by removing extra whitespace and empty lines.

        Args:
            content: Content to clean

        Returns:
            Cleaned content
        """
        # Remove multiple consecutive newlines
        import re

        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)

        # Strip leading/trailing whitespace
        content = content.strip()

        return content

    @classmethod
    def format_thinking_content(cls, thinking_content: str) -> str:
        """
        Format thinking content for better readability.

        Args:
            thinking_content: Raw thinking content

        Returns:
            Formatted thinking content
        """
        if not thinking_content:
            return ""

        # Add timestamp
        import datetime

        timestamp = datetime.datetime.now().isoformat()

        formatted = f"=== Thinking Process ({timestamp}) ===\n\n"
        formatted += thinking_content
        formatted += "\n\n=== End Thinking ==="

        return formatted


class VLLMResponse(BaseModel):
    """Response model for vLLM API responses."""

    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    reasoning_content: Optional[str] = None


class RetryConfig(BaseModel):
    """Configuration for retry logic."""

    max_retries: int = Field(default=3, ge=0, le=10)
    base_delay: float = Field(default=1.0, ge=0.1, le=10.0)
    max_delay: float = Field(default=60.0, ge=1.0, le=300.0)
    exponential_base: float = Field(default=2.0, ge=1.1, le=5.0)
    jitter: bool = Field(default=True)


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for vLLM endpoint health monitoring.

    Prevents cascading failures by temporarily stopping requests to failing endpoints.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            success_threshold: Number of successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0

        logger.info(f"Circuit breaker initialized: threshold={failure_threshold}")

    async def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise Exception("Circuit breaker is OPEN - endpoint unavailable")
            else:
                # Try to recover
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception:
            await self._on_failure()
            raise

    async def _on_success(self):
        """Handle successful request."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED - service recovered")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0  # Reset failure count on success

    async def _on_failure(self):
        """Handle failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker OPEN - {self.failure_count} failures")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker back to OPEN - recovery failed")

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "is_available": self.state != CircuitBreakerState.OPEN
            or (time.time() - self.last_failure_time) >= self.recovery_timeout,
        }


class HealthMonitor:
    """
    Health monitoring for vLLM endpoints.

    Provides health checks, metrics collection, and endpoint status tracking.
    """

    def __init__(self, base_url: str, timeout: float = 5.0):
        """
        Initialize health monitor.

        Args:
            base_url: vLLM endpoint base URL
            timeout: Health check timeout
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.last_check_time = 0.0
        self.last_check_status = False
        self.check_count = 0
        self.success_count = 0

    async def check_health(self) -> Dict[str, Any]:
        """
        Perform health check on vLLM endpoint.

        Returns:
            Health check result
        """
        start_time = time.time()
        self.check_count += 1

        try:
            # Create a temporary session for health check
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Try to get models endpoint (lightweight check)
                url = urljoin(self.base_url, "/v1/models")

                async with session.get(url) as response:
                    response_time = (time.time() - start_time) * 1000

                    if response.status == 200:
                        self.success_count += 1
                        self.last_check_status = True

                        # Try to parse response to ensure it's valid
                        try:
                            data = await response.json()
                            model_count = len(data.get("data", []))
                        except Exception:
                            model_count = 0

                        result = {
                            "status": "healthy",
                            "response_time_ms": response_time,
                            "endpoint": self.base_url,
                            "model_count": model_count,
                            "timestamp": time.time(),
                        }
                    else:
                        self.last_check_status = False
                        result = {
                            "status": "unhealthy",
                            "response_time_ms": response_time,
                            "endpoint": self.base_url,
                            "error": f"HTTP {response.status}",
                            "timestamp": time.time(),
                        }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.last_check_status = False
            result = {
                "status": "unhealthy",
                "response_time_ms": response_time,
                "endpoint": self.base_url,
                "error": str(e),
                "timestamp": time.time(),
            }

        self.last_check_time = time.time()
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get health monitoring statistics."""
        return {
            "endpoint": self.base_url,
            "check_count": self.check_count,
            "success_count": self.success_count,
            "success_rate": self.success_count / max(self.check_count, 1),
            "last_check_time": self.last_check_time,
            "last_check_status": self.last_check_status,
        }


class QwenVLLM(BaseLLM):
    """
    Custom CrewAI BaseLLM implementation for Qwen3-8B-AWQ via vLLM.

    This class provides:
    - OpenAI-compatible API communication with vLLM
    - Connection pooling using aiohttp
    - Exponential backoff retry logic
    - Thinking content parsing and separation
    - Tool calling support
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
        thinking_mode: bool = True,
        retry_config: Optional[RetryConfig] = None,
        endpoint_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize QwenVLLM handler.

        Args:
            base_url: vLLM server base URL (defaults to config)
            model: Model name (defaults to config)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            thinking_mode: Whether to enable thinking mode
            retry_config: Retry configuration
            endpoint_config: Endpoint-specific configuration
            **kwargs: Additional parameters
        """
        super().__init__(model=model, **kwargs)

        settings = get_settings()
        chat_config = settings.vllm.get_chat_config()

        # Use provided config or fall back to settings
        self.base_url = base_url or chat_config.base_url
        self.model = model or chat_config.model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_mode = thinking_mode

        # Configure retry settings
        self.retry_config = retry_config or RetryConfig(
            max_retries=chat_config.max_retries, base_delay=1.0, max_delay=60.0
        )

        # Store endpoint configuration
        self.endpoint_config = endpoint_config or {}
        self.api_key = chat_config.api_key

        # Initialize connection pool using connection manager
        self.connection_pool: Optional[ConnectionPool] = None
        self._connection_manager_initialized = False

        # Initialize health monitoring and circuit breaker
        self.health_monitor = HealthMonitor(base_url=self.base_url, timeout=5.0)

        # Configure circuit breaker based on settings
        if settings.vllm.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=chat_config.circuit_breaker_threshold,
                recovery_timeout=chat_config.circuit_breaker_timeout,
                success_threshold=3,
            )
        else:
            self.circuit_breaker = None

        # Track request statistics
        self.request_count = 0
        self.error_count = 0
        self.total_tokens = 0

        # Detect endpoint type for logging
        endpoint_type = self._detect_endpoint_type(self.base_url)

        logger.info(
            f"Initialized QwenVLLM handler: model={self.model}, "
            f"endpoint_type={endpoint_type}, base_url={self.base_url}, "
            f"temperature={self.temperature}, thinking_mode={self.thinking_mode}"
        )

    async def _ensure_connection_pool(self) -> None:
        """Ensure connection pool is initialized."""
        if not self._connection_manager_initialized:
            settings = get_settings()
            chat_config = settings.vllm.get_chat_config()

            # Get connection manager and create pool
            connection_manager = await get_connection_manager()
            self.connection_pool = await connection_manager.get_or_create_pool(
                endpoint_config=chat_config, pool_id=f"qwen_chat_{hash(self.base_url)}"
            )
            self._connection_manager_initialized = True

    async def call(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make async call to vLLM with OpenAI-compatible request.

        Args:
            messages: List of chat messages
            tools: Available tools for function calling
            **kwargs: Additional parameters

        Returns:
            Dict containing response with separated thinking content
        """
        try:
            # Ensure connection pool is initialized
            await self._ensure_connection_pool()

            # Convert messages to proper format
            formatted_messages = self._format_messages(messages)

            # Build request payload
            payload = self._build_request_payload(
                messages=formatted_messages, tools=tools, **kwargs
            )

            # Make request with circuit breaker protection (if enabled)
            if self.circuit_breaker:
                response = await self.circuit_breaker.call(
                    self._make_request_with_retry, payload
                )
            else:
                response = await self._make_request_with_retry(payload)

            # Parse and return response
            return self._parse_response(response)

        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in QwenVLLM.call: {e}", exc_info=True)
            raise

    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling."""
        return True

    async def _make_request_with_retry(self, payload: Dict[str, Any]) -> VLLMResponse:
        """
        Make HTTP request to vLLM with exponential backoff retry.

        Args:
            payload: Request payload

        Returns:
            VLLMResponse object

        Raises:
            aiohttp.ClientError: If all retries fail
        """
        last_exception = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                response_data = await self._make_request(payload)
                self.request_count += 1

                # Update token statistics
                if "usage" in response_data:
                    self.total_tokens += response_data["usage"].get("total_tokens", 0)

                return VLLMResponse(**response_data)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e

                if attempt < self.retry_config.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.retry_config.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed: {e}")
                    self.error_count += 1

        # If we get here, all retries failed
        raise last_exception or Exception("Request failed after all retries")

    async def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make single HTTP request to vLLM endpoint.

        Args:
            payload: Request payload

        Returns:
            Response data as dict
        """
        if not self.connection_pool:
            raise RuntimeError("Connection pool not initialized")

        url = urljoin(self.base_url, "/v1/chat/completions")

        response = await self.connection_pool.make_request(
            method="POST", url=url, json=payload
        )

        if response.status != 200:
            error_text = await response.text()
            raise aiohttp.ClientResponseError(
                request_info=response.request_info,
                history=response.history,
                status=response.status,
                message=f"vLLM request failed: {error_text}",
            )

        return await response.json()

    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate delay for exponential backoff with jitter.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds
        """
        delay = min(
            self.retry_config.base_delay
            * (self.retry_config.exponential_base**attempt),
            self.retry_config.max_delay,
        )

        if self.retry_config.jitter:
            import random

            delay *= 0.5 + random.random() * 0.5  # Add 0-50% jitter

        return delay

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format messages for OpenAI-compatible API.

        Args:
            messages: Raw messages

        Returns:
            Formatted messages
        """
        formatted = []

        for msg in messages:
            if isinstance(msg, dict):
                formatted_msg = {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                }

                # Handle tool calls
                if "tool_calls" in msg:
                    formatted_msg["tool_calls"] = msg["tool_calls"]

                # Handle tool call ID for tool messages
                if msg.get("role") == "tool" and "tool_call_id" in msg:
                    formatted_msg["tool_call_id"] = msg["tool_call_id"]

                formatted.append(formatted_msg)
            else:
                # Handle string messages
                formatted.append({"role": "user", "content": str(msg)})

        return formatted

    def _build_request_payload(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Build OpenAI-compatible request payload.

        Args:
            messages: Formatted messages
            tools: Available tools
            **kwargs: Additional parameters

        Returns:
            Request payload
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False,  # We don't support streaming in this implementation
        }

        # Add max_tokens if specified
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Add tools if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Add other OpenAI parameters
        for param in ["top_p", "presence_penalty", "frequency_penalty", "stop"]:
            if param in kwargs:
                payload[param] = kwargs[param]

        return payload

    def _parse_response(self, response: VLLMResponse) -> Dict[str, Any]:
        """
        Parse vLLM response and separate thinking content.

        Args:
            response: VLLM response object

        Returns:
            Parsed response with thinking content separated
        """
        if not response.choices:
            raise ValueError("No choices in vLLM response")

        choice = response.choices[0]
        message = choice.get("message", {})

        # Extract content and thinking using the parser
        raw_content = message.get("content", "")
        reasoning_content = response.reasoning_content

        # Use ThinkingContentParser to handle both modes
        thinking_mode = getattr(self, "thinking_mode", True)
        content, thinking_content = ThinkingContentParser.parse_response(
            content=raw_content,
            reasoning_content=reasoning_content,
            thinking_mode=thinking_mode,
        )

        # Build response
        result = {
            "content": content,
            "thinking_content": thinking_content,
            "finish_reason": choice.get("finish_reason", "stop"),
            "usage": {
                "prompt_tokens": response.usage.get("prompt_tokens", 0),
                "completion_tokens": response.usage.get("completion_tokens", 0),
                "total_tokens": response.usage.get("total_tokens", 0),
            },
        }

        # Handle tool calls
        if "tool_calls" in message:
            result["tool_calls"] = message["tool_calls"]

        # Log thinking content for debugging (if present)
        if thinking_content:
            logger.debug(
                f"Extracted thinking content: {len(thinking_content)} characters"
            )

        return result

    async def check_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Health check results including endpoint status and circuit breaker state
        """
        health_result = await self.health_monitor.check_health()
        circuit_state = (
            self.circuit_breaker.get_state()
            if self.circuit_breaker
            else {"state": "disabled", "is_available": True}
        )

        # Additional endpoint information
        endpoint_info = {
            "base_url": self.base_url,
            "model": self.model,
            "endpoint_type": self._detect_endpoint_type(self.base_url),
            "thinking_mode": self.thinking_mode,
            "api_key_configured": bool(self.api_key),
            "connection_pool_size": self.connection_pool.pool_size,
            "timeout": self.connection_pool.timeout.total,
        }

        # Check if connection pool is healthy
        pool_healthy = True
        pool_info = {}
        try:
            session = await self.connection_pool.get_session()
            pool_info = {
                "session_closed": session.closed,
                "connector_limit": session.connector.limit
                if session.connector
                else None,
                "connector_limit_per_host": session.connector.limit_per_host
                if session.connector
                else None,
            }
        except Exception as ex:
            pool_healthy = False
            pool_info = {"error": str(ex)}

        return {
            "endpoint_health": health_result,
            "circuit_breaker": circuit_state,
            "endpoint_info": endpoint_info,
            "connection_pool": {"healthy": pool_healthy, "info": pool_info},
            "handler_stats": self.get_stats(),
            "overall_status": "healthy"
            if (
                health_result["status"] == "healthy"
                and circuit_state["is_available"]
                and pool_healthy
            )
            else "unhealthy",
        }

    async def is_healthy(self) -> bool:
        """
        Quick health check.

        Returns:
            True if endpoint is healthy and available
        """
        try:
            health = await self.check_health()
            return health["overall_status"] == "healthy"
        except Exception:
            return False

    def supports_external_endpoints(self) -> bool:
        """Check if handler supports external endpoints like Runpod."""
        return True

    def _detect_endpoint_type(self, url: str) -> str:
        """
        Detect the type of endpoint based on URL.

        Args:
            url: Endpoint URL

        Returns:
            Endpoint type string
        """
        url_lower = url.lower()

        if "runpod" in url_lower:
            return "runpod"
        elif "localhost" in url_lower or "127.0.0.1" in url_lower:
            return "local"
        elif any(
            cloud in url_lower for cloud in ["aws", "gcp", "azure", "huggingface"]
        ):
            return "cloud"
        else:
            return "external"

    def configure_external_endpoint(
        self, endpoint_url: str, api_key: Optional[str] = None, **kwargs
    ):
        """
        Configure external vLLM endpoint (e.g., Runpod, cloud providers).

        Args:
            endpoint_url: External endpoint URL
            api_key: API key if required
            **kwargs: Additional configuration options
                - pool_size: Connection pool size
                - timeout: Request timeout
                - health_timeout: Health check timeout
                - headers: Additional headers
                - verify_ssl: SSL verification (default: True)
        """
        old_url = self.base_url
        self.base_url = endpoint_url.rstrip("/")
        self.api_key = api_key

        endpoint_type = self._detect_endpoint_type(self.base_url)

        # Close existing connection pool
        asyncio.create_task(self.connection_pool.close())

        # Create new connection pool with updated configuration
        self.connection_pool = ConnectionPool(
            base_url=self.base_url,
            pool_size=kwargs.get("pool_size", 10),
            timeout=kwargs.get("timeout", 60),
            api_key=api_key,
            headers=kwargs.get("headers", {}),
        )

        # Update health monitor
        self.health_monitor = HealthMonitor(
            base_url=self.base_url, timeout=kwargs.get("health_timeout", 5.0)
        )

        # Reset circuit breaker state when changing endpoints
        if self.circuit_breaker:
            self.circuit_breaker.state = CircuitBreakerState.CLOSED
            self.circuit_breaker.failure_count = 0
            self.circuit_breaker.success_count = 0

        # Update endpoint configuration
        self.endpoint_config.update(kwargs)

        logger.info(
            f"Configured {endpoint_type} endpoint: {self.base_url} "
            f"(changed from {old_url})"
        )

        # Log additional configuration details
        if api_key:
            logger.info("API key authentication configured")
        if kwargs.get("headers"):
            logger.info(f"Custom headers configured: {list(kwargs['headers'].keys())}")

    def configure_runpod_endpoint(
        self, pod_id: str, api_key: str, region: str = "us-east-1", **kwargs
    ):
        """
        Configure Runpod-specific endpoint with proper URL formatting.

        Args:
            pod_id: Runpod instance ID
            api_key: Runpod API key
            region: Runpod region
            **kwargs: Additional configuration
        """
        # Construct Runpod URL
        runpod_url = f"https://{pod_id}-8000.proxy.runpod.net"

        # Runpod-specific headers
        headers = kwargs.get("headers", {})
        headers.update({"X-Runpod-Region": region, "X-Runpod-Pod-ID": pod_id})

        self.configure_external_endpoint(
            endpoint_url=runpod_url, api_key=api_key, headers=headers, **kwargs
        )

        logger.info(f"Configured Runpod endpoint: pod_id={pod_id}, region={region}")

    def configure_huggingface_endpoint(self, model_id: str, api_key: str, **kwargs):
        """
        Configure Hugging Face Inference Endpoints.

        Args:
            model_id: Hugging Face model ID
            api_key: Hugging Face API key
            **kwargs: Additional configuration
        """
        # Construct HF Inference Endpoint URL
        hf_url = f"https://api-inference.huggingface.co/models/{model_id}"

        # HF-specific headers
        headers = kwargs.get("headers", {})
        headers.update(
            {
                "X-Use-Cache": "false",  # Disable caching for real-time inference
            }
        )

        self.configure_external_endpoint(
            endpoint_url=hf_url, api_key=api_key, headers=headers, **kwargs
        )

        logger.info(f"Configured Hugging Face endpoint: model_id={model_id}")

    async def close(self):
        """Close connection pool and cleanup resources."""
        await self.connection_pool.close()
        logger.info("QwenVLLM handler closed")

    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        health_stats = self.health_monitor.get_stats()
        circuit_stats = (
            self.circuit_breaker.get_state()
            if self.circuit_breaker
            else {"state": "disabled"}
        )

        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "total_tokens": self.total_tokens,
            "error_rate": self.error_count / max(self.request_count, 1),
            "model": self.model,
            "base_url": self.base_url,
            "endpoint_type": self._detect_endpoint_type(self.base_url),
            "thinking_mode": self.thinking_mode,
            "api_key_configured": bool(self.api_key),
            "health_monitor": health_stats,
            "circuit_breaker": circuit_stats,
            "retry_config": {
                "max_retries": self.retry_config.max_retries,
                "base_delay": self.retry_config.base_delay,
                "max_delay": self.retry_config.max_delay,
            },
        }

    def get_endpoint_config(self) -> Dict[str, Any]:
        """
        Get current endpoint configuration.

        Returns:
            Current endpoint configuration
        """
        return {
            "base_url": self.base_url,
            "model": self.model,
            "endpoint_type": self._detect_endpoint_type(self.base_url),
            "api_key_configured": bool(self.api_key),
            "thinking_mode": self.thinking_mode,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "connection_pool_size": self.connection_pool.pool_size,
            "timeout": self.connection_pool.timeout.total,
            "retry_config": {
                "max_retries": self.retry_config.max_retries,
                "base_delay": self.retry_config.base_delay,
                "max_delay": self.retry_config.max_delay,
            },
            "circuit_breaker_enabled": self.circuit_breaker is not None,
            "health_monitoring_enabled": True,
            "custom_config": self.endpoint_config,
        }

    @classmethod
    def create_for_local(
        cls,
        base_url: str = "http://localhost:8000",
        model: str = "Qwen/Qwen3-8B-AWQ",
        **kwargs,
    ) -> "QwenVLLM":
        """
        Create QwenVLLM instance for local vLLM server.

        Args:
            base_url: Local vLLM server URL
            model: Model name
            **kwargs: Additional configuration

        Returns:
            Configured QwenVLLM instance
        """
        return cls(base_url=base_url, model=model, **kwargs)

    @classmethod
    def create_for_runpod(
        cls,
        pod_id: str,
        api_key: str,
        model: str = "Qwen/Qwen3-8B-AWQ",
        region: str = "us-east-1",
        **kwargs,
    ) -> "QwenVLLM":
        """
        Create QwenVLLM instance for Runpod endpoint.

        Args:
            pod_id: Runpod instance ID
            api_key: Runpod API key
            model: Model name
            region: Runpod region
            **kwargs: Additional configuration

        Returns:
            Configured QwenVLLM instance
        """
        runpod_url = f"https://{pod_id}-8000.proxy.runpod.net"

        headers = kwargs.get("headers", {})
        headers.update({"X-Runpod-Region": region, "X-Runpod-Pod-ID": pod_id})

        endpoint_config = kwargs.get("endpoint_config", {})
        endpoint_config["headers"] = headers
        endpoint_config["runpod_pod_id"] = pod_id
        endpoint_config["runpod_region"] = region

        return cls(
            base_url=runpod_url,
            model=model,
            endpoint_config=endpoint_config,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["headers", "endpoint_config"]
            },
        )

    @classmethod
    def create_for_external(
        cls, base_url: str, model: str, api_key: Optional[str] = None, **kwargs
    ) -> "QwenVLLM":
        """
        Create QwenVLLM instance for external endpoint.

        Args:
            base_url: External endpoint URL
            model: Model name
            api_key: API key if required
            **kwargs: Additional configuration

        Returns:
            Configured QwenVLLM instance
        """
        return cls(
            base_url=base_url,
            model=model,
            endpoint_config={"api_key": api_key} if api_key else {},
            **kwargs,
        )
