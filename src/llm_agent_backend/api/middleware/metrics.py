"""
Metrics collection middleware for Prometheus monitoring.

This module provides middleware for collecting application metrics
including request counts, response times, and system metrics.
"""

import logging
import time
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from ...config import get_settings


logger = logging.getLogger(__name__)


# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

REQUEST_SIZE = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    buckets=[100, 1000, 10000, 100000, 1000000]
)

RESPONSE_SIZE = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint'],
    buckets=[100, 1000, 10000, 100000, 1000000]
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Number of active HTTP requests'
)

# Application-specific metrics
CHAT_COMPLETIONS_TOTAL = Counter(
    'chat_completions_total',
    'Total chat completion requests',
    ['model', 'status']
)

CHAT_COMPLETION_DURATION = Histogram(
    'chat_completion_duration_seconds',
    'Chat completion processing time',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

TOKEN_USAGE_TOTAL = Counter(
    'token_usage_total',
    'Total tokens used',
    ['model', 'type']  # type: prompt, completion
)

TOOL_EXECUTIONS_TOTAL = Counter(
    'tool_executions_total',
    'Total tool executions',
    ['tool_name', 'status']
)

TOOL_EXECUTION_DURATION = Histogram(
    'tool_execution_duration_seconds',
    'Tool execution time',
    ['tool_name'],
    buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
)

# System metrics
SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes'
)

SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

AGENT_TASKS_ACTIVE = Gauge(
    'agent_tasks_active',
    'Number of active agent tasks'
)

AGENT_TASKS_TOTAL = Counter(
    'agent_tasks_total',
    'Total agent tasks processed',
    ['status']
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting HTTP and application metrics.
    
    Collects metrics for:
    - HTTP request/response statistics
    - Chat completion performance
    - Tool execution metrics
    - System resource usage
    """
    
    def __init__(self, app, **kwargs):
        """Initialize metrics middleware."""
        super().__init__(app, **kwargs)
        self.settings = get_settings()
        
        # Paths to exclude from detailed metrics
        self.exclude_paths = {
            "/metrics",
            "/health/live",
        }
        
        logger.info("Metrics middleware initialized")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with metrics collection.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
            
        Returns:
            Response: HTTP response
        """
        # Skip metrics for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Increment active requests
        ACTIVE_REQUESTS.inc()
        
        start_time = time.time()
        method = request.method
        path = self._normalize_path(request.url.path)
        
        # Get request size
        request_size = self._get_request_size(request)
        if request_size > 0:
            REQUEST_SIZE.labels(method=method, endpoint=path).observe(request_size)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Record metrics
            status_code = str(response.status_code)
            
            REQUEST_COUNT.labels(
                method=method,
                endpoint=path,
                status_code=status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=path
            ).observe(processing_time)
            
            # Get response size
            response_size = self._get_response_size(response)
            if response_size > 0:
                RESPONSE_SIZE.labels(
                    method=method,
                    endpoint=path
                ).observe(response_size)
            
            # Record chat completion metrics if applicable
            if path == "/v1/chat/completions" and response.status_code == 200:
                await self._record_chat_completion_metrics(request, response, processing_time)
            
            return response
            
        except Exception:
            # Record error metrics
            processing_time = time.time() - start_time
            
            REQUEST_COUNT.labels(
                method=method,
                endpoint=path,
                status_code="500"
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=path
            ).observe(processing_time)
            
            raise
        
        finally:
            # Decrement active requests
            ACTIVE_REQUESTS.dec()
    
    def _normalize_path(self, path: str) -> str:
        """
        Normalize path for metrics to avoid high cardinality.
        
        Args:
            path: Request path
            
        Returns:
            str: Normalized path
        """
        # Replace dynamic segments with placeholders
        if path.startswith("/v1/models/"):
            return "/v1/models/{model_id}"
        
        # Remove query parameters
        if "?" in path:
            path = path.split("?")[0]
        
        return path
    
    def _get_request_size(self, request: Request) -> int:
        """
        Get request content length.
        
        Args:
            request: HTTP request
            
        Returns:
            int: Request size in bytes
        """
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                return int(content_length)
            except ValueError:
                pass
        
        return 0
    
    def _get_response_size(self, response: Response) -> int:
        """
        Get response content length.
        
        Args:
            response: HTTP response
            
        Returns:
            int: Response size in bytes
        """
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                return int(content_length)
            except ValueError:
                pass
        
        return 0
    
    async def _record_chat_completion_metrics(
        self,
        request: Request,
        response: Response,
        processing_time: float
    ) -> None:
        """
        Record chat completion specific metrics.
        
        Args:
            request: HTTP request
            response: HTTP response
            processing_time: Processing time in seconds
        """
        try:
            # Extract model from request (would need to parse request body)
            # For now, use default model
            model = self.settings.vllm.chat.model
            
            # Record completion metrics
            CHAT_COMPLETIONS_TOTAL.labels(
                model=model,
                status="success"
            ).inc()
            
            CHAT_COMPLETION_DURATION.labels(
                model=model
            ).observe(processing_time)
            
            # Token usage would be extracted from response body
            # This would require parsing the response, which is complex in middleware
            # Better to record these metrics in the actual handler
            
        except Exception as e:
            logger.warning(f"Failed to record chat completion metrics: {e}")


class MetricsCollector:
    """
    Utility class for collecting application-specific metrics.
    
    Used by other components to record metrics outside of HTTP requests.
    """
    
    @staticmethod
    def record_chat_completion(
        model: str,
        status: str,
        duration: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ) -> None:
        """
        Record chat completion metrics.
        
        Args:
            model: Model name
            status: Completion status (success, error, timeout)
            duration: Processing duration in seconds
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
        """
        CHAT_COMPLETIONS_TOTAL.labels(model=model, status=status).inc()
        CHAT_COMPLETION_DURATION.labels(model=model).observe(duration)
        
        if prompt_tokens > 0:
            TOKEN_USAGE_TOTAL.labels(model=model, type="prompt").inc(prompt_tokens)
        
        if completion_tokens > 0:
            TOKEN_USAGE_TOTAL.labels(model=model, type="completion").inc(completion_tokens)
    
    @staticmethod
    def record_tool_execution(
        tool_name: str,
        status: str,
        duration: float
    ) -> None:
        """
        Record tool execution metrics.
        
        Args:
            tool_name: Name of the executed tool
            status: Execution status (success, error, timeout)
            duration: Execution duration in seconds
        """
        TOOL_EXECUTIONS_TOTAL.labels(tool_name=tool_name, status=status).inc()
        TOOL_EXECUTION_DURATION.labels(tool_name=tool_name).observe(duration)
    
    @staticmethod
    def record_agent_task(status: str) -> None:
        """
        Record agent task metrics.
        
        Args:
            status: Task status (completed, failed, timeout)
        """
        AGENT_TASKS_TOTAL.labels(status=status).inc()
    
    @staticmethod
    def set_active_tasks(count: int) -> None:
        """
        Set number of active agent tasks.
        
        Args:
            count: Number of active tasks
        """
        AGENT_TASKS_ACTIVE.set(count)
    
    @staticmethod
    def update_system_metrics() -> None:
        """Update system resource metrics."""
        try:
            import psutil
            
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(memory.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            SYSTEM_CPU_USAGE.set(cpu_percent)
            
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")


def get_metrics() -> str:
    """
    Get Prometheus metrics in text format.
    
    Returns:
        str: Prometheus metrics
    """
    # Update system metrics before returning
    MetricsCollector.update_system_metrics()
    
    return generate_latest().decode('utf-8')


def get_metrics_content_type() -> str:
    """
    Get Prometheus metrics content type.
    
    Returns:
        str: Content type for metrics response
    """
    return CONTENT_TYPE_LATEST