"""
Structured logging middleware with correlation IDs.

This module provides comprehensive request/response logging with
correlation IDs for tracing requests through the system.
"""

import json
import logging
import time
import uuid
from typing import Dict, Any, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...config import get_settings


logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured request/response logging.
    
    Provides:
    - Request correlation IDs
    - Structured JSON logging
    - Request/response timing
    - Sensitive data filtering
    - Configurable log levels
    """
    
    def __init__(self, app, **kwargs):
        """Initialize logging middleware."""
        super().__init__(app, **kwargs)
        self.settings = get_settings()
        
        # Paths to exclude from access logging
        self.exclude_paths = {
            "/health/live",
            "/health/ready",
        }
        
        # Headers to exclude from logging (sensitive data)
        self.sensitive_headers = {
            "authorization",
            "x-api-key",
            "cookie",
            "set-cookie",
        }
        
        # Request body fields to exclude (sensitive data)
        self.sensitive_body_fields = {
            "password",
            "token",
            "secret",
            "key",
            "api_key",
        }
        
        logger.info("Logging middleware initialized")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with structured logging.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
            
        Returns:
            Response: HTTP response
        """
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.request_id = correlation_id
        
        # Skip detailed logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        start_time = time.time()
        
        # Log request
        await self._log_request(request, correlation_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log response
            await self._log_response(request, response, correlation_id, processing_time)
            
            # Add correlation ID to response headers
            response.headers["X-Request-ID"] = correlation_id
            
            return response
            
        except Exception as e:
            # Log error
            processing_time = time.time() - start_time
            await self._log_error(request, e, correlation_id, processing_time)
            raise
    
    async def _log_request(self, request: Request, correlation_id: str) -> None:
        """
        Log incoming request details.
        
        Args:
            request: HTTP request
            correlation_id: Request correlation ID
        """
        # Get client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Get authentication info if available
        auth_info = {}
        if hasattr(request.state, 'auth_user') and request.state.auth_user:
            auth_info = {
                "user_id": request.state.auth_user.get('id'),
                "auth_method": getattr(request.state, 'auth_method', 'unknown')
            }
        
        # Prepare request body (if applicable and not too large)
        request_body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Only log body for JSON content and if not too large
                content_type = request.headers.get("content-type", "")
                if "application/json" in content_type:
                    # Read body (this consumes the stream, but FastAPI handles this)
                    body_bytes = await request.body()
                    if len(body_bytes) < 10000:  # Only log if less than 10KB
                        body_str = body_bytes.decode('utf-8')
                        request_body = self._filter_sensitive_data(json.loads(body_str))
            except Exception:
                # If we can't parse the body, don't log it
                pass
        
        # Create log entry
        log_data = {
            "event": "request_started",
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params) if request.query_params else None,
            "headers": self._filter_headers(dict(request.headers)),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "body": request_body,
            "auth": auth_info if auth_info else None,
            "timestamp": time.time(),
        }
        
        # Remove None values
        log_data = {k: v for k, v in log_data.items() if v is not None}
        
        if self.settings.enable_access_logs:
            logger.info("Request started", extra={"structured_data": log_data})
    
    async def _log_response(
        self,
        request: Request,
        response: Response,
        correlation_id: str,
        processing_time: float
    ) -> None:
        """
        Log response details.
        
        Args:
            request: HTTP request
            response: HTTP response
            correlation_id: Request correlation ID
            processing_time: Request processing time in seconds
        """
        # Create log entry
        log_data = {
            "event": "request_completed",
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "processing_time_ms": round(processing_time * 1000, 2),
            "response_headers": self._filter_headers(dict(response.headers)),
            "timestamp": time.time(),
        }
        
        # Add response size if available
        content_length = response.headers.get("content-length")
        if content_length:
            log_data["response_size_bytes"] = int(content_length)
        
        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = logging.ERROR
        elif response.status_code >= 400:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO
        
        if self.settings.enable_access_logs:
            logger.log(log_level, "Request completed", extra={"structured_data": log_data})
    
    async def _log_error(
        self,
        request: Request,
        error: Exception,
        correlation_id: str,
        processing_time: float
    ) -> None:
        """
        Log request error details.
        
        Args:
            request: HTTP request
            error: Exception that occurred
            correlation_id: Request correlation ID
            processing_time: Request processing time in seconds
        """
        # Create log entry
        log_data = {
            "event": "request_error",
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "processing_time_ms": round(processing_time * 1000, 2),
            "timestamp": time.time(),
        }
        
        logger.error("Request failed", extra={"structured_data": log_data}, exc_info=True)
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address from request.
        
        Args:
            request: HTTP request
            
        Returns:
            str: Client IP address
        """
        # Check for forwarded headers (from load balancers/proxies)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _filter_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Filter sensitive headers from logging.
        
        Args:
            headers: Request/response headers
            
        Returns:
            Dict: Filtered headers
        """
        filtered = {}
        
        for key, value in headers.items():
            key_lower = key.lower()
            
            if key_lower in self.sensitive_headers:
                # Mask sensitive headers
                if key_lower == "authorization" and value.startswith("Bearer "):
                    filtered[key] = "Bearer ***"
                else:
                    filtered[key] = "***"
            else:
                filtered[key] = value
        
        return filtered
    
    def _filter_sensitive_data(self, data: Any) -> Any:
        """
        Recursively filter sensitive data from request/response bodies.
        
        Args:
            data: Data to filter (dict, list, or primitive)
            
        Returns:
            Any: Filtered data
        """
        if isinstance(data, dict):
            filtered = {}
            for key, value in data.items():
                key_lower = key.lower()
                
                if any(sensitive in key_lower for sensitive in self.sensitive_body_fields):
                    filtered[key] = "***"
                else:
                    filtered[key] = self._filter_sensitive_data(value)
            
            return filtered
        
        elif isinstance(data, list):
            return [self._filter_sensitive_data(item) for item in data]
        
        else:
            # Primitive value, return as-is
            return data


class StructuredLogger:
    """
    Utility class for structured logging throughout the application.
    
    Provides consistent structured logging with correlation ID support.
    """
    
    def __init__(self, name: str):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically __name__)
        """
        self.logger = logging.getLogger(name)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with structured data."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with structured data."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with structured data."""
        self._log(logging.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with structured data."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs) -> None:
        """
        Internal logging method.
        
        Args:
            level: Log level
            message: Log message
            **kwargs: Additional structured data
        """
        # Add timestamp if not present
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = time.time()
        
        # Create extra data for structured logging
        extra = {"structured_data": kwargs} if kwargs else {}
        
        self.logger.log(level, message, extra=extra)