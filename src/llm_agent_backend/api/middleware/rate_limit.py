"""
Rate limiting middleware using token bucket algorithm.

This module provides rate limiting functionality to prevent abuse
and ensure fair usage of the API endpoints.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Tuple
from collections import defaultdict

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ...config import get_settings
from ...models.api import ErrorResponse, ErrorDetail, ErrorType


logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Token bucket implementation for rate limiting.
    
    The token bucket algorithm allows for burst traffic while maintaining
    an average rate limit over time.
    """
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens in bucket
            refill_rate: Rate at which tokens are added (tokens per second)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            bool: True if tokens were consumed, False if not enough tokens
        """
        async with self._lock:
            now = time.time()
            
            # Add tokens based on time elapsed
            time_elapsed = now - self.last_refill
            tokens_to_add = time_elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_status(self) -> Dict[str, float]:
        """
        Get current bucket status.
        
        Returns:
            Dict: Bucket status information
        """
        return {
            "tokens": self.tokens,
            "capacity": self.capacity,
            "refill_rate": self.refill_rate,
            "last_refill": self.last_refill
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm.
    
    Implements per-client rate limiting with configurable limits
    and proper error responses.
    """
    
    def __init__(self, app, **kwargs):
        """Initialize rate limiting middleware."""
        super().__init__(app, **kwargs)
        self.settings = get_settings()
        
        # Rate limit configuration
        self.requests_per_minute = self.settings.rate_limit.requests_per_minute
        self.burst_size = self.settings.rate_limit.burst_size
        self.enabled = self.settings.rate_limit.enable_rate_limiting
        
        # Convert to tokens per second
        self.refill_rate = self.requests_per_minute / 60.0
        
        # Client buckets (in production, use Redis or similar)
        self.client_buckets: Dict[str, TokenBucket] = {}
        self.bucket_cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        
        # Paths that are exempt from rate limiting
        self.exempt_paths = {
            "/health",
            "/health/live",
            "/health/ready",
            "/",
        }
        
        logger.info(
            f"Rate limiting initialized: {self.requests_per_minute} req/min, "
            f"burst: {self.burst_size}, enabled: {self.enabled}"
        )
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with rate limiting.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
            
        Returns:
            Response: HTTP response
        """
        # Skip rate limiting if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip rate limiting for exempt paths
        if self._is_exempt_path(request.url.path):
            return await call_next(request)
        
        # Skip rate limiting for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        allowed, retry_after = await self._check_rate_limit(client_id)
        
        if not allowed:
            return self._create_rate_limit_response(retry_after, client_id)
        
        # Add rate limit headers to response
        response = await call_next(request)
        self._add_rate_limit_headers(response, client_id)
        
        return response
    
    def _is_exempt_path(self, path: str) -> bool:
        """
        Check if a path is exempt from rate limiting.
        
        Args:
            path: Request path
            
        Returns:
            bool: True if path is exempt
        """
        return path in self.exempt_paths
    
    def _get_client_id(self, request: Request) -> str:
        """
        Get client identifier for rate limiting.
        
        Uses multiple methods to identify clients:
        1. Authenticated user ID (if available)
        2. API key (if available)
        3. IP address (fallback)
        
        Args:
            request: HTTP request
            
        Returns:
            str: Client identifier
        """
        # Try to get authenticated user ID
        if hasattr(request.state, 'auth_user') and request.state.auth_user:
            user_id = request.state.auth_user.get('id')
            if user_id:
                return f"user:{user_id}"
        
        # Try to get API key
        api_key_header = getattr(self.settings.auth, 'api_key_header', 'X-API-Key')
        api_key = request.headers.get(api_key_header)
        if api_key:
            # Use first 8 characters of API key for identification
            return f"api_key:{api_key[:8]}"
        
        # Fallback to IP address
        client_ip = self._get_client_ip(request)
        return f"ip:{client_ip}"
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address from request.
        
        Handles various proxy headers to get the real client IP.
        
        Args:
            request: HTTP request
            
        Returns:
            str: Client IP address
        """
        # Check for forwarded headers (from load balancers/proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    async def _check_rate_limit(self, client_id: str) -> Tuple[bool, Optional[float]]:
        """
        Check if client is within rate limits.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Tuple: (allowed, retry_after_seconds)
        """
        # Clean up old buckets periodically
        await self._cleanup_buckets()
        
        # Get or create bucket for client
        if client_id not in self.client_buckets:
            self.client_buckets[client_id] = TokenBucket(
                capacity=self.burst_size,
                refill_rate=self.refill_rate
            )
        
        bucket = self.client_buckets[client_id]
        
        # Try to consume a token
        allowed = await bucket.consume(1)
        
        if not allowed:
            # Calculate retry after time
            # Time needed to get at least 1 token
            retry_after = 1.0 / self.refill_rate
            return False, retry_after
        
        return True, None
    
    async def _cleanup_buckets(self) -> None:
        """
        Clean up old, unused token buckets to prevent memory leaks.
        """
        now = time.time()
        
        # Only cleanup every few minutes
        if now - self.last_cleanup < self.bucket_cleanup_interval:
            return
        
        # Remove buckets that haven't been used recently
        cutoff_time = now - self.bucket_cleanup_interval
        buckets_to_remove = []
        
        for client_id, bucket in self.client_buckets.items():
            if bucket.last_refill < cutoff_time:
                buckets_to_remove.append(client_id)
        
        for client_id in buckets_to_remove:
            del self.client_buckets[client_id]
        
        self.last_cleanup = now
        
        if buckets_to_remove:
            logger.debug(f"Cleaned up {len(buckets_to_remove)} old rate limit buckets")
    
    def _add_rate_limit_headers(self, response: Response, client_id: str) -> None:
        """
        Add rate limit headers to response.
        
        Args:
            response: HTTP response
            client_id: Client identifier
        """
        if client_id in self.client_buckets:
            bucket = self.client_buckets[client_id]
            status = bucket.get_status()
            
            # Add standard rate limit headers
            response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(int(status["tokens"]))
            response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
            
            # Add custom headers for debugging
            if self.settings.is_development():
                response.headers["X-RateLimit-Burst"] = str(self.burst_size)
                response.headers["X-RateLimit-RefillRate"] = str(self.refill_rate)
    
    def _create_rate_limit_response(
        self,
        retry_after: Optional[float],
        client_id: str
    ) -> JSONResponse:
        """
        Create rate limit exceeded error response.
        
        Args:
            retry_after: Seconds to wait before retrying
            client_id: Client identifier
            
        Returns:
            JSONResponse: Rate limit error response
        """
        error_response = ErrorResponse(
            error=ErrorDetail(
                type=ErrorType.RATE_LIMIT,
                code="rate_limit_exceeded",
                message=f"Rate limit exceeded. Try again in {retry_after:.1f} seconds.",
                details={
                    "limit": self.requests_per_minute,
                    "window": "1 minute",
                    "retry_after": retry_after
                }
            )
        )
        
        headers = {
            "X-RateLimit-Limit": str(self.requests_per_minute),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(time.time() + (retry_after or 60))),
        }
        
        if retry_after:
            headers["Retry-After"] = str(int(retry_after))
        
        logger.warning(f"Rate limit exceeded for client: {client_id}")
        
        return JSONResponse(
            status_code=429,
            content=error_response.model_dump(),
            headers=headers
        )