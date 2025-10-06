"""
Timing utilities for performance measurement and monitoring.

This module provides decorators and context managers for measuring
execution time of functions and code blocks.
"""

import asyncio
import time
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Generator, TypeVar

from .logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def measure_time(func_name: str = "") -> Callable[[F], F]:
    """
    Decorator to measure execution time of synchronous functions.
    
    Args:
        func_name: Optional custom name for the function in logs
        
    Returns:
        Decorated function that logs execution time
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = func_name or f"{func.__module__}.{func.__name__}"
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                duration = end_time - start_time
                logger.info(
                    "Function execution completed",
                    function=name,
                    duration_seconds=duration,
                    duration_ms=duration * 1000
                )
        
        return wrapper  # type: ignore
    return decorator


def async_timer(func_name: str = "") -> Callable[[F], F]:
    """
    Decorator to measure execution time of asynchronous functions.
    
    Args:
        func_name: Optional custom name for the function in logs
        
    Returns:
        Decorated async function that logs execution time
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = func_name or f"{func.__module__}.{func.__name__}"
            start_time = time.perf_counter()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                duration = end_time - start_time
                logger.info(
                    "Async function execution completed",
                    function=name,
                    duration_seconds=duration,
                    duration_ms=duration * 1000
                )
        
        return wrapper  # type: ignore
    return decorator


@contextmanager
def timer(operation_name: str) -> Generator[None, None, None]:
    """
    Context manager to measure execution time of code blocks.
    
    Args:
        operation_name: Name of the operation being timed
        
    Example:
        with timer("database_query"):
            result = db.query("SELECT * FROM users")
    """
    start_time = time.perf_counter()
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(
            "Operation completed",
            operation=operation_name,
            duration_seconds=duration,
            duration_ms=duration * 1000
        )


@asynccontextmanager
async def async_timer_context(operation_name: str) -> AsyncGenerator[None, None]:
    """
    Async context manager to measure execution time of async code blocks.
    
    Args:
        operation_name: Name of the operation being timed
        
    Example:
        async with async_timer_context("api_request"):
            response = await client.get("/api/data")
    """
    start_time = time.perf_counter()
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(
            "Async operation completed",
            operation=operation_name,
            duration_seconds=duration,
            duration_ms=duration * 1000
        )


class PerformanceMonitor:
    """
    Performance monitoring utility for tracking operation metrics.
    """
    
    def __init__(self) -> None:
        self._metrics: dict[str, list[float]] = {}
    
    def record(self, operation: str, duration: float) -> None:
        """Record a timing measurement for an operation."""
        if operation not in self._metrics:
            self._metrics[operation] = []
        self._metrics[operation].append(duration)
    
    def get_stats(self, operation: str) -> dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self._metrics:
            return {}
        
        durations = self._metrics[operation]
        return {
            "count": len(durations),
            "total": sum(durations),
            "average": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
        }
    
    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_stats(op) for op in self._metrics.keys()}
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()