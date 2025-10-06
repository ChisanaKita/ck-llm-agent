"""
Logging configuration and utilities for the LLM Agent Backend.

This module provides structured logging setup with correlation IDs,
JSON formatting, and different log levels for different environments.
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, Processor


# Context variable for correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def add_correlation_id(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add correlation ID to log events."""
    cid = correlation_id.get()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def add_timestamp(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add ISO timestamp to log events."""
    import datetime
    event_dict["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    enable_access_logs: bool = True
) -> None:
    """
    Set up structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ("json" or "console")
        enable_access_logs: Whether to enable access logs
    """
    # Configure structlog
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        add_correlation_id,
        add_timestamp,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]
    
    if log_format == "json":
        processors.extend([
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer()
        ])
    else:
        processors.extend([
            structlog.processors.ExceptionPrettyPrinter(),
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Configure uvicorn access logs
    if not enable_access_logs:
        logging.getLogger("uvicorn.access").disabled = True
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def set_correlation_id(cid: Optional[str] = None) -> str:
    """
    Set correlation ID for the current context.
    
    Args:
        cid: Correlation ID to set. If None, generates a new UUID.
        
    Returns:
        The correlation ID that was set
    """
    if cid is None:
        cid = str(uuid.uuid4())
    correlation_id.set(cid)
    return cid


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return correlation_id.get()


def clear_correlation_id() -> None:
    """Clear the current correlation ID."""
    correlation_id.set(None)