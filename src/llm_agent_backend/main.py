"""
Main entry point for the LLM Agent Backend application.

This module provides the main entry point for running the FastAPI application
with uvicorn server.
"""

import logging
import sys
from typing import Optional

import uvicorn

from .config import get_settings
from .api.app import app


def setup_logging() -> None:
    """Configure application logging."""
    settings = get_settings()
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    if settings.is_development():
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    else:
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def main(
    host: Optional[str] = None,
    port: Optional[int] = None,
    reload: Optional[bool] = None,
    workers: Optional[int] = None
) -> None:
    """
    Run the FastAPI application with uvicorn.
    
    Args:
        host: Host to bind to (overrides config)
        port: Port to bind to (overrides config)
        reload: Enable auto-reload (overrides config)
        workers: Number of worker processes (overrides config)
    """
    settings = get_settings()
    
    # Setup logging
    setup_logging()
    
    # Determine configuration
    run_host = host or settings.host
    run_port = port or settings.port
    run_reload = reload if reload is not None else settings.is_development()
    run_workers = workers or (1 if run_reload else settings.workers)
    
    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Host: {run_host}:{run_port}")
    logger.info(f"Workers: {run_workers}")
    logger.info(f"Reload: {run_reload}")
    
    # Run the application
    uvicorn.run(
        "llm_agent_backend.api.app:app",
        host=run_host,
        port=run_port,
        reload=run_reload,
        workers=run_workers if not run_reload else 1,  # Reload doesn't work with multiple workers
        log_level=settings.log_level.lower(),
        access_log=settings.enable_access_logs,
    )


if __name__ == "__main__":
    main()