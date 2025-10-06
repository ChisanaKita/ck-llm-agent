"""
FastAPI application setup for the LLM Agent Backend.

This module creates and configures the main FastAPI application with
OpenAI-compatible endpoints, middleware, and error handling.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import get_settings
from ..models.api import ErrorResponse, ErrorDetail, ErrorType
from ..services.agent_manager import AgentManager
from ..services.connection_manager import get_connection_manager, shutdown_connection_manager
from ..services.resource_manager import get_resource_manager, shutdown_resource_manager
from .routes import chat, health, models
from .middleware import (
    AuthenticationMiddleware,
    RateLimitMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown tasks.
    
    Handles initialization of the AgentManager and other services during
    startup, and cleanup during shutdown.
    """
    settings = get_settings()
    
    try:
        # Startup
        logger.info(f"Starting {settings.app_name} v{settings.app_version}")
        
        # Initialize connection manager first
        connection_manager = await get_connection_manager()
        app.state.connection_manager = connection_manager
        
        # Initialize resource manager for background tasks
        resource_manager = await get_resource_manager()
        app.state.resource_manager = resource_manager
        
        # Initialize AgentManager
        agent_manager = AgentManager()
        await agent_manager.initialize()
        
        # Store in app state for access in routes
        app.state.agent_manager = agent_manager
        app.state.start_time = time.time()
        
        logger.info("Application startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}", exc_info=True)
        raise
    
    finally:
        # Shutdown
        logger.info("Starting application shutdown...")
        
        try:
            # Cleanup AgentManager first
            if hasattr(app.state, 'agent_manager'):
                await app.state.agent_manager.shutdown()
            
            # Shutdown resource manager
            await shutdown_resource_manager()
            
            # Shutdown connection manager last
            await shutdown_connection_manager()
            
            logger.info("Application shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    settings = get_settings()
    
    # Create FastAPI app with OpenAPI configuration
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-grade LLM agent backend with CrewAI, vLLM, and MCP integration",
        docs_url="/docs" if settings.is_development() else None,
        redoc_url="/redoc" if settings.is_development() else None,
        openapi_url="/openapi.json" if settings.is_development() else None,
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development() else [],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Add custom middleware (order matters - last added is executed first)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    # Add authentication and rate limiting if configured
    if settings.auth:
        app.add_middleware(AuthenticationMiddleware)
    
    if settings.rate_limit.enable_rate_limiting:
        app.add_middleware(RateLimitMiddleware)
    
    # Include routers
    app.include_router(chat.router, prefix="/v1", tags=["Chat Completions"])
    app.include_router(health.router, tags=["Health"])
    app.include_router(models.router, prefix="/v1", tags=["Models"])
    
    # Global exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions with proper error responses."""
        logger.error(f"Unhandled exception in {request.method} {request.url}: {exc}", exc_info=True)
        
        error_response = ErrorResponse(
            error=ErrorDetail(
                type=ErrorType.SERVER_ERROR,
                code="internal_server_error",
                message="An internal server error occurred",
                details={"path": str(request.url.path)} if settings.is_development() else None
            ),
            request_id=getattr(request.state, 'request_id', None)
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump()
        )
    
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle 404 errors with OpenAI-compatible format."""
        error_response = ErrorResponse(
            error=ErrorDetail(
                type=ErrorType.NOT_FOUND,
                code="not_found",
                message=f"The requested endpoint {request.url.path} was not found",
                param="path"
            ),
            request_id=getattr(request.state, 'request_id', None)
        )
        
        return JSONResponse(
            status_code=404,
            content=error_response.model_dump()
        )
    
    @app.exception_handler(405)
    async def method_not_allowed_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle 405 errors with OpenAI-compatible format."""
        error_response = ErrorResponse(
            error=ErrorDetail(
                type=ErrorType.INVALID_REQUEST,
                code="method_not_allowed",
                message=f"Method {request.method} not allowed for {request.url.path}",
                param="method"
            ),
            request_id=getattr(request.state, 'request_id', None)
        )
        
        return JSONResponse(
            status_code=405,
            content=error_response.model_dump()
        )
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root() -> Dict[str, Any]:
        """Root endpoint with basic service information."""
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "status": "running",
            "docs_url": "/docs" if settings.is_development() else None
        }
    
    return app


# Create the application instance
app = create_app()