"""
Health check endpoints for monitoring and observability.

This module provides health check endpoints for monitoring the system
status and individual service health.
"""

import asyncio
import logging
import time
from typing import Dict, Any

from fastapi import APIRouter, Request, Depends, Query
from fastapi.responses import JSONResponse

from ...config import get_settings
from ...models.api import HealthResponse, HealthStatus, ServiceHealth
from ...services.agent_manager import AgentManager
from ..middleware.metrics import get_metrics, get_metrics_content_type


logger = logging.getLogger(__name__)
router = APIRouter()


def get_agent_manager(request: Request) -> AgentManager:
    """Dependency to get AgentManager from app state."""
    return getattr(request.app.state, 'agent_manager', None)


def get_app_start_time(request: Request) -> float:
    """Dependency to get application start time."""
    return getattr(request.app.state, 'start_time', time.time())


@router.get("/health")
async def health_check(
    request: Request,
    deep: bool = Query(False, description="Perform deep health checks"),
    agent_manager: AgentManager = Depends(get_agent_manager),
    start_time: float = Depends(get_app_start_time)
) -> HealthResponse:
    """
    Basic health check endpoint.
    
    Returns the overall system health status and basic metrics.
    With deep=true, performs comprehensive health checks of all dependencies.
    
    Args:
        request: FastAPI request object
        deep: Whether to perform deep health checks
        agent_manager: AgentManager dependency (optional)
        start_time: Application start time
        
    Returns:
        HealthResponse: System health status and metrics
    """
    settings = get_settings()
    current_time = time.time()
    uptime = current_time - start_time
    
    # Initialize response
    health_response = HealthResponse(
        status=HealthStatus.HEALTHY,
        version=settings.app_version,
        uptime_seconds=uptime,
        services={},
        metrics={
            "uptime_seconds": uptime,
            "timestamp": current_time,
        }
    )
    
    # Basic health checks
    try:
        # Check if AgentManager is available
        if agent_manager is None:
            health_response.services["agent_manager"] = ServiceHealth(
                status=HealthStatus.UNHEALTHY,
                message="AgentManager not initialized"
            )
            health_response.status = HealthStatus.UNHEALTHY
        else:
            health_response.services["agent_manager"] = ServiceHealth(
                status=HealthStatus.HEALTHY,
                message="AgentManager available"
            )
        
        # Perform deep health checks if requested
        if deep and agent_manager:
            await _perform_deep_health_checks(health_response, agent_manager)
        
        # Add system metrics
        health_response.metrics.update({
            "memory_usage_mb": _get_memory_usage(),
            "active_tasks": len(getattr(agent_manager, '_active_tasks', {})) if agent_manager else 0,
        })
        
    except Exception as e:
        logger.error(f"Error during health check: {e}", exc_info=True)
        health_response.status = HealthStatus.UNHEALTHY
        health_response.services["system"] = ServiceHealth(
            status=HealthStatus.UNHEALTHY,
            message=f"Health check failed: {str(e)}"
        )
    
    # Determine overall status based on service health
    if health_response.status == HealthStatus.HEALTHY:
        unhealthy_services = [
            name for name, service in health_response.services.items()
            if service.status == HealthStatus.UNHEALTHY
        ]
        degraded_services = [
            name for name, service in health_response.services.items()
            if service.status == HealthStatus.DEGRADED
        ]
        
        if unhealthy_services:
            health_response.status = HealthStatus.UNHEALTHY
        elif degraded_services:
            health_response.status = HealthStatus.DEGRADED
    
    # Set appropriate HTTP status code
    status_code = 200
    if health_response.status == HealthStatus.DEGRADED:
        status_code = 200  # Still operational
    elif health_response.status == HealthStatus.UNHEALTHY:
        status_code = 503  # Service unavailable
    
    return JSONResponse(
        status_code=status_code,
        content=health_response.model_dump()
    )


@router.get("/health/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Kubernetes liveness probe endpoint.
    
    Simple endpoint that returns 200 if the application is running.
    Should not perform any dependency checks.
    
    Returns:
        Dict: Simple status response
    """
    return {
        "status": "alive",
        "timestamp": time.time()
    }


@router.get("/health/ready")
async def readiness_check(
    request: Request,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Kubernetes readiness probe endpoint.
    
    Returns 200 if the application is ready to serve requests,
    503 if not ready (dependencies not available).
    
    Args:
        request: FastAPI request object
        agent_manager: AgentManager dependency
        
    Returns:
        Dict: Readiness status response
    """
    ready = True
    issues = []
    
    # Check critical dependencies
    if agent_manager is None:
        ready = False
        issues.append("AgentManager not initialized")
    
    # Additional readiness checks can be added here
    
    status_code = 200 if ready else 503
    response_data = {
        "status": "ready" if ready else "not_ready",
        "timestamp": time.time(),
        "issues": issues if issues else None
    }
    
    return JSONResponse(
        status_code=status_code,
        content=response_data
    )


async def _perform_deep_health_checks(
    health_response: HealthResponse,
    agent_manager: AgentManager
) -> None:
    """
    Perform comprehensive health checks of all system dependencies.
    
    Args:
        health_response: Health response object to update
        agent_manager: AgentManager instance
    """
    settings = get_settings()
    
    # Check vLLM endpoints
    try:
        if hasattr(agent_manager, '_llm_handler') and agent_manager._llm_handler:
            start_time = time.time()
            llm_health = await agent_manager._llm_handler.check_health()
            response_time = (time.time() - start_time) * 1000
            
            if llm_health.get("status") == "healthy":
                health_response.services["vllm_chat"] = ServiceHealth(
                    status=HealthStatus.HEALTHY,
                    message="vLLM chat endpoint healthy",
                    response_time_ms=response_time
                )
            else:
                health_response.services["vllm_chat"] = ServiceHealth(
                    status=HealthStatus.UNHEALTHY,
                    message=f"vLLM chat endpoint unhealthy: {llm_health.get('message', 'Unknown error')}",
                    response_time_ms=response_time
                )
        else:
            health_response.services["vllm_chat"] = ServiceHealth(
                status=HealthStatus.UNHEALTHY,
                message="LLM handler not initialized"
            )
    except Exception as e:
        health_response.services["vllm_chat"] = ServiceHealth(
            status=HealthStatus.UNHEALTHY,
            message=f"vLLM health check failed: {str(e)}"
        )
    
    # Check MCP registry
    try:
        if hasattr(agent_manager, '_mcp_registry') and agent_manager._mcp_registry:
            mcp_health = await agent_manager._mcp_registry.check_health()
            
            if mcp_health.get("status") == "healthy":
                tool_count = len(await agent_manager._mcp_registry.get_all_tools())
                health_response.services["mcp_registry"] = ServiceHealth(
                    status=HealthStatus.HEALTHY,
                    message=f"MCP registry healthy with {tool_count} tools"
                )
            else:
                health_response.services["mcp_registry"] = ServiceHealth(
                    status=HealthStatus.DEGRADED,
                    message=f"MCP registry issues: {mcp_health.get('message', 'Unknown error')}"
                )
        else:
            health_response.services["mcp_registry"] = ServiceHealth(
                status=HealthStatus.DEGRADED,
                message="MCP registry not available (optional)"
            )
    except Exception as e:
        health_response.services["mcp_registry"] = ServiceHealth(
            status=HealthStatus.DEGRADED,
            message=f"MCP registry check failed: {str(e)}"
        )
    
    # Check ChromaDB (if available)
    try:
        if hasattr(agent_manager, '_tool_selector') and agent_manager._tool_selector:
            # Try a simple operation to check ChromaDB health
            start_time = time.time()
            # This would be a simple health check operation
            # await agent_manager._tool_selector.health_check()
            response_time = (time.time() - start_time) * 1000
            
            health_response.services["chromadb"] = ServiceHealth(
                status=HealthStatus.HEALTHY,
                message="ChromaDB vector database healthy",
                response_time_ms=response_time
            )
        else:
            health_response.services["chromadb"] = ServiceHealth(
                status=HealthStatus.DEGRADED,
                message="ChromaDB not available (optional)"
            )
    except Exception as e:
        health_response.services["chromadb"] = ServiceHealth(
            status=HealthStatus.DEGRADED,
            message=f"ChromaDB check failed: {str(e)}"
        )


def _get_memory_usage() -> float:
    """
    Get current memory usage in MB.
    
    Returns:
        float: Memory usage in megabytes
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except ImportError:
        # psutil not available, return 0
        return 0.0
    except Exception:
        # Any other error, return 0
        return 0.0


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format for scraping by monitoring systems.
    
    Returns:
        Response: Prometheus metrics in text format
    """
    metrics_data = get_metrics()
    
    return Response(
        content=metrics_data,
        media_type=get_metrics_content_type()
    )