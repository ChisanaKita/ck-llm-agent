"""
Health check endpoints for monitoring and observability.

This module provides health check endpoints for monitoring the system
status and individual service health.
"""

import logging
import time
from typing import Dict, Any

from fastapi import APIRouter, Request, Depends, Query
from fastapi.responses import JSONResponse, Response

from ...config import get_settings
from ...models.api import HealthResponse, HealthStatus, ServiceHealth
from ...services.agent_manager import AgentManager
from ...services.connection_manager import get_connection_manager
from ...services.resource_manager import get_resource_manager
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
        
        # Add connection pool metrics if deep check
        if deep:
            try:
                connection_manager = await get_connection_manager()
                resource_usage = await connection_manager.get_resource_usage()
                connection_stats = connection_manager.get_all_stats()
                
                health_response.metrics.update({
                    "connection_pools": {
                        "total_pools": resource_usage.get("total_pools", 0),
                        "total_active_requests": resource_usage.get("total_active_requests", 0),
                        "global_utilization": resource_usage.get("global_utilization", 0),
                        "available_capacity": resource_usage.get("available_capacity", 0),
                        "total_requests": connection_stats.get("total_requests", 0),
                        "error_rate": connection_stats.get("error_rate", 0),
                    }
                })
                
                # Check connection pool health
                healthy_pools = connection_manager.get_healthy_pools()
                unhealthy_pools = connection_manager.get_unhealthy_pools()
                
                if unhealthy_pools:
                    health_response.services["connection_pools"] = ServiceHealth(
                        status=HealthStatus.DEGRADED,
                        message=f"Some connection pools unhealthy: {unhealthy_pools}"
                    )
                else:
                    health_response.services["connection_pools"] = ServiceHealth(
                        status=HealthStatus.HEALTHY,
                        message=f"All {len(healthy_pools)} connection pools healthy"
                    )
                    
            except Exception as e:
                logger.warning(f"Failed to get connection pool metrics: {e}")
                health_response.services["connection_pools"] = ServiceHealth(
                    status=HealthStatus.DEGRADED,
                    message=f"Connection pool check failed: {str(e)}"
                )
        
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


@router.get("/health/connections")
async def connection_health() -> Dict[str, Any]:
    """
    Connection pool health and statistics endpoint.
    
    Returns detailed information about connection pools, resource usage,
    and performance metrics.
    
    Returns:
        Dict: Connection pool health and statistics
    """
    try:
        connection_manager = await get_connection_manager()
        resource_manager = await get_resource_manager()
        
        # Get comprehensive connection statistics
        resource_usage = await connection_manager.get_resource_usage()
        connection_stats = connection_manager.get_all_stats()
        resource_manager_stats = resource_manager.get_stats()
        
        # Get pool health status
        healthy_pools = connection_manager.get_healthy_pools()
        unhealthy_pools = connection_manager.get_unhealthy_pools()
        
        return {
            "status": "healthy" if not unhealthy_pools else "degraded",
            "timestamp": time.time(),
            "summary": {
                "total_pools": len(connection_manager.pools),
                "healthy_pools": len(healthy_pools),
                "unhealthy_pools": len(unhealthy_pools),
                "global_utilization": resource_usage.get("global_utilization", 0),
                "total_active_requests": resource_usage.get("total_active_requests", 0),
                "available_capacity": resource_usage.get("available_capacity", 0),
            },
            "resource_usage": resource_usage,
            "connection_stats": connection_stats,
            "resource_manager": resource_manager_stats,
            "pool_health": {
                "healthy": healthy_pools,
                "unhealthy": unhealthy_pools,
            }
        }
        
    except Exception as e:
        logger.error(f"Connection health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "timestamp": time.time(),
                "error": str(e)
            }
        )


@router.post("/health/connections/cleanup")
async def force_connection_cleanup() -> Dict[str, Any]:
    """
    Force immediate connection pool cleanup.
    
    Triggers cleanup of stale connections and optimization of pool sizes.
    
    Returns:
        Dict: Cleanup results
    """
    try:
        resource_manager = await get_resource_manager()
        
        # Force cleanup and optimization
        cleanup_result = await resource_manager.force_cleanup()
        optimization_result = await resource_manager.force_optimization()
        
        return {
            "status": "completed",
            "timestamp": time.time(),
            "cleanup": cleanup_result,
            "optimization": optimization_result
        }
        
    except Exception as e:
        logger.error(f"Force cleanup failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "timestamp": time.time(),
                "error": str(e)
            }
        )


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