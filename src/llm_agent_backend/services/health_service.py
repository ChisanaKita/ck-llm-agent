"""
Health monitoring service for vLLM endpoints and system components.

This service provides centralized health monitoring for all vLLM endpoints,
circuit breakers, and system dependencies with comprehensive status reporting.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Avoid circular import - import endpoint_manager when needed
from ..models.api import HealthStatus, ServiceHealth, HealthResponse
from ..config import get_settings


logger = logging.getLogger(__name__)


class ComponentType(str, Enum):
    """Types of system components that can be monitored."""
    VLLM_CHAT = "vllm_chat"
    VLLM_EMBEDDING = "vllm_embedding"
    CHROMADB = "chromadb"
    MCP_SERVER = "mcp_server"
    CIRCUIT_BREAKER = "circuit_breaker"
    CONNECTION_POOL = "connection_pool"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component_id: str
    component_type: ComponentType
    status: HealthStatus
    message: Optional[str] = None
    response_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class SystemHealthService:
    """
    Centralized health monitoring service for all system components.
    
    Provides comprehensive health checks for vLLM endpoints, circuit breakers,
    connection pools, and other system dependencies.
    """
    
    def __init__(self):
        """Initialize the health service."""
        self.settings = get_settings()
        self.start_time = time.time()
        self._health_cache: Dict[str, HealthCheckResult] = {}
        self._cache_ttl = 30  # Cache health results for 30 seconds
        
        logger.info("System health service initialized")
    
    async def check_system_health(self, include_details: bool = True) -> HealthResponse:
        """
        Perform comprehensive system health check.
        
        Args:
            include_details: Whether to include detailed service health info
            
        Returns:
            Complete system health response
        """
        start_time = time.time()
        
        # Check all components
        health_results = await self._check_all_components()
        
        # Determine overall system status
        overall_status = self._determine_overall_status(health_results)
        
        # Build service health details
        services = {}
        if include_details:
            services = self._build_service_health_map(health_results)
        
        # Calculate uptime
        uptime_seconds = time.time() - self.start_time
        
        response = HealthResponse(
            status=overall_status,
            timestamp=int(time.time()),
            version=self.settings.app_version,
            uptime_seconds=uptime_seconds,
            services=services
        )
        
        check_duration = (time.time() - start_time) * 1000
        logger.debug(f"System health check completed in {check_duration:.2f}ms")
        
        return response
    
    async def check_vllm_endpoints(self) -> Dict[str, HealthCheckResult]:
        """
        Check health of all vLLM endpoints.
        
        Returns:
            Health check results for all endpoints
        """
        endpoint_manager = get_endpoint_manager()
        endpoint_statuses = await endpoint_manager.check_all_health()
        
        results = {}
        for endpoint_id, status in endpoint_statuses.items():
            # Determine component type based on endpoint ID
            if "chat" in endpoint_id.lower():
                component_type = ComponentType.VLLM_CHAT
            elif "embedding" in endpoint_id.lower():
                component_type = ComponentType.VLLM_EMBEDDING
            else:
                component_type = ComponentType.VLLM_CHAT  # Default
            
            # Convert to health check result
            health_status = HealthStatus.HEALTHY if status.is_healthy else HealthStatus.UNHEALTHY
            
            results[endpoint_id] = HealthCheckResult(
                component_id=endpoint_id,
                component_type=component_type,
                status=health_status,
                message=status.error_message or "Endpoint operational",
                response_time_ms=status.response_time_ms,
                metadata={
                    "endpoint_type": status.endpoint_type.value,
                    "base_url": status.base_url,
                    "model": status.model,
                    "error_count": status.error_count,
                    "success_count": status.success_count,
                    "last_check": status.last_check
                }
            )
        
        return results
    
    async def check_circuit_breakers(self) -> Dict[str, HealthCheckResult]:
        """
        Check status of all circuit breakers.
        
        Returns:
            Circuit breaker health status
        """
        endpoint_manager = get_endpoint_manager()
        results = {}
        
        for endpoint_id in endpoint_manager.list_endpoints():
            handler = endpoint_manager.get_endpoint(endpoint_id)
            if handler and handler.circuit_breaker:
                breaker_state = handler.circuit_breaker.get_state()
                
                # Determine health based on circuit breaker state
                if breaker_state["state"] == "closed":
                    status = HealthStatus.HEALTHY
                    message = "Circuit breaker closed - normal operation"
                elif breaker_state["state"] == "half_open":
                    status = HealthStatus.DEGRADED
                    message = "Circuit breaker half-open - testing recovery"
                else:  # open
                    status = HealthStatus.UNHEALTHY
                    message = f"Circuit breaker open - {breaker_state['failure_count']} failures"
                
                results[f"{endpoint_id}_circuit_breaker"] = HealthCheckResult(
                    component_id=f"{endpoint_id}_circuit_breaker",
                    component_type=ComponentType.CIRCUIT_BREAKER,
                    status=status,
                    message=message,
                    metadata=breaker_state
                )
        
        return results
    
    async def check_connection_pools(self) -> Dict[str, HealthCheckResult]:
        """
        Check health of connection pools.
        
        Returns:
            Connection pool health status
        """
        endpoint_manager = get_endpoint_manager()
        results = {}
        
        for endpoint_id in endpoint_manager.list_endpoints():
            handler = endpoint_manager.get_endpoint(endpoint_id)
            if handler and handler.connection_pool:
                pool = handler.connection_pool
                
                # Check if session is healthy
                try:
                    session = await pool.get_session()
                    is_healthy = not session.closed
                    
                    status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
                    message = "Connection pool operational" if is_healthy else "Connection pool closed"
                    
                    results[f"{endpoint_id}_connection_pool"] = HealthCheckResult(
                        component_id=f"{endpoint_id}_connection_pool",
                        component_type=ComponentType.CONNECTION_POOL,
                        status=status,
                        message=message,
                        metadata={
                            "base_url": pool.base_url,
                            "pool_size": pool.pool_size,
                            "timeout": pool.timeout.total,
                            "session_closed": session.closed,
                            "api_key_configured": bool(pool.api_key)
                        }
                    )
                    
                except Exception as e:
                    results[f"{endpoint_id}_connection_pool"] = HealthCheckResult(
                        component_id=f"{endpoint_id}_connection_pool",
                        component_type=ComponentType.CONNECTION_POOL,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Connection pool error: {str(e)}",
                        metadata={"error": str(e)}
                    )
        
        return results
    
    async def _check_all_components(self) -> List[HealthCheckResult]:
        """
        Check health of all system components.
        
        Returns:
            List of all health check results
        """
        all_results = []
        
        # Check vLLM endpoints
        try:
            vllm_results = await self.check_vllm_endpoints()
            all_results.extend(vllm_results.values())
        except Exception as e:
            logger.error(f"Error checking vLLM endpoints: {e}")
            all_results.append(HealthCheckResult(
                component_id="vllm_endpoints",
                component_type=ComponentType.VLLM_CHAT,
                status=HealthStatus.UNHEALTHY,
                message=f"vLLM endpoint check failed: {str(e)}"
            ))
        
        # Check circuit breakers
        try:
            breaker_results = await self.check_circuit_breakers()
            all_results.extend(breaker_results.values())
        except Exception as e:
            logger.error(f"Error checking circuit breakers: {e}")
            all_results.append(HealthCheckResult(
                component_id="circuit_breakers",
                component_type=ComponentType.CIRCUIT_BREAKER,
                status=HealthStatus.UNHEALTHY,
                message=f"Circuit breaker check failed: {str(e)}"
            ))
        
        # Check connection pools
        try:
            pool_results = await self.check_connection_pools()
            all_results.extend(pool_results.values())
        except Exception as e:
            logger.error(f"Error checking connection pools: {e}")
            all_results.append(HealthCheckResult(
                component_id="connection_pools",
                component_type=ComponentType.CONNECTION_POOL,
                status=HealthStatus.UNHEALTHY,
                message=f"Connection pool check failed: {str(e)}"
            ))
        
        return all_results
    
    def _determine_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """
        Determine overall system health status from component results.
        
        Args:
            results: List of component health check results
            
        Returns:
            Overall system health status
        """
        if not results:
            return HealthStatus.UNHEALTHY
        
        # Count status types
        degraded_count = sum(1 for r in results if r.status == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for r in results if r.status == HealthStatus.UNHEALTHY)
        
        total_count = len(results)
        
        # Determine overall status based on component health
        if unhealthy_count == 0:
            if degraded_count == 0:
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.DEGRADED
        else:
            # If more than 50% of components are unhealthy, system is unhealthy
            if unhealthy_count > total_count * 0.5:
                return HealthStatus.UNHEALTHY
            else:
                return HealthStatus.DEGRADED
    
    def _build_service_health_map(self, results: List[HealthCheckResult]) -> Dict[str, ServiceHealth]:
        """
        Build service health map from health check results.
        
        Args:
            results: List of health check results
            
        Returns:
            Map of service names to health status
        """
        services = {}
        
        for result in results:
            service_health = ServiceHealth(
                status=result.status,
                message=result.message,
                last_check=int(result.timestamp),
                response_time_ms=result.response_time_ms
            )
            
            services[result.component_id] = service_health
        
        return services
    
    async def get_endpoint_summary(self) -> Dict[str, Any]:
        """
        Get summary of all endpoint configurations and health.
        
        Returns:
            Endpoint summary with health information
        """
        endpoint_manager = get_endpoint_manager()
        summary = endpoint_manager.get_endpoint_summary()
        
        # Add health information
        health_results = await self.check_vllm_endpoints()
        
        for endpoint_id, endpoint_info in summary["endpoints"].items():
            if endpoint_id in health_results:
                health_result = health_results[endpoint_id]
                endpoint_info.update({
                    "health_status": health_result.status.value,
                    "last_health_check": health_result.timestamp,
                    "response_time_ms": health_result.response_time_ms,
                    "health_message": health_result.message
                })
        
        return summary
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self.start_time
    
    def reset_start_time(self):
        """Reset the start time (useful for testing)."""
        self.start_time = time.time()


# Global health service instance
_health_service: Optional[SystemHealthService] = None


def get_health_service() -> SystemHealthService:
    """
    Get the global health service instance.
    
    Returns:
        SystemHealthService instance
    """
    global _health_service
    
    if _health_service is None:
        _health_service = SystemHealthService()
    
    return _health_service


async def initialize_health_service():
    """Initialize the health service."""
    service = get_health_service()
    logger.info("Health service initialized")
    return service


async def cleanup_health_service():
    """Cleanup the health service."""
    global _health_service
    
    if _health_service:
        _health_service = None
    
    logger.info("Health service cleaned up")