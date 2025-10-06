"""
Advanced Connection Pooling and Resource Management.

This module provides comprehensive connection pooling for aiohttp sessions,
resource management, graceful shutdown handling, and connection health monitoring
for both chat and embedding vLLM endpoints.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import weakref

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import VLLMEndpointConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ConnectionState(Enum):
    """Connection pool states."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTDOWN = "shutdown"


@dataclass
class ConnectionStats:
    """Statistics for connection pool monitoring."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    total_response_time: float = 0.0
    connection_errors: int = 0
    timeout_errors: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    
    def record_request_success(self, response_time: float) -> None:
        """Record successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_response_time += response_time
        self.average_response_time = self.total_response_time / self.successful_requests
    
    def record_request_failure(self, error: str, is_timeout: bool = False) -> None:
        """Record failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_error = error
        self.last_error_time = datetime.utcnow()
        
        if is_timeout:
            self.timeout_errors += 1
        else:
            self.connection_errors += 1
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_requests / max(self.total_requests, 1)
    
    def get_error_rate(self) -> float:
        """Calculate error rate."""
        return self.failed_requests / max(self.total_requests, 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "failed_connections": self.failed_connections,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.get_success_rate(),
            "error_rate": self.get_error_rate(),
            "average_response_time": self.average_response_time,
            "connection_errors": self.connection_errors,
            "timeout_errors": self.timeout_errors,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None
        }


class ConnectionPool:
    """
    Advanced connection pool for aiohttp sessions with health monitoring.
    
    Provides connection pooling, health checks, automatic retry logic,
    and graceful resource management for vLLM endpoints.
    """
    
    def __init__(
        self,
        endpoint_config: VLLMEndpointConfig,
        pool_id: str,
        max_concurrent_requests: int = 50,
        health_check_interval: int = 30,
        connection_timeout: int = 10,
        read_timeout: int = 60
    ):
        """
        Initialize connection pool.
        
        Args:
            endpoint_config: vLLM endpoint configuration
            pool_id: Unique identifier for this pool
            max_concurrent_requests: Maximum concurrent requests
            health_check_interval: Health check interval in seconds
            connection_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds
        """
        self.config = endpoint_config
        self.pool_id = pool_id
        self.max_concurrent_requests = max_concurrent_requests
        self.health_check_interval = health_check_interval
        
        # Connection management
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # State management
        self.state = ConnectionState.INITIALIZING
        self.stats = ConnectionStats()
        
        # Health monitoring
        self._last_health_check: Optional[datetime] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._consecutive_failures = 0
        self._circuit_breaker_open = False
        self._circuit_breaker_open_time: Optional[datetime] = None
        
        # Timeouts
        self.timeout = aiohttp.ClientTimeout(
            total=read_timeout,
            connect=connection_timeout,
            sock_read=read_timeout
        )
        
        # Shutdown handling
        self._shutdown_event = asyncio.Event()
        self._active_requests: Set[asyncio.Task] = set()
        
        logger.info(f"ConnectionPool {pool_id} initialized for {endpoint_config.base_url}")
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        logger.info(f"Initializing connection pool {self.pool_id}")
        
        try:
            # Create TCP connector with optimized settings
            self._connector = aiohttp.TCPConnector(
                limit=self.config.connection_pool_size,
                limit_per_host=self.config.connection_pool_size,
                ttl_dns_cache=300,  # 5 minutes DNS cache
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
                force_close=False,
                ssl=False  # Assuming HTTP for vLLM
            )
            
            # Create session with connector
            headers = {
                "Content-Type": "application/json",
                "User-Agent": f"llm-agent-backend-pool-{self.pool_id}/1.0"
            }
            
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=self.timeout,
                headers=headers,
                raise_for_status=False  # Handle status codes manually
            )
            
            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            # Perform initial health check
            await self._perform_health_check()
            
            self.state = ConnectionState.HEALTHY
            logger.info(f"Connection pool {self.pool_id} initialized successfully")
            
        except Exception as e:
            self.state = ConnectionState.UNHEALTHY
            logger.error(f"Failed to initialize connection pool {self.pool_id}: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the connection pool."""
        logger.info(f"Shutting down connection pool {self.pool_id}")
        
        self.state = ConnectionState.SHUTDOWN
        self._shutdown_event.set()
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Wait for active requests to complete (with timeout)
        if self._active_requests:
            logger.info(f"Waiting for {len(self._active_requests)} active requests to complete")
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._active_requests, return_exceptions=True),
                    timeout=30.0  # 30 second timeout for graceful shutdown
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for active requests, forcing shutdown")
                
                # Cancel remaining requests
                for task in self._active_requests:
                    if not task.done():
                        task.cancel()
        
        # Close session and connector
        if self._session:
            await self._session.close()
        
        if self._connector:
            await self._connector.close()
        
        logger.info(f"Connection pool {self.pool_id} shutdown complete")
    
    @asynccontextmanager
    async def get_session(self):
        """
        Get session with automatic resource management.
        
        Yields:
            aiohttp.ClientSession: Session for making requests
        """
        if self.state == ConnectionState.SHUTDOWN:
            raise RuntimeError(f"Connection pool {self.pool_id} is shutdown")
        
        if self._circuit_breaker_open:
            await self._check_circuit_breaker()
        
        if not self._session or self._session.closed:
            raise RuntimeError(f"Connection pool {self.pool_id} session not available")
        
        async with self._semaphore:
            try:
                yield self._session
            finally:
                pass  # Cleanup handled by session lifecycle
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def make_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """
        Make HTTP request with retry logic and monitoring.
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request arguments
            
        Returns:
            HTTP response
        """
        if self.state == ConnectionState.SHUTDOWN:
            raise RuntimeError(f"Connection pool {self.pool_id} is shutdown")
        
        if self._circuit_breaker_open:
            await self._check_circuit_breaker()
        
        start_time = time.time()
        request_task = None
        
        try:
            async with self.get_session() as session:
                # Create request task for tracking
                request_task = asyncio.create_task(
                    session.request(method, url, **kwargs)
                )
                self._active_requests.add(request_task)
                
                response = await request_task
                
                # Record success
                response_time = time.time() - start_time
                self.stats.record_request_success(response_time)
                self._consecutive_failures = 0
                
                # Update connection state
                if self.state == ConnectionState.DEGRADED:
                    self.state = ConnectionState.HEALTHY
                
                return response
                
        except asyncio.TimeoutError as e:
            self.stats.record_request_failure(str(e), is_timeout=True)
            self._handle_request_failure()
            raise
            
        except aiohttp.ClientError as e:
            self.stats.record_request_failure(str(e))
            self._handle_request_failure()
            raise
            
        except Exception as e:
            self.stats.record_request_failure(str(e))
            self._handle_request_failure()
            raise
            
        finally:
            if request_task:
                self._active_requests.discard(request_task)
    
    def _handle_request_failure(self) -> None:
        """Handle request failure and update circuit breaker."""
        self._consecutive_failures += 1
        
        # Update state based on failure count
        if self._consecutive_failures >= self.config.circuit_breaker_threshold:
            self._circuit_breaker_open = True
            self._circuit_breaker_open_time = datetime.utcnow()
            self.state = ConnectionState.UNHEALTHY
            
            logger.warning(
                f"Circuit breaker opened for pool {self.pool_id} "
                f"after {self._consecutive_failures} consecutive failures"
            )
        elif self._consecutive_failures >= 2:
            self.state = ConnectionState.DEGRADED
    
    async def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker should be closed."""
        if not self._circuit_breaker_open or not self._circuit_breaker_open_time:
            return
        
        # Check if timeout period has passed
        timeout_duration = timedelta(seconds=self.config.circuit_breaker_timeout)
        if datetime.utcnow() - self._circuit_breaker_open_time > timeout_duration:
            logger.info(f"Attempting to close circuit breaker for pool {self.pool_id}")
            
            try:
                # Perform health check
                await self._perform_health_check()
                
                # If successful, close circuit breaker
                self._circuit_breaker_open = False
                self._circuit_breaker_open_time = None
                self._consecutive_failures = 0
                self.state = ConnectionState.HEALTHY
                
                logger.info(f"Circuit breaker closed for pool {self.pool_id}")
                
            except Exception as e:
                logger.warning(f"Health check failed, keeping circuit breaker open: {e}")
        else:
            raise RuntimeError(f"Circuit breaker open for pool {self.pool_id}")
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.health_check_interval
                )
            except asyncio.TimeoutError:
                if not self._shutdown_event.is_set():
                    await self._perform_health_check()
            except Exception as e:
                logger.error(f"Error in health check loop for pool {self.pool_id}: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _perform_health_check(self) -> None:
        """Perform health check on the endpoint."""
        try:
            # Simple health check - try to connect to the endpoint
            health_url = f"{self.config.base_url}/health"
            
            async with self.get_session() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        self._last_health_check = datetime.utcnow()
                        logger.debug(f"Health check passed for pool {self.pool_id}")
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
                        
        except Exception as e:
            logger.warning(f"Health check failed for pool {self.pool_id}: {e}")
            if self.state == ConnectionState.HEALTHY:
                self.state = ConnectionState.DEGRADED
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        stats_dict = self.stats.to_dict()
        
        # Add pool-specific information
        stats_dict.update({
            "pool_id": self.pool_id,
            "endpoint_url": self.config.base_url,
            "state": self.state.value,
            "circuit_breaker_open": self._circuit_breaker_open,
            "consecutive_failures": self._consecutive_failures,
            "max_concurrent_requests": self.max_concurrent_requests,
            "active_requests": len(self._active_requests),
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
            "session_closed": self._session.closed if self._session else True
        })
        
        return stats_dict
    
    def is_healthy(self) -> bool:
        """Check if connection pool is healthy."""
        return self.state in [ConnectionState.HEALTHY, ConnectionState.DEGRADED]


class ConnectionManager:
    """
    Global connection manager for all vLLM endpoints.
    
    Manages multiple connection pools, provides unified interface,
    and handles resource lifecycle management.
    """
    
    def __init__(self, max_total_connections: int = 200):
        """
        Initialize connection manager.
        
        Args:
            max_total_connections: Maximum total connections across all pools
        """
        self.max_total_connections = max_total_connections
        
        # Connection pools
        self.pools: Dict[str, ConnectionPool] = {}
        self._pool_refs: Dict[str, weakref.ReferenceType] = {}
        
        # Global semaphore for connection limiting
        self._global_semaphore = asyncio.Semaphore(max_total_connections)
        
        # Statistics
        self.total_pools_created = 0
        self.total_pools_destroyed = 0
        
        # Shutdown handling
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"ConnectionManager initialized with max {max_total_connections} connections")
    
    async def get_or_create_pool(
        self,
        endpoint_config: VLLMEndpointConfig,
        pool_id: Optional[str] = None
    ) -> ConnectionPool:
        """
        Get existing or create new connection pool.
        
        Args:
            endpoint_config: Endpoint configuration
            pool_id: Optional pool identifier (generated if None)
            
        Returns:
            Connection pool instance
        """
        if not pool_id:
            pool_id = f"pool_{endpoint_config.base_url.replace('://', '_').replace('/', '_')}"
        
        # Check if pool already exists
        if pool_id in self.pools:
            pool = self.pools[pool_id]
            if pool.is_healthy():
                return pool
            else:
                # Remove unhealthy pool
                await self._remove_pool(pool_id)
        
        # Create new pool
        pool = ConnectionPool(
            endpoint_config=endpoint_config,
            pool_id=pool_id,
            max_concurrent_requests=min(
                endpoint_config.connection_pool_size,
                self.max_total_connections // max(len(self.pools) + 1, 1)
            )
        )
        
        await pool.initialize()
        
        self.pools[pool_id] = pool
        self.total_pools_created += 1
        
        # Create weak reference for cleanup
        def cleanup_callback(ref):
            self._pool_refs.pop(pool_id, None)
        
        self._pool_refs[pool_id] = weakref.ref(pool, cleanup_callback)
        
        logger.info(f"Created connection pool {pool_id} for {endpoint_config.base_url}")
        return pool
    
    async def get_pool(self, pool_id: str) -> Optional[ConnectionPool]:
        """
        Get existing connection pool.
        
        Args:
            pool_id: Pool identifier
            
        Returns:
            Connection pool or None if not found
        """
        return self.pools.get(pool_id)
    
    async def remove_pool(self, pool_id: str) -> bool:
        """
        Remove connection pool.
        
        Args:
            pool_id: Pool identifier
            
        Returns:
            True if pool was removed, False if not found
        """
        return await self._remove_pool(pool_id)
    
    async def _remove_pool(self, pool_id: str) -> bool:
        """Internal pool removal."""
        pool = self.pools.pop(pool_id, None)
        if pool:
            await pool.shutdown()
            self.total_pools_destroyed += 1
            self._pool_refs.pop(pool_id, None)
            logger.info(f"Removed connection pool {pool_id}")
            return True
        return False
    
    async def shutdown_all(self, timeout: float = 30.0) -> None:
        """
        Shutdown all connection pools with timeout.
        
        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        logger.info(f"Shutting down all {len(self.pools)} connection pools")
        
        self._shutdown_event.set()
        
        # Shutdown all pools concurrently with timeout
        shutdown_tasks = []
        for pool_id, pool in list(self.pools.items()):
            task = asyncio.create_task(self._remove_pool(pool_id))
            shutdown_tasks.append(task)
        
        if shutdown_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*shutdown_tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Shutdown timeout after {timeout}s, forcing closure")
                
                # Cancel remaining tasks
                for task in shutdown_tasks:
                    if not task.done():
                        task.cancel()
        
        logger.info("All connection pools shutdown complete")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all connection pools."""
        pool_stats = {}
        total_connections = 0
        total_requests = 0
        total_errors = 0
        
        for pool_id, pool in self.pools.items():
            stats = pool.get_stats()
            pool_stats[pool_id] = stats
            
            total_connections += stats.get("total_connections", 0)
            total_requests += stats.get("total_requests", 0)
            total_errors += stats.get("failed_requests", 0)
        
        return {
            "total_pools": len(self.pools),
            "total_pools_created": self.total_pools_created,
            "total_pools_destroyed": self.total_pools_destroyed,
            "total_connections": total_connections,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / max(total_requests, 1),
            "max_total_connections": self.max_total_connections,
            "pools": pool_stats
        }
    
    def get_healthy_pools(self) -> List[str]:
        """Get list of healthy pool IDs."""
        return [
            pool_id for pool_id, pool in self.pools.items()
            if pool.is_healthy()
        ]
    
    def get_unhealthy_pools(self) -> List[str]:
        """Get list of unhealthy pool IDs."""
        return [
            pool_id for pool_id, pool in self.pools.items()
            if not pool.is_healthy()
        ]
    
    async def cleanup_stale_pools(self, max_idle_time: int = 300) -> int:
        """
        Clean up stale connection pools that haven't been used recently.
        
        Args:
            max_idle_time: Maximum idle time in seconds before cleanup
            
        Returns:
            Number of pools cleaned up
        """
        current_time = time.time()
        stale_pools = []
        
        for pool_id, pool in self.pools.items():
            # Check if pool has been idle for too long
            if (hasattr(pool, '_last_health_check') and 
                pool._last_health_check and
                (current_time - pool._last_health_check.timestamp()) > max_idle_time):
                
                # Only cleanup if pool is not actively being used
                if len(pool._active_requests) == 0:
                    stale_pools.append(pool_id)
        
        # Remove stale pools
        cleanup_count = 0
        for pool_id in stale_pools:
            if await self._remove_pool(pool_id):
                cleanup_count += 1
                logger.info(f"Cleaned up stale connection pool: {pool_id}")
        
        return cleanup_count
    
    async def optimize_pool_sizes(self) -> None:
        """
        Optimize connection pool sizes based on usage patterns.
        
        Adjusts pool sizes dynamically based on request patterns and
        available system resources.
        """
        if not self.pools:
            return
        
        # Calculate optimal distribution
        optimal_size_per_pool = max(
            1, 
            self.max_total_connections // len(self.pools)
        )
        
        for pool_id, pool in self.pools.items():
            current_active = len(pool._active_requests)
            
            # Adjust semaphore limit based on usage
            if current_active > pool.max_concurrent_requests * 0.8:
                # High usage - consider increasing limit
                new_limit = min(
                    optimal_size_per_pool,
                    pool.max_concurrent_requests + 5
                )
                if new_limit > pool.max_concurrent_requests:
                    pool.max_concurrent_requests = new_limit
                    # Create new semaphore with updated limit
                    pool._semaphore = asyncio.Semaphore(new_limit)
                    logger.debug(f"Increased pool {pool_id} limit to {new_limit}")
            
            elif current_active < pool.max_concurrent_requests * 0.2:
                # Low usage - consider decreasing limit
                new_limit = max(
                    1,
                    pool.max_concurrent_requests - 2
                )
                if new_limit < pool.max_concurrent_requests:
                    pool.max_concurrent_requests = new_limit
                    # Create new semaphore with updated limit
                    pool._semaphore = asyncio.Semaphore(new_limit)
                    logger.debug(f"Decreased pool {pool_id} limit to {new_limit}")
    
    async def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get detailed resource usage information.
        
        Returns:
            Resource usage statistics
        """
        total_active_requests = sum(
            len(pool._active_requests) for pool in self.pools.values()
        )
        
        total_max_connections = sum(
            pool.max_concurrent_requests for pool in self.pools.values()
        )
        
        pool_details = {}
        for pool_id, pool in self.pools.items():
            pool_details[pool_id] = {
                "active_requests": len(pool._active_requests),
                "max_concurrent": pool.max_concurrent_requests,
                "utilization": len(pool._active_requests) / max(pool.max_concurrent_requests, 1),
                "state": pool.state.value,
                "consecutive_failures": pool._consecutive_failures,
                "circuit_breaker_open": pool._circuit_breaker_open,
                "session_closed": pool._session.closed if pool._session else True
            }
        
        return {
            "total_pools": len(self.pools),
            "total_active_requests": total_active_requests,
            "total_max_connections": total_max_connections,
            "global_utilization": total_active_requests / max(total_max_connections, 1),
            "available_capacity": self.max_total_connections - total_active_requests,
            "pool_details": pool_details,
            "memory_usage": {
                "processing_batches": len(getattr(self, '_processing_batches', {})),
                "weak_references": len(self._pool_refs)
            }
        }


# Global connection manager instance
_connection_manager: Optional[ConnectionManager] = None


async def get_connection_manager() -> ConnectionManager:
    """Get global connection manager instance."""
    global _connection_manager
    
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    
    return _connection_manager


async def shutdown_connection_manager() -> None:
    """Shutdown global connection manager."""
    global _connection_manager
    
    if _connection_manager:
        await _connection_manager.shutdown_all()
        _connection_manager = None