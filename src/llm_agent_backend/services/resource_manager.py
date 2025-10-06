"""
Resource Management Service.

This module provides background resource management tasks including
connection pool optimization, cleanup of stale resources, and
system resource monitoring.
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, Optional

from .connection_manager import get_connection_manager
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ResourceManager:
    """
    Background resource management service.
    
    Handles periodic cleanup, optimization, and monitoring of system resources
    including connection pools, memory usage, and performance metrics.
    """
    
    def __init__(
        self,
        cleanup_interval: int = 300,  # 5 minutes
        optimization_interval: int = 600,  # 10 minutes
        monitoring_interval: int = 60,  # 1 minute
    ):
        """
        Initialize resource manager.
        
        Args:
            cleanup_interval: Interval for cleanup tasks in seconds
            optimization_interval: Interval for optimization tasks in seconds
            monitoring_interval: Interval for monitoring tasks in seconds
        """
        self.cleanup_interval = cleanup_interval
        self.optimization_interval = optimization_interval
        self.monitoring_interval = monitoring_interval
        
        # Task management
        self._cleanup_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self.cleanup_runs = 0
        self.optimization_runs = 0
        self.monitoring_runs = 0
        self.last_cleanup_time: Optional[datetime] = None
        self.last_optimization_time: Optional[datetime] = None
        self.last_monitoring_time: Optional[datetime] = None
        
        # Resource metrics
        self.resource_metrics: Dict[str, Any] = {}
        
    async def start(self) -> None:
        """Start background resource management tasks."""
        logger.info("Starting resource manager")
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Resource manager started successfully")
    
    async def stop(self) -> None:
        """Stop background resource management tasks."""
        logger.info("Stopping resource manager")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel and wait for tasks
        tasks = [
            self._cleanup_task,
            self._optimization_task,
            self._monitoring_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Resource manager stopped")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup task loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.cleanup_interval
                )
            except asyncio.TimeoutError:
                if not self._shutdown_event.is_set():
                    await self._run_cleanup()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _optimization_loop(self) -> None:
        """Background optimization task loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.optimization_interval
                )
            except asyncio.TimeoutError:
                if not self._shutdown_event.is_set():
                    await self._run_optimization()
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring task loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.monitoring_interval
                )
            except asyncio.TimeoutError:
                if not self._shutdown_event.is_set():
                    await self._run_monitoring()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _run_cleanup(self) -> None:
        """Run cleanup tasks."""
        start_time = time.time()
        
        try:
            logger.debug("Running resource cleanup")
            
            # Get connection manager
            connection_manager = await get_connection_manager()
            
            # Cleanup stale connection pools
            stale_pools_cleaned = await connection_manager.cleanup_stale_pools(
                max_idle_time=600  # 10 minutes
            )
            
            if stale_pools_cleaned > 0:
                logger.info(f"Cleaned up {stale_pools_cleaned} stale connection pools")
            
            # Update statistics
            self.cleanup_runs += 1
            self.last_cleanup_time = datetime.utcnow()
            
            cleanup_time = time.time() - start_time
            logger.debug(f"Cleanup completed in {cleanup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Cleanup task failed: {e}")
    
    async def _run_optimization(self) -> None:
        """Run optimization tasks."""
        start_time = time.time()
        
        try:
            logger.debug("Running resource optimization")
            
            # Get connection manager
            connection_manager = await get_connection_manager()
            
            # Optimize connection pool sizes
            await connection_manager.optimize_pool_sizes()
            
            # Update statistics
            self.optimization_runs += 1
            self.last_optimization_time = datetime.utcnow()
            
            optimization_time = time.time() - start_time
            logger.debug(f"Optimization completed in {optimization_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Optimization task failed: {e}")
    
    async def _run_monitoring(self) -> None:
        """Run monitoring tasks."""
        start_time = time.time()
        
        try:
            # Get connection manager
            connection_manager = await get_connection_manager()
            
            # Collect resource usage metrics
            resource_usage = await connection_manager.get_resource_usage()
            connection_stats = connection_manager.get_all_stats()
            
            # Update resource metrics
            self.resource_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "resource_usage": resource_usage,
                "connection_stats": connection_stats,
                "manager_stats": {
                    "cleanup_runs": self.cleanup_runs,
                    "optimization_runs": self.optimization_runs,
                    "monitoring_runs": self.monitoring_runs,
                    "last_cleanup": self.last_cleanup_time.isoformat() if self.last_cleanup_time else None,
                    "last_optimization": self.last_optimization_time.isoformat() if self.last_optimization_time else None,
                }
            }
            
            # Log warnings for high resource usage
            global_utilization = resource_usage.get("global_utilization", 0)
            if global_utilization > 0.8:
                logger.warning(f"High connection pool utilization: {global_utilization:.2%}")
            
            # Check for unhealthy pools
            unhealthy_pools = connection_manager.get_unhealthy_pools()
            if unhealthy_pools:
                logger.warning(f"Unhealthy connection pools detected: {unhealthy_pools}")
            
            # Update statistics
            self.monitoring_runs += 1
            self.last_monitoring_time = datetime.utcnow()
            
            monitoring_time = time.time() - start_time
            logger.debug(f"Monitoring completed in {monitoring_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Monitoring task failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get resource manager statistics.
        
        Returns:
            Resource manager statistics
        """
        return {
            "cleanup_runs": self.cleanup_runs,
            "optimization_runs": self.optimization_runs,
            "monitoring_runs": self.monitoring_runs,
            "last_cleanup_time": self.last_cleanup_time.isoformat() if self.last_cleanup_time else None,
            "last_optimization_time": self.last_optimization_time.isoformat() if self.last_optimization_time else None,
            "last_monitoring_time": self.last_monitoring_time.isoformat() if self.last_monitoring_time else None,
            "intervals": {
                "cleanup_interval": self.cleanup_interval,
                "optimization_interval": self.optimization_interval,
                "monitoring_interval": self.monitoring_interval,
            },
            "current_metrics": self.resource_metrics,
        }
    
    async def force_cleanup(self) -> Dict[str, Any]:
        """
        Force immediate cleanup run.
        
        Returns:
            Cleanup results
        """
        logger.info("Forcing immediate resource cleanup")
        
        start_time = time.time()
        await self._run_cleanup()
        cleanup_time = time.time() - start_time
        
        return {
            "status": "completed",
            "cleanup_time": cleanup_time,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def force_optimization(self) -> Dict[str, Any]:
        """
        Force immediate optimization run.
        
        Returns:
            Optimization results
        """
        logger.info("Forcing immediate resource optimization")
        
        start_time = time.time()
        await self._run_optimization()
        optimization_time = time.time() - start_time
        
        return {
            "status": "completed",
            "optimization_time": optimization_time,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None


async def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance."""
    global _resource_manager
    
    if _resource_manager is None:
        # Create resource manager with configuration
        _resource_manager = ResourceManager(
            cleanup_interval=300,  # 5 minutes
            optimization_interval=600,  # 10 minutes
            monitoring_interval=60,  # 1 minute
        )
        
        # Start background tasks
        await _resource_manager.start()
    
    return _resource_manager


async def shutdown_resource_manager() -> None:
    """Shutdown global resource manager."""
    global _resource_manager
    
    if _resource_manager:
        await _resource_manager.stop()
        _resource_manager = None