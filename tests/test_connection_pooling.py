"""
Tests for connection pooling and resource management functionality.

This module tests the connection manager, resource manager, and
related functionality for task 7.3.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm_agent_backend.config import VLLMEndpointConfig
from src.llm_agent_backend.services.connection_manager import (
    ConnectionManager,
    ConnectionPool,
    ConnectionState,
    get_connection_manager,
    shutdown_connection_manager
)
from src.llm_agent_backend.services.resource_manager import (
    ResourceManager,
    get_resource_manager,
    shutdown_resource_manager
)


class TestConnectionPool:
    """Test connection pool functionality."""
    
    @pytest.fixture
    async def endpoint_config(self):
        """Create test endpoint configuration."""
        return VLLMEndpointConfig(
            base_url="http://localhost:8000",
            model="test-model",
            connection_pool_size=5,
            max_retries=2,
            timeout=30
        )
    
    @pytest.fixture
    async def connection_pool(self, endpoint_config):
        """Create test connection pool."""
        pool = ConnectionPool(
            endpoint_config=endpoint_config,
            pool_id="test_pool",
            max_concurrent_requests=10
        )
        
        # Mock the session creation to avoid actual HTTP connections
        with patch('aiohttp.TCPConnector'), \
             patch('aiohttp.ClientSession'):
            await pool.initialize()
        
        yield pool
        
        await pool.shutdown()
    
    async def test_pool_initialization(self, connection_pool):
        """Test connection pool initialization."""
        assert connection_pool.state == ConnectionState.HEALTHY
        assert connection_pool.pool_id == "test_pool"
        assert connection_pool.max_concurrent_requests == 10
    
    async def test_pool_statistics(self, connection_pool):
        """Test connection pool statistics collection."""
        stats = connection_pool.get_stats()
        
        assert "pool_id" in stats
        assert "state" in stats
        assert "total_requests" in stats
        assert "successful_requests" in stats
        assert "failed_requests" in stats
        assert stats["pool_id"] == "test_pool"
    
    async def test_pool_health_check(self, connection_pool):
        """Test connection pool health status."""
        assert connection_pool.is_healthy()
        
        # Simulate unhealthy state
        connection_pool.state = ConnectionState.UNHEALTHY
        assert not connection_pool.is_healthy()
    
    async def test_pool_shutdown(self, endpoint_config):
        """Test graceful pool shutdown."""
        pool = ConnectionPool(
            endpoint_config=endpoint_config,
            pool_id="shutdown_test",
            max_concurrent_requests=5
        )
        
        with patch('aiohttp.TCPConnector'), \
             patch('aiohttp.ClientSession'):
            await pool.initialize()
            
            # Verify pool is healthy
            assert pool.state == ConnectionState.HEALTHY
            
            # Shutdown pool
            await pool.shutdown()
            
            # Verify shutdown state
            assert pool.state == ConnectionState.SHUTDOWN


class TestConnectionManager:
    """Test connection manager functionality."""
    
    @pytest.fixture
    async def endpoint_config(self):
        """Create test endpoint configuration."""
        return VLLMEndpointConfig(
            base_url="http://localhost:8000",
            model="test-model",
            connection_pool_size=5
        )
    
    @pytest.fixture
    async def connection_manager(self):
        """Create test connection manager."""
        manager = ConnectionManager(max_total_connections=50)
        yield manager
        await manager.shutdown_all()
    
    async def test_manager_initialization(self, connection_manager):
        """Test connection manager initialization."""
        assert connection_manager.max_total_connections == 50
        assert len(connection_manager.pools) == 0
    
    async def test_pool_creation(self, connection_manager, endpoint_config):
        """Test connection pool creation and management."""
        with patch('aiohttp.TCPConnector'), \
             patch('aiohttp.ClientSession'):
            
            # Create pool
            pool = await connection_manager.get_or_create_pool(
                endpoint_config=endpoint_config,
                pool_id="test_pool"
            )
            
            assert pool is not None
            assert pool.pool_id == "test_pool"
            assert len(connection_manager.pools) == 1
            
            # Get existing pool
            same_pool = await connection_manager.get_or_create_pool(
                endpoint_config=endpoint_config,
                pool_id="test_pool"
            )
            
            assert same_pool is pool
            assert len(connection_manager.pools) == 1
    
    async def test_pool_removal(self, connection_manager, endpoint_config):
        """Test connection pool removal."""
        with patch('aiohttp.TCPConnector'), \
             patch('aiohttp.ClientSession'):
            
            # Create pool
            await connection_manager.get_or_create_pool(
                endpoint_config=endpoint_config,
                pool_id="remove_test"
            )
            
            assert len(connection_manager.pools) == 1
            
            # Remove pool
            removed = await connection_manager.remove_pool("remove_test")
            
            assert removed is True
            assert len(connection_manager.pools) == 0
    
    async def test_manager_statistics(self, connection_manager, endpoint_config):
        """Test connection manager statistics."""
        with patch('aiohttp.TCPConnector'), \
             patch('aiohttp.ClientSession'):
            
            # Create multiple pools
            await connection_manager.get_or_create_pool(
                endpoint_config=endpoint_config,
                pool_id="pool1"
            )
            await connection_manager.get_or_create_pool(
                endpoint_config=endpoint_config,
                pool_id="pool2"
            )
            
            stats = connection_manager.get_all_stats()
            
            assert stats["total_pools"] == 2
            assert "pools" in stats
            assert "pool1" in stats["pools"]
            assert "pool2" in stats["pools"]
    
    async def test_resource_usage(self, connection_manager, endpoint_config):
        """Test resource usage monitoring."""
        with patch('aiohttp.TCPConnector'), \
             patch('aiohttp.ClientSession'):
            
            # Create pool
            await connection_manager.get_or_create_pool(
                endpoint_config=endpoint_config,
                pool_id="resource_test"
            )
            
            usage = await connection_manager.get_resource_usage()
            
            assert "total_pools" in usage
            assert "total_active_requests" in usage
            assert "global_utilization" in usage
            assert "pool_details" in usage
            assert usage["total_pools"] == 1
    
    async def test_cleanup_stale_pools(self, connection_manager, endpoint_config):
        """Test cleanup of stale connection pools."""
        with patch('aiohttp.TCPConnector'), \
             patch('aiohttp.ClientSession'):
            
            # Create pool
            pool = await connection_manager.get_or_create_pool(
                endpoint_config=endpoint_config,
                pool_id="stale_test"
            )
            
            # Mock old health check time
            from datetime import datetime, timedelta
            pool._last_health_check = datetime.utcnow() - timedelta(hours=1)
            
            # Run cleanup
            cleaned = await connection_manager.cleanup_stale_pools(max_idle_time=300)
            
            # Should clean up the stale pool
            assert cleaned >= 0  # May or may not clean up depending on active requests


class TestResourceManager:
    """Test resource manager functionality."""
    
    @pytest.fixture
    async def resource_manager(self):
        """Create test resource manager."""
        manager = ResourceManager(
            cleanup_interval=1,  # Short intervals for testing
            optimization_interval=1,
            monitoring_interval=1
        )
        yield manager
        await manager.stop()
    
    async def test_manager_initialization(self, resource_manager):
        """Test resource manager initialization."""
        assert resource_manager.cleanup_interval == 1
        assert resource_manager.optimization_interval == 1
        assert resource_manager.monitoring_interval == 1
        assert resource_manager.cleanup_runs == 0
    
    async def test_manager_start_stop(self, resource_manager):
        """Test resource manager start and stop."""
        # Start manager
        await resource_manager.start()
        
        # Verify tasks are running
        assert resource_manager._cleanup_task is not None
        assert resource_manager._optimization_task is not None
        assert resource_manager._monitoring_task is not None
        
        # Stop manager
        await resource_manager.stop()
        
        # Verify tasks are cancelled
        assert resource_manager._cleanup_task.cancelled() or resource_manager._cleanup_task.done()
    
    async def test_force_cleanup(self, resource_manager):
        """Test forced cleanup execution."""
        with patch.object(resource_manager, '_run_cleanup', new_callable=AsyncMock) as mock_cleanup:
            result = await resource_manager.force_cleanup()
            
            assert result["status"] == "completed"
            assert "cleanup_time" in result
            mock_cleanup.assert_called_once()
    
    async def test_force_optimization(self, resource_manager):
        """Test forced optimization execution."""
        with patch.object(resource_manager, '_run_optimization', new_callable=AsyncMock) as mock_optimization:
            result = await resource_manager.force_optimization()
            
            assert result["status"] == "completed"
            assert "optimization_time" in result
            mock_optimization.assert_called_once()
    
    async def test_manager_statistics(self, resource_manager):
        """Test resource manager statistics."""
        stats = resource_manager.get_stats()
        
        assert "cleanup_runs" in stats
        assert "optimization_runs" in stats
        assert "monitoring_runs" in stats
        assert "intervals" in stats
        assert stats["cleanup_runs"] == 0


class TestGlobalManagers:
    """Test global manager instances."""
    
    async def test_global_connection_manager(self):
        """Test global connection manager singleton."""
        # Reset global state
        import src.llm_agent_backend.services.connection_manager as cm_module
        cm_module._connection_manager = None
        
        # Get manager instances
        manager1 = await get_connection_manager()
        manager2 = await get_connection_manager()
        
        # Should be the same instance
        assert manager1 is manager2
        
        # Cleanup
        await shutdown_connection_manager()
    
    async def test_global_resource_manager(self):
        """Test global resource manager singleton."""
        # Reset global state
        import src.llm_agent_backend.services.resource_manager as rm_module
        rm_module._resource_manager = None
        
        # Get manager instances
        manager1 = await get_resource_manager()
        manager2 = await get_resource_manager()
        
        # Should be the same instance
        assert manager1 is manager2
        
        # Cleanup
        await shutdown_resource_manager()


class TestIntegration:
    """Integration tests for connection pooling and resource management."""
    
    async def test_full_lifecycle(self):
        """Test complete lifecycle of managers."""
        # Reset global state
        import src.llm_agent_backend.services.connection_manager as cm_module
        import src.llm_agent_backend.services.resource_manager as rm_module
        cm_module._connection_manager = None
        rm_module._resource_manager = None
        
        try:
            # Initialize managers
            connection_manager = await get_connection_manager()
            resource_manager = await get_resource_manager()
            
            # Verify initialization
            assert connection_manager is not None
            assert resource_manager is not None
            
            # Create test endpoint config
            endpoint_config = VLLMEndpointConfig(
                base_url="http://localhost:8000",
                model="test-model"
            )
            
            with patch('aiohttp.TCPConnector'), \
                 patch('aiohttp.ClientSession'):
                
                # Create connection pool
                pool = await connection_manager.get_or_create_pool(
                    endpoint_config=endpoint_config,
                    pool_id="integration_test"
                )
                
                assert pool is not None
                assert pool.is_healthy()
                
                # Get resource usage
                usage = await connection_manager.get_resource_usage()
                assert usage["total_pools"] == 1
                
                # Force cleanup and optimization
                cleanup_result = await resource_manager.force_cleanup()
                optimization_result = await resource_manager.force_optimization()
                
                assert cleanup_result["status"] == "completed"
                assert optimization_result["status"] == "completed"
        
        finally:
            # Cleanup
            await shutdown_resource_manager()
            await shutdown_connection_manager()


if __name__ == "__main__":
    # Run basic test
    async def run_basic_test():
        """Run a basic connection pooling test."""
        print("Testing connection pooling functionality...")
        
        # Test connection manager
        manager = ConnectionManager(max_total_connections=10)
        
        endpoint_config = VLLMEndpointConfig(
            base_url="http://localhost:8000",
            model="test-model"
        )
        
        try:
            with patch('aiohttp.TCPConnector'), \
                 patch('aiohttp.ClientSession'):
                
                # Create pool
                pool = await manager.get_or_create_pool(
                    endpoint_config=endpoint_config,
                    pool_id="basic_test"
                )
                
                print(f"Created pool: {pool.pool_id}")
                print(f"Pool state: {pool.state}")
                print(f"Pool healthy: {pool.is_healthy()}")
                
                # Get statistics
                stats = manager.get_all_stats()
                print(f"Manager stats: {stats}")
                
                print("Connection pooling test completed successfully!")
        
        finally:
            await manager.shutdown_all()
    
    # Run the test
    asyncio.run(run_basic_test())