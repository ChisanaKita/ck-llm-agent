"""
Endpoint Manager for vLLM configurations.

This module provides centralized management of vLLM endpoint configurations,
supporting local, Runpod, and other external service configurations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from .qwen_vllm import QwenVLLM
from ..config import get_settings, VLLMEndpointConfig


logger = logging.getLogger(__name__)


class EndpointType(str, Enum):
    """Supported endpoint types."""
    LOCAL = "local"
    RUNPOD = "runpod"
    HUGGINGFACE = "huggingface"
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    EXTERNAL = "external"


@dataclass
class EndpointStatus:
    """Endpoint status information."""
    endpoint_id: str
    endpoint_type: EndpointType
    base_url: str
    model: str
    is_healthy: bool
    last_check: float
    error_count: int
    success_count: int
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None


class VLLMEndpointManager:
    """
    Manages multiple vLLM endpoint configurations and health monitoring.
    
    Provides centralized management of different vLLM endpoints including
    local servers, Runpod instances, and other cloud providers.
    """
    
    def __init__(self):
        """Initialize endpoint manager."""
        self.endpoints: Dict[str, QwenVLLM] = {}
        self.endpoint_configs: Dict[str, Dict[str, Any]] = {}
        self.health_check_interval = 30  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Load configuration from settings
        self._load_from_settings()
    
    def _load_from_settings(self):
        """Load endpoint configurations from application settings."""
        settings = get_settings()
        
        # Add primary chat endpoint
        chat_config = settings.vllm.get_chat_config()
        self.add_endpoint(
            endpoint_id="primary_chat",
            config=chat_config,
            endpoint_type=self._detect_endpoint_type(chat_config.base_url)
        )
        
        # Add embedding endpoint if different from chat
        embedding_config = settings.vllm.get_embedding_config()
        if embedding_config.base_url != chat_config.base_url:
            self.add_endpoint(
                endpoint_id="primary_embedding",
                config=embedding_config,
                endpoint_type=self._detect_endpoint_type(embedding_config.base_url)
            )
        
        logger.info(f"Loaded {len(self.endpoints)} endpoints from settings")
    
    def _detect_endpoint_type(self, url: str) -> EndpointType:
        """
        Detect endpoint type from URL.
        
        Args:
            url: Endpoint URL
            
        Returns:
            Detected endpoint type
        """
        url_lower = url.lower()
        
        if 'runpod' in url_lower:
            return EndpointType.RUNPOD
        elif 'huggingface' in url_lower or 'hf.co' in url_lower:
            return EndpointType.HUGGINGFACE
        elif 'amazonaws.com' in url_lower:
            return EndpointType.AWS
        elif 'googleapis.com' in url_lower or 'gcp' in url_lower:
            return EndpointType.GCP
        elif 'azure' in url_lower:
            return EndpointType.AZURE
        elif 'localhost' in url_lower or '127.0.0.1' in url_lower:
            return EndpointType.LOCAL
        else:
            return EndpointType.EXTERNAL
    
    def add_endpoint(
        self,
        endpoint_id: str,
        config: Union[VLLMEndpointConfig, Dict[str, Any]],
        endpoint_type: Optional[EndpointType] = None,
        **kwargs
    ) -> QwenVLLM:
        """
        Add a new vLLM endpoint.
        
        Args:
            endpoint_id: Unique identifier for the endpoint
            config: Endpoint configuration
            endpoint_type: Type of endpoint (auto-detected if not provided)
            **kwargs: Additional configuration options
            
        Returns:
            Created QwenVLLM instance
        """
        if isinstance(config, dict):
            # Convert dict to VLLMEndpointConfig
            endpoint_config = VLLMEndpointConfig(**config)
        else:
            endpoint_config = config
        
        if not endpoint_type:
            endpoint_type = self._detect_endpoint_type(endpoint_config.base_url)
        
        # Create QwenVLLM instance based on endpoint type
        if endpoint_type == EndpointType.RUNPOD:
            # Extract Runpod-specific configuration
            pod_id = kwargs.get('pod_id') or self._extract_runpod_id(endpoint_config.base_url)
            if pod_id and endpoint_config.api_key:
                handler = QwenVLLM.create_for_runpod(
                    pod_id=pod_id,
                    api_key=endpoint_config.api_key,
                    model=endpoint_config.model,
                    **kwargs
                )
            else:
                handler = QwenVLLM.create_for_external(
                    base_url=endpoint_config.base_url,
                    model=endpoint_config.model,
                    api_key=endpoint_config.api_key,
                    **kwargs
                )
        elif endpoint_type == EndpointType.LOCAL:
            handler = QwenVLLM.create_for_local(
                base_url=endpoint_config.base_url,
                model=endpoint_config.model,
                **kwargs
            )
        else:
            handler = QwenVLLM.create_for_external(
                base_url=endpoint_config.base_url,
                model=endpoint_config.model,
                api_key=endpoint_config.api_key,
                **kwargs
            )
        
        # Store endpoint
        self.endpoints[endpoint_id] = handler
        self.endpoint_configs[endpoint_id] = {
            'config': endpoint_config,
            'type': endpoint_type,
            'kwargs': kwargs
        }
        
        logger.info(
            f"Added {endpoint_type.value} endpoint '{endpoint_id}': "
            f"{endpoint_config.base_url} ({endpoint_config.model})"
        )
        
        return handler
    
    def _extract_runpod_id(self, url: str) -> Optional[str]:
        """
        Extract Runpod pod ID from URL.
        
        Args:
            url: Runpod URL
            
        Returns:
            Pod ID if found
        """
        import re
        
        # Match pattern: https://{pod_id}-8000.proxy.runpod.net
        match = re.search(r'https://([^-]+)-\d+\.proxy\.runpod\.net', url)
        return match.group(1) if match else None
    
    def get_endpoint(self, endpoint_id: str) -> Optional[QwenVLLM]:
        """
        Get endpoint by ID.
        
        Args:
            endpoint_id: Endpoint identifier
            
        Returns:
            QwenVLLM instance if found
        """
        return self.endpoints.get(endpoint_id)
    
    def get_primary_chat_endpoint(self) -> Optional[QwenVLLM]:
        """Get the primary chat endpoint."""
        return self.get_endpoint("primary_chat")
    
    def get_primary_embedding_endpoint(self) -> Optional[QwenVLLM]:
        """Get the primary embedding endpoint."""
        return self.get_endpoint("primary_embedding") or self.get_endpoint("primary_chat")
    
    def list_endpoints(self) -> List[str]:
        """List all endpoint IDs."""
        return list(self.endpoints.keys())
    
    def remove_endpoint(self, endpoint_id: str) -> bool:
        """
        Remove an endpoint.
        
        Args:
            endpoint_id: Endpoint identifier
            
        Returns:
            True if endpoint was removed
        """
        if endpoint_id in self.endpoints:
            # Close the endpoint
            asyncio.create_task(self.endpoints[endpoint_id].close())
            
            # Remove from tracking
            del self.endpoints[endpoint_id]
            del self.endpoint_configs[endpoint_id]
            
            logger.info(f"Removed endpoint '{endpoint_id}'")
            return True
        
        return False
    
    async def check_all_health(self) -> Dict[str, EndpointStatus]:
        """
        Check health of all endpoints.
        
        Returns:
            Health status for all endpoints
        """
        results = {}
        
        for endpoint_id, handler in self.endpoints.items():
            try:
                health_data = await handler.check_health()
                endpoint_config = self.endpoint_configs[endpoint_id]
                
                results[endpoint_id] = EndpointStatus(
                    endpoint_id=endpoint_id,
                    endpoint_type=endpoint_config['type'],
                    base_url=handler.base_url,
                    model=handler.model,
                    is_healthy=health_data['overall_status'] == 'healthy',
                    last_check=health_data['endpoint_health'].get('timestamp', 0),
                    error_count=handler.error_count,
                    success_count=handler.request_count - handler.error_count,
                    response_time_ms=health_data['endpoint_health'].get('response_time_ms'),
                    error_message=health_data['endpoint_health'].get('error')
                )
                
            except Exception as e:
                results[endpoint_id] = EndpointStatus(
                    endpoint_id=endpoint_id,
                    endpoint_type=self.endpoint_configs[endpoint_id]['type'],
                    base_url=handler.base_url,
                    model=handler.model,
                    is_healthy=False,
                    last_check=0,
                    error_count=handler.error_count,
                    success_count=handler.request_count - handler.error_count,
                    error_message=str(e)
                )
        
        return results
    
    async def get_healthy_endpoints(self) -> List[str]:
        """
        Get list of healthy endpoint IDs.
        
        Returns:
            List of healthy endpoint IDs
        """
        health_status = await self.check_all_health()
        return [
            endpoint_id for endpoint_id, status in health_status.items()
            if status.is_healthy
        ]
    
    async def start_health_monitoring(self):
        """Start background health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Started endpoint health monitoring")
    
    async def stop_health_monitoring(self):
        """Stop background health monitoring."""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
        
        logger.info("Stopped endpoint health monitoring")
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while self._running:
            try:
                health_status = await self.check_all_health()
                
                # Log unhealthy endpoints
                unhealthy = [
                    endpoint_id for endpoint_id, status in health_status.items()
                    if not status.is_healthy
                ]
                
                if unhealthy:
                    logger.warning(f"Unhealthy endpoints: {unhealthy}")
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def close_all(self):
        """Close all endpoints and cleanup."""
        await self.stop_health_monitoring()
        
        for endpoint_id, handler in self.endpoints.items():
            try:
                await handler.close()
            except Exception as e:
                logger.error(f"Error closing endpoint '{endpoint_id}': {e}")
        
        self.endpoints.clear()
        self.endpoint_configs.clear()
        
        logger.info("Closed all endpoints")
    
    def get_endpoint_summary(self) -> Dict[str, Any]:
        """
        Get summary of all endpoints.
        
        Returns:
            Summary information for all endpoints
        """
        summary = {
            'total_endpoints': len(self.endpoints),
            'endpoint_types': {},
            'endpoints': {}
        }
        
        for endpoint_id, handler in self.endpoints.items():
            config = self.endpoint_configs[endpoint_id]
            endpoint_type = config['type']
            
            # Count by type
            if endpoint_type.value not in summary['endpoint_types']:
                summary['endpoint_types'][endpoint_type.value] = 0
            summary['endpoint_types'][endpoint_type.value] += 1
            
            # Endpoint details
            summary['endpoints'][endpoint_id] = {
                'type': endpoint_type.value,
                'base_url': handler.base_url,
                'model': handler.model,
                'request_count': handler.request_count,
                'error_count': handler.error_count,
                'error_rate': handler.error_count / max(handler.request_count, 1)
            }
        
        return summary


# Global endpoint manager instance
_endpoint_manager: Optional[VLLMEndpointManager] = None


def get_endpoint_manager() -> VLLMEndpointManager:
    """
    Get the global endpoint manager instance.
    
    Returns:
        VLLMEndpointManager instance
    """
    global _endpoint_manager
    
    if _endpoint_manager is None:
        _endpoint_manager = VLLMEndpointManager()
    
    return _endpoint_manager


async def initialize_endpoints():
    """Initialize the endpoint manager and start health monitoring."""
    manager = get_endpoint_manager()
    await manager.start_health_monitoring()
    logger.info("Endpoint manager initialized")


async def cleanup_endpoints():
    """Cleanup all endpoints and stop monitoring."""
    global _endpoint_manager
    
    if _endpoint_manager:
        await _endpoint_manager.close_all()
        _endpoint_manager = None
    
    logger.info("Endpoint manager cleaned up")