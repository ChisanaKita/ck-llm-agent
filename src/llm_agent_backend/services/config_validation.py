"""
Configuration validation service for vLLM endpoints.

This service validates vLLM endpoint configurations, tests connectivity,
and provides configuration recommendations for different deployment scenarios.
"""

import asyncio
import logging
from typing import Dict, Optional, Any
from urllib.parse import urlparse
import aiohttp

from ..config import VLLMEndpointConfig, get_settings
from ..handlers.endpoint_manager import EndpointType


logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class EndpointValidator:
    """
    Validates vLLM endpoint configurations and connectivity.
    
    Provides comprehensive validation for different endpoint types including
    local servers, Runpod instances, and other cloud providers.
    """
    
    def __init__(self):
        """Initialize the endpoint validator."""
        self.timeout = 10.0  # Connection timeout for validation
        
    async def validate_endpoint_config(
        self, 
        config: VLLMEndpointConfig,
        endpoint_type: Optional[EndpointType] = None
    ) -> Dict[str, Any]:
        """
        Validate a single endpoint configuration.
        
        Args:
            config: Endpoint configuration to validate
            endpoint_type: Type of endpoint (auto-detected if not provided)
            
        Returns:
            Validation result with status and details
        """
        result = {
            "valid": False,
            "endpoint_type": endpoint_type,
            "issues": [],
            "warnings": [],
            "connectivity": None,
            "model_info": None,
            "recommendations": []
        }
        
        try:
            # Basic configuration validation
            self._validate_basic_config(config, result)
            
            # Detect endpoint type if not provided
            if not endpoint_type:
                endpoint_type = self._detect_endpoint_type(config.base_url)
                result["endpoint_type"] = endpoint_type
            
            # Type-specific validation
            self._validate_endpoint_type_specific(config, endpoint_type, result)
            
            # Test connectivity
            connectivity_result = await self._test_connectivity(config)
            result["connectivity"] = connectivity_result
            
            # Get model information if connected
            if connectivity_result.get("connected", False):
                model_info = await self._get_model_info(config)
                result["model_info"] = model_info
                
                # Validate model compatibility
                self._validate_model_compatibility(config, model_info, result)
            
            # Generate recommendations
            self._generate_recommendations(config, endpoint_type, result)
            
            # Determine overall validity
            result["valid"] = len(result["issues"]) == 0 and connectivity_result.get("connected", False)
            
        except Exception as e:
            result["issues"].append(f"Validation error: {str(e)}")
            logger.error(f"Error validating endpoint config: {e}", exc_info=True)
        
        return result
    
    def _validate_basic_config(self, config: VLLMEndpointConfig, result: Dict[str, Any]):
        """Validate basic configuration parameters."""
        # URL validation
        try:
            parsed_url = urlparse(config.base_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                result["issues"].append("Invalid base URL format")
            elif parsed_url.scheme not in ["http", "https"]:
                result["issues"].append("Base URL must use http or https scheme")
        except Exception:
            result["issues"].append("Failed to parse base URL")
        
        # Model validation
        if not config.model or not config.model.strip():
            result["issues"].append("Model name is required")
        
        # Timeout validation
        if config.timeout < 5:
            result["warnings"].append("Timeout is very low (< 5s), may cause connection issues")
        elif config.timeout > 300:
            result["warnings"].append("Timeout is very high (> 300s), may cause hanging requests")
        
        # Retry validation
        if config.max_retries < 1:
            result["issues"].append("Max retries must be at least 1")
        elif config.max_retries > 10:
            result["warnings"].append("High retry count may cause long delays on failures")
        
        # Connection pool validation
        if config.connection_pool_size < 1:
            result["issues"].append("Connection pool size must be at least 1")
        elif config.connection_pool_size > 100:
            result["warnings"].append("Large connection pool may consume excessive resources")
    
    def _detect_endpoint_type(self, url: str) -> EndpointType:
        """Detect endpoint type from URL."""
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
    
    def _validate_endpoint_type_specific(
        self, 
        config: VLLMEndpointConfig, 
        endpoint_type: EndpointType, 
        result: Dict[str, Any]
    ):
        """Perform endpoint type-specific validation."""
        if endpoint_type == EndpointType.RUNPOD:
            self._validate_runpod_config(config, result)
        elif endpoint_type == EndpointType.LOCAL:
            self._validate_local_config(config, result)
        elif endpoint_type in [EndpointType.AWS, EndpointType.GCP, EndpointType.AZURE]:
            self._validate_cloud_config(config, result)
        elif endpoint_type == EndpointType.HUGGINGFACE:
            self._validate_huggingface_config(config, result)
    
    def _validate_runpod_config(self, config: VLLMEndpointConfig, result: Dict[str, Any]):
        """Validate Runpod-specific configuration."""
        # Check URL format
        if not config.base_url.endswith('.proxy.runpod.net'):
            result["warnings"].append("URL doesn't match standard Runpod proxy format")
        
        # API key validation
        if not config.api_key:
            result["issues"].append("Runpod endpoints typically require an API key")
        
        # Port validation
        if ':8000' not in config.base_url and ':8001' not in config.base_url:
            result["warnings"].append("Runpod endpoints typically use port 8000 or 8001")
    
    def _validate_local_config(self, config: VLLMEndpointConfig, result: Dict[str, Any]):
        """Validate local endpoint configuration."""
        # Check if using localhost or local IP
        if 'localhost' not in config.base_url and '127.0.0.1' not in config.base_url:
            result["warnings"].append("Local endpoint should use localhost or 127.0.0.1")
        
        # API key usually not needed for local
        if config.api_key:
            result["warnings"].append("Local endpoints typically don't require API keys")
        
        # Check common ports
        if ':8000' not in config.base_url and ':8001' not in config.base_url:
            result["warnings"].append("vLLM typically runs on port 8000 or 8001")
    
    def _validate_cloud_config(self, config: VLLMEndpointConfig, result: Dict[str, Any]):
        """Validate cloud provider configuration."""
        # HTTPS requirement for cloud endpoints
        if not config.base_url.startswith('https://'):
            result["issues"].append("Cloud endpoints should use HTTPS")
        
        # API key typically required
        if not config.api_key:
            result["warnings"].append("Cloud endpoints typically require authentication")
    
    def _validate_huggingface_config(self, config: VLLMEndpointConfig, result: Dict[str, Any]):
        """Validate Hugging Face endpoint configuration."""
        # HTTPS requirement
        if not config.base_url.startswith('https://'):
            result["issues"].append("Hugging Face endpoints should use HTTPS")
        
        # API key validation
        if not config.api_key:
            result["warnings"].append("Hugging Face endpoints may require an API token")
    
    async def _test_connectivity(self, config: VLLMEndpointConfig) -> Dict[str, Any]:
        """Test connectivity to the endpoint."""
        result = {
            "connected": False,
            "response_time_ms": None,
            "status_code": None,
            "error": None
        }
        
        try:
            # Build headers
            headers = {"Content-Type": "application/json"}
            if config.api_key:
                if 'runpod' in config.base_url.lower():
                    headers["Authorization"] = f"Bearer {config.api_key}"
                else:
                    headers["X-API-Key"] = config.api_key
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                start_time = asyncio.get_event_loop().time()
                
                # Try to get models endpoint (lightweight check)
                url = f"{config.base_url.rstrip('/')}/v1/models"
                
                async with session.get(url) as response:
                    end_time = asyncio.get_event_loop().time()
                    response_time = (end_time - start_time) * 1000
                    
                    result["response_time_ms"] = response_time
                    result["status_code"] = response.status
                    
                    if response.status == 200:
                        result["connected"] = True
                    else:
                        result["error"] = f"HTTP {response.status}"
                        
        except asyncio.TimeoutError:
            result["error"] = f"Connection timeout after {self.timeout}s"
        except aiohttp.ClientConnectorError as e:
            result["error"] = f"Connection failed: {str(e)}"
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
        
        return result
    
    async def _get_model_info(self, config: VLLMEndpointConfig) -> Optional[Dict[str, Any]]:
        """Get model information from the endpoint."""
        try:
            headers = {"Content-Type": "application/json"}
            if config.api_key:
                if 'runpod' in config.base_url.lower():
                    headers["Authorization"] = f"Bearer {config.api_key}"
                else:
                    headers["X-API-Key"] = config.api_key
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                url = f"{config.base_url.rstrip('/')}/v1/models"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                        
        except Exception as e:
            logger.debug(f"Failed to get model info: {e}")
        
        return None
    
    def _validate_model_compatibility(
        self, 
        config: VLLMEndpointConfig, 
        model_info: Optional[Dict[str, Any]], 
        result: Dict[str, Any]
    ):
        """Validate model compatibility with configuration."""
        if not model_info:
            result["warnings"].append("Could not retrieve model information")
            return
        
        # Check if configured model is available
        available_models = []
        if "data" in model_info:
            available_models = [model.get("id", "") for model in model_info["data"]]
        
        if available_models:
            if config.model not in available_models:
                result["warnings"].append(
                    f"Configured model '{config.model}' not found in available models: {available_models}"
                )
            
            # Check for Qwen model compatibility
            if "qwen" in config.model.lower():
                qwen_models = [m for m in available_models if "qwen" in m.lower()]
                if not qwen_models:
                    result["warnings"].append("No Qwen models found on endpoint")
    
    def _generate_recommendations(
        self, 
        config: VLLMEndpointConfig, 
        endpoint_type: EndpointType, 
        result: Dict[str, Any]
    ):
        """Generate configuration recommendations."""
        recommendations = []
        
        # Timeout recommendations
        if endpoint_type == EndpointType.LOCAL and config.timeout > 60:
            recommendations.append("Consider reducing timeout for local endpoints (30-60s)")
        elif endpoint_type in [EndpointType.RUNPOD, EndpointType.EXTERNAL] and config.timeout < 60:
            recommendations.append("Consider increasing timeout for external endpoints (60-120s)")
        
        # Connection pool recommendations
        if config.connection_pool_size < 5:
            recommendations.append("Consider increasing connection pool size for better concurrency")
        
        # Retry recommendations
        if endpoint_type == EndpointType.LOCAL and config.max_retries > 3:
            recommendations.append("Local endpoints typically need fewer retries (1-3)")
        elif endpoint_type in [EndpointType.RUNPOD, EndpointType.EXTERNAL] and config.max_retries < 3:
            recommendations.append("External endpoints benefit from more retries (3-5)")
        
        # Security recommendations
        if endpoint_type != EndpointType.LOCAL and not config.base_url.startswith('https://'):
            recommendations.append("Use HTTPS for external endpoints")
        
        # Circuit breaker recommendations
        if config.circuit_breaker_threshold < 3:
            recommendations.append("Consider higher circuit breaker threshold (3-5) to avoid false positives")
        
        result["recommendations"] = recommendations


class ConfigurationService:
    """
    Service for managing and validating vLLM endpoint configurations.
    
    Provides configuration validation, testing, and management capabilities
    for different types of vLLM endpoints.
    """
    
    def __init__(self):
        """Initialize the configuration service."""
        self.validator = EndpointValidator()
        self.settings = get_settings()
    
    async def validate_all_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate all configured endpoints.
        
        Returns:
            Validation results for all endpoints
        """
        results = {}
        
        # Validate chat endpoint
        chat_config = self.settings.vllm.get_chat_config()
        results["chat"] = await self.validator.validate_endpoint_config(chat_config)
        
        # Validate embedding endpoint if different
        embedding_config = self.settings.vllm.get_embedding_config()
        if embedding_config.base_url != chat_config.base_url:
            results["embedding"] = await self.validator.validate_endpoint_config(embedding_config)
        
        return results
    
    async def test_endpoint_connectivity(self, base_url: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Test connectivity to a specific endpoint.
        
        Args:
            base_url: Endpoint URL to test
            api_key: Optional API key
            
        Returns:
            Connectivity test results
        """
        # Create temporary config for testing
        test_config = VLLMEndpointConfig(
            base_url=base_url,
            model="test",  # Model doesn't matter for connectivity test
            api_key=api_key
        )
        
        return await self.validator._test_connectivity(test_config)
    
    def get_recommended_config(self, endpoint_type: EndpointType) -> Dict[str, Any]:
        """
        Get recommended configuration for an endpoint type.
        
        Args:
            endpoint_type: Type of endpoint
            
        Returns:
            Recommended configuration parameters
        """
        base_config = {
            "timeout": 60,
            "max_retries": 3,
            "connection_pool_size": 10,
            "health_check_interval": 30,
            "circuit_breaker_threshold": 5,
            "circuit_breaker_timeout": 60
        }
        
        if endpoint_type == EndpointType.LOCAL:
            base_config.update({
                "timeout": 30,
                "max_retries": 2,
                "health_check_interval": 15
            })
        elif endpoint_type == EndpointType.RUNPOD:
            base_config.update({
                "timeout": 90,
                "max_retries": 4,
                "health_check_interval": 45
            })
        elif endpoint_type in [EndpointType.AWS, EndpointType.GCP, EndpointType.AZURE]:
            base_config.update({
                "timeout": 120,
                "max_retries": 5,
                "health_check_interval": 60
            })
        
        return base_config
    
    def generate_runpod_config(
        self, 
        pod_id: str, 
        api_key: str, 
        model: str = "Qwen/Qwen3-8B-AWQ",
        port: int = 8000
    ) -> VLLMEndpointConfig:
        """
        Generate Runpod endpoint configuration.
        
        Args:
            pod_id: Runpod instance ID
            api_key: Runpod API key
            model: Model name
            port: Port number
            
        Returns:
            Configured VLLMEndpointConfig
        """
        base_url = f"https://{pod_id}-{port}.proxy.runpod.net"
        recommended = self.get_recommended_config(EndpointType.RUNPOD)
        
        return VLLMEndpointConfig(
            base_url=base_url,
            model=model,
            api_key=api_key,
            **recommended
        )


# Global configuration service instance
_config_service: Optional[ConfigurationService] = None


def get_config_service() -> ConfigurationService:
    """
    Get the global configuration service instance.
    
    Returns:
        ConfigurationService instance
    """
    global _config_service
    
    if _config_service is None:
        _config_service = ConfigurationService()
    
    return _config_service