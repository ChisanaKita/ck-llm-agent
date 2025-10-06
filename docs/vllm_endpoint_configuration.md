# vLLM Endpoint Configuration and Health Monitoring

This document describes the enhanced vLLM endpoint configuration system that supports local, Runpod, and other external service configurations with comprehensive health monitoring and circuit breaker patterns.

> **Quick Start**: For configuration examples and testing utilities, see [vLLM Endpoint Examples](vllm_endpoint_examples.md).

## Overview

The enhanced QwenVLLM handler provides:

- **Multi-endpoint Support**: Local, Runpod, Hugging Face, and generic external endpoints
- **Health Monitoring**: Continuous endpoint health checks with detailed metrics
- **Circuit Breaker Pattern**: Automatic failure detection and recovery
- **Connection Pooling**: Efficient HTTP connection management with authentication
- **Dynamic Configuration**: Runtime endpoint reconfiguration support

## Configuration

### Environment Variables

```bash
# Primary chat endpoint
VLLM__CHAT__BASE_URL=http://localhost:8000
VLLM__CHAT__MODEL=Qwen/Qwen3-8B-AWQ
VLLM__CHAT__API_KEY=your-api-key  # Optional

# Primary embedding endpoint  
VLLM__EMBEDDING__BASE_URL=http://localhost:8001
VLLM__EMBEDDING__MODEL=Qwen/Qwen3-Embedding-0.6B
VLLM__EMBEDDING__API_KEY=your-api-key  # Optional

# Runpod configuration (overrides primary endpoints if set)
VLLM__RUNPOD_CHAT_ENDPOINT=https://pod-id-8000.proxy.runpod.net
VLLM__RUNPOD_EMBEDDING_ENDPOINT=https://pod-id-8001.proxy.runpod.net
VLLM__RUNPOD_API_KEY=your-runpod-api-key

# Health monitoring settings
VLLM__ENABLE_HEALTH_MONITORING=true
VLLM__ENABLE_CIRCUIT_BREAKER=true
```

### Programmatic Configuration

#### Local vLLM Server

```python
from llm_agent_backend.handlers import QwenVLLM

# Create handler for local vLLM server
handler = QwenVLLM.create_for_local(
    base_url="http://localhost:8000",
    model="Qwen/Qwen3-8B-AWQ",
    thinking_mode=True,
    temperature=0.6
)
```

#### Runpod Configuration

```python
# Create handler for Runpod instance
handler = QwenVLLM.create_for_runpod(
    pod_id="your-pod-id",
    api_key="your-runpod-api-key",
    model="Qwen/Qwen3-8B-AWQ",
    region="us-east-1"
)

# Or configure existing handler for Runpod
handler.configure_runpod_endpoint(
    pod_id="your-pod-id",
    api_key="your-runpod-api-key",
    region="us-east-1"
)
```

#### External Endpoints

```python
# Create handler for generic external endpoint
handler = QwenVLLM.create_for_external(
    base_url="https://api.your-provider.com/v1",
    model="your-model",
    api_key="your-api-key"
)

# Configure Hugging Face Inference Endpoints
handler.configure_huggingface_endpoint(
    model_id="Qwen/Qwen3-8B-AWQ",
    api_key="your-hf-token"
)
```

## Endpoint Manager

The `VLLMEndpointManager` provides centralized management of multiple endpoints:

```python
from llm_agent_backend.handlers import get_endpoint_manager, VLLMEndpointConfig

manager = get_endpoint_manager()

# Add endpoints
local_config = VLLMEndpointConfig(
    base_url="http://localhost:8000",
    model="Qwen/Qwen3-8B-AWQ"
)

runpod_config = VLLMEndpointConfig(
    base_url="https://pod-id-8000.proxy.runpod.net",
    model="Qwen/Qwen3-8B-AWQ",
    api_key="runpod-key"
)

manager.add_endpoint("local", local_config)
manager.add_endpoint("runpod", runpod_config)

# Get endpoints
chat_handler = manager.get_primary_chat_endpoint()
embedding_handler = manager.get_primary_embedding_endpoint()

# Health monitoring
health_status = await manager.check_all_health()
healthy_endpoints = await manager.get_healthy_endpoints()
```

## Health Monitoring

### Health Check Results

```python
# Perform comprehensive health check
health = await handler.check_health()

print(health)
# {
#     'endpoint_health': {
#         'status': 'healthy',
#         'response_time_ms': 45.2,
#         'endpoint': 'https://api.example.com',
#         'model_count': 1,
#         'timestamp': 1696598400.0
#     },
#     'circuit_breaker': {
#         'state': 'closed',
#         'failure_count': 0,
#         'is_available': True
#     },
#     'endpoint_info': {
#         'base_url': 'https://api.example.com',
#         'model': 'Qwen/Qwen3-8B-AWQ',
#         'endpoint_type': 'external',
#         'thinking_mode': True,
#         'api_key_configured': True
#     },
#     'overall_status': 'healthy'
# }
```

### Monitoring Metrics

```python
# Get handler statistics
stats = handler.get_stats()

print(stats)
# {
#     'request_count': 150,
#     'error_count': 3,
#     'error_rate': 0.02,
#     'total_tokens': 45000,
#     'model': 'Qwen/Qwen3-8B-AWQ',
#     'endpoint_type': 'runpod',
#     'health_monitor': {
#         'check_count': 20,
#         'success_rate': 0.95
#     },
#     'circuit_breaker': {
#         'state': 'closed',
#         'failure_count': 0
#     }
# }
```

## Circuit Breaker Pattern

The circuit breaker automatically handles endpoint failures:

### States

- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Endpoint failing, requests are rejected immediately  
- **HALF_OPEN**: Testing recovery, limited requests allowed

### Configuration

```python
from llm_agent_backend.handlers import CircuitBreaker

# Custom circuit breaker configuration
circuit_breaker = CircuitBreaker(
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=60.0,    # Wait 60s before testing recovery
    success_threshold=3       # Close after 3 successful requests
)
```

### Usage

```python
# Circuit breaker protects requests automatically
try:
    response = await handler.call(messages)
except Exception as e:
    if "Circuit breaker is OPEN" in str(e):
        # Handle circuit breaker rejection
        print("Endpoint temporarily unavailable")
    else:
        # Handle other errors
        print(f"Request failed: {e}")
```

## Connection Pooling

### Configuration

```python
from llm_agent_backend.handlers import ConnectionPool

# Custom connection pool
pool = ConnectionPool(
    base_url="https://api.example.com",
    pool_size=20,           # Max concurrent connections
    timeout=60,             # Request timeout in seconds
    api_key="your-key",     # Authentication
    headers={               # Custom headers
        "Custom-Header": "value"
    }
)
```

### Authentication

The connection pool supports multiple authentication methods:

- **API Key**: `X-API-Key` header (default)
- **Bearer Token**: `Authorization: Bearer <token>` (Runpod)
- **Custom Headers**: Provider-specific authentication

## Error Handling

### Retry Logic

```python
from llm_agent_backend.handlers import RetryConfig

# Custom retry configuration
retry_config = RetryConfig(
    max_retries=5,           # Maximum retry attempts
    base_delay=1.0,          # Initial delay in seconds
    max_delay=60.0,          # Maximum delay in seconds
    exponential_base=2.0,    # Exponential backoff multiplier
    jitter=True              # Add random jitter to delays
)

handler = QwenVLLM(
    base_url="https://api.example.com",
    model="Qwen/Qwen3-8B-AWQ",
    retry_config=retry_config
)
```

### Error Categories

1. **Connection Errors**: Network timeouts, DNS failures
2. **Authentication Errors**: Invalid API keys, expired tokens
3. **Rate Limiting**: HTTP 429 responses
4. **Server Errors**: HTTP 5xx responses
5. **Circuit Breaker**: Endpoint temporarily unavailable

## Best Practices

### Production Deployment

1. **Use External Endpoints**: Configure Runpod or cloud providers for production
2. **Enable Health Monitoring**: Monitor endpoint health continuously
3. **Configure Circuit Breakers**: Prevent cascading failures
4. **Set Appropriate Timeouts**: Balance responsiveness and reliability
5. **Monitor Metrics**: Track request counts, error rates, and response times

### Development Setup

1. **Local vLLM Server**: Use local endpoints for development
2. **Mock Endpoints**: Use test endpoints for unit testing
3. **Health Check Intervals**: Use shorter intervals for faster feedback
4. **Debug Logging**: Enable detailed logging for troubleshooting

### Security Considerations

1. **API Key Management**: Store keys in environment variables or secret managers
2. **TLS Encryption**: Use HTTPS for all external endpoints
3. **Network Security**: Restrict access to vLLM endpoints
4. **Audit Logging**: Log authentication and access events

## Troubleshooting

### Common Issues

#### Connection Failures

```python
# Check endpoint health
health = await handler.check_health()
if health['overall_status'] != 'healthy':
    print(f"Endpoint issue: {health['endpoint_health'].get('error')}")
```

#### Authentication Problems

```python
# Verify API key configuration
config = handler.get_endpoint_config()
if not config['api_key_configured']:
    print("API key not configured")
```

#### Circuit Breaker Issues

```python
# Check circuit breaker state
stats = handler.get_stats()
circuit_state = stats['circuit_breaker']['state']
if circuit_state == 'open':
    print("Circuit breaker is open - endpoint unavailable")
```

### Debugging

Enable debug logging to see detailed request/response information:

```python
import logging

logging.getLogger('llm_agent_backend.handlers').setLevel(logging.DEBUG)
```

## Migration Guide

### From Legacy Configuration

Old configuration:
```python
handler = QwenVLLM(
    base_url="http://localhost:8000",
    model="Qwen/Qwen3-8B-AWQ"
)
```

New configuration:
```python
# Explicit factory method
handler = QwenVLLM.create_for_local(
    base_url="http://localhost:8000",
    model="Qwen/Qwen3-8B-AWQ"
)

# Or use endpoint manager
manager = get_endpoint_manager()
handler = manager.get_primary_chat_endpoint()
```

### Environment Variables

Legacy variables are still supported but new nested format is recommended:

```bash
# Legacy (still works)
VLLM_CHAT_BASE_URL=http://localhost:8000
VLLM_CHAT_MODEL=Qwen/Qwen3-8B-AWQ

# New format (recommended)
VLLM__CHAT__BASE_URL=http://localhost:8000
VLLM__CHAT__MODEL=Qwen/Qwen3-8B-AWQ
```