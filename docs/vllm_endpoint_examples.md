# vLLM Endpoint Configuration Examples

This document provides examples of how to configure different types of vLLM endpoints for the LLM Agent Backend.

## Environment Variables

The system supports configuration through environment variables with the `VLLM__` prefix:

### Basic Configuration

```bash
# Chat endpoint (primary)
VLLM__CHAT__BASE_URL=http://localhost:8000
VLLM__CHAT__MODEL=Qwen/Qwen3-8B-AWQ
VLLM__CHAT__TIMEOUT=60
VLLM__CHAT__MAX_RETRIES=3
VLLM__CHAT__CONNECTION_POOL_SIZE=10

# Embedding endpoint (if different from chat)
VLLM__EMBEDDING__BASE_URL=http://localhost:8001
VLLM__EMBEDDING__MODEL=Qwen/Qwen3-Embedding-0.6B
VLLM__EMBEDDING__TIMEOUT=30
VLLM__EMBEDDING__MAX_RETRIES=2
```

### Health Monitoring and Circuit Breaker

```bash
# Health monitoring
VLLM__ENABLE_HEALTH_MONITORING=true
VLLM__ENABLE_CIRCUIT_BREAKER=true

# Circuit breaker settings
VLLM__CHAT__CIRCUIT_BREAKER_THRESHOLD=5
VLLM__CHAT__CIRCUIT_BREAKER_TIMEOUT=60
VLLM__CHAT__HEALTH_CHECK_INTERVAL=30
```

## Local vLLM Server

### Standard Local Setup

```bash
# Local vLLM server on default port
VLLM__CHAT__BASE_URL=http://localhost:8000
VLLM__CHAT__MODEL=Qwen/Qwen3-8B-AWQ
VLLM__CHAT__TIMEOUT=30
VLLM__CHAT__MAX_RETRIES=2
```

### Local with Custom Port

```bash
# Local vLLM server on custom port
VLLM__CHAT__BASE_URL=http://localhost:8080
VLLM__CHAT__MODEL=Qwen/Qwen3-8B-AWQ
```

### Separate Chat and Embedding Servers

```bash
# Chat server
VLLM__CHAT__BASE_URL=http://localhost:8000
VLLM__CHAT__MODEL=Qwen/Qwen3-8B-AWQ

# Embedding server
VLLM__EMBEDDING__BASE_URL=http://localhost:8001
VLLM__EMBEDDING__MODEL=Qwen/Qwen3-Embedding-0.6B
```

## Runpod Configuration

### Basic Runpod Setup

```bash
# Runpod endpoint
VLLM__RUNPOD_CHAT_ENDPOINT=https://your-pod-id-8000.proxy.runpod.net
VLLM__RUNPOD_API_KEY=your-runpod-api-key

# Model configuration
VLLM__CHAT__MODEL=Qwen/Qwen3-8B-AWQ
VLLM__CHAT__TIMEOUT=90
VLLM__CHAT__MAX_RETRIES=4
```

### Runpod with Separate Embedding Server

```bash
# Chat endpoint
VLLM__RUNPOD_CHAT_ENDPOINT=https://chat-pod-id-8000.proxy.runpod.net
VLLM__RUNPOD_EMBEDDING_ENDPOINT=https://embed-pod-id-8001.proxy.runpod.net
VLLM__RUNPOD_API_KEY=your-runpod-api-key

# Models
VLLM__CHAT__MODEL=Qwen/Qwen3-8B-AWQ
VLLM__EMBEDDING__MODEL=Qwen/Qwen3-Embedding-0.6B
```

### Runpod with Custom Configuration

```bash
# Runpod with optimized settings
VLLM__RUNPOD_CHAT_ENDPOINT=https://your-pod-id-8000.proxy.runpod.net
VLLM__RUNPOD_API_KEY=your-runpod-api-key

# Optimized for Runpod latency
VLLM__CHAT__TIMEOUT=120
VLLM__CHAT__MAX_RETRIES=5
VLLM__CHAT__CONNECTION_POOL_SIZE=15
VLLM__CHAT__HEALTH_CHECK_INTERVAL=45
VLLM__CHAT__CIRCUIT_BREAKER_THRESHOLD=3
```

## Cloud Provider Configurations

### AWS (SageMaker or EC2)

```bash
# AWS endpoint
VLLM__CHAT__BASE_URL=https://your-endpoint.us-east-1.amazonaws.com
VLLM__CHAT__API_KEY=your-aws-api-key
VLLM__CHAT__MODEL=Qwen/Qwen3-8B-AWQ
VLLM__CHAT__TIMEOUT=120
VLLM__CHAT__MAX_RETRIES=5
```

### Google Cloud Platform

```bash
# GCP endpoint
VLLM__CHAT__BASE_URL=https://your-endpoint.googleapis.com
VLLM__CHAT__API_KEY=your-gcp-api-key
VLLM__CHAT__MODEL=Qwen/Qwen3-8B-AWQ
```

### Azure

```bash
# Azure endpoint
VLLM__CHAT__BASE_URL=https://your-endpoint.azure.com
VLLM__CHAT__API_KEY=your-azure-api-key
VLLM__CHAT__MODEL=Qwen/Qwen3-8B-AWQ
```

### Hugging Face Inference Endpoints

```bash
# Hugging Face endpoint
VLLM__CHAT__BASE_URL=https://your-endpoint.hf.co
VLLM__CHAT__API_KEY=your-hf-token
VLLM__CHAT__MODEL=Qwen/Qwen3-8B-AWQ
```

## Advanced Configuration

### High Availability Setup

```bash
# Primary endpoint
VLLM__CHAT__BASE_URL=https://primary-endpoint.com
VLLM__CHAT__API_KEY=primary-api-key

# Circuit breaker for quick failover
VLLM__CHAT__CIRCUIT_BREAKER_THRESHOLD=3
VLLM__CHAT__CIRCUIT_BREAKER_TIMEOUT=30
VLLM__CHAT__HEALTH_CHECK_INTERVAL=15
```

### Performance Optimized

```bash
# High-performance configuration
VLLM__CHAT__BASE_URL=http://localhost:8000
VLLM__CHAT__CONNECTION_POOL_SIZE=20
VLLM__CHAT__TIMEOUT=45
VLLM__CHAT__MAX_RETRIES=2
VLLM__CHAT__HEALTH_CHECK_INTERVAL=10

# Disable circuit breaker for local high-performance setup
VLLM__ENABLE_CIRCUIT_BREAKER=false
```

### Development/Testing

```bash
# Development configuration
VLLM__CHAT__BASE_URL=http://localhost:8000
VLLM__CHAT__MODEL=Qwen/Qwen3-8B-AWQ
VLLM__CHAT__TIMEOUT=30
VLLM__CHAT__MAX_RETRIES=1
VLLM__CHAT__CONNECTION_POOL_SIZE=5

# Frequent health checks for development
VLLM__CHAT__HEALTH_CHECK_INTERVAL=10
VLLM__ENABLE_HEALTH_MONITORING=true
```

## Testing Configuration

You can test your endpoint configuration using the built-in testing utility:

### Test a Specific Endpoint

```bash
# Test local endpoint
python -m llm_agent_backend.utils.endpoint_tester test http://localhost:8000

# Test Runpod endpoint
python -m llm_agent_backend.utils.endpoint_tester test \
  https://your-pod-id-8000.proxy.runpod.net \
  --api-key your-runpod-api-key

# Test with custom model
python -m llm_agent_backend.utils.endpoint_tester test \
  http://localhost:8000 \
  --model Qwen/Qwen3-8B-AWQ \
  --timeout 60
```

### Test Chat Completion

```bash
# Test chat functionality
python -m llm_agent_backend.utils.endpoint_tester chat \
  http://localhost:8000 \
  --message "Hello, how are you?"

# Test with Runpod
python -m llm_agent_backend.utils.endpoint_tester chat \
  https://your-pod-id-8000.proxy.runpod.net \
  --api-key your-runpod-api-key \
  --message "Explain quantum computing"
```

### Test All Configured Endpoints

```bash
# Test all endpoints from configuration
python -m llm_agent_backend.utils.endpoint_tester test-all
```

### Benchmark Performance

```bash
# Benchmark endpoint performance
python -m llm_agent_backend.utils.endpoint_tester benchmark \
  http://localhost:8000 \
  --requests 20 \
  --concurrency 5
```

## Configuration Validation

The system automatically validates configurations on startup and provides detailed error messages for common issues:

- **Invalid URLs**: Checks for proper HTTP/HTTPS format
- **Missing API keys**: Warns when external endpoints lack authentication
- **Timeout settings**: Validates reasonable timeout values
- **Connection pools**: Ensures proper pool sizing
- **Model compatibility**: Verifies model availability on endpoints

## Troubleshooting

### Common Issues

1. **Connection Timeout**
   - Increase `VLLM__CHAT__TIMEOUT` for external endpoints
   - Check network connectivity to the endpoint

2. **Authentication Errors**
   - Verify `VLLM__CHAT__API_KEY` or `VLLM__RUNPOD_API_KEY`
   - Check API key format for the specific provider

3. **Circuit Breaker Triggering**
   - Increase `VLLM__CHAT__CIRCUIT_BREAKER_THRESHOLD`
   - Check endpoint health and stability

4. **Model Not Found**
   - Verify the model name matches what's available on the endpoint
   - Use the test utility to list available models

### Health Check Endpoints

The system provides health check endpoints for monitoring:

- `GET /health` - Basic health check
- `GET /health/deep` - Comprehensive health check including all endpoints
- `GET /metrics` - Prometheus metrics

### Logging

Enable debug logging to troubleshoot configuration issues:

```bash
LOG_LEVEL=DEBUG
```

This will provide detailed information about endpoint connections, health checks, and circuit breaker state changes.