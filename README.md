# LLM Agent Backend

A production-grade asynchronous service that provides intelligent chat interactions through CrewAI-orchestrated agents powered by vLLM-served Qwen3 models with MCP tool integration and semantic search capabilities.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI chat completions API
- **CrewAI Agent Orchestration**: Intelligent agent management and task processing
- **vLLM Integration**: High-performance inference with Qwen3-8B-AWQ model
- **MCP Tool Integration**: Dynamic tool discovery and execution via Model Context Protocol
- **Semantic Search**: ChromaDB-powered vector search for intelligent tool selection
- **Thinking Mode**: Support for Qwen3's reasoning capabilities with transparent thought processes
- **Production Ready**: Comprehensive monitoring, logging, and error handling
- **Containerized**: Docker support with multi-architecture builds (ARM64/AMD64)

## Architecture

The system follows a modular architecture with clear separation of concerns:

- **FastAPI**: HTTP API layer with authentication and rate limiting
- **CrewAI**: Agent orchestration and task management
- **vLLM**: High-performance model inference serving
- **ChromaDB**: Vector database for semantic tool selection
- **MCP**: Extensible tool integration protocol

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- NVIDIA GPU with CUDA support (for vLLM inference)

### Local Development

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd llm-agent-backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start vLLM servers** (requires GPU):
   ```bash
   # Chat server (Qwen3-8B-AWQ)
   vllm serve Qwen/Qwen3-8B-AWQ \
     --enable-reasoning \
     --reasoning-parser qwen3 \
     --enable-auto-tool-choice \
     --tool-call-parser hermes \
     --max-model-len 8192 \
     --gpu-memory-utilization 0.9 \
     --quantization awq \
     --port 8000

   # Embedding server (Qwen3-Embedding-0.6B) - in another terminal
   vllm serve Qwen/Qwen3-Embedding-0.6B \
     --max-model-len 2048 \
     --gpu-memory-utilization 0.45 \
     --port 8001
   ```

5. **Run the application**:
   ```bash
   python -m uvicorn llm_agent_backend.api.app:app --reload --port 8080
   ```

### Docker Deployment

1. **Using Docker Compose** (recommended):
   ```bash
   docker-compose up -d
   ```

2. **Build and run manually**:
   ```bash
   docker build -t llm-agent-backend .
   docker run -p 8080:8080 llm-agent-backend
   ```

## Configuration

The application uses environment variables for configuration. See `.env.example` for all available options.

### Key Configuration Sections

- **vLLM Endpoints**: Configure chat and embedding server URLs
- **ChromaDB**: Vector database settings for tool embeddings
- **Authentication**: Optional JWT and API key authentication
- **Rate Limiting**: Request throttling configuration
- **MCP Integration**: Tool discovery and selection settings

### External vLLM Deployment

For production deployments, you can use external vLLM servers (e.g., Runpod):

```bash
# Configure external endpoints in .env
VLLM__CHAT_BASE_URL=https://your-runpod-chat-endpoint.com
VLLM__EMBEDDING_BASE_URL=https://your-runpod-embedding-endpoint.com
```

## API Usage

### Chat Completions

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-AWQ",
    "messages": [
      {"role": "user", "content": "Hello, how can you help me?"}
    ],
    "thinking_mode": true,
    "temperature": 0.7
  }'
```

### Health Check

```bash
curl http://localhost:8080/health
```

### Metrics

```bash
curl http://localhost:8080/metrics
```

## Development

### Project Structure

```
src/llm_agent_backend/
├── __init__.py          # Package initialization
├── config.py            # Configuration management
├── api/                 # FastAPI application and routes
├── models/              # Pydantic data models
├── services/            # Business logic services
├── handlers/            # Request handlers and middleware
└── utils/               # Utility functions and helpers
```

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src/llm_agent_backend --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Monitoring and Observability

The application provides comprehensive monitoring capabilities:

- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Prometheus Metrics**: Request metrics, performance counters, and health indicators
- **Health Checks**: Deep dependency health monitoring
- **Performance Tracking**: Request timing and resource usage metrics

## Security

- **Authentication**: JWT token and API key support
- **Rate Limiting**: Configurable request throttling
- **Input Validation**: Comprehensive request sanitization
- **Secure Logging**: Sensitive data exclusion from logs

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or contributions, please:

1. Check the [Issues](https://github.com/yourusername/llm-agent-backend/issues) page
2. Review the documentation in the `docs/` directory
3. Join our community discussions

## Roadmap

- [ ] Enhanced tool selection algorithms
- [ ] Multi-model support
- [ ] Advanced caching strategies
- [ ] Kubernetes deployment manifests
- [ ] Performance optimization guides
- [ ] Extended MCP protocol support