"""
Configuration management for the LLM Agent Backend.

This module provides centralized configuration using Pydantic Settings with
environment variable support and validation.
"""

import os
from functools import lru_cache
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class VLLMConfig(BaseModel):
    """Configuration for vLLM endpoints."""
    
    chat_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for vLLM chat inference server"
    )
    embedding_base_url: str = Field(
        default="http://localhost:8001", 
        description="Base URL for vLLM embedding server"
    )
    chat_model: str = Field(
        default="Qwen/Qwen3-8B-AWQ",
        description="Model name for chat completions"
    )
    embedding_model: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description="Model name for embeddings"
    )
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout: int = Field(default=60, ge=5, le=300)
    connection_pool_size: int = Field(default=10, ge=1, le=100)


class ChromaDBConfig(BaseModel):
    """Configuration for ChromaDB vector database."""
    
    persist_directory: str = Field(
        default="./data/chromadb",
        description="Directory for ChromaDB persistence"
    )
    collection_name: str = Field(
        default="tool_embeddings",
        description="Collection name for tool embeddings"
    )
    embedding_dimension: int = Field(
        default=1024,
        description="Dimension of embedding vectors"
    )
    similarity_metric: str = Field(
        default="cosine",
        description="Similarity metric for vector search"
    )


class AuthConfig(BaseModel):
    """Configuration for authentication and security."""
    
    secret_key: str = Field(
        description="Secret key for JWT token signing"
    )
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30, ge=1, le=1440)
    api_key_header: str = Field(default="X-API-Key")


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""
    
    requests_per_minute: int = Field(default=60, ge=1, le=10000)
    burst_size: int = Field(default=10, ge=1, le=100)
    enable_rate_limiting: bool = Field(default=True)


class MCPConfig(BaseModel):
    """Configuration for MCP (Model Context Protocol) integration."""
    
    config_path: str = Field(
        default="./.kiro/settings/mcp.json",
        description="Path to MCP configuration file"
    )
    auto_discover: bool = Field(
        default=True,
        description="Enable automatic MCP server discovery"
    )
    tool_selection_limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of tools to select per request"
    )


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application settings
    app_name: str = Field(default="LLM Agent Backend")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    environment: str = Field(default="development")
    
    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=32)
    
    # Logging settings
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    enable_access_logs: bool = Field(default=True)
    
    # Component configurations
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)
    auth: Optional[AuthConfig] = None
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    
    # Performance settings
    max_concurrent_requests: int = Field(default=100, ge=1, le=1000)
    request_timeout: int = Field(default=300, ge=30, le=3600)
    enable_caching: bool = Field(default=True)
    cache_ttl: int = Field(default=3600, ge=60, le=86400)
    
    # Health check settings
    health_check_interval: int = Field(default=30, ge=5, le=300)
    dependency_timeout: int = Field(default=5, ge=1, le=30)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False
        
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed_environments = ["development", "staging", "production"]
        if v not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level setting."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    This function uses lru_cache to ensure settings are loaded only once
    and reused throughout the application lifecycle.
    """
    return Settings()