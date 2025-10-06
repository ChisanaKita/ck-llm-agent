"""
LLM Agent Backend - Production-grade asynchronous service for intelligent chat interactions.

This package provides a FastAPI-based backend that orchestrates CrewAI agents powered by
vLLM-served Qwen3 models with MCP tool integration and semantic search capabilities.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]