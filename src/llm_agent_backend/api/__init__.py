"""
API layer for the LLM Agent Backend.

This package contains FastAPI application setup, route definitions,
and API-specific components.
"""

from .app import app, create_app

__all__ = [
    "app",
    "create_app",
]