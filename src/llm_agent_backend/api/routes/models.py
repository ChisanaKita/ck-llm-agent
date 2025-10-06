"""
Models endpoint for OpenAI API compatibility.

This module provides the /v1/models endpoint to list available models,
maintaining compatibility with OpenAI API clients.
"""

import logging
from typing import List

from fastapi import APIRouter, Request

from ...config import get_settings
from ...models.api import ModelsResponse, ModelInfo


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/models")
async def list_models(request: Request) -> ModelsResponse:
    """
    List available models.
    
    Returns information about available models in OpenAI-compatible format.
    This endpoint helps maintain compatibility with OpenAI API clients.
    
    Args:
        request: FastAPI request object
        
    Returns:
        ModelsResponse: List of available models
    """
    settings = get_settings()
    
    # Define available models based on configuration
    models: List[ModelInfo] = []
    
    # Add chat model
    chat_model = settings.vllm.chat.model
    models.append(
        ModelInfo(
            id=chat_model,
            owned_by="llm-agent-backend",
            permission=["read"]
        )
    )
    
    # Add embedding model if different from chat model
    embedding_model = settings.vllm.embedding.model
    if embedding_model != chat_model:
        models.append(
            ModelInfo(
                id=embedding_model,
                owned_by="llm-agent-backend",
                permission=["read"]
            )
        )
    
    # Add common model aliases for compatibility
    model_aliases = [
        "gpt-3.5-turbo",  # Common alias for compatibility
        "gpt-4",          # Common alias for compatibility
        "qwen3-8b",       # Simplified alias
        "qwen3",          # Even simpler alias
    ]
    
    for alias in model_aliases:
        if alias not in [m.id for m in models]:
            models.append(
                ModelInfo(
                    id=alias,
                    owned_by="llm-agent-backend",
                    permission=["read"],
                    root=chat_model  # Point to actual model
                )
            )
    
    logger.debug(f"Returning {len(models)} available models")
    
    return ModelsResponse(data=models)


@router.get("/models/{model_id}")
async def get_model(model_id: str, request: Request) -> ModelInfo:
    """
    Get information about a specific model.
    
    Args:
        model_id: Model identifier
        request: FastAPI request object
        
    Returns:
        ModelInfo: Model information
        
    Raises:
        HTTPException: If model not found
    """
    settings = get_settings()
    
    # Check if it's one of our configured models
    chat_model = settings.vllm.chat.model
    embedding_model = settings.vllm.embedding.model
    
    if model_id in [chat_model, embedding_model]:
        return ModelInfo(
            id=model_id,
            owned_by="llm-agent-backend",
            permission=["read"]
        )
    
    # Check if it's an alias
    model_aliases = {
        "gpt-3.5-turbo": chat_model,
        "gpt-4": chat_model,
        "qwen3-8b": chat_model,
        "qwen3": chat_model,
    }
    
    if model_id in model_aliases:
        return ModelInfo(
            id=model_id,
            owned_by="llm-agent-backend",
            permission=["read"],
            root=model_aliases[model_id]
        )
    
    # Model not found
    from fastapi import HTTPException
    from ...models.api import ErrorResponse, ErrorDetail, ErrorType
    
    raise HTTPException(
        status_code=404,
        detail=ErrorResponse(
            error=ErrorDetail(
                type=ErrorType.NOT_FOUND,
                code="model_not_found",
                message=f"Model '{model_id}' not found",
                param="model_id"
            )
        ).model_dump()
    )