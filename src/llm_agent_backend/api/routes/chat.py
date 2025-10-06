"""
Chat completions endpoint implementation.

This module provides the OpenAI-compatible /v1/chat/completions endpoint
with support for Qwen3 thinking mode and MCP tool integration.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from ...config import get_settings
from ...models.api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ErrorResponse,
    ErrorDetail,
    ErrorType,
    Choice,
    ChatMessage,
    MessageRole,
    FinishReason,
    TokenUsage,
)
from ...services.agent_manager import AgentManager


logger = logging.getLogger(__name__)
router = APIRouter()


def get_agent_manager(request: Request) -> AgentManager:
    """Dependency to get AgentManager from app state."""
    if not hasattr(request.app.state, 'agent_manager'):
        raise HTTPException(
            status_code=503,
            detail="Agent manager not initialized"
        )
    return request.app.state.agent_manager


@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    http_request: Request,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> ChatCompletionResponse:
    """
    Create a chat completion using the CrewAI agent system.
    
    This endpoint is compatible with the OpenAI Chat Completions API while
    supporting additional features like Qwen3 thinking mode and MCP tools.
    
    Args:
        request: Chat completion request parameters
        http_request: FastAPI request object for accessing state
        agent_manager: AgentManager dependency
        
    Returns:
        ChatCompletionResponse: Generated completion with thinking content
        
    Raises:
        HTTPException: For various error conditions
    """
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4()))
    start_time = time.time()
    
    try:
        logger.info(f"Processing chat completion request {request_id}")
        
        # Validate request
        if not request.messages:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        type=ErrorType.INVALID_REQUEST,
                        code="invalid_messages",
                        message="Messages list cannot be empty",
                        param="messages"
                    ),
                    request_id=request_id
                ).model_dump()
            )
        
        # Check if streaming is requested (not yet implemented)
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        type=ErrorType.INVALID_REQUEST,
                        code="streaming_not_supported",
                        message="Streaming responses are not yet implemented",
                        param="stream"
                    ),
                    request_id=request_id
                ).model_dump()
            )
        
        # Process the chat completion through AgentManager
        try:
            result = await agent_manager.process_chat_completion(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking_mode=request.thinking_mode,
                user=request.user,
                request_id=request_id
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out")
            raise HTTPException(
                status_code=408,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        type=ErrorType.SERVER_ERROR,
                        code="request_timeout",
                        message="Request processing timed out",
                    ),
                    request_id=request_id
                ).model_dump()
            )
        
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}", exc_info=True)
            
            # Check if it's a known error type
            if "rate_limit" in str(e).lower():
                status_code = 429
                error_type = ErrorType.RATE_LIMIT
                error_code = "rate_limit_exceeded"
            elif "authentication" in str(e).lower():
                status_code = 401
                error_type = ErrorType.AUTHENTICATION
                error_code = "authentication_failed"
            elif "model" in str(e).lower() and "not found" in str(e).lower():
                status_code = 404
                error_type = ErrorType.NOT_FOUND
                error_code = "model_not_found"
            else:
                status_code = 500
                error_type = ErrorType.SERVER_ERROR
                error_code = "processing_error"
            
            raise HTTPException(
                status_code=status_code,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        type=error_type,
                        code=error_code,
                        message=str(e),
                    ),
                    request_id=request_id
                ).model_dump()
            )
        
        # Build response
        processing_time = time.time() - start_time
        
        # Extract response components from agent result
        content = result.get("content", "")
        thinking_content = result.get("thinking_content")
        tool_calls = result.get("tool_calls", [])
        usage = result.get("usage", {})
        
        # Determine finish reason
        if tool_calls:
            finish_reason = FinishReason.TOOL_CALLS
        elif usage.get("completion_tokens", 0) >= (request.max_tokens or 4096):
            finish_reason = FinishReason.LENGTH
        else:
            finish_reason = FinishReason.STOP
        
        # Create response message
        response_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content if content else None,
            tool_calls=tool_calls if tool_calls else None
        )
        
        # Create choice
        choice = Choice(
            index=0,
            message=response_message,
            finish_reason=finish_reason
        )
        
        # Create token usage
        token_usage = TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0)
        )
        
        # Create final response
        response = ChatCompletionResponse(
            model=request.model,
            choices=[choice],
            usage=token_usage,
            thinking_content=thinking_content
        )
        
        logger.info(
            f"Completed request {request_id} in {processing_time:.2f}s, "
            f"tokens: {token_usage.total_tokens}"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        # Handle any unexpected errors
        logger.error(f"Unexpected error in chat completion {request_id}: {e}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(
                    type=ErrorType.SERVER_ERROR,
                    code="internal_error",
                    message="An internal error occurred while processing the request",
                ),
                request_id=request_id
            ).model_dump()
        )


@router.post("/chat/completions/stream")
async def create_chat_completion_stream(
    request: ChatCompletionRequest,
    http_request: Request,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> StreamingResponse:
    """
    Create a streaming chat completion (placeholder for future implementation).
    
    Args:
        request: Chat completion request parameters
        http_request: FastAPI request object
        agent_manager: AgentManager dependency
        
    Returns:
        StreamingResponse: Server-sent events stream
        
    Raises:
        HTTPException: Not implemented error
    """
    raise HTTPException(
        status_code=501,
        detail=ErrorResponse(
            error=ErrorDetail(
                type=ErrorType.SERVER_ERROR,
                code="not_implemented",
                message="Streaming chat completions are not yet implemented",
            ),
            request_id=getattr(http_request.state, 'request_id', None)
        ).model_dump()
    )