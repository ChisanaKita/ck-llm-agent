"""
Authentication middleware for JWT token and API key validation.

This module provides middleware for validating JWT tokens and API keys
with support for different access levels and proper error handling.
"""

import logging
import time
from typing import Optional, Dict, Any, List

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from jose import JWTError, jwt

from ...config import get_settings
from ...models.api import ErrorResponse, ErrorDetail, ErrorType


logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for JWT token and API key authentication.
    
    Supports multiple authentication methods:
    - JWT tokens in Authorization header
    - API keys in X-API-Key header
    - Different access levels based on authentication method
    """
    
    def __init__(self, app, **kwargs):
        """Initialize authentication middleware."""
        super().__init__(app, **kwargs)
        self.settings = get_settings()
        
        # Paths that don't require authentication
        self.public_paths = {
            "/",
            "/health",
            "/health/live", 
            "/health/ready",
            "/docs",
            "/redoc",
            "/openapi.json",
        }
        
        # Paths that require authentication
        self.protected_paths = {
            "/v1/chat/completions",
            "/v1/models",
        }
        
        logger.info("Authentication middleware initialized")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request authentication.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
            
        Returns:
            Response: HTTP response
        """
        # Skip authentication for public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)
        
        # Skip authentication for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Perform authentication
        auth_result = await self._authenticate_request(request)
        
        if not auth_result["authenticated"]:
            return self._create_auth_error_response(
                auth_result["error_type"],
                auth_result["message"],
                getattr(request.state, 'request_id', None)
            )
        
        # Add authentication info to request state
        request.state.auth_user = auth_result.get("user")
        request.state.auth_method = auth_result.get("method")
        request.state.auth_scopes = auth_result.get("scopes", [])
        
        return await call_next(request)
    
    def _is_public_path(self, path: str) -> bool:
        """
        Check if a path is public (doesn't require authentication).
        
        Args:
            path: Request path
            
        Returns:
            bool: True if path is public
        """
        # Exact match
        if path in self.public_paths:
            return True
        
        # Check for path prefixes that should be public
        public_prefixes = ["/health"]
        for prefix in public_prefixes:
            if path.startswith(prefix):
                return True
        
        return False
    
    async def _authenticate_request(self, request: Request) -> Dict[str, Any]:
        """
        Authenticate the incoming request.
        
        Args:
            request: HTTP request to authenticate
            
        Returns:
            Dict: Authentication result with status and details
        """
        # Try JWT authentication first
        jwt_result = await self._authenticate_jwt(request)
        if jwt_result["authenticated"]:
            return jwt_result
        
        # Try API key authentication
        api_key_result = await self._authenticate_api_key(request)
        if api_key_result["authenticated"]:
            return api_key_result
        
        # No valid authentication found
        return {
            "authenticated": False,
            "error_type": ErrorType.AUTHENTICATION,
            "message": "Authentication required. Provide a valid JWT token or API key."
        }
    
    async def _authenticate_jwt(self, request: Request) -> Dict[str, Any]:
        """
        Authenticate using JWT token from Authorization header.
        
        Args:
            request: HTTP request
            
        Returns:
            Dict: JWT authentication result
        """
        if not self.settings.auth:
            return {"authenticated": False}
        
        # Get Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return {"authenticated": False}
        
        # Parse Bearer token
        if not auth_header.startswith("Bearer "):
            return {
                "authenticated": False,
                "error_type": ErrorType.AUTHENTICATION,
                "message": "Invalid Authorization header format. Use 'Bearer <token>'"
            }
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        
        try:
            # Decode and validate JWT token
            payload = jwt.decode(
                token,
                self.settings.auth.secret_key,
                algorithms=[self.settings.auth.algorithm]
            )
            
            # Check token expiration
            exp = payload.get("exp")
            if exp and exp < time.time():
                return {
                    "authenticated": False,
                    "error_type": ErrorType.AUTHENTICATION,
                    "message": "JWT token has expired"
                }
            
            # Extract user information
            user_id = payload.get("sub")
            scopes = payload.get("scopes", [])
            
            logger.debug(f"JWT authentication successful for user: {user_id}")
            
            return {
                "authenticated": True,
                "method": "jwt",
                "user": {
                    "id": user_id,
                    "email": payload.get("email"),
                    "name": payload.get("name"),
                },
                "scopes": scopes,
                "token_payload": payload
            }
            
        except JWTError as e:
            logger.warning(f"JWT validation failed: {e}")
            return {
                "authenticated": False,
                "error_type": ErrorType.AUTHENTICATION,
                "message": f"Invalid JWT token: {str(e)}"
            }
        
        except Exception as e:
            logger.error(f"JWT authentication error: {e}", exc_info=True)
            return {
                "authenticated": False,
                "error_type": ErrorType.AUTHENTICATION,
                "message": "JWT authentication failed"
            }
    
    async def _authenticate_api_key(self, request: Request) -> Dict[str, Any]:
        """
        Authenticate using API key from X-API-Key header.
        
        Args:
            request: HTTP request
            
        Returns:
            Dict: API key authentication result
        """
        if not self.settings.auth:
            return {"authenticated": False}
        
        # Get API key header
        api_key_header = self.settings.auth.api_key_header
        api_key = request.headers.get(api_key_header)
        
        if not api_key:
            return {"authenticated": False}
        
        # Validate API key (this would typically check against a database)
        # For now, we'll use environment variables for simplicity
        valid_api_keys = self._get_valid_api_keys()
        
        if api_key not in valid_api_keys:
            logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
            return {
                "authenticated": False,
                "error_type": ErrorType.AUTHENTICATION,
                "message": "Invalid API key"
            }
        
        # Get API key details
        api_key_info = valid_api_keys[api_key]
        
        logger.debug(f"API key authentication successful for: {api_key_info.get('name', 'unknown')}")
        
        return {
            "authenticated": True,
            "method": "api_key",
            "user": {
                "id": api_key_info.get("user_id", "api_key_user"),
                "name": api_key_info.get("name", "API Key User"),
                "type": "api_key"
            },
            "scopes": api_key_info.get("scopes", ["read", "write"]),
            "api_key_info": api_key_info
        }
    
    def _get_valid_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """
        Get valid API keys from configuration.
        
        In production, this would typically query a database.
        For now, we use environment variables.
        
        Returns:
            Dict: Mapping of API keys to their details
        """
        import os
        
        valid_keys = {}
        
        # Check for environment variables with API key pattern
        # API_KEY_<NAME>=<key>:<scopes>
        for env_var, value in os.environ.items():
            if env_var.startswith("API_KEY_"):
                name = env_var[8:].lower()  # Remove "API_KEY_" prefix
                
                # Parse value (format: "key:scope1,scope2" or just "key")
                if ":" in value:
                    key, scopes_str = value.split(":", 1)
                    scopes = [s.strip() for s in scopes_str.split(",")]
                else:
                    key = value
                    scopes = ["read", "write"]  # Default scopes
                
                valid_keys[key] = {
                    "name": name,
                    "user_id": f"api_key_{name}",
                    "scopes": scopes,
                    "created_from": env_var
                }
        
        # Add a default development API key if in development mode
        if self.settings.is_development() and not valid_keys:
            dev_key = "dev-api-key-12345"
            valid_keys[dev_key] = {
                "name": "development",
                "user_id": "dev_user",
                "scopes": ["read", "write", "admin"],
                "created_from": "default_dev"
            }
            logger.info(f"Added development API key: {dev_key}")
        
        return valid_keys
    
    def _create_auth_error_response(
        self,
        error_type: ErrorType,
        message: str,
        request_id: Optional[str] = None
    ) -> JSONResponse:
        """
        Create an authentication error response.
        
        Args:
            error_type: Type of authentication error
            message: Error message
            request_id: Request correlation ID
            
        Returns:
            JSONResponse: Error response
        """
        error_response = ErrorResponse(
            error=ErrorDetail(
                type=error_type,
                code="authentication_failed",
                message=message
            ),
            request_id=request_id
        )
        
        status_code = 401 if error_type == ErrorType.AUTHENTICATION else 403
        
        return JSONResponse(
            status_code=status_code,
            content=error_response.model_dump(),
            headers={"WWW-Authenticate": "Bearer"}
        )