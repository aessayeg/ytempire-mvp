"""
JWT Authentication Middleware
Owner: Security Engineer #1
"""

import logging
from typing import Optional, Tuple, List
from fastapi import HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse
import time
from datetime import datetime

from app.services.auth_service import AuthService
from app.repositories.user_repository import UserRepository
from app.core.metrics import metrics
from app.core.database import AsyncSessionLocal
from app.models.user import User

logger = logging.getLogger(__name__)

# Paths that don't require authentication
PUBLIC_PATHS = {
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/auth/register",
    "/api/v1/auth/login",
    "/api/v1/auth/refresh",
    "/api/v1/youtube/oauth/callback",
    "/api/v1/webhooks",  # Webhooks might have their own auth
    "/health",
    "/metrics"
}

# Paths that require authentication
PROTECTED_PREFIXES = [
    "/api/v1/users",
    "/api/v1/channels",
    "/api/v1/videos",
    "/api/v1/analytics",
    "/api/v1/youtube"
]


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """JWT Authentication Middleware for protecting API endpoints."""
    
    def __init__(self, app):
        super().__init__(app)
        self.security = HTTPBearer(auto_error=False)
    
    async def dispatch(self, request: StarletteRequest, call_next) -> StarletteResponse:
        """Process request through JWT authentication."""
        start_time = time.time()
        
        # Skip authentication for public paths
        if self._is_public_path(request.url.path):
            response = await call_next(request)
            self._add_timing_header(response, start_time)
            return response
        
        # Skip authentication for non-protected paths
        if not self._requires_auth(request.url.path):
            response = await call_next(request)
            self._add_timing_header(response, start_time)
            return response
        
        try:
            # Extract and validate JWT token
            user = await self._authenticate_request(request)
            
            if not user:
                return self._create_unauthorized_response("Authentication required")
            
            # Add user to request state
            request.state.user = user
            request.state.user_id = user.id
            request.state.authenticated = True
            
            # Log successful authentication
            logger.debug(f"Authenticated user: {user.email} for {request.method} {request.url.path}")
            
            # Continue with request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            self._add_timing_header(response, start_time)
            
            return response
            
        except HTTPException as e:
            return self._create_error_response(e.status_code, e.detail)
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return self._create_error_response(500, "Internal authentication error")
    
    def _is_public_path(self, path: str) -> bool:
        """Check if path is in public paths list."""
        return path in PUBLIC_PATHS or path.startswith("/static/")
    
    def _requires_auth(self, path: str) -> bool:
        """Check if path requires authentication."""
        return any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES)
    
    async def _authenticate_request(self, request: StarletteRequest) -> Optional[User]:
        """Authenticate request using JWT token."""
        # Extract Authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header missing"
            )
        
        # Parse Bearer token
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format"
            )
        
        token = auth_header.split(" ")[1]
        
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing"
            )
        
        # Validate token and get user
        try:
            async with AsyncSessionLocal() as db:
                user_repo = UserRepository(db)
                auth_service = AuthService(user_repo)
                
                # Verify token and get user
                user = await auth_service.get_current_user(token)
                
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or expired token"
                    )
                
                if not user.is_active:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User account is inactive"
                    )
                
                return user
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token validation failed"
            )
    
    def _create_unauthorized_response(self, detail: str) -> StarletteResponse:
        """Create unauthorized response."""
        from starlette.responses import JSONResponse
        
        return JSONResponse(
            status_code=401,
            content={
                "detail": detail,
                "error": "unauthorized",
                "timestamp": datetime.utcnow().isoformat()
            },
            headers={
                "WWW-Authenticate": "Bearer",
                "X-Auth-Error": detail
            }
        )
    
    def _create_error_response(self, status_code: int, detail: str) -> StarletteResponse:
        """Create error response."""
        from starlette.responses import JSONResponse
        
        return JSONResponse(
            status_code=status_code,
            content={
                "detail": detail,
                "error": "authentication_error",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def _add_security_headers(self, response: StarletteResponse):
        """Add security headers to response."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    def _add_timing_header(self, response: StarletteResponse, start_time: float):
        """Add request timing header."""
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware to prevent abuse."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}
        self.window_start = {}
    
    async def dispatch(self, request: StarletteRequest, call_next) -> StarletteResponse:
        """Apply rate limiting based on client IP."""
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Clean old entries
        self._cleanup_old_entries(current_time)
        
        # Check rate limit
        if self._is_rate_limited(client_ip, current_time):
            # Record rate limit hit
            metrics.record_rate_limit_hit(client_ip)
            return self._create_rate_limit_response()
        
        # Record request
        self._record_request(client_ip, current_time)
        
        # Continue with request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self._get_remaining_requests(client_ip)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response
    
    def _get_client_ip(self, request: StarletteRequest) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers (when behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    def _is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        """Check if client has exceeded rate limit."""
        if client_ip not in self.request_counts:
            return False
        
        window_start = self.window_start.get(client_ip, current_time)
        
        # If more than 60 seconds have passed, reset
        if current_time - window_start >= 60:
            self.request_counts[client_ip] = 0
            self.window_start[client_ip] = current_time
            return False
        
        return self.request_counts[client_ip] >= self.requests_per_minute
    
    def _record_request(self, client_ip: str, current_time: float):
        """Record a request for the client."""
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = 0
            self.window_start[client_ip] = current_time
        
        self.request_counts[client_ip] += 1
    
    def _get_remaining_requests(self, client_ip: str) -> int:
        """Get remaining requests for client."""
        if client_ip not in self.request_counts:
            return self.requests_per_minute
        
        return max(0, self.requests_per_minute - self.request_counts[client_ip])
    
    def _cleanup_old_entries(self, current_time: float):
        """Clean up old rate limit entries."""
        clients_to_remove = []
        
        for client_ip, window_start in self.window_start.items():
            if current_time - window_start >= 300:  # 5 minutes
                clients_to_remove.append(client_ip)
        
        for client_ip in clients_to_remove:
            self.request_counts.pop(client_ip, None)
            self.window_start.pop(client_ip, None)
    
    def _create_rate_limit_response(self) -> StarletteResponse:
        """Create rate limit exceeded response."""
        from starlette.responses import JSONResponse
        
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded",
                "error": "too_many_requests",
                "retry_after": 60,
                "timestamp": datetime.utcnow().isoformat()
            },
            headers={
                "Retry-After": "60",
                "X-RateLimit-Limit": str(self.requests_per_minute),
                "X-RateLimit-Remaining": "0"
            }
        )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all requests."""
    
    async def dispatch(self, request: StarletteRequest, call_next) -> StarletteResponse:
        """Log request and response details."""
        start_time = time.time()
        
        # Get user info if available
        user_info = "anonymous"
        if hasattr(request.state, 'user') and request.state.user:
            user_info = f"{request.state.user.email} ({request.state.user.id})"
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} - User: {user_info} - "
            f"IP: {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {response.status_code} - "
            f"Time: {process_time:.3f}s - "
            f"User: {user_info}"
        )
        
        return response


# Helper function to get current user from request
def get_current_user_from_request(request: Request) -> Optional[User]:
    """Extract current user from request state (set by middleware)."""
    if hasattr(request.state, 'user'):
        return request.state.user
    return None


# Helper function to check if user is authenticated
def is_authenticated(request: Request) -> bool:
    """Check if current request is authenticated."""
    return getattr(request.state, 'authenticated', False)


# Helper function to require authentication
def require_authentication(request: Request) -> User:
    """Require authentication and return user or raise exception."""
    if not is_authenticated(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    user = get_current_user_from_request(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user