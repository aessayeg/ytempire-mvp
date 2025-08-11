"""
Security Headers Middleware
Implements comprehensive security headers for API protection
"""
from typing import Callable
from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import hashlib
import secrets
from datetime import datetime

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add comprehensive security headers to all responses
    Implements OWASP recommended security headers
    """
    
    def __init__(
        self,
        app: ASGIApp,
        enable_hsts: bool = True,
        enable_csp: bool = True,
        csp_report_uri: str = None,
        frame_options: str = "DENY",
        content_type_options: str = "nosniff",
        xss_protection: str = "1; mode=block",
        referrer_policy: str = "strict-origin-when-cross-origin",
        permissions_policy: str = "geolocation=(), microphone=(), camera=()"
    ):
        super().__init__(app)
        self.enable_hsts = enable_hsts
        self.enable_csp = enable_csp
        self.csp_report_uri = csp_report_uri
        self.frame_options = frame_options
        self.content_type_options = content_type_options
        self.xss_protection = xss_protection
        self.referrer_policy = referrer_policy
        self.permissions_policy = permissions_policy
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate nonce for CSP
        nonce = secrets.token_urlsafe(16)
        request.state.csp_nonce = nonce
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response, nonce)
        
        return response
    
    def _add_security_headers(self, response: Response, nonce: str):
        """Add all security headers to response"""
        
        # 1. Strict-Transport-Security (HSTS)
        # Forces HTTPS for 1 year, including subdomains
        if self.enable_hsts:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        # 2. Content-Security-Policy (CSP)
        # Prevents XSS, clickjacking, and other code injection attacks
        if self.enable_csp:
            csp_directives = [
                "default-src 'self'",
                f"script-src 'self' 'nonce-{nonce}' https://cdn.jsdelivr.net",
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
                "font-src 'self' https://fonts.gstatic.com",
                "img-src 'self' data: https:",
                "connect-src 'self' wss: https://api.stripe.com",
                "frame-ancestors 'none'",
                "base-uri 'self'",
                "form-action 'self'",
                "upgrade-insecure-requests",
            ]
            
            if self.csp_report_uri:
                csp_directives.append(f"report-uri {self.csp_report_uri}")
            
            response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        # 3. X-Frame-Options
        # Prevents clickjacking attacks
        response.headers["X-Frame-Options"] = self.frame_options
        
        # 4. X-Content-Type-Options
        # Prevents MIME type sniffing
        response.headers["X-Content-Type-Options"] = self.content_type_options
        
        # 5. X-XSS-Protection
        # Enable browser XSS filter (legacy but still useful)
        response.headers["X-XSS-Protection"] = self.xss_protection
        
        # 6. Referrer-Policy
        # Controls referrer information sent with requests
        response.headers["Referrer-Policy"] = self.referrer_policy
        
        # 7. Permissions-Policy (formerly Feature-Policy)
        # Controls browser features and APIs
        response.headers["Permissions-Policy"] = self.permissions_policy
        
        # 8. Cache-Control for sensitive data
        # Prevent caching of sensitive responses
        if response.status_code == 200:
            # For API responses, set appropriate cache headers
            if "api" in str(response.headers.get("content-type", "")):
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
        
        # 9. Additional security headers
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        
        # 10. Remove sensitive headers
        headers_to_remove = [
            "Server",
            "X-Powered-By",
            "X-AspNet-Version",
            "X-AspNetMvc-Version"
        ]
        for header in headers_to_remove:
            response.headers.pop(header, None)


class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """
    Enhanced CORS middleware with security considerations
    """
    
    def __init__(
        self,
        app: ASGIApp,
        allowed_origins: list = None,
        allowed_methods: list = None,
        allowed_headers: list = None,
        allow_credentials: bool = False,
        max_age: int = 3600
    ):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["http://localhost:3000"]
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allowed_headers = allowed_headers or ["*"]
        self.allow_credentials = allow_credentials
        self.max_age = max_age
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Handle preflight requests
        if request.method == "OPTIONS":
            return self._create_preflight_response(request)
        
        # Process request
        response = await call_next(request)
        
        # Add CORS headers
        self._add_cors_headers(request, response)
        
        return response
    
    def _create_preflight_response(self, request: Request) -> Response:
        """Create response for preflight OPTIONS request"""
        response = Response(content="", status_code=200)
        self._add_cors_headers(request, response)
        return response
    
    def _add_cors_headers(self, request: Request, response: Response):
        """Add CORS headers to response"""
        origin = request.headers.get("origin")
        
        # Validate origin
        if origin and self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            
            if self.allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"
            
            if request.method == "OPTIONS":
                response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
                response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
                response.headers["Access-Control-Max-Age"] = str(self.max_age)
        
        # Always vary on Origin for security
        response.headers["Vary"] = "Origin"
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed"""
        if "*" in self.allowed_origins:
            return True
        return origin in self.allowed_origins


class ContentSecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for content validation and sanitization
    """
    
    def __init__(
        self,
        app: ASGIApp,
        max_content_length: int = 10 * 1024 * 1024,  # 10MB default
        allowed_content_types: list = None
    ):
        super().__init__(app)
        self.max_content_length = max_content_length
        self.allowed_content_types = allowed_content_types or [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_content_length:
            return Response(
                content={"detail": "Request entity too large"},
                status_code=413,
                headers={"Content-Type": "application/json"}
            )
        
        # Validate content type for POST/PUT/PATCH requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "").split(";")[0].strip()
            if content_type and not self._is_content_type_allowed(content_type):
                return Response(
                    content={"detail": f"Content type '{content_type}' not allowed"},
                    status_code=415,
                    headers={"Content-Type": "application/json"}
                )
        
        # Process request
        response = await call_next(request)
        
        return response
    
    def _is_content_type_allowed(self, content_type: str) -> bool:
        """Check if content type is allowed"""
        return any(
            allowed in content_type 
            for allowed in self.allowed_content_types
        )


def setup_security_headers(app):
    """
    Setup all security middleware for the FastAPI application
    """
    from backend.app.core.config import settings
    
    # Add security headers middleware
    app.add_middleware(
        SecurityHeadersMiddleware,
        enable_hsts=settings.ENVIRONMENT == "production",
        enable_csp=True,
        csp_report_uri="/api/v1/security/csp-report",
        frame_options="DENY",
        content_type_options="nosniff",
        xss_protection="1; mode=block",
        referrer_policy="strict-origin-when-cross-origin"
    )
    
    # Add CORS security middleware
    app.add_middleware(
        CORSSecurityMiddleware,
        allowed_origins=settings.BACKEND_CORS_ORIGINS,
        allowed_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allowed_headers=["*"],
        allow_credentials=True,
        max_age=3600
    )
    
    # Add content security middleware
    app.add_middleware(
        ContentSecurityMiddleware,
        max_content_length=settings.MAX_UPLOAD_SIZE,
        allowed_content_types=[
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
            "video/mp4",
            "audio/mpeg",
            "image/jpeg",
            "image/png"
        ]
    )
    
    return app