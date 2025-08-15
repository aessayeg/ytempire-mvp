"""
Response Compression Middleware
Reduces response size by 60-80% using gzip compression
"""
import gzip
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import MutableHeaders
import io

class CompressionMiddleware(BaseHTTPMiddleware):
    """
    Middleware to compress responses using gzip
    Significantly reduces bandwidth and improves response times
    """
    
    def __init__(self, app, minimum_size: int = 1000, compression_level: int = 6):
        """
        Initialize compression middleware
        
        Args:
            app: FastAPI application
            minimum_size: Minimum response size to compress (bytes)
            compression_level: Gzip compression level (1-9, higher = better compression but slower)
        """
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compression_level = compression_level
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return await call_next(request)
        
        # Process request
        response = await call_next(request)
        
        # Check if response should be compressed
        if not self._should_compress(response):
            return response
        
        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        # Check minimum size
        if len(body) < self.minimum_size:
            # Return original response
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        
        # Compress body
        compressed_body = gzip.compress(body, compresslevel=self.compression_level)
        
        # Update headers
        headers = MutableHeaders(response.headers)
        headers["content-encoding"] = "gzip"
        headers["content-length"] = str(len(compressed_body))
        headers["vary"] = "Accept-Encoding"
        
        # Return compressed response
        return Response(
            content=compressed_body,
            status_code=response.status_code,
            headers=dict(headers),
            media_type=response.media_type
        )
    
    def _should_compress(self, response: Response) -> bool:
        """Check if response should be compressed"""
        
        # Don't compress if already encoded
        if "content-encoding" in response.headers:
            return False
        
        # Check content type
        content_type = response.headers.get("content-type", "").lower()
        
        # Compress these content types
        compressible_types = [
            "application/json",
            "application/javascript",
            "application/xml",
            "text/css",
            "text/html",
            "text/plain",
            "text/xml",
            "application/x-javascript",
            "application/vnd.api+json",
        ]
        
        return any(ct in content_type for ct in compressible_types)


class BrotliCompressionMiddleware(BaseHTTPMiddleware):
    """
    Alternative middleware using Brotli compression (better than gzip)
    Requires: pip install brotli
    """
    
    def __init__(self, app, minimum_size: int = 1000, quality: int = 4):
        """
        Initialize Brotli compression
        
        Args:
            app: FastAPI application  
            minimum_size: Minimum response size to compress (bytes)
            quality: Brotli quality (0-11, higher = better but slower)
        """
        super().__init__(app)
        self.minimum_size = minimum_size
        self.quality = quality
        
        try:
            import brotli
            self.brotli = brotli
            self.enabled = True
        except ImportError:
            self.enabled = False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)
        
        # Check if client accepts brotli
        accept_encoding = request.headers.get("accept-encoding", "")
        if "br" not in accept_encoding.lower():
            return await call_next(request)
        
        # Process request
        response = await call_next(request)
        
        # Check if response should be compressed
        if "content-encoding" in response.headers:
            return response
        
        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        # Check minimum size
        if len(body) < self.minimum_size:
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        
        # Compress with Brotli
        compressed_body = self.brotli.compress(body, quality=self.quality)
        
        # Update headers
        headers = MutableHeaders(response.headers)
        headers["content-encoding"] = "br"
        headers["content-length"] = str(len(compressed_body))
        headers["vary"] = "Accept-Encoding"
        
        return Response(
            content=compressed_body,
            status_code=response.status_code,
            headers=dict(headers),
            media_type=response.media_type
        )