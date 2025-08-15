"""
Custom exception classes for YTEmpire
"""
from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class YTEmpireException(Exception):
    """Base exception class for YTEmpire"""
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or "YTEMPIRE_ERROR"
        self.details = details or {}
        super().__init__(self.message)


class APIException(YTEmpireException):
    """Base API exception"""
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)
        self.status_code = status_code


# Authentication Exceptions
class AuthenticationException(APIException):
    """Authentication failed exception"""
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTH_FAILED",
            details=details
        )


class InvalidTokenException(AuthenticationException):
    """Invalid token exception"""
    def __init__(self):
        super().__init__(
            message="Invalid or expired token",
            details={"error": "token_invalid"}
        )


class InsufficientPermissionsException(APIException):
    """Insufficient permissions exception"""
    def __init__(self, resource: Optional[str] = None):
        message = f"Insufficient permissions to access {resource}" if resource else "Insufficient permissions"
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="INSUFFICIENT_PERMISSIONS"
        )


# Resource Exceptions
class ResourceNotFoundException(APIException):
    """Resource not found exception"""
    def __init__(self, resource_type: str, resource_id: Optional[str] = None):
        message = f"{resource_type} not found"
        if resource_id:
            message += f": {resource_id}"
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="RESOURCE_NOT_FOUND",
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


class ResourceAlreadyExistsException(APIException):
    """Resource already exists exception"""
    def __init__(self, resource_type: str, identifier: Optional[str] = None):
        message = f"{resource_type} already exists"
        if identifier:
            message += f": {identifier}"
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            error_code="RESOURCE_EXISTS",
            details={"resource_type": resource_type, "identifier": identifier}
        )


# Validation Exceptions
class ValidationException(APIException):
    """Data validation exception"""
    def __init__(self, field: str, message: str, value: Optional[Any] = None):
        super().__init__(
            message=f"Validation error for field '{field}': {message}",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details={"field": field, "value": value, "error": message}
        )


class InvalidRequestException(APIException):
    """Invalid request exception"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="INVALID_REQUEST",
            details=details
        )


# Business Logic Exceptions
class QuotaExceededException(APIException):
    """Quota exceeded exception"""
    def __init__(self, quota_type: str, limit: int, current: int):
        super().__init__(
            message=f"{quota_type} quota exceeded. Limit: {limit}, Current: {current}",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="QUOTA_EXCEEDED",
            details={"quota_type": quota_type, "limit": limit, "current": current}
        )


class ThresholdExceededException(APIException):
    """Cost threshold exceeded exception"""
    def __init__(
        self,
        threshold_type: str,
        threshold_value: float,
        current_value: float,
        service: Optional[str] = None
    ):
        message = f"{threshold_type} cost threshold exceeded: ${current_value:.2f} > ${threshold_value:.2f}"
        super().__init__(
            message=message,
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            error_code="THRESHOLD_EXCEEDED",
            details={
                "threshold_type": threshold_type,
                "threshold_value": threshold_value,
                "current_value": current_value,
                "service": service
            }
        )


class PaymentRequiredException(APIException):
    """Payment required exception"""
    def __init__(self, message: str = "Payment required to continue", feature: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            error_code="PAYMENT_REQUIRED",
            details={"feature": feature}
        )


# External Service Exceptions
class ExternalServiceException(YTEmpireException):
    """External service error"""
    def __init__(
        self,
        service: str,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[Dict] = None
    ):
        super().__init__(
            message=f"{service} error: {message}",
            error_code=f"{service.upper()}_ERROR",
            details=details or {}
        )
        self.service = service
        self.status_code = status_code


class YouTubeAPIException(ExternalServiceException):
    """YouTube API exception"""
    def __init__(self, message: str, quota_used: Optional[int] = None):
        details = {"quota_used": quota_used} if quota_used else {}
        super().__init__(
            service="YouTube",
            message=message,
            details=details
        )


class OpenAIException(ExternalServiceException):
    """OpenAI API exception"""
    def __init__(self, message: str, model: Optional[str] = None, tokens_used: Optional[int] = None):
        details = {}
        if model:
            details["model"] = model
        if tokens_used:
            details["tokens_used"] = tokens_used
        super().__init__(
            service="OpenAI",
            message=message,
            details=details
        )


class ElevenLabsException(ExternalServiceException):
    """ElevenLabs API exception"""
    def __init__(self, message: str, characters_used: Optional[int] = None):
        details = {"characters_used": characters_used} if characters_used else {}
        super().__init__(
            service="ElevenLabs",
            message=message,
            details=details
        )


# Database Exceptions
class DatabaseException(YTEmpireException):
    """Database operation exception"""
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details={"operation": operation}
        )


class TransactionException(DatabaseException):
    """Database transaction exception"""
    def __init__(self, message: str = "Transaction failed"):
        super().__init__(
            message=message,
            operation="transaction"
        )


# Processing Exceptions
class VideoProcessingException(YTEmpireException):
    """Video processing exception"""
    def __init__(self, message: str, video_id: Optional[str] = None, stage: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="VIDEO_PROCESSING_ERROR",
            details={"video_id": video_id, "stage": stage}
        )


class ContentGenerationException(YTEmpireException):
    """Content generation exception"""
    def __init__(self, message: str, content_type: str, reason: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="CONTENT_GENERATION_ERROR",
            details={"content_type": content_type, "reason": reason}
        )


# Configuration Exceptions
class ConfigurationException(YTEmpireException):
    """Configuration error"""
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key}
        )


class MissingConfigurationException(ConfigurationException):
    """Missing configuration exception"""
    def __init__(self, config_key: str):
        super().__init__(
            message=f"Required configuration missing: {config_key}",
            config_key=config_key
        )


# Rate Limiting Exceptions
class RateLimitException(APIException):
    """Rate limit exceeded exception"""
    def __init__(
        self,
        limit: int,
        window: str,
        retry_after: Optional[int] = None
    ):
        message = f"Rate limit exceeded: {limit} requests per {window}"
        details = {"limit": limit, "window": window}
        if retry_after:
            details["retry_after_seconds"] = retry_after
            
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details
        )