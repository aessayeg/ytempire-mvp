# 5. API SPECIFICATIONS - YTEMPIRE

**Version 2.0 | January 2025**  
**Document Status: Consolidated & Standardized**  
**Last Updated: January 2025**

---

## 5.1 REST API Contract

### API Design Principles

```yaml
Core Principles:
  RESTful Standards:
    - Resource-based URLs
    - HTTP verbs for actions
    - Stateless operations
    - Hypermedia links
  
  Consistency:
    - Uniform response structure
    - Standard error format
    - Predictable naming
    - Common patterns
  
  Versioning:
    - URL path versioning (/v1/)
    - Backward compatibility
    - Deprecation notices
    - Migration guides
  
  Security:
    - HTTPS only
    - Authentication required
    - Rate limiting
    - Input validation
```

### Base URL Structure

```
Production: https://api.ytempire.com/v1
Staging:    https://staging-api.ytempire.com/v1
Local:      http://localhost:8000/v1
```

### Standard Response Format

```json
{
  "success": true,
  "data": {
    "type": "resource_type",
    "id": "resource_id",
    "attributes": {
      // Resource attributes
    },
    "relationships": {
      // Related resources
    }
  },
  "meta": {
    "timestamp": "2025-01-15T10:30:00Z",
    "version": "1.0",
    "request_id": "req_abc123"
  },
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 100,
    "total_pages": 5
  }
}
```

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Validation failed for the provided input",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format",
        "code": "INVALID_FORMAT"
      }
    ],
    "request_id": "req_xyz789",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

### HTTP Status Codes

```yaml
Success Codes:
  200 OK: Successful GET, PUT
  201 Created: Successful POST
  202 Accepted: Async operation started
  204 No Content: Successful DELETE

Client Error Codes:
  400 Bad Request: Invalid input
  401 Unauthorized: Missing/invalid auth
  403 Forbidden: Insufficient permissions
  404 Not Found: Resource doesn't exist
  409 Conflict: Resource conflict
  422 Unprocessable Entity: Validation error
  429 Too Many Requests: Rate limit exceeded

Server Error Codes:
  500 Internal Server Error: Server error
  502 Bad Gateway: External service error
  503 Service Unavailable: Maintenance/overload
  504 Gateway Timeout: External service timeout
```

---

## 5.2 OpenAPI 3.0 Specification

### OpenAPI Definition

```yaml
openapi: 3.0.3
info:
  title: YTEMPIRE API
  description: Automated YouTube content generation platform API
  version: 1.0.0
  contact:
    name: API Support
    email: api@ytempire.com
  license:
    name: Proprietary
    
servers:
  - url: https://api.ytempire.com/v1
    description: Production server
  - url: https://staging-api.ytempire.com/v1
    description: Staging server
  - url: http://localhost:8000/v1
    description: Local development

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
    
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

  schemas:
    User:
      type: object
      required:
        - email
        - username
      properties:
        id:
          type: string
          format: uuid
        email:
          type: string
          format: email
        username:
          type: string
          minLength: 3
          maxLength: 50
        tier:
          type: string
          enum: [free, starter, growth, scale]
        created_at:
          type: string
          format: date-time
    
    Channel:
      type: object
      properties:
        id:
          type: string
          pattern: '^ch_[a-f0-9]{16}$'
        name:
          type: string
        youtube_channel_id:
          type: string
        status:
          type: string
          enum: [active, paused, deleted]
        
    Video:
      type: object
      properties:
        id:
          type: string
          pattern: '^vid_[a-f0-9]{16}$'
        title:
          type: string
        status:
          type: string
          enum: [pending, processing, completed, failed, published]
        cost:
          type: number
          format: float
        duration_seconds:
          type: integer
```

### API Documentation Generation

```python
# FastAPI automatic OpenAPI generation
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI()

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="YTEMPIRE API",
        version="1.0.0",
        description="Automated YouTube content generation platform",
        routes=app.routes,
    )
    
    # Add custom components
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

---

## 5.3 Endpoint Documentation

### Authentication Endpoints

```yaml
POST /auth/register:
  description: Register new user account
  request:
    body:
      email: string
      password: string
      username: string
  response:
    201: User created successfully
    409: Email already exists
    422: Validation error

POST /auth/login:
  description: Authenticate user and receive tokens
  request:
    body:
      email: string
      password: string
  response:
    200: Authentication successful
    401: Invalid credentials
    429: Too many attempts

POST /auth/refresh:
  description: Refresh access token
  request:
    body:
      refresh_token: string
  response:
    200: New tokens generated
    401: Invalid refresh token

POST /auth/logout:
  description: Invalidate user session
  security: BearerAuth
  response:
    204: Logged out successfully
```

### User Management Endpoints

```yaml
GET /users/me:
  description: Get current user profile
  security: BearerAuth
  response:
    200: User profile data
    401: Unauthorized

PUT /users/me:
  description: Update user profile
  security: BearerAuth
  request:
    body:
      username?: string
      timezone?: string
      notification_preferences?: object
  response:
    200: Profile updated
    422: Validation error

GET /users/me/usage:
  description: Get usage statistics
  security: BearerAuth
  query:
    start_date?: date
    end_date?: date
  response:
    200: Usage statistics
    
DELETE /users/me:
  description: Delete user account
  security: BearerAuth
  response:
    204: Account deleted
    403: Cannot delete with active subscription
```

### Channel Management Endpoints

```yaml
GET /channels:
  description: List user's channels
  security: BearerAuth
  query:
    page?: integer
    per_page?: integer
    status?: enum[active,paused]
    sort?: string
  response:
    200: List of channels with pagination

POST /channels:
  description: Create new channel
  security: BearerAuth
  request:
    body:
      name: string
      niche: string
      youtube_channel_id?: string
      settings: object
  response:
    201: Channel created
    403: Channel limit reached
    422: Validation error

GET /channels/{channel_id}:
  description: Get channel details
  security: BearerAuth
  parameters:
    channel_id: string (path)
  response:
    200: Channel details
    404: Channel not found

PUT /channels/{channel_id}:
  description: Update channel settings
  security: BearerAuth
  parameters:
    channel_id: string (path)
  request:
    body:
      name?: string
      status?: enum[active,paused]
      settings?: object
  response:
    200: Channel updated
    404: Channel not found

DELETE /channels/{channel_id}:
  description: Delete channel
  security: BearerAuth
  parameters:
    channel_id: string (path)
  response:
    204: Channel deleted
    404: Channel not found
    409: Channel has active videos

POST /channels/{channel_id}/sync:
  description: Sync with YouTube
  security: BearerAuth
  parameters:
    channel_id: string (path)
  response:
    202: Sync started
    404: Channel not found
```

### Video Management Endpoints

```yaml
GET /videos:
  description: List videos across all channels
  security: BearerAuth
  query:
    channel_id?: string
    status?: enum[pending,processing,completed,failed,published]
    start_date?: date
    end_date?: date
    page?: integer
    per_page?: integer
    sort?: string
  response:
    200: List of videos with pagination

POST /videos/generate:
  description: Queue new video for generation
  security: BearerAuth
  request:
    body:
      channel_id: string
      topic: string
      style: enum[educational,entertainment,review,tutorial]
      duration_target: integer (seconds)
      quality: enum[standard,high,premium]
      schedule_for?: datetime
  response:
    202: Video queued for generation
    403: Daily limit reached
    422: Validation error

GET /videos/{video_id}:
  description: Get video details
  security: BearerAuth
  parameters:
    video_id: string (path)
  response:
    200: Video details
    404: Video not found

PUT /videos/{video_id}:
  description: Update video metadata
  security: BearerAuth
  parameters:
    video_id: string (path)
  request:
    body:
      title?: string
      description?: string
      tags?: array[string]
      thumbnail_url?: string
  response:
    200: Video updated
    404: Video not found
    409: Cannot edit published video

DELETE /videos/{video_id}:
  description: Delete video
  security: BearerAuth
  parameters:
    video_id: string (path)
  response:
    204: Video deleted
    404: Video not found
    409: Cannot delete published video

POST /videos/{video_id}/publish:
  description: Publish video to YouTube
  security: BearerAuth
  parameters:
    video_id: string (path)
  request:
    body:
      schedule_time?: datetime
      privacy: enum[public,unlisted,private]
  response:
    202: Publishing started
    404: Video not found
    409: Video already published

GET /videos/{video_id}/cost:
  description: Get detailed cost breakdown
  security: BearerAuth
  parameters:
    video_id: string (path)
  response:
    200: Cost breakdown details
    404: Video not found
```

### Analytics Endpoints

```yaml
GET /analytics/overview:
  description: Get dashboard overview metrics
  security: BearerAuth
  query:
    period?: enum[today,week,month,year]
  response:
    200: Overview metrics

GET /analytics/channels/{channel_id}:
  description: Get channel analytics
  security: BearerAuth
  parameters:
    channel_id: string (path)
  query:
    start_date: date
    end_date: date
    metrics?: array[views,revenue,subscribers]
  response:
    200: Channel analytics data
    404: Channel not found

GET /analytics/videos/{video_id}:
  description: Get video performance metrics
  security: BearerAuth
  parameters:
    video_id: string (path)
  response:
    200: Video analytics
    404: Video not found

GET /analytics/revenue:
  description: Get revenue analytics
  security: BearerAuth
  query:
    group_by?: enum[day,week,month]
    start_date: date
    end_date: date
  response:
    200: Revenue data

GET /analytics/costs:
  description: Get cost analytics
  security: BearerAuth
  query:
    group_by?: enum[service,channel,day]
    start_date: date
    end_date: date
  response:
    200: Cost breakdown data
```

### Admin Endpoints

```yaml
GET /admin/users:
  description: List all users (admin only)
  security: BearerAuth
  query:
    page?: integer
    per_page?: integer
    tier?: string
    status?: string
  response:
    200: User list
    403: Insufficient permissions

GET /admin/system/health:
  description: System health check
  security: BearerAuth
  response:
    200: System health status
    403: Insufficient permissions

GET /admin/system/metrics:
  description: System performance metrics
  security: BearerAuth
  response:
    200: System metrics
    403: Insufficient permissions

POST /admin/users/{user_id}/impersonate:
  description: Impersonate user (admin only)
  security: BearerAuth
  parameters:
    user_id: string (path)
  response:
    200: Impersonation token
    403: Insufficient permissions
    404: User not found
```

---

## 5.4 Authentication & Authorization

### JWT Token Structure

```python
# Token payload structure
{
    "sub": "user_id",           # Subject (user ID)
    "email": "user@example.com", # User email
    "username": "username",      # Username
    "role": "user",              # Role (user/admin)
    "tier": "growth",            # Subscription tier
    "exp": 1705325400,          # Expiration timestamp
    "iat": 1705324500,          # Issued at timestamp
    "jti": "unique_token_id",   # JWT ID for revocation
    "type": "access"             # Token type (access/refresh)
}
```

### Authentication Flow

```python
# Authentication implementation
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

class AuthService:
    def __init__(self):
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12
        )
        self.SECRET_KEY = settings.SECRET_KEY
        self.ALGORITHM = "HS256"
        self.ACCESS_TOKEN_EXPIRE_MINUTES = 15
        self.REFRESH_TOKEN_EXPIRE_DAYS = 7
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password)
    
    def create_access_token(self, data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({
            "exp": expire,
            "type": "access",
            "iat": datetime.utcnow()
        })
        return jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
    
    def create_refresh_token(self, data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({
            "exp": expire,
            "type": "refresh",
            "iat": datetime.utcnow()
        })
        return jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
```

### Authorization Levels

```python
# Role-based access control
from enum import Enum
from typing import List

class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

class UserTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    GROWTH = "growth"
    SCALE = "scale"

# Permission definitions
PERMISSIONS = {
    UserRole.USER: [
        "read:own_profile",
        "update:own_profile",
        "create:channels",
        "read:own_channels",
        "update:own_channels",
        "delete:own_channels",
        "create:videos",
        "read:own_videos",
        "update:own_videos",
        "delete:own_videos",
        "read:own_analytics"
    ],
    UserRole.ADMIN: [
        # Inherits all USER permissions plus:
        "read:all_users",
        "read:all_channels",
        "read:all_videos",
        "read:system_metrics",
        "impersonate:users"
    ],
    UserRole.SUPER_ADMIN: [
        # Inherits all ADMIN permissions plus:
        "update:all_users",
        "delete:all_users",
        "update:system_config",
        "manage:api_keys"
    ]
}

# Tier limitations
TIER_LIMITS = {
    UserTier.FREE: {
        "channels": 1,
        "videos_per_day": 1,
        "video_quality": "standard",
        "analytics_retention_days": 7
    },
    UserTier.STARTER: {
        "channels": 3,
        "videos_per_day": 5,
        "video_quality": "high",
        "analytics_retention_days": 30
    },
    UserTier.GROWTH: {
        "channels": 5,
        "videos_per_day": 10,
        "video_quality": "premium",
        "analytics_retention_days": 90
    },
    UserTier.SCALE: {
        "channels": 20,
        "videos_per_day": 50,
        "video_quality": "premium",
        "analytics_retention_days": 365
    }
}
```

### API Key Management

```python
# API key structure
API_KEY_FORMAT = "yte_live_[32_random_characters]"
# Example: yte_live_sk4f8g2h1j5k9l3m7n6p0q2r4t8v1w3x

# API key permissions
API_KEY_SCOPES = [
    "videos:read",
    "videos:write",
    "channels:read",
    "channels:write",
    "analytics:read",
    "webhooks:write"
]

# Rate limiting by API key
API_KEY_RATE_LIMITS = {
    "default": "1000/hour",
    "premium": "10000/hour",
    "enterprise": "unlimited"
}
```

---

## 5.5 Error Handling Standards

### Error Code System

```python
# Error code structure: [Category][Number]
# Categories: AUTH, VAL, RATE, PERM, RES, EXT, SYS

ERROR_CODES = {
    # Authentication Errors (AUTH)
    "AUTH001": "Invalid credentials",
    "AUTH002": "Token expired",
    "AUTH003": "Token invalid",
    "AUTH004": "Refresh token required",
    "AUTH005": "Account locked",
    
    # Validation Errors (VAL)
    "VAL001": "Required field missing",
    "VAL002": "Invalid format",
    "VAL003": "Value out of range",
    "VAL004": "Duplicate value",
    "VAL005": "Invalid enum value",
    
    # Rate Limiting Errors (RATE)
    "RATE001": "API rate limit exceeded",
    "RATE002": "Daily video limit reached",
    "RATE003": "Channel limit reached",
    "RATE004": "Concurrent request limit",
    
    # Permission Errors (PERM)
    "PERM001": "Insufficient permissions",
    "PERM002": "Resource access denied",
    "PERM003": "Feature not available in tier",
    "PERM004": "Admin access required",
    
    # Resource Errors (RES)
    "RES001": "Resource not found",
    "RES002": "Resource already exists",
    "RES003": "Resource in use",
    "RES004": "Resource limit exceeded",
    
    # External Service Errors (EXT)
    "EXT001": "YouTube API error",
    "EXT002": "OpenAI API error",
    "EXT003": "Payment processing error",
    "EXT004": "External service timeout",
    "EXT005": "Quota exceeded",
    
    # System Errors (SYS)
    "SYS001": "Internal server error",
    "SYS002": "Database connection error",
    "SYS003": "Service unavailable",
    "SYS004": "Maintenance mode"
}
```

### Error Response Examples

```json
// 400 Bad Request - Validation Error
{
  "success": false,
  "error": {
    "code": "VAL002",
    "message": "Invalid format",
    "details": [
      {
        "field": "email",
        "message": "Email must be a valid email address",
        "provided_value": "invalid-email"
      }
    ],
    "request_id": "req_abc123",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}

// 401 Unauthorized - Authentication Error
{
  "success": false,
  "error": {
    "code": "AUTH002",
    "message": "Token expired",
    "details": {
      "expired_at": "2025-01-15T10:00:00Z",
      "current_time": "2025-01-15T10:30:00Z"
    },
    "request_id": "req_def456",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}

// 429 Too Many Requests - Rate Limit
{
  "success": false,
  "error": {
    "code": "RATE001",
    "message": "API rate limit exceeded",
    "details": {
      "limit": 100,
      "remaining": 0,
      "reset_at": "2025-01-15T11:00:00Z"
    },
    "request_id": "req_ghi789",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}

// 500 Internal Server Error
{
  "success": false,
  "error": {
    "code": "SYS001",
    "message": "Internal server error",
    "details": {
      "trace_id": "trace_xyz123",
      "support_contact": "support@ytempire.com"
    },
    "request_id": "req_jkl012",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

### Error Handling Implementation

```python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import traceback
import logging

logger = logging.getLogger(__name__)

class APIError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: dict = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

async def error_handler(request: Request, exc: Exception):
    request_id = request.state.request_id if hasattr(request.state, 'request_id') else 'unknown'
    
    if isinstance(exc, APIError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
    
    elif isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": f"HTTP{exc.status_code}",
                    "message": exc.detail,
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
    
    else:
        # Log unexpected errors
        logger.error(f"Unexpected error: {traceback.format_exc()}")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "SYS001",
                    "message": "Internal server error",
                    "details": {
                        "trace_id": request_id,
                        "support_contact": "support@ytempire.com"
                    },
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
```

---

## Document Control

- **Version**: 2.0
- **Last Updated**: January 2025
- **Next Review**: February 2025
- **Owner**: API Development Engineer
- **Approved By**: Backend Team Lead

---

## Navigation

- [← Previous: Technical Architecture](./4-technical-architecture.md)
- [→ Next: Database Design](./6-database-design.md)