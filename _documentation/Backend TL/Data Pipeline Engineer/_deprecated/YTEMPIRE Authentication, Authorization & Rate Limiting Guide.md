# YTEMPIRE Authentication, Authorization & Rate Limiting Guide
**Version 1.0 | January 2025**  
**Owner: API Development Engineer**  
**Security Classification: Confidential**

---

## Executive Summary

This document provides comprehensive implementation guidelines for authentication, authorization, and rate limiting in the YTEMPIRE API. These security measures ensure platform integrity while maintaining excellent developer experience for our automated YouTube content platform.

---

## 1. Authentication Flow

### 1.1 JWT-Based Authentication Architecture

```python
# Authentication flow implementation
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List
import jwt
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
import redis
from pydantic import BaseModel
import secrets
import json

class AuthConfig:
    """Authentication configuration"""
    
    # JWT Configuration
    JWT_SECRET_KEY = "your-secret-key-from-env"  # Load from environment
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 15
    REFRESH_TOKEN_EXPIRE_DAYS = 30
    
    # Password hashing
    PWD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    # Redis for token blacklist
    REDIS_CLIENT = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    # Token prefixes
    ACCESS_TOKEN_PREFIX = "ytempire_access_"
    REFRESH_TOKEN_PREFIX = "ytempire_refresh_"
    
    # Security settings
    ROTATE_REFRESH_TOKENS = True
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION = 300  # 5 minutes

class TokenPayload(BaseModel):
    """JWT token payload structure"""
    sub: str  # Subject (user ID)
    type: str  # Token type (access/refresh)
    exp: int  # Expiration timestamp
    iat: int  # Issued at timestamp
    jti: str  # JWT ID for revocation
    scope: List[str]  # User permissions
    
class AuthService:
    """Core authentication service"""
    
    def __init__(self):
        self.security = HTTPBearer()
        
    def create_access_token(self, user_id: str, scopes: List[str]) -> str:
        """Generate access token"""
        
        payload = {
            "sub": user_id,
            "type": "access",
            "exp": datetime.utcnow() + timedelta(minutes=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES),
            "iat": datetime.utcnow(),
            "jti": f"at_{generate_unique_id()}",
            "scope": scopes
        }
        
        token = jwt.encode(payload, AuthConfig.JWT_SECRET_KEY, algorithm=AuthConfig.JWT_ALGORITHM)
        
        # Store token metadata in Redis for tracking
        AuthConfig.REDIS_CLIENT.setex(
            f"{AuthConfig.ACCESS_TOKEN_PREFIX}{payload['jti']}",
            AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            json.dumps({
                "user_id": user_id,
                "created_at": payload["iat"],
                "scopes": scopes
            })
        )
        
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """Generate refresh token"""
        
        payload = {
            "sub": user_id,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(days=AuthConfig.REFRESH_TOKEN_EXPIRE_DAYS),
            "iat": datetime.utcnow(),
            "jti": f"rt_{generate_unique_id()}"
        }
        
        token = jwt.encode(payload, AuthConfig.JWT_SECRET_KEY, algorithm=AuthConfig.JWT_ALGORITHM)
        
        # Store refresh token in Redis
        AuthConfig.REDIS_CLIENT.setex(
            f"{AuthConfig.REFRESH_TOKEN_PREFIX}{payload['jti']}",
            AuthConfig.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
            user_id
        )
        
        return token
    
    def verify_token(self, credentials: HTTPAuthorizationCredentials) -> TokenPayload:
        """Verify and decode JWT token"""
        
        token = credentials.credentials
        
        try:
            # Decode token
            payload = jwt.decode(
                token, 
                AuthConfig.JWT_SECRET_KEY, 
                algorithms=[AuthConfig.JWT_ALGORITHM]
            )
            
            # Check if token is blacklisted
            if self.is_token_blacklisted(payload["jti"]):
                raise HTTPException(status_code=401, detail="Token has been revoked")
            
            return TokenPayload(**payload)
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def is_token_blacklisted(self, jti: str) -> bool:
        """Check if token is in blacklist"""
        
        # Check access token blacklist
        if AuthConfig.REDIS_CLIENT.exists(f"blacklist_{jti}"):
            return True
            
        return False
    
    def revoke_token(self, jti: str, exp: int):
        """Add token to blacklist"""
        
        ttl = exp - int(datetime.utcnow().timestamp())
        if ttl > 0:
            AuthConfig.REDIS_CLIENT.setex(f"blacklist_{jti}", ttl, "revoked")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return AuthConfig.PWD_CONTEXT.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return AuthConfig.PWD_CONTEXT.verify(plain_password, hashed_password)
```

### 1.2 Login Flow Implementation

```python
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import Dict
import asyncio

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/login", response_model=AuthResponse)
async def login(
    request: LoginRequest,
    client_request: Request,
    db: Session = Depends(get_db),
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
) -> Dict:
    """
    User login endpoint
    
    Flow:
    1. Validate credentials
    2. Generate tokens
    3. Log authentication event
    4. Return tokens
    """
    
    # Rate limit login attempts
    await rate_limiter.check_rate_limit(
        key=f"login:{request.email}",
        limit=5,
        window=300  # 5 attempts per 5 minutes
    )
    
    # Check if account is locked
    lockout_key = f"lockout:{request.email}"
    if AuthConfig.REDIS_CLIENT.exists(lockout_key):
        raise HTTPException(status_code=429, detail="Account temporarily locked due to multiple failed attempts")
    
    # Validate user credentials
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not auth_service.verify_password(request.password, user.hashed_password):
        # Increment failed attempts
        failed_key = f"failed_login:{request.email}"
        failed_attempts = AuthConfig.REDIS_CLIENT.incr(failed_key)
        AuthConfig.REDIS_CLIENT.expire(failed_key, 300)  # Reset after 5 minutes
        
        # Lock account if too many failures
        if failed_attempts >= AuthConfig.MAX_LOGIN_ATTEMPTS:
            AuthConfig.REDIS_CLIENT.setex(lockout_key, AuthConfig.LOCKOUT_DURATION, "locked")
        
        # Log failed attempt
        await log_auth_event(
            event_type="login_failed",
            email=request.email,
            ip_address=client_request.client.host
        )
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check if account is active
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    
    # Clear failed attempts on successful login
    AuthConfig.REDIS_CLIENT.delete(f"failed_login:{request.email}")
    
    # Generate tokens
    access_token = auth_service.create_access_token(
        user_id=str(user.id),
        scopes=user.get_scopes()
    )
    refresh_token = auth_service.create_refresh_token(user_id=str(user.id))
    
    # Log successful authentication
    await log_auth_event(
        event_type="login_success",
        user_id=str(user.id),
        ip_address=client_request.client.host
    )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    return {
        "data": {
            "type": "auth",
            "attributes": {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_in": AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                "token_type": "Bearer"
            },
            "relationships": {
                "user": {
                    "type": "user",
                    "id": str(user.id)
                }
            }
        }
    }

@router.post("/refresh", response_model=AuthResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    db: Session = Depends(get_db)
) -> Dict:
    """
    Refresh access token
    
    Flow:
    1. Validate refresh token
    2. Check if user still active
    3. Generate new access token
    4. Optionally rotate refresh token
    """
    
    try:
        # Verify refresh token
        payload = jwt.decode(
            request.refresh_token,
            AuthConfig.JWT_SECRET_KEY,
            algorithms=[AuthConfig.JWT_ALGORITHM]
        )
        
        if payload["type"] != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        # Check if refresh token exists in Redis
        if not AuthConfig.REDIS_CLIENT.exists(f"{AuthConfig.REFRESH_TOKEN_PREFIX}{payload['jti']}"):
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        # Get user
        user = db.query(User).filter(User.id == payload["sub"]).first()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User not found or inactive")
        
        # Generate new access token
        new_access_token = auth_service.create_access_token(
            user_id=str(user.id),
            scopes=user.get_scopes()
        )
        
        # Optional: Rotate refresh token for enhanced security
        if AuthConfig.ROTATE_REFRESH_TOKENS:
            # Revoke old refresh token
            AuthConfig.REDIS_CLIENT.delete(f"{AuthConfig.REFRESH_TOKEN_PREFIX}{payload['jti']}")
            
            # Generate new refresh token
            new_refresh_token = auth_service.create_refresh_token(user_id=str(user.id))
        else:
            new_refresh_token = request.refresh_token
        
        return {
            "data": {
                "type": "auth",
                "attributes": {
                    "access_token": new_access_token,
                    "refresh_token": new_refresh_token,
                    "expires_in": AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                    "token_type": "Bearer"
                }
            }
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    token: TokenPayload = Depends(verify_token)
) -> Dict:
    """
    Logout user and revoke tokens
    
    Flow:
    1. Revoke access token
    2. Revoke associated refresh token
    3. Clear any session data
    4. Log logout event
    """
    
    # Revoke current access token
    auth_service.revoke_token(token.jti, token.exp)
    
    # Revoke all user's refresh tokens
    pattern = f"{AuthConfig.REFRESH_TOKEN_PREFIX}*"
    for key in AuthConfig.REDIS_CLIENT.scan_iter(match=pattern):
        if AuthConfig.REDIS_CLIENT.get(key) == str(current_user.id):
            AuthConfig.REDIS_CLIENT.delete(key)
    
    # Log logout event
    await log_auth_event(
        event_type="logout",
        user_id=str(current_user.id)
    )
    
    return {"message": "Successfully logged out"}

# Dependency to get current user
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    
    token_payload = auth_service.verify_token(credentials)
    
    user = db.query(User).filter(User.id == token_payload.sub).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    if not user.is_active:
        raise HTTPException(status_code=403, detail="User account is disabled")
    
    return user
```

---

## 2. Authorization System

### 2.1 Role-Based Access Control (RBAC)

```python
from enum import Enum
from typing import List, Set
from functools import wraps

class Role(Enum):
    """System roles"""
    ADMIN = "admin"
    USER = "user"
    PREMIUM_USER = "premium_user"
    API_USER = "api_user"

class Permission(Enum):
    """System permissions"""
    
    # Channel permissions
    CHANNEL_CREATE = "channel:create"
    CHANNEL_READ = "channel:read"
    CHANNEL_UPDATE = "channel:update"
    CHANNEL_DELETE = "channel:delete"
    
    # Video permissions
    VIDEO_CREATE = "video:create"
    VIDEO_READ = "video:read"
    VIDEO_UPDATE = "video:update"
    VIDEO_DELETE = "video:delete"
    VIDEO_PUBLISH = "video:publish"
    
    # Analytics permissions
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_EXPORT = "analytics:export"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"

# Role-Permission mapping
ROLE_PERMISSIONS = {
    Role.USER: {
        Permission.CHANNEL_CREATE,
        Permission.CHANNEL_READ,
        Permission.CHANNEL_UPDATE,
        Permission.VIDEO_CREATE,
        Permission.VIDEO_READ,
        Permission.VIDEO_UPDATE,
        Permission.VIDEO_PUBLISH,
        Permission.ANALYTICS_READ
    },
    Role.PREMIUM_USER: {
        # Inherits all USER permissions
        Permission.CHANNEL_CREATE,
        Permission.CHANNEL_READ,
        Permission.CHANNEL_UPDATE,
        Permission.CHANNEL_DELETE,
        Permission.VIDEO_CREATE,
        Permission.VIDEO_READ,
        Permission.VIDEO_UPDATE,
        Permission.VIDEO_DELETE,
        Permission.VIDEO_PUBLISH,
        Permission.ANALYTICS_READ,
        Permission.ANALYTICS_EXPORT
    },
    Role.ADMIN: {
        # All permissions
        *Permission
    },
    Role.API_USER: {
        # Limited API access
        Permission.CHANNEL_READ,
        Permission.VIDEO_READ,
        Permission.ANALYTICS_READ
    }
}

class AuthorizationService:
    """Handle authorization logic"""
    
    def __init__(self):
        self.role_permissions = ROLE_PERMISSIONS
    
    def has_permission(
        self, 
        user_roles: List[Role], 
        required_permission: Permission
    ) -> bool:
        """Check if user has required permission"""
        
        user_permissions = set()
        for role in user_roles:
            if role in self.role_permissions:
                user_permissions.update(self.role_permissions[role])
        
        return required_permission in user_permissions
    
    def check_resource_ownership(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        db: Session
    ) -> bool:
        """Verify user owns the resource"""
        
        if resource_type == "channel":
            channel = db.query(Channel).filter(
                Channel.id == resource_id,
                Channel.user_id == user_id
            ).first()
            return channel is not None
            
        elif resource_type == "video":
            video = db.query(Video).filter(
                Video.id == resource_id
            ).first()
            if video:
                # Check if user owns the channel that owns the video
                return self.check_resource_ownership(
                    user_id, "channel", video.channel_id, db
                )
        
        return False

# Authorization decorators
def require_permission(permission: Permission):
    """Decorator to check permissions"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current user from kwargs
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # Check permission
            auth_service = AuthorizationService()
            if not auth_service.has_permission(current_user.roles, permission):
                raise HTTPException(
                    status_code=403, 
                    detail=f"Permission '{permission.value}' required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_resource_ownership(resource_type: str, resource_id_param: str):
    """Decorator to check resource ownership"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            resource_id = kwargs.get(resource_id_param)
            db = kwargs.get('db')
            
            if not all([current_user, resource_id, db]):
                raise HTTPException(status_code=400, detail="Missing required parameters")
            
            auth_service = AuthorizationService()
            if not auth_service.check_resource_ownership(
                str(current_user.id), resource_type, resource_id, db
            ):
                raise HTTPException(
                    status_code=403,
                    detail=f"You don't have access to this {resource_type}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

### 2.2 API Endpoint Authorization Examples

```python
# Example protected endpoints with authorization
@router.get("/channels/{channel_id}")
@require_permission(Permission.CHANNEL_READ)
@require_resource_ownership("channel", "channel_id")
async def get_channel(
    channel_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get channel details - requires ownership"""
    
    channel = db.query(Channel).filter(Channel.id == channel_id).first()
    return {"data": channel.to_dict()}

@router.post("/videos/{video_id}/publish")
@require_permission(Permission.VIDEO_PUBLISH)
@require_resource_ownership("video", "video_id")
async def publish_video(
    video_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Publish video - requires ownership and publish permission"""
    
    # Publishing logic here
    pass

@router.get("/admin/users")
@require_permission(Permission.ADMIN_USERS)
async def list_all_users(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Admin endpoint - requires admin permission"""
    
    users = db.query(User).all()
    return {"data": [user.to_dict() for user in users]}
```

### 2.3 Dynamic Permission Checking

```python
class DynamicAuthorizationService:
    """Handle complex authorization scenarios"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def can_create_channel(self, user: User) -> Tuple[bool, Optional[str]]:
        """Check if user can create more channels"""
        
        # Check user's subscription level
        channel_limit = {
            "free": 1,
            "basic": 5,
            "premium": 25,
            "enterprise": 100
        }.get(user.subscription_tier, 1)
        
        # Count existing channels
        current_channels = self.db.query(Channel).filter(
            Channel.user_id == user.id,
            Channel.status != "deleted"
        ).count()
        
        if current_channels >= channel_limit:
            return False, f"Channel limit reached ({channel_limit} channels for {user.subscription_tier} tier)"
        
        return True, None
    
    def can_generate_video(self, user: User, channel: Channel) -> Tuple[bool, Optional[str]]:
        """Check if user can generate more videos"""
        
        # Check daily video generation limit
        today = datetime.utcnow().date()
        daily_limit = {
            "free": 5,
            "basic": 20,
            "premium": 100,
            "enterprise": 500
        }.get(user.subscription_tier, 5)
        
        # Count today's videos
        todays_videos = self.db.query(Video).filter(
            Video.channel_id == channel.id,
            Video.created_at >= today,
            Video.created_at < today + timedelta(days=1)
        ).count()
        
        if todays_videos >= daily_limit:
            return False, f"Daily video limit reached ({daily_limit} videos/day for {user.subscription_tier} tier)"
        
        # Check if channel is active
        if channel.status != "active":
            return False, "Channel is not active"
        
        # Check if user has enough credits
        if user.video_credits <= 0:
            return False, "Insufficient video credits"
        
        return True, None
    
    def get_user_permissions(self, user: User) -> Dict[str, bool]:
        """Get all user permissions as a dictionary"""
        
        auth_service = AuthorizationService()
        all_permissions = {}
        
        for permission in Permission:
            all_permissions[permission.value] = auth_service.has_permission(
                user.roles, permission
            )
        
        # Add dynamic permissions
        can_create_channel, _ = self.can_create_channel(user)
        all_permissions["dynamic:create_channel"] = can_create_channel
        
        return all_permissions

# Usage in endpoints
@router.post("/channels")
async def create_channel(
    request: ChannelCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create channel with dynamic authorization"""
    
    # Check static permission
    if not AuthorizationService().has_permission(current_user.roles, Permission.CHANNEL_CREATE):
        raise HTTPException(status_code=403, detail="No permission to create channels")
    
    # Check dynamic authorization
    dynamic_auth = DynamicAuthorizationService(db)
    can_create, reason = dynamic_auth.can_create_channel(current_user)
    
    if not can_create:
        raise HTTPException(status_code=403, detail=reason)
    
    # Create channel
    channel = Channel(**request.dict(), user_id=current_user.id)
    db.add(channel)
    db.commit()
    
    return {"data": channel.to_dict()}
```

---

## 3. Rate Limiting Implementation

### 3.1 Rate Limiter Core

```python
import asyncio
from typing import Optional, Dict, Tuple
import time
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"

@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    
    # Default limits by tier
    LIMITS = {
        "anonymous": {
            "requests_per_minute": 10,
            "requests_per_hour": 100,
            "requests_per_day": 500
        },
        "free": {
            "requests_per_minute": 30,
            "requests_per_hour": 500,
            "requests_per_day": 5000,
            "videos_per_day": 5
        },
        "basic": {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "requests_per_day": 20000,
            "videos_per_day": 20
        },
        "premium": {
            "requests_per_minute": 300,
            "requests_per_hour": 5000,
            "requests_per_day": 100000,
            "videos_per_day": 100
        },
        "enterprise": {
            "requests_per_minute": 1000,
            "requests_per_hour": 20000,
            "requests_per_day": 500000,
            "videos_per_day": 500
        }
    }
    
    # Endpoint-specific limits
    ENDPOINT_LIMITS = {
        "/api/v1/videos/generate": {
            "requests_per_minute": 5,
            "requests_per_hour": 20
        },
        "/api/v1/auth/login": {
            "requests_per_minute": 5,
            "requests_per_hour": 20
        },
        "/api/v1/analytics/export": {
            "requests_per_minute": 2,
            "requests_per_hour": 10
        }
    }

class RateLimiter:
    """Advanced rate limiting implementation"""
    
    def __init__(self, redis_client: redis.Redis, strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW):
        self.redis = redis_client
        self.strategy = strategy
        self.config = RateLimitConfig()
    
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: int,
        cost: int = 1
    ) -> Tuple[bool, Dict]:
        """Check if request is within rate limit"""
        
        if self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._sliding_window_check(key, limit, window, cost)
        elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._token_bucket_check(key, limit, window, cost)
        else:
            return await self._fixed_window_check(key, limit, window, cost)
    
    async def _sliding_window_check(
        self,
        key: str,
        limit: int,
        window: int,
        cost: int
    ) -> Tuple[bool, Dict]:
        """Sliding window rate limiting"""
        
        now = time.time()
        pipeline = self.redis.pipeline()
        
        # Remove old entries
        pipeline.zremrangebyscore(key, 0, now - window)
        
        # Count current requests in window
        pipeline.zcard(key)
        
        # Add current request
        pipeline.zadd(key, {str(now): now})
        
        # Set expiry
        pipeline.expire(key, window)
        
        results = pipeline.execute()
        current_requests = results[1]
        
        if current_requests + cost > limit:
            # Calculate when the oldest request will expire
            oldest = self.redis.zrange(key, 0, 0, withscores=True)
            if oldest:
                reset_time = oldest[0][1] + window
            else:
                reset_time = now + window
            
            return False, {
                "limit": limit,
                "remaining": max(0, limit - current_requests),
                "reset": int(reset_time),
                "retry_after": int(reset_time - now)
            }
        
        return True, {
            "limit": limit,
            "remaining": limit - current_requests - cost,
            "reset": int(now + window)
        }
    
    async def _token_bucket_check(
        self,
        key: str,
        capacity: int,
        refill_rate: int,
        cost: int
    ) -> Tuple[bool, Dict]:
        """Token bucket rate limiting"""
        
        bucket_key = f"bucket:{key}"
        now = time.time()
        
        # Get current bucket state
        bucket_data = self.redis.hgetall(bucket_key)
        
        if not bucket_data:
            # Initialize bucket
            tokens = capacity
            last_refill = now
        else:
            tokens = float(bucket_data.get('tokens', capacity))
            last_refill = float(bucket_data.get('last_refill', now))
            
            # Calculate tokens to add
            time_passed = now - last_refill
            tokens_to_add = time_passed * refill_rate
            tokens = min(capacity, tokens + tokens_to_add)
        
        if tokens >= cost:
            # Consume tokens
            tokens -= cost
            
            # Update bucket
            self.redis.hset(bucket_key, mapping={
                'tokens': tokens,
                'last_refill': now
            })
            self.redis.expire(bucket_key, 3600)  # 1 hour expiry
            
            return True, {
                "limit": capacity,
                "remaining": int(tokens),
                "refill_rate": refill_rate
            }
        else:
            # Calculate wait time
            tokens_needed = cost - tokens
            wait_time = tokens_needed / refill_rate
            
            return False, {
                "limit": capacity,
                "remaining": int(tokens),
                "retry_after": int(wait_time),
                "refill_rate": refill_rate
            }
    
    async def _fixed_window_check(
        self,
        key: str,
        limit: int,
        window: int,
        cost: int
    ) -> Tuple[bool, Dict]:
        """Fixed window rate limiting"""
        
        now = int(time.time())
        window_start = now - (now % window)
        window_key = f"{key}:{window_start}"
        
        # Increment counter
        current = self.redis.incrby(window_key, cost)
        
        # Set expiry on first request
        if current == cost:
            self.redis.expire(window_key, window)
        
        if current > limit:
            return False, {
                "limit": limit,
                "remaining": max(0, limit - current + cost),
                "reset": window_start + window,
                "retry_after": window_start + window - now
            }
        
        return True, {
            "limit": limit,
            "remaining": limit - current,
            "reset": window_start + window
        }
    
    def get_user_limits(self, user_tier: str) -> Dict:
        """Get rate limits for user tier"""
        
        return self.config.LIMITS.get(user_tier, self.config.LIMITS["anonymous"])
    
    def get_endpoint_limits(self, endpoint: str) -> Optional[Dict]:
        """Get endpoint-specific limits"""
        
        return self.config.ENDPOINT_LIMITS.get(endpoint)
```

### 3.2 Rate Limiting Middleware

```python
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import hashlib

class RateLimitMiddleware:
    """FastAPI rate limiting middleware"""
    
    def __init__(self, app, rate_limiter: RateLimiter):
        self.app = app
        self.rate_limiter = rate_limiter
    
    async def __call__(self, request: Request, call_next):
        """Process rate limiting for each request"""
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
        
        # Get user identification
        user_id = await self._get_user_id(request)
        user_tier = await self._get_user_tier(user_id)
        
        # Get applicable limits
        endpoint_limits = self.rate_limiter.get_endpoint_limits(request.url.path)
        tier_limits = self.rate_limiter.get_user_limits(user_tier)
        
        # Check endpoint-specific limits first
        if endpoint_limits:
            for period, limit in endpoint_limits.items():
                window = self._parse_period(period)
                key = f"endpoint:{request.url.path}:{user_id}:{period}"
                
                allowed, info = await self.rate_limiter.check_rate_limit(
                    key, limit, window
                )
                
                if not allowed:
                    return self._rate_limit_response(info)
        
        # Check tier-based limits
        for period, limit in tier_limits.items():
            if period.startswith("requests_"):
                window = self._parse_period(period)
                key = f"tier:{user_tier}:{user_id}:{period}"
                
                allowed, info = await self.rate_limiter.check_rate_limit(
                    key, limit, window
                )
                
                if not allowed:
                    return self._rate_limit_response(info)
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        # Add rate limit info to headers
        if hasattr(response, 'headers'):
            response.headers["X-RateLimit-Limit"] = str(tier_limits.get("requests_per_minute", 0))
            response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", 0))
            response.headers["X-RateLimit-Reset"] = str(info.get("reset", 0))
        
        return response
    
    async def _get_user_id(self, request: Request) -> str:
        """Extract user ID from request"""
        
        # Try to get from JWT token
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                token = auth_header.split(" ")[1]
                payload = jwt.decode(token, AuthConfig.JWT_SECRET_KEY, algorithms=[AuthConfig.JWT_ALGORITHM])
                return payload.get("sub", "anonymous")
            except:
                pass
        
        # Fallback to IP address
        client_ip = request.client.host
        return hashlib.sha256(client_ip.encode()).hexdigest()
    
    async def _get_user_tier(self, user_id: str) -> str:
        """Get user's subscription tier"""
        
        if user_id == "anonymous":
            return "anonymous"
        
        # Look up user tier in cache/database
        cached_tier = self.rate_limiter.redis.get(f"user_tier:{user_id}")
        if cached_tier:
            return cached_tier
        
        # Default to free tier
        return "free"
    
    def _parse_period(self, period: str) -> int:
        """Convert period string to seconds"""
        
        if "minute" in period:
            return 60
        elif "hour" in period:
            return 3600
        elif "day" in period:
            return 86400
        else:
            return 60  # Default to 1 minute
    
    def _rate_limit_response(self, info: Dict) -> Response:
        """Create rate limit error response"""
        
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "status": 429,
                    "code": "RATE_LIMIT_EXCEEDED",
                    "title": "Too Many Requests",
                    "detail": "Rate limit exceeded. Please retry after some time.",
                    "meta": {
                        "limit": info.get("limit"),
                        "remaining": info.get("remaining", 0),
                        "reset": info.get("reset"),
                        "retry_after": info.get("retry_after")
                    }
                }
            },
            headers={
                "X-RateLimit-Limit": str(info.get("limit", 0)),
                "X-RateLimit-Remaining": str(info.get("remaining", 0)),
                "X-RateLimit-Reset": str(info.get("reset", 0)),
                "Retry-After": str(info.get("retry_after", 60))
            }
        )
```

### 3.3 Cost-Based Rate Limiting

```python
class CostBasedRateLimiter:
    """Rate limiting based on operation cost"""
    
    # Operation costs
    OPERATION_COSTS = {
        "video_generate": 50,
        "channel_create": 10,
        "analytics_export": 20,
        "bulk_operation": 100,
        "api_read": 1,
        "api_write": 5
    }
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
    
    async def check_operation_limit(
        self,
        user_id: str,
        operation: str,
        user_tier: str
    ) -> Tuple[bool, Optional[str]]:
        """Check if user can perform costly operation"""
        
        cost = self.OPERATION_COSTS.get(operation, 1)
        
        # Get user's daily token budget
        daily_budget = {
            "free": 100,
            "basic": 500,
            "premium": 2000,
            "enterprise": 10000
        }.get(user_tier, 100)
        
        # Check token bucket
        key = f"tokens:{user_id}:daily"
        allowed, info = await self.rate_limiter.check_rate_limit(
            key=key,
            limit=daily_budget,
            window=86400,  # 24 hours
            cost=cost
        )
        
        if not allowed:
            return False, f"Daily operation budget exceeded. Cost: {cost}, Remaining: {info['remaining']}"
        
        return True, None

# Usage example
@router.post("/videos/generate")
async def generate_video(
    request: VideoGenerateRequest,
    current_user: User = Depends(get_current_user),
    rate_limiter: CostBasedRateLimiter = Depends(get_cost_rate_limiter)
):
    """Generate video with cost-based rate limiting"""
    
    # Check operation limit
    allowed, reason = await rate_limiter.check_operation_limit(
        user_id=str(current_user.id),
        operation="video_generate",
        user_tier=current_user.subscription_tier
    )
    
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)
    
    # Proceed with video generation
    pass
```

---

## 4. Security Best Practices

### 4.1 Token Security

```python
class TokenSecurityConfig:
    """Enhanced token security configuration"""
    
    # Token rotation
    ROTATE_REFRESH_TOKENS = True
    MAX_REFRESH_TOKEN_REUSE = 2
    
    # Token binding
    BIND_TOKEN_TO_IP = False  # Can break with mobile users
    BIND_TOKEN_TO_USER_AGENT = True
    
    # Additional security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
    }

# Middleware to add security headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    for header, value in TokenSecurityConfig.SECURITY_HEADERS.items():
        response.headers[header] = value
    return response
```

### 4.2 Audit Logging

```python
class AuthAuditLogger:
    """Log authentication and authorization events"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def log_auth_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        details: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Log authentication event"""
        
        event = AuthAuditLog(
            event_type=event_type,
            user_id=user_id,
            details=json.dumps(details or {}),
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow()
        )
        
        self.db.add(event)
        await self.db.commit()
        
        # Alert on suspicious events
        if event_type in ["login_failed_multiple", "token_theft_detected", "privilege_escalation"]:
            await self.send_security_alert(event)
    
    async def send_security_alert(self, event: AuthAuditLog):
        """Send security alert for suspicious events"""
        
        alert_message = f"""
        Security Alert: {event.event_type}
        
        User ID: {event.user_id}
        IP Address: {event.ip_address}
        Timestamp: {event.timestamp}
        Details: {event.details}
        """
        
        # Send to monitoring system or security team
        await send_notification("security", alert_message)
```

### 4.3 Advanced Security Measures

```python
class AdvancedSecurityService:
    """Advanced security implementations"""
    
    def __init__(self):
        self.suspicious_patterns = {
            "rapid_permission_changes": 5,  # Max 5 permission changes per hour
            "multiple_device_login": 3,     # Max 3 devices simultaneously
            "geo_location_change": 1000     # Max 1000km distance change per hour
        }
    
    async def detect_suspicious_activity(
        self,
        user_id: str,
        activity_type: str,
        metadata: Dict
    ) -> bool:
        """Detect suspicious user activity patterns"""
        
        if activity_type == "login":
            # Check for multiple device login
            active_sessions = await self.get_active_sessions(user_id)
            if len(active_sessions) >= self.suspicious_patterns["multiple_device_login"]:
                await self.flag_suspicious_activity(
                    user_id,
                    "Multiple device login detected",
                    metadata
                )
                return True
        
        elif activity_type == "location_change":
            # Check for impossible travel
            last_location = await self.get_last_location(user_id)
            if last_location:
                distance = self.calculate_distance(
                    last_location,
                    metadata["current_location"]
                )
                time_diff = metadata["timestamp"] - last_location["timestamp"]
                
                if distance > self.suspicious_patterns["geo_location_change"]:
                    await self.flag_suspicious_activity(
                        user_id,
                        "Impossible travel detected",
                        metadata
                    )
                    return True
        
        return False
    
    async def implement_adaptive_authentication(
        self,
        user: User,
        login_context: Dict
    ) -> Dict[str, Any]:
        """Implement risk-based authentication"""
        
        risk_score = 0
        factors = []
        
        # Check login time
        if not self.is_normal_login_time(user, login_context["timestamp"]):
            risk_score += 20
            factors.append("unusual_login_time")
        
        # Check device
        if not await self.is_known_device(user.id, login_context["device_id"]):
            risk_score += 30
            factors.append("unknown_device")
        
        # Check location
        if not await self.is_known_location(user.id, login_context["ip_address"]):
            risk_score += 25
            factors.append("unknown_location")
        
        # Determine authentication requirements
        if risk_score < 30:
            return {"require_mfa": False, "risk_level": "low"}
        elif risk_score < 60:
            return {"require_mfa": True, "risk_level": "medium"}
        else:
            return {
                "require_mfa": True,
                "require_email_verification": True,
                "risk_level": "high",
                "factors": factors
            }
```

---

## 5. Implementation Checklist

### 5.1 Authentication Implementation

- [ ] JWT token generation with proper expiration
- [ ] Secure password hashing with bcrypt
- [ ] Token refresh mechanism
- [ ] Token revocation and blacklisting
- [ ] Account lockout after failed attempts
- [ ] Security headers implementation
- [ ] Session management
- [ ] Multi-factor authentication (future)

### 5.2 Authorization Implementation

- [ ] RBAC system with roles and permissions
- [ ] Resource ownership verification
- [ ] Dynamic permission checking
- [ ] Permission decorators for endpoints
- [ ] Subscription tier limits
- [ ] Admin access controls
- [ ] API scopes implementation

### 5.3 Rate Limiting Implementation

- [ ] Multiple rate limiting strategies
- [ ] User tier-based limits
- [ ] Endpoint-specific limits
- [ ] Cost-based rate limiting
- [ ] Rate limit headers in responses
- [ ] Graceful degradation
- [ ] Rate limit bypass for admins

### 5.4 Security Monitoring

- [ ] Authentication event logging
- [ ] Failed login tracking
- [ ] Suspicious activity detection
- [ ] Security alerts system
- [ ] Audit trail maintenance
- [ ] Compliance reporting
- [ ] Regular security reviews

---

## Document Control

- **Version**: 1.0
- **Last Updated**: January 2025
- **Owner**: API Development Engineer
- **Security Review**: Required quarterly
- **Classification**: Confidential

**Security Notice**: This document contains sensitive security implementation details. Handle with appropriate care and limit distribution to authorized personnel only.

**Next Steps**:
1. Implement core authentication flow
2. Set up RBAC system
3. Configure rate limiting
4. Enable audit logging
5. Conduct security testing
6. Schedule penetration testing