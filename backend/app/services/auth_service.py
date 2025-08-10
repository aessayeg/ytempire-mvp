"""
Authentication Service
Owner: Backend Team Lead
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext
import redis.asyncio as redis
import uuid
import logging

from app.core.config import settings
from app.repositories.user_repository import UserRepository
from app.core.metrics import metrics
from app.schemas.auth import UserRegister, Token
from app.models.user import User

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Redis client for token blacklist
redis_client: Optional[redis.Redis] = None


async def get_redis_client() -> redis.Redis:
    """Get Redis client for token management."""
    global redis_client
    if redis_client is None:
        redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    return redis_client


class AuthService:
    """Authentication service for user management."""
    
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash password."""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4())  # JWT ID for revocation
        })
        
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": str(uuid.uuid4())
        })
        
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    async def authenticate_user(self, identifier: str, password: str) -> Optional[User]:
        """Authenticate user with email/username and password."""
        user = await self.user_repo.get_by_email_or_username(identifier)
        
        if not user:
            return None
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated"
            )
        
        if not self.verify_password(password, user.hashed_password):
            return None
        
        return user
    
    async def register_user(self, user_data: UserRegister) -> User:
        """Register a new user."""
        # Check if email already exists
        if await self.user_repo.email_exists(user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Check if username already exists
        if await self.user_repo.username_exists(user_data.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        # Validate password strength
        if len(user_data.password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long"
            )
        
        # Hash password and create user
        hashed_password = self.get_password_hash(user_data.password)
        user = await self.user_repo.create_user(user_data, hashed_password)
        
        logger.info(f"New user registered: {user.email}")
        return user
    
    async def login_user(self, identifier: str, password: str) -> Token:
        """Login user and return tokens."""
        user = await self.authenticate_user(identifier, password)
        
        if not user:
            # Record failed authentication
            metrics.record_user_auth(success=False)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email/username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Record successful authentication
        metrics.record_user_auth(success=True)
        
        # Create tokens
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = self.create_access_token(
            data={"sub": user.id, "email": user.email, "username": user.username},
            expires_delta=access_token_expires
        )
        
        refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        refresh_token = self.create_refresh_token(
            data={"sub": user.id, "email": user.email},
            expires_delta=refresh_token_expires
        )
        
        # Record token issuance
        metrics.record_jwt_token_issued("access")
        metrics.record_jwt_token_issued("refresh")
        
        # Update last login
        await self.user_repo.update_last_login(user.id)
        
        logger.info(f"User logged in: {user.email}")
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer"
        )
    
    async def refresh_access_token(self, refresh_token: str) -> Token:
        """Refresh access token using refresh token."""
        try:
            # Check if token is blacklisted
            redis_client = await get_redis_client()
            is_blacklisted = await redis_client.get(f"blacklist:{refresh_token}")
            
            if is_blacklisted:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            # Decode refresh token
            payload = jwt.decode(refresh_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            user_id: str = payload.get("sub")
            token_type: str = payload.get("type")
            
            if user_id is None or token_type != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )
            
            # Get user from database
            user = await self.user_repo.get_by_id(user_id)
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive"
                )
            
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Create new access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = self.create_access_token(
            data={"sub": user.id, "email": user.email, "username": user.username},
            expires_delta=access_token_expires
        )
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer"
        )
    
    async def logout_user(self, token: str) -> bool:
        """Logout user by blacklisting token."""
        try:
            # Decode token to get expiration
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            exp = payload.get("exp")
            jti = payload.get("jti")
            
            if exp and jti:
                # Calculate TTL for blacklist (until token expires)
                current_time = datetime.utcnow().timestamp()
                ttl = int(exp - current_time)
                
                if ttl > 0:
                    # Add token to blacklist in Redis
                    redis_client = await get_redis_client()
                    await redis_client.setex(f"blacklist:{token}", ttl, "1")
                    
                    logger.info(f"Token blacklisted: {jti}")
                    return True
            
        except JWTError:
            pass
        
        return False
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            # Check if token is blacklisted
            redis_client = await get_redis_client()
            is_blacklisted = await redis_client.get(f"blacklist:{token}")
            
            if is_blacklisted:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            # Decode token
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            user_id: str = payload.get("sub")
            
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            
            return payload
            
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def get_current_user(self, token: str) -> User:
        """Get current user from token."""
        payload = await self.verify_token(token)
        user_id = payload.get("sub")
        
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is inactive"
            )
        
        return user
    
    async def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Change user password."""
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        if not self.verify_password(current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Validate new password
        if len(new_password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be at least 8 characters long"
            )
        
        # Update password
        new_hashed_password = self.get_password_hash(new_password)
        await self.user_repo.update_password(user_id, new_hashed_password)
        
        logger.info(f"Password changed for user: {user.email}")
        return True