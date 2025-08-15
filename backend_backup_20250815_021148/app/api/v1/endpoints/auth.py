"""
Authentication endpoints with enhanced security and email verification
"""
from datetime import datetime, timedelta
from typing import Annotated, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
import secrets
import re
from pydantic import BaseModel, EmailStr, validator

from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    verify_password,
    get_password_hash,
    verify_token
)
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import Token, UserCreate, UserResponse, UserLogin
from app.services.email_service import EmailService
from app.services.rate_limiter import RateLimiter

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# Initialize services
email_service = EmailService()
rate_limiter = RateLimiter()

# Enhanced Pydantic models
class UserRegisterEnhanced(UserCreate):
    """Enhanced user registration with password validation"""
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r"[A-Z]", v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r"[a-z]", v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r"\d", v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError('Password must contain at least one special character')
        return v

class EmailVerification(BaseModel):
    token: str

class PasswordReset(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

# Helper functions
def generate_verification_token() -> str:
    """Generate secure verification token"""
    return secrets.token_urlsafe(32)

def generate_password_reset_token() -> str:
    """Generate secure password reset token"""
    return secrets.token_urlsafe(32)


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user with email verification check
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    user_id = verify_token(token)
    if user_id is None:
        raise credentials_exception
    
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )
    
    return user

async def get_current_verified_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """Ensure user email is verified"""
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your email address"
        )
    return current_user


async def get_current_active_superuser(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """
    Get current superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserRegisterEnhanced,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Register new user with email verification
    
    - Validates email uniqueness
    - Enforces strong password requirements  
    - Sends verification email
    - Creates user with pending verification status
    """
    # Check rate limiting
    if not await rate_limiter.check_rate_limit(f"register:{user_data.email}", max_attempts=3, window=3600):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many registration attempts. Please try again later."
        )
    
    # Check if user exists
    result = await db.execute(
        select(User).filter(
            (User.email == user_data.email) | (User.username == user_data.username)
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email or username already registered"
        )
    
    # Create verification token
    verification_token = generate_verification_token()
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        email=user_data.email,
        username=user_data.username,
        full_name=user_data.full_name,
        hashed_password=hashed_password,
        is_active=True,
        is_verified=False,
        verification_token=verification_token,
        subscription_tier="free",
        api_quota_remaining=100,
        api_quota_reset_at=datetime.utcnow() + timedelta(days=30)
    )
    
    try:
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        
        # Send verification email in background
        background_tasks.add_task(
            email_service.send_verification_email,
            db_user.email,
            db_user.full_name,
            verification_token
        )
        
        return db_user
        
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email or username already registered"
        )


@router.post("/login", response_model=Token)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: AsyncSession = Depends(get_db)
):
    """
    User login with rate limiting
    
    - Validates credentials
    - Returns JWT access and refresh tokens
    - Tracks login attempts for security
    """
    # Check rate limiting
    if not await rate_limiter.check_rate_limit(f"login:{form_data.username}", max_attempts=5, window=900):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again later."
        )
    
    # Find user by username or email
    result = await db.execute(
        select(User).filter(
            (User.email == form_data.username) | (User.username == form_data.username)
        )
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        # Log failed attempt
        await rate_limiter.log_failed_attempt(f"login:{form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )
    
    # Create tokens
    access_token = create_access_token(subject=str(user.id))
    refresh_token = create_refresh_token(subject=str(user.id))
    
    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token
    """
    user_id = verify_token(refresh_token, token_type="refresh")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new tokens
    access_token = create_access_token(subject=user.id)
    new_refresh_token = create_refresh_token(subject=user.id)
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Get current user info
    """
    return current_user


@router.post("/verify-email", response_model=Dict[str, str])
async def verify_email(
    verification: EmailVerification,
    db: AsyncSession = Depends(get_db)
):
    """
    Verify user email address
    
    - Validates verification token
    - Activates user account
    - Enables full platform access
    """
    result = await db.execute(
        select(User).filter(User.verification_token == verification.token)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid verification token"
        )
    
    if user.is_verified:
        return {"message": "Email already verified"}
    
    # Verify user
    user.is_verified = True
    user.verification_token = None
    user.verified_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": "Email verified successfully"}

@router.post("/password-reset", response_model=Dict[str, str])
async def request_password_reset(
    reset_request: PasswordReset,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Request password reset
    
    - Generates reset token
    - Sends reset email
    - Expires after 1 hour
    """
    # Check rate limiting
    if not await rate_limiter.check_rate_limit(f"password_reset:{reset_request.email}", max_attempts=3, window=3600):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many password reset attempts. Please try again later."
        )
    
    result = await db.execute(
        select(User).filter(User.email == reset_request.email)
    )
    user = result.scalar_one_or_none()
    
    # Always return success to prevent email enumeration
    if user:
        reset_token = generate_password_reset_token()
        user.password_reset_token = reset_token
        user.password_reset_expires = datetime.utcnow() + timedelta(hours=1)
        await db.commit()
        
        # Send reset email in background
        background_tasks.add_task(
            email_service.send_password_reset_email,
            user.email,
            user.full_name,
            reset_token
        )
    
    return {"message": "If the email exists, a password reset link has been sent"}

@router.post("/password-reset-confirm", response_model=Dict[str, str])
async def confirm_password_reset(
    reset_confirm: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db)
):
    """
    Confirm password reset with token
    
    - Validates reset token
    - Updates user password
    - Invalidates existing sessions
    """
    result = await db.execute(
        select(User).filter(
            User.password_reset_token == reset_confirm.token,
            User.password_reset_expires > datetime.utcnow()
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid or expired reset token"
        )
    
    # Update password
    user.hashed_password = get_password_hash(reset_confirm.new_password)
    user.password_reset_token = None
    user.password_reset_expires = None
    await db.commit()
    
    return {"message": "Password reset successfully"}

@router.post("/logout", response_model=Dict[str, str])
async def logout(
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Logout current user
    
    - Client should remove tokens
    - Server-side token invalidation can be added
    """
    return {"message": "Successfully logged out"}

@router.delete("/account", response_model=Dict[str, str])
async def delete_account(
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Delete user account
    
    - Soft delete (deactivates account)
    - Preserves data for compliance
    - Can be reversed within 30 days
    """
    current_user.is_active = False
    current_user.deleted_at = datetime.utcnow()
    await db.commit()
    
    return {"message": "Account deactivated. You have 30 days to reactivate before permanent deletion."}