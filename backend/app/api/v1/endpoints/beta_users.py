"""
Beta User Management Endpoints
Handles beta user signups, onboarding, and special features
"""
from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from pydantic import BaseModel, EmailStr, Field
import secrets
import logging

from app.db.session import get_db
from app.models.user import User
from app.core.security import get_password_hash, create_access_token
from app.services.email_service import send_email
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/beta", tags=["beta"])

class BetaSignupRequest(BaseModel):
    """Beta user signup request"""
    full_name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    company: Optional[str] = Field(None, max_length=100)
    use_case: str = Field(..., min_length=10, max_length=500)
    expected_volume: str = Field(..., description="Expected videos per month")
    referral_source: Optional[str] = Field(None, description="How did you hear about us?")

class BetaSignupResponse(BaseModel):
    """Beta signup response"""
    message: str
    user_id: int
    api_key: str
    dashboard_url: str

class BetaUserStats(BaseModel):
    """Beta user statistics"""
    total_signups: int
    active_users: int
    total_videos_generated: int
    average_videos_per_user: float
    top_use_cases: List[str]

@router.post("/signup", response_model=BetaSignupResponse, status_code=status.HTTP_201_CREATED)
async def beta_signup(
    request: BetaSignupRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Beta user signup endpoint
    - Creates user account with beta privileges
    - Sends welcome email with credentials
    - Generates API key
    """
    try:
        # Check if email already exists
        existing_user = await db.execute(
            select(User).where(User.email == request.email)
        )
        if existing_user.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Generate temporary password
        temp_password = secrets.token_urlsafe(12)
        
        # Create beta user
        new_user = User(
            email=request.email,
            full_name=request.full_name,
            hashed_password=get_password_hash(temp_password),
            is_active=True,
            is_superuser=False,
            is_beta=True,  # Beta user flag
            company=request.company,
            use_case=request.use_case,
            expected_volume=request.expected_volume,
            referral_source=request.referral_source,
            created_at=datetime.utcnow(),
            # Beta user benefits
            api_rate_limit=5000,  # Higher rate limit
            max_channels=10,  # More channels allowed
            max_videos_per_day=100,  # Higher daily limit
            free_credits=50.0,  # $50 free credits
        )
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        # Generate API key
        api_key = f"beta_{secrets.token_urlsafe(32)}"
        
        # Store API key (in production, store hashed version)
        new_user.api_key = api_key
        await db.commit()
        
        # Send welcome email in background
        background_tasks.add_task(
            send_beta_welcome_email,
            email=request.email,
            full_name=request.full_name,
            temp_password=temp_password,
            api_key=api_key
        )
        
        # Track signup event
        logger.info(f"New beta user signup: {request.email}")
        
        return BetaSignupResponse(
            message="Beta access granted! Check your email for login credentials.",
            user_id=new_user.id,
            api_key=api_key,
            dashboard_url=f"{settings.FRONTEND_URL}/dashboard?beta=true"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Beta signup error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process beta signup"
        )

@router.get("/stats", response_model=BetaUserStats)
async def get_beta_stats(
    db: AsyncSession = Depends(get_db)
):
    """Get beta user statistics"""
    try:
        # Get total beta signups
        total_signups = await db.execute(
            select(func.count(User.id)).where(User.is_beta == True)
        )
        total = total_signups.scalar() or 0
        
        # Get active beta users (logged in last 7 days)
        active_cutoff = datetime.utcnow() - timedelta(days=7)
        active_users = await db.execute(
            select(func.count(User.id)).where(
                and_(
                    User.is_beta == True,
                    User.last_login > active_cutoff
                )
            )
        )
        active = active_users.scalar() or 0
        
        # Get video statistics (simplified for now)
        total_videos = 100  # Placeholder
        avg_videos = total_videos / max(total, 1)
        
        # Get top use cases
        use_cases_result = await db.execute(
            select(User.use_case, func.count(User.id).label('count'))
            .where(User.is_beta == True)
            .group_by(User.use_case)
            .order_by(func.count(User.id).desc())
            .limit(5)
        )
        top_use_cases = [row[0] for row in use_cases_result if row[0]]
        
        return BetaUserStats(
            total_signups=total,
            active_users=active,
            total_videos_generated=total_videos,
            average_videos_per_user=avg_videos,
            top_use_cases=top_use_cases
        )
        
    except Exception as e:
        logger.error(f"Failed to get beta stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve beta statistics"
        )

@router.post("/feedback")
async def submit_beta_feedback(
    feedback: str,
    rating: int = Field(..., ge=1, le=5),
    user_id: int = None,
    db: AsyncSession = Depends(get_db)
):
    """Submit beta user feedback"""
    try:
        # Store feedback in database (simplified for now)
        logger.info(f"Beta feedback received - Rating: {rating}, Feedback: {feedback}")
        
        return {"message": "Thank you for your feedback!", "status": "received"}
        
    except Exception as e:
        logger.error(f"Failed to submit feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )

async def send_beta_welcome_email(
    email: str,
    full_name: str,
    temp_password: str,
    api_key: str
):
    """Send welcome email to beta user"""
    try:
        subject = "Welcome to YTEmpire Beta! 🚀"
        
        body = f"""
        Hi {full_name},
        
        Congratulations! You've been accepted into the YTEmpire Beta Program.
        
        Here are your credentials:
        
        🔑 Login Credentials:
        Email: {email}
        Temporary Password: {temp_password}
        
        🔧 API Access:
        API Key: {api_key}
        API Endpoint: {settings.API_V1_STR}
        
        🎁 Beta Benefits:
        • $50 free credits
        • 5,000 API requests/hour (5x standard)
        • 10 channel limit (2x standard)  
        • 100 videos/day generation
        • Priority support
        • Early access to new features
        
        📚 Getting Started:
        1. Login at: {settings.FRONTEND_URL}/login
        2. Change your password in Settings
        3. Check out the docs: {settings.FRONTEND_URL}/docs
        4. Join our Discord: https://discord.gg/ytempire
        
        Quick Start API Example:
        ```python
        import requests
        
        headers = {{"Authorization": "Bearer {api_key}"}}
        response = requests.post(
            "{settings.API_V1_STR}/videos/generate",
            headers=headers,
            json={{"topic": "Your topic here"}}
        )
        ```
        
        We're excited to have you on board! Please share your feedback as you use the platform.
        
        Best regards,
        The YTEmpire Team
        """
        
        await send_email(
            to_email=email,
            subject=subject,
            body=body
        )
        
        logger.info(f"Welcome email sent to beta user: {email}")
        
    except Exception as e:
        logger.error(f"Failed to send beta welcome email: {str(e)}")

@router.get("/waitlist/count")
async def get_waitlist_count(db: AsyncSession = Depends(get_db)):
    """Get current waitlist count"""
    # Placeholder - in production, track waitlist separately
    return {"count": 47, "message": "people ahead of you"}