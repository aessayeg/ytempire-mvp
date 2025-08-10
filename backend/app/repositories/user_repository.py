"""
User Repository
Owner: Backend Team Lead
"""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload
import uuid
from datetime import datetime

from app.models.user import User
from app.schemas.auth import UserRegister


class UserRepository:
    """Repository for user data operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.db.execute(
            select(User).where(User.email == email.lower())
        )
        return result.scalar_one_or_none()
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        result = await self.db.execute(
            select(User).where(User.username == username.lower())
        )
        return result.scalar_one_or_none()
    
    async def get_by_email_or_username(self, identifier: str) -> Optional[User]:
        """Get user by email or username."""
        result = await self.db.execute(
            select(User).where(
                and_(
                    User.is_active == True,
                    (User.email == identifier.lower()) | (User.username == identifier.lower())
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def create_user(self, user_data: UserRegister, hashed_password: str) -> User:
        """Create a new user."""
        user = User(
            id=str(uuid.uuid4()),
            email=user_data.email.lower(),
            username=user_data.username.lower(),
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            is_active=True,
            is_verified=False,
            subscription_tier=user_data.subscription_tier or "FREE",
            preferences={},
            usage_stats={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user
    
    async def update_last_login(self, user_id: str) -> bool:
        """Update user's last login timestamp."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if user:
            user.last_login = datetime.utcnow()
            user.updated_at = datetime.utcnow()
            await self.db.commit()
            return True
        return False
    
    async def update_password(self, user_id: str, hashed_password: str) -> bool:
        """Update user password."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if user:
            user.hashed_password = hashed_password
            user.updated_at = datetime.utcnow()
            await self.db.commit()
            return True
        return False
    
    async def verify_user(self, user_id: str) -> bool:
        """Mark user as verified."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if user:
            user.is_verified = True
            user.updated_at = datetime.utcnow()
            await self.db.commit()
            return True
        return False
    
    async def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user account."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if user:
            user.is_active = False
            user.updated_at = datetime.utcnow()
            await self.db.commit()
            return True
        return False
    
    async def email_exists(self, email: str) -> bool:
        """Check if email already exists."""
        result = await self.db.execute(
            select(User.id).where(User.email == email.lower())
        )
        return result.scalar_one_or_none() is not None
    
    async def username_exists(self, username: str) -> bool:
        """Check if username already exists."""
        result = await self.db.execute(
            select(User.id).where(User.username == username.lower())
        )
        return result.scalar_one_or_none() is not None