"""
Authentication Tests
Owner: QA Engineer #1
"""

import pytest
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.schemas.auth import UserRegister, PasswordChange
from app.services.auth_service import AuthService
from app.repositories.user_repository import UserRepository
from app.models.user import User, SubscriptionTier


@pytest.mark.unit
@pytest.mark.auth
class TestAuthService:
    """Test authentication service functionality."""
    
    @pytest.mark.asyncio
    async def test_password_hashing(self, auth_service: AuthService):
        """Test password hashing and verification."""
        password = "test_password_123"
        
        # Hash password
        hashed = auth_service.get_password_hash(password)
        assert hashed != password
        assert len(hashed) > 0
        
        # Verify password
        assert auth_service.verify_password(password, hashed)
        assert not auth_service.verify_password("wrong_password", hashed)
    
    @pytest.mark.asyncio
    async def test_token_creation(self, auth_service: AuthService):
        """Test JWT token creation."""
        data = {"sub": "test_user", "email": "test@example.com"}
        
        # Create access token
        access_token = auth_service.create_access_token(data)
        assert access_token is not None
        assert isinstance(access_token, str)
        
        # Create refresh token
        refresh_token = auth_service.create_refresh_token(data)
        assert refresh_token is not None
        assert isinstance(refresh_token, str)
        assert refresh_token != access_token
    
    @pytest.mark.asyncio
    async def test_user_registration_success(self, auth_service: AuthService):
        """Test successful user registration."""
        user_data = UserRegister(
            email="newuser@example.com",
            username="newuser",
            password="securepassword123",
            full_name="New User"
        )
        
        user = await auth_service.register_user(user_data)
        
        assert user.email == user_data.email.lower()
        assert user.username == user_data.username.lower()
        assert user.full_name == user_data.full_name
        assert user.is_active is True
        assert user.is_verified is False
        assert user.subscription_tier == SubscriptionTier.FREE
        assert user.hashed_password != user_data.password
    
    @pytest.mark.asyncio
    async def test_user_registration_duplicate_email(self, auth_service: AuthService):
        """Test user registration with duplicate email."""
        user_data = UserRegister(
            email="duplicate@example.com",
            username="user1",
            password="password123",
            full_name="User One"
        )
        
        # Register first user
        await auth_service.register_user(user_data)
        
        # Try to register another user with same email
        user_data2 = UserRegister(
            email="duplicate@example.com",
            username="user2",
            password="password123",
            full_name="User Two"
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await auth_service.register_user(user_data2)
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Email already registered" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_user_registration_duplicate_username(self, auth_service: AuthService):
        """Test user registration with duplicate username."""
        user_data = UserRegister(
            email="user1@example.com",
            username="duplicateuser",
            password="password123",
            full_name="User One"
        )
        
        # Register first user
        await auth_service.register_user(user_data)
        
        # Try to register another user with same username
        user_data2 = UserRegister(
            email="user2@example.com",
            username="duplicateuser",
            password="password123",
            full_name="User Two"
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await auth_service.register_user(user_data2)
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Username already taken" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_user_registration_weak_password(self, auth_service: AuthService):
        """Test user registration with weak password."""
        user_data = UserRegister(
            email="test@example.com",
            username="testuser",
            password="weak",  # Too short
            full_name="Test User"
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await auth_service.register_user(user_data)
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "at least 8 characters" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_user_login_success(self, auth_service: AuthService, test_user: User):
        """Test successful user login."""
        token = await auth_service.login_user("test@example.com", "testpassword123")
        
        assert token.access_token is not None
        assert token.refresh_token is not None
        assert token.token_type == "bearer"
    
    @pytest.mark.asyncio
    async def test_user_login_wrong_password(self, auth_service: AuthService, test_user: User):
        """Test user login with wrong password."""
        with pytest.raises(HTTPException) as exc_info:
            await auth_service.login_user("test@example.com", "wrongpassword")
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Incorrect email/username or password" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_user_login_nonexistent_user(self, auth_service: AuthService):
        """Test user login with non-existent user."""
        with pytest.raises(HTTPException) as exc_info:
            await auth_service.login_user("nonexistent@example.com", "password123")
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Incorrect email/username or password" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_token_verification(self, auth_service: AuthService, test_user: User):
        """Test JWT token verification."""
        token = await auth_service.login_user("test@example.com", "testpassword123")
        
        # Verify access token
        payload = await auth_service.verify_token(token.access_token)
        assert payload["sub"] == test_user.id
        assert payload["email"] == test_user.email
    
    @pytest.mark.asyncio
    async def test_get_current_user(self, auth_service: AuthService, test_user: User):
        """Test getting current user from token."""
        token = await auth_service.login_user("test@example.com", "testpassword123")
        
        current_user = await auth_service.get_current_user(token.access_token)
        assert current_user.id == test_user.id
        assert current_user.email == test_user.email
    
    @pytest.mark.asyncio
    async def test_refresh_token(self, auth_service: AuthService, test_user: User):
        """Test access token refresh."""
        token = await auth_service.login_user("test@example.com", "testpassword123")
        
        # Refresh access token
        new_token = await auth_service.refresh_access_token(token.refresh_token)
        
        assert new_token.access_token != token.access_token
        assert new_token.refresh_token == token.refresh_token
        assert new_token.token_type == "bearer"
    
    @pytest.mark.asyncio
    async def test_change_password(self, auth_service: AuthService, test_user: User):
        """Test password change."""
        await auth_service.change_password(
            test_user.id,
            "testpassword123",
            "newpassword456"
        )
        
        # Try to login with new password
        token = await auth_service.login_user("test@example.com", "newpassword456")
        assert token is not None
        
        # Old password should not work
        with pytest.raises(HTTPException):
            await auth_service.login_user("test@example.com", "testpassword123")
    
    @pytest.mark.asyncio
    async def test_change_password_wrong_current(self, auth_service: AuthService, test_user: User):
        """Test password change with wrong current password."""
        with pytest.raises(HTTPException) as exc_info:
            await auth_service.change_password(
                test_user.id,
                "wrongpassword",
                "newpassword456"
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Current password is incorrect" in str(exc_info.value.detail)


@pytest.mark.unit
@pytest.mark.auth
class TestUserRepository:
    """Test user repository functionality."""
    
    @pytest.mark.asyncio
    async def test_create_user(self, user_repo: UserRepository):
        """Test user creation."""
        user_data = UserRegister(
            email="repo@example.com",
            username="repouser",
            password="password123",
            full_name="Repo User"
        )
        
        hashed_password = "hashed_password"
        user = await user_repo.create_user(user_data, hashed_password)
        
        assert user.email == user_data.email.lower()
        assert user.username == user_data.username.lower()
        assert user.hashed_password == hashed_password
        assert user.is_active is True
    
    @pytest.mark.asyncio
    async def test_get_by_email(self, user_repo: UserRepository, test_user: User):
        """Test get user by email."""
        user = await user_repo.get_by_email("test@example.com")
        assert user is not None
        assert user.id == test_user.id
        
        # Test case insensitivity
        user = await user_repo.get_by_email("TEST@EXAMPLE.COM")
        assert user is not None
        assert user.id == test_user.id
    
    @pytest.mark.asyncio
    async def test_get_by_username(self, user_repo: UserRepository, test_user: User):
        """Test get user by username."""
        user = await user_repo.get_by_username("testuser")
        assert user is not None
        assert user.id == test_user.id
        
        # Test case insensitivity
        user = await user_repo.get_by_username("TESTUSER")
        assert user is not None
        assert user.id == test_user.id
    
    @pytest.mark.asyncio
    async def test_get_by_email_or_username(self, user_repo: UserRepository, test_user: User):
        """Test get user by email or username."""
        # Test with email
        user = await user_repo.get_by_email_or_username("test@example.com")
        assert user is not None
        assert user.id == test_user.id
        
        # Test with username
        user = await user_repo.get_by_email_or_username("testuser")
        assert user is not None
        assert user.id == test_user.id
    
    @pytest.mark.asyncio
    async def test_update_last_login(self, user_repo: UserRepository, test_user: User):
        """Test updating last login timestamp."""
        success = await user_repo.update_last_login(test_user.id)
        assert success is True
        
        # Verify update
        updated_user = await user_repo.get_by_id(test_user.id)
        assert updated_user.last_login is not None
        assert updated_user.last_login > test_user.created_at
    
    @pytest.mark.asyncio
    async def test_email_exists(self, user_repo: UserRepository, test_user: User):
        """Test email existence check."""
        assert await user_repo.email_exists("test@example.com") is True
        assert await user_repo.email_exists("nonexistent@example.com") is False
    
    @pytest.mark.asyncio
    async def test_username_exists(self, user_repo: UserRepository, test_user: User):
        """Test username existence check."""
        assert await user_repo.username_exists("testuser") is True
        assert await user_repo.username_exists("nonexistent") is False