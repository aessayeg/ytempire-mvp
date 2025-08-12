"""
Unit tests for authentication endpoints
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from fastapi import HTTPException
from app.api.v1.endpoints.auth import (
    login, register, get_current_user, verify_email
)
from app.models.user import User
from app.core.security import verify_password, get_password_hash, create_access_token

class TestAuthEndpoints:
    """Test authentication endpoints"""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session"""
        db = AsyncMock()
        return db
    
    @pytest.fixture
    def mock_user(self):
        """Mock user object"""
        user = Mock(spec=User)
        user.id = 1
        user.email = "test@example.com"
        user.full_name = "Test User"
        user.hashed_password = get_password_hash("testpass123")
        user.is_active = True
        user.is_verified = True
        user.is_superuser = False
        return user
    
    @pytest.mark.asyncio
    async def test_register_success(self, mock_db):
        """Test successful user registration"""
        # Arrange
        mock_db.execute.return_value.scalar_one_or_none.return_value = None
        mock_db.add = Mock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()
        
        registration_data = {
            "email": "newuser@example.com",
            "password": "securepass123",
            "full_name": "New User"
        }
        
        # Act
        with patch('app.api.v1.endpoints.auth.get_db', return_value=mock_db):
            with patch('app.api.v1.endpoints.auth.send_verification_email') as mock_send:
                result = await register(registration_data, mock_db)
        
        # Assert
        assert result["email"] == registration_data["email"]
        assert "id" in result
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, mock_db, mock_user):
        """Test registration with duplicate email"""
        # Arrange
        mock_db.execute.return_value.scalar_one_or_none.return_value = mock_user
        
        registration_data = {
            "email": "test@example.com",
            "password": "testpass123",
            "full_name": "Test User"
        }
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await register(registration_data, mock_db)
        
        assert exc_info.value.status_code == 400
        assert "already registered" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_login_success(self, mock_db, mock_user):
        """Test successful login"""
        # Arrange
        mock_db.execute.return_value.scalar_one_or_none.return_value = mock_user
        
        login_data = {
            "username": "test@example.com",
            "password": "testpass123"
        }
        
        # Act
        result = await login(login_data, mock_db)
        
        # Assert
        assert result["access_token"] is not None
        assert result["token_type"] == "bearer"
        assert result["user"]["email"] == mock_user.email
    
    @pytest.mark.asyncio
    async def test_login_invalid_password(self, mock_db, mock_user):
        """Test login with invalid password"""
        # Arrange
        mock_db.execute.return_value.scalar_one_or_none.return_value = mock_user
        
        login_data = {
            "username": "test@example.com",
            "password": "wrongpassword"
        }
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await login(login_data, mock_db)
        
        assert exc_info.value.status_code == 401
        assert "Incorrect email or password" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_login_inactive_user(self, mock_db, mock_user):
        """Test login with inactive user"""
        # Arrange
        mock_user.is_active = False
        mock_db.execute.return_value.scalar_one_or_none.return_value = mock_user
        
        login_data = {
            "username": "test@example.com",
            "password": "testpass123"
        }
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await login(login_data, mock_db)
        
        assert exc_info.value.status_code == 403
        assert "Account is inactive" in str(exc_info.value.detail)
    
    def test_create_access_token(self):
        """Test access token creation"""
        # Arrange
        data = {"sub": "test@example.com", "user_id": 1}
        
        # Act
        token = create_access_token(data)
        
        # Assert
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 50
    
    def test_verify_password(self):
        """Test password verification"""
        # Arrange
        plain_password = "testpass123"
        hashed_password = get_password_hash(plain_password)
        
        # Act & Assert
        assert verify_password(plain_password, hashed_password) is True
        assert verify_password("wrongpass", hashed_password) is False
    
    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self, mock_db, mock_user):
        """Test getting current user with valid token"""
        # Arrange
        token_data = {"sub": mock_user.email, "user_id": mock_user.id}
        token = create_access_token(token_data)
        
        mock_db.execute.return_value.scalar_one_or_none.return_value = mock_user
        
        # Act
        with patch('app.api.v1.endpoints.auth.jwt.decode', return_value=token_data):
            result = await get_current_user(token, mock_db)
        
        # Assert
        assert result.id == mock_user.id
        assert result.email == mock_user.email
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, mock_db):
        """Test getting current user with invalid token"""
        # Arrange
        invalid_token = "invalid.token.here"
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(invalid_token, mock_db)
        
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_verify_email_success(self, mock_db, mock_user):
        """Test email verification"""
        # Arrange
        mock_user.is_verified = False
        mock_db.execute.return_value.scalar_one_or_none.return_value = mock_user
        mock_db.commit = AsyncMock()
        
        verification_token = "valid_token_123"
        
        # Act
        with patch('app.api.v1.endpoints.auth.verify_token', return_value={"user_id": mock_user.id}):
            result = await verify_email(verification_token, mock_db)
        
        # Assert
        assert result["message"] == "Email verified successfully"
        assert mock_user.is_verified is True
        mock_db.commit.assert_called_once()