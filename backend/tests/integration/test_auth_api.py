"""
Authentication API Integration Tests
Owner: QA Engineer #1
"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


@pytest.mark.integration
@pytest.mark.auth
@pytest.mark.api
class TestAuthAPI:
    """Test authentication API endpoints."""
    
    def test_register_user_success(self, client: TestClient):
        """Test successful user registration."""
        user_data = {
            "email": "apitest@example.com",
            "username": "apitest",
            "password": "testpassword123",
            "full_name": "API Test User"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == user_data["email"]
        assert data["username"] == user_data["username"]
        assert data["full_name"] == user_data["full_name"]
        assert data["is_active"] is True
        assert data["is_verified"] is False
        assert "id" in data
        assert "created_at" in data
    
    def test_register_user_invalid_email(self, client: TestClient):
        """Test user registration with invalid email."""
        user_data = {
            "email": "invalid-email",
            "username": "testuser",
            "password": "testpassword123"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 422
    
    def test_register_user_short_password(self, client: TestClient):
        """Test user registration with short password."""
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "short"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 422
    
    def test_login_success(self, client: TestClient, test_user: User):
        """Test successful user login."""
        login_data = {
            "username": "test@example.com",
            "password": "testpassword123"
        }
        
        response = client.post("/api/v1/auth/login", data=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        
        # Verify token structure (should be JWT)
        access_token = data["access_token"]
        assert len(access_token.split(".")) == 3  # JWT has 3 parts
    
    def test_login_with_username(self, client: TestClient, test_user: User):
        """Test login with username instead of email."""
        login_data = {
            "username": "testuser",
            "password": "testpassword123"
        }
        
        response = client.post("/api/v1/auth/login", data=login_data)
        assert response.status_code == 200
    
    def test_login_wrong_password(self, client: TestClient, test_user: User):
        """Test login with wrong password."""
        login_data = {
            "username": "test@example.com",
            "password": "wrongpassword"
        }
        
        response = client.post("/api/v1/auth/login", data=login_data)
        assert response.status_code == 401
        assert "Incorrect email/username or password" in response.json()["detail"]
    
    def test_login_nonexistent_user(self, client: TestClient):
        """Test login with non-existent user."""
        login_data = {
            "username": "nonexistent@example.com",
            "password": "password123"
        }
        
        response = client.post("/api/v1/auth/login", data=login_data)
        assert response.status_code == 401
    
    def test_refresh_token_success(self, client: TestClient, test_user: User):
        """Test successful token refresh."""
        # First, login to get tokens
        login_data = {
            "username": "test@example.com",
            "password": "testpassword123"
        }
        
        login_response = client.post("/api/v1/auth/login", data=login_data)
        tokens = login_response.json()
        
        # Refresh access token
        refresh_data = {
            "refresh_token": tokens["refresh_token"]
        }
        
        response = client.post("/api/v1/auth/refresh", json=refresh_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        
        # New access token should be different
        assert data["access_token"] != tokens["access_token"]
        # Refresh token should be the same
        assert data["refresh_token"] == tokens["refresh_token"]
    
    def test_refresh_invalid_token(self, client: TestClient):
        """Test token refresh with invalid token."""
        refresh_data = {
            "refresh_token": "invalid.token.here"
        }
        
        response = client.post("/api/v1/auth/refresh", json=refresh_data)
        assert response.status_code == 401
        assert "Invalid refresh token" in response.json()["detail"]
    
    def test_get_current_user(self, client: TestClient, test_user: User):
        """Test getting current user information."""
        # Login first
        login_data = {
            "username": "test@example.com",
            "password": "testpassword123"
        }
        
        login_response = client.post("/api/v1/auth/login", data=login_data)
        tokens = login_response.json()
        
        # Get current user
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_user.id
        assert data["email"] == test_user.email
        assert data["username"] == test_user.username
        assert data["is_active"] is True
    
    def test_get_current_user_invalid_token(self, client: TestClient):
        """Test getting current user with invalid token."""
        headers = {"Authorization": "Bearer invalid.token.here"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == 401
        assert "Invalid token" in response.json()["detail"]
    
    def test_get_current_user_no_token(self, client: TestClient):
        """Test getting current user without token."""
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401
    
    def test_change_password_success(self, client: TestClient, test_user: User):
        """Test successful password change."""
        # Login first
        login_data = {
            "username": "test@example.com",
            "password": "testpassword123"
        }
        
        login_response = client.post("/api/v1/auth/login", data=login_data)
        tokens = login_response.json()
        
        # Change password
        password_data = {
            "current_password": "testpassword123",
            "new_password": "newpassword456"
        }
        
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        response = client.post("/api/v1/auth/change-password", json=password_data, headers=headers)
        
        assert response.status_code == 200
        assert "successfully changed" in response.json()["message"]
        
        # Verify new password works
        new_login_data = {
            "username": "test@example.com",
            "password": "newpassword456"
        }
        
        new_login_response = client.post("/api/v1/auth/login", data=new_login_data)
        assert new_login_response.status_code == 200
        
        # Verify old password doesn't work
        old_login_response = client.post("/api/v1/auth/login", data=login_data)
        assert old_login_response.status_code == 401
    
    def test_change_password_wrong_current(self, client: TestClient, test_user: User):
        """Test password change with wrong current password."""
        # Login first
        login_data = {
            "username": "test@example.com",
            "password": "testpassword123"
        }
        
        login_response = client.post("/api/v1/auth/login", data=login_data)
        tokens = login_response.json()
        
        # Try to change password with wrong current password
        password_data = {
            "current_password": "wrongpassword",
            "new_password": "newpassword456"
        }
        
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        response = client.post("/api/v1/auth/change-password", json=password_data, headers=headers)
        
        assert response.status_code == 400
        assert "Current password is incorrect" in response.json()["detail"]
    
    def test_logout_success(self, client: TestClient, test_user: User):
        """Test successful logout."""
        # Login first
        login_data = {
            "username": "test@example.com",
            "password": "testpassword123"
        }
        
        login_response = client.post("/api/v1/auth/login", data=login_data)
        tokens = login_response.json()
        
        # Logout
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        response = client.post("/api/v1/auth/logout", headers=headers)
        
        assert response.status_code == 200
        assert "Successfully logged out" in response.json()["message"]
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, async_client: AsyncClient):
        """Test concurrent authentication operations."""
        import asyncio
        
        # Create multiple users concurrently
        async def register_user(username_suffix: str):
            user_data = {
                "email": f"concurrent{username_suffix}@example.com",
                "username": f"concurrent{username_suffix}",
                "password": "testpassword123"
            }
            response = await async_client.post("/api/v1/auth/register", json=user_data)
            return response
        
        # Register 5 users concurrently
        tasks = [register_user(str(i)) for i in range(5)]
        responses = await asyncio.gather(*tasks)
        
        # All registrations should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "id" in data
            assert data["is_active"] is True