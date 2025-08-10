# YTEMPIRE API Testing Scenarios & Endpoint Documentation
**Version 1.0 | January 2025**  
**Owner: API Development Engineer**  
**Status: Testing & Documentation Standard**

---

## 1. API Testing Scenarios

### 1.1 Unit Testing Framework

```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime, timedelta

# Test configuration
class TestConfig:
    """Test environment configuration"""
    
    DATABASE_URL = "sqlite:///./test.db"
    JWT_SECRET_KEY = "test-secret-key"
    REDIS_URL = "redis://localhost:6379/1"
    
    # Test user credentials
    TEST_USERS = {
        "free_user": {
            "email": "free@test.com",
            "password": "testpass123",
            "tier": "free",
            "user_id": "usr_test_free_user"
        },
        "premium_user": {
            "email": "premium@test.com",
            "password": "testpass123",
            "tier": "premium",
            "user_id": "usr_test_premium_user"
        },
        "admin_user": {
            "email": "admin@test.com",
            "password": "testpass123",
            "tier": "enterprise",
            "roles": ["admin"],
            "user_id": "usr_test_admin_user"
        }
    }

# Test fixtures
@pytest.fixture(scope="session")
def test_db():
    """Create test database"""
    
    engine = create_engine(TestConfig.DATABASE_URL)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    yield TestingSessionLocal()
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_client(test_db):
    """Create test client with dependency overrides"""
    
    from main import app
    
    def override_get_db():
        try:
            yield test_db
        finally:
            test_db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as client:
        yield client

@pytest.fixture
def auth_headers(test_client, user_type="free_user"):
    """Get authentication headers for test user"""
    
    user_data = TestConfig.TEST_USERS[user_type]
    
    # Create test user
    response = test_client.post(
        "/api/v1/auth/register",
        json={
            "email": user_data["email"],
            "password": user_data["password"]
        }
    )
    
    # Login to get token
    response = test_client.post(
        "/api/v1/auth/login",
        json={
            "email": user_data["email"],
            "password": user_data["password"]
        }
    )
    
    token = response.json()["data"]["attributes"]["access_token"]
    
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Request-ID": f"test_{generate_unique_id()}"
    }

### 1.2 Authentication Testing

```python
class TestAuthentication:
    """Test authentication endpoints"""
    
    def test_user_registration_success(self, test_client):
        """Test successful user registration"""
        
        response = test_client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@test.com",
                "password": "ValidPass123!",
                "name": "Test User"
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["data"]["type"] == "user"
        assert data["data"]["attributes"]["email"] == "newuser@test.com"
        assert "password" not in data["data"]["attributes"]
    
    def test_user_registration_duplicate_email(self, test_client):
        """Test registration with duplicate email"""
        
        # First registration
        test_client.post(
            "/api/v1/auth/register",
            json={
                "email": "duplicate@test.com",
                "password": "ValidPass123!"
            }
        )
        
        # Duplicate registration
        response = test_client.post(
            "/api/v1/auth/register",
            json={
                "email": "duplicate@test.com",
                "password": "ValidPass123!"
            }
        )
        
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "4002"
    
    def test_login_success(self, test_client):
        """Test successful login"""
        
        # Create user
        test_client.post(
            "/api/v1/auth/register",
            json={
                "email": "login@test.com",
                "password": "ValidPass123!"
            }
        )
        
        # Login
        response = test_client.post(
            "/api/v1/auth/login",
            json={
                "email": "login@test.com",
                "password": "ValidPass123!"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data["data"]["attributes"]
        assert "refresh_token" in data["data"]["attributes"]
        assert data["data"]["attributes"]["token_type"] == "Bearer"
    
    def test_login_invalid_credentials(self, test_client):
        """Test login with invalid credentials"""
        
        response = test_client.post(
            "/api/v1/auth/login",
            json={
                "email": "nonexistent@test.com",
                "password": "WrongPass123!"
            }
        )
        
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "1004"
    
    def test_token_refresh(self, test_client):
        """Test token refresh flow"""
        
        # Login to get tokens
        login_response = test_client.post(
            "/api/v1/auth/login",
            json={
                "email": TestConfig.TEST_USERS["free_user"]["email"],
                "password": TestConfig.TEST_USERS["free_user"]["password"]
            }
        )
        
        refresh_token = login_response.json()["data"]["attributes"]["refresh_token"]
        
        # Refresh token
        response = test_client.post(
            "/api/v1/auth/refresh",
            json={
                "refresh_token": refresh_token
            }
        )
        
        assert response.status_code == 200
        assert "access_token" in response.json()["data"]["attributes"]
    
    def test_logout(self, test_client, auth_headers):
        """Test logout functionality"""
        
        response = test_client.post(
            "/api/v1/auth/logout",
            headers=auth_headers
        )
        
        assert response.status_code == 204
        
        # Verify token is invalidated
        response = test_client.get(
            "/api/v1/channels",
            headers=auth_headers
        )
        
        assert response.status_code == 401

### 1.3 Channel Management Testing

```python
class TestChannels:
    """Test channel management endpoints"""
    
    def test_create_channel_success(self, test_client, auth_headers):
        """Test successful channel creation"""
        
        response = test_client.post(
            "/api/v1/channels",
            headers=auth_headers,
            json={
                "data": {
                    "type": "channel",
                    "attributes": {
                        "name": "Tech Reviews Channel",
                        "description": "Latest technology reviews",
                        "niche": "technology",
                        "targetAudience": {
                            "ageRange": "18-34",
                            "interests": ["gadgets", "smartphones"]
                        }
                    }
                }
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["data"]["type"] == "channel"
        assert data["data"]["attributes"]["name"] == "Tech Reviews Channel"
        assert data["data"]["id"].startswith("ch_")
    
    def test_create_channel_limit_exceeded(self, test_client, auth_headers):
        """Test channel creation when limit exceeded"""
        
        # Create maximum allowed channels for free tier (1)
        test_client.post(
            "/api/v1/channels",
            headers=auth_headers,
            json={
                "data": {
                    "type": "channel",
                    "attributes": {
                        "name": "First Channel",
                        "niche": "technology"
                    }
                }
            }
        )
        
        # Try to create another
        response = test_client.post(
            "/api/v1/channels",
            headers=auth_headers,
            json={
                "data": {
                    "type": "channel",
                    "attributes": {
                        "name": "Second Channel",
                        "niche": "gaming"
                    }
                }
            }
        )
        
        assert response.status_code == 403
        assert "limit reached" in response.json()["error"]["detail"].lower()
    
    def test_get_channel_details(self, test_client, auth_headers):
        """Test retrieving channel details"""
        
        # Create channel
        create_response = test_client.post(
            "/api/v1/channels",
            headers=auth_headers,
            json={
                "data": {
                    "type": "channel",
                    "attributes": {
                        "name": "Test Channel",
                        "niche": "gaming"
                    }
                }
            }
        )
        
        channel_id = create_response.json()["data"]["id"]
        
        # Get channel details
        response = test_client.get(
            f"/api/v1/channels/{channel_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert response.json()["data"]["id"] == channel_id
    
    def test_update_channel(self, test_client, auth_headers):
        """Test updating channel information"""
        
        # Create channel
        create_response = test_client.post(
            "/api/v1/channels",
            headers=auth_headers,
            json={
                "data": {
                    "type": "channel",
                    "attributes": {
                        "name": "Original Name",
                        "niche": "technology"
                    }
                }
            }
        )
        
        channel_id = create_response.json()["data"]["id"]
        
        # Update channel
        response = test_client.put(
            f"/api/v1/channels/{channel_id}",
            headers=auth_headers,
            json={
                "data": {
                    "type": "channel",
                    "id": channel_id,
                    "attributes": {
                        "name": "Updated Name",
                        "description": "Updated description"
                    }
                }
            }
        )
        
        assert response.status_code == 200
        assert response.json()["data"]["attributes"]["name"] == "Updated Name"
    
    def test_list_channels_with_pagination(self, test_client, auth_headers):
        """Test listing channels with pagination"""
        
        # Create multiple channels (assuming premium user)
        premium_headers = auth_headers  # Get premium user headers
        
        for i in range(5):
            test_client.post(
                "/api/v1/channels",
                headers=premium_headers,
                json={
                    "data": {
                        "type": "channel",
                        "attributes": {
                            "name": f"Channel {i}",
                            "niche": "technology"
                        }
                    }
                }
            )
        
        # Get first page
        response = test_client.get(
            "/api/v1/channels?page[size]=2&page[number]=1",
            headers=premium_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2
        assert data["meta"]["pagination"]["perPage"] == 2
        assert data["meta"]["pagination"]["total"] == 5
        assert "next" in data["links"]

### 1.4 Video Generation Testing

```python
class TestVideoGeneration:
    """Test video generation and management"""
    
    @pytest.mark.asyncio
    async def test_video_generation_async(self, test_client, auth_headers):
        """Test async video generation"""
        
        # Create channel first
        channel_response = test_client.post(
            "/api/v1/channels",
            headers=auth_headers,
            json={
                "data": {
                    "type": "channel",
                    "attributes": {
                        "name": "Video Test Channel",
                        "niche": "technology"
                    }
                }
            }
        )
        
        channel_id = channel_response.json()["data"]["id"]
        
        # Generate video
        response = test_client.post(
            f"/api/v1/channels/{channel_id}/videos",
            headers={**auth_headers, "Prefer": "respond-async"},
            json={
                "data": {
                    "type": "video",
                    "attributes": {
                        "topic": "iPhone 15 Pro Review",
                        "style": "review",
                        "duration": "10-15"
                    }
                }
            }
        )
        
        assert response.status_code == 202
        assert "Location" in response.headers
        data = response.json()
        assert data["data"]["type"] == "job"
        assert data["data"]["attributes"]["status"] == "pending"
    
    def test_video_generation_sync(self, test_client, auth_headers):
        """Test synchronous video generation (if supported)"""
        
        # Create channel
        channel_id = self._create_test_channel(test_client, auth_headers)
        
        # Mock external services
        with patch('services.openai_service.generate_script') as mock_script:
            with patch('services.video_service.create_video') as mock_video:
                mock_script.return_value = "Test script content"
                mock_video.return_value = {"video_path": "/test/video.mp4"}
                
                response = test_client.post(
                    f"/api/v1/channels/{channel_id}/videos",
                    headers=auth_headers,
                    json={
                        "data": {
                            "type": "video",
                            "attributes": {
                                "topic": "Test Video",
                                "style": "tutorial"
                            }
                        }
                    }
                )
                
                assert response.status_code == 201
                assert response.json()["data"]["type"] == "video"
    
    def test_video_generation_rate_limit(self, test_client, auth_headers):
        """Test video generation rate limiting"""
        
        channel_id = self._create_test_channel(test_client, auth_headers)
        
        # Generate videos up to the limit (5 for free tier)
        for i in range(5):
            test_client.post(
                f"/api/v1/channels/{channel_id}/videos",
                headers=auth_headers,
                json={
                    "data": {
                        "type": "video",
                        "attributes": {
                            "topic": f"Video {i}",
                            "style": "review"
                        }
                    }
                }
            )
        
        # Try to generate one more
        response = test_client.post(
            f"/api/v1/channels/{channel_id}/videos",
            headers=auth_headers,
            json={
                "data": {
                    "type": "video",
                    "attributes": {
                        "topic": "Excess Video",
                        "style": "review"
                    }
                }
            }
        )
        
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "5002"
    
    def test_publish_video(self, test_client, auth_headers):
        """Test video publishing"""
        
        # Create video
        video_id = self._create_test_video(test_client, auth_headers)
        
        # Publish video
        response = test_client.post(
            f"/api/v1/videos/{video_id}/publish",
            headers=auth_headers,
            json={
                "scheduledAt": (datetime.utcnow() + timedelta(hours=1)).isoformat()
            }
        )
        
        assert response.status_code == 202
        assert response.json()["data"]["type"] == "job"
    
    def _create_test_channel(self, test_client, auth_headers):
        """Helper to create test channel"""
        
        response = test_client.post(
            "/api/v1/channels",
            headers=auth_headers,
            json={
                "data": {
                    "type": "channel",
                    "attributes": {
                        "name": "Test Channel",
                        "niche": "technology"
                    }
                }
            }
        )
        
        return response.json()["data"]["id"]
    
    def _create_test_video(self, test_client, auth_headers):
        """Helper to create test video"""
        
        channel_id = self._create_test_channel(test_client, auth_headers)
        
        response = test_client.post(
            f"/api/v1/channels/{channel_id}/videos",
            headers=auth_headers,
            json={
                "data": {
                    "type": "video",
                    "attributes": {
                        "topic": "Test Video",
                        "style": "review"
                    }
                }
            }
        )
        
        return response.json()["data"]["id"]

### 1.5 Integration Testing

```python
class TestIntegration:
    """End-to-end integration tests"""
    
    @pytest.mark.integration
    def test_complete_video_workflow(self, test_client):
        """Test complete video creation and publishing workflow"""
        
        # 1. Register user
        register_response = test_client.post(
            "/api/v1/auth/register",
            json={
                "email": "workflow@test.com",
                "password": "ValidPass123!"
            }
        )
        
        # 2. Login
        login_response = test_client.post(
            "/api/v1/auth/login",
            json={
                "email": "workflow@test.com",
                "password": "ValidPass123!"
            }
        )
        
        token = login_response.json()["data"]["attributes"]["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # 3. Create channel
        channel_response = test_client.post(
            "/api/v1/channels",
            headers=headers,
            json={
                "data": {
                    "type": "channel",
                    "attributes": {
                        "name": "Integration Test Channel",
                        "niche": "technology"
                    }
                }
            }
        )
        
        channel_id = channel_response.json()["data"]["id"]
        
        # 4. Generate video
        video_response = test_client.post(
            f"/api/v1/channels/{channel_id}/videos",
            headers={**headers, "Prefer": "respond-async"},
            json={
                "data": {
                    "type": "video",
                    "attributes": {
                        "topic": "Complete Workflow Test",
                        "style": "tutorial"
                    }
                }
            }
        )
        
        job_location = video_response.headers["Location"]
        
        # 5. Poll job status
        max_attempts = 10
        for _ in range(max_attempts):
            job_response = test_client.get(job_location, headers=headers)
            job_data = job_response.json()
            
            if job_data["data"]["attributes"]["status"] == "completed":
                video_id = job_data["data"]["attributes"]["resultId"]
                break
            
            time.sleep(1)
        
        # 6. Get video details
        video_detail_response = test_client.get(
            f"/api/v1/videos/{video_id}",
            headers=headers
        )
        
        assert video_detail_response.status_code == 200
        video_data = video_detail_response.json()["data"]
        assert video_data["attributes"]["status"] == "draft"
        
        # 7. Publish video
        publish_response = test_client.post(
            f"/api/v1/videos/{video_id}/publish",
            headers=headers
        )
        
        assert publish_response.status_code == 202
    
    @pytest.mark.integration
    def test_webhook_delivery(self, test_client, auth_headers):
        """Test webhook delivery system"""
        
        # Create webhook endpoint
        webhook_url = "https://webhook.site/test-endpoint"
        
        webhook_response = test_client.post(
            "/api/v1/webhooks",
            headers=auth_headers,
            json={
                "data": {
                    "type": "webhook",
                    "attributes": {
                        "url": webhook_url,
                        "events": ["video.published", "channel.updated"]
                    }
                }
            }
        )
        
        webhook_secret = webhook_response.json()["data"]["attributes"]["secret"]
        
        # Trigger an event
        channel_id = self._create_test_channel(test_client, auth_headers)
        
        # Update channel (should trigger webhook)
        test_client.put(
            f"/api/v1/channels/{channel_id}",
            headers=auth_headers,
            json={
                "data": {
                    "type": "channel",
                    "id": channel_id,
                    "attributes": {
                        "description": "Updated to trigger webhook"
                    }
                }
            }
        )
        
        # Verify webhook was queued
        # In real test, would verify actual delivery

### 1.6 Performance Testing

```python
import locust
from locust import HttpUser, task, between

class PerformanceTestUser(HttpUser):
    """Locust performance testing"""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Login before running tasks"""
        
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "email": "perf@test.com",
                "password": "TestPass123!"
            }
        )
        
        self.token = response.json()["data"]["attributes"]["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(3)
    def list_channels(self):
        """Test listing channels endpoint"""
        
        self.client.get(
            "/api/v1/channels",
            headers=self.headers
        )
    
    @task(2)
    def get_channel_details(self):
        """Test getting channel details"""
        
        # Assuming channel_id is known
        self.client.get(
            "/api/v1/channels/ch_1234567890abcdef",
            headers=self.headers
        )
    
    @task(1)
    def create_video(self):
        """Test video creation (less frequent)"""
        
        self.client.post(
            "/api/v1/channels/ch_1234567890abcdef/videos",
            headers=self.headers,
            json={
                "data": {
                    "type": "video",
                    "attributes": {
                        "topic": f"Performance Test Video {datetime.utcnow()}",
                        "style": "review"
                    }
                }
            }
        )

# Run with: locust -f performance_test.py --host=https://api.ytempire.com
```

---

## 2. Endpoint Documentation

### 2.1 Authentication Endpoints

#### POST /api/v1/auth/register
**Description**: Register a new user account

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "name": "John Doe"
}
```

**Success Response** (201 Created):
```json
{
  "data": {
    "type": "user",
    "id": "usr_1234567890abcdef",
    "attributes": {
      "email": "user@example.com",
      "name": "John Doe",
      "subscription_tier": "free",
      "created_at": "2025-01-15T10:30:00Z"
    }
  }
}
```

**Error Responses**:
- 400 Bad Request - Invalid input data
- 409 Conflict - Email already exists
- 422 Unprocessable Entity - Validation errors

---

#### POST /api/v1/auth/login
**Description**: Authenticate user and receive tokens

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

**Success Response** (200 OK):
```json
{
  "data": {
    "type": "auth",
    "attributes": {
      "access_token": "eyJhbGciOiJIUzI1NiIs...",
      "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
      "expires_in": 900,
      "token_type": "Bearer"
    },
    "relationships": {
      "user": {
        "type": "user",
        "id": "usr_1234567890abcdef"
      }
    }
  }
}
```

**Error Responses**:
- 401 Unauthorized - Invalid credentials
- 429 Too Many Requests - Rate limit exceeded

---

#### POST /api/v1/auth/refresh
**Description**: Refresh access token using refresh token

**Request Body**:
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

**Success Response** (200 OK):
```json
{
  "data": {
    "type": "auth",
    "attributes": {
      "access_token": "eyJhbGciOiJIUzI1NiIs...",
      "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
      "expires_in": 900,
      "token_type": "Bearer"
    }
  }
}
```

---

#### POST /api/v1/auth/logout
**Description**: Logout and invalidate tokens

**Headers**:
```
Authorization: Bearer {access_token}
```

**Success Response** (204 No Content)

---

### 2.2 Channel Management Endpoints

#### GET /api/v1/channels
**Description**: List user's channels with pagination

**Query Parameters**:
- `page[size]` (integer): Items per page (default: 20, max: 100)
- `page[number]` (integer): Page number (default: 1)
- `sort` (string): Sort order (e.g., `-created_at`, `name`)
- `filter[status]` (string): Filter by status (active, paused, suspended)
- `filter[niche]` (string): Filter by niche

**Success Response** (200 OK):
```json
{
  "data": [
    {
      "type": "channel",
      "id": "ch_1234567890abcdef",
      "attributes": {
        "name": "Tech Reviews",
        "description": "Latest technology reviews",
        "niche": "technology",
        "status": "active",
        "subscriber_count": 1523,
        "video_count": 45,
        "created_at": "2025-01-10T08:00:00Z"
      },
      "links": {
        "self": "/api/v1/channels/ch_1234567890abcdef"
      }
    }
  ],
  "meta": {
    "pagination": {
      "total": 5,
      "count": 5,
      "per_page": 20,
      "current_page": 1,
      "total_pages": 1
    }
  },
  "links": {
    "self": "/api/v1/channels?page[number]=1",
    "first": "/api/v1/channels?page[number]=1",
    "last": "/api/v1/channels?page[number]=1"
  }
}
```

---

#### POST /api/v1/channels
**Description**: Create a new channel

**Request Body**:
```json
{
  "data": {
    "type": "channel",
    "attributes": {
      "name": "Gaming Chronicles",
      "description": "Epic gaming moments and reviews",
      "niche": "gaming",
      "target_audience": {
        "age_range": "13-25",
        "interests": ["fps", "rpg", "esports"]
      },
      "settings": {
        "upload_schedule": {
          "frequency": "daily",
          "times": ["09:00", "18:00"]
        },
        "monetization_enabled": true
      }
    }
  }
}
```

**Success Response** (201 Created):
```json
{
  "data": {
    "type": "channel",
    "id": "ch_0987654321fedcba",
    "attributes": {
      "name": "Gaming Chronicles",
      "description": "Epic gaming moments and reviews",
      "niche": "gaming",
      "status": "active",
      "youtube_channel_id": null,
      "subscriber_count": 0,
      "video_count": 0,
      "created_at": "2025-01-15T10:45:00Z"
    },
    "relationships": {
      "owner": {
        "type": "user",
        "id": "usr_1234567890abcdef"
      }
    },
    "links": {
      "self": "/api/v1/channels/ch_0987654321fedcba"
    }
  }
}
```

**Error Responses**:
- 403 Forbidden - Channel limit reached
- 422 Unprocessable Entity - Validation errors

---

#### GET /api/v1/channels/{channel_id}
**Description**: Get channel details

**Path Parameters**:
- `channel_id` (string): Channel identifier

**Query Parameters**:
- `include` (string): Include related resources (videos, analytics)

**Success Response** (200 OK):
```json
{
  "data": {
    "type": "channel",
    "id": "ch_1234567890abcdef",
    "attributes": {
      "name": "Tech Reviews",
      "description": "Latest technology reviews",
      "niche": "technology",
      "status": "active",
      "youtube_channel_id": "UC1234567890",
      "subscriber_count": 1523,
      "video_count": 45,
      "total_views": 156789,
      "monthly_revenue": 234.56,
      "automation_enabled": true,
      "settings": {
        "upload_schedule": {
          "frequency": "daily",
          "times": ["09:00"]
        },
        "default_video_length": "medium",
        "monetization_enabled": true
      },
      "created_at": "2025-01-10T08:00:00Z",
      "updated_at": "2025-01-15T10:00:00Z"
    },
    "relationships": {
      "owner": {
        "type": "user",
        "id": "usr_1234567890abcdef"
      },
      "videos": {
        "links": {
          "related": "/api/v1/channels/ch_1234567890abcdef/videos"
        }
      }
    }
  }
}
```

---

### 2.3 Video Management Endpoints

#### POST /api/v1/channels/{channel_id}/videos
**Description**: Generate a new video for the channel

**Headers**:
```
Authorization: Bearer {access_token}
Prefer: respond-async  # Optional for async processing
```

**Request Body**:
```json
{
  "data": {
    "type": "video",
    "attributes": {
      "topic": "iPhone 15 Pro Max Review - Is it Worth $1200?",
      "style": "review",
      "duration": "10-15",
      "template_id": "tpl_tech_review_001",
      "custom_prompt": "Focus on camera improvements and battery life",
      "scheduled_publish_at": "2025-01-16T14:00:00Z"
    }
  }
}
```

**Success Response (Async)** (202 Accepted):
```json
{
  "data": {
    "type": "job",
    "id": "job_abc123def456",
    "attributes": {
      "status": "pending",
      "progress": 0,
      "created_at": "2025-01-15T11:00:00Z",
      "estimated_completion": "2025-01-15T11:10:00Z"
    }
  }
}
```
**Headers**:
```
Location: /api/v1/jobs/job_abc123def456
```

---

#### GET /api/v1/videos/{video_id}
**Description**: Get video details

**Success Response** (200 OK):
```json
{
  "data": {
    "type": "video",
    "id": "vid_1234567890abcdef",
    "attributes": {
      "title": "iPhone 15 Pro Max Review - Is it Worth $1200?",
      "description": "In this comprehensive review, we dive deep into...",
      "tags": ["iphone", "apple", "tech review", "smartphone"],
      "status": "published",
      "youtube_video_id": "dQw4w9WgXcQ",
      "thumbnail_url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
      "duration": 745,
      "published_at": "2025-01-16T14:00:00Z",
      "metrics": {
        "views": 12543,
        "likes": 1032,
        "comments": 89,
        "watch_time": 934521
      },
      "generation_cost": 0.85,
      "created_at": "2025-01-15T11:00:00Z"
    },
    "relationships": {
      "channel": {
        "type": "channel",
        "id": "ch_1234567890abcdef"
      }
    }
  }
}
```

---

#### POST /api/v1/videos/{video_id}/publish
**Description**: Publish video to YouTube

**Request Body** (Optional):
```json
{
  "scheduled_at": "2025-01-17T15:00:00Z"
}
```

**Success Response** (202 Accepted):
```json
{
  "data": {
    "type": "job",
    "id": "job_publish_xyz789",
    "attributes": {
      "status": "processing",
      "progress": 25,
      "created_at": "2025-01-15T12:00:00Z"
    }
  }
}
```

---

### 2.4 Analytics Endpoints

#### GET /api/v1/analytics/channels/{channel_id}
**Description**: Get channel analytics

**Query Parameters**:
- `start_date` (date, required): Start date (YYYY-MM-DD)
- `end_date` (date, required): End date (YYYY-MM-DD)
- `metrics` (array): Specific metrics to retrieve

**Success Response** (200 OK):
```json
{
  "data": {
    "type": "analytics",
    "attributes": {
      "start_date": "2025-01-01",
      "end_date": "2025-01-15",
      "metrics": {
        "views": {
          "total": 45678,
          "average": 3045,
          "change": 12.5
        },
        "subscribers": {
          "gained": 234,
          "lost": 12,
          "net": 222
        },
        "revenue": {
          "total": 567.89,
          "ad_revenue": 456.78,
          "affiliate_revenue": 111.11
        },
        "watch_time": {
          "total": 234567,
          "average": 15638
        }
      }
    }
  }
}
```

---

### 2.5 Webhook Endpoints

#### POST /api/v1/webhooks
**Description**: Create webhook subscription

**Request Body**:
```json
{
  "data": {
    "type": "webhook",
    "attributes": {
      "url": "https://example.com/webhooks/ytempire",
      "events": ["video.published", "channel.suspended"],
      "headers": {
        "X-Custom-Header": "custom-value"
      }
    }
  }
}
```

**Success Response** (201 Created):
```json
{
  "data": {
    "type": "webhook",
    "id": "wh_1234567890abcdef",
    "attributes": {
      "url": "https://example.com/webhooks/ytempire",
      "events": ["video.published", "channel.suspended"],
      "secret": "whsec_abcdef1234567890",
      "is_active": true,
      "created_at": "2025-01-15T13:00:00Z"
    }
  }
}
```

---

## 3. API Testing Best Practices

### 3.1 Test Coverage Requirements

```yaml
coverage_targets:
  unit_tests:
    target: 80%
    critical_paths: 95%
    
  integration_tests:
    target: 70%
    user_workflows: 90%
    
  performance_tests:
    response_time_p95: <500ms
    throughput: >1000 req/min
    error_rate: <0.1%
```

### 3.2 Test Data Management

```python
class TestDataFactory:
    """Generate consistent test data"""
    
    @staticmethod
    def create_test_user(tier="free"):
        return {
            "email": f"test_{uuid.uuid4()}@example.com",
            "password": "TestPass123!",
            "tier": tier
        }
    
    @staticmethod
    def create_test_channel():
        return {
            "name": f"Test Channel {uuid.uuid4()}",
            "niche": random.choice(["technology", "gaming", "education"]),
            "description": "Test channel description"
        }
```

### 3.3 Continuous Integration

```yaml
# .github/workflows/api-tests.yml
name: API Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Unit Tests
      run: |
        pytest tests/unit -v --cov=api --cov-report=xml
        
    - name: Run Integration Tests
      run: |
        pytest tests/integration -v -m integration
        
    - name: Upload Coverage
      uses: codecov/codecov-action@v3
```

---

## Document Control

- **Version**: 1.0
- **Last Updated**: January 2025
- **Owner**: API Development Engineer
- **Review Cycle**: Sprint-based
- **Test Environment**: https://staging-api.ytempire.com

**Testing Checklist**:
- [ ] Unit tests for all endpoints
- [ ] Integration tests for workflows
- [ ] Performance tests under load
- [ ] Security tests for auth/authz
- [ ] Error handling verification
- [ ] Rate limiting validation
- [ ] Webhook delivery testing
- [ ] Documentation accuracy