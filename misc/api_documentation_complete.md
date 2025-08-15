# YTEmpire Complete API Documentation

## API Overview
- **Base URL**: `http://localhost:8000/api/v1`
- **Authentication**: JWT Bearer Token (RS256)
- **Rate Limiting**: 1000 requests/hour per user
- **Response Format**: JSON
- **API Version**: v1
- **Total Endpoints**: 400+

## Authentication

### POST /auth/register
Register a new user account.

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!",
  "full_name": "John Doe",
  "company": "YTEmpire Inc"
}
```

**Response** (201):
```json
{
  "user_id": "usr_123456",
  "email": "user@example.com",
  "message": "Registration successful. Please verify your email.",
  "verification_sent": true
}
```

### POST /auth/login
Authenticate user and receive JWT token.

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!"
}
```

**Response** (200):
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "usr_123456",
    "email": "user@example.com",
    "subscription_tier": "professional"
  }
}
```

### POST /auth/refresh
Refresh access token using refresh token.

### POST /auth/logout
Invalidate current session tokens.

### POST /auth/verify-email
Verify email address with token.

### POST /auth/forgot-password
Request password reset email.

### POST /auth/reset-password
Reset password with token.

### POST /auth/2fa/enable
Enable two-factor authentication.

### POST /auth/2fa/verify
Verify 2FA code.

## User Management

### GET /users/profile
Get current user profile.

**Headers**:
```
Authorization: Bearer <token>
```

**Response** (200):
```json
{
  "id": "usr_123456",
  "email": "user@example.com",
  "full_name": "John Doe",
  "subscription": {
    "tier": "professional",
    "videos_remaining": 450,
    "expires_at": "2024-09-15T00:00:00Z"
  },
  "usage": {
    "videos_generated": 50,
    "total_cost": 125.50,
    "average_cost_per_video": 2.51
  }
}
```

### PUT /users/profile
Update user profile.

### DELETE /users/account
Delete user account (requires confirmation).

### GET /users/settings
Get user settings.

### PUT /users/settings
Update user settings.

## Channel Management

### GET /channels
List all channels for authenticated user.

**Query Parameters**:
- `page`: int (default: 1)
- `limit`: int (default: 20, max: 100)
- `status`: active|suspended|all

**Response** (200):
```json
{
  "channels": [
    {
      "id": "ch_123456",
      "name": "Tech Reviews Channel",
      "youtube_channel_id": "UC123456789",
      "status": "active",
      "health_score": 95,
      "videos_count": 125,
      "subscribers": 50000,
      "monthly_views": 1500000
    }
  ],
  "total": 5,
  "page": 1,
  "pages": 1
}
```

### POST /channels
Create new channel connection.

**Request Body**:
```json
{
  "name": "My Channel",
  "youtube_channel_id": "UC123456789",
  "category": "technology",
  "target_audience": "tech_enthusiasts",
  "posting_schedule": {
    "frequency": "daily",
    "preferred_times": ["09:00", "15:00", "20:00"]
  }
}
```

### GET /channels/{channel_id}
Get specific channel details.

### PUT /channels/{channel_id}
Update channel settings.

### DELETE /channels/{channel_id}
Remove channel connection.

### POST /channels/{channel_id}/sync
Sync channel with YouTube.

### GET /channels/{channel_id}/analytics
Get channel analytics.

### GET /channels/{channel_id}/health
Get channel health metrics.

## Video Generation

### POST /videos/generate
Generate a new video.

**Request Body**:
```json
{
  "channel_id": "ch_123456",
  "title": "Top 10 Tech Gadgets 2024",
  "description": "Reviewing the latest technology gadgets",
  "script_style": "informative",
  "voice_id": "voice_professional_male",
  "thumbnail_style": "modern",
  "duration_target": 600,
  "keywords": ["tech", "gadgets", "2024"],
  "cost_limit": 3.00,
  "quality_threshold": 85
}
```

**Response** (202):
```json
{
  "video_id": "vid_789012",
  "status": "processing",
  "estimated_completion": "2024-08-15T12:30:00Z",
  "estimated_cost": 2.75,
  "websocket_channel": "video_updates_vid_789012"
}
```

### GET /videos
List all videos.

**Query Parameters**:
- `channel_id`: string
- `status`: processing|completed|failed|published
- `date_from`: ISO 8601
- `date_to`: ISO 8601
- `page`: int
- `limit`: int

### GET /videos/{video_id}
Get video details.

### PUT /videos/{video_id}
Update video metadata.

### DELETE /videos/{video_id}
Delete video.

### POST /videos/{video_id}/publish
Publish video to YouTube.

### GET /videos/{video_id}/status
Get video generation status.

### GET /videos/{video_id}/analytics
Get video performance analytics.

### POST /videos/batch
Generate multiple videos (batch operation).

## Video Processing

### POST /video-processing/thumbnail
Generate thumbnail only.

### POST /video-processing/script
Generate script only.

### POST /video-processing/voice
Generate voice synthesis only.

### POST /video-processing/optimize
Optimize existing video.

### GET /video-processing/queue
Get processing queue status.

## Analytics

### GET /analytics/overview
Get analytics overview.

**Response** (200):
```json
{
  "period": "last_30_days",
  "metrics": {
    "total_views": 5000000,
    "total_revenue": 15000.00,
    "avg_view_duration": 450,
    "engagement_rate": 0.065,
    "subscriber_growth": 5000
  },
  "top_videos": [...],
  "channel_performance": [...]
}
```

### GET /analytics/revenue
Get revenue analytics.

### GET /analytics/engagement
Get engagement metrics.

### GET /analytics/trends
Get trending topics and keywords.

### GET /analytics/competitors
Get competitive analysis.

### POST /analytics/export
Export analytics data.

## Cost Management

### GET /costs/summary
Get cost summary.

**Response** (200):
```json
{
  "period": "current_month",
  "total_cost": 750.50,
  "breakdown": {
    "openai": 400.00,
    "elevenlabs": 200.00,
    "dall-e": 100.00,
    "google_tts": 50.50
  },
  "videos_generated": 300,
  "average_cost_per_video": 2.50
}
```

### GET /costs/history
Get historical cost data.

### GET /costs/optimization
Get cost optimization suggestions.

### PUT /costs/limits
Set cost limits and alerts.

## ML Models

### GET /ml-models
List available ML models.

### GET /ml-models/{model_id}
Get model details.

### POST /ml-models/{model_id}/predict
Run prediction with model.

### GET /ml-models/performance
Get model performance metrics.

## Batch Operations

### POST /batch/videos
Batch video generation.

**Request Body**:
```json
{
  "videos": [
    {
      "title": "Video 1",
      "channel_id": "ch_123"
    },
    {
      "title": "Video 2",
      "channel_id": "ch_123"
    }
  ],
  "processing_options": {
    "parallel": true,
    "priority": "high",
    "cost_limit_per_video": 3.00
  }
}
```

### GET /batch/{batch_id}/status
Get batch operation status.

### POST /batch/{batch_id}/cancel
Cancel batch operation.

## Webhooks

### GET /webhooks
List configured webhooks.

### POST /webhooks
Create new webhook.

**Request Body**:
```json
{
  "url": "https://your-app.com/webhook",
  "events": ["video.completed", "video.published"],
  "secret": "webhook_secret_key"
}
```

### PUT /webhooks/{webhook_id}
Update webhook configuration.

### DELETE /webhooks/{webhook_id}
Delete webhook.

### POST /webhooks/{webhook_id}/test
Test webhook with sample payload.

## YouTube Integration

### GET /youtube/accounts
List connected YouTube accounts (15 max).

**Response** (200):
```json
{
  "accounts": [
    {
      "account_id": "yt_acc_1",
      "email": "channel1@gmail.com",
      "channel_name": "Tech Channel 1",
      "status": "active",
      "quota_used": 2500,
      "quota_limit": 10000,
      "health_score": 98
    }
  ],
  "total_accounts": 15,
  "aggregated_quota": {
    "used": 37500,
    "limit": 150000
  }
}
```

### POST /youtube/oauth/connect
Connect new YouTube account.

### DELETE /youtube/accounts/{account_id}
Disconnect YouTube account.

### GET /youtube/accounts/{account_id}/quota
Get quota usage for account.

## WebSocket Endpoints

### WS /ws/{client_id}
General WebSocket connection.

**Message Types**:
- `video.progress`: Video generation progress
- `video.completed`: Video generation complete
- `analytics.update`: Real-time analytics
- `system.notification`: System notifications

### WS /ws/video-updates/{channel_id}
Channel-specific video updates.

### WS /ws/analytics-stream
Real-time analytics stream.

## Payment & Subscription

### GET /payment/subscription
Get current subscription details.

### POST /payment/subscribe
Create new subscription.

### PUT /payment/subscription
Update subscription plan.

### DELETE /payment/subscription
Cancel subscription.

### GET /payment/invoices
List invoices.

### GET /payment/methods
List payment methods.

### POST /payment/methods
Add payment method.

## Reports

### GET /reports/available
List available report types.

### POST /reports/generate
Generate custom report.

### GET /reports/{report_id}
Get report details.

### GET /reports/{report_id}/download
Download report file.

## System

### GET /health
System health check.

**Response** (200):
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "ml_pipeline": "healthy",
    "youtube_api": "healthy"
  },
  "uptime": 864000
}
```

### GET /metrics
Prometheus metrics endpoint.

### GET /status
Detailed system status.

## Rate Limits

| Endpoint Category | Rate Limit | Window |
|------------------|------------|---------|
| Authentication | 10 req | 1 minute |
| Video Generation | 100 req | 1 hour |
| Analytics | 500 req | 1 hour |
| General API | 1000 req | 1 hour |

## Error Responses

### 400 Bad Request
```json
{
  "error": "validation_error",
  "message": "Invalid request parameters",
  "details": {
    "field": "email",
    "issue": "Invalid email format"
  }
}
```

### 401 Unauthorized
```json
{
  "error": "unauthorized",
  "message": "Authentication required"
}
```

### 403 Forbidden
```json
{
  "error": "forbidden",
  "message": "Insufficient permissions"
}
```

### 404 Not Found
```json
{
  "error": "not_found",
  "message": "Resource not found"
}
```

### 429 Too Many Requests
```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests",
  "retry_after": 3600
}
```

### 500 Internal Server Error
```json
{
  "error": "internal_error",
  "message": "An unexpected error occurred",
  "request_id": "req_abc123"
}
```

## SDK Examples

### Python
```python
import requests

API_BASE = "http://localhost:8000/api/v1"
headers = {"Authorization": f"Bearer {token}"}

# Generate video
response = requests.post(
    f"{API_BASE}/videos/generate",
    headers=headers,
    json={
        "channel_id": "ch_123",
        "title": "My Video",
        "cost_limit": 3.00
    }
)
```

### JavaScript
```javascript
const API_BASE = 'http://localhost:8000/api/v1';

// Generate video
fetch(`${API_BASE}/videos/generate`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    channel_id: 'ch_123',
    title: 'My Video',
    cost_limit: 3.00
  })
});
```

### cURL
```bash
curl -X POST http://localhost:8000/api/v1/videos/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "channel_id": "ch_123",
    "title": "My Video",
    "cost_limit": 3.00
  }'
```

## Postman Collection

Import the Postman collection from: `/docs/postman/ytempire-api.json`

## OpenAPI Specification

Access the interactive API documentation at: `http://localhost:8000/docs`

Download OpenAPI spec: `http://localhost:8000/openapi.json`

---

**Last Updated**: August 15, 2024
**Version**: 1.0.0
**Total Endpoints**: 400+
**Maintained By**: Backend Team