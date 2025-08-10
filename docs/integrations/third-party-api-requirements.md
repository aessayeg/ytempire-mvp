# Third-Party API Integration Requirements

**Owner**: Integration Specialist  
**Created**: Day 1  
**Version**: 1.0

---

## Overview

This document outlines all third-party API integrations required for the YTEmpire platform, including authentication methods, rate limits, cost structures, and implementation requirements.

---

## 1. OpenAI API

### Purpose
Script generation, content optimization, and intelligent content creation

### API Details
- **Base URL**: `https://api.openai.com/v1`
- **Authentication**: Bearer token (API Key)
- **Rate Limits**: 
  - GPT-3.5: 3,500 RPM, 90,000 TPM
  - GPT-4: 500 RPM, 10,000 TPM
- **Retry Strategy**: Exponential backoff with jitter

### Endpoints Required
```yaml
/chat/completions:
  method: POST
  purpose: Generate video scripts
  model: gpt-3.5-turbo-1106 (primary)
  fallback: gpt-4-1106-preview (premium)

/embeddings:
  method: POST
  purpose: Content similarity analysis
  model: text-embedding-ada-002

/moderations:
  method: POST
  purpose: Content safety checking
  model: text-moderation-latest
```

### Cost Structure
| Model | Input | Output |
|-------|-------|--------|
| GPT-3.5-turbo-1106 | $0.001/1K tokens | $0.002/1K tokens |
| GPT-4-1106-preview | $0.01/1K tokens | $0.03/1K tokens |
| text-embedding-ada-002 | $0.0001/1K tokens | - |

### Implementation Requirements
```python
# Configuration
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "organization": os.getenv("OPENAI_ORG_ID"),
    "max_retries": 3,
    "timeout": 30,
    "max_tokens": 2000,
    "temperature": 0.7,
    "top_p": 0.9
}

# Error Handling
RETRY_ERRORS = [429, 500, 502, 503, 504]
FATAL_ERRORS = [401, 403, 404]
```

---

## 2. YouTube Data API v3

### Purpose
Channel management, video upload, analytics retrieval

### API Details
- **Base URL**: `https://www.googleapis.com/youtube/v3`
- **Authentication**: OAuth 2.0
- **Quota**: 10,000 units per day (per project)
- **Quota Management**: 15-account rotation system

### Endpoints Required
```yaml
/channels:
  method: GET
  purpose: Retrieve channel information
  quota_cost: 1 unit

/videos:
  method: POST (insert)
  purpose: Upload videos
  quota_cost: 1600 units
  
/videos:
  method: PUT (update)
  purpose: Update video metadata
  quota_cost: 50 units

/playlists:
  method: POST
  purpose: Create playlists
  quota_cost: 50 units

/playlistItems:
  method: POST
  purpose: Add videos to playlists
  quota_cost: 50 units

/analytics:
  method: GET
  purpose: Retrieve video analytics
  quota_cost: 1 unit
```

### Quota Optimization Strategy
```python
# Quota allocation per operation
QUOTA_LIMITS = {
    "daily_total": 10000,
    "upload": 1600,
    "update": 50,
    "read": 1,
    "buffer": 1000  # Reserve for errors
}

# Account rotation logic
def get_youtube_account():
    accounts = load_accounts()
    for account in accounts:
        if account.quota_remaining > required_quota:
            return account
    raise QuotaExhausted("All accounts quota exceeded")
```

### OAuth 2.0 Flow
1. Redirect user to Google authorization
2. Receive authorization code
3. Exchange for access/refresh tokens
4. Store encrypted tokens per channel
5. Auto-refresh before expiration

---

## 3. ElevenLabs API

### Purpose
High-quality voice synthesis for video narration

### API Details
- **Base URL**: `https://api.elevenlabs.io/v1`
- **Authentication**: API Key (xi-api-key header)
- **Rate Limits**: 
  - Free: 10,000 characters/month
  - Starter: 30,000 characters/month
  - Creator: 100,000 characters/month
- **Concurrency**: 2 concurrent requests

### Endpoints Required
```yaml
/text-to-speech/{voice_id}:
  method: POST
  purpose: Generate speech from text
  params:
    model_id: eleven_monolingual_v1
    voice_settings:
      stability: 0.5
      similarity_boost: 0.75

/voices:
  method: GET
  purpose: List available voices
  
/user:
  method: GET
  purpose: Check usage and limits

/history:
  method: GET
  purpose: Retrieve generation history
```

### Voice Configuration
```python
VOICE_PROFILES = {
    "narrator_male": {
        "voice_id": "21m00Tcm4TlvDq8ikWAM",
        "name": "Rachel",
        "settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True
        }
    },
    "narrator_female": {
        "voice_id": "AZnzlk1XvdvUeBnXmlld",
        "name": "Domi",
        "settings": {
            "stability": 0.6,
            "similarity_boost": 0.8
        }
    }
}
```

### Cost Structure
- Starter: $5/month for 30,000 characters
- Creator: $22/month for 100,000 characters
- Professional: $99/month for 500,000 characters
- Character counting: ~150 words = 1000 characters

---

## 4. Google Cloud Text-to-Speech

### Purpose
Backup TTS service for cost optimization

### API Details
- **Base URL**: `https://texttospeech.googleapis.com/v1`
- **Authentication**: Service Account JSON key
- **Rate Limits**: 1000 requests/minute
- **Free Tier**: 1 million characters/month

### Endpoints Required
```yaml
/text:synthesize:
  method: POST
  purpose: Convert text to speech
  params:
    input:
      text: "script content"
    voice:
      languageCode: "en-US"
      name: "en-US-Neural2-F"
    audioConfig:
      audioEncoding: "MP3"
      speakingRate: 1.0
      pitch: 0.0
```

### Voice Options
```python
GOOGLE_VOICES = {
    "standard": {
        "name": "en-US-Standard-F",
        "gender": "FEMALE",
        "cost_per_million": 4.00
    },
    "wavenet": {
        "name": "en-US-Wavenet-F",
        "gender": "FEMALE",
        "cost_per_million": 16.00
    },
    "neural2": {
        "name": "en-US-Neural2-F",
        "gender": "FEMALE",
        "cost_per_million": 16.00
    }
}
```

---

## 5. Pexels API

### Purpose
Stock footage and images for video content

### API Details
- **Base URL**: `https://api.pexels.com/v1`
- **Authentication**: API Key (Authorization header)
- **Rate Limits**: 200 requests/hour
- **Cost**: Free with attribution

### Endpoints Required
```yaml
/search:
  method: GET
  purpose: Search for images
  params:
    query: "search term"
    per_page: 15
    page: 1

/videos/search:
  method: GET
  purpose: Search for videos
  params:
    query: "search term"
    per_page: 15
    min_duration: 5
    max_duration: 30
```

---

## 6. Pixabay API

### Purpose
Alternative source for royalty-free media

### API Details
- **Base URL**: `https://pixabay.com/api`
- **Authentication**: API Key (key parameter)
- **Rate Limits**: 5000 requests/hour
- **Cost**: Free

### Endpoints Required
```yaml
/:
  method: GET
  purpose: Search media
  params:
    q: "search term"
    image_type: "photo"
    video_type: "film"
    min_width: 1920
    min_height: 1080
```

---

## 7. Stripe API

### Purpose
Payment processing and subscription management

### API Details
- **Base URL**: `https://api.stripe.com/v1`
- **Authentication**: Bearer token (Secret Key)
- **Rate Limits**: 100 requests/second
- **Webhook Security**: Signature verification

### Endpoints Required
```yaml
/customers:
  method: POST
  purpose: Create customer

/subscriptions:
  method: POST
  purpose: Create subscription

/payment_intents:
  method: POST
  purpose: Process payments

/invoices:
  method: GET
  purpose: Retrieve invoices

/webhook_endpoints:
  method: POST
  purpose: Register webhooks
```

### Webhook Events
```python
STRIPE_WEBHOOKS = [
    "customer.subscription.created",
    "customer.subscription.updated",
    "customer.subscription.deleted",
    "invoice.payment_succeeded",
    "invoice.payment_failed",
    "payment_intent.succeeded",
    "payment_intent.payment_failed"
]
```

---

## 8. SendGrid API

### Purpose
Transactional email delivery

### API Details
- **Base URL**: `https://api.sendgrid.com/v3`
- **Authentication**: Bearer token (API Key)
- **Rate Limits**: 10,000 emails/second
- **Free Tier**: 100 emails/day

### Endpoints Required
```yaml
/mail/send:
  method: POST
  purpose: Send emails
  
/templates:
  method: GET/POST
  purpose: Manage email templates

/stats:
  method: GET
  purpose: Email statistics
```

### Email Templates
```python
EMAIL_TEMPLATES = {
    "welcome": "d-xxxxx",
    "password_reset": "d-xxxxx",
    "video_complete": "d-xxxxx",
    "subscription_confirm": "d-xxxxx",
    "payment_failed": "d-xxxxx"
}
```

---

## 9. AWS S3 API

### Purpose
Video and media storage

### API Details
- **Endpoint**: Region-specific
- **Authentication**: AWS Signature v4
- **Rate Limits**: 3,500 PUT/COPY/POST/DELETE per second
- **Cost**: $0.023 per GB/month + transfer costs

### Operations Required
```yaml
PutObject:
  purpose: Upload videos/images
  
GetObject:
  purpose: Retrieve media
  
DeleteObject:
  purpose: Remove old content
  
CreateMultipartUpload:
  purpose: Large video uploads
  
GeneratePresignedUrl:
  purpose: Temporary access URLs
```

---

## Integration Best Practices

### 1. Authentication Management
```python
class APIKeyManager:
    def __init__(self):
        self.keys = self.load_encrypted_keys()
        self.rotation_schedule = {}
    
    def get_key(self, service):
        key = self.keys.get(service)
        if self.should_rotate(service):
            key = self.rotate_key(service)
        return key
    
    def rotate_key(self, service):
        # Implement key rotation logic
        pass
```

### 2. Rate Limiting
```python
class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
    
    async def acquire(self):
        now = time.time()
        # Remove old calls outside window
        while self.calls and self.calls[0] < now - self.period:
            self.calls.popleft()
        
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            await asyncio.sleep(sleep_time)
            return await self.acquire()
        
        self.calls.append(now)
```

### 3. Error Handling
```python
class APIClient:
    async def make_request(self, endpoint, data, retries=3):
        for attempt in range(retries):
            try:
                response = await self._request(endpoint, data)
                return response
            except RateLimitError:
                await self.handle_rate_limit(attempt)
            except TemporaryError:
                await self.exponential_backoff(attempt)
            except PermanentError:
                raise
        raise MaxRetriesExceeded()
```

### 4. Cost Tracking
```python
class CostTracker:
    def track_api_call(self, service, operation, units):
        cost = self.calculate_cost(service, operation, units)
        self.record_cost(service, operation, cost)
        
        if self.is_approaching_limit(service):
            self.send_alert(service)
        
        return cost
```

### 5. Fallback Strategies
```python
SERVICE_FALLBACKS = {
    "elevenlabs": ["google_tts", "local_tts"],
    "openai_gpt4": ["openai_gpt3.5", "local_llm"],
    "pexels": ["pixabay", "unsplash"],
    "stripe": ["paddle", "manual_processing"]
}
```

---

## Monitoring & Alerting

### Metrics to Track
- API call volume by service
- Error rates and types
- Response times
- Cost per service
- Quota usage percentage
- Rate limit hits

### Alert Thresholds
```yaml
alerts:
  - name: High API Error Rate
    condition: error_rate > 5%
    severity: warning
    
  - name: Quota Near Limit
    condition: quota_used > 80%
    severity: critical
    
  - name: Cost Overrun
    condition: daily_cost > budget * 1.1
    severity: critical
    
  - name: Service Unavailable
    condition: consecutive_errors > 5
    severity: critical
```

---

## Security Considerations

### API Key Storage
- Store in environment variables
- Encrypt at rest
- Use key management service
- Implement key rotation
- Audit key usage

### Request Security
- Use HTTPS only
- Implement request signing
- Add request timestamps
- Use webhook signatures
- Implement IP whitelisting where possible

### Data Protection
- Minimize data in logs
- Encrypt sensitive data
- Implement PII masking
- Use secure connections
- Regular security audits

---

*Last Updated: Day 1*  
*Next Review: Day 5*