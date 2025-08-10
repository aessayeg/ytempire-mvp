# API Rate Limits & Quota Documentation

## OpenAI API Limits

### GPT-4 Turbo
- **Rate Limit**: 10,000 TPM (Tokens Per Minute)
- **Daily Limit**: 450,000 tokens
- **Requests Per Minute**: 60 RPM
- **Max Tokens per Request**: 4,096
- **Context Window**: 128,000 tokens

### GPT-3.5 Turbo
- **Rate Limit**: 90,000 TPM
- **Daily Limit**: 2,000,000 tokens
- **Requests Per Minute**: 3,500 RPM
- **Max Tokens per Request**: 4,096
- **Context Window**: 16,385 tokens

### DALL-E 3
- **Rate Limit**: 5 images per minute
- **Monthly Limit**: 150 images (Standard tier)
- **Resolution Options**: 1024x1024, 1024x1792, 1792x1024

### Whisper API
- **File Size Limit**: 25 MB
- **Rate Limit**: 50 RPM
- **Supported Formats**: mp3, mp4, mpeg, mpga, m4a, wav, webm

## ElevenLabs API Limits

### Starter Plan ($5/month)
- **Character Limit**: 30,000 characters/month
- **Voice Limit**: 10 custom voices
- **Concurrent Requests**: 2

### Creator Plan ($22/month)
- **Character Limit**: 100,000 characters/month
- **Voice Limit**: 30 custom voices
- **Concurrent Requests**: 5
- **Commercial Use**: Allowed

### Pro Plan ($99/month)
- **Character Limit**: 500,000 characters/month
- **Voice Limit**: 160 custom voices
- **Concurrent Requests**: 10
- **Commercial Use**: Allowed
- **Priority Queue**: Yes

## Google Cloud API Limits

### Text-to-Speech (TTS)
- **Rate Limit**: 1,000 requests per minute
- **Character Limit**: 5,000 characters per request
- **Monthly Free Tier**: 1 million characters (Standard voices)
- **Monthly Free Tier**: 1 million characters (WaveNet/Neural2 voices)

### Cloud Translation
- **Rate Limit**: 3,000 requests per 100 seconds
- **Character Limit**: 30,000 characters per request
- **Monthly Free Tier**: 500,000 characters

### Cloud Vision
- **Rate Limit**: 1,800 requests per minute
- **Image Size Limit**: 20 MB
- **Monthly Free Tier**: 1,000 units

## YouTube Data API v3 Quotas

### Default Quota
- **Daily Quota**: 10,000 units
- **Quota Cost by Operation**:
  - Search: 100 units
  - Video upload: 1,600 units
  - Video update: 50 units
  - Playlist insert: 50 units
  - Channel list: 1 unit
  - Video list: 1 unit
  - Comment insert: 50 units

### Quota Management Strategy
- Use multiple API keys (rotate between 3-5 keys)
- Batch operations where possible
- Cache frequently accessed data
- Implement exponential backoff for rate limit errors

## Anthropic Claude API Limits

### Claude 3 Opus
- **Rate Limit**: 5 requests per minute
- **Token Limit**: 200,000 tokens per request
- **Monthly Token Limit**: 10 million tokens

### Claude 3 Sonnet
- **Rate Limit**: 50 requests per minute
- **Token Limit**: 200,000 tokens per request
- **Monthly Token Limit**: 50 million tokens

## Rate Limiting Implementation

### Retry Strategy
```python
# Exponential backoff configuration
MAX_RETRIES = 3
BASE_DELAY = 1  # seconds
MAX_DELAY = 60  # seconds
EXPONENTIAL_BASE = 2
```

### Rate Limiter Configuration
```python
RATE_LIMITS = {
    "openai_gpt4": {"calls": 60, "period": 60},  # 60 calls per minute
    "openai_gpt35": {"calls": 3500, "period": 60},  # 3500 calls per minute
    "elevenlabs": {"calls": 5, "period": 60},  # 5 calls per minute
    "google_tts": {"calls": 1000, "period": 60},  # 1000 calls per minute
    "youtube": {"units": 10000, "period": 86400},  # 10000 units per day
}
```

## Cost Optimization Strategies

### Token Optimization
1. Use GPT-3.5 for simple tasks
2. Implement prompt caching
3. Batch similar requests
4. Use streaming for long responses
5. Implement token counting before requests

### API Cost Tracking
```python
# Cost per 1K tokens (USD)
API_COSTS = {
    "gpt-4-turbo": 0.01,  # Input
    "gpt-4-turbo-output": 0.03,  # Output
    "gpt-3.5-turbo": 0.0005,  # Input
    "gpt-3.5-turbo-output": 0.0015,  # Output
    "elevenlabs": 0.30,  # Per 1K characters
    "google-tts-standard": 4.00,  # Per 1M characters
    "google-tts-neural": 16.00,  # Per 1M characters
}
```

## Monitoring & Alerts

### Usage Thresholds
- **Warning**: 70% of daily/monthly limit
- **Critical**: 90% of daily/monthly limit
- **Emergency Stop**: 95% of limit

### Alert Channels
1. Email notifications
2. Slack webhooks
3. Dashboard indicators
4. Prometheus metrics

## Error Handling

### Common Error Codes
- `429`: Rate limit exceeded
- `402`: Payment required (quota exceeded)
- `503`: Service temporarily unavailable
- `401`: Invalid API key

### Response Headers to Monitor
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Time when limit resets
- `Retry-After`: Seconds to wait before retry

## Best Practices

1. **Always implement exponential backoff**
2. **Monitor usage in real-time**
3. **Set up billing alerts**
4. **Use caching aggressively**
5. **Implement circuit breakers**
6. **Log all API calls and costs**
7. **Review usage weekly**
8. **Optimize prompts for token efficiency**

## Emergency Procedures

### Rate Limit Exceeded
1. Switch to backup API key
2. Reduce request frequency
3. Queue non-critical requests
4. Alert operations team

### Quota Exhausted
1. Switch to alternative service
2. Purchase additional quota
3. Pause non-essential operations
4. Review and optimize usage patterns

---

*Last Updated: 2024*
*Maintained by: YTEmpire AI/ML Team*