# YTEMPIRE YouTube API Integration Guide

## Document Control
- **Version**: 1.0
- **Date**: January 2025
- **Author**: VP of Engineering
- **Audience**: Analytics Engineer, Backend Team, Data Team
- **Status**: FINAL - Ready for Implementation

---

## 1. YouTube API Architecture Overview

### 1.1 API Services Used

```yaml
youtube_apis:
  data_api_v3:
    purpose: "Channel management, video uploads, metadata"
    quota_cost: "Variable (1-1600 units per operation)"
    endpoints:
      - channels.list
      - videos.insert
      - videos.update
      - playlists.insert
      - thumbnails.set
      
  analytics_api_v2:
    purpose: "Performance metrics, revenue data"
    quota_cost: "1 unit per request"
    endpoints:
      - reports.query
      
  reporting_api_v1:
    purpose: "Bulk data downloads (ZERO quota cost)"
    quota_cost: "0 units - our secret weapon"
    reports:
      - channel_basic_a2
      - channel_traffic_source_a2
      - channel_combined_a2
```

### 1.2 OAuth 2.0 Flow Implementation

```python
class YouTubeOAuthManager:
    """
    Manages OAuth flow for 50 users connecting their YouTube channels
    """
    
    SCOPES = [
        'https://www.googleapis.com/auth/youtube',
        'https://www.googleapis.com/auth/youtube.upload',
        'https://www.googleapis.com/auth/yt-analytics.readonly',
        'https://www.googleapis.com/auth/yt-analytics-monetary.readonly'
    ]
    
    def initiate_oauth_flow(self, user_id: str) -> str:
        """
        Step 1: Generate authorization URL for user
        """
        flow = google_auth_oauthlib.flow.Flow.from_client_config(
            self.client_config,
            scopes=self.SCOPES
        )
        
        flow.redirect_uri = f'https://app.ytempire.com/auth/youtube/callback'
        
        auth_url, state = flow.authorization_url(
            access_type='offline',  # For refresh tokens
            include_granted_scopes='true',
            prompt='consent'  # Force consent to get refresh token
        )
        
        # Store state in Redis with 10-minute expiry
        self.redis.setex(f"oauth_state:{state}", 600, user_id)
        
        return auth_url
    
    def handle_oauth_callback(self, state: str, code: str) -> dict:
        """
        Step 2: Exchange authorization code for tokens
        """
        # Verify state
        user_id = self.redis.get(f"oauth_state:{state}")
        if not user_id:
            raise SecurityError("Invalid OAuth state")
        
        # Exchange code for tokens
        flow = google_auth_oauthlib.flow.Flow.from_client_config(
            self.client_config,
            scopes=self.SCOPES,
            state=state
        )
        
        flow.fetch_token(code=code)
        
        # Store encrypted tokens
        tokens = {
            'access_token': flow.credentials.token,
            'refresh_token': flow.credentials.refresh_token,
            'expiry': flow.credentials.expiry.isoformat()
        }
        
        self.store_user_tokens(user_id, tokens)
        
        return {'status': 'success', 'user_id': user_id}
```

### 1.3 API Quota Management Strategy

```python
class QuotaManager:
    """
    Manages 10,000 daily quota units across 250 channels
    """
    
    DAILY_QUOTA = 10000
    RESERVED_QUOTA = 1000  # Emergency buffer
    
    # Operation costs in quota units
    OPERATION_COSTS = {
        'video_upload': 1600,
        'thumbnail_set': 50,
        'video_update': 50,
        'channel_list': 1,
        'analytics_query': 1,
        'playlist_insert': 50
    }
    
    async def check_quota_availability(self, operation: str) -> bool:
        """
        Check if operation can proceed within quota limits
        """
        current_usage = await self.get_today_usage()
        operation_cost = self.OPERATION_COSTS.get(operation, 1)
        
        if current_usage + operation_cost > (self.DAILY_QUOTA - self.RESERVED_QUOTA):
            # Try quota redistribution
            return await self.redistribute_quota(operation)
        
        return True
    
    async def redistribute_quota(self, operation: str) -> bool:
        """
        Intelligent quota redistribution across users
        """
        if operation in ['video_upload', 'thumbnail_set']:
            # High priority - borrow from low-usage users
            inactive_quota = await self.get_inactive_user_quota()
            if inactive_quota > self.OPERATION_COSTS[operation]:
                await self.borrow_quota(operation)
                return True
        
        return False
    
    def get_quota_efficient_strategy(self):
        """
        Returns quota-efficient data fetching strategy
        """
        return {
            'primary': 'Use YouTube Reporting API (0 quota)',
            'secondary': 'Cache all responses for 24 hours',
            'tertiary': 'Batch operations where possible',
            'emergency': 'Defer non-critical operations to next day'
        }
```

### 1.4 Rate Limiting Implementation

```python
class YouTubeRateLimiter:
    """
    Prevents hitting YouTube's rate limits
    """
    
    # YouTube's undocumented rate limits (discovered through testing)
    RATE_LIMITS = {
        'videos.insert': {'requests': 100, 'window': 3600},  # 100 per hour
        'videos.update': {'requests': 500, 'window': 3600},  # 500 per hour
        'thumbnails.set': {'requests': 200, 'window': 3600}, # 200 per hour
        'analytics.query': {'requests': 1000, 'window': 60}  # 1000 per minute
    }
    
    async def can_proceed(self, endpoint: str, user_id: str) -> bool:
        """
        Check if request can proceed without hitting rate limits
        """
        key = f"rate_limit:{endpoint}:{user_id}"
        window = self.RATE_LIMITS[endpoint]['window']
        limit = self.RATE_LIMITS[endpoint]['requests']
        
        current = await self.redis.incr(key)
        
        if current == 1:
            await self.redis.expire(key, window)
        
        if current > limit:
            # Calculate backoff time
            ttl = await self.redis.ttl(key)
            raise RateLimitError(f"Rate limit exceeded. Retry in {ttl} seconds")
        
        return True
```

### 1.5 Error Handling Strategy

```python
class YouTubeErrorHandler:
    """
    Comprehensive error handling for YouTube API
    """
    
    ERROR_STRATEGIES = {
        'quotaExceeded': {
            'retry': False,
            'action': 'defer_to_tomorrow',
            'alert': 'critical'
        },
        'rateLimitExceeded': {
            'retry': True,
            'backoff': 'exponential',
            'max_retries': 3,
            'alert': 'warning'
        },
        'authError': {
            'retry': True,
            'action': 'refresh_token',
            'alert': 'info'
        },
        'processingFailure': {
            'retry': True,
            'delay': 300,  # 5 minutes
            'max_retries': 5,
            'alert': 'warning'
        },
        'videoNotFound': {
            'retry': False,
            'action': 'mark_as_deleted',
            'alert': 'info'
        }
    }
    
    async def handle_api_error(self, error: Exception, context: dict) -> dict:
        """
        Handle YouTube API errors with appropriate strategies
        """
        error_type = self.identify_error_type(error)
        strategy = self.ERROR_STRATEGIES.get(error_type)
        
        if strategy['retry']:
            return await self.retry_with_strategy(context, strategy)
        else:
            return await self.execute_fallback(context, strategy)
```

---

## 2. YouTube Data Collection Pipeline

### 2.1 Data Collection Architecture

```python
class YouTubeDataCollector:
    """
    Orchestrates all YouTube data collection
    """
    
    def __init__(self):
        self.reporting_api = YouTubeReportingAPI()  # 0 quota cost
        self.analytics_api = YouTubeAnalyticsAPI()  # Minimal quota
        self.data_api = YouTubeDataAPI()            # High quota cost
        
    async def collect_channel_data(self, user_id: str, channel_id: str):
        """
        Hierarchical data collection strategy
        """
        # Level 1: Reporting API (0 quota) - 95% of data needs
        bulk_data = await self.reporting_api.get_channel_report(channel_id)
        
        # Level 2: Cached data - 4% of data needs
        cached_data = await self.get_cached_metrics(channel_id)
        
        # Level 3: Analytics API (low quota) - 0.9% of data needs
        if self.needs_fresh_analytics(channel_id):
            analytics = await self.analytics_api.get_realtime_metrics(channel_id)
        
        # Level 4: Data API (high quota) - 0.1% of data needs (emergencies only)
        if self.critical_update_needed(channel_id):
            await self.data_api.update_video_metadata(channel_id)
        
        return self.merge_data_sources(bulk_data, cached_data, analytics)
```

### 2.2 Specific API Endpoints Usage

```yaml
# YouTube Data API v3 Endpoints
data_api_endpoints:
  channels.list:
    purpose: "Get channel metadata"
    parameters:
      part: "snippet,statistics,contentDetails"
      id: "{channel_id}"
    quota_cost: 1
    cache_ttl: 86400  # 24 hours
    
  videos.list:
    purpose: "Get video details"
    parameters:
      part: "snippet,statistics,contentDetails"
      id: "{video_ids}"  # Batch up to 50
    quota_cost: 1
    cache_ttl: 3600   # 1 hour
    
  videos.insert:
    purpose: "Upload new video"
    parameters:
      part: "snippet,status"
      body: "{video_metadata}"
    quota_cost: 1600
    rate_limit: "100/hour"
    
  thumbnails.set:
    purpose: "Set video thumbnail"
    parameters:
      videoId: "{video_id}"
    quota_cost: 50
    rate_limit: "200/hour"

# YouTube Analytics API v2 Endpoints
analytics_api_endpoints:
  reports.query:
    purpose: "Get analytics data"
    parameters:
      ids: "channel=={channel_id}"
      startDate: "7daysAgo"
      endDate: "today"
      metrics: "views,estimatedMinutesWatched,averageViewDuration,subscribersGained"
      dimensions: "day"
    quota_cost: 1
    cache_ttl: 1800  # 30 minutes
```

### 2.3 Webhook Integration

```python
class YouTubeWebhookHandler:
    """
    Handles YouTube push notifications for real-time updates
    """
    
    async def setup_channel_webhooks(self, channel_id: str):
        """
        Subscribe to YouTube push notifications
        """
        # YouTube supports webhooks for:
        # - New video uploads
        # - Video updates
        # - Channel updates
        
        subscription = {
            'id': channel_id,
            'type': 'channel',
            'callback_url': f'https://api.ytempire.com/webhooks/youtube/{channel_id}',
            'topic': 'https://www.youtube.com/xml/feeds/videos.xml',
            'verify_token': self.generate_verify_token(channel_id)
        }
        
        # Subscribe via PubSubHubbub
        response = await self.subscribe_to_hub(subscription)
        
        return response
    
    async def handle_webhook(self, channel_id: str, data: dict):
        """
        Process incoming YouTube webhooks
        """
        event_type = data.get('eventType')
        
        if event_type == 'video.upload':
            await self.process_new_video(channel_id, data)
        elif event_type == 'video.update':
            await self.process_video_update(channel_id, data)
        
        # Update cache immediately
        await self.invalidate_cache(channel_id)
```

---

## 3. Analytics Engineer Integration Points

### 3.1 Data Flow to Analytics System

```python
class YouTubeAnalyticsIntegration:
    """
    Integration points for Analytics Engineer
    """
    
    def get_analytics_schema(self):
        """
        Schema for analytics consumption
        """
        return {
            'youtube_metrics': {
                'channel_id': 'VARCHAR(50)',
                'date': 'DATE',
                'views': 'BIGINT',
                'watch_time_minutes': 'DECIMAL(15,2)',
                'subscribers_gained': 'INTEGER',
                'subscribers_lost': 'INTEGER',
                'estimated_revenue': 'DECIMAL(10,2)',
                'impressions': 'BIGINT',
                'click_through_rate': 'DECIMAL(5,4)',
                'average_view_duration': 'DECIMAL(10,2)'
            },
            'video_performance': {
                'video_id': 'VARCHAR(50)',
                'published_at': 'TIMESTAMP',
                'views': 'BIGINT',
                'likes': 'INTEGER',
                'dislikes': 'INTEGER',
                'comments': 'INTEGER',
                'shares': 'INTEGER',
                'watch_time_minutes': 'DECIMAL(15,2)',
                'retention_data': 'JSONB'
            }
        }
    
    def get_data_pipeline_schedule(self):
        """
        When data becomes available
        """
        return {
            'reporting_api_data': {
                'frequency': 'daily',
                'available_at': '14:00 UTC',  # YouTube processes overnight
                'latency': '24-48 hours'
            },
            'analytics_api_data': {
                'frequency': 'hourly',
                'available_at': ':15',  # 15 minutes past hour
                'latency': '3-4 hours'
            },
            'realtime_data': {
                'frequency': 'continuous',
                'available_at': 'immediate',
                'latency': '1-2 minutes'
            }
        }
```

### 3.2 Error Codes and Monitoring

```yaml
youtube_error_codes:
  quota_errors:
    - code: 403
      reason: "quotaExceeded"
      message: "The request cannot be completed because you have exceeded your quota"
      action: "Switch to Reporting API or wait until midnight PT"
      
  auth_errors:
    - code: 401
      reason: "authError"
      message: "Invalid Credentials"
      action: "Refresh OAuth token"
      
    - code: 403
      reason: "forbidden"
      message: "Access forbidden"
      action: "Re-authenticate user"
      
  rate_limit_errors:
    - code: 429
      reason: "rateLimitExceeded"
      message: "Too many requests"
      action: "Exponential backoff"
      
  data_errors:
    - code: 404
      reason: "videoNotFound"
      message: "Video not found"
      action: "Mark as deleted in database"
      
    - code: 400
      reason: "invalidRequest"
      message: "Invalid request parameters"
      action: "Log and fix request"
```

---

## 4. Best Practices and Optimization

### 4.1 Quota Optimization Techniques

1. **Use Reporting API First** (0 quota cost)
   - Covers 95% of analytics needs
   - 24-48 hour data delay acceptable for most metrics

2. **Batch Operations**
   - Videos.list: Request 50 videos at once (same quota cost)
   - Aggregate updates before API calls

3. **Intelligent Caching**
   - Channel data: 24-hour cache
   - Video statistics: 1-hour cache
   - Analytics data: 30-minute cache

4. **Off-Peak Processing**
   - Schedule heavy operations for 2-6 AM PT
   - YouTube quotas reset at midnight PT

### 4.2 Performance Optimization

```python
# Connection pooling for YouTube APIs
youtube_service = build('youtube', 'v3', 
    credentials=credentials,
    cache_discovery=False,  # Disable discovery doc caching
    num_retries=3
)

# Use batch requests where possible
batch = youtube_service.new_batch_http_request()
for video_id in video_ids[:50]:  # Max 50 per batch
    batch.add(youtube_service.videos().list(
        part="statistics",
        id=video_id
    ))
batch.execute()
```

---

## 5. Monitoring and Alerting

### 5.1 Key Metrics to Track

```yaml
youtube_api_metrics:
  quota_usage:
    metric: "youtube_api_quota_used"
    threshold: 8000  # Alert at 80%
    action: "Switch to cache-only mode"
    
  api_latency:
    metric: "youtube_api_response_time_ms"
    threshold: 5000  # 5 seconds
    action: "Check for API degradation"
    
  error_rate:
    metric: "youtube_api_error_rate"
    threshold: 0.05  # 5%
    action: "Investigate error patterns"
    
  token_refresh_failures:
    metric: "youtube_oauth_refresh_failures"
    threshold: 3
    action: "Re-authenticate user"
```

### 5.2 Dashboard Requirements

```sql
-- Queries for Grafana dashboards

-- Quota usage over time
SELECT 
    date_trunc('hour', created_at) as hour,
    SUM(quota_cost) as quota_used,
    10000 - SUM(quota_cost) as quota_remaining
FROM youtube_api_calls
WHERE created_at >= CURRENT_DATE
GROUP BY hour;

-- API success rate by endpoint
SELECT 
    endpoint,
    COUNT(*) FILTER (WHERE status = 'success') * 100.0 / COUNT(*) as success_rate,
    AVG(response_time_ms) as avg_latency,
    COUNT(*) as total_calls
FROM youtube_api_calls
WHERE created_at >= NOW() - INTERVAL '1 hour'
GROUP BY endpoint;

-- Channel sync status
SELECT 
    u.email,
    COUNT(c.channel_id) as channel_count,
    SUM(CASE WHEN c.last_sync < NOW() - INTERVAL '25 hours' THEN 1 ELSE 0 END) as stale_channels,
    MIN(c.last_sync) as oldest_sync
FROM users u
JOIN channels c ON u.user_id = c.user_id
GROUP BY u.email;
```

---

## Next Steps for Analytics Engineer

1. **Implement quota tracking tables** as defined in section 3.1
2. **Set up Grafana dashboards** using queries from section 5.2
3. **Configure alerts** based on thresholds in section 5.1
4. **Create data quality checks** for YouTube data ingestion
5. **Build analytics views** combining YouTube data with internal metrics

This integration guide provides the complete YouTube API implementation details needed for the MVP. The Analytics Engineer should focus on the data flow, monitoring, and optimization aspects while the Backend team handles the OAuth flow and API client implementation.