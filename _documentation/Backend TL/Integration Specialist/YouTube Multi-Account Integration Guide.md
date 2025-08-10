# YouTube Multi-Account Integration Guide

**Document Version**: 1.0  
**For**: Integration Specialist  
**Classification**: CRITICAL - BUSINESS ESSENTIAL  
**Last Updated**: January 2025

---

## 🎯 YouTube Integration Overview

### The Challenge
Managing 15 YouTube accounts (12 active + 3 reserve) to upload 50 videos daily while avoiding quota limits, strikes, and maintaining account health.

### Your Solution Architecture
```
┌─────────────────────────────────────────┐
│        15 YouTube Accounts Pool         │
├─────────────────────────────────────────┤
│  Active Fleet (12)  │  Reserve Fleet (3)│
├─────────────────────────────────────────┤
│    Health Monitor   │   Quota Tracker   │
├─────────────────────────────────────────┤
│         Intelligent Rotation Engine      │
└─────────────────────────────────────────┘
```

---

## 📊 Account Management Strategy

### Account Distribution Model

```python
# Your implementation target
YOUTUBE_ACCOUNTS = {
    "total": 15,
    "active_pool": 12,
    "reserve_pool": 3,
    "daily_video_target": 50,
    "max_per_account_daily": 5,  # Conservative limit
    "quota_per_account": 10000,
    "upload_cost": 1600,  # quota units
    "safety_buffer": 0.20  # Never use last 20% of quota
}
```

### Account Naming Convention
```
ytempire_prod_01 through ytempire_prod_12 (Active)
ytempire_reserve_01 through ytempire_reserve_03 (Emergency)
```

---

## 🔐 OAuth 2.0 Implementation

### Step 1: Initial Setup

```python
# OAuth Configuration - Store in environment variables
YOUTUBE_OAUTH_CONFIG = {
    "client_id": "YOUR_CLIENT_ID",
    "client_secret": "YOUR_CLIENT_SECRET",
    "redirect_uri": "http://localhost:8000/api/v1/auth/youtube/callback",
    "scopes": [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube",
        "https://www.googleapis.com/auth/youtube.readonly",
        "https://www.googleapis.com/auth/youtubepartner"
    ]
}
```

### Step 2: Account Authentication Flow

```python
class YouTubeAccountAuthenticator:
    """Your implementation for OAuth flow"""
    
    async def authenticate_account(self, account_number: int):
        """Authenticate a single YouTube account"""
        
        # 1. Generate OAuth URL
        auth_url = self.generate_oauth_url(account_number)
        print(f"Visit: {auth_url}")
        
        # 2. Wait for callback with code
        auth_code = await self.wait_for_callback()
        
        # 3. Exchange for tokens
        tokens = await self.exchange_code_for_tokens(auth_code)
        
        # 4. Store encrypted in database
        await self.store_tokens_securely(account_number, tokens)
        
        # 5. Test the authentication
        await self.verify_account_access(account_number)
        
        return f"Account ytempire_prod_{account_number:02d} authenticated!"
```

### Step 3: Token Management

```python
class TokenManager:
    """Manage OAuth tokens with automatic refresh"""
    
    async def get_valid_token(self, account_id: str):
        """Always return a valid token, refreshing if needed"""
        
        token_data = await self.get_stored_token(account_id)
        
        # Check if token expires in next 5 minutes
        if self.token_expires_soon(token_data, minutes=5):
            # Refresh proactively
            new_token = await self.refresh_token(
                token_data['refresh_token']
            )
            await self.update_stored_token(account_id, new_token)
            return new_token['access_token']
        
        return token_data['access_token']
```

---

## 📈 Quota Management System

### Daily Quota Allocation

```yaml
quota_distribution:
  total_daily_quota: 150,000  # 15 accounts × 10,000
  
  allocation_strategy:
    uploads: 70%       # 105,000 units = 65 videos
    metadata: 15%      # 22,500 units
    analytics: 10%     # 15,000 units
    emergency: 5%      # 7,500 units (never touch)
  
  per_video_cost:
    upload: 1600
    thumbnail: 50
    metadata: 50
    total: 1700 units
```

### Quota Tracking Implementation

```python
class QuotaManager:
    """Your quota management system"""
    
    def __init__(self):
        self.redis_client = Redis()
        self.quota_limits = self.load_quota_config()
    
    async def check_quota_availability(self, account_id: str) -> dict:
        """Check if account has quota for upload"""
        
        # Get current usage from Redis
        key = f"quota:{account_id}:{datetime.now().strftime('%Y%m%d')}"
        current_usage = int(self.redis_client.get(key) or 0)
        
        # Calculate availability
        daily_limit = 10000
        safety_buffer = 2000  # Keep 20% buffer
        usable_quota = daily_limit - safety_buffer
        
        available = usable_quota - current_usage
        can_upload = available >= 1700  # Cost per video
        
        return {
            "account_id": account_id,
            "current_usage": current_usage,
            "available": available,
            "can_upload": can_upload,
            "health_score": (usable_quota - current_usage) / usable_quota
        }
    
    async def consume_quota(self, account_id: str, amount: int):
        """Consume quota units after successful operation"""
        
        key = f"quota:{account_id}:{datetime.now().strftime('%Y%m%d')}"
        new_usage = self.redis_client.incrby(key, amount)
        
        # Set expiry to reset at midnight PST
        self.redis_client.expireat(key, self.get_next_reset_time())
        
        # Alert if approaching limit
        if new_usage > 8000:
            await self.send_quota_alert(account_id, new_usage)
        
        return new_usage
```

---

## 🔄 Intelligent Account Rotation

### Account Selection Algorithm

```python
class AccountRotationEngine:
    """Your intelligent rotation system"""
    
    async def select_best_account(self) -> str:
        """Select optimal account for next upload"""
        
        accounts = await self.get_all_accounts()
        
        # Score each account
        scored_accounts = []
        for account in accounts:
            score = await self.calculate_account_score(account)
            scored_accounts.append((account, score))
        
        # Sort by score (highest first)
        scored_accounts.sort(key=lambda x: x[1], reverse=True)
        
        # Return best available account
        best_account = scored_accounts[0][0]
        
        # Log selection
        await self.log_account_selection(best_account)
        
        return best_account['id']
    
    async def calculate_account_score(self, account: dict) -> float:
        """Calculate health score for account selection"""
        
        score = 100.0
        
        # Factor 1: Quota availability (40% weight)
        quota_available = account['quota_available'] / 10000
        score *= (0.6 + 0.4 * quota_available)
        
        # Factor 2: Daily uploads (30% weight)
        uploads_today = account['uploads_today']
        upload_factor = max(0, (5 - uploads_today) / 5)
        score *= (0.7 + 0.3 * upload_factor)
        
        # Factor 3: Recent errors (20% weight)
        if account['last_error_time']:
            hours_since_error = (datetime.now() - account['last_error_time']).hours
            error_factor = min(1.0, hours_since_error / 24)
            score *= (0.8 + 0.2 * error_factor)
        
        # Factor 4: Time since last use (10% weight)
        if account['last_upload_time']:
            minutes_since_use = (datetime.now() - account['last_upload_time']).seconds / 60
            rest_factor = min(1.0, minutes_since_use / 60)  # Best after 1 hour rest
            score *= (0.9 + 0.1 * rest_factor)
        
        # Penalty for reserve accounts (use only if needed)
        if account['is_reserve']:
            score *= 0.1  # Heavy penalty
        
        return score
```

### Batch Upload Distribution

```python
class BatchUploadDistributor:
    """Distribute multiple videos across accounts"""
    
    async def distribute_batch(self, videos: list) -> dict:
        """Distribute video batch across available accounts"""
        
        distribution = {}
        available_accounts = await self.get_available_accounts()
        
        # Group videos by priority
        priority_videos = sorted(videos, key=lambda x: x['priority'], reverse=True)
        
        # Distribute round-robin with limits
        account_index = 0
        for video in priority_videos:
            account = available_accounts[account_index % len(available_accounts)]
            
            # Check account daily limit
            if account['uploads_today'] >= 5:
                # Skip this account, try next
                account_index += 1
                if account_index >= len(available_accounts):
                    # All accounts full, queue for tomorrow
                    await self.queue_for_tomorrow(video)
                    continue
                account = available_accounts[account_index % len(available_accounts)]
            
            # Assign video to account
            if account['id'] not in distribution:
                distribution[account['id']] = []
            distribution[account['id']].append(video)
            
            # Update account usage
            account['uploads_today'] += 1
            account_index += 1
        
        return distribution
```

---

## 🚨 Error Handling & Recovery

### Error Classification & Actions

```python
ERROR_HANDLING_MATRIX = {
    "quotaExceeded": {
        "action": "switch_account",
        "retry": True,
        "wait_time": 0,
        "alert": True,
        "severity": "warning"
    },
    "uploadLimitExceeded": {
        "action": "defer_to_tomorrow",
        "retry": False,
        "wait_time": 86400,
        "alert": True,
        "severity": "warning"
    },
    "authenticationError": {
        "action": "refresh_token",
        "retry": True,
        "wait_time": 5,
        "alert": False,
        "severity": "info"
    },
    "rateLimitExceeded": {
        "action": "exponential_backoff",
        "retry": True,
        "wait_time": 60,
        "alert": False,
        "severity": "info"
    },
    "invalidVideo": {
        "action": "reprocess_video",
        "retry": True,
        "wait_time": 0,
        "alert": True,
        "severity": "error"
    },
    "duplicateVideo": {
        "action": "skip",
        "retry": False,
        "wait_time": 0,
        "alert": True,
        "severity": "warning"
    }
}
```

### Recovery Implementation

```python
class YouTubeErrorHandler:
    """Your error recovery system"""
    
    async def handle_upload_error(self, error: Exception, video_id: str, account_id: str):
        """Handle YouTube API errors with appropriate recovery"""
        
        error_type = self.classify_error(error)
        strategy = ERROR_HANDLING_MATRIX.get(error_type)
        
        if not strategy:
            # Unknown error, log and escalate
            await self.log_unknown_error(error, video_id)
            raise error
        
        # Execute recovery strategy
        if strategy['action'] == 'switch_account':
            # Mark current account as unhealthy
            await self.mark_account_unhealthy(account_id)
            
            # Find alternative account
            new_account = await self.select_best_account(exclude=account_id)
            
            # Retry with new account
            return await self.retry_upload(video_id, new_account)
        
        elif strategy['action'] == 'refresh_token':
            # Refresh OAuth token
            await self.refresh_account_token(account_id)
            
            # Retry with same account
            return await self.retry_upload(video_id, account_id)
        
        elif strategy['action'] == 'exponential_backoff':
            # Calculate wait time
            wait_time = strategy['wait_time'] * (2 ** self.get_retry_count(video_id))
            
            # Wait and retry
            await asyncio.sleep(wait_time)
            return await self.retry_upload(video_id, account_id)
        
        # Send alerts if needed
        if strategy['alert']:
            await self.send_error_alert(error_type, video_id, account_id, strategy['severity'])
```

---

## 📊 Account Health Monitoring

### Health Metrics Dashboard

```python
class AccountHealthMonitor:
    """Monitor and maintain account health"""
    
    HEALTH_METRICS = {
        "quota_usage": {"weight": 0.3, "threshold": 0.8},
        "error_rate": {"weight": 0.3, "threshold": 0.05},
        "upload_success": {"weight": 0.2, "threshold": 0.95},
        "api_latency": {"weight": 0.1, "threshold": 2000},
        "daily_uploads": {"weight": 0.1, "threshold": 5}
    }
    
    async def calculate_health_score(self, account_id: str) -> dict:
        """Calculate comprehensive health score"""
        
        metrics = await self.collect_account_metrics(account_id)
        score = 100.0
        breakdown = {}
        
        for metric_name, config in self.HEALTH_METRICS.items():
            metric_value = metrics[metric_name]
            metric_score = self.calculate_metric_score(metric_value, config)
            
            score *= (1 - config['weight'] + config['weight'] * metric_score)
            breakdown[metric_name] = metric_score
        
        return {
            "account_id": account_id,
            "overall_score": score,
            "breakdown": breakdown,
            "status": self.get_status_from_score(score),
            "recommendations": self.get_recommendations(breakdown)
        }
    
    def get_status_from_score(self, score: float) -> str:
        if score >= 90:
            return "healthy"
        elif score >= 70:
            return "warning"
        elif score >= 50:
            return "degraded"
        else:
            return "critical"
```

---

## 🎯 Daily Upload Schedule

### Optimal Upload Distribution

```yaml
upload_schedule:
  morning_batch:  # 6:00 AM - 10:00 AM PST
    accounts: [prod_01, prod_02, prod_03, prod_04]
    videos: 20
    reasoning: "Peak user activity, fresh content"
  
  afternoon_batch:  # 12:00 PM - 4:00 PM PST
    accounts: [prod_05, prod_06, prod_07, prod_08]
    videos: 20
    reasoning: "Lunch break viewership"
  
  evening_batch:  # 6:00 PM - 10:00 PM PST
    accounts: [prod_09, prod_10, prod_11, prod_12]
    videos: 10
    reasoning: "Prime time engagement"
  
  reserve_accounts:
    usage: "Only when primary accounts hit limits"
    threshold: "When any account health < 50%"
```

---

## 🔧 Implementation Checklist

### Week 1 Tasks

#### Day 1-2: Setup
- [ ] Configure OAuth for all 15 accounts
- [ ] Store tokens securely in database
- [ ] Test token refresh mechanism
- [ ] Verify all accounts can upload

#### Day 3-4: Core Implementation
- [ ] Implement quota tracking in Redis
- [ ] Build account rotation algorithm
- [ ] Create health monitoring system
- [ ] Set up error handling matrix

#### Day 5: Testing
- [ ] Test with 10 video uploads
- [ ] Verify account rotation works
- [ ] Check quota tracking accuracy
- [ ] Validate error recovery

### Critical Functions to Implement

```python
# Your core interface
class YouTubeIntegrationManager:
    """Main interface for YouTube operations"""
    
    async def upload_video(self, video_path: str, metadata: dict) -> str:
        """Upload a video using best available account"""
        
        # 1. Select best account
        account = await self.select_best_account()
        
        # 2. Check quota
        if not await self.check_quota(account):
            account = await self.select_alternative_account()
        
        # 3. Upload with error handling
        try:
            youtube_id = await self.execute_upload(account, video_path, metadata)
            
            # 4. Update tracking
            await self.update_account_metrics(account, success=True)
            
            return youtube_id
            
        except Exception as e:
            # 5. Handle errors
            return await self.handle_upload_error(e, video_path, metadata, account)
```

---

## 📈 Monitoring & Alerts

### Key Metrics to Track

```yaml
monitoring_dashboard:
  real_time:
    - Current quota usage per account
    - Active uploads in progress
    - Account health scores
    - Error rate (last hour)
  
  daily_summary:
    - Total videos uploaded
    - Quota efficiency (used vs available)
    - Account distribution balance
    - Average upload time
  
  alerts:
    critical:
      - Any account banned/suspended
      - All accounts at >90% quota
      - Upload success rate <90%
    
    warning:
      - Single account at >80% quota
      - Account health score <70%
      - Unusual error patterns
```

---

## 🚀 Pro Tips for Success

### 1. **Always Have Spare Capacity**
Never use more than 80% of available quota. Keep reserve accounts truly reserved.

### 2. **Monitor Natural Patterns**
Upload at varied times, with varied content. Avoid patterns that look automated.

### 3. **Respect Rate Limits**
Even within quota, respect unwritten rate limits. Space uploads by 5+ minutes per account.

### 4. **Track Everything**
Log every API call, every error, every success. Data helps prevent future issues.

### 5. **Plan for Failures**
Always have a Plan B. If primary accounts fail, reserve accounts activate automatically.

---

## 📞 Support & Escalation

### When to Escalate
- Any account suspension or ban
- Quota calculation discrepancies
- Systematic upload failures (>10%)
- Authentication issues lasting >1 hour

### Escalation Path
1. **First**: Check this documentation
2. **Second**: Backend Team Lead
3. **Third**: CTO (critical issues only)
4. **Emergency**: 24/7 on-call hotline

---

**Remember**: You're managing the crown jewels of our platform. These YouTube accounts are our direct connection to millions of viewers and our path to $50M ARR. Handle with care, optimize aggressively, and always have a backup plan!