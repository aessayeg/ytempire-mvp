# YTEMPIRE Integration Specialist Guide - Part 1: Core Integrations

**Document Version**: 1.0  
**Date**: January 2025  
**Author**: Backend Team Lead  
**Audience**: Integration Specialist  
**Scope**: YouTube API, OAuth 2.0, and Quota Management

---

## Table of Contents
1. [YouTube API Integration Guide](#1-youtube-api-integration-guide)
2. [OAuth 2.0 Implementation Guide](#2-oauth-20-implementation-guide)
3. [API Quota Management Strategy](#3-api-quota-management-strategy)
4. [Service Fallback Strategies](#4-service-fallback-strategies)

---

## 1. YouTube API Integration Guide

### 1.1 Overview and Architecture

As the Integration Specialist, you'll manage **15 YouTube accounts** (12 active + 3 reserve) to support our target of **50 videos per day** while maintaining strict quota compliance.

```python
# YouTube Integration Configuration
YOUTUBE_CONFIG = {
    "total_accounts": 15,
    "active_accounts": 12,
    "reserve_accounts": 3,
    "daily_video_target": 50,
    "videos_per_account_limit": 5,
    "api_version": "v3",
    "quota_per_project": 10000,  # Daily limit
    "upload_cost": 1600,  # Quota units per upload
}
```

### 1.2 Multi-Account Management Implementation

#### Account Pool Architecture

```python
class YouTubeAccountPool:
    """
    Manages 15 YouTube accounts with health scoring and rotation
    Your primary responsibility as Integration Specialist
    """
    
    def __init__(self):
        self.accounts = self._initialize_accounts()
        self.health_scorer = AccountHealthScorer()
        self.quota_tracker = QuotaTracker()
        
    def _initialize_accounts(self) -> list:
        """Initialize all 15 YouTube accounts"""
        accounts = []
        
        # Production accounts (1-12)
        for i in range(1, 13):
            accounts.append({
                "id": f"ytempire_prod_{i:02d}",
                "type": "production",
                "daily_limit": 5,
                "quota_allocated": 7000,  # 70% of 10k
                "health_score": 1.0,
                "status": "active"
            })
        
        # Reserve accounts (13-15)
        for i in range(13, 16):
            accounts.append({
                "id": f"ytempire_reserve_{i:02d}",
                "type": "reserve",
                "daily_limit": 2,
                "quota_allocated": 3000,  # Emergency use only
                "health_score": 1.0,
                "status": "standby"
            })
        
        return accounts
    
    async def select_optimal_account(self, priority: int = 5) -> dict:
        """
        Select the best account for upload based on multiple factors
        This is called for EVERY video upload
        """
        
        # Get accounts with capacity
        available = await self._get_available_accounts()
        
        if not available:
            # Activate reserve account
            return await self._activate_reserve_account()
        
        # Score each account
        scored_accounts = []
        for account in available:
            score = await self._calculate_account_score(account)
            scored_accounts.append((account, score))
        
        # Sort by score (highest first)
        scored_accounts.sort(key=lambda x: x[1], reverse=True)
        
        selected = scored_accounts[0][0]
        
        # Log selection
        await self._log_account_selection(selected, priority)
        
        return selected
    
    async def _calculate_account_score(self, account: dict) -> float:
        """
        Calculate account health score
        
        Factors:
        - Available quota (40% weight)
        - Upload success rate (30% weight)
        - Time since last upload (20% weight)
        - Account age (10% weight)
        """
        
        score = 0.0
        
        # Quota availability
        quota_used = await self.quota_tracker.get_used_quota(account["id"])
        quota_available = account["quota_allocated"] - quota_used
        quota_score = (quota_available / account["quota_allocated"]) * 0.4
        score += quota_score
        
        # Success rate
        success_rate = await self._get_success_rate(account["id"])
        score += success_rate * 0.3
        
        # Time distribution
        hours_since_upload = await self._get_hours_since_upload(account["id"])
        time_score = min(hours_since_upload / 24, 1.0) * 0.2
        score += time_score
        
        # Account age bonus
        age_days = await self._get_account_age(account["id"])
        age_score = min(age_days / 30, 1.0) * 0.1
        score += age_score
        
        return score
```

### 1.3 YouTube API Client Implementation

```python
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
import asyncio
import aiohttp

class YouTubeAPIClient:
    """
    Core YouTube API client with retry logic and error handling
    """
    
    def __init__(self, account_id: str):
        self.account_id = account_id
        self.service = None
        self.credentials = None
        self._initialize_service()
        
    def _initialize_service(self):
        """Initialize YouTube API service"""
        
        # Load credentials for account
        self.credentials = self._load_credentials(self.account_id)
        
        # Build service
        self.service = build(
            'youtube', 
            'v3',
            credentials=self.credentials,
            cache_discovery=False
        )
    
    async def upload_video(self, video_data: dict) -> dict:
        """
        Upload video with resumable support
        
        Args:
            video_data: {
                'file_path': str,
                'title': str,
                'description': str,
                'tags': list,
                'category_id': str,
                'privacy_status': str
            }
        """
        
        try:
            # Prepare video metadata
            body = {
                'snippet': {
                    'title': video_data['title'][:100],  # YouTube limit
                    'description': video_data['description'][:5000],
                    'tags': video_data['tags'][:500],  # 500 chars total
                    'categoryId': video_data['category_id'],
                    'defaultLanguage': 'en',
                    'defaultAudioLanguage': 'en'
                },
                'status': {
                    'privacyStatus': video_data.get('privacy_status', 'public'),
                    'selfDeclaredMadeForKids': False,
                    'embeddable': True,
                    'publicStatsViewable': True
                }
            }
            
            # Create media upload
            media = MediaFileUpload(
                video_data['file_path'],
                chunksize=50 * 1024 * 1024,  # 50MB chunks
                resumable=True,
                mimetype='video/mp4'
            )
            
            # Create insert request
            insert_request = self.service.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            # Execute with progress tracking
            response = await self._execute_resumable_upload(
                insert_request,
                video_data.get('progress_callback')
            )
            
            return {
                'success': True,
                'youtube_id': response['id'],
                'youtube_url': f"https://youtube.com/watch?v={response['id']}",
                'account_id': self.account_id
            }
            
        except Exception as e:
            return await self._handle_upload_error(e, video_data)
    
    async def _execute_resumable_upload(self, request, progress_callback=None):
        """Execute resumable upload with progress tracking"""
        
        response = None
        error_count = 0
        max_retries = 5
        
        while response is None and error_count < max_retries:
            try:
                status, response = request.next_chunk()
                
                if status and progress_callback:
                    progress = int(status.progress() * 100)
                    await progress_callback(progress)
                    
            except HttpError as e:
                if e.resp.status in [500, 502, 503, 504]:
                    # Retry with exponential backoff
                    error_count += 1
                    await asyncio.sleep(2 ** error_count)
                else:
                    raise
                    
        return response
    
    async def update_video_metadata(self, video_id: str, updates: dict) -> dict:
        """Update video metadata after upload"""
        
        body = {
            'id': video_id,
            'snippet': updates.get('snippet', {}),
            'status': updates.get('status', {})
        }
        
        request = self.service.videos().update(
            part='snippet,status',
            body=body
        )
        
        response = request.execute()
        return response
    
    async def set_thumbnail(self, video_id: str, thumbnail_path: str) -> dict:
        """Set custom thumbnail for video"""
        
        media = MediaFileUpload(
            thumbnail_path,
            mimetype='image/jpeg'
        )
        
        request = self.service.thumbnails().set(
            videoId=video_id,
            media_body=media
        )
        
        response = request.execute()
        return response
```

### 1.4 Error Handling Matrix

```python
class YouTubeErrorHandler:
    """
    Comprehensive error handling for YouTube API
    """
    
    ERROR_STRATEGIES = {
        "quotaExceeded": {
            "action": "switch_account",
            "retry": True,
            "alert_level": "critical",
            "fallback": self._switch_to_next_account
        },
        "uploadLimitExceeded": {
            "action": "defer_24_hours",
            "retry": False,
            "alert_level": "warning",
            "fallback": self._defer_upload
        },
        "rateLimitExceeded": {
            "action": "exponential_backoff",
            "retry": True,
            "max_retries": 5,
            "fallback": self._apply_backoff
        },
        "authError": {
            "action": "refresh_token",
            "retry": True,
            "alert_level": "high",
            "fallback": self._refresh_oauth_token
        },
        "processingFailed": {
            "action": "retry_upload",
            "retry": True,
            "max_retries": 3,
            "fallback": self._retry_upload
        },
        "duplicateUpload": {
            "action": "skip",
            "retry": False,
            "alert_level": "info",
            "fallback": self._handle_duplicate
        }
    }
    
    async def handle_error(self, error: Exception, context: dict) -> dict:
        """
        Main error handling dispatcher
        
        Args:
            error: The exception that occurred
            context: Context about the operation (account_id, video_id, etc.)
        """
        
        error_type = self._classify_error(error)
        strategy = self.ERROR_STRATEGIES.get(
            error_type,
            self._get_default_strategy()
        )
        
        # Log error with full context
        await self._log_error(error, error_type, context)
        
        # Send alert if needed
        if strategy.get("alert_level"):
            await self._send_alert(error_type, strategy["alert_level"], context)
        
        # Execute fallback strategy
        if strategy["retry"]:
            return await strategy["fallback"](error, context)
        else:
            return {
                "success": False,
                "error": str(error),
                "strategy": strategy["action"]
            }
```

---

## 2. OAuth 2.0 Implementation Guide

### 2.1 OAuth Flow Architecture

```python
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
import pickle
import os

class YouTubeOAuthManager:
    """
    Manages OAuth 2.0 flow for all 15 YouTube accounts
    Critical for maintaining authenticated access
    """
    
    OAUTH_CONFIG = {
        "client_secrets_file": "config/client_secrets.json",
        "scopes": [
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube",
            "https://www.googleapis.com/auth/youtube.force-ssl",
            "https://www.googleapis.com/auth/youtubepartner"
        ],
        "redirect_uri": "http://localhost:8000/api/v1/auth/youtube/callback",
        "access_type": "offline",
        "prompt": "consent"
    }
    
    def __init__(self):
        self.flows = {}  # Active OAuth flows
        self.credentials_store = CredentialsStore()
        
    async def initiate_oauth_flow(self, account_id: str) -> str:
        """
        Start OAuth flow for a YouTube account
        
        Returns:
            Authorization URL for user to visit
        """
        
        flow = Flow.from_client_secrets_file(
            self.OAUTH_CONFIG["client_secrets_file"],
            scopes=self.OAUTH_CONFIG["scopes"]
        )
        
        flow.redirect_uri = self.OAUTH_CONFIG["redirect_uri"]
        
        # Generate authorization URL
        authorization_url, state = flow.authorization_url(
            access_type=self.OAUTH_CONFIG["access_type"],
            prompt=self.OAUTH_CONFIG["prompt"],
            include_granted_scopes='true',
            state=account_id  # Use account_id as state
        )
        
        # Store flow for callback
        self.flows[state] = flow
        
        # Log OAuth initiation
        await self._log_oauth_start(account_id, state)
        
        return authorization_url
    
    async def handle_oauth_callback(self, state: str, code: str) -> dict:
        """
        Handle OAuth callback and store credentials
        
        Args:
            state: State parameter (account_id)
            code: Authorization code from Google
        """
        
        if state not in self.flows:
            raise ValueError(f"Invalid state: {state}")
        
        flow = self.flows[state]
        account_id = state
        
        try:
            # Exchange code for tokens
            flow.fetch_token(code=code)
            
            # Get credentials
            credentials = flow.credentials
            
            # Store encrypted credentials
            await self.credentials_store.save(account_id, credentials)
            
            # Clean up flow
            del self.flows[state]
            
            # Test credentials
            test_result = await self._test_credentials(account_id, credentials)
            
            return {
                "success": True,
                "account_id": account_id,
                "channel_id": test_result.get("channel_id"),
                "channel_title": test_result.get("channel_title"),
                "expires_in": credentials.expiry
            }
            
        except Exception as e:
            await self._log_oauth_error(account_id, str(e))
            raise
    
    async def refresh_credentials(self, account_id: str) -> Credentials:
        """
        Refresh expired credentials
        Called automatically 30 minutes before expiry
        """
        
        # Load existing credentials
        credentials = await self.credentials_store.load(account_id)
        
        if not credentials:
            raise ValueError(f"No credentials found for {account_id}")
        
        # Check if refresh needed
        if credentials.expired or not credentials.valid:
            try:
                # Refresh the token
                credentials.refresh(Request())
                
                # Save updated credentials
                await self.credentials_store.save(account_id, credentials)
                
                # Log refresh
                await self._log_token_refresh(account_id)
                
            except Exception as e:
                # Handle refresh failure
                await self._handle_refresh_failure(account_id, e)
                raise
        
        return credentials
```

### 2.2 Credentials Storage and Security

```python
import cryptography
from cryptography.fernet import Fernet
import json
import aiofiles

class CredentialsStore:
    """
    Secure storage for OAuth credentials
    Implements encryption at rest
    """
    
    def __init__(self):
        self.encryption_key = self._load_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        self.storage_path = "/secure/credentials"
        
    async def save(self, account_id: str, credentials: Credentials):
        """
        Save credentials with encryption
        """
        
        # Convert credentials to dict
        cred_data = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes,
            'expiry': credentials.expiry.isoformat() if credentials.expiry else None
        }
        
        # Encrypt data
        encrypted = self.cipher.encrypt(
            json.dumps(cred_data).encode()
        )
        
        # Save to file
        file_path = f"{self.storage_path}/{account_id}.enc"
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(encrypted)
        
        # Also save to database (encrypted)
        await self._save_to_database(account_id, encrypted)
    
    async def load(self, account_id: str) -> Credentials:
        """
        Load and decrypt credentials
        """
        
        # Try file first
        file_path = f"{self.storage_path}/{account_id}.enc"
        
        if os.path.exists(file_path):
            async with aiofiles.open(file_path, 'rb') as f:
                encrypted = await f.read()
        else:
            # Fallback to database
            encrypted = await self._load_from_database(account_id)
            
        if not encrypted:
            return None
        
        # Decrypt
        decrypted = self.cipher.decrypt(encrypted)
        cred_data = json.loads(decrypted.decode())
        
        # Reconstruct credentials
        credentials = Credentials(
            token=cred_data['token'],
            refresh_token=cred_data['refresh_token'],
            token_uri=cred_data['token_uri'],
            client_id=cred_data['client_id'],
            client_secret=cred_data['client_secret'],
            scopes=cred_data['scopes']
        )
        
        if cred_data['expiry']:
            credentials.expiry = datetime.fromisoformat(cred_data['expiry'])
        
        return credentials
```

### 2.3 Automated Token Refresh System

```python
class TokenRefreshScheduler:
    """
    Proactively refreshes tokens before expiry
    Runs as background task
    """
    
    def __init__(self):
        self.oauth_manager = YouTubeOAuthManager()
        self.refresh_buffer = 1800  # 30 minutes before expiry
        self.check_interval = 300  # Check every 5 minutes
        
    async def start_refresh_monitor(self):
        """
        Start background token refresh monitor
        """
        
        while True:
            try:
                await self._check_and_refresh_tokens()
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Token refresh monitor error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
    
    async def _check_and_refresh_tokens(self):
        """
        Check all accounts and refresh tokens as needed
        """
        
        # Get all account IDs
        accounts = await self._get_all_accounts()
        
        for account_id in accounts:
            try:
                # Load credentials
                credentials = await self.oauth_manager.credentials_store.load(account_id)
                
                if not credentials:
                    logger.warning(f"No credentials for {account_id}")
                    continue
                
                # Check if refresh needed
                if self._needs_refresh(credentials):
                    logger.info(f"Refreshing token for {account_id}")
                    await self.oauth_manager.refresh_credentials(account_id)
                    
            except Exception as e:
                logger.error(f"Failed to refresh {account_id}: {e}")
                await self._alert_refresh_failure(account_id, e)
    
    def _needs_refresh(self, credentials: Credentials) -> bool:
        """
        Check if credentials need refresh
        """
        
        if not credentials.expiry:
            return False
            
        time_until_expiry = (credentials.expiry - datetime.now()).total_seconds()
        return time_until_expiry <= self.refresh_buffer
```

---

## 3. API Quota Management Strategy

### 3.1 Quota Architecture

```python
class QuotaManagementSystem:
    """
    Comprehensive quota management for 15 YouTube accounts
    Critical for preventing API limit violations
    """
    
    # YouTube API quota costs
    QUOTA_COSTS = {
        "videos.insert": 1600,      # Upload video
        "videos.update": 50,        # Update metadata
        "videos.delete": 50,        # Delete video
        "thumbnails.set": 50,       # Set thumbnail
        "playlists.insert": 50,     # Create playlist
        "playlistItems.insert": 50, # Add to playlist
        "channels.list": 1,         # Read channel info
        "videos.list": 1,           # List videos
        "search.list": 100,         # Search
        "commentThreads.list": 1,  # Read comments
        "captions.insert": 400,     # Add captions
        "captions.update": 450,     # Update captions
    }
    
    def __init__(self):
        self.daily_limit = 10000  # Per project
        self.redis_client = Redis()
        self.alert_manager = AlertManager()
        
    async def check_quota_availability(self, account_id: str, operation: str) -> dict:
        """
        Check if quota is available for operation
        
        Returns:
            {
                'available': bool,
                'current_usage': int,
                'operation_cost': int,
                'remaining': int,
                'reset_time': datetime
            }
        """
        
        operation_cost = self.QUOTA_COSTS.get(operation, 0)
        current_usage = await self._get_current_usage(account_id)
        remaining = self.daily_limit - current_usage
        
        return {
            'available': remaining >= operation_cost,
            'current_usage': current_usage,
            'operation_cost': operation_cost,
            'remaining': remaining,
            'reset_time': self._get_reset_time(),
            'percentage_used': (current_usage / self.daily_limit) * 100
        }
    
    async def consume_quota(self, account_id: str, operation: str) -> bool:
        """
        Consume quota for an operation
        Returns False if quota exceeded
        """
        
        # Check availability first
        check = await self.check_quota_availability(account_id, operation)
        
        if not check['available']:
            await self._handle_quota_exceeded(account_id, operation)
            return False
        
        # Consume quota atomically
        cost = self.QUOTA_COSTS[operation]
        new_usage = await self._increment_usage(account_id, cost)
        
        # Check thresholds
        await self._check_alert_thresholds(account_id, new_usage)
        
        # Log consumption
        await self._log_quota_consumption(account_id, operation, cost, new_usage)
        
        return True
    
    async def _increment_usage(self, account_id: str, cost: int) -> int:
        """
        Atomically increment quota usage
        """
        
        key = f"quota:{account_id}:{datetime.now().strftime('%Y%m%d')}"
        
        # Increment and get new value
        new_value = await self.redis_client.incrby(key, cost)
        
        # Set expiry to end of day Pacific Time
        await self.redis_client.expireat(key, self._get_reset_timestamp())
        
        return new_value
    
    async def _check_alert_thresholds(self, account_id: str, current_usage: int):
        """
        Check and alert on quota thresholds
        """
        
        percentage = (current_usage / self.daily_limit) * 100
        
        if percentage >= 90:
            # Critical alert - stop using account
            await self.alert_manager.send_critical({
                'type': 'quota_critical',
                'account_id': account_id,
                'usage': percentage,
                'action': 'STOP_USING_ACCOUNT'
            })
            
        elif percentage >= 80:
            # Warning - prepare to switch
            await self.alert_manager.send_warning({
                'type': 'quota_warning',
                'account_id': account_id,
                'usage': percentage,
                'action': 'PREPARE_ACCOUNT_SWITCH'
            })
            
        elif percentage >= 70:
            # Info - monitor closely
            await self.alert_manager.send_info({
                'type': 'quota_info',
                'account_id': account_id,
                'usage': percentage
            })
```

### 3.2 Quota Distribution Strategy

```python
class QuotaDistributionStrategy:
    """
    Intelligent quota distribution across accounts and time
    """
    
    def __init__(self):
        self.total_accounts = 15
        self.daily_videos = 50
        self.uploads_per_account = 5  # Conservative limit
        
    async def calculate_daily_distribution(self) -> dict:
        """
        Calculate optimal quota distribution for the day
        """
        
        distribution = {
            'time_blocks': [],
            'account_assignments': {},
            'quota_allocations': {}
        }
        
        # Divide day into 3 blocks for quota distribution
        time_blocks = [
            {'name': 'morning', 'hours': '06:00-12:00', 'videos': 20},
            {'name': 'afternoon', 'hours': '12:00-18:00', 'videos': 20},
            {'name': 'evening', 'hours': '18:00-23:00', 'videos': 10}
        ]
        
        distribution['time_blocks'] = time_blocks
        
        # Assign accounts to time blocks
        for block in time_blocks:
            accounts_needed = math.ceil(block['videos'] / self.uploads_per_account)
            assigned_accounts = await self._select_accounts_for_block(
                accounts_needed,
                block['name']
            )
            distribution['account_assignments'][block['name']] = assigned_accounts
        
        # Calculate quota allocation per account
        for account_id in await self._get_all_accounts():
            allocation = await self._calculate_account_allocation(account_id)
            distribution['quota_allocations'][account_id] = allocation
        
        return distribution
    
    async def _calculate_account_allocation(self, account_id: str) -> dict:
        """
        Calculate quota allocation for specific account
        """
        
        return {
            'upload_quota': 7000,      # 70% for uploads (4 videos)
            'metadata_quota': 1000,     # 10% for metadata operations
            'thumbnail_quota': 500,     # 5% for thumbnails
            'analytics_quota': 500,     # 5% for reading analytics
            'buffer_quota': 1000,       # 10% emergency buffer
            'total': 10000
        }
```

### 3.3 Quota Monitoring Dashboard

```python
class QuotaMonitoringDashboard:
    """
    Real-time quota monitoring and visualization
    """
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.websocket_server = WebSocketServer()
        
    async def get_quota_dashboard_data(self) -> dict:
        """
        Get current quota status for all accounts
        """
        
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'accounts': [],
            'total_usage': 0,
            'total_capacity': 150000,  # 15 accounts × 10k
            'alerts': []
        }
        
        for account_id in await self._get_all_accounts():
            account_data = await self._get_account_quota_status(account_id)
            dashboard_data['accounts'].append(account_data)
            dashboard_data['total_usage'] += account_data['usage']
            
            # Check for alerts
            if account_data['percentage'] >= 80:
                dashboard_data['alerts'].append({
                    'account_id': account_id,
                    'level': 'critical' if account_data['percentage'] >= 90 else 'warning',
                    'message': f"Account {account_id} at {account_data['percentage']}% quota"
                })
        
        # Calculate overall health
        dashboard_data['health_score'] = self._calculate_health_score(dashboard_data)
        
        return dashboard_data
    
    async def _get_account_quota_status(self, account_id: str) -> dict:
        """
        Get detailed quota status for an account
        """
        
        usage = await self._get_current_usage(account_id)
        
        return {
            'account_id': account_id,
            'usage': usage,
            'limit': 10000,
            'remaining': 10000 - usage,
            'percentage': (usage / 10000) * 100,
            'reset_time': self._get_reset_time(),
            'status': self._get_status_color(usage),
            'uploads_today': await self._get_upload_count(account_id),
            'last_operation': await self._get_last_operation(account_id)
        }
    
    def _get_status_color(self, usage: int) -> str:
        """Get status color based on usage"""
        
        percentage = (usage / 10000) * 100
        
        if percentage >= 90:
            return 'red'
        elif percentage >= 80:
            return 'orange'
        elif percentage >= 70:
            return 'yellow'
        else:
            return 'green'
```

---

## 4. Service Fallback Strategies

### 4.1 Fallback Architecture

```python
class ServiceFallbackManager:
    """
    Manages fallback strategies for all integrated services
    Ensures system resilience when primary services fail
    """
    
    FALLBACK_CHAINS = {
        'voice_synthesis': [
            {'service': 'elevenlabs', 'cost': 0.50, 'quality': 'premium'},
            {'service': 'google_tts', 'cost': 0.20, 'quality': 'standard'},
            {'service': 'aws_polly', 'cost': 0.15, 'quality': 'basic'},
            {'service': 'local_tts', 'cost': 0.01, 'quality': 'emergency'}
        ],
        'script_generation': [
            {'service': 'openai_gpt4', 'cost': 0.80, 'quality': 'premium'},
            {'service': 'openai_gpt35', 'cost': 0.40, 'quality': 'standard'},
            {'service': 'claude', 'cost': 0.60, 'quality': 'good'},
            {'service': 'template_engine', 'cost': 0.05, 'quality': 'basic'}
        ],
        'video_upload': [
            {'service': 'youtube_primary', 'accounts': 12},
            {'service': 'youtube_reserve', 'accounts': 3},
            {'service': 'queue_for_retry', 'delay': 3600},
            {'service': 'manual_intervention', 'alert': 'critical'}
        ]
    }
    
    async def execute_with_fallback(
        self,
        service_type: str,
        operation: callable,
        context: dict
    ) -> dict:
        """
        Execute operation with automatic fallback
        """
        
        fallback_chain = self.FALLBACK_CHAINS.get(service_type, [])
        last_error = None
        
        for i, service_config in enumerate(fallback_chain):
            try:
                # Check if service is healthy
                if not await self._is_service_healthy(service_config['service']):
                    continue
                
                # Check cost constraints
                if not await self._check_cost_constraint(service_config, context):
                    continue
                
                # Attempt operation
                result = await operation(service_config, context)
                
                # Log successful fallback if not primary
                if i > 0:
                    await self._log_fallback_success(service_type, i, service_config)
                
                return result
                
            except Exception as e:
                last_error = e
                await self._log_fallback_attempt(service_type, i, service_config, e)
                
                # Check if we should continue trying
                if not self._should_continue_fallback(e):
                    break
        
        # All fallbacks failed
        await self._handle_all_fallbacks_failed(service_type, last_error, context)
        raise ServiceUnavailableError(f"All fallbacks failed for {service_type}: {last_error}")
```

### 4.2 Circuit Breaker Implementation

```python
from enum import Enum
from datetime import datetime, timedelta
import asyncio

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """
    Circuit breaker pattern for service protection
    """
    
    def __init__(self, service_name: str, config: dict = None):
        self.service_name = service_name
        self.state = CircuitState.CLOSED
        
        # Configuration
        self.failure_threshold = config.get('failure_threshold', 5)
        self.recovery_timeout = config.get('recovery_timeout', 60)
        self.half_open_requests = config.get('half_open_requests', 3)
        
        # State tracking
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_count = 0
        self.success_count = 0
        
    async def call(self, func: callable, *args, **kwargs):
        """
        Execute function through circuit breaker
        """
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_count = 0
            else:
                raise CircuitOpenError(
                    f"Circuit breaker OPEN for {self.service_name}"
                )
        
        # Execute function
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call"""
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_requests:
                # Circuit recovered
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker CLOSED for {self.service_name}")
        else:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery, reopen circuit
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker REOPENED for {self.service_name}")
            
        elif self.failure_count >= self.failure_threshold:
            # Open circuit
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPENED for {self.service_name}")
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset circuit"""
        
        if not self.last_failure_time:
            return True
            
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout
```

### 4.3 Health Check System

```python
class ServiceHealthChecker:
    """
    Proactive health checking for all integrated services
    """
    
    def __init__(self):
        self.health_checks = {}
        self.check_interval = 30  # seconds
        self.timeout = 5  # seconds
        
    async def start_health_monitoring(self):
        """
        Start background health monitoring
        """
        
        services = [
            'youtube_api',
            'openai_api',
            'elevenlabs_api',
            'google_tts_api',
            'stripe_api'
        ]
        
        for service in services:
            asyncio.create_task(self._monitor_service(service))
    
    async def _monitor_service(self, service_name: str):
        """
        Monitor individual service health
        """
        
        while True:
            try:
                health_status = await self._check_service_health(service_name)
                
                # Update health status
                self.health_checks[service_name] = {
                    'status': health_status,
                    'timestamp': datetime.now(),
                    'response_time': health_status.get('response_time'),
                    'error': health_status.get('error')
                }
                
                # Alert on status change
                await self._check_status_change(service_name, health_status)
                
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {e}")
                
            await asyncio.sleep(self.check_interval)
    
    async def _check_service_health(self, service_name: str) -> dict:
        """
        Perform health check for specific service
        """
        
        health_check_endpoints = {
            'youtube_api': self._check_youtube_health,
            'openai_api': self._check_openai_health,
            'elevenlabs_api': self._check_elevenlabs_health,
            'google_tts_api': self._check_google_tts_health,
            'stripe_api': self._check_stripe_health
        }
        
        check_func = health_check_endpoints.get(service_name)
        if not check_func:
            return {'status': 'unknown', 'error': 'No health check defined'}
        
        return await check_func()
    
    async def _check_youtube_health(self) -> dict:
        """Check YouTube API health"""
        
        try:
            start_time = time.time()
            
            # Test with minimal quota operation
            service = build('youtube', 'v3', credentials=self._get_test_credentials())
            response = service.channels().list(
                part='id',
                mine=True
            ).execute()
            
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'response_time': response_time,
                'quota_available': await self._check_quota_availability()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
```

---

## Key Implementation Notes

### For You as Integration Specialist

1. **Account Management Priority**
   - Always maintain at least 3 reserve accounts ready
   - Monitor account health scores continuously
   - Rotate accounts proactively, not reactively

2. **Quota Management**
   - Never exceed 80% quota on any account
   - Implement hard stops at 90%
   - Track quota consumption in real-time

3. **OAuth Token Management**
   - Refresh tokens 30 minutes before expiry
   - Store credentials encrypted
   - Maintain backup authentication methods

4. **Fallback Strategies**
   - Test fallback chains weekly
   - Monitor service health proactively
   - Document all fallback activations

5. **Error Handling**
   - Log all errors with full context
   - Alert on critical failures immediately
   - Maintain error pattern analysis

### Critical Metrics to Monitor

- **Account Health**: Score > 0.7 for all accounts
- **Quota Usage**: < 80% for all accounts
- **Token Validity**: 100% tokens valid
- **Upload Success Rate**: > 98%
- **Fallback Activation Rate**: < 5%

### Daily Checklist

- [ ] Check all 15 account health scores
- [ ] Verify token validity for all accounts
- [ ] Review quota usage patterns
- [ ] Test one fallback chain
- [ ] Check error logs for patterns
- [ ] Update account rotation schedule

---

**Document Status**: Part 1 Complete
**Next**: See Part 2 for Payment Systems, Webhooks, and Testing Protocols