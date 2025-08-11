"""
YouTube Multi-Account Management System
Handles 15 YouTube accounts with intelligent rotation and quota management
"""
import os
import json
import redis
import logging
import random
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow

logger = logging.getLogger(__name__)

# Redis connection for distributed state management
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "1"))  # Use DB 1 for YouTube accounts

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)


class AccountStatus(Enum):
    """YouTube account status states"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    QUOTA_EXCEEDED = "quota_exceeded"
    AUTH_EXPIRED = "auth_expired"
    ERROR = "error"
    COOLING_DOWN = "cooling_down"


class OperationType(Enum):
    """YouTube API operation types with quota costs"""
    UPLOAD = 1600  # Video upload
    LIST = 1  # List videos/channels
    UPDATE = 50  # Update video metadata
    DELETE = 50  # Delete video
    INSERT_PLAYLIST = 50  # Create playlist
    ANALYTICS = 1  # Analytics read


@dataclass
class YouTubeAccount:
    """YouTube account configuration and state"""
    account_id: str
    email: str
    channel_id: str
    channel_name: str
    credentials_json: str  # OAuth2 credentials
    refresh_token: str
    status: AccountStatus
    quota_used: int
    quota_reset_time: datetime
    last_used: datetime
    error_count: int
    health_score: float  # 0-100 score
    total_uploads: int
    total_views: int
    strikes: int
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage"""
        data = asdict(self)
        data['status'] = self.status.value
        data['quota_reset_time'] = self.quota_reset_time.isoformat()
        data['last_used'] = self.last_used.isoformat()
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'YouTubeAccount':
        """Create from dictionary (Redis storage)"""
        data['status'] = AccountStatus(data['status'])
        data['quota_reset_time'] = datetime.fromisoformat(data['quota_reset_time'])
        data['last_used'] = datetime.fromisoformat(data['last_used'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class YouTubeMultiAccountManager:
    """Manages multiple YouTube accounts with intelligent rotation"""
    
    DAILY_QUOTA_LIMIT = 10000  # YouTube API daily quota per account
    QUOTA_BUFFER = 1000  # Keep buffer to avoid hitting exact limit
    MAX_ERROR_COUNT = 5  # Max errors before suspending account
    COOLDOWN_HOURS = 6  # Hours to cool down after quota exceeded
    HEALTH_SCORE_THRESHOLD = 30  # Minimum health score to use account
    
    def __init__(self):
        """Initialize the multi-account manager"""
        self.accounts: List[YouTubeAccount] = []
        self.redis = redis_client
        self.load_accounts()
        
    def load_accounts(self):
        """Load all YouTube accounts from configuration/Redis"""
        try:
            # Load from Redis if exists
            account_keys = self.redis.keys("youtube:account:*")
            
            if account_keys:
                for key in account_keys:
                    account_data = self.redis.hgetall(key)
                    if account_data:
                        account = YouTubeAccount.from_dict(account_data)
                        self.accounts.append(account)
                        logger.info(f"Loaded YouTube account: {account.email}")
            else:
                # Initialize from environment or config file
                self._initialize_accounts_from_config()
                
            logger.info(f"Loaded {len(self.accounts)} YouTube accounts")
            
        except Exception as e:
            logger.error(f"Failed to load YouTube accounts: {e}")
            
    def _initialize_accounts_from_config(self):
        """Initialize accounts from configuration file"""
        config_path = "config/youtube_accounts.json"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            for acc_config in config.get('accounts', []):
                account = YouTubeAccount(
                    account_id=acc_config['account_id'],
                    email=acc_config['email'],
                    channel_id=acc_config.get('channel_id', ''),
                    channel_name=acc_config.get('channel_name', ''),
                    credentials_json=acc_config.get('credentials_json', ''),
                    refresh_token=acc_config.get('refresh_token', ''),
                    status=AccountStatus.ACTIVE,
                    quota_used=0,
                    quota_reset_time=datetime.utcnow() + timedelta(days=1),
                    last_used=datetime.utcnow(),
                    error_count=0,
                    health_score=100.0,
                    total_uploads=0,
                    total_views=0,
                    strikes=0,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                self.accounts.append(account)
                self.save_account(account)
                
    def save_account(self, account: YouTubeAccount):
        """Save account state to Redis"""
        try:
            key = f"youtube:account:{account.account_id}"
            account.updated_at = datetime.utcnow()
            self.redis.hset(key, mapping=account.to_dict())
            self.redis.expire(key, 86400 * 30)  # 30 days expiry
        except Exception as e:
            logger.error(f"Failed to save account {account.email}: {e}")
            
    def get_best_account(self) -> Optional[YouTubeAccount]:
        """Get the best available YouTube account based on health score"""
        available_accounts = [
            acc for acc in self.accounts
            if self._is_account_available(acc)
        ]
        
        if not available_accounts:
            logger.error("No available YouTube accounts")
            return None
            
        # Sort by health score and quota availability
        available_accounts.sort(
            key=lambda x: (x.health_score, self.DAILY_QUOTA_LIMIT - x.quota_used),
            reverse=True
        )
        
        selected = available_accounts[0]
        logger.info(f"Selected YouTube account: {selected.email} (health: {selected.health_score})")
        
        return selected
        
    def _is_account_available(self, account: YouTubeAccount) -> bool:
        """Check if account is available for use"""
        # Check status
        if account.status not in [AccountStatus.ACTIVE, AccountStatus.COOLING_DOWN]:
            return False
            
        # Check health score
        if account.health_score < self.HEALTH_SCORE_THRESHOLD:
            return False
            
        # Check quota
        if account.quota_used >= (self.DAILY_QUOTA_LIMIT - self.QUOTA_BUFFER):
            # Check if quota should reset
            if datetime.utcnow() >= account.quota_reset_time:
                self._reset_account_quota(account)
            else:
                return False
                
        # Check cooldown
        if account.status == AccountStatus.COOLING_DOWN:
            cooldown_end = account.last_used + timedelta(hours=self.COOLDOWN_HOURS)
            if datetime.utcnow() >= cooldown_end:
                account.status = AccountStatus.ACTIVE
                self.save_account(account)
            else:
                return False
                
        return True
        
    def _reset_account_quota(self, account: YouTubeAccount):
        """Reset daily quota for account"""
        logger.info(f"Resetting quota for account: {account.email}")
        account.quota_used = 0
        account.quota_reset_time = datetime.utcnow() + timedelta(days=1)
        account.status = AccountStatus.ACTIVE
        self.save_account(account)
        
    def use_account(self, account: YouTubeAccount, operation: OperationType) -> bool:
        """Mark account as used and update quota"""
        try:
            quota_cost = operation.value
            
            # Check if operation would exceed quota
            if account.quota_used + quota_cost > self.DAILY_QUOTA_LIMIT:
                logger.warning(f"Operation would exceed quota for {account.email}")
                account.status = AccountStatus.QUOTA_EXCEEDED
                self.save_account(account)
                return False
                
            # Update account usage
            account.quota_used += quota_cost
            account.last_used = datetime.utcnow()
            
            # Update health score based on quota usage
            quota_percentage = (account.quota_used / self.DAILY_QUOTA_LIMIT) * 100
            account.health_score = max(0, 100 - quota_percentage - (account.error_count * 10))
            
            self.save_account(account)
            
            logger.info(
                f"Used account {account.email} for {operation.name}. "
                f"Quota: {account.quota_used}/{self.DAILY_QUOTA_LIMIT}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update account usage: {e}")
            return False
            
    def handle_account_error(self, account: YouTubeAccount, error: Exception):
        """Handle error for a YouTube account"""
        account.error_count += 1
        
        # Determine error type and update status
        error_str = str(error).lower()
        
        if 'quota' in error_str:
            account.status = AccountStatus.QUOTA_EXCEEDED
            account.last_used = datetime.utcnow()
            logger.warning(f"Quota exceeded for account {account.email}")
            
        elif 'credentials' in error_str or 'auth' in error_str:
            account.status = AccountStatus.AUTH_EXPIRED
            logger.error(f"Authentication expired for account {account.email}")
            
        elif account.error_count >= self.MAX_ERROR_COUNT:
            account.status = AccountStatus.SUSPENDED
            logger.error(f"Suspending account {account.email} due to excessive errors")
            
        else:
            account.status = AccountStatus.COOLING_DOWN
            logger.warning(f"Cooling down account {account.email} after error")
            
        # Update health score
        account.health_score = max(0, account.health_score - 20)
        
        self.save_account(account)
        
    def get_youtube_service(self, account: YouTubeAccount):
        """Get authenticated YouTube service for an account"""
        try:
            # Parse credentials
            if account.credentials_json:
                creds_data = json.loads(account.credentials_json)
                credentials = Credentials(
                    token=creds_data.get('token'),
                    refresh_token=account.refresh_token,
                    token_uri=creds_data.get('token_uri'),
                    client_id=creds_data.get('client_id'),
                    client_secret=creds_data.get('client_secret'),
                    scopes=creds_data.get('scopes', [
                        'https://www.googleapis.com/auth/youtube.upload',
                        'https://www.googleapis.com/auth/youtube',
                        'https://www.googleapis.com/auth/youtubepartner'
                    ])
                )
                
                # Refresh if needed
                if credentials.expired and credentials.refresh_token:
                    credentials.refresh(Request())
                    # Update stored credentials
                    account.credentials_json = credentials.to_json()
                    self.save_account(account)
                    
                # Build YouTube service
                service = build('youtube', 'v3', credentials=credentials)
                return service
                
            else:
                logger.error(f"No credentials for account {account.email}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get YouTube service for {account.email}: {e}")
            self.handle_account_error(account, e)
            return None
            
    def rotate_to_next_account(self, current_account: Optional[YouTubeAccount] = None) -> Optional[YouTubeAccount]:
        """Rotate to the next available account"""
        if current_account:
            # Mark current as cooling down
            current_account.status = AccountStatus.COOLING_DOWN
            self.save_account(current_account)
            
        # Get next best account
        return self.get_best_account()
        
    def get_account_stats(self) -> Dict[str, Any]:
        """Get statistics for all accounts"""
        total_quota = sum(acc.quota_used for acc in self.accounts)
        active_accounts = sum(1 for acc in self.accounts if acc.status == AccountStatus.ACTIVE)
        
        return {
            'total_accounts': len(self.accounts),
            'active_accounts': active_accounts,
            'total_quota_used': total_quota,
            'total_quota_available': len(self.accounts) * self.DAILY_QUOTA_LIMIT,
            'average_health_score': sum(acc.health_score for acc in self.accounts) / len(self.accounts) if self.accounts else 0,
            'accounts': [
                {
                    'email': acc.email,
                    'status': acc.status.value,
                    'health_score': acc.health_score,
                    'quota_used': acc.quota_used,
                    'quota_limit': self.DAILY_QUOTA_LIMIT
                }
                for acc in self.accounts
            ]
        }
        
    async def upload_video_with_rotation(self, video_path: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Upload video with automatic account rotation on failure"""
        max_retries = min(3, len(self.accounts))
        
        for attempt in range(max_retries):
            account = self.get_best_account()
            
            if not account:
                logger.error("No available YouTube accounts for upload")
                return None
                
            try:
                # Get YouTube service
                service = self.get_youtube_service(account)
                if not service:
                    continue
                    
                # Mark account as being used
                if not self.use_account(account, OperationType.UPLOAD):
                    continue
                    
                # Perform upload (simplified - actual implementation would use resumable upload)
                logger.info(f"Uploading video using account: {account.email}")
                
                # TODO: Implement actual video upload logic here
                # This is a placeholder for the upload implementation
                result = {
                    'video_id': f"video_{account.account_id}_{datetime.utcnow().timestamp()}",
                    'channel_id': account.channel_id,
                    'account_used': account.email,
                    'upload_time': datetime.utcnow().isoformat()
                }
                
                # Update account stats
                account.total_uploads += 1
                self.save_account(account)
                
                return result
                
            except Exception as e:
                logger.error(f"Upload failed with account {account.email}: {e}")
                self.handle_account_error(account, e)
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying with different account (attempt {attempt + 2}/{max_retries})")
                    await asyncio.sleep(2)  # Brief delay before retry
                    
        logger.error("All upload attempts failed")
        return None
        
    def setup_oauth_for_account(self, account_index: int) -> Optional[str]:
        """Setup OAuth2 flow for a specific account"""
        try:
            # OAuth2 configuration
            client_config = {
                "installed": {
                    "client_id": os.getenv(f"YOUTUBE_CLIENT_ID_{account_index}"),
                    "client_secret": os.getenv(f"YOUTUBE_CLIENT_SECRET_{account_index}"),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"]
                }
            }
            
            # Create flow
            flow = Flow.from_client_config(
                client_config,
                scopes=[
                    'https://www.googleapis.com/auth/youtube.upload',
                    'https://www.googleapis.com/auth/youtube',
                    'https://www.googleapis.com/auth/youtubepartner',
                    'https://www.googleapis.com/auth/youtube.force-ssl'
                ]
            )
            
            flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
            
            # Get authorization URL
            auth_url, _ = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent'
            )
            
            logger.info(f"OAuth URL for account {account_index}: {auth_url}")
            return auth_url
            
        except Exception as e:
            logger.error(f"Failed to setup OAuth for account {account_index}: {e}")
            return None
            
    def complete_oauth_for_account(self, account_index: int, auth_code: str) -> bool:
        """Complete OAuth2 flow with authorization code"""
        try:
            # Recreate flow
            client_config = {
                "installed": {
                    "client_id": os.getenv(f"YOUTUBE_CLIENT_ID_{account_index}"),
                    "client_secret": os.getenv(f"YOUTUBE_CLIENT_SECRET_{account_index}"),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"]
                }
            }
            
            flow = Flow.from_client_config(
                client_config,
                scopes=[
                    'https://www.googleapis.com/auth/youtube.upload',
                    'https://www.googleapis.com/auth/youtube',
                    'https://www.googleapis.com/auth/youtubepartner',
                    'https://www.googleapis.com/auth/youtube.force-ssl'
                ],
                redirect_uri='urn:ietf:wg:oauth:2.0:oob'
            )
            
            # Exchange code for token
            flow.fetch_token(code=auth_code)
            credentials = flow.credentials
            
            # Save credentials for account
            if account_index < len(self.accounts):
                account = self.accounts[account_index]
                account.credentials_json = credentials.to_json()
                account.refresh_token = credentials.refresh_token
                account.status = AccountStatus.ACTIVE
                self.save_account(account)
                
                logger.info(f"OAuth completed for account {account.email}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to complete OAuth for account {account_index}: {e}")
            
        return False


# Singleton instance
_manager_instance = None


def get_youtube_manager() -> YouTubeMultiAccountManager:
    """Get singleton instance of YouTube multi-account manager"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = YouTubeMultiAccountManager()
    return _manager_instance