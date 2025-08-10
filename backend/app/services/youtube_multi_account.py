"""
YouTube Multi-Account Management Service
Handles rotation and management of 15 YouTube accounts
"""
import os
import json
import asyncio
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum
import aiofiles
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.http import MediaFileUpload
import httplib2
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import run_flow
from tenacity import retry, stop_after_attempt, wait_exponential
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

logger = logging.getLogger(__name__)

class AccountStatus(Enum):
    """YouTube account status"""
    ACTIVE = "active"
    QUOTA_EXCEEDED = "quota_exceeded"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    DISABLED = "disabled"
    PENDING_AUTH = "pending_auth"

@dataclass
class YouTubeAccount:
    """Individual YouTube account configuration"""
    account_id: str
    email: str
    channel_id: Optional[str] = None
    client_secrets_file: str = None
    credentials_file: str = None
    api_key: Optional[str] = None
    status: AccountStatus = AccountStatus.PENDING_AUTH
    quota_used: int = 0
    quota_limit: int = 10000
    last_used: Optional[datetime] = None
    error_count: int = 0
    health_score: float = 100.0
    authenticated_service: Any = None
    metadata: Dict = field(default_factory=dict)
    
    def is_available(self) -> bool:
        """Check if account is available for use"""
        return (
            self.status == AccountStatus.ACTIVE and
            self.quota_used < self.quota_limit * 0.9 and  # 90% quota threshold
            self.health_score > 50.0
        )
    
    def update_health_score(self, success: bool):
        """Update health score based on operation result"""
        if success:
            self.health_score = min(100.0, self.health_score + 5.0)
            self.error_count = 0
        else:
            self.health_score = max(0.0, self.health_score - 10.0)
            self.error_count += 1
            
        # Auto-disable if too many errors
        if self.error_count >= 5:
            self.status = AccountStatus.ERROR
        elif self.health_score <= 25.0:
            self.status = AccountStatus.DISABLED

class MultiAccountYouTubeService:
    """Manages multiple YouTube accounts with rotation and health monitoring"""
    
    def __init__(self, accounts_config_path: str = "config/youtube_accounts.json"):
        self.accounts: Dict[str, YouTubeAccount] = {}
        self.accounts_config_path = accounts_config_path
        self.redis_client: Optional[redis.Redis] = None
        self.rotation_strategy = "health_weighted"  # Options: round_robin, health_weighted, least_used
        self.current_index = 0
        self.initialized = False
        
    async def initialize(self):
        """Initialize multi-account service"""
        try:
            # Initialize Redis for distributed coordination
            self.redis_client = await redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379"),
                encoding="utf-8",
                decode_responses=True
            )
            
            # Load accounts configuration
            await self.load_accounts_config()
            
            # Initialize each account
            for account_id in self.accounts:
                await self.initialize_account(account_id)
                
            self.initialized = True
            logger.info(f"Initialized {len(self.accounts)} YouTube accounts")
            
        except Exception as e:
            logger.error(f"Failed to initialize multi-account service: {e}")
            raise
            
    async def load_accounts_config(self):
        """Load accounts configuration from file or environment"""
        config_path = Path(self.accounts_config_path)
        
        if config_path.exists():
            async with aiofiles.open(config_path, 'r') as f:
                config = json.loads(await f.read())
        else:
            # Create default configuration for 15 accounts
            config = self.create_default_config()
            await self.save_accounts_config(config)
            
        for account_config in config["accounts"]:
            account = YouTubeAccount(
                account_id=account_config["account_id"],
                email=account_config["email"],
                channel_id=account_config.get("channel_id"),
                client_secrets_file=account_config.get("client_secrets_file"),
                credentials_file=account_config.get("credentials_file"),
                api_key=account_config.get("api_key"),
                quota_limit=account_config.get("quota_limit", 10000)
            )
            self.accounts[account.account_id] = account
            
    def create_default_config(self) -> Dict:
        """Create default configuration for 15 accounts"""
        accounts = []
        for i in range(1, 16):
            accounts.append({
                "account_id": f"youtube_account_{i:02d}",
                "email": f"ytempire.account{i:02d}@gmail.com",
                "client_secrets_file": f"config/youtube_secrets/client_secret_{i:02d}.json",
                "credentials_file": f"config/youtube_credentials/credentials_{i:02d}.json",
                "api_key": os.getenv(f"YOUTUBE_API_KEY_{i:02d}"),
                "quota_limit": 10000
            })
            
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "accounts": accounts
        }
        
    async def save_accounts_config(self, config: Dict):
        """Save accounts configuration to file"""
        config_path = Path(self.accounts_config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(config_path, 'w') as f:
            await f.write(json.dumps(config, indent=2))
            
    async def initialize_account(self, account_id: str):
        """Initialize a single YouTube account"""
        account = self.accounts.get(account_id)
        if not account:
            return
            
        try:
            # Check for API key first (for read-only operations)
            if account.api_key:
                service = build(
                    "youtube", "v3",
                    developerKey=account.api_key,
                    cache_discovery=False
                )
                # Test the API key
                service.videos().list(part="id", chart="mostPopular", maxResults=1).execute()
                account.status = AccountStatus.ACTIVE
                logger.info(f"Account {account_id} initialized with API key")
                
            # Check for OAuth credentials (for write operations)
            if account.credentials_file and Path(account.credentials_file).exists():
                store = Storage(account.credentials_file)
                credentials = store.get()
                
                if credentials and not credentials.invalid:
                    account.authenticated_service = build(
                        "youtube", "v3",
                        http=credentials.authorize(httplib2.Http()),
                        cache_discovery=False
                    )
                    account.status = AccountStatus.ACTIVE
                    logger.info(f"Account {account_id} authenticated with OAuth")
                else:
                    account.status = AccountStatus.PENDING_AUTH
                    logger.warning(f"Account {account_id} needs authentication")
            else:
                account.status = AccountStatus.PENDING_AUTH
                
            # Load quota usage from Redis
            await self.load_account_quota(account_id)
            
        except Exception as e:
            logger.error(f"Failed to initialize account {account_id}: {e}")
            account.status = AccountStatus.ERROR
            account.update_health_score(False)
            
    async def load_account_quota(self, account_id: str):
        """Load quota usage from Redis"""
        if not self.redis_client:
            return
            
        try:
            quota_key = f"youtube:quota:{account_id}:{datetime.now().strftime('%Y%m%d')}"
            quota_used = await self.redis_client.get(quota_key)
            
            if quota_used:
                self.accounts[account_id].quota_used = int(quota_used)
                
        except Exception as e:
            logger.error(f"Failed to load quota for {account_id}: {e}")
            
    async def update_account_quota(self, account_id: str, units: int):
        """Update account quota usage"""
        account = self.accounts.get(account_id)
        if not account:
            return
            
        account.quota_used += units
        account.last_used = datetime.now()
        
        # Check quota limit
        if account.quota_used >= account.quota_limit * 0.95:
            account.status = AccountStatus.QUOTA_EXCEEDED
            logger.warning(f"Account {account_id} quota exceeded: {account.quota_used}/{account.quota_limit}")
            
        # Store in Redis with daily expiry
        if self.redis_client:
            try:
                quota_key = f"youtube:quota:{account_id}:{datetime.now().strftime('%Y%m%d')}"
                await self.redis_client.setex(
                    quota_key,
                    86400,  # 24 hours
                    str(account.quota_used)
                )
            except Exception as e:
                logger.error(f"Failed to update quota in Redis: {e}")
                
    async def get_best_account(self, required_units: int = 100) -> Optional[YouTubeAccount]:
        """Get the best available account based on strategy"""
        available_accounts = [
            acc for acc in self.accounts.values()
            if acc.is_available() and (acc.quota_used + required_units) < acc.quota_limit
        ]
        
        if not available_accounts:
            # Try to reset quota for accounts if it's a new day
            await self.reset_daily_quotas()
            available_accounts = [
                acc for acc in self.accounts.values()
                if acc.is_available() and (acc.quota_used + required_units) < acc.quota_limit
            ]
            
        if not available_accounts:
            logger.error("No available YouTube accounts")
            return None
            
        if self.rotation_strategy == "round_robin":
            account = available_accounts[self.current_index % len(available_accounts)]
            self.current_index += 1
            
        elif self.rotation_strategy == "health_weighted":
            # Select based on health score
            weights = [acc.health_score for acc in available_accounts]
            total_weight = sum(weights)
            if total_weight == 0:
                account = random.choice(available_accounts)
            else:
                weights = [w/total_weight for w in weights]
                account = random.choices(available_accounts, weights=weights)[0]
                
        elif self.rotation_strategy == "least_used":
            # Select account with lowest quota usage
            account = min(available_accounts, key=lambda x: x.quota_used)
            
        else:
            account = random.choice(available_accounts)
            
        return account
        
    async def reset_daily_quotas(self):
        """Reset daily quotas for all accounts"""
        current_date = datetime.now().strftime('%Y%m%d')
        
        for account_id, account in self.accounts.items():
            # Check if it's a new day
            if account.last_used and account.last_used.date() < datetime.now().date():
                account.quota_used = 0
                if account.status == AccountStatus.QUOTA_EXCEEDED:
                    account.status = AccountStatus.ACTIVE
                logger.info(f"Reset quota for account {account_id}")
                
    async def execute_with_rotation(
        self,
        operation_func,
        required_units: int = 100,
        max_retries: int = 3,
        **kwargs
    ):
        """Execute an operation with automatic account rotation"""
        retries = 0
        last_error = None
        
        while retries < max_retries:
            account = await self.get_best_account(required_units)
            
            if not account:
                raise Exception("No available YouTube accounts for operation")
                
            try:
                # Execute the operation
                result = await operation_func(account, **kwargs)
                
                # Update account metrics on success
                await self.update_account_quota(account.account_id, required_units)
                account.update_health_score(True)
                
                # Log successful operation
                await self.log_operation(account.account_id, "success", operation_func.__name__)
                
                return result
                
            except HttpError as e:
                last_error = e
                account.update_health_score(False)
                
                if e.resp.status == 429:  # Rate limit
                    account.status = AccountStatus.RATE_LIMITED
                    logger.warning(f"Account {account.account_id} rate limited")
                elif e.resp.status == 403:  # Quota exceeded
                    account.status = AccountStatus.QUOTA_EXCEEDED
                    logger.warning(f"Account {account.account_id} quota exceeded")
                else:
                    logger.error(f"Operation failed on account {account.account_id}: {e}")
                    
                await self.log_operation(account.account_id, "error", operation_func.__name__, str(e))
                retries += 1
                
            except Exception as e:
                last_error = e
                account.update_health_score(False)
                logger.error(f"Unexpected error on account {account.account_id}: {e}")
                await self.log_operation(account.account_id, "error", operation_func.__name__, str(e))
                retries += 1
                
        raise Exception(f"Operation failed after {max_retries} retries: {last_error}")
        
    async def log_operation(self, account_id: str, status: str, operation: str, error: str = None):
        """Log operation to Redis for monitoring"""
        if not self.redis_client:
            return
            
        try:
            log_entry = {
                "account_id": account_id,
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "status": status,
                "error": error
            }
            
            # Store in Redis list with expiry
            log_key = f"youtube:operations:{datetime.now().strftime('%Y%m%d')}"
            await self.redis_client.lpush(log_key, json.dumps(log_entry))
            await self.redis_client.expire(log_key, 86400 * 7)  # Keep logs for 7 days
            
        except Exception as e:
            logger.error(f"Failed to log operation: {e}")
            
    async def get_account_statistics(self) -> Dict:
        """Get statistics for all accounts"""
        stats = {
            "total_accounts": len(self.accounts),
            "active_accounts": 0,
            "total_quota_used": 0,
            "total_quota_limit": 0,
            "accounts": []
        }
        
        for account_id, account in self.accounts.items():
            if account.status == AccountStatus.ACTIVE:
                stats["active_accounts"] += 1
                
            stats["total_quota_used"] += account.quota_used
            stats["total_quota_limit"] += account.quota_limit
            
            stats["accounts"].append({
                "account_id": account_id,
                "email": account.email,
                "status": account.status.value,
                "quota_used": account.quota_used,
                "quota_limit": account.quota_limit,
                "quota_percentage": (account.quota_used / account.quota_limit * 100) if account.quota_limit > 0 else 0,
                "health_score": account.health_score,
                "error_count": account.error_count,
                "last_used": account.last_used.isoformat() if account.last_used else None
            })
            
        stats["quota_usage_percentage"] = (
            (stats["total_quota_used"] / stats["total_quota_limit"] * 100)
            if stats["total_quota_limit"] > 0 else 0
        )
        
        return stats
        
    async def authenticate_account(self, account_id: str, client_secrets_path: str = None):
        """Manually authenticate a YouTube account"""
        account = self.accounts.get(account_id)
        if not account:
            raise ValueError(f"Account {account_id} not found")
            
        try:
            secrets_file = client_secrets_path or account.client_secrets_file
            
            if not secrets_file or not Path(secrets_file).exists():
                raise ValueError(f"Client secrets file not found: {secrets_file}")
                
            flow = flow_from_clientsecrets(
                secrets_file,
                scope=[
                    "https://www.googleapis.com/auth/youtube",
                    "https://www.googleapis.com/auth/youtube.upload",
                    "https://www.googleapis.com/auth/youtube.readonly",
                    "https://www.googleapis.com/auth/youtubepartner",
                    "https://www.googleapis.com/auth/youtube.force-ssl"
                ]
            )
            
            store = Storage(account.credentials_file)
            credentials = run_flow(flow, store)
            
            account.authenticated_service = build(
                "youtube", "v3",
                http=credentials.authorize(httplib2.Http()),
                cache_discovery=False
            )
            
            account.status = AccountStatus.ACTIVE
            logger.info(f"Successfully authenticated account {account_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to authenticate account {account_id}: {e}")
            account.status = AccountStatus.ERROR
            return False
            
    async def health_check(self) -> Dict:
        """Perform health check on all accounts"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "healthy_accounts": 0,
            "unhealthy_accounts": 0,
            "accounts": {}
        }
        
        for account_id, account in self.accounts.items():
            try:
                if account.api_key:
                    # Test with simple API call
                    service = build("youtube", "v3", developerKey=account.api_key, cache_discovery=False)
                    service.videos().list(part="id", chart="mostPopular", maxResults=1).execute()
                    
                    account.status = AccountStatus.ACTIVE
                    account.update_health_score(True)
                    results["healthy_accounts"] += 1
                    results["accounts"][account_id] = "healthy"
                else:
                    results["accounts"][account_id] = "no_api_key"
                    
            except Exception as e:
                account.update_health_score(False)
                results["unhealthy_accounts"] += 1
                results["accounts"][account_id] = f"error: {str(e)}"
                
        return results


# Integration with existing YouTube service
class YouTubeServiceWrapper:
    """Wrapper to integrate multi-account service with existing YouTube operations"""
    
    def __init__(self):
        self.multi_account_service = MultiAccountYouTubeService()
        
    async def initialize(self):
        """Initialize the service"""
        await self.multi_account_service.initialize()
        
    async def search_videos(self, query: str, **kwargs):
        """Search videos with automatic account rotation"""
        async def operation(account: YouTubeAccount, **op_kwargs):
            service = build("youtube", "v3", developerKey=account.api_key, cache_discovery=False)
            
            search_params = {
                "q": op_kwargs.get("query"),
                "type": "video",
                "part": "id,snippet",
                "maxResults": op_kwargs.get("max_results", 25)
            }
            
            response = service.search().list(**search_params).execute()
            return response
            
        return await self.multi_account_service.execute_with_rotation(
            operation,
            required_units=100,  # Search costs 100 units
            query=query,
            **kwargs
        )
        
    async def upload_video(self, video_file_path: str, title: str, description: str, **kwargs):
        """Upload video with automatic account rotation"""
        async def operation(account: YouTubeAccount, **op_kwargs):
            if not account.authenticated_service:
                raise ValueError(f"Account {account.account_id} not authenticated for uploads")
                
            body = {
                "snippet": {
                    "title": op_kwargs.get("title"),
                    "description": op_kwargs.get("description"),
                    "tags": op_kwargs.get("tags", []),
                    "categoryId": op_kwargs.get("category_id", "22")
                },
                "status": {
                    "privacyStatus": op_kwargs.get("privacy_status", "private"),
                    "selfDeclaredMadeForKids": False
                }
            }
            
            media = MediaFileUpload(
                op_kwargs.get("video_file_path"),
                chunksize=1024 * 1024,
                resumable=True,
                mimetype="video/*"
            )
            
            request = account.authenticated_service.videos().insert(
                part=",".join(body.keys()),
                body=body,
                media_body=media
            )
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    logger.info(f"Upload progress: {int(status.progress() * 100)}%")
                    
            return response
            
        return await self.multi_account_service.execute_with_rotation(
            operation,
            required_units=1600,  # Upload costs 1600 units
            video_file_path=video_file_path,
            title=title,
            description=description,
            **kwargs
        )
        
    async def get_statistics(self):
        """Get multi-account statistics"""
        return await self.multi_account_service.get_account_statistics()
        
    async def health_check(self):
        """Perform health check"""
        return await self.multi_account_service.health_check()