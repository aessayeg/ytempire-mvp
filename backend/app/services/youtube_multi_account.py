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
    host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
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
        data["status"] = self.status.value
        data["quota_reset_time"] = self.quota_reset_time.isoformat()
        data["last_used"] = self.last_used.isoformat()
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "YouTubeAccount":
        """Create from dictionary (Redis storage)"""
        data["status"] = AccountStatus(data["status"])
        data["quota_reset_time"] = datetime.fromisoformat(data["quota_reset_time"])
        data["last_used"] = datetime.fromisoformat(data["last_used"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class YouTubeMultiAccountManager:
    """Manages multiple YouTube accounts with intelligent rotation and advanced monitoring"""

    DAILY_QUOTA_LIMIT = 10000  # YouTube API daily quota per account
    QUOTA_BUFFER = 1000  # Keep buffer to avoid hitting exact limit
    MAX_ERROR_COUNT = 5  # Max errors before suspending account
    COOLDOWN_HOURS = 6  # Hours to cool down after quota exceeded
    HEALTH_SCORE_THRESHOLD = 30  # Minimum health score to use account

    # Enhanced configuration for 15 account management
    TARGET_ACCOUNT_COUNT = 15  # Target number of accounts to manage
    FAILOVER_THRESHOLD = 3  # Minimum healthy accounts before alerting
    ROTATION_STRATEGY = (
        "weighted_round_robin"  # weighted_round_robin, health_based, quota_based
    )
    HEALTH_CHECK_INTERVAL = 300  # Health check every 5 minutes
    QUOTA_REDISTRIBUTION = True  # Enable quota redistribution across accounts

    # Advanced monitoring thresholds
    CRITICAL_HEALTH_THRESHOLD = 20  # Critical health alert threshold
    WARNING_QUOTA_THRESHOLD = 0.8  # Warning when quota usage exceeds 80%
    ERROR_SPIKE_THRESHOLD = 5  # Alert when errors spike within time window

    # Performance optimization
    CONCURRENT_OPERATIONS = 3  # Max concurrent operations per account
    OPERATION_TIMEOUT = 30  # Operation timeout in seconds
    RETRY_BACKOFF_BASE = 2  # Exponential backoff base for retries

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
            with open(config_path, "r") as f:
                config = json.load(f)

            for acc_config in config.get("accounts", []):
                account = YouTubeAccount(
                    account_id=acc_config["account_id"],
                    email=acc_config["email"],
                    channel_id=acc_config.get("channel_id", ""),
                    channel_name=acc_config.get("channel_name", ""),
                    credentials_json=acc_config.get("credentials_json", ""),
                    refresh_token=acc_config.get("refresh_token", ""),
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
                    updated_at=datetime.utcnow(),
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
            acc for acc in self.accounts if self._is_account_available(acc)
        ]

        if not available_accounts:
            logger.error("No available YouTube accounts")
            return None

        # Sort by health score and quota availability
        available_accounts.sort(
            key=lambda x: (x.health_score, self.DAILY_QUOTA_LIMIT - x.quota_used),
            reverse=True,
        )

        selected = available_accounts[0]
        logger.info(
            f"Selected YouTube account: {selected.email} (health: {selected.health_score})"
        )

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
            account.health_score = max(
                0, 100 - quota_percentage - (account.error_count * 10)
            )

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

        if "quota" in error_str:
            account.status = AccountStatus.QUOTA_EXCEEDED
            account.last_used = datetime.utcnow()
            logger.warning(f"Quota exceeded for account {account.email}")

        elif "credentials" in error_str or "auth" in error_str:
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
                    token=creds_data.get("token"),
                    refresh_token=account.refresh_token,
                    token_uri=creds_data.get("token_uri"),
                    client_id=creds_data.get("client_id"),
                    client_secret=creds_data.get("client_secret"),
                    scopes=creds_data.get(
                        "scopes",
                        [
                            "https://www.googleapis.com/auth/youtube.upload",
                            "https://www.googleapis.com/auth/youtube",
                            "https://www.googleapis.com/auth/youtubepartner",
                        ],
                    ),
                )

                # Refresh if needed
                if credentials.expired and credentials.refresh_token:
                    credentials.refresh(Request())
                    # Update stored credentials
                    account.credentials_json = credentials.to_json()
                    self.save_account(account)

                # Build YouTube service
                service = build("youtube", "v3", credentials=credentials)
                return service

            else:
                logger.error(f"No credentials for account {account.email}")
                return None

        except Exception as e:
            logger.error(f"Failed to get YouTube service for {account.email}: {e}")
            self.handle_account_error(account, e)
            return None

    def rotate_to_next_account(
        self, current_account: Optional[YouTubeAccount] = None
    ) -> Optional[YouTubeAccount]:
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
        active_accounts = sum(
            1 for acc in self.accounts if acc.status == AccountStatus.ACTIVE
        )

        return {
            "total_accounts": len(self.accounts),
            "active_accounts": active_accounts,
            "total_quota_used": total_quota,
            "total_quota_available": len(self.accounts) * self.DAILY_QUOTA_LIMIT,
            "average_health_score": sum(acc.health_score for acc in self.accounts)
            / len(self.accounts)
            if self.accounts
            else 0,
            "accounts": [
                {
                    "email": acc.email,
                    "status": acc.status.value,
                    "health_score": acc.health_score,
                    "quota_used": acc.quota_used,
                    "quota_limit": self.DAILY_QUOTA_LIMIT,
                }
                for acc in self.accounts
            ],
        }

    async def upload_video_with_rotation(
        self, video_path: str, metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
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
                    "video_id": f"video_{account.account_id}_{datetime.utcnow().timestamp()}",
                    "channel_id": account.channel_id,
                    "account_used": account.email,
                    "upload_time": datetime.utcnow().isoformat(),
                }

                # Update account stats
                account.total_uploads += 1
                self.save_account(account)

                return result

            except Exception as e:
                logger.error(f"Upload failed with account {account.email}: {e}")
                self.handle_account_error(account, e)

                if attempt < max_retries - 1:
                    logger.info(
                        f"Retrying with different account (attempt {attempt + 2}/{max_retries})"
                    )
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
                    "client_secret": os.getenv(
                        f"YOUTUBE_CLIENT_SECRET_{account_index}"
                    ),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
                }
            }

            # Create flow
            flow = Flow.from_client_config(
                client_config,
                scopes=[
                    "https://www.googleapis.com/auth/youtube.upload",
                    "https://www.googleapis.com/auth/youtube",
                    "https://www.googleapis.com/auth/youtubepartner",
                    "https://www.googleapis.com/auth/youtube.force-ssl",
                ],
            )

            flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

            # Get authorization URL
            auth_url, _ = flow.authorization_url(
                access_type="offline", include_granted_scopes="true", prompt="consent"
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
                    "client_secret": os.getenv(
                        f"YOUTUBE_CLIENT_SECRET_{account_index}"
                    ),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
                }
            }

            flow = Flow.from_client_config(
                client_config,
                scopes=[
                    "https://www.googleapis.com/auth/youtube.upload",
                    "https://www.googleapis.com/auth/youtube",
                    "https://www.googleapis.com/auth/youtubepartner",
                    "https://www.googleapis.com/auth/youtube.force-ssl",
                ],
                redirect_uri="urn:ietf:wg:oauth:2.0:oob",
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

    # Enhanced methods for 15 account management

    async def initialize_account_pool(self, target_count: int = None) -> Dict[str, Any]:
        """Initialize and maintain a pool of YouTube accounts"""
        if target_count is None:
            target_count = self.TARGET_ACCOUNT_COUNT

        initialization_results = {
            "target_count": target_count,
            "current_count": len(self.accounts),
            "healthy_count": 0,
            "initialized": [],
            "failed": [],
            "warnings": [],
        }

        # Health check existing accounts
        await self.perform_health_check()

        # Count healthy accounts
        healthy_accounts = [
            acc for acc in self.accounts if self._is_account_healthy(acc)
        ]
        initialization_results["healthy_count"] = len(healthy_accounts)

        # Initialize missing accounts if needed
        missing_count = target_count - len(healthy_accounts)
        if missing_count > 0:
            logger.info(f"Need to initialize {missing_count} additional accounts")
            initialization_results["warnings"].append(
                f"Missing {missing_count} accounts from target pool"
            )

        # Validate account distribution
        account_distribution = self._analyze_account_distribution()
        initialization_results.update(account_distribution)

        return initialization_results

    def _is_account_healthy(self, account: YouTubeAccount) -> bool:
        """Enhanced health check for an account"""
        # Basic availability check
        if not self._is_account_available(account):
            return False

        # Health score threshold
        if account.health_score < self.HEALTH_SCORE_THRESHOLD:
            return False

        # Check for recent errors
        if account.error_count > 3:
            return False

        # Check strikes
        if account.strikes > 2:
            return False

        # Check quota sustainability
        quota_usage_rate = account.quota_used / self.DAILY_QUOTA_LIMIT
        if quota_usage_rate > self.WARNING_QUOTA_THRESHOLD:
            logger.warning(
                f"Account {account.email} quota usage at {quota_usage_rate:.1%}"
            )

        return True

    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check on all accounts"""
        health_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_accounts": len(self.accounts),
            "healthy_accounts": 0,
            "critical_accounts": 0,
            "warning_accounts": 0,
            "suspended_accounts": 0,
            "account_details": [],
            "alerts": [],
        }

        for account in self.accounts:
            try:
                # Test API connectivity
                service = self.get_youtube_service(account)
                if service:
                    # Quick API test - list channels (low quota cost)
                    try:
                        response = (
                            service.channels()
                            .list(part="snippet,statistics", mine=True)
                            .execute()
                        )

                        # Update account stats from API
                        if "items" in response and response["items"]:
                            channel_data = response["items"][0]
                            account.channel_name = channel_data["snippet"]["title"]
                            account.total_views = int(
                                channel_data["statistics"].get("viewCount", 0)
                            )

                        # Mark as successful API call
                        account.error_count = max(0, account.error_count - 1)

                    except Exception as e:
                        logger.warning(f"API test failed for {account.email}: {e}")
                        account.error_count += 1

                # Determine health status
                if account.health_score >= 80:
                    health_results["healthy_accounts"] += 1
                    status = "healthy"
                elif account.health_score >= self.CRITICAL_HEALTH_THRESHOLD:
                    health_results["warning_accounts"] += 1
                    status = "warning"
                else:
                    health_results["critical_accounts"] += 1
                    status = "critical"
                    health_results["alerts"].append(f"Critical health: {account.email}")

                if account.status == AccountStatus.SUSPENDED:
                    health_results["suspended_accounts"] += 1
                    status = "suspended"

                # Update account health trend
                self._update_health_trend(account)

                health_results["account_details"].append(
                    {
                        "email": account.email,
                        "health_score": account.health_score,
                        "status": status,
                        "quota_used": account.quota_used,
                        "quota_percentage": (
                            account.quota_used / self.DAILY_QUOTA_LIMIT
                        )
                        * 100,
                        "error_count": account.error_count,
                        "last_used": account.last_used.isoformat(),
                        "strikes": account.strikes,
                    }
                )

                # Save updated account state
                self.save_account(account)

            except Exception as e:
                logger.error(f"Health check failed for {account.email}: {e}")
                health_results["account_details"].append(
                    {"email": account.email, "status": "error", "error": str(e)}
                )

        # Generate alerts for critical conditions
        if health_results["healthy_accounts"] < self.FAILOVER_THRESHOLD:
            health_results["alerts"].append(
                f"CRITICAL: Only {health_results['healthy_accounts']} healthy accounts remaining"
            )

        return health_results

    def _update_health_trend(self, account: YouTubeAccount):
        """Update health trend metrics for an account"""
        try:
            # Store health trend in Redis
            key = f"youtube:health_trend:{account.account_id}"
            timestamp = int(datetime.utcnow().timestamp())

            # Store last 24 hours of health scores
            self.redis.zadd(key, {timestamp: account.health_score})
            self.redis.expire(key, 86400)  # 24 hours

            # Remove old entries (older than 24 hours)
            old_timestamp = timestamp - 86400
            self.redis.zremrangebyscore(key, 0, old_timestamp)

        except Exception as e:
            logger.error(f"Failed to update health trend for {account.email}: {e}")

    def get_account_with_strategy(
        self, strategy: str = None
    ) -> Optional[YouTubeAccount]:
        """Get account using specified rotation strategy"""
        if strategy is None:
            strategy = self.ROTATION_STRATEGY

        available_accounts = [
            acc for acc in self.accounts if self._is_account_healthy(acc)
        ]

        if not available_accounts:
            logger.error("No healthy accounts available")
            return None

        if strategy == "health_based":
            # Select account with highest health score
            return max(available_accounts, key=lambda x: x.health_score)

        elif strategy == "quota_based":
            # Select account with most available quota
            return max(
                available_accounts, key=lambda x: self.DAILY_QUOTA_LIMIT - x.quota_used
            )

        elif strategy == "weighted_round_robin":
            # Weighted selection based on health and quota
            weights = []
            for acc in available_accounts:
                quota_weight = (
                    self.DAILY_QUOTA_LIMIT - acc.quota_used
                ) / self.DAILY_QUOTA_LIMIT
                health_weight = acc.health_score / 100
                combined_weight = (quota_weight * 0.6) + (health_weight * 0.4)
                weights.append(combined_weight)

            # Select based on weights
            import random

            selected = random.choices(available_accounts, weights=weights)[0]
            return selected

        else:
            # Default: round robin
            return available_accounts[0]

    async def redistribute_quota(self) -> Dict[str, Any]:
        """Redistribute quota across accounts to optimize usage"""
        if not self.QUOTA_REDISTRIBUTION:
            return {"status": "disabled"}

        # Calculate total available quota
        total_quota_available = 0
        quota_distribution = {}

        for account in self.accounts:
            if self._is_account_healthy(account):
                available = self.DAILY_QUOTA_LIMIT - account.quota_used
                total_quota_available += available
                quota_distribution[account.email] = {
                    "used": account.quota_used,
                    "available": available,
                    "health_score": account.health_score,
                }

        redistribution_results = {
            "total_accounts": len(self.accounts),
            "healthy_accounts": len(quota_distribution),
            "total_quota_available": total_quota_available,
            "quota_distribution": quota_distribution,
            "recommendations": [],
        }

        # Generate optimization recommendations
        if total_quota_available < 5000:  # Less than 5k quota across all accounts
            redistribution_results["recommendations"].append(
                "WARNING: Low total quota remaining across all accounts"
            )

        # Identify accounts with very high usage
        high_usage_accounts = [
            email
            for email, data in quota_distribution.items()
            if (data["used"] / self.DAILY_QUOTA_LIMIT) > 0.9
        ]

        if high_usage_accounts:
            redistribution_results["recommendations"].append(
                f"Consider cooling down high-usage accounts: {', '.join(high_usage_accounts)}"
            )

        return redistribution_results

    def _analyze_account_distribution(self) -> Dict[str, Any]:
        """Analyze account distribution and identify optimization opportunities"""
        status_counts = {}
        health_distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        quota_distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        for account in self.accounts:
            # Status distribution
            status = account.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

            # Health distribution
            if account.health_score >= 80:
                health_distribution["excellent"] += 1
            elif account.health_score >= 60:
                health_distribution["good"] += 1
            elif account.health_score >= 40:
                health_distribution["fair"] += 1
            else:
                health_distribution["poor"] += 1

            # Quota distribution
            quota_pct = (account.quota_used / self.DAILY_QUOTA_LIMIT) * 100
            if quota_pct < 25:
                quota_distribution["low"] += 1
            elif quota_pct < 50:
                quota_distribution["medium"] += 1
            elif quota_pct < 80:
                quota_distribution["high"] += 1
            else:
                quota_distribution["critical"] += 1

        return {
            "status_distribution": status_counts,
            "health_distribution": health_distribution,
            "quota_distribution": quota_distribution,
        }

    async def auto_failover_check(self) -> Dict[str, Any]:
        """Check if automatic failover is needed and execute if necessary"""
        healthy_count = sum(1 for acc in self.accounts if self._is_account_healthy(acc))

        failover_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "healthy_account_count": healthy_count,
            "failover_threshold": self.FAILOVER_THRESHOLD,
            "failover_needed": healthy_count < self.FAILOVER_THRESHOLD,
            "actions_taken": [],
        }

        if failover_results["failover_needed"]:
            logger.critical(
                f"FAILOVER TRIGGERED: Only {healthy_count} healthy accounts remaining!"
            )

            # Attempt to recover suspended/cooling accounts
            recovered = await self._attempt_account_recovery()
            failover_results["actions_taken"].extend(recovered)

            # Reset quota for accounts if new day
            reset_count = self._emergency_quota_reset()
            if reset_count > 0:
                failover_results["actions_taken"].append(
                    f"Emergency quota reset for {reset_count} accounts"
                )

            # Send critical alerts
            await self._send_failover_alert(failover_results)

            # Update healthy count after recovery attempts
            healthy_count = sum(
                1 for acc in self.accounts if self._is_account_healthy(acc)
            )
            failover_results["healthy_account_count_after_recovery"] = healthy_count

        return failover_results

    async def _attempt_account_recovery(self) -> List[str]:
        """Attempt to recover suspended or cooling accounts"""
        recovery_actions = []

        for account in self.accounts:
            if account.status == AccountStatus.COOLING_DOWN:
                # Check if cooldown period has passed
                cooldown_end = account.last_used + timedelta(
                    hours=self.COOLDOWN_HOURS // 2
                )  # Reduced for emergency
                if datetime.utcnow() >= cooldown_end:
                    account.status = AccountStatus.ACTIVE
                    account.error_count = max(
                        0, account.error_count - 2
                    )  # Reduce error count
                    self.save_account(account)
                    recovery_actions.append(
                        f"Recovered cooling account: {account.email}"
                    )

            elif account.status == AccountStatus.QUOTA_EXCEEDED:
                # Check if quota should reset
                if datetime.utcnow() >= account.quota_reset_time:
                    self._reset_account_quota(account)
                    recovery_actions.append(f"Reset quota for: {account.email}")

        return recovery_actions

    def _emergency_quota_reset(self) -> int:
        """Emergency quota reset for accounts that might be eligible"""
        reset_count = 0
        current_hour = datetime.utcnow().hour

        # If it's past midnight, consider emergency reset
        if current_hour >= 0 and current_hour <= 6:
            for account in self.accounts:
                if account.status == AccountStatus.QUOTA_EXCEEDED:
                    time_since_reset = (
                        datetime.utcnow() - account.quota_reset_time + timedelta(days=1)
                    )
                    if (
                        time_since_reset.total_seconds() > 21600
                    ):  # 6 hours past reset time
                        self._reset_account_quota(account)
                        reset_count += 1

        return reset_count

    async def _send_failover_alert(self, failover_info: Dict[str, Any]):
        """Send critical failover alert"""
        try:
            # Store alert in Redis for monitoring systems
            alert_key = f"youtube:failover_alert:{int(datetime.utcnow().timestamp())}"
            self.redis.setex(
                alert_key, 3600, json.dumps(failover_info)
            )  # Store for 1 hour

            logger.critical(f"FAILOVER ALERT: {json.dumps(failover_info, indent=2)}")

        except Exception as e:
            logger.error(f"Failed to send failover alert: {e}")

    async def get_advanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all 15 accounts"""
        stats = self.get_account_stats()

        # Add advanced metrics
        advanced_stats = {
            **stats,
            "failover_status": await self.auto_failover_check(),
            "health_check_results": await self.perform_health_check(),
            "quota_redistribution": await self.redistribute_quota(),
            "account_distribution": self._analyze_account_distribution(),
            # Performance metrics
            "performance_metrics": {
                "average_upload_success_rate": self._calculate_success_rate(),
                "quota_efficiency": self._calculate_quota_efficiency(),
                "account_rotation_frequency": self._get_rotation_frequency(),
                "error_rate_trend": self._get_error_trend(),
            },
            # Operational status
            "operational_status": {
                "target_account_count": self.TARGET_ACCOUNT_COUNT,
                "current_healthy_count": sum(
                    1 for acc in self.accounts if self._is_account_healthy(acc)
                ),
                "failover_ready": sum(
                    1 for acc in self.accounts if self._is_account_healthy(acc)
                )
                >= self.FAILOVER_THRESHOLD,
                "quota_sustainability": self._assess_quota_sustainability(),
                "last_health_check": datetime.utcnow().isoformat(),
            },
        }

        return advanced_stats

    def _calculate_success_rate(self) -> float:
        """Calculate overall upload success rate"""
        if not self.accounts:
            return 0.0

        total_uploads = sum(acc.total_uploads for acc in self.accounts)
        total_errors = sum(acc.error_count for acc in self.accounts)

        if total_uploads + total_errors == 0:
            return 100.0

        return (total_uploads / (total_uploads + total_errors)) * 100

    def _calculate_quota_efficiency(self) -> float:
        """Calculate quota efficiency across accounts"""
        total_quota = len(self.accounts) * self.DAILY_QUOTA_LIMIT
        total_used = sum(acc.quota_used for acc in self.accounts)

        if total_quota == 0:
            return 0.0

        return (total_used / total_quota) * 100

    def _get_rotation_frequency(self) -> Dict[str, Any]:
        """Get account rotation frequency statistics"""
        # This would typically be stored in Redis with timestamps
        return {
            "rotations_last_hour": 0,  # Placeholder - implement with Redis tracking
            "most_used_account": max(self.accounts, key=lambda x: x.total_uploads).email
            if self.accounts
            else None,
            "least_used_account": min(
                self.accounts, key=lambda x: x.total_uploads
            ).email
            if self.accounts
            else None,
        }

    def _get_error_trend(self) -> Dict[str, float]:
        """Get error trend across accounts"""
        if not self.accounts:
            return {"current": 0.0, "trend": "stable"}

        total_errors = sum(acc.error_count for acc in self.accounts)
        avg_errors = total_errors / len(self.accounts)

        return {
            "average_errors_per_account": avg_errors,
            "total_errors": total_errors,
            "accounts_with_errors": sum(
                1 for acc in self.accounts if acc.error_count > 0
            ),
            "trend": "increasing"
            if avg_errors > 2
            else "stable"
            if avg_errors > 1
            else "improving",
        }

    def _assess_quota_sustainability(self) -> str:
        """Assess quota sustainability across the account pool"""
        healthy_accounts = [
            acc for acc in self.accounts if self._is_account_healthy(acc)
        ]

        if not healthy_accounts:
            return "critical"

        total_available = sum(
            self.DAILY_QUOTA_LIMIT - acc.quota_used for acc in healthy_accounts
        )
        avg_available = total_available / len(healthy_accounts)

        if avg_available > 5000:
            return "excellent"
        elif avg_available > 2000:
            return "good"
        elif avg_available > 500:
            return "fair"
        else:
            return "poor"


# Singleton instance
_manager_instance = None


def get_youtube_manager() -> YouTubeMultiAccountManager:
    """Get singleton instance of YouTube multi-account manager"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = YouTubeMultiAccountManager()
    return _manager_instance
