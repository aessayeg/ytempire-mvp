"""
YouTube OAuth Service
Handles OAuth flow for multiple YouTube accounts
"""
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import os

from app.core.config import settings

logger = logging.getLogger(__name__)


class YouTubeOAuthService:
    """
    Manages OAuth authentication for multiple YouTube accounts
    """

    # OAuth2 scopes required for YouTube operations
    SCOPES = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube",
        "https://www.googleapis.com/auth/youtube.readonly",
        "https://www.googleapis.com/auth/youtubepartner",
    ]

    def __init__(self):
        self.redirect_uri = f"{settings.API_V1_STR}/youtube/oauth/callback"
        self.accounts_file = "backend/config/youtube_accounts.json"
        self.load_accounts()

    def load_accounts(self):
        """Load YouTube accounts configuration"""
        try:
            with open(self.accounts_file, "r") as f:
                self.accounts_config = json.load(f)
                self.accounts = {
                    acc["account_id"]: acc for acc in self.accounts_config["accounts"]
                }
                logger.info(f"Loaded {len(self.accounts)} YouTube accounts")
        except Exception as e:
            logger.error(f"Failed to load YouTube accounts: {e}")
            self.accounts = {}

    def save_accounts(self):
        """Save updated accounts configuration"""
        try:
            self.accounts_config["accounts"] = list(self.accounts.values())
            with open(self.accounts_file, "w") as f:
                json.dump(self.accounts_config, f, indent=2)
            logger.info("YouTube accounts configuration saved")
        except Exception as e:
            logger.error(f"Failed to save YouTube accounts: {e}")

    def get_auth_url(
        self, account_id: str, client_secrets_file: Optional[str] = None
    ) -> str:
        """
        Generate OAuth authorization URL for a specific account

        Args:
            account_id: YouTube account identifier
            client_secrets_file: Path to client secrets JSON file

        Returns:
            Authorization URL for user to grant permissions
        """
        try:
            if account_id not in self.accounts:
                raise ValueError(f"Account {account_id} not found")

            # Use provided secrets file or environment variable
            secrets_file = client_secrets_file or os.environ.get(
                "GOOGLE_CLIENT_SECRETS_FILE"
            )
            if not secrets_file or not os.path.exists(secrets_file):
                # Create temporary client secrets from environment variables
                secrets_file = self._create_temp_secrets_file()

            # Create OAuth flow
            flow = Flow.from_client_secrets_file(
                secrets_file, scopes=self.SCOPES, redirect_uri=self.redirect_uri
            )

            # Store state for verification
            flow.state = f"{account_id}:{os.urandom(16).hex()}"

            # Generate authorization URL
            auth_url, _ = flow.authorization_url(
                prompt="consent", access_type="offline", include_granted_scopes="true"
            )

            logger.info(f"Generated auth URL for account {account_id}")
            return auth_url

        except Exception as e:
            logger.error(f"Failed to generate auth URL: {e}")
            raise

    def handle_callback(
        self, account_id: str, authorization_code: str
    ) -> Dict[str, Any]:
        """
        Handle OAuth callback and store credentials

        Args:
            account_id: YouTube account identifier
            authorization_code: Authorization code from OAuth callback

        Returns:
            Account information with stored credentials
        """
        try:
            if account_id not in self.accounts:
                raise ValueError(f"Account {account_id} not found")

            # Create OAuth flow
            secrets_file = (
                os.environ.get("GOOGLE_CLIENT_SECRETS_FILE")
                or self._create_temp_secrets_file()
            )
            flow = Flow.from_client_secrets_file(
                secrets_file, scopes=self.SCOPES, redirect_uri=self.redirect_uri
            )

            # Exchange authorization code for credentials
            flow.fetch_token(code=authorization_code)
            credentials = flow.credentials

            # Store credentials
            account = self.accounts[account_id]
            account["refresh_token"] = credentials.refresh_token
            account["access_token"] = credentials.token
            account["token_expiry"] = (
                (datetime.utcnow() + timedelta(seconds=credentials.expiry)).isoformat()
                if credentials.expiry
                else None
            )
            account["authorized"] = True
            account["authorized_at"] = datetime.utcnow().isoformat()

            # Get channel information
            youtube = build("youtube", "v3", credentials=credentials)
            channel_response = (
                youtube.channels().list(part="snippet,statistics", mine=True).execute()
            )

            if channel_response.get("items"):
                channel = channel_response["items"][0]
                account["channel_id"] = channel["id"]
                account["channel_name"] = channel["snippet"]["title"]
                account["subscriber_count"] = int(
                    channel["statistics"].get("subscriberCount", 0)
                )

            # Save updated configuration
            self.save_accounts()

            logger.info(f"Successfully authorized account {account_id}")
            return account

        except Exception as e:
            logger.error(f"Failed to handle OAuth callback: {e}")
            raise

    def get_credentials(self, account_id: str) -> Optional[Credentials]:
        """
        Get valid credentials for a YouTube account

        Args:
            account_id: YouTube account identifier

        Returns:
            Valid Google OAuth2 credentials or None
        """
        try:
            if account_id not in self.accounts:
                return None

            account = self.accounts[account_id]

            if not account.get("refresh_token"):
                logger.warning(f"No refresh token for account {account_id}")
                return None

            # Create credentials from stored tokens
            credentials = Credentials(
                token=account.get("access_token"),
                refresh_token=account.get("refresh_token"),
                token_uri="https://oauth2.googleapis.com/token",
                client_id=os.environ.get("GOOGLE_CLIENT_ID"),
                client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
                scopes=self.SCOPES,
            )

            # Refresh if expired
            if not credentials.valid:
                if credentials.expired and credentials.refresh_token:
                    credentials.refresh(Request())

                    # Update stored tokens
                    account["access_token"] = credentials.token
                    account["token_expiry"] = (
                        datetime.utcnow() + timedelta(seconds=3600)
                    ).isoformat()
                    self.save_accounts()

                    logger.info(f"Refreshed credentials for account {account_id}")

            return credentials

        except Exception as e:
            logger.error(f"Failed to get credentials for account {account_id}: {e}")
            return None

    def get_youtube_service(self, account_id: str):
        """
        Get authenticated YouTube API service for an account

        Args:
            account_id: YouTube account identifier

        Returns:
            Authenticated YouTube API service object
        """
        credentials = self.get_credentials(account_id)
        if not credentials:
            raise ValueError(f"No valid credentials for account {account_id}")

        return build("youtube", "v3", credentials=credentials)

    def list_authorized_accounts(self) -> list:
        """Get list of authorized YouTube accounts"""
        authorized = []
        for account_id, account in self.accounts.items():
            if account.get("refresh_token"):
                authorized.append(
                    {
                        "account_id": account_id,
                        "email": account.get("email"),
                        "channel_name": account.get("channel_name"),
                        "channel_id": account.get("channel_id"),
                        "authorized_at": account.get("authorized_at"),
                    }
                )
        return authorized

    def _create_temp_secrets_file(self) -> str:
        """Create temporary client secrets file from environment variables"""
        client_id = os.environ.get("GOOGLE_CLIENT_ID")
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise ValueError("Google OAuth credentials not configured in environment")

        secrets = {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [self.redirect_uri],
            }
        }

        # Save to temporary file
        temp_file = "/tmp/client_secrets.json"
        with open(temp_file, "w") as f:
            json.dump(secrets, f)

        return temp_file

    def check_account_health(self, account_id: str) -> Dict[str, Any]:
        """
        Check health status of a YouTube account

        Args:
            account_id: YouTube account identifier

        Returns:
            Health status information
        """
        try:
            youtube = self.get_youtube_service(account_id)

            # Check quota usage
            # Note: YouTube doesn't provide direct quota API, this is a placeholder
            # In production, track API calls and estimate quota usage

            # Try a simple API call to verify credentials work
            response = youtube.channels().list(part="snippet", mine=True).execute()

            return {
                "account_id": account_id,
                "status": "healthy",
                "channel_id": response["items"][0]["id"]
                if response.get("items")
                else None,
                "checked_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Health check failed for account {account_id}: {e}")
            return {
                "account_id": account_id,
                "status": "unhealthy",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat(),
            }


# Global instance
youtube_oauth_service = YouTubeOAuthService()
