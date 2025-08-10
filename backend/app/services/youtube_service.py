"""
YouTube API Service
Owner: Integration Specialist
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import io
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
import httplib2

from app.core.config import settings
from app.services.vault_service import get_api_keys, encrypt_sensitive_data, decrypt_sensitive_data

logger = logging.getLogger(__name__)

# YouTube API scopes
SCOPES = [
    'https://www.googleapis.com/auth/youtube.upload',
    'https://www.googleapis.com/auth/youtube.readonly',
    'https://www.googleapis.com/auth/youtube.force-ssl'
]

# API quotas and limits
QUOTA_LIMITS = {
    'video_upload': 1600,  # Cost per upload
    'search': 100,         # Cost per search
    'analytics': 200,      # Cost per analytics request
    'daily_limit': 10000   # Total daily quota
}


class YouTubeService:
    """YouTube API service for video management."""
    
    def __init__(self):
        self.service = None
        self.credentials = None
        self.quota_used = 0
        self.last_quota_reset = datetime.utcnow().date()
    
    async def initialize(self, user_id: str) -> bool:
        """Initialize YouTube service with user credentials."""
        try:
            # Get API keys from Vault
            api_keys = await get_api_keys()
            
            if not api_keys.get('youtube_client_id') or not api_keys.get('youtube_client_secret'):
                logger.error("YouTube API credentials not found in Vault")
                return False
            
            # Load or create credentials for the user
            credentials = await self._load_user_credentials(user_id, api_keys)
            
            if not credentials or not credentials.valid:
                logger.error(f"Invalid YouTube credentials for user {user_id}")
                return False
            
            # Build service
            self.service = build('youtube', 'v3', credentials=credentials)
            self.credentials = credentials
            
            logger.info(f"YouTube service initialized for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize YouTube service: {str(e)}")
            return False
    
    async def _load_user_credentials(self, user_id: str, api_keys: Dict) -> Optional[Credentials]:
        """Load user's YouTube credentials."""
        try:
            from app.services.vault_service import get_vault_service
            vault = await get_vault_service()
            
            # Try to get stored credentials
            stored_creds = await vault.get_secret(f"ytempire/users/{user_id}/youtube_creds")
            
            if stored_creds:
                # Decrypt and restore credentials
                decrypted_creds = await decrypt_sensitive_data(stored_creds.get('encrypted_token'))
                if decrypted_creds:
                    creds_data = json.loads(decrypted_creds)
                    credentials = Credentials.from_authorized_user_info(creds_data)
                    
                    # Refresh if expired
                    if credentials.expired and credentials.refresh_token:
                        credentials.refresh(Request())
                        # Update stored credentials
                        await self._store_user_credentials(user_id, credentials)
                    
                    return credentials
            
            # No stored credentials, need OAuth flow
            # In production, this would redirect user to OAuth flow
            logger.warning(f"No YouTube credentials found for user {user_id}. OAuth flow required.")
            return None
            
        except Exception as e:
            logger.error(f"Error loading YouTube credentials: {str(e)}")
            return None
    
    async def _store_user_credentials(self, user_id: str, credentials: Credentials) -> bool:
        """Store user's YouTube credentials securely."""
        try:
            # Encrypt credentials
            creds_json = credentials.to_json()
            encrypted_creds = await encrypt_sensitive_data(creds_json)
            
            if not encrypted_creds:
                return False
            
            from app.services.vault_service import get_vault_service
            vault = await get_vault_service()
            
            # Store encrypted credentials
            success = await vault.set_secret(
                f"ytempire/users/{user_id}/youtube_creds",
                {
                    'encrypted_token': encrypted_creds,
                    'updated_at': datetime.utcnow().isoformat()
                }
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing YouTube credentials: {str(e)}")
            return False
    
    async def start_oauth_flow(self, user_id: str) -> str:
        """Start OAuth flow for YouTube authorization."""
        try:
            api_keys = await get_api_keys()
            
            client_config = {
                'web': {
                    'client_id': api_keys['youtube_client_id'],
                    'client_secret': api_keys['youtube_client_secret'],
                    'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
                    'token_uri': 'https://oauth2.googleapis.com/token',
                    'redirect_uris': ['http://localhost:8000/api/v1/youtube/callback']
                }
            }
            
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            flow.redirect_uri = 'http://localhost:8000/api/v1/youtube/callback'
            
            auth_url, _ = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                state=user_id  # Pass user_id in state
            )
            
            return auth_url
            
        except Exception as e:
            logger.error(f"Error starting OAuth flow: {str(e)}")
            return ""
    
    async def handle_oauth_callback(self, user_id: str, code: str) -> bool:
        """Handle OAuth callback and store credentials."""
        try:
            api_keys = await get_api_keys()
            
            client_config = {
                'web': {
                    'client_id': api_keys['youtube_client_id'],
                    'client_secret': api_keys['youtube_client_secret'],
                    'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
                    'token_uri': 'https://oauth2.googleapis.com/token',
                    'redirect_uris': ['http://localhost:8000/api/v1/youtube/callback']
                }
            }
            
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            flow.redirect_uri = 'http://localhost:8000/api/v1/youtube/callback'
            
            # Exchange code for credentials
            flow.fetch_token(code=code)
            credentials = flow.credentials
            
            # Store credentials
            success = await self._store_user_credentials(user_id, credentials)
            
            if success:
                logger.info(f"YouTube authorization completed for user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error handling OAuth callback: {str(e)}")
            return False
    
    def _check_quota(self, operation: str) -> bool:
        """Check if operation is within quota limits."""
        # Reset quota if new day
        current_date = datetime.utcnow().date()
        if current_date > self.last_quota_reset:
            self.quota_used = 0
            self.last_quota_reset = current_date
        
        operation_cost = QUOTA_LIMITS.get(operation, 1)
        
        if self.quota_used + operation_cost > QUOTA_LIMITS['daily_limit']:
            logger.warning(f"YouTube quota limit reached. Used: {self.quota_used}")
            return False
        
        self.quota_used += operation_cost
        return True
    
    async def upload_video(
        self,
        video_file_path: str,
        title: str,
        description: str,
        tags: List[str] = None,
        category_id: str = "22",  # People & Blogs
        privacy_status: str = "private",
        thumbnail_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload video to YouTube."""
        
        if not self.service:
            raise Exception("YouTube service not initialized")
        
        if not self._check_quota('video_upload'):
            raise Exception("YouTube quota limit exceeded")
        
        try:
            # Prepare video metadata
            body = {
                'snippet': {
                    'title': title[:100],  # YouTube title limit
                    'description': description[:5000],  # YouTube description limit
                    'tags': tags[:30] if tags else [],  # Max 30 tags
                    'categoryId': category_id
                },
                'status': {
                    'privacyStatus': privacy_status,
                    'embeddable': True,
                    'selfDeclaredMadeForKids': False
                }
            }
            
            # Create media upload
            media = MediaFileUpload(
                video_file_path,
                chunksize=-1,
                resumable=True,
                mimetype='video/*'
            )
            
            # Execute upload
            insert_request = self.service.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            # Resumable upload
            video_id = None
            response = None
            error = None
            retry = 0
            
            while response is None and retry < 3:
                try:
                    status, response = insert_request.next_chunk()
                    if response is not None:
                        if 'id' in response:
                            video_id = response['id']
                            logger.info(f"Video uploaded successfully: {video_id}")
                        else:
                            raise Exception(f"Upload failed: {response}")
                except HttpError as e:
                    if e.resp.status in [500, 502, 503, 504]:
                        # Retriable error
                        retry += 1
                        await asyncio.sleep(2 ** retry)
                    else:
                        raise e
            
            if not video_id:
                raise Exception("Failed to upload video after retries")
            
            # Upload thumbnail if provided
            if thumbnail_path and os.path.exists(thumbnail_path):
                await self._upload_thumbnail(video_id, thumbnail_path)
            
            return {
                'video_id': video_id,
                'url': f'https://www.youtube.com/watch?v={video_id}',
                'status': 'uploaded',
                'privacy_status': privacy_status
            }
            
        except Exception as e:
            logger.error(f"Video upload failed: {str(e)}")
            raise e
    
    async def _upload_thumbnail(self, video_id: str, thumbnail_path: str) -> bool:
        """Upload thumbnail for video."""
        try:
            if not self._check_quota('video_upload'):  # Same cost as upload
                return False
            
            media = MediaFileUpload(thumbnail_path, mimetype='image/*')
            
            self.service.thumbnails().set(
                videoId=video_id,
                media_body=media
            ).execute()
            
            logger.info(f"Thumbnail uploaded for video {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Thumbnail upload failed: {str(e)}")
            return False
    
    async def get_video_stats(self, video_id: str) -> Dict[str, Any]:
        """Get video statistics."""
        if not self.service:
            raise Exception("YouTube service not initialized")
        
        if not self._check_quota('analytics'):
            raise Exception("YouTube quota limit exceeded")
        
        try:
            response = self.service.videos().list(
                part='statistics,snippet,status',
                id=video_id
            ).execute()
            
            if not response.get('items'):
                return {}
            
            item = response['items'][0]
            stats = item.get('statistics', {})
            snippet = item.get('snippet', {})
            status = item.get('status', {})
            
            return {
                'video_id': video_id,
                'title': snippet.get('title'),
                'published_at': snippet.get('publishedAt'),
                'views': int(stats.get('viewCount', 0)),
                'likes': int(stats.get('likeCount', 0)),
                'comments': int(stats.get('commentCount', 0)),
                'privacy_status': status.get('privacyStatus'),
                'upload_status': status.get('uploadStatus'),
                'description': snippet.get('description', ''),
                'tags': snippet.get('tags', [])
            }
            
        except Exception as e:
            logger.error(f"Failed to get video stats: {str(e)}")
            return {}
    
    async def get_channel_info(self) -> Dict[str, Any]:
        """Get channel information."""
        if not self.service:
            raise Exception("YouTube service not initialized")
        
        if not self._check_quota('analytics'):
            raise Exception("YouTube quota limit exceeded")
        
        try:
            response = self.service.channels().list(
                part='snippet,statistics,status',
                mine=True
            ).execute()
            
            if not response.get('items'):
                return {}
            
            item = response['items'][0]
            snippet = item.get('snippet', {})
            stats = item.get('statistics', {})
            
            return {
                'channel_id': item.get('id'),
                'title': snippet.get('title'),
                'description': snippet.get('description'),
                'custom_url': snippet.get('customUrl'),
                'subscriber_count': int(stats.get('subscriberCount', 0)),
                'video_count': int(stats.get('videoCount', 0)),
                'view_count': int(stats.get('viewCount', 0)),
                'created_at': snippet.get('publishedAt')
            }
            
        except Exception as e:
            logger.error(f"Failed to get channel info: {str(e)}")
            return {}
    
    async def update_video(
        self,
        video_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        privacy_status: Optional[str] = None
    ) -> bool:
        """Update video metadata."""
        if not self.service:
            raise Exception("YouTube service not initialized")
        
        try:
            # Get current video details
            current = self.service.videos().list(
                part='snippet,status',
                id=video_id
            ).execute()
            
            if not current.get('items'):
                return False
            
            video = current['items'][0]
            snippet = video['snippet']
            status = video['status']
            
            # Update fields
            if title:
                snippet['title'] = title[:100]
            if description:
                snippet['description'] = description[:5000]
            if tags:
                snippet['tags'] = tags[:30]
            if privacy_status:
                status['privacyStatus'] = privacy_status
            
            # Update video
            self.service.videos().update(
                part='snippet,status',
                body={
                    'id': video_id,
                    'snippet': snippet,
                    'status': status
                }
            ).execute()
            
            logger.info(f"Video {video_id} updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update video {video_id}: {str(e)}")
            return False
    
    async def delete_video(self, video_id: str) -> bool:
        """Delete video from YouTube."""
        if not self.service:
            raise Exception("YouTube service not initialized")
        
        try:
            self.service.videos().delete(id=video_id).execute()
            logger.info(f"Video {video_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete video {video_id}: {str(e)}")
            return False
    
    async def search_videos(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for videos."""
        if not self.service:
            raise Exception("YouTube service not initialized")
        
        if not self._check_quota('search'):
            raise Exception("YouTube quota limit exceeded")
        
        try:
            response = self.service.search().list(
                q=query,
                part='snippet',
                type='video',
                maxResults=min(max_results, 50),
                order='relevance'
            ).execute()
            
            results = []
            for item in response.get('items', []):
                snippet = item.get('snippet', {})
                results.append({
                    'video_id': item['id']['videoId'],
                    'title': snippet.get('title'),
                    'description': snippet.get('description'),
                    'channel_title': snippet.get('channelTitle'),
                    'published_at': snippet.get('publishedAt'),
                    'thumbnail_url': snippet.get('thumbnails', {}).get('default', {}).get('url')
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Video search failed: {str(e)}")
            return []
    
    async def get_quota_usage(self) -> Dict[str, Any]:
        """Get current quota usage."""
        return {
            'quota_used': self.quota_used,
            'daily_limit': QUOTA_LIMITS['daily_limit'],
            'remaining': QUOTA_LIMITS['daily_limit'] - self.quota_used,
            'last_reset': self.last_quota_reset.isoformat(),
            'percentage_used': round((self.quota_used / QUOTA_LIMITS['daily_limit']) * 100, 2)
        }


# Global YouTube service instances (per user)
_youtube_services: Dict[str, YouTubeService] = {}


async def get_youtube_service(user_id: str) -> YouTubeService:
    """Get or create YouTube service for user."""
    if user_id not in _youtube_services:
        service = YouTubeService()
        success = await service.initialize(user_id)
        if success:
            _youtube_services[user_id] = service
        else:
            raise Exception(f"Failed to initialize YouTube service for user {user_id}")
    
    return _youtube_services[user_id]


async def cleanup_youtube_services():
    """Cleanup YouTube service instances."""
    global _youtube_services
    _youtube_services.clear()
    logger.info("YouTube services cleaned up")