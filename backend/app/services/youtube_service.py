"""
YouTube API Integration Service
Handles all YouTube Data API v3 operations
"""
import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.http import MediaFileUpload
import httplib2
from dataclasses import dataclass
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

@dataclass
class YouTubeConfig:
    """YouTube API configuration"""
    api_key: str
    client_secrets_file: str = "client_secrets.json"
    oauth_scope: List[str] = None
    api_service_name: str = "youtube"
    api_version: str = "v3"
    max_results: int = 50
    upload_chunk_size: int = 1024 * 1024  # 1MB chunks
    
    def __post_init__(self):
        if self.oauth_scope is None:
            self.oauth_scope = [
                "https://www.googleapis.com/auth/youtube",
                "https://www.googleapis.com/auth/youtube.upload",
                "https://www.googleapis.com/auth/youtube.readonly",
                "https://www.googleapis.com/auth/youtubepartner",
                "https://www.googleapis.com/auth/youtube.force-ssl"
            ]

class YouTubeService:
    """Main YouTube API service"""
    
    def __init__(self, config: YouTubeConfig = None):
        self.config = config or YouTubeConfig(
            api_key=os.getenv("YOUTUBE_API_KEY", "")
        )
        self.youtube = None
        self.authenticated_service = None
        self._quota_tracker = QuotaTracker()
        
    def initialize_api_client(self):
        """Initialize YouTube API client with API key"""
        if not self.config.api_key:
            raise ValueError("YouTube API key not configured")
            
        self.youtube = build(
            self.config.api_service_name,
            self.config.api_version,
            developerKey=self.config.api_key,
            cache_discovery=False
        )
        logger.info("YouTube API client initialized")
        
    def authenticate_oauth(self, credentials_file: str = "youtube_credentials.json"):
        """Authenticate using OAuth 2.0 for write operations"""
        store = Storage(credentials_file)
        credentials = store.get()
        
        if not credentials or credentials.invalid:
            flow = flow_from_clientsecrets(
                self.config.client_secrets_file,
                scope=self.config.oauth_scope
            )
            credentials = run_flow(flow, store)
            
        self.authenticated_service = build(
            self.config.api_service_name,
            self.config.api_version,
            http=credentials.authorize(httplib2.Http()),
            cache_discovery=False
        )
        logger.info("YouTube OAuth authentication successful")
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_videos(
        self,
        query: str,
        channel_id: Optional[str] = None,
        max_results: int = 25,
        order: str = "relevance",
        published_after: Optional[datetime] = None
    ) -> List[Dict]:
        """Search for videos on YouTube"""
        try:
            if not self.youtube:
                self.initialize_api_client()
                
            search_params = {
                "q": query,
                "type": "video",
                "part": "id,snippet",
                "maxResults": min(max_results, self.config.max_results),
                "order": order
            }
            
            if channel_id:
                search_params["channelId"] = channel_id
                
            if published_after:
                search_params["publishedAfter"] = published_after.isoformat() + "Z"
                
            response = self.youtube.search().list(**search_params).execute()
            
            self._quota_tracker.add_units(100)  # Search costs 100 quota units
            
            videos = []
            for item in response.get("items", []):
                videos.append({
                    "video_id": item["id"]["videoId"],
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"],
                    "channel_id": item["snippet"]["channelId"],
                    "channel_title": item["snippet"]["channelTitle"],
                    "published_at": item["snippet"]["publishedAt"],
                    "thumbnail_url": item["snippet"]["thumbnails"]["high"]["url"]
                })
                
            return videos
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            raise
            
    async def get_video_details(self, video_id: str) -> Dict:
        """Get detailed information about a video"""
        try:
            if not self.youtube:
                self.initialize_api_client()
                
            response = self.youtube.videos().list(
                part="snippet,statistics,contentDetails,status",
                id=video_id
            ).execute()
            
            self._quota_tracker.add_units(3)  # Video details cost 3 quota units
            
            if not response["items"]:
                return None
                
            video = response["items"][0]
            
            return {
                "video_id": video["id"],
                "title": video["snippet"]["title"],
                "description": video["snippet"]["description"],
                "channel_id": video["snippet"]["channelId"],
                "channel_title": video["snippet"]["channelTitle"],
                "published_at": video["snippet"]["publishedAt"],
                "duration": self._parse_duration(video["contentDetails"]["duration"]),
                "privacy_status": video["status"]["privacyStatus"],
                "tags": video["snippet"].get("tags", []),
                "category_id": video["snippet"]["categoryId"],
                "statistics": {
                    "view_count": int(video["statistics"].get("viewCount", 0)),
                    "like_count": int(video["statistics"].get("likeCount", 0)),
                    "comment_count": int(video["statistics"].get("commentCount", 0)),
                    "favorite_count": int(video["statistics"].get("favoriteCount", 0))
                }
            }
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            raise
            
    async def get_channel_details(self, channel_id: str) -> Dict:
        """Get channel information and statistics"""
        try:
            if not self.youtube:
                self.initialize_api_client()
                
            response = self.youtube.channels().list(
                part="snippet,statistics,contentDetails,brandingSettings",
                id=channel_id
            ).execute()
            
            self._quota_tracker.add_units(3)  # Channel details cost 3 quota units
            
            if not response["items"]:
                return None
                
            channel = response["items"][0]
            
            return {
                "channel_id": channel["id"],
                "title": channel["snippet"]["title"],
                "description": channel["snippet"]["description"],
                "custom_url": channel["snippet"].get("customUrl"),
                "published_at": channel["snippet"]["publishedAt"],
                "country": channel["snippet"].get("country"),
                "uploads_playlist": channel["contentDetails"]["relatedPlaylists"]["uploads"],
                "statistics": {
                    "subscriber_count": int(channel["statistics"].get("subscriberCount", 0)),
                    "view_count": int(channel["statistics"].get("viewCount", 0)),
                    "video_count": int(channel["statistics"].get("videoCount", 0))
                },
                "keywords": channel["brandingSettings"]["channel"].get("keywords", "").split()
            }
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            raise
            
    async def get_channel_videos(
        self,
        channel_id: str,
        max_results: int = 50,
        order: str = "date"
    ) -> List[Dict]:
        """Get videos from a specific channel"""
        try:
            # First get the uploads playlist ID
            channel_details = await self.get_channel_details(channel_id)
            if not channel_details:
                return []
                
            uploads_playlist_id = channel_details["uploads_playlist"]
            
            if not self.youtube:
                self.initialize_api_client()
                
            videos = []
            next_page_token = None
            
            while len(videos) < max_results:
                response = self.youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    playlistId=uploads_playlist_id,
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_page_token
                ).execute()
                
                self._quota_tracker.add_units(3)  # Playlist items cost 3 quota units
                
                for item in response.get("items", []):
                    videos.append({
                        "video_id": item["contentDetails"]["videoId"],
                        "title": item["snippet"]["title"],
                        "description": item["snippet"]["description"],
                        "published_at": item["contentDetails"]["videoPublishedAt"],
                        "thumbnail_url": item["snippet"]["thumbnails"]["high"]["url"]
                    })
                    
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
                    
            return videos
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            raise
            
    async def upload_video(
        self,
        video_file_path: str,
        title: str,
        description: str,
        tags: List[str],
        category_id: str = "22",  # People & Blogs
        privacy_status: str = "private",
        thumbnail_path: Optional[str] = None
    ) -> Dict:
        """Upload a video to YouTube"""
        try:
            if not self.authenticated_service:
                self.authenticate_oauth()
                
            body = {
                "snippet": {
                    "title": title,
                    "description": description,
                    "tags": tags,
                    "categoryId": category_id
                },
                "status": {
                    "privacyStatus": privacy_status,
                    "selfDeclaredMadeForKids": False
                }
            }
            
            # Create media upload object
            media = MediaFileUpload(
                video_file_path,
                chunksize=self.config.upload_chunk_size,
                resumable=True,
                mimetype="video/*"
            )
            
            # Execute upload
            request = self.authenticated_service.videos().insert(
                part=",".join(body.keys()),
                body=body,
                media_body=media
            )
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    logger.info(f"Upload progress: {int(status.progress() * 100)}%")
                    
            self._quota_tracker.add_units(1600)  # Video upload costs 1600 quota units
            
            video_id = response["id"]
            
            # Upload thumbnail if provided
            if thumbnail_path and os.path.exists(thumbnail_path):
                await self.upload_thumbnail(video_id, thumbnail_path)
                
            return {
                "video_id": video_id,
                "title": response["snippet"]["title"],
                "description": response["snippet"]["description"],
                "published_at": response["snippet"]["publishedAt"],
                "privacy_status": response["status"]["privacyStatus"],
                "upload_status": response["status"]["uploadStatus"]
            }
            
        except HttpError as e:
            logger.error(f"YouTube upload error: {e}")
            raise
            
    async def upload_thumbnail(self, video_id: str, thumbnail_path: str) -> bool:
        """Upload a custom thumbnail for a video"""
        try:
            if not self.authenticated_service:
                self.authenticate_oauth()
                
            media = MediaFileUpload(
                thumbnail_path,
                mimetype="image/jpeg",
                resumable=True
            )
            
            request = self.authenticated_service.thumbnails().set(
                videoId=video_id,
                media_body=media
            )
            
            response = request.execute()
            
            self._quota_tracker.add_units(50)  # Thumbnail upload costs 50 quota units
            
            return True
            
        except HttpError as e:
            logger.error(f"Thumbnail upload error: {e}")
            return False
            
    async def update_video(
        self,
        video_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category_id: Optional[str] = None,
        privacy_status: Optional[str] = None
    ) -> Dict:
        """Update video metadata"""
        try:
            if not self.authenticated_service:
                self.authenticate_oauth()
                
            # First get current video details
            video = await self.get_video_details(video_id)
            
            # Build update body
            body = {
                "id": video_id,
                "snippet": {
                    "title": title or video["title"],
                    "description": description or video["description"],
                    "tags": tags or video["tags"],
                    "categoryId": category_id or video["category_id"]
                }
            }
            
            if privacy_status:
                body["status"] = {"privacyStatus": privacy_status}
                
            request = self.authenticated_service.videos().update(
                part="snippet,status",
                body=body
            )
            
            response = request.execute()
            
            self._quota_tracker.add_units(50)  # Video update costs 50 quota units
            
            return {
                "video_id": response["id"],
                "title": response["snippet"]["title"],
                "description": response["snippet"]["description"],
                "privacy_status": response.get("status", {}).get("privacyStatus")
            }
            
        except HttpError as e:
            logger.error(f"Video update error: {e}")
            raise
            
    async def delete_video(self, video_id: str) -> bool:
        """Delete a video from YouTube"""
        try:
            if not self.authenticated_service:
                self.authenticate_oauth()
                
            request = self.authenticated_service.videos().delete(id=video_id)
            request.execute()
            
            self._quota_tracker.add_units(50)  # Video deletion costs 50 quota units
            
            return True
            
        except HttpError as e:
            logger.error(f"Video deletion error: {e}")
            return False
            
    async def get_analytics(
        self,
        channel_id: str,
        start_date: datetime,
        end_date: datetime,
        metrics: List[str] = None
    ) -> Dict:
        """Get YouTube Analytics data"""
        # Note: This requires YouTube Analytics API which is separate
        # Placeholder for analytics implementation
        if metrics is None:
            metrics = ["views", "likes", "comments", "shares", "estimatedMinutesWatched"]
            
        # This would use the YouTube Analytics API
        # For now, return mock data structure
        return {
            "channel_id": channel_id,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "metrics": {metric: 0 for metric in metrics}
        }
        
    async def get_trending_videos(
        self,
        region_code: str = "US",
        category_id: Optional[str] = None,
        max_results: int = 50
    ) -> List[Dict]:
        """Get trending videos for a specific region"""
        try:
            if not self.youtube:
                self.initialize_api_client()
                
            params = {
                "part": "snippet,statistics",
                "chart": "mostPopular",
                "regionCode": region_code,
                "maxResults": min(max_results, self.config.max_results)
            }
            
            if category_id:
                params["videoCategoryId"] = category_id
                
            response = self.youtube.videos().list(**params).execute()
            
            self._quota_tracker.add_units(3)  # Trending videos cost 3 quota units
            
            videos = []
            for item in response.get("items", []):
                videos.append({
                    "video_id": item["id"],
                    "title": item["snippet"]["title"],
                    "channel_id": item["snippet"]["channelId"],
                    "channel_title": item["snippet"]["channelTitle"],
                    "published_at": item["snippet"]["publishedAt"],
                    "view_count": int(item["statistics"].get("viewCount", 0)),
                    "like_count": int(item["statistics"].get("likeCount", 0)),
                    "category_id": item["snippet"]["categoryId"]
                })
                
            return videos
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            raise
            
    def _parse_duration(self, duration: str) -> int:
        """Parse ISO 8601 duration to seconds"""
        # Format: PT#H#M#S
        import re
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
        if not match:
            return 0
            
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        
        return hours * 3600 + minutes * 60 + seconds
        
    def get_quota_usage(self) -> Dict:
        """Get current quota usage statistics"""
        return self._quota_tracker.get_stats()


class QuotaTracker:
    """Track YouTube API quota usage"""
    
    def __init__(self, daily_limit: int = 10000):
        self.daily_limit = daily_limit
        self.usage = []
        self.reset_time = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
        
    def add_units(self, units: int):
        """Add quota units used"""
        now = datetime.now()
        if now >= self.reset_time:
            self.usage = []
            self.reset_time = now.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            
        self.usage.append({"units": units, "timestamp": now})
        
    def get_stats(self) -> Dict:
        """Get quota usage statistics"""
        total_used = sum(u["units"] for u in self.usage)
        remaining = self.daily_limit - total_used
        
        return {
            "daily_limit": self.daily_limit,
            "used": total_used,
            "remaining": remaining,
            "percentage_used": (total_used / self.daily_limit) * 100,
            "reset_time": self.reset_time.isoformat(),
            "operations_count": len(self.usage)
        }
        
    def is_quota_available(self, required_units: int) -> bool:
        """Check if enough quota is available"""
        total_used = sum(u["units"] for u in self.usage)
        return (total_used + required_units) <= self.daily_limit