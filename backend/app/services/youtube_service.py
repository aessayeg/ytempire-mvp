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

# OAuth2 imports - using google-auth instead of deprecated oauth2client
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials as OAuth2Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False
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
                "https://www.googleapis.com/auth/youtube.force-ssl",
            ]


class YouTubeService:
    """Main YouTube API service"""

    def __init__(self, config: YouTubeConfig = None):
        self.config = config or YouTubeConfig(api_key=os.getenv("YOUTUBE_API_KEY", ""))
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
            cache_discovery=False,
        )
        logger.info("YouTube API client initialized")

    def authenticate_oauth(self, credentials_file: str = "youtube_credentials.json"):
        """Authenticate using OAuth 2.0 for write operations"""
        if not OAUTH_AVAILABLE:
            logger.warning("OAuth dependencies not available, using API key only")
            self.authenticated_service = self.youtube
            return

        credentials = None
        # Load existing credentials if they exist
        if os.path.exists(credentials_file):
            credentials = OAuth2Credentials.from_authorized_user_file(
                credentials_file, self.config.oauth_scope
            )

        # If there are no (valid) credentials available, request authorization
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                if not os.path.exists(self.config.client_secrets_file):
                    logger.warning(
                        "Client secrets file not found, using API key authentication"
                    )
                    self.authenticated_service = self.youtube
                    return

                flow = InstalledAppFlow.from_client_secrets_file(
                    self.config.client_secrets_file, self.config.oauth_scope
                )
                credentials = flow.run_local_server(port=0)

            # Save credentials for next run
            with open(credentials_file, "w") as token:
                token.write(credentials.to_json())

        self.authenticated_service = build(
            self.config.api_service_name,
            self.config.api_version,
            credentials=credentials,
            cache_discovery=False,
        )
        logger.info("YouTube OAuth authentication successful")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def search_videos(
        self,
        query: str,
        channel_id: Optional[str] = None,
        max_results: int = 25,
        order: str = "relevance",
        published_after: Optional[datetime] = None,
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
                "order": order,
            }

            if channel_id:
                search_params["channelId"] = channel_id

            if published_after:
                search_params["publishedAfter"] = published_after.isoformat() + "Z"

            response = self.youtube.search().list(**search_params).execute()

            self._quota_tracker.add_units(100)  # Search costs 100 quota units

            videos = []
            for item in response.get("items", []):
                videos.append(
                    {
                        "video_id": item["id"]["videoId"],
                        "title": item["snippet"]["title"],
                        "description": item["snippet"]["description"],
                        "channel_id": item["snippet"]["channelId"],
                        "channel_title": item["snippet"]["channelTitle"],
                        "published_at": item["snippet"]["publishedAt"],
                        "thumbnail_url": item["snippet"]["thumbnails"]["high"]["url"],
                    }
                )

            return videos

        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            raise

    async def get_video_details(self, video_id: str) -> Dict:
        """Get detailed information about a video"""
        try:
            if not self.youtube:
                self.initialize_api_client()

            response = (
                self.youtube.videos()
                .list(part="snippet,statistics,contentDetails,status", id=video_id)
                .execute()
            )

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
                    "favorite_count": int(video["statistics"].get("favoriteCount", 0)),
                },
            }

        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            raise

    async def get_channel_details(self, channel_id: str) -> Dict:
        """Get channel information and statistics"""
        try:
            if not self.youtube:
                self.initialize_api_client()

            response = (
                self.youtube.channels()
                .list(
                    part="snippet,statistics,contentDetails,brandingSettings",
                    id=channel_id,
                )
                .execute()
            )

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
                "uploads_playlist": channel["contentDetails"]["relatedPlaylists"][
                    "uploads"
                ],
                "statistics": {
                    "subscriber_count": int(
                        channel["statistics"].get("subscriberCount", 0)
                    ),
                    "view_count": int(channel["statistics"].get("viewCount", 0)),
                    "video_count": int(channel["statistics"].get("videoCount", 0)),
                },
                "keywords": channel["brandingSettings"]["channel"]
                .get("keywords", "")
                .split(),
            }

        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            raise

    async def get_channel_videos(
        self, channel_id: str, max_results: int = 50, order: str = "date"
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
                response = (
                    self.youtube.playlistItems()
                    .list(
                        part="snippet,contentDetails",
                        playlistId=uploads_playlist_id,
                        maxResults=min(50, max_results - len(videos)),
                        pageToken=next_page_token,
                    )
                    .execute()
                )

                self._quota_tracker.add_units(3)  # Playlist items cost 3 quota units

                for item in response.get("items", []):
                    videos.append(
                        {
                            "video_id": item["contentDetails"]["videoId"],
                            "title": item["snippet"]["title"],
                            "description": item["snippet"]["description"],
                            "published_at": item["contentDetails"]["videoPublishedAt"],
                            "thumbnail_url": item["snippet"]["thumbnails"]["high"][
                                "url"
                            ],
                        }
                    )

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
        thumbnail_path: Optional[str] = None,
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
                    "categoryId": category_id,
                },
                "status": {
                    "privacyStatus": privacy_status,
                    "selfDeclaredMadeForKids": False,
                },
            }

            # Create media upload object
            media = MediaFileUpload(
                video_file_path,
                chunksize=self.config.upload_chunk_size,
                resumable=True,
                mimetype="video/*",
            )

            # Execute upload
            request = self.authenticated_service.videos().insert(
                part=",".join(body.keys()), body=body, media_body=media
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
                "upload_status": response["status"]["uploadStatus"],
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
                thumbnail_path, mimetype="image/jpeg", resumable=True
            )

            request = self.authenticated_service.thumbnails().set(
                videoId=video_id, media_body=media
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
        privacy_status: Optional[str] = None,
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
                    "categoryId": category_id or video["category_id"],
                },
            }

            if privacy_status:
                body["status"] = {"privacyStatus": privacy_status}

            request = self.authenticated_service.videos().update(
                part="snippet,status", body=body
            )

            response = request.execute()

            self._quota_tracker.add_units(50)  # Video update costs 50 quota units

            return {
                "video_id": response["id"],
                "title": response["snippet"]["title"],
                "description": response["snippet"]["description"],
                "privacy_status": response.get("status", {}).get("privacyStatus"),
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
        metrics: List[str] = None,
    ) -> Dict:
        """Get YouTube Analytics data using Analytics API v2"""
        try:
            if not self.authenticated_service:
                self.authenticate_oauth()

            if metrics is None:
                metrics = [
                    "views",
                    "likes",
                    "comments",
                    "shares",
                    "estimatedMinutesWatched",
                    "averageViewDuration",
                    "subscribersGained",
                    "subscribersLost",
                ]

            # Build the analytics API client
            analytics_service = build(
                "youtubeAnalytics",
                "v2",
                http=self.authenticated_service._http,
                cache_discovery=False,
            )

            # Query analytics data
            response = (
                analytics_service.reports()
                .query(
                    ids=f"channel=={channel_id}",
                    startDate=start_date.strftime("%Y-%m-%d"),
                    endDate=end_date.strftime("%Y-%m-%d"),
                    metrics=",".join(metrics),
                    dimensions="day",
                )
                .execute()
            )

            self._quota_tracker.add_units(10)  # Analytics API quota usage

            # Process the response
            analytics_data = {
                "channel_id": channel_id,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "metrics": {},
                "daily_data": [],
            }

            if "rows" in response:
                # Extract column headers
                headers = [col["name"] for col in response["columnHeaders"]]

                # Process each row
                for row in response["rows"]:
                    daily_data = dict(zip(headers, row))
                    analytics_data["daily_data"].append(daily_data)

                # Calculate totals
                for metric in metrics:
                    if metric in headers[1:]:  # Skip day column
                        metric_index = headers.index(metric)
                        total = sum(
                            row[metric_index]
                            for row in response["rows"]
                            if len(row) > metric_index
                        )
                        analytics_data["metrics"][metric] = total

            return analytics_data

        except HttpError as e:
            logger.error(f"YouTube Analytics API error: {e}")
            # Fallback to mock data if API fails
            return {
                "channel_id": channel_id,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "metrics": {metric: 0 for metric in metrics or []},
                "error": str(e),
            }

    async def get_trending_videos(
        self,
        region_code: str = "US",
        category_id: Optional[str] = None,
        max_results: int = 50,
    ) -> List[Dict]:
        """Get trending videos for a specific region"""
        try:
            if not self.youtube:
                self.initialize_api_client()

            params = {
                "part": "snippet,statistics",
                "chart": "mostPopular",
                "regionCode": region_code,
                "maxResults": min(max_results, self.config.max_results),
            }

            if category_id:
                params["videoCategoryId"] = category_id

            response = self.youtube.videos().list(**params).execute()

            self._quota_tracker.add_units(3)  # Trending videos cost 3 quota units

            videos = []
            for item in response.get("items", []):
                videos.append(
                    {
                        "video_id": item["id"],
                        "title": item["snippet"]["title"],
                        "channel_id": item["snippet"]["channelId"],
                        "channel_title": item["snippet"]["channelTitle"],
                        "published_at": item["snippet"]["publishedAt"],
                        "view_count": int(item["statistics"].get("viewCount", 0)),
                        "like_count": int(item["statistics"].get("likeCount", 0)),
                        "category_id": item["snippet"]["categoryId"],
                    }
                )

            return videos

        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            raise

    def _parse_duration(self, duration: str) -> int:
        """Parse ISO 8601 duration to seconds"""
        # Format: PT#H#M#S
        import re

        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
        if not match:
            return 0

        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)

        return hours * 3600 + minutes * 60 + seconds

    def get_quota_usage(self) -> Dict:
        """Get current quota usage statistics"""
        return self._quota_tracker.get_stats()

    async def create_playlist(
        self, title: str, description: str = "", privacy_status: str = "private"
    ) -> Dict:
        """Create a new playlist"""
        try:
            if not self.authenticated_service:
                self.authenticate_oauth()

            body = {
                "snippet": {"title": title, "description": description},
                "status": {"privacyStatus": privacy_status},
            }

            response = (
                self.authenticated_service.playlists()
                .insert(part="snippet,status", body=body)
                .execute()
            )

            self._quota_tracker.add_units(50)  # Playlist creation costs 50 units

            return {
                "playlist_id": response["id"],
                "title": response["snippet"]["title"],
                "description": response["snippet"]["description"],
                "privacy_status": response["status"]["privacyStatus"],
            }

        except HttpError as e:
            logger.error(f"Playlist creation error: {e}")
            raise

    async def add_video_to_playlist(
        self, playlist_id: str, video_id: str, position: Optional[int] = None
    ) -> bool:
        """Add a video to a playlist"""
        try:
            if not self.authenticated_service:
                self.authenticate_oauth()

            body = {
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {"kind": "youtube#video", "videoId": video_id},
                }
            }

            if position is not None:
                body["snippet"]["position"] = position

            self.authenticated_service.playlistItems().insert(
                part="snippet", body=body
            ).execute()

            self._quota_tracker.add_units(50)  # Playlist item insert costs 50 units

            return True

        except HttpError as e:
            logger.error(f"Add to playlist error: {e}")
            return False

    async def get_video_comments(
        self, video_id: str, max_results: int = 100, order: str = "relevance"
    ) -> List[Dict]:
        """Get comments for a video"""
        try:
            if not self.youtube:
                self.initialize_api_client()

            comments = []
            next_page_token = None

            while len(comments) < max_results:
                response = (
                    self.youtube.commentThreads()
                    .list(
                        part="snippet,replies",
                        videoId=video_id,
                        maxResults=min(100, max_results - len(comments)),
                        order=order,
                        pageToken=next_page_token,
                    )
                    .execute()
                )

                self._quota_tracker.add_units(3)  # Comment threads cost 3 units

                for item in response.get("items", []):
                    top_comment = item["snippet"]["topLevelComment"]["snippet"]
                    comment_data = {
                        "comment_id": item["id"],
                        "author_name": top_comment["authorDisplayName"],
                        "author_channel_id": top_comment["authorChannelId"]["value"]
                        if "authorChannelId" in top_comment
                        else None,
                        "text": top_comment["textDisplay"],
                        "like_count": top_comment["likeCount"],
                        "published_at": top_comment["publishedAt"],
                        "reply_count": item["snippet"]["totalReplyCount"],
                    }

                    # Add replies if present
                    if "replies" in item:
                        replies = []
                        for reply in item["replies"]["comments"]:
                            reply_snippet = reply["snippet"]
                            replies.append(
                                {
                                    "author_name": reply_snippet["authorDisplayName"],
                                    "text": reply_snippet["textDisplay"],
                                    "like_count": reply_snippet["likeCount"],
                                    "published_at": reply_snippet["publishedAt"],
                                }
                            )
                        comment_data["replies"] = replies

                    comments.append(comment_data)

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

            return comments

        except HttpError as e:
            logger.error(f"Get comments error: {e}")
            raise

    async def moderate_comment(
        self,
        comment_id: str,
        action: str = "approve",  # approve, reject, hold_for_review
    ) -> bool:
        """Moderate a comment (approve/reject/hold)"""
        try:
            if not self.authenticated_service:
                self.authenticate_oauth()

            if action == "approve":
                self.authenticated_service.comments().setModerationStatus(
                    id=comment_id, moderationStatus="published"
                ).execute()
            elif action == "reject":
                self.authenticated_service.comments().setModerationStatus(
                    id=comment_id, moderationStatus="rejected"
                ).execute()
            elif action == "hold_for_review":
                self.authenticated_service.comments().setModerationStatus(
                    id=comment_id, moderationStatus="heldForReview"
                ).execute()
            else:
                raise ValueError(f"Invalid moderation action: {action}")

            self._quota_tracker.add_units(50)  # Comment moderation costs 50 units

            return True

        except HttpError as e:
            logger.error(f"Comment moderation error: {e}")
            return False

    async def set_video_thumbnail_from_options(
        self, video_id: str, thumbnail_option: str = "default"  # default, 1, 2, 3
    ) -> bool:
        """Set video thumbnail from YouTube's auto-generated options"""
        try:
            if not self.authenticated_service:
                self.authenticate_oauth()

            # YouTube automatically generates thumbnails, we can only upload custom ones
            # This is a placeholder for thumbnail selection from auto-generated options
            # In practice, you'd need to use a custom thumbnail

            logger.info(
                f"Thumbnail selection not directly supported by API for video {video_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Thumbnail selection error: {e}")
            return False

    async def get_video_thumbnail_options(self, video_id: str) -> List[Dict]:
        """Get available thumbnail options for a video"""
        try:
            video_details = await self.get_video_details(video_id)

            if not video_details:
                return []

            # Extract thumbnail URLs from video details
            thumbnails = []
            # YouTube provides default, medium, high, standard, maxres thumbnails
            for quality in ["default", "medium", "high", "standard", "maxres"]:
                thumbnail_url = f"https://img.youtube.com/vi/{video_id}/{quality}.jpg"
                thumbnails.append(
                    {
                        "quality": quality,
                        "url": thumbnail_url,
                        "recommended": quality == "maxres",  # Recommend highest quality
                    }
                )

            return thumbnails

        except Exception as e:
            logger.error(f"Get thumbnail options error: {e}")
            return []


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
            "operations_count": len(self.usage),
        }

    def is_quota_available(self, required_units: int) -> bool:
        """Check if enough quota is available"""
        total_used = sum(u["units"] for u in self.usage)
        return (total_used + required_units) <= self.daily_limit
