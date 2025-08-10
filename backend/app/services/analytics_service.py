"""
Analytics service for fetching and processing YouTube channel analytics
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import httpx

from app.models.channel import Channel
from app.models.video import Video
from app.core.config import settings
from app.db.session import get_db

logger = logging.getLogger(__name__)

class AnalyticsService:
    """
    Service for managing YouTube analytics and metrics
    """
    
    def __init__(self):
        self.youtube_api_base = "https://www.googleapis.com/youtube/v3"
        self.analytics_api_base = "https://youtubeanalytics.googleapis.com/v2"
    
    async def fetch_channel_analytics(
        self,
        channel_id: str,
        youtube_channel_id: str,
        api_key: str
    ) -> Dict[str, Any]:
        """
        Fetch comprehensive channel analytics from YouTube
        """
        try:
            async with httpx.AsyncClient() as client:
                # Fetch channel statistics
                channel_response = await client.get(
                    f"{self.youtube_api_base}/channels",
                    params={
                        "part": "statistics,snippet",
                        "id": youtube_channel_id,
                        "key": api_key
                    }
                )
                
                if channel_response.status_code != 200:
                    logger.error(f"Failed to fetch channel stats: {channel_response.text}")
                    return {}
                
                channel_data = channel_response.json()
                
                if not channel_data.get("items"):
                    logger.warning(f"No channel data found for {youtube_channel_id}")
                    return {}
                
                stats = channel_data["items"][0]["statistics"]
                snippet = channel_data["items"][0]["snippet"]
                
                # Update channel in database
                async with get_db() as db:
                    await db.execute(
                        update(Channel)
                        .where(Channel.id == channel_id)
                        .values(
                            subscriber_count=int(stats.get("subscriberCount", 0)),
                            total_views=int(stats.get("viewCount", 0)),
                            total_videos=int(stats.get("videoCount", 0)),
                            last_sync_at=datetime.utcnow()
                        )
                    )
                    await db.commit()
                
                return {
                    "subscriber_count": int(stats.get("subscriberCount", 0)),
                    "view_count": int(stats.get("viewCount", 0)),
                    "video_count": int(stats.get("videoCount", 0)),
                    "channel_title": snippet.get("title"),
                    "channel_description": snippet.get("description"),
                    "published_at": snippet.get("publishedAt")
                }
                
        except Exception as e:
            logger.error(f"Error fetching channel analytics: {str(e)}")
            return {}
    
    async def fetch_video_analytics(
        self,
        video_id: str,
        youtube_video_id: str,
        api_key: str
    ) -> Dict[str, Any]:
        """
        Fetch analytics for a specific video
        """
        try:
            async with httpx.AsyncClient() as client:
                # Fetch video statistics
                response = await client.get(
                    f"{self.youtube_api_base}/videos",
                    params={
                        "part": "statistics,snippet,contentDetails",
                        "id": youtube_video_id,
                        "key": api_key
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to fetch video stats: {response.text}")
                    return {}
                
                data = response.json()
                
                if not data.get("items"):
                    logger.warning(f"No video data found for {youtube_video_id}")
                    return {}
                
                stats = data["items"][0]["statistics"]
                snippet = data["items"][0]["snippet"]
                content = data["items"][0]["contentDetails"]
                
                # Update video in database
                async with get_db() as db:
                    await db.execute(
                        update(Video)
                        .where(Video.id == video_id)
                        .values(
                            view_count=int(stats.get("viewCount", 0)),
                            like_count=int(stats.get("likeCount", 0)),
                            dislike_count=int(stats.get("dislikeCount", 0)),
                            comment_count=int(stats.get("commentCount", 0)),
                            updated_at=datetime.utcnow()
                        )
                    )
                    await db.commit()
                
                return {
                    "view_count": int(stats.get("viewCount", 0)),
                    "like_count": int(stats.get("likeCount", 0)),
                    "comment_count": int(stats.get("commentCount", 0)),
                    "duration": content.get("duration"),
                    "title": snippet.get("title"),
                    "description": snippet.get("description"),
                    "published_at": snippet.get("publishedAt")
                }
                
        except Exception as e:
            logger.error(f"Error fetching video analytics: {str(e)}")
            return {}
    
    async def calculate_engagement_rate(
        self,
        views: int,
        likes: int,
        comments: int,
        shares: int = 0
    ) -> float:
        """
        Calculate engagement rate for content
        """
        if views == 0:
            return 0.0
        
        total_engagements = likes + comments + shares
        engagement_rate = (total_engagements / views) * 100
        
        return round(engagement_rate, 2)
    
    async def get_trending_topics(
        self,
        category: str,
        region: str = "US",
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get trending topics for a specific category
        """
        try:
            api_key = api_key or settings.YOUTUBE_API_KEY
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.youtube_api_base}/videos",
                    params={
                        "part": "snippet,statistics",
                        "chart": "mostPopular",
                        "regionCode": region,
                        "videoCategoryId": self._get_category_id(category),
                        "maxResults": 10,
                        "key": api_key
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to fetch trending topics: {response.text}")
                    return []
                
                data = response.json()
                trending = []
                
                for item in data.get("items", []):
                    trending.append({
                        "title": item["snippet"]["title"],
                        "video_id": item["id"],
                        "view_count": int(item["statistics"].get("viewCount", 0)),
                        "published_at": item["snippet"]["publishedAt"],
                        "tags": item["snippet"].get("tags", [])[:5]
                    })
                
                return trending
                
        except Exception as e:
            logger.error(f"Error fetching trending topics: {str(e)}")
            return []
    
    def _get_category_id(self, category: str) -> str:
        """
        Map category names to YouTube category IDs
        """
        category_map = {
            "gaming": "20",
            "education": "27",
            "technology": "28",
            "entertainment": "24",
            "music": "10",
            "sports": "17",
            "news": "25",
            "howto": "26",
            "travel": "19",
            "food": "26",
            "fashion": "26",
            "fitness": "17",
            "business": "27",
            "science": "28"
        }
        
        return category_map.get(category.lower(), "24")  # Default to entertainment
    
    async def analyze_competition(
        self,
        category: str,
        subscriber_range: tuple = (1000, 100000),
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze competitor channels in the same category
        """
        try:
            api_key = api_key or settings.YOUTUBE_API_KEY
            
            async with httpx.AsyncClient() as client:
                # Search for channels in the category
                response = await client.get(
                    f"{self.youtube_api_base}/search",
                    params={
                        "part": "snippet",
                        "type": "channel",
                        "q": category,
                        "maxResults": 10,
                        "order": "relevance",
                        "key": api_key
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to search channels: {response.text}")
                    return []
                
                data = response.json()
                channel_ids = [item["id"]["channelId"] for item in data.get("items", [])]
                
                if not channel_ids:
                    return []
                
                # Get detailed channel information
                channels_response = await client.get(
                    f"{self.youtube_api_base}/channels",
                    params={
                        "part": "statistics,snippet",
                        "id": ",".join(channel_ids),
                        "key": api_key
                    }
                )
                
                if channels_response.status_code != 200:
                    return []
                
                channels_data = channels_response.json()
                competitors = []
                
                for item in channels_data.get("items", []):
                    stats = item["statistics"]
                    subscriber_count = int(stats.get("subscriberCount", 0))
                    
                    # Filter by subscriber range
                    if subscriber_range[0] <= subscriber_count <= subscriber_range[1]:
                        competitors.append({
                            "channel_id": item["id"],
                            "channel_title": item["snippet"]["title"],
                            "subscriber_count": subscriber_count,
                            "view_count": int(stats.get("viewCount", 0)),
                            "video_count": int(stats.get("videoCount", 0)),
                            "avg_views": int(stats.get("viewCount", 0)) // max(int(stats.get("videoCount", 1)), 1),
                            "description": item["snippet"]["description"][:200]
                        })
                
                # Sort by subscriber count
                competitors.sort(key=lambda x: x["subscriber_count"], reverse=True)
                
                return competitors[:5]  # Return top 5 competitors
                
        except Exception as e:
            logger.error(f"Error analyzing competition: {str(e)}")
            return []
    
    async def calculate_revenue_estimate(
        self,
        views: int,
        category: str,
        is_monetized: bool = True
    ) -> Dict[str, float]:
        """
        Estimate revenue based on views and category
        """
        if not is_monetized:
            return {"low": 0.0, "mid": 0.0, "high": 0.0}
        
        # CPM rates by category (in dollars)
        cpm_rates = {
            "gaming": {"low": 0.5, "mid": 2.0, "high": 4.0},
            "education": {"low": 2.0, "mid": 5.0, "high": 10.0},
            "technology": {"low": 2.0, "mid": 6.0, "high": 12.0},
            "entertainment": {"low": 0.5, "mid": 2.5, "high": 5.0},
            "music": {"low": 0.3, "mid": 1.5, "high": 3.0},
            "business": {"low": 3.0, "mid": 8.0, "high": 15.0},
            "default": {"low": 0.5, "mid": 2.0, "high": 4.0}
        }
        
        rates = cpm_rates.get(category.lower(), cpm_rates["default"])
        
        # Calculate revenue (CPM is per 1000 views)
        revenue_low = (views / 1000) * rates["low"]
        revenue_mid = (views / 1000) * rates["mid"]
        revenue_high = (views / 1000) * rates["high"]
        
        return {
            "low": round(revenue_low, 2),
            "mid": round(revenue_mid, 2),
            "high": round(revenue_high, 2)
        }

# Global instance
analytics_service = AnalyticsService()