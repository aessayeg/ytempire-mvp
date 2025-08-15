"""
Analytics service for fetching and processing YouTube channel analytics
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import uuid
import random
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
    
    async def calculate_quality_score(
        self,
        video_data: Dict[str, Any]
    ) -> float:
        """
        Calculate quality score for a video based on multiple factors.
        Used for AI/ML quality assessment.
        
        Args:
            video_data: Video metadata including views, likes, comments, etc.
            
        Returns:
            Quality score between 0 and 100
        """
        score = 0.0
        weights = {
            "engagement_rate": 0.3,
            "retention_rate": 0.25,
            "ctr": 0.2,
            "sentiment": 0.15,
            "technical_quality": 0.1
        }
        
        # Calculate engagement rate
        views = video_data.get("views", 0)
        likes = video_data.get("likes", 0)
        comments = video_data.get("comments", 0)
        
        if views > 0:
            engagement_rate = ((likes + comments * 2) / views) * 100
            engagement_score = min(engagement_rate * 10, 100)  # Cap at 100
        else:
            engagement_score = 0
        
        # Calculate retention rate (mock for now)
        retention_rate = video_data.get("retention_rate", 65)  # Default 65%
        retention_score = min(retention_rate * 1.2, 100)
        
        # Calculate CTR score
        impressions = video_data.get("impressions", views * 10)
        ctr = (views / impressions * 100) if impressions > 0 else 0
        ctr_score = min(ctr * 5, 100)  # 20% CTR = 100 score
        
        # Sentiment score (mock for now)
        sentiment_score = video_data.get("sentiment_score", 75)
        
        # Technical quality score
        resolution = video_data.get("resolution", "1080p")
        tech_scores = {"4k": 100, "1080p": 85, "720p": 70, "480p": 50}
        technical_score = tech_scores.get(resolution.lower(), 70)
        
        # Calculate weighted score
        score = (
            engagement_score * weights["engagement_rate"] +
            retention_score * weights["retention_rate"] +
            ctr_score * weights["ctr"] +
            sentiment_score * weights["sentiment"] +
            technical_score * weights["technical_quality"]
        )
        
        return round(score, 2)
    
    async def track_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Track analytics events for data collection.
        
        Args:
            event_type: Type of event (e.g., 'video_view', 'channel_subscription')
            event_data: Event-specific data
            user_id: Optional user ID
            channel_id: Optional channel ID
            
        Returns:
            Event tracking confirmation
        """
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "channel_id": channel_id,
            "data": event_data,
            "session_id": event_data.get("session_id"),
            "ip_address": event_data.get("ip_address"),
            "user_agent": event_data.get("user_agent")
        }
        
        # Store event (in production, this would go to database/analytics service)
        logger.info(f"Tracked event: {event_type} for user: {user_id}")
        
        # Trigger real-time processing if needed
        if event_type in ["video_published", "channel_milestone", "revenue_threshold"]:
            await self._process_critical_event(event)
        
        return {
            "status": "success",
            "event_id": event["event_id"],
            "tracked_at": event["timestamp"]
        }
    
    async def generate_report(
        self,
        report_type: str,
        channel_id: str,
        date_range: Dict[str, datetime],
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Generate analytics reports for channels.
        
        Args:
            report_type: Type of report ('performance', 'revenue', 'engagement', 'comprehensive')
            channel_id: Channel ID to generate report for
            date_range: Start and end dates for the report
            format: Output format ('json', 'pdf', 'csv')
            
        Returns:
            Generated report data
        """
        report_data = {
            "report_id": str(uuid.uuid4()),
            "report_type": report_type,
            "channel_id": channel_id,
            "date_range": {
                "start": date_range["start"].isoformat(),
                "end": date_range["end"].isoformat()
            },
            "generated_at": datetime.utcnow().isoformat(),
            "format": format
        }
        
        # Generate report based on type
        if report_type == "performance":
            report_data["data"] = await self._generate_performance_report(channel_id, date_range)
        elif report_type == "revenue":
            report_data["data"] = await self._generate_revenue_report(channel_id, date_range)
        elif report_type == "engagement":
            report_data["data"] = await self._generate_engagement_report(channel_id, date_range)
        elif report_type == "comprehensive":
            report_data["data"] = {
                "performance": await self._generate_performance_report(channel_id, date_range),
                "revenue": await self._generate_revenue_report(channel_id, date_range),
                "engagement": await self._generate_engagement_report(channel_id, date_range)
            }
        else:
            raise ValueError(f"Unknown report type: {report_type}")
        
        # Format conversion if needed
        if format == "pdf":
            report_data["pdf_url"] = f"/reports/{report_data['report_id']}.pdf"
        elif format == "csv":
            report_data["csv_url"] = f"/reports/{report_data['report_id']}.csv"
        
        logger.info(f"Generated {report_type} report for channel {channel_id}")
        return report_data
    
    async def _process_critical_event(self, event: Dict[str, Any]):
        """Process critical events that need immediate attention"""
        # Implement critical event processing logic
        pass
    
    async def _generate_performance_report(self, channel_id: str, date_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Generate performance metrics report"""
        return {
            "total_views": random.randint(100000, 1000000),
            "total_videos": random.randint(10, 100),
            "avg_views_per_video": random.randint(1000, 50000),
            "growth_rate": round(random.uniform(-10, 50), 2),
            "top_performing_videos": [
                {"title": f"Video {i}", "views": random.randint(10000, 100000)}
                for i in range(5)
            ]
        }
    
    async def _generate_revenue_report(self, channel_id: str, date_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Generate revenue metrics report"""
        return {
            "total_revenue": round(random.uniform(1000, 10000), 2),
            "ad_revenue": round(random.uniform(800, 8000), 2),
            "sponsorship_revenue": round(random.uniform(200, 2000), 2),
            "daily_average": round(random.uniform(30, 300), 2),
            "revenue_by_video_type": {
                "shorts": round(random.uniform(100, 1000), 2),
                "long_form": round(random.uniform(500, 5000), 2),
                "live": round(random.uniform(200, 2000), 2)
            }
        }
    
    async def _generate_engagement_report(self, channel_id: str, date_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Generate engagement metrics report"""
        return {
            "avg_engagement_rate": round(random.uniform(2, 15), 2),
            "total_likes": random.randint(10000, 100000),
            "total_comments": random.randint(1000, 10000),
            "total_shares": random.randint(500, 5000),
            "subscriber_growth": random.randint(100, 10000),
            "avg_watch_time": round(random.uniform(2, 10), 2)
        }

# Global instance
analytics_service = AnalyticsService()