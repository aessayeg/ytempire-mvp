"""
YouTube Analytics Data Pipeline
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging
import asyncio
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class VideoAnalytics:
    """Video analytics data structure"""
    video_id: str
    title: str
    published_at: datetime
    views: int
    likes: int
    dislikes: int
    comments: int
    shares: int
    watch_time_minutes: float
    average_view_duration: float
    impression_ctr: float
    unique_viewers: int
    subscriber_change: int
    revenue: float
    rpm: float
    cpm: float
    

@dataclass
class ChannelAnalytics:
    """Channel analytics data structure"""
    channel_id: str
    date: datetime
    subscribers: int
    subscriber_change: int
    views: int
    watch_time_minutes: float
    estimated_revenue: float
    videos_published: int
    average_view_duration: float
    

class YouTubeAnalyticsExtractor:
    """Extract analytics data from YouTube API"""
    
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.youtube = None
        self.analytics = None
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize YouTube API clients"""
        try:
            self.youtube = build(
                'youtube', 
                'v3', 
                developerKey=self.api_keys[self.current_key_index]
            )
            self.analytics = build(
                'youtubeAnalytics',
                'v2',
                developerKey=self.api_keys[self.current_key_index]
            )
        except Exception as e:
            logger.error(f"Failed to initialize YouTube clients: {e}")
            raise
            
    def _rotate_api_key(self):
        """Rotate to next API key when quota exceeded"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logger.info(f"Rotating to API key {self.current_key_index + 1}")
        self._initialize_clients()
        
    async def extract_channel_analytics(self, 
                                       channel_id: str,
                                       start_date: datetime,
                                       end_date: datetime) -> ChannelAnalytics:
        """Extract channel analytics for date range"""
        try:
            # Format dates for API
            start = start_date.strftime('%Y-%m-%d')
            end = end_date.strftime('%Y-%m-%d')
            
            # Get channel statistics
            response = self.analytics.reports().query(
                ids=f'channel=={channel_id}',
                startDate=start,
                endDate=end,
                metrics='views,estimatedMinutesWatched,subscribersGained,subscribersLost,estimatedRevenue',
                dimensions='day'
            ).execute()
            
            # Process response
            if 'rows' in response:
                data = response['rows'][0]
                return ChannelAnalytics(
                    channel_id=channel_id,
                    date=end_date,
                    subscribers=self._get_subscriber_count(channel_id),
                    subscriber_change=data[3] - data[4],  # gained - lost
                    views=data[1],
                    watch_time_minutes=data[2],
                    estimated_revenue=data[5] if len(data) > 5 else 0.0,
                    videos_published=self._count_videos_published(channel_id, start_date, end_date),
                    average_view_duration=data[2] / max(data[1], 1)
                )
            
            return None
            
        except HttpError as e:
            if e.resp.status == 403:  # Quota exceeded
                self._rotate_api_key()
                return await self.extract_channel_analytics(channel_id, start_date, end_date)
            logger.error(f"YouTube API error: {e}")
            raise
            
    async def extract_video_analytics(self, video_id: str) -> VideoAnalytics:
        """Extract analytics for a specific video"""
        try:
            # Get video details
            video_response = self.youtube.videos().list(
                part='statistics,snippet,contentDetails',
                id=video_id
            ).execute()
            
            if not video_response['items']:
                return None
                
            video = video_response['items'][0]
            stats = video['statistics']
            snippet = video['snippet']
            
            # Get analytics data
            analytics_response = self.analytics.reports().query(
                ids=f'video=={video_id}',
                startDate='2020-01-01',  # From beginning
                endDate=datetime.now().strftime('%Y-%m-%d'),
                metrics='estimatedMinutesWatched,averageViewDuration,views,likes,shares,comments,subscribersGained,estimatedRevenue,impressions,impressionClickThroughRate',
                dimensions='video'
            ).execute()
            
            analytics_data = {}
            if 'rows' in analytics_response and analytics_response['rows']:
                row = analytics_response['rows'][0]
                analytics_data = {
                    'watch_time_minutes': row[1],
                    'average_view_duration': row[2],
                    'impression_ctr': row[10] if len(row) > 10 else 0.0,
                    'subscriber_change': row[7] if len(row) > 7 else 0,
                    'revenue': row[8] if len(row) > 8 else 0.0,
                }
            
            return VideoAnalytics(
                video_id=video_id,
                title=snippet['title'],
                published_at=datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00')),
                views=int(stats.get('viewCount', 0)),
                likes=int(stats.get('likeCount', 0)),
                dislikes=int(stats.get('dislikeCount', 0)),
                comments=int(stats.get('commentCount', 0)),
                shares=analytics_data.get('shares', 0),
                watch_time_minutes=analytics_data.get('watch_time_minutes', 0.0),
                average_view_duration=analytics_data.get('average_view_duration', 0.0),
                impression_ctr=analytics_data.get('impression_ctr', 0.0),
                unique_viewers=int(stats.get('viewCount', 0)) * 0.7,  # Estimate
                subscriber_change=analytics_data.get('subscriber_change', 0),
                revenue=analytics_data.get('revenue', 0.0),
                rpm=self._calculate_rpm(analytics_data.get('revenue', 0.0), int(stats.get('viewCount', 0))),
                cpm=self._calculate_cpm(analytics_data.get('revenue', 0.0), int(stats.get('viewCount', 0)))
            )
            
        except HttpError as e:
            if e.resp.status == 403:
                self._rotate_api_key()
                return await self.extract_video_analytics(video_id)
            logger.error(f"YouTube API error: {e}")
            raise
            
    async def extract_trending_videos(self, 
                                     region: str = 'US',
                                     category_id: str = '0',
                                     max_results: int = 50) -> List[Dict[str, Any]]:
        """Extract currently trending videos"""
        try:
            response = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                chart='mostPopular',
                regionCode=region,
                categoryId=category_id if category_id != '0' else None,
                maxResults=max_results
            ).execute()
            
            trending = []
            for item in response.get('items', []):
                trending.append({
                    'video_id': item['id'],
                    'title': item['snippet']['title'],
                    'channel_id': item['snippet']['channelId'],
                    'published_at': item['snippet']['publishedAt'],
                    'views': int(item['statistics'].get('viewCount', 0)),
                    'likes': int(item['statistics'].get('likeCount', 0)),
                    'comments': int(item['statistics'].get('commentCount', 0)),
                    'duration': self._parse_duration(item['contentDetails']['duration']),
                    'tags': item['snippet'].get('tags', []),
                    'category_id': item['snippet']['categoryId'],
                })
                
            return trending
            
        except HttpError as e:
            if e.resp.status == 403:
                self._rotate_api_key()
                return await self.extract_trending_videos(region, category_id, max_results)
            logger.error(f"YouTube API error: {e}")
            raise
            
    async def extract_competitor_data(self, 
                                     competitor_channel_ids: List[str]) -> List[Dict[str, Any]]:
        """Extract competitor channel data"""
        competitors = []
        
        for channel_id in competitor_channel_ids:
            try:
                # Get channel details
                response = self.youtube.channels().list(
                    part='snippet,statistics,contentDetails',
                    id=channel_id
                ).execute()
                
                if response['items']:
                    channel = response['items'][0]
                    
                    # Get recent videos
                    videos_response = self.youtube.search().list(
                        part='snippet',
                        channelId=channel_id,
                        order='date',
                        maxResults=10,
                        type='video'
                    ).execute()
                    
                    recent_videos = []
                    for video in videos_response.get('items', []):
                        video_data = await self.extract_video_analytics(video['id']['videoId'])
                        if video_data:
                            recent_videos.append(asdict(video_data))
                    
                    competitors.append({
                        'channel_id': channel_id,
                        'channel_name': channel['snippet']['title'],
                        'subscribers': int(channel['statistics'].get('subscriberCount', 0)),
                        'total_views': int(channel['statistics'].get('viewCount', 0)),
                        'video_count': int(channel['statistics'].get('videoCount', 0)),
                        'recent_videos': recent_videos,
                        'average_views': np.mean([v['views'] for v in recent_videos]) if recent_videos else 0,
                        'upload_frequency': self._calculate_upload_frequency(recent_videos)
                    })
                    
            except Exception as e:
                logger.error(f"Error extracting competitor {channel_id}: {e}")
                continue
                
        return competitors
        
    def _get_subscriber_count(self, channel_id: str) -> int:
        """Get current subscriber count for channel"""
        try:
            response = self.youtube.channels().list(
                part='statistics',
                id=channel_id
            ).execute()
            
            if response['items']:
                return int(response['items'][0]['statistics'].get('subscriberCount', 0))
            return 0
            
        except Exception as e:
            logger.error(f"Error getting subscriber count: {e}")
            return 0
            
    def _count_videos_published(self, 
                               channel_id: str,
                               start_date: datetime,
                               end_date: datetime) -> int:
        """Count videos published in date range"""
        try:
            response = self.youtube.search().list(
                part='id',
                channelId=channel_id,
                publishedAfter=start_date.isoformat() + 'Z',
                publishedBefore=end_date.isoformat() + 'Z',
                type='video',
                maxResults=50
            ).execute()
            
            return response.get('pageInfo', {}).get('totalResults', 0)
            
        except Exception as e:
            logger.error(f"Error counting videos: {e}")
            return 0
            
    def _calculate_rpm(self, revenue: float, views: int) -> float:
        """Calculate Revenue Per Mille (thousand views)"""
        if views > 0:
            return (revenue / views) * 1000
        return 0.0
        
    def _calculate_cpm(self, revenue: float, views: int) -> float:
        """Calculate Cost Per Mille for advertisers"""
        # CPM is typically 45% of RPM (YouTube's cut)
        rpm = self._calculate_rpm(revenue, views)
        return rpm / 0.55
        
    def _parse_duration(self, duration: str) -> int:
        """Parse ISO 8601 duration to seconds"""
        # Format: PT#H#M#S
        import re
        pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
        match = pattern.match(duration)
        
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds
        return 0
        
    def _calculate_upload_frequency(self, videos: List[Dict[str, Any]]) -> float:
        """Calculate average days between uploads"""
        if len(videos) < 2:
            return 0.0
            
        dates = sorted([
            datetime.fromisoformat(v['published_at'].replace('Z', '+00:00'))
            for v in videos
        ])
        
        total_days = (dates[-1] - dates[0]).days
        return total_days / (len(videos) - 1) if len(videos) > 1 else 0.0


class DataVersioning:
    """Version control for training data"""
    
    def __init__(self, storage_path: str = "./data/versions"):
        self.storage_path = storage_path
        self.current_version = None
        self.versions = []
        
    def create_version(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> str:
        """Create new data version"""
        version_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        version_info = {
            'version_id': version_id,
            'created_at': datetime.now().isoformat(),
            'shape': data.shape,
            'columns': list(data.columns),
            'metadata': metadata
        }
        
        # Save data
        data.to_parquet(f"{self.storage_path}/{version_id}_data.parquet")
        
        # Save version info
        with open(f"{self.storage_path}/{version_id}_info.json", 'w') as f:
            json.dump(version_info, f, indent=2)
            
        self.versions.append(version_info)
        self.current_version = version_id
        
        logger.info(f"Created data version: {version_id}")
        return version_id
        
    def load_version(self, version_id: str) -> pd.DataFrame:
        """Load specific data version"""
        try:
            data = pd.read_parquet(f"{self.storage_path}/{version_id}_data.parquet")
            logger.info(f"Loaded data version: {version_id}")
            return data
        except Exception as e:
            logger.error(f"Failed to load version {version_id}: {e}")
            raise
            
    def get_latest_version(self) -> str:
        """Get latest version ID"""
        if self.versions:
            return self.versions[-1]['version_id']
        return None