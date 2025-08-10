"""
YouTube Analytics Data Connector
Owner: Data Engineer
"""

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class YouTubeAnalyticsConnector:
    """
    Connector for YouTube Analytics API
    Handles data extraction for channels and videos
    """
    
    def __init__(self, credentials: Credentials):
        """Initialize YouTube Analytics API client"""
        self.youtube_analytics = build('youtubeAnalytics', 'v2', credentials=credentials)
        self.youtube = build('youtube', 'v3', credentials=credentials)
        
    def get_channel_analytics(
        self, 
        channel_id: str,
        start_date: str,
        end_date: str,
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Get analytics data for a channel
        
        Args:
            channel_id: YouTube channel ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            metrics: List of metrics to fetch
        
        Returns:
            DataFrame with analytics data
        """
        if metrics is None:
            metrics = [
                'views',
                'estimatedMinutesWatched',
                'averageViewDuration',
                'subscribersGained',
                'subscribersLost',
                'likes',
                'dislikes',
                'comments',
                'shares'
            ]
        
        try:
            response = self.youtube_analytics.reports().query(
                ids=f'channel=={channel_id}',
                startDate=start_date,
                endDate=end_date,
                metrics=','.join(metrics),
                dimensions='day',
                sort='day'
            ).execute()
            
            # Convert to DataFrame
            headers = [h['name'] for h in response['columnHeaders']]
            rows = response.get('rows', [])
            df = pd.DataFrame(rows, columns=headers)
            
            # Add metadata
            df['channel_id'] = channel_id
            df['fetched_at'] = datetime.utcnow()
            
            return df
            
        except HttpError as e:
            logger.error(f"Error fetching channel analytics: {e}")
            raise
    
    def get_video_analytics(
        self,
        video_id: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get analytics data for a specific video
        
        Args:
            video_id: YouTube video ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with video analytics
        """
        metrics = [
            'views',
            'estimatedMinutesWatched',
            'averageViewDuration',
            'averageViewPercentage',
            'likes',
            'dislikes',
            'comments',
            'shares',
            'subscribersGained'
        ]
        
        try:
            response = self.youtube_analytics.reports().query(
                ids='channel==MINE',
                filters=f'video=={video_id}',
                startDate=start_date,
                endDate=end_date,
                metrics=','.join(metrics),
                dimensions='day',
                sort='day'
            ).execute()
            
            headers = [h['name'] for h in response['columnHeaders']]
            rows = response.get('rows', [])
            df = pd.DataFrame(rows, columns=headers)
            
            df['video_id'] = video_id
            df['fetched_at'] = datetime.utcnow()
            
            return df
            
        except HttpError as e:
            logger.error(f"Error fetching video analytics: {e}")
            raise
    
    def get_traffic_sources(
        self,
        channel_id: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Get traffic source breakdown for channel"""
        try:
            response = self.youtube_analytics.reports().query(
                ids=f'channel=={channel_id}',
                startDate=start_date,
                endDate=end_date,
                metrics='views,estimatedMinutesWatched',
                dimensions='insightTrafficSourceType',
                sort='-views'
            ).execute()
            
            headers = [h['name'] for h in response['columnHeaders']]
            rows = response.get('rows', [])
            df = pd.DataFrame(rows, columns=headers)
            
            return df
            
        except HttpError as e:
            logger.error(f"Error fetching traffic sources: {e}")
            raise
    
    def get_demographics(
        self,
        channel_id: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Get demographic data for channel"""
        demographics = {}
        
        # Age and Gender
        try:
            response = self.youtube_analytics.reports().query(
                ids=f'channel=={channel_id}',
                startDate=start_date,
                endDate=end_date,
                metrics='viewerPercentage',
                dimensions='ageGroup,gender',
                sort='-viewerPercentage'
            ).execute()
            
            headers = [h['name'] for h in response['columnHeaders']]
            rows = response.get('rows', [])
            demographics['age_gender'] = pd.DataFrame(rows, columns=headers)
            
        except HttpError as e:
            logger.error(f"Error fetching demographics: {e}")
        
        # Geography
        try:
            response = self.youtube_analytics.reports().query(
                ids=f'channel=={channel_id}',
                startDate=start_date,
                endDate=end_date,
                metrics='views,estimatedMinutesWatched',
                dimensions='country',
                sort='-views',
                maxResults=25
            ).execute()
            
            headers = [h['name'] for h in response['columnHeaders']]
            rows = response.get('rows', [])
            demographics['geography'] = pd.DataFrame(rows, columns=headers)
            
        except HttpError as e:
            logger.error(f"Error fetching geography data: {e}")
        
        return demographics
    
    def get_retention_data(
        self,
        video_id: str
    ) -> pd.DataFrame:
        """Get audience retention data for a video"""
        try:
            response = self.youtube_analytics.reports().query(
                ids='channel==MINE',
                filters=f'video=={video_id}',
                metrics='audienceWatchRatio',
                dimensions='elapsedVideoTimeRatio'
            ).execute()
            
            headers = [h['name'] for h in response['columnHeaders']]
            rows = response.get('rows', [])
            df = pd.DataFrame(rows, columns=headers)
            
            # Convert ratio to percentage
            df['elapsed_percentage'] = df['elapsedVideoTimeRatio'] * 100
            df['retention_percentage'] = df['audienceWatchRatio'] * 100
            
            return df
            
        except HttpError as e:
            logger.error(f"Error fetching retention data: {e}")
            raise
    
    def get_search_terms(
        self,
        channel_id: str,
        start_date: str,
        end_date: str,
        max_results: int = 25
    ) -> pd.DataFrame:
        """Get search terms that led to channel views"""
        try:
            response = self.youtube_analytics.reports().query(
                ids=f'channel=={channel_id}',
                startDate=start_date,
                endDate=end_date,
                metrics='views',
                dimensions='insightTrafficSourceDetail',
                filters='insightTrafficSourceType==YT_SEARCH',
                sort='-views',
                maxResults=max_results
            ).execute()
            
            headers = [h['name'] for h in response['columnHeaders']]
            rows = response.get('rows', [])
            df = pd.DataFrame(rows, columns=headers)
            
            df.rename(columns={'insightTrafficSourceDetail': 'search_term'}, inplace=True)
            
            return df
            
        except HttpError as e:
            logger.error(f"Error fetching search terms: {e}")
            raise
    
    def get_realtime_metrics(
        self,
        channel_id: str
    ) -> Dict[str, Any]:
        """Get real-time metrics (last 48 hours)"""
        end_date = datetime.utcnow().strftime('%Y-%m-%d')
        start_date = (datetime.utcnow() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        try:
            response = self.youtube_analytics.reports().query(
                ids=f'channel=={channel_id}',
                startDate=start_date,
                endDate=end_date,
                metrics='views,estimatedMinutesWatched,subscribersGained',
                dimensions='hour'
            ).execute()
            
            return {
                'period': '48_hours',
                'data': response.get('rows', []),
                'totals': {
                    'views': sum(r[1] for r in response.get('rows', [])),
                    'watch_time_minutes': sum(r[2] for r in response.get('rows', [])),
                    'subscribers_gained': sum(r[3] for r in response.get('rows', []))
                }
            }
            
        except HttpError as e:
            logger.error(f"Error fetching realtime metrics: {e}")
            raise
    
    def get_playlist_analytics(
        self,
        playlist_id: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Get analytics for a playlist"""
        try:
            response = self.youtube_analytics.reports().query(
                ids='channel==MINE',
                filters=f'playlist=={playlist_id}',
                startDate=start_date,
                endDate=end_date,
                metrics='views,estimatedMinutesWatched,playlistStarts,viewsPerPlaylistStart',
                dimensions='day',
                sort='day'
            ).execute()
            
            headers = [h['name'] for h in response['columnHeaders']]
            rows = response.get('rows', [])
            df = pd.DataFrame(rows, columns=headers)
            
            df['playlist_id'] = playlist_id
            
            return df
            
        except HttpError as e:
            logger.error(f"Error fetching playlist analytics: {e}")
            raise