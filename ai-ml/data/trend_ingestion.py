"""
Trend Data Ingestion Pipeline
Owner: ML Engineer

YouTube trending data collection and preprocessing for trend prediction models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time
import json

from app.core.config import settings
from app.core.database import get_db
from app.models.analytics import TrendData, KeywordTrend

logger = logging.getLogger(__name__)


class TrendDataIngestion:
    """Data ingestion pipeline for YouTube trends and keyword data."""
    
    def __init__(self, api_keys: List[str] = None):
        self.api_keys = api_keys or settings.YOUTUBE_API_KEYS.split(',')
        self.current_key_index = 0
        self.youtube_service = None
        self.rate_limits = {
            'youtube_api': {'requests': 0, 'reset_time': datetime.now()},
            'trends_api': {'requests': 0, 'reset_time': datetime.now()}
        }
        
    def get_youtube_service(self):
        """Get YouTube API service with key rotation."""
        try:
            if not self.youtube_service or self._should_rotate_key():
                current_key = self.api_keys[self.current_key_index]
                self.youtube_service = build('youtube', 'v3', developerKey=current_key)
                logger.info(f"Using YouTube API key index: {self.current_key_index}")
            
            return self.youtube_service
            
        except Exception as e:
            logger.error(f"Failed to create YouTube service: {str(e)}")
            self._rotate_api_key()
            raise
    
    def _should_rotate_key(self) -> bool:
        """Check if API key should be rotated due to quota limits."""
        return self.rate_limits['youtube_api']['requests'] > 9000  # Conservative limit
    
    def _rotate_api_key(self):
        """Rotate to next API key."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.youtube_service = None
        self.rate_limits['youtube_api'] = {'requests': 0, 'reset_time': datetime.now()}
        logger.info(f"Rotated to API key index: {self.current_key_index}")
    
    async def collect_trending_videos(self, region_code: str = 'US', category_id: str = '0') -> List[Dict]:
        """Collect trending videos from YouTube API."""
        try:
            youtube = self.get_youtube_service()
            
            request = youtube.videos().list(
                part='snippet,statistics,contentDetails',
                chart='mostPopular',
                regionCode=region_code,
                videoCategoryId=category_id,
                maxResults=50
            )
            
            response = request.execute()
            self.rate_limits['youtube_api']['requests'] += 1
            
            trending_videos = []
            for video in response['items']:
                video_data = {
                    'video_id': video['id'],
                    'title': video['snippet']['title'],
                    'description': video['snippet']['description'],
                    'channel_id': video['snippet']['channelId'],
                    'channel_title': video['snippet']['channelTitle'],
                    'published_at': video['snippet']['publishedAt'],
                    'category_id': video['snippet']['categoryId'],
                    'tags': video['snippet'].get('tags', []),
                    'view_count': int(video['statistics'].get('viewCount', 0)),
                    'like_count': int(video['statistics'].get('likeCount', 0)),
                    'comment_count': int(video['statistics'].get('commentCount', 0)),
                    'duration': video['contentDetails']['duration'],
                    'collected_at': datetime.now().isoformat(),
                    'region_code': region_code,
                    'category_id': category_id
                }
                trending_videos.append(video_data)
            
            logger.info(f"Collected {len(trending_videos)} trending videos for {region_code}")
            return trending_videos
            
        except HttpError as e:
            if e.resp.status == 403:
                logger.warning("YouTube API quota exceeded, rotating key")
                self._rotate_api_key()
                return await self.collect_trending_videos(region_code, category_id)
            else:
                logger.error(f"YouTube API error: {str(e)}")
                raise
    
    async def collect_search_trends(self, keywords: List[str], timeframe: str = 'today 12-m') -> Dict[str, List]:
        """Collect search trend data for keywords."""
        try:
            from pytrends.request import TrendReq
            
            # Initialize pytrends
            pytrends = TrendReq(hl='en-US', tz=360)
            
            trends_data = {}
            
            # Process keywords in batches of 5 (pytrends limit)
            for i in range(0, len(keywords), 5):
                batch = keywords[i:i+5]
                
                try:
                    # Build payload
                    pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo='US', gprop='youtube')
                    
                    # Get interest over time
                    interest_over_time = pytrends.interest_over_time()
                    
                    if not interest_over_time.empty:
                        for keyword in batch:
                            if keyword in interest_over_time.columns:
                                trends_data[keyword] = interest_over_time[keyword].tolist()
                    
                    # Rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Failed to get trends for batch {batch}: {str(e)}")
                    continue
            
            logger.info(f"Collected trend data for {len(trends_data)} keywords")
            return trends_data
            
        except Exception as e:
            logger.error(f"Search trends collection failed: {str(e)}")
            return {}
    
    async def extract_trending_keywords(self, video_data: List[Dict]) -> List[str]:
        """Extract trending keywords from video titles and tags."""
        try:
            from collections import Counter
            import re
            
            # Common stop words to filter out
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
                'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
            }
            
            all_words = []
            
            # Extract words from titles
            for video in video_data:
                title = video.get('title', '').lower()
                tags = video.get('tags', [])
                
                # Clean and extract words from title
                title_words = re.findall(r'\b[a-z]+\b', title)
                title_words = [word for word in title_words if word not in stop_words and len(word) > 2]
                all_words.extend(title_words)
                
                # Add tags
                tag_words = [tag.lower() for tag in tags if len(tag) > 2]
                all_words.extend(tag_words)
            
            # Count word frequency
            word_counts = Counter(all_words)
            
            # Get top trending keywords
            trending_keywords = [word for word, count in word_counts.most_common(100) if count > 2]
            
            logger.info(f"Extracted {len(trending_keywords)} trending keywords")
            return trending_keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return []
    
    async def enrich_video_data(self, videos: List[Dict]) -> List[Dict]:
        """Enrich video data with additional metrics and features."""
        try:
            enriched_videos = []
            
            for video in videos:
                enriched = video.copy()
                
                # Calculate engagement rate
                views = video.get('view_count', 0)
                likes = video.get('like_count', 0)
                comments = video.get('comment_count', 0)
                
                if views > 0:
                    engagement_rate = (likes + comments) / views * 100
                    enriched['engagement_rate'] = round(engagement_rate, 4)
                else:
                    enriched['engagement_rate'] = 0
                
                # Parse duration to seconds
                duration_str = video.get('duration', 'PT0S')
                enriched['duration_seconds'] = self._parse_duration(duration_str)
                
                # Calculate views per hour since publish
                published_at = datetime.fromisoformat(video['published_at'].replace('Z', '+00:00'))
                hours_since_publish = (datetime.now(published_at.tzinfo) - published_at).total_seconds() / 3600
                
                if hours_since_publish > 0:
                    enriched['views_per_hour'] = views / hours_since_publish
                else:
                    enriched['views_per_hour'] = 0
                
                # Extract title features
                title = video.get('title', '')
                enriched['title_length'] = len(title)
                enriched['title_word_count'] = len(title.split())
                enriched['has_caps'] = int(any(word.isupper() for word in title.split()))
                enriched['has_numbers'] = int(any(char.isdigit() for char in title))
                enriched['question_marks'] = title.count('?')
                enriched['exclamation_marks'] = title.count('!')
                
                enriched_videos.append(enriched)
            
            logger.info(f"Enriched {len(enriched_videos)} videos with additional features")
            return enriched_videos
            
        except Exception as e:
            logger.error(f"Video enrichment failed: {str(e)}")
            return videos
    
    def _parse_duration(self, duration_str: str) -> int:
        """Parse YouTube duration string to seconds."""
        try:
            import re
            
            # Parse PT format (PT1H2M3S)
            pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
            match = re.match(pattern, duration_str)
            
            if not match:
                return 0
            
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            
            return hours * 3600 + minutes * 60 + seconds
            
        except Exception:
            return 0
    
    async def store_trend_data(self, trend_data: Dict, data_type: str = 'video'):
        """Store collected trend data in database."""
        try:
            db = next(get_db())
            
            if data_type == 'video':
                for video in trend_data.get('videos', []):
                    trend_record = TrendData(
                        data_type='video',
                        source='youtube_api',
                        data_point=video,
                        collected_at=datetime.fromisoformat(video['collected_at'])
                    )
                    db.add(trend_record)
            
            elif data_type == 'keyword':
                for keyword, values in trend_data.items():
                    for i, value in enumerate(values):
                        keyword_trend = KeywordTrend(
                            keyword=keyword,
                            search_volume=value,
                            timestamp=datetime.now() - timedelta(days=len(values)-i),
                            source='google_trends'
                        )
                        db.add(keyword_trend)
            
            db.commit()
            logger.info(f"Stored {data_type} trend data in database")
            
        except Exception as e:
            logger.error(f"Failed to store trend data: {str(e)}")
            if 'db' in locals():
                db.rollback()
            raise
    
    async def run_full_ingestion_pipeline(self, regions: List[str] = ['US'], categories: List[str] = ['0']):
        """Run complete trend data ingestion pipeline."""
        try:
            logger.info("Starting full trend data ingestion pipeline")
            
            all_videos = []
            
            # Collect trending videos for all regions and categories
            for region in regions:
                for category in categories:
                    videos = await self.collect_trending_videos(region, category)
                    all_videos.extend(videos)
                    
                    # Rate limiting
                    await asyncio.sleep(1)
            
            # Enrich video data
            enriched_videos = await self.enrich_video_data(all_videos)
            
            # Extract trending keywords
            keywords = await self.extract_trending_keywords(enriched_videos)
            
            # Collect keyword trends
            keyword_trends = await self.collect_search_trends(keywords[:50])  # Limit to top 50
            
            # Store all data
            await self.store_trend_data({'videos': enriched_videos}, 'video')
            
            if keyword_trends:
                await self.store_trend_data(keyword_trends, 'keyword')
            
            logger.info(f"Ingestion pipeline completed: {len(enriched_videos)} videos, {len(keyword_trends)} keywords")
            
            return {
                'videos_collected': len(enriched_videos),
                'keywords_collected': len(keyword_trends),
                'regions': regions,
                'categories': categories,
                'completed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {str(e)}")
            raise


# Utility functions for data preprocessing
def preprocess_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess trend data for model training."""
    try:
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Normalize engagement metrics
        if 'engagement_rate' in df.columns:
            df['engagement_rate'] = df['engagement_rate'].clip(0, 100)
        
        # Create time features
        if 'published_at' in df.columns:
            df['published_at'] = pd.to_datetime(df['published_at'])
            df['hour_of_day'] = df['published_at'].dt.hour
            df['day_of_week'] = df['published_at'].dt.dayofweek
            df['month'] = df['published_at'].dt.month
        
        return df
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        return df


# Example usage
async def main():
    """Example usage of trend ingestion pipeline."""
    ingestion = TrendDataIngestion()
    
    # Run ingestion for multiple regions
    result = await ingestion.run_full_ingestion_pipeline(
        regions=['US', 'GB', 'CA'],
        categories=['0', '24', '25']  # All, Entertainment, News & Politics
    )
    
    print(f"Ingestion completed: {result}")


if __name__ == "__main__":
    asyncio.run(main())