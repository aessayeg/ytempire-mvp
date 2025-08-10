"""
Comprehensive tests for YouTube service
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.services.youtube_service import (
    YouTubeService, QuotaTracker, YouTubeError,
    VideoUploadError, ChannelNotFoundError
)


class TestYouTubeService:
    """Test YouTube service functionality"""
    
    @pytest.fixture
    def youtube_service(self):
        """Create YouTube service instance"""
        with patch('backend.app.services.youtube_service.build'):
            service = YouTubeService()
            service.youtube = Mock()
            service.analytics = Mock()
            return service
    
    @pytest.fixture
    def quota_tracker(self):
        """Create quota tracker instance"""
        return QuotaTracker()
    
    @pytest.mark.asyncio
    async def test_upload_video_success(self, youtube_service):
        """Test successful video upload"""
        # Mock YouTube API responses
        youtube_service.youtube.videos().insert().next_chunk.return_value = (None, {
            'id': 'test_video_id',
            'snippet': {'title': 'Test Video'}
        })
        
        result = await youtube_service.upload_video(
            video_file_path='test.mp4',
            title='Test Video',
            description='Test Description',
            tags=['test', 'video']
        )
        
        assert result['id'] == 'test_video_id'
        assert result['snippet']['title'] == 'Test Video'
    
    @pytest.mark.asyncio
    async def test_upload_video_quota_exceeded(self, youtube_service):
        """Test video upload with quota exceeded"""
        youtube_service.quota_tracker.get_remaining_quota = Mock(return_value=0)
        
        with pytest.raises(YouTubeError) as exc_info:
            await youtube_service.upload_video(
                video_file_path='test.mp4',
                title='Test Video',
                description='Test Description',
                tags=['test']
            )
        
        assert 'quota exceeded' in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_search_videos(self, youtube_service):
        """Test video search functionality"""
        mock_response = {
            'items': [
                {
                    'id': {'videoId': 'video1'},
                    'snippet': {'title': 'Video 1'}
                },
                {
                    'id': {'videoId': 'video2'},
                    'snippet': {'title': 'Video 2'}
                }
            ],
            'pageInfo': {'totalResults': 2}
        }
        
        youtube_service.youtube.search().list().execute.return_value = mock_response
        
        results = await youtube_service.search_videos(
            query='test query',
            max_results=10
        )
        
        assert len(results['items']) == 2
        assert results['items'][0]['id']['videoId'] == 'video1'
    
    @pytest.mark.asyncio
    async def test_get_channel_details(self, youtube_service):
        """Test getting channel details"""
        mock_response = {
            'items': [{
                'id': 'channel123',
                'snippet': {'title': 'Test Channel'},
                'statistics': {
                    'subscriberCount': '1000',
                    'viewCount': '50000'
                }
            }]
        }
        
        youtube_service.youtube.channels().list().execute.return_value = mock_response
        
        result = await youtube_service.get_channel_details('channel123')
        
        assert result['id'] == 'channel123'
        assert result['snippet']['title'] == 'Test Channel'
        assert result['statistics']['subscriberCount'] == '1000'
    
    @pytest.mark.asyncio
    async def test_get_channel_details_not_found(self, youtube_service):
        """Test getting channel details when channel not found"""
        youtube_service.youtube.channels().list().execute.return_value = {'items': []}
        
        with pytest.raises(ChannelNotFoundError):
            await youtube_service.get_channel_details('nonexistent')
    
    @pytest.mark.asyncio
    async def test_get_video_analytics(self, youtube_service):
        """Test getting video analytics"""
        mock_response = {
            'rows': [
                ['video1', '1000', '50', '10'],
                ['video2', '2000', '100', '20']
            ],
            'columnHeaders': [
                {'name': 'video'},
                {'name': 'views'},
                {'name': 'likes'},
                {'name': 'comments'}
            ]
        }
        
        youtube_service.analytics.reports().query().execute.return_value = mock_response
        
        result = await youtube_service.get_video_analytics(
            video_ids=['video1', 'video2'],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
        
        assert len(result['rows']) == 2
        assert result['rows'][0][1] == '1000'  # views for video1
    
    @pytest.mark.asyncio
    async def test_update_video_metadata(self, youtube_service):
        """Test updating video metadata"""
        mock_response = {
            'id': 'video123',
            'snippet': {
                'title': 'Updated Title',
                'description': 'Updated Description'
            }
        }
        
        youtube_service.youtube.videos().update().execute.return_value = mock_response
        
        result = await youtube_service.update_video(
            video_id='video123',
            title='Updated Title',
            description='Updated Description'
        )
        
        assert result['snippet']['title'] == 'Updated Title'
        assert result['snippet']['description'] == 'Updated Description'
    
    @pytest.mark.asyncio
    async def test_delete_video(self, youtube_service):
        """Test deleting a video"""
        youtube_service.youtube.videos().delete().execute.return_value = {}
        
        result = await youtube_service.delete_video('video123')
        
        assert result is True
        youtube_service.youtube.videos().delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_comments(self, youtube_service):
        """Test getting video comments"""
        mock_response = {
            'items': [
                {
                    'id': 'comment1',
                    'snippet': {
                        'topLevelComment': {
                            'snippet': {
                                'textDisplay': 'Great video!',
                                'authorDisplayName': 'User1'
                            }
                        }
                    }
                }
            ]
        }
        
        youtube_service.youtube.commentThreads().list().execute.return_value = mock_response
        
        comments = await youtube_service.get_video_comments('video123')
        
        assert len(comments['items']) == 1
        assert comments['items'][0]['snippet']['topLevelComment']['snippet']['textDisplay'] == 'Great video!'
    
    def test_quota_tracker_initialization(self, quota_tracker):
        """Test quota tracker initialization"""
        assert quota_tracker.daily_limit == 10000
        assert quota_tracker.get_remaining_quota() == 10000
    
    def test_quota_tracker_use_quota(self, quota_tracker):
        """Test using quota"""
        initial_quota = quota_tracker.get_remaining_quota()
        quota_tracker.use_quota(100)
        
        assert quota_tracker.get_remaining_quota() == initial_quota - 100
    
    def test_quota_tracker_reset(self, quota_tracker):
        """Test quota reset after 24 hours"""
        quota_tracker.use_quota(5000)
        assert quota_tracker.get_remaining_quota() == 5000
        
        # Simulate time passing
        quota_tracker.last_reset = datetime.now() - timedelta(hours=25)
        remaining = quota_tracker.get_remaining_quota()
        
        assert remaining == 10000  # Reset to full quota
    
    @pytest.mark.asyncio
    async def test_batch_upload_videos(self, youtube_service):
        """Test batch video upload"""
        videos = [
            {'path': 'video1.mp4', 'title': 'Video 1'},
            {'path': 'video2.mp4', 'title': 'Video 2'}
        ]
        
        youtube_service.upload_video = AsyncMock(return_value={'id': 'test_id'})
        
        results = await youtube_service.batch_upload_videos(videos)
        
        assert len(results) == 2
        assert youtube_service.upload_video.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_channel_videos(self, youtube_service):
        """Test getting all videos from a channel"""
        mock_response = {
            'items': [
                {'id': {'videoId': 'video1'}},
                {'id': {'videoId': 'video2'}}
            ],
            'nextPageToken': None
        }
        
        youtube_service.youtube.search().list().execute.return_value = mock_response
        
        videos = await youtube_service.get_channel_videos('channel123')
        
        assert len(videos['items']) == 2
    
    @pytest.mark.asyncio
    async def test_create_playlist(self, youtube_service):
        """Test creating a playlist"""
        mock_response = {
            'id': 'playlist123',
            'snippet': {'title': 'Test Playlist'}
        }
        
        youtube_service.youtube.playlists().insert().execute.return_value = mock_response
        
        result = await youtube_service.create_playlist(
            title='Test Playlist',
            description='Test Description'
        )
        
        assert result['id'] == 'playlist123'
    
    @pytest.mark.asyncio
    async def test_add_video_to_playlist(self, youtube_service):
        """Test adding video to playlist"""
        mock_response = {'id': 'playlistItem123'}
        
        youtube_service.youtube.playlistItems().insert().execute.return_value = mock_response
        
        result = await youtube_service.add_video_to_playlist(
            playlist_id='playlist123',
            video_id='video123'
        )
        
        assert result['id'] == 'playlistItem123'
    
    @pytest.mark.asyncio
    async def test_error_handling(self, youtube_service):
        """Test error handling in YouTube service"""
        youtube_service.youtube.videos().list().execute.side_effect = Exception("API Error")
        
        with pytest.raises(YouTubeError):
            await youtube_service.get_video_details('video123')
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, youtube_service):
        """Test retry mechanism for failed requests"""
        youtube_service.youtube.videos().list().execute.side_effect = [
            Exception("Temporary error"),
            Exception("Temporary error"),
            {'items': [{'id': 'video123'}]}
        ]
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await youtube_service.get_video_details_with_retry('video123')
        
        assert result['items'][0]['id'] == 'video123'
    
    @pytest.mark.asyncio
    async def test_thumbnail_upload(self, youtube_service):
        """Test thumbnail upload"""
        youtube_service.youtube.thumbnails().set().execute.return_value = {
            'items': [{'default': {'url': 'thumbnail_url'}}]
        }
        
        result = await youtube_service.upload_thumbnail(
            video_id='video123',
            thumbnail_path='thumbnail.jpg'
        )
        
        assert 'items' in result
    
    @pytest.mark.asyncio
    async def test_get_subscriptions(self, youtube_service):
        """Test getting channel subscriptions"""
        mock_response = {
            'items': [
                {'snippet': {'resourceId': {'channelId': 'channel1'}}},
                {'snippet': {'resourceId': {'channelId': 'channel2'}}}
            ]
        }
        
        youtube_service.youtube.subscriptions().list().execute.return_value = mock_response
        
        subs = await youtube_service.get_subscriptions()
        
        assert len(subs['items']) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])