"""
YouTube Upload Tasks
Owner: Integration Specialist

YouTube API integration for video uploads with comprehensive error handling.
"""

from celery import current_task
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import os
from pathlib import Path

from app.core.celery_app import celery_app
from app.core.config import settings
from app.services.youtube_service import YouTubeService
from app.services.vault_service import VaultService
from app.repositories.video_repository import VideoRepository
from app.repositories.channel_repository import ChannelRepository
from app.core.database import AsyncSessionLocal
from app.core.metrics import metrics
from app.models.video import VideoStatus

logger = logging.getLogger(__name__)


class YouTubeUploadError(Exception):
    """Custom exception for YouTube upload errors."""
    pass


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 300})
def upload_to_youtube_task(self, compilation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload compiled video to YouTube.
    
    Args:
        compilation_data: Video compilation results from previous task
        
    Returns:
        YouTube upload results with video ID and metadata
    """
    try:
        video_id = compilation_data['id']
        logger.info(f"Starting YouTube upload for video: {video_id}")
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'youtube_upload', 'progress': 10, 'video_id': video_id}
        )
        
        # Validate upload requirements
        validate_upload_requirements(compilation_data)
        
        # Initialize services
        async with AsyncSessionLocal() as db:
            video_repo = VideoRepository(db)
            channel_repo = ChannelRepository(db)
            youtube_service = YouTubeService()
            
            # Get video and channel information
            video = await video_repo.get_by_id(video_id)
            if not video:
                raise YouTubeUploadError(f"Video not found: {video_id}")
            
            channel = await channel_repo.get_by_id(video.channel_id)
            if not channel:
                raise YouTubeUploadError(f"Channel not found: {video.channel_id}")
            
            # Update progress
            current_task.update_state(
                state='PROGRESS',
                meta={'stage': 'preparing_upload', 'progress': 20, 'video_id': video_id}
            )
            
            # Prepare upload data
            upload_data = prepare_youtube_upload_data(compilation_data, video, channel)
            
            # Update progress
            current_task.update_state(
                state='PROGRESS',
                meta={'stage': 'uploading_to_youtube', 'progress': 30, 'video_id': video_id}
            )
            
            # Perform YouTube upload
            youtube_result = await perform_youtube_upload(
                youtube_service, upload_data, video, channel
            )
            
            # Update progress
            current_task.update_state(
                state='PROGRESS',
                meta={'stage': 'processing_upload_result', 'progress': 80, 'video_id': video_id}
            )
            
            # Update video record with YouTube information
            await update_video_with_youtube_data(video_repo, video_id, youtube_result)
            
            # Update progress
            current_task.update_state(
                state='PROGRESS',
                meta={'stage': 'upload_complete', 'progress': 100, 'video_id': video_id}
            )
            
            # Prepare final result
            result = {
                **compilation_data,
                'youtube_video_id': youtube_result['youtube_video_id'],
                'youtube_url': youtube_result['youtube_url'],
                'youtube_status': youtube_result['status'],
                'upload_cost': youtube_result.get('cost', 0),
                'youtube_upload_completed': True,
                'upload_metadata': {
                    'uploaded_at': datetime.utcnow().isoformat(),
                    'processing_info': youtube_result.get('processing_info', {}),
                    'privacy_status': upload_data['privacy_status'],
                    'category_id': upload_data['category_id']
                }
            }
            
            # Record metrics
            metrics.record_video_uploaded(compilation_data['user_id'], video.channel_id)
            
            logger.info(f"YouTube upload completed for video: {video_id}, YouTube ID: {youtube_result['youtube_video_id']}")
            return result
        
    except Exception as e:
        logger.error(f"YouTube upload failed for video {compilation_data.get('id')}: {str(e)}")
        
        # Update video status to upload failed
        try:
            async with AsyncSessionLocal() as db:
                video_repo = VideoRepository(db)
                await video_repo.update_status(compilation_data['id'], VideoStatus.UPLOAD_FAILED)
        except:
            pass
        
        raise


def validate_upload_requirements(compilation_data: Dict[str, Any]) -> None:
    """Validate that all required data is available for YouTube upload."""
    
    required_fields = ['id', 'user_id', 'channel_id', 'video_file_path', 'title']
    for field in required_fields:
        if field not in compilation_data:
            raise YouTubeUploadError(f"Missing required field: {field}")
    
    # Check if video file exists
    video_file_path = compilation_data.get('video_file_path')
    if not os.path.exists(video_file_path):
        raise YouTubeUploadError(f"Video file not found: {video_file_path}")
    
    # Check file size (YouTube limit is 256GB, but we'll use a reasonable limit)
    file_size_mb = os.path.getsize(video_file_path) / (1024 * 1024)
    if file_size_mb > 2048:  # 2GB limit
        raise YouTubeUploadError(f"Video file too large: {file_size_mb:.2f}MB")
    
    # Check video duration from metadata
    duration = compilation_data.get('video_duration', 0)
    if duration > 43200:  # 12 hours limit
        raise YouTubeUploadError(f"Video too long: {duration}s")


def prepare_youtube_upload_data(
    compilation_data: Dict[str, Any], 
    video: Any, 
    channel: Any
) -> Dict[str, Any]:
    """Prepare data for YouTube upload."""
    
    # Get video metadata
    title = compilation_data.get('title', 'Untitled Video')
    description = compilation_data.get('description', '')
    tags = compilation_data.get('tags', [])
    
    # Ensure title is within YouTube limits (100 characters)
    if len(title) > 100:
        title = title[:97] + "..."
    
    # Ensure description is within YouTube limits (5000 characters)
    if len(description) > 5000:
        description = description[:4997] + "..."
    
    # Limit tags (YouTube allows up to 500 characters total)
    tag_string = ', '.join(tags)
    if len(tag_string) > 500:
        # Truncate tags to fit limit
        truncated_tags = []
        current_length = 0
        for tag in tags:
            if current_length + len(tag) + 2 <= 500:  # +2 for ", "
                truncated_tags.append(tag)
                current_length += len(tag) + 2
            else:
                break
        tags = truncated_tags
    
    # Generate enhanced description
    enhanced_description = generate_enhanced_description(description, compilation_data)
    
    upload_data = {
        'video_file_path': compilation_data['video_file_path'],
        'thumbnail_file_path': compilation_data.get('thumbnail_file_path'),
        'title': title,
        'description': enhanced_description,
        'tags': tags,
        'category_id': determine_category_id(compilation_data),
        'privacy_status': determine_privacy_status(channel),
        'language': 'en',
        'default_language': 'en',
        'recording_date': datetime.utcnow().isoformat(),
        'location': None,  # Can be added if available
        'license': 'youtube',  # Standard YouTube license
        'embeddable': True,
        'public_stats_viewable': True,
        'made_for_kids': False,
        'monetization': {
            'enabled': True,
            'ad_formats': ['overlay', 'skippable', 'non_skippable', 'bumper']
        }
    }
    
    return upload_data


async def perform_youtube_upload(
    youtube_service: YouTubeService,
    upload_data: Dict[str, Any],
    video: Any,
    channel: Any
) -> Dict[str, Any]:
    """Perform the actual YouTube upload."""
    
    try:
        # Initialize YouTube API credentials for the channel
        await youtube_service.initialize_credentials(channel.id, channel.oauth_tokens)
        
        # Upload video
        upload_result = await youtube_service.upload_video(
            video_file_path=upload_data['video_file_path'],
            title=upload_data['title'],
            description=upload_data['description'],
            tags=upload_data['tags'],
            category_id=upload_data['category_id'],
            privacy_status=upload_data['privacy_status'],
            thumbnail_file_path=upload_data.get('thumbnail_file_path')
        )
        
        if not upload_result or not upload_result.get('id'):
            raise YouTubeUploadError("YouTube upload failed - no video ID returned")
        
        youtube_video_id = upload_result['id']
        youtube_url = f"https://www.youtube.com/watch?v={youtube_video_id}"
        
        # Get upload status
        status_info = upload_result.get('status', {})
        
        # Calculate upload cost (minimal for YouTube API calls)
        upload_cost = 0.01  # Small cost for API usage
        
        result = {
            'youtube_video_id': youtube_video_id,
            'youtube_url': youtube_url,
            'status': status_info.get('uploadStatus', 'uploaded'),
            'privacy_status': status_info.get('privacyStatus', upload_data['privacy_status']),
            'processing_info': {
                'processing_status': status_info.get('processingStatus'),
                'processing_progress': status_info.get('processingProgress'),
                'processing_failure_reason': status_info.get('processingFailureReason'),
                'file_details_availability_status': status_info.get('fileDetailsAvailabilityStatus')
            },
            'cost': upload_cost,
            'upload_timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"YouTube upload successful: {youtube_video_id}")
        return result
        
    except Exception as e:
        logger.error(f"YouTube upload error: {str(e)}")
        raise YouTubeUploadError(f"Upload failed: {str(e)}")


async def update_video_with_youtube_data(
    video_repo: VideoRepository,
    video_id: str,
    youtube_result: Dict[str, Any]
) -> None:
    """Update video record with YouTube upload information."""
    
    try:
        update_data = {
            'youtube_video_id': youtube_result['youtube_video_id'],
            'youtube_url': youtube_result['youtube_url'],
            'status': VideoStatus.PUBLISHED,
            'published_at': datetime.utcnow(),
            'metadata': {
                'youtube_status': youtube_result['status'],
                'youtube_privacy_status': youtube_result['privacy_status'],
                'youtube_processing_info': youtube_result['processing_info'],
                'upload_cost': youtube_result['cost']
            }
        }
        
        await video_repo.update(video_id, update_data)
        
    except Exception as e:
        logger.error(f"Failed to update video with YouTube data: {str(e)}")
        # Don't raise - upload was successful, just database update failed


def generate_enhanced_description(base_description: str, compilation_data: Dict[str, Any]) -> str:
    """Generate enhanced description with timestamps and additional info."""
    
    # Start with base description
    enhanced = base_description
    
    # Add timestamps section
    duration = compilation_data.get('video_duration', 0)
    if duration > 60:  # Only add timestamps for videos longer than 1 minute
        enhanced += "\n\n📍 TIMESTAMPS:\n"
        enhanced += "00:00 Introduction\n"
        
        # Calculate middle timestamp
        middle_time = int(duration // 2)
        middle_minutes = middle_time // 60
        middle_seconds = middle_time % 60
        enhanced += f"{middle_minutes:02d}:{middle_seconds:02d} Main Content\n"
        
        # Calculate end timestamp
        end_time = int(duration - 30)  # 30 seconds before end
        if end_time > middle_time + 30:  # Only if there's enough gap
            end_minutes = end_time // 60
            end_seconds = end_time % 60
            enhanced += f"{end_minutes:02d}:{end_seconds:02d} Conclusion\n"
    
    # Add engagement call-to-action
    enhanced += "\n\n👍 If you enjoyed this video, please like and subscribe for more content!\n"
    enhanced += "💬 Let me know your thoughts in the comments below.\n"
    
    # Add hashtags
    tags = compilation_data.get('tags', [])
    if tags:
        hashtags = [f"#{tag.replace(' ', '')}" for tag in tags[:3]]  # Max 3 hashtags
        enhanced += f"\n🏷️ {' '.join(hashtags)}\n"
    
    # Add generation info (subtle)
    enhanced += "\n---\n"
    enhanced += "🤖 This video was created using AI-powered content generation.\n"
    enhanced += f"📅 Generated on {datetime.utcnow().strftime('%Y-%m-%d')}\n"
    
    return enhanced


def determine_category_id(compilation_data: Dict[str, Any]) -> str:
    """Determine appropriate YouTube category ID based on content."""
    
    # YouTube category IDs
    CATEGORIES = {
        'education': '27',
        'science_technology': '28',
        'entertainment': '24',
        'howto_style': '26',
        'news_politics': '25',
        'people_blogs': '22',
        'gaming': '20',
        'comedy': '23',
        'music': '10',
        'sports': '17'
    }
    
    # Analyze content for category hints
    content = (
        compilation_data.get('title', '') + ' ' +
        compilation_data.get('description', '') + ' ' +
        ' '.join(compilation_data.get('tags', []))
    ).lower()
    
    # Category keywords
    if any(word in content for word in ['tutorial', 'how to', 'guide', 'learn', 'education']):
        return CATEGORIES['education']
    elif any(word in content for word in ['technology', 'tech', 'ai', 'programming', 'science']):
        return CATEGORIES['science_technology']
    elif any(word in content for word in ['funny', 'comedy', 'humor', 'laugh', 'joke']):
        return CATEGORIES['comedy']
    elif any(word in content for word in ['game', 'gaming', 'play', 'player']):
        return CATEGORIES['gaming']
    elif any(word in content for word in ['music', 'song', 'audio', 'sound']):
        return CATEGORIES['music']
    elif any(word in content for word in ['news', 'politics', 'current', 'events']):
        return CATEGORIES['news_politics']
    elif any(word in content for word in ['style', 'fashion', 'beauty', 'lifestyle']):
        return CATEGORIES['howto_style']
    elif any(word in content for word in ['sport', 'fitness', 'exercise', 'workout']):
        return CATEGORIES['sports']
    elif any(word in content for word in ['entertainment', 'fun', 'interesting']):
        return CATEGORIES['entertainment']
    else:
        return CATEGORIES['people_blogs']  # Default category


def determine_privacy_status(channel: Any) -> str:
    """Determine privacy status based on channel settings."""
    
    # Check channel auto-publish settings
    if hasattr(channel, 'auto_publish') and channel.auto_publish:
        return 'public'
    
    # Check if channel has minimum subscriber requirement
    if hasattr(channel, 'min_subscribers_for_public'):
        # This would require checking current subscriber count
        # For now, default to unlisted for new channels
        return 'unlisted'
    
    # Default to unlisted for safety
    return 'unlisted'


@celery_app.task(bind=True)
def check_youtube_processing_status(self, youtube_video_id: str, video_id: str) -> Dict[str, Any]:
    """
    Check YouTube video processing status.
    This is a separate task that can be scheduled to run periodically.
    """
    try:
        async with AsyncSessionLocal() as db:
            video_repo = VideoRepository(db)
            channel_repo = ChannelRepository(db)
            youtube_service = YouTubeService()
            
            # Get video and channel
            video = await video_repo.get_by_id(video_id)
            if not video:
                raise YouTubeUploadError(f"Video not found: {video_id}")
            
            channel = await channel_repo.get_by_id(video.channel_id)
            if not channel:
                raise YouTubeUploadError(f"Channel not found: {video.channel_id}")
            
            # Initialize YouTube credentials
            await youtube_service.initialize_credentials(channel.id, channel.oauth_tokens)
            
            # Check processing status
            status_info = await youtube_service.get_video_status(youtube_video_id)
            
            if status_info:
                processing_status = status_info.get('processingStatus')
                
                # Update video metadata with current status
                await video_repo.update(video_id, {
                    'metadata': {
                        **(video.metadata or {}),
                        'youtube_processing_status': processing_status,
                        'last_status_check': datetime.utcnow().isoformat()
                    }
                })
                
                logger.info(f"Processing status for {youtube_video_id}: {processing_status}")
                return {
                    'youtube_video_id': youtube_video_id,
                    'video_id': video_id,
                    'processing_status': processing_status,
                    'status_info': status_info
                }
            
            return {'error': 'Failed to get status'}
            
    except Exception as e:
        logger.error(f"Failed to check YouTube processing status: {str(e)}")
        raise


@celery_app.task(bind=True)
def update_youtube_video_metadata(self, youtube_video_id: str, video_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update YouTube video metadata (title, description, tags, etc.)
    """
    try:
        async with AsyncSessionLocal() as db:
            video_repo = VideoRepository(db)
            channel_repo = ChannelRepository(db)
            youtube_service = YouTubeService()
            
            # Get video and channel
            video = await video_repo.get_by_id(video_id)
            if not video:
                raise YouTubeUploadError(f"Video not found: {video_id}")
            
            channel = await channel_repo.get_by_id(video.channel_id)
            if not channel:
                raise YouTubeUploadError(f"Channel not found: {video.channel_id}")
            
            # Initialize YouTube credentials
            await youtube_service.initialize_credentials(channel.id, channel.oauth_tokens)
            
            # Update video metadata on YouTube
            result = await youtube_service.update_video_metadata(youtube_video_id, updates)
            
            if result:
                logger.info(f"Updated YouTube video metadata for {youtube_video_id}")
                return {
                    'youtube_video_id': youtube_video_id,
                    'video_id': video_id,
                    'updates_applied': updates,
                    'success': True
                }
            else:
                return {'error': 'Failed to update metadata'}
                
    except Exception as e:
        logger.error(f"Failed to update YouTube video metadata: {str(e)}")
        raise


@celery_app.task(bind=True)
def schedule_video_publication(self, video_id: str, publish_time: datetime) -> Dict[str, Any]:
    """
    Schedule video publication at a specific time.
    """
    try:
        async with AsyncSessionLocal() as db:
            video_repo = VideoRepository(db)
            channel_repo = ChannelRepository(db)
            youtube_service = YouTubeService()
            
            video = await video_repo.get_by_id(video_id)
            if not video or not video.youtube_video_id:
                raise YouTubeUploadError(f"Video not ready for scheduling: {video_id}")
            
            channel = await channel_repo.get_by_id(video.channel_id)
            if not channel:
                raise YouTubeUploadError(f"Channel not found: {video.channel_id}")
            
            # Initialize YouTube credentials
            await youtube_service.initialize_credentials(channel.id, channel.oauth_tokens)
            
            # Schedule publication
            result = await youtube_service.schedule_video_publication(
                video.youtube_video_id, 
                publish_time
            )
            
            if result:
                # Update video record
                await video_repo.update(video_id, {
                    'scheduled_publish_at': publish_time,
                    'metadata': {
                        **(video.metadata or {}),
                        'publication_scheduled': True,
                        'scheduled_at': datetime.utcnow().isoformat()
                    }
                })
                
                logger.info(f"Scheduled video publication: {video_id} at {publish_time}")
                return {
                    'video_id': video_id,
                    'youtube_video_id': video.youtube_video_id,
                    'publish_time': publish_time.isoformat(),
                    'scheduled': True
                }
            else:
                return {'error': 'Failed to schedule publication'}
                
    except Exception as e:
        logger.error(f"Failed to schedule video publication: {str(e)}")
        raise