"""
YouTube Tasks for Celery
Handles YouTube uploads, channel management, and video publishing
"""

import logging
import asyncio
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from celery import Task

from app.core.celery_app import celery_app, TaskPriority
from app.services.youtube_service import YouTubeService
from app.services.youtube_multi_account import get_youtube_manager
from app.services.notification_service import notification_service
from app.db.session import AsyncSessionLocal
from app.models.video import Video, VideoStatus
from app.models.channel import Channel
from sqlalchemy import select, and_

logger = logging.getLogger(__name__)


class YouTubeTask(Task):
    """Base class for YouTube tasks"""

    _youtube_service = None
    _youtube_manager = None

    @property
    def youtube_service(self):
        if self._youtube_service is None:
            self._youtube_service = YouTubeService()
        return self._youtube_service

    @property
    def youtube_manager(self):
        if self._youtube_manager is None:
            self._youtube_manager = get_youtube_manager()
        return self._youtube_manager


@celery_app.task(
    bind=True,
    base=YouTubeTask,
    name="youtube.upload_video",
    queue="youtube_upload",
    max_retries=3,
    default_retry_delay=120,
)
def upload_video_to_youtube(
    self,
    video_id: str,
    video_path: str,
    title: str,
    description: str,
    tags: List[str],
    channel_id: str,
    thumbnail_path: Optional[str] = None,
    scheduled_time: Optional[str] = None,
    privacy: str = "private",
) -> Dict[str, Any]:
    """
    Upload video to YouTube

    Args:
        video_id: Internal video ID
        video_path: Path to video file
        title: Video title
        description: Video description
        tags: Video tags
        channel_id: Channel ID to upload to
        thumbnail_path: Path to thumbnail image
        scheduled_time: Time to schedule video publication
        privacy: Privacy setting (private, unlisted, public)
    """
    try:
        logger.info(f"Uploading video {video_id} to YouTube")

        async def upload():
            async with AsyncSessionLocal() as db:
                # Get channel credentials
                channel = await db.get(Channel, channel_id)
                if not channel:
                    raise ValueError(f"Channel {channel_id} not found")

                # Update video status
                video = await db.get(Video, video_id)
                if video:
                    video.status = VideoStatus.UPLOADING
                    await db.commit()

                # Use multi-account manager for upload
                youtube_account = (
                    await self.youtube_manager.get_best_account_for_upload(
                        channel.youtube_channel_id
                    )
                )

                # Upload video
                upload_result = await self.youtube_service.upload_video(
                    video_path=video_path,
                    title=title,
                    description=description,
                    tags=tags,
                    category_id="22",  # People & Blogs
                    privacy_status=privacy,
                    thumbnail_path=thumbnail_path,
                    scheduled_publish_time=scheduled_time,
                    channel_credentials={
                        "api_key": youtube_account["api_key"],
                        "refresh_token": youtube_account["refresh_token"],
                        "channel_id": channel.youtube_channel_id,
                    },
                )

                # Update video with YouTube ID
                if video and upload_result.get("video_id"):
                    video.youtube_video_id = upload_result["video_id"]
                    video.youtube_url = (
                        f"https://youtube.com/watch?v={upload_result['video_id']}"
                    )
                    video.status = (
                        VideoStatus.PUBLISHED
                        if privacy == "public"
                        else VideoStatus.SCHEDULED
                    )
                    video.published_at = (
                        datetime.utcnow() if privacy == "public" else None
                    )
                    await db.commit()

                # Update account quota
                await self.youtube_manager.update_account_quota(
                    youtube_account["account_id"], operation="upload"
                )

                return upload_result

        result = asyncio.run(upload())

        # Send notification
        asyncio.run(
            notification_service.send_notification(
                user_id=channel_id,  # Assuming channel_id relates to user
                title="Video Uploaded",
                message=f"Your video '{title}' has been uploaded to YouTube",
                type="success",
            )
        )

        return {
            "success": True,
            "youtube_video_id": result.get("video_id"),
            "youtube_url": result.get("video_url"),
            "upload_time": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"YouTube upload failed: {str(e)}")

        # Update video status to failed
        asyncio.run(update_video_status(video_id, VideoStatus.UPLOAD_FAILED))

        raise self.retry(exc=e)


@celery_app.task(
    bind=True, base=YouTubeTask, name="youtube.update_metadata", queue="youtube_upload"
)
def update_video_metadata(
    self, video_id: str, youtube_video_id: str, updates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update video metadata on YouTube

    Args:
        video_id: Internal video ID
        youtube_video_id: YouTube video ID
        updates: Dictionary of fields to update
    """
    try:
        logger.info(f"Updating metadata for video {youtube_video_id}")

        async def update():
            async with AsyncSessionLocal() as db:
                video = await db.get(Video, video_id)
                if not video:
                    raise ValueError(f"Video {video_id} not found")

                channel = await db.get(Channel, video.channel_id)

                # Update on YouTube
                result = await self.youtube_service.update_video(
                    video_id=youtube_video_id,
                    updates=updates,
                    api_key=channel.youtube_api_key,
                )

                # Update local database
                if updates.get("title"):
                    video.title = updates["title"]
                if updates.get("description"):
                    video.description = updates["description"]
                if updates.get("tags"):
                    video.tags = updates["tags"]

                await db.commit()
                return result

        result = asyncio.run(update())

        return {
            "success": True,
            "updated_fields": list(updates.keys()),
            "youtube_video_id": youtube_video_id,
        }

    except Exception as e:
        logger.error(f"Metadata update failed: {str(e)}")
        return {"success": False, "error": str(e)}


@celery_app.task(
    bind=True, base=YouTubeTask, name="youtube.delete_video", queue="youtube_upload"
)
def delete_video_from_youtube(
    self, video_id: str, youtube_video_id: str, channel_id: str
) -> Dict[str, Any]:
    """
    Delete video from YouTube

    Args:
        video_id: Internal video ID
        youtube_video_id: YouTube video ID
        channel_id: Channel ID
    """
    try:
        logger.info(f"Deleting video {youtube_video_id} from YouTube")

        async def delete():
            async with AsyncSessionLocal() as db:
                channel = await db.get(Channel, channel_id)

                # Delete from YouTube
                await self.youtube_service.delete_video(
                    video_id=youtube_video_id, api_key=channel.youtube_api_key
                )

                # Update video status
                video = await db.get(Video, video_id)
                if video:
                    video.status = VideoStatus.DELETED
                    video.youtube_video_id = None
                    video.youtube_url = None
                    await db.commit()

        asyncio.run(delete())

        return {
            "success": True,
            "deleted_video_id": youtube_video_id,
            "deleted_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Video deletion failed: {str(e)}")
        return {"success": False, "error": str(e)}


@celery_app.task(name="youtube.check_upload_status", queue="youtube_upload")
def check_upload_status(video_id: str) -> Dict[str, Any]:
    """
    Check the status of a video upload

    Args:
        video_id: Internal video ID
    """
    try:

        async def check():
            async with AsyncSessionLocal() as db:
                video = await db.get(Video, video_id)
                if not video:
                    return {"status": "not_found"}

                if video.youtube_video_id:
                    channel = await db.get(Channel, video.channel_id)
                    youtube_service = YouTubeService()

                    # Check YouTube status
                    yt_status = await youtube_service.get_video_status(
                        video.youtube_video_id, channel.youtube_api_key
                    )

                    return {
                        "status": video.status,
                        "youtube_status": yt_status,
                        "youtube_video_id": video.youtube_video_id,
                        "youtube_url": video.youtube_url,
                    }

                return {"status": video.status, "youtube_status": None}

        status_data = asyncio.run(check())

        return {"success": True, "video_id": video_id, **status_data}

    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return {"success": False, "error": str(e)}


@celery_app.task(name="youtube.sync_channel", queue="youtube_upload")
def sync_channel_videos(channel_id: str, max_videos: int = 50) -> Dict[str, Any]:
    """
    Sync videos from YouTube channel

    Args:
        channel_id: Internal channel ID
        max_videos: Maximum number of videos to sync
    """
    try:
        logger.info(f"Syncing videos for channel {channel_id}")

        async def sync():
            async with AsyncSessionLocal() as db:
                channel = await db.get(Channel, channel_id)
                if not channel:
                    raise ValueError(f"Channel {channel_id} not found")

                youtube_service = YouTubeService()

                # Get videos from YouTube
                videos = await youtube_service.get_channel_videos(
                    channel.youtube_channel_id,
                    channel.youtube_api_key,
                    max_results=max_videos,
                )

                synced_count = 0
                for yt_video in videos:
                    # Check if video exists in database
                    existing = await db.execute(
                        select(Video).where(Video.youtube_video_id == yt_video["id"])
                    )

                    if not existing.scalar():
                        # Create new video record
                        video = Video(
                            channel_id=channel_id,
                            title=yt_video["title"],
                            description=yt_video["description"],
                            youtube_video_id=yt_video["id"],
                            youtube_url=f"https://youtube.com/watch?v={yt_video['id']}",
                            status=VideoStatus.PUBLISHED,
                            published_at=yt_video.get("published_at"),
                        )
                        db.add(video)
                        synced_count += 1

                await db.commit()
                return synced_count, len(videos)

        synced, total = asyncio.run(sync())

        return {
            "success": True,
            "channel_id": channel_id,
            "videos_found": total,
            "videos_synced": synced,
            "sync_time": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Channel sync failed: {str(e)}")
        return {"success": False, "error": str(e)}


@celery_app.task(name="youtube.schedule_publish", queue="youtube_upload")
def schedule_video_publish(video_id: str, publish_time: str) -> Dict[str, Any]:
    """
    Schedule a video for publication

    Args:
        video_id: Internal video ID
        publish_time: ISO format datetime string
    """
    try:
        logger.info(f"Scheduling video {video_id} for {publish_time}")

        # Convert string to datetime
        publish_dt = datetime.fromisoformat(publish_time)

        # Schedule the upload task
        upload_video_to_youtube.apply_async(args=[video_id], eta=publish_dt)

        # Update video status
        asyncio.run(update_video_status(video_id, VideoStatus.SCHEDULED))

        return {"success": True, "video_id": video_id, "scheduled_for": publish_time}

    except Exception as e:
        logger.error(f"Scheduling failed: {str(e)}")
        return {"success": False, "error": str(e)}


async def update_video_status(video_id: str, status: VideoStatus):
    """Update video status in database"""
    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if video:
            video.status = status
            await db.commit()
