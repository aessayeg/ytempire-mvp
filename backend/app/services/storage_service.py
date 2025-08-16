"""
Storage Service
Handles file storage and retrieval operations
"""
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Union, BinaryIO
from datetime import datetime
import logging
import aiofiles
from PIL import Image
import io

logger = logging.getLogger(__name__)


class StorageService:
    """Service for handling file storage operations"""

    def __init__(self, base_path: str = "./uploads"):
        """Initialize storage service

        Args:
            base_path: Base directory for file storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.avatars_path = self.base_path / "avatars"
        self.videos_path = self.base_path / "videos"
        self.thumbnails_path = self.base_path / "thumbnails"
        self.temp_path = self.base_path / "temp"

        for path in [
            self.avatars_path,
            self.videos_path,
            self.thumbnails_path,
            self.temp_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    async def save_avatar(self, file: BinaryIO, user_id: str) -> str:
        """Save user avatar

        Args:
            file: File object to save
            user_id: User ID

        Returns:
            URL/path of saved avatar
        """
        try:
            # Generate unique filename
            file_extension = self._get_file_extension(
                file.filename if hasattr(file, "filename") else "avatar.jpg"
            )
            filename = f"{user_id}_{uuid.uuid4().hex}{file_extension}"
            file_path = self.avatars_path / filename

            # Save file
            contents = await file.read() if hasattr(file, "read") else file
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(contents)

            # Process image (resize if needed)
            await self._process_avatar(file_path)

            return f"/uploads/avatars/{filename}"

        except Exception as e:
            logger.error(f"Error saving avatar: {e}")
            raise

    async def save_video(self, file: BinaryIO, video_id: str) -> str:
        """Save video file

        Args:
            file: Video file object
            video_id: Video ID

        Returns:
            URL/path of saved video
        """
        try:
            file_extension = self._get_file_extension(
                file.filename if hasattr(file, "filename") else "video.mp4"
            )
            filename = f"{video_id}_{uuid.uuid4().hex}{file_extension}"
            file_path = self.videos_path / filename

            contents = await file.read() if hasattr(file, "read") else file
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(contents)

            return f"/uploads/videos/{filename}"

        except Exception as e:
            logger.error(f"Error saving video: {e}")
            raise

    async def save_thumbnail(self, file: BinaryIO, video_id: str) -> str:
        """Save video thumbnail

        Args:
            file: Thumbnail image file
            video_id: Video ID

        Returns:
            URL/path of saved thumbnail
        """
        try:
            file_extension = self._get_file_extension(
                file.filename if hasattr(file, "filename") else "thumb.jpg"
            )
            filename = f"{video_id}_thumb_{uuid.uuid4().hex}{file_extension}"
            file_path = self.thumbnails_path / filename

            contents = await file.read() if hasattr(file, "read") else file
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(contents)

            # Process thumbnail
            await self._process_thumbnail(file_path)

            return f"/uploads/thumbnails/{filename}"

        except Exception as e:
            logger.error(f"Error saving thumbnail: {e}")
            raise

    async def delete_file(self, file_path: str) -> bool:
        """Delete a file

        Args:
            file_path: Path of file to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            full_path = self.base_path / file_path.lstrip("/uploads/")
            if full_path.exists():
                full_path.unlink()
                return True
            return False

        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False

    async def get_file_url(self, file_path: str) -> Optional[str]:
        """Get URL for a stored file

        Args:
            file_path: Path of the file

        Returns:
            URL of the file or None if not found
        """
        full_path = self.base_path / file_path.lstrip("/uploads/")
        if full_path.exists():
            return file_path
        return None

    async def _process_avatar(self, file_path: Path, max_size: tuple = (256, 256)):
        """Process avatar image (resize if needed)

        Args:
            file_path: Path to avatar file
            max_size: Maximum dimensions (width, height)
        """
        try:
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize if larger than max_size
                img.thumbnail(max_size, Image.Resampling.LANCZOS)

                # Save optimized image
                img.save(file_path, "JPEG", quality=85, optimize=True)

        except Exception as e:
            logger.warning(f"Could not process avatar image: {e}")

    async def _process_thumbnail(self, file_path: Path, size: tuple = (1280, 720)):
        """Process thumbnail image

        Args:
            file_path: Path to thumbnail file
            size: Target dimensions (width, height)
        """
        try:
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize to exact dimensions
                img = img.resize(size, Image.Resampling.LANCZOS)

                # Save optimized image
                img.save(file_path, "JPEG", quality=90, optimize=True)

        except Exception as e:
            logger.warning(f"Could not process thumbnail image: {e}")

    def _get_file_extension(self, filename: str) -> str:
        """Get file extension from filename

        Args:
            filename: Original filename

        Returns:
            File extension with dot (e.g., '.jpg')
        """
        extension = Path(filename).suffix.lower()
        if not extension:
            extension = ".bin"
        return extension

    async def cleanup_temp_files(self, older_than_hours: int = 24):
        """Clean up temporary files older than specified hours

        Args:
            older_than_hours: Delete files older than this many hours
        """
        try:
            current_time = datetime.now()
            for file_path in self.temp_path.iterdir():
                if file_path.is_file():
                    file_age_hours = (
                        current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    ).total_seconds() / 3600
                    if file_age_hours > older_than_hours:
                        file_path.unlink()
                        logger.info(f"Deleted temp file: {file_path}")

        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")

    def get_storage_stats(self) -> dict:
        """Get storage statistics

        Returns:
            Dictionary with storage statistics
        """
        try:
            stats = {
                "total_files": 0,
                "total_size_bytes": 0,
                "avatars_count": 0,
                "videos_count": 0,
                "thumbnails_count": 0,
                "temp_files_count": 0,
            }

            for category, path in [
                ("avatars_count", self.avatars_path),
                ("videos_count", self.videos_path),
                ("thumbnails_count", self.thumbnails_path),
                ("temp_files_count", self.temp_path),
            ]:
                for file_path in path.iterdir():
                    if file_path.is_file():
                        stats[category] += 1
                        stats["total_files"] += 1
                        stats["total_size_bytes"] += file_path.stat().st_size

            # Convert to MB
            stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)

            return stats

        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}


# Global storage service instance
storage_service = StorageService()
