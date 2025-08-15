"""
Mock Video Generator for Testing without FFmpeg
Creates placeholder video files for testing the pipeline
"""
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class MockVideoGenerator:
    """Creates mock video files for testing when FFmpeg is not available"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "ytempire_mock_videos"
        self.temp_dir.mkdir(exist_ok=True)
        
    async def create_mock_video(
        self,
        audio_path: str,
        visuals: List[Dict],
        output_path: str,
        title: str = "",
        subtitles: str = ""
    ) -> Dict[str, Any]:
        """Create a mock video file for testing"""
        try:
            # Create a mock MP4 file (just a text file with metadata)
            mock_video_data = {
                "type": "mock_video",
                "title": title,
                "audio_source": audio_path,
                "visuals_count": len(visuals),
                "subtitles": subtitles[:100] if subtitles else None,
                "created_at": datetime.utcnow().isoformat(),
                "duration": 120,  # Mock 2 minute video
                "resolution": "1920x1080",
                "fps": 30,
                "codec": "h264",
                "format": "mp4"
            }
            
            # Write mock data to output path
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a small binary file to simulate video
            with open(output_file, 'wb') as f:
                # Write a simple MP4 header-like structure
                f.write(b'ftypmp42')  # File type box
                f.write(b'\x00' * 1024)  # Some padding to make it look like a video file
                # Add the metadata as JSON at the end
                metadata = json.dumps(mock_video_data).encode('utf-8')
                f.write(b'MOCK_VIDEO_METADATA:')
                f.write(metadata)
            
            file_size = os.path.getsize(output_file)
            
            logger.info(f"Created mock video at {output_file} ({file_size} bytes)")
            
            return {
                "video_path": str(output_file),
                "duration": mock_video_data["duration"],
                "file_size": file_size / (1024 * 1024),  # Convert to MB
                "resolution": mock_video_data["resolution"],
                "fps": mock_video_data["fps"],
                "processing_time": 0.5,  # Mock processing time
                "visuals_count": len(visuals),
                "is_mock": True,
                "mock_metadata": mock_video_data
            }
            
        except Exception as e:
            logger.error(f"Failed to create mock video: {e}")
            raise
            
    async def create_mock_audio(self, text: str, output_path: str) -> Dict[str, Any]:
        """Create a mock audio file for testing"""
        try:
            # Create mock MP3 file
            mock_audio_data = {
                "type": "mock_audio",
                "text": text[:500],  # First 500 chars
                "duration": len(text) / 15,  # Estimate 15 chars per second
                "format": "mp3",
                "bitrate": "192k",
                "sample_rate": 48000
            }
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a small binary file to simulate audio
            with open(output_file, 'wb') as f:
                # Write MP3 header-like bytes
                f.write(b'ID3')  # ID3 tag
                f.write(b'\x00' * 512)  # Some padding
                # Add metadata
                metadata = json.dumps(mock_audio_data).encode('utf-8')
                f.write(b'MOCK_AUDIO_METADATA:')
                f.write(metadata)
            
            file_size = os.path.getsize(output_file)
            
            logger.info(f"Created mock audio at {output_file} ({file_size} bytes)")
            
            return {
                "audio_path": str(output_file),
                "duration": mock_audio_data["duration"],
                "characters": len(text),
                "cost": 0.001,  # Mock cost
                "format": mock_audio_data["format"],
                "is_mock": True
            }
            
        except Exception as e:
            logger.error(f"Failed to create mock audio: {e}")
            raise
            
    async def create_mock_thumbnail(self, title: str, output_path: str) -> Dict[str, Any]:
        """Create a mock thumbnail for testing"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a simple image
            img = Image.new('RGB', (1280, 720), color='blue')
            draw = ImageDraw.Draw(img)
            
            # Add title text
            try:
                font = ImageFont.truetype("arial.ttf", 60)
            except:
                font = ImageFont.load_default()
            
            # Center the text
            text = title[:50]  # Limit title length
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (1280 - text_width) // 2
            y = (720 - text_height) // 2
            
            draw.text((x, y), text, font=font, fill='white')
            
            # Add "MOCK" watermark
            draw.text((10, 10), "MOCK THUMBNAIL", font=font, fill='yellow')
            
            # Save image
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_file, 'JPEG', quality=95)
            
            file_size = os.path.getsize(output_file)
            
            logger.info(f"Created mock thumbnail at {output_file} ({file_size} bytes)")
            
            return {
                "thumbnail_path": str(output_file),
                "resolution": "1280x720",
                "file_size": file_size,
                "format": "jpeg",
                "is_mock": True
            }
            
        except Exception as e:
            logger.error(f"Failed to create mock thumbnail: {e}")
            # Create a placeholder file even if PIL fails
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'wb') as f:
                f.write(b'MOCK_THUMBNAIL')
            return {
                "thumbnail_path": str(output_file),
                "resolution": "1280x720",
                "file_size": 14,
                "format": "jpeg",
                "is_mock": True,
                "error": str(e)
            }

# Global instance
mock_generator = MockVideoGenerator()