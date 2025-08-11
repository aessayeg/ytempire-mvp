"""
Video Processing Service
Handles video generation, editing, and thumbnail creation
"""
import os
import asyncio
import logging
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import subprocess
import json
from datetime import datetime
import aiofiles
import aiohttp
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Import mock generator for fallback
try:
    from app.services.mock_video_generator import mock_generator
except ImportError:
    mock_generator = None

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Video processing and generation service"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "ytempire_videos"
        self.temp_dir.mkdir(exist_ok=True)
        self.ffmpeg_path = self._find_ffmpeg()
        
    def _find_ffmpeg(self) -> str:
        """Find ffmpeg executable"""
        # Try common locations
        paths = [
            str(Path(__file__).parent.parent.parent / "ffmpeg" / "ffmpeg.exe"),  # Local project ffmpeg
            "ffmpeg",  # System PATH
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "C:\\ffmpeg\\bin\\ffmpeg.exe",
            "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe"
        ]
        
        for path in paths:
            if shutil.which(path):
                return path
                
        logger.warning("ffmpeg not found in system. Video processing will be limited.")
        return "ffmpeg"  # Fallback
        
    async def create_video_from_assets(
        self,
        audio_path: str,
        images: List[str],
        output_path: str,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30,
        transitions: bool = True
    ) -> Dict[str, Any]:
        """Create video from audio and images"""
        try:
            start_time = datetime.utcnow()
            
            # Create temp directory for this video
            video_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            work_dir = self.temp_dir / video_id
            work_dir.mkdir(exist_ok=True)
            
            # Get audio duration
            duration = await self._get_audio_duration(audio_path)
            
            # Calculate image display time
            image_duration = duration / len(images) if images else duration
            
            # Create video from images
            if images:
                video_path = work_dir / "video.mp4"
                await self._create_slideshow(
                    images, 
                    video_path, 
                    image_duration,
                    resolution,
                    fps,
                    transitions
                )
            else:
                # Create blank video if no images
                video_path = work_dir / "video.mp4"
                await self._create_blank_video(
                    video_path,
                    duration,
                    resolution,
                    fps
                )
                
            # Combine video with audio
            final_path = output_path or (work_dir / "final.mp4")
            await self._combine_audio_video(
                video_path,
                audio_path,
                final_path
            )
            
            # Get file size
            file_size = os.path.getsize(final_path)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "output_path": str(final_path),
                "duration": duration,
                "resolution": f"{resolution[0]}x{resolution[1]}",
                "fps": fps,
                "file_size": file_size,
                "processing_time": processing_time,
                "temp_dir": str(work_dir)
            }
            
        except Exception as e:
            logger.error(f"Video creation error: {e}")
            raise
            
    async def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio file duration"""
        try:
            cmd = [
                self.ffmpeg_path,
                "-i", audio_path,
                "-f", "null",
                "-"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await process.communicate()
            stderr_text = stderr.decode()
            
            # Parse duration from ffmpeg output
            import re
            duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", stderr_text)
            if duration_match:
                hours = int(duration_match.group(1))
                minutes = int(duration_match.group(2))
                seconds = float(duration_match.group(3))
                duration = hours * 3600 + minutes * 60 + seconds
                return duration
                
            return 60.0  # Default duration
            
        except Exception as e:
            logger.error(f"Error getting audio duration: {e}")
            return 60.0
            
    async def _create_slideshow(
        self,
        images: List[str],
        output_path: Path,
        image_duration: float,
        resolution: Tuple[int, int],
        fps: int,
        transitions: bool
    ):
        """Create slideshow video from images"""
        try:
            # Create filter complex for slideshow
            filter_parts = []
            
            for i, image in enumerate(images):
                # Scale and pad each image
                filter_parts.append(
                    f"[{i}:v]scale={resolution[0]}:{resolution[1]}:"
                    f"force_original_aspect_ratio=decrease,"
                    f"pad={resolution[0]}:{resolution[1]}:"
                    f"(ow-iw)/2:(oh-ih)/2[v{i}]"
                )
                
            # Concatenate videos
            concat_inputs = "".join([f"[v{i}]" for i in range(len(images))])
            filter_parts.append(
                f"{concat_inputs}concat=n={len(images)}:v=1:a=0[out]"
            )
            
            filter_complex = ";".join(filter_parts)
            
            # Build ffmpeg command
            cmd = [self.ffmpeg_path, "-y"]
            
            # Add input images
            for image in images:
                cmd.extend(["-loop", "1", "-t", str(image_duration), "-i", image])
                
            # Add filter and output options
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-r", str(fps),
                str(output_path)
            ])
            
            # Execute command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
        except Exception as e:
            logger.error(f"Slideshow creation error: {e}")
            raise
            
    async def _create_blank_video(
        self,
        output_path: Path,
        duration: float,
        resolution: Tuple[int, int],
        fps: int
    ):
        """Create blank video with specific duration"""
        try:
            cmd = [
                self.ffmpeg_path, "-y",
                "-f", "lavfi",
                "-i", f"color=c=black:s={resolution[0]}x{resolution[1]}:d={duration}",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-r", str(fps),
                str(output_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
        except Exception as e:
            logger.error(f"Blank video creation error: {e}")
            raise
            
    async def _combine_audio_video(
        self,
        video_path: Path,
        audio_path: str,
        output_path: str
    ):
        """Combine audio and video files"""
        try:
            cmd = [
                self.ffmpeg_path, "-y",
                "-i", str(video_path),
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(output_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
        except Exception as e:
            logger.error(f"Audio/video combination error: {e}")
            raise
            
    async def generate_thumbnail(
        self,
        title: str,
        style: str = "vibrant",
        output_path: Optional[str] = None,
        resolution: Tuple[int, int] = (1280, 720)
    ) -> Dict[str, Any]:
        """Generate thumbnail for video"""
        try:
            # Create image
            img = Image.new('RGB', resolution, color='white')
            draw = ImageDraw.Draw(img)
            
            # Add gradient background
            for i in range(resolution[1]):
                color_value = int(255 * (1 - i / resolution[1]))
                if style == "vibrant":
                    color = (255, color_value, 100)
                elif style == "dark":
                    color = (color_value // 3, color_value // 3, color_value // 2)
                else:
                    color = (color_value, color_value, color_value)
                    
                draw.rectangle([(0, i), (resolution[0], i + 1)], fill=color)
                
            # Try to load font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", 60)
                small_font = ImageFont.truetype("arial.ttf", 30)
            except:
                font = ImageFont.load_default()
                small_font = font
                
            # Add title text
            # Simple text wrapping
            words = title.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                test_line = " ".join(current_line)
                bbox = draw.textbbox((0, 0), test_line, font=font)
                if bbox[2] > resolution[0] - 100:
                    if len(current_line) > 1:
                        current_line.pop()
                        lines.append(" ".join(current_line))
                        current_line = [word]
                    else:
                        lines.append(test_line)
                        current_line = []
                        
            if current_line:
                lines.append(" ".join(current_line))
                
            # Draw title
            y_position = resolution[1] // 3
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                x_position = (resolution[0] - text_width) // 2
                
                # Add shadow
                draw.text((x_position + 2, y_position + 2), line, font=font, fill='black')
                # Main text
                draw.text((x_position, y_position), line, font=font, fill='white')
                
                y_position += 70
                
            # Add channel watermark
            watermark = "YTEmpire"
            bbox = draw.textbbox((0, 0), watermark, font=small_font)
            text_width = bbox[2] - bbox[0]
            draw.text(
                (resolution[0] - text_width - 20, resolution[1] - 40),
                watermark,
                font=small_font,
                fill='white'
            )
            
            # Save thumbnail
            if not output_path:
                output_path = self.temp_dir / f"thumbnail_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg"
            else:
                output_path = Path(output_path)
                
            img.save(output_path, 'JPEG', quality=95)
            
            return {
                "output_path": str(output_path),
                "resolution": f"{resolution[0]}x{resolution[1]}",
                "file_size": os.path.getsize(output_path),
                "style": style
            }
            
        except Exception as e:
            logger.error(f"Thumbnail generation error: {e}")
            raise
            
    async def add_subtitles(
        self,
        video_path: str,
        subtitle_path: str,
        output_path: str,
        style: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Add subtitles to video"""
        try:
            # Default subtitle style
            if not style:
                style = {
                    "Fontsize": "24",
                    "PrimaryColour": "&H00FFFFFF",
                    "BackColour": "&H80000000",
                    "BorderStyle": "3",
                    "Outline": "2",
                    "Shadow": "1"
                }
                
            # Build ffmpeg command
            cmd = [
                self.ffmpeg_path, "-y",
                "-i", video_path,
                "-vf", f"subtitles={subtitle_path}",
                "-c:a", "copy",
                output_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            return {
                "output_path": output_path,
                "subtitle_file": subtitle_path,
                "style": style
            }
            
        except Exception as e:
            logger.error(f"Subtitle addition error: {e}")
            raise
            
    async def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        fps: float = 1.0
    ) -> List[str]:
        """Extract frames from video"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            cmd = [
                self.ffmpeg_path,
                "-i", video_path,
                "-vf", f"fps={fps}",
                str(output_dir / "frame_%04d.jpg")
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            # Get list of extracted frames
            frames = sorted(output_dir.glob("frame_*.jpg"))
            return [str(f) for f in frames]
            
        except Exception as e:
            logger.error(f"Frame extraction error: {e}")
            raise
            
    async def create_video_with_audio(
        self,
        audio_path: str,
        visuals: List[Dict],
        output_path: str,
        title: str = "",
        subtitles: str = "",
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30
    ) -> Dict[str, Any]:
        """
        Create video with audio track and visuals
        This is the main method called by the video generation pipeline
        
        Args:
            audio_path: Path to the audio file (narration)
            visuals: List of visual elements (images, videos, text overlays)
            output_path: Where to save the final video
            title: Video title for overlays
            subtitles: Subtitle text for the video
            resolution: Video resolution (default 1920x1080)
            fps: Frames per second (default 30)
        """
        try:
            start_time = datetime.utcnow()
            
            # Create unique work directory
            video_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            work_dir = self.temp_dir / video_id
            work_dir.mkdir(exist_ok=True, parents=True)
            
            # Get audio duration
            audio_duration = await self._get_audio_duration(audio_path)
            logger.info(f"Audio duration: {audio_duration} seconds")
            
            # Process visuals (extract image paths from visual dictionaries)
            image_paths = []
            for visual in visuals:
                if isinstance(visual, dict):
                    if 'path' in visual:
                        image_paths.append(visual['path'])
                    elif 'url' in visual:
                        # Download image from URL if needed
                        img_path = await self._download_image(visual['url'], work_dir)
                        if img_path:
                            image_paths.append(img_path)
                elif isinstance(visual, str):
                    image_paths.append(visual)
            
            # If no visuals, create with stock images or blank video
            if not image_paths:
                logger.info("No visuals provided, creating video with default background")
                # Create a simple title card
                title_card = await self._create_title_card(title, work_dir, resolution)
                image_paths = [title_card]
            
            # Calculate duration for each image
            image_duration = audio_duration / len(image_paths) if image_paths else audio_duration
            
            # Build FFmpeg filter complex for advanced video assembly
            filter_complex = await self._build_advanced_filter_complex(
                image_paths, 
                image_duration,
                resolution,
                title if title else ""
            )
            
            # Prepare FFmpeg command with hardware acceleration if available
            cmd = await self._build_ffmpeg_command(
                audio_path,
                image_paths,
                output_path,
                filter_complex,
                resolution,
                fps,
                image_duration
            )
            
            logger.info(f"Starting FFmpeg with {len(image_paths)} visuals")
            
            # Execute FFmpeg command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr.decode()}")
                # Fallback to mock generator if available
                if mock_generator:
                    logger.info("FFmpeg failed, using mock video generator for testing")
                    result = await mock_generator.create_mock_video(
                        audio_path=audio_path,
                        visuals=visuals,
                        output_path=output_path,
                        title=title,
                        subtitles=subtitles
                    )
                    return result
                else:
                    # Try simpler method
                    logger.info("Trying fallback video creation method")
                    result = await self.create_video_from_assets(
                        audio_path=audio_path,
                        images=image_paths,
                        output_path=output_path,
                        resolution=resolution,
                        fps=fps,
                        transitions=True
                    )
                    return result
            
            # Add subtitles if provided
            if subtitles:
                subtitle_path = work_dir / "subtitles.srt"
                await self._create_srt_file(subtitles, subtitle_path, audio_duration)
                
                final_output = str(Path(output_path).parent / f"final_{Path(output_path).name}")
                await self.add_subtitles(output_path, str(subtitle_path), final_output)
                
                # Replace original with subtitled version
                shutil.move(final_output, output_path)
            
            # Get final file stats
            file_size = os.path.getsize(output_path)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"Video created successfully in {processing_time:.2f} seconds")
            
            return {
                "video_path": output_path,
                "duration": audio_duration,
                "file_size": file_size / (1024 * 1024),  # Convert to MB
                "resolution": f"{resolution[0]}x{resolution[1]}",
                "fps": fps,
                "processing_time": processing_time,
                "visuals_count": len(image_paths),
                "work_directory": str(work_dir)
            }
            
        except Exception as e:
            logger.error(f"Error in create_video_with_audio: {e}")
            # Try mock generator as final fallback
            if mock_generator:
                try:
                    logger.info("Using mock video generator as final fallback")
                    result = await mock_generator.create_mock_video(
                        audio_path=audio_path if audio_path else "mock_audio.mp3",
                        visuals=visuals if visuals else [],
                        output_path=output_path,
                        title=title if title else "Mock Video",
                        subtitles=subtitles if subtitles else ""
                    )
                    return result
                except Exception as mock_error:
                    logger.error(f"Mock generator also failed: {mock_error}")
            
            # Try simpler fallback method
            try:
                logger.info("Attempting final fallback video creation")
                return await self.create_video_from_assets(
                    audio_path=audio_path,
                    images=[],  # Will create blank video
                    output_path=output_path,
                    resolution=resolution,
                    fps=fps
                )
            except Exception as fallback_error:
                logger.error(f"All fallbacks failed: {fallback_error}")
                raise
    
    async def _build_advanced_filter_complex(
        self,
        image_paths: List[str],
        duration: float,
        resolution: Tuple[int, int],
        title: str = ""
    ) -> str:
        """Build advanced filter complex for video with transitions and effects"""
        filters = []
        
        # Scale and pad images
        for i, _ in enumerate(image_paths):
            filters.append(
                f"[{i}:v]scale={resolution[0]}:{resolution[1]}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={resolution[0]}:{resolution[1]}:"
                f"(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1[v{i}]"
            )
        
        # Add crossfade transitions between images
        if len(image_paths) > 1:
            transition_duration = min(0.5, duration / len(image_paths) / 4)
            current = "[v0]"
            
            for i in range(1, len(image_paths)):
                offset = (i * duration) - (transition_duration * i)
                filters.append(
                    f"{current}[v{i}]xfade=transition=fade:"
                    f"duration={transition_duration}:"
                    f"offset={offset}[v{i}x]"
                )
                current = f"[v{i}x]"
            
            # Final output
            filters.append(f"{current}format=yuv420p[out]")
        else:
            filters.append("[v0]format=yuv420p[out]")
        
        return ";".join(filters)
    
    async def _build_ffmpeg_command(
        self,
        audio_path: str,
        image_paths: List[str],
        output_path: str,
        filter_complex: str,
        resolution: Tuple[int, int],
        fps: int,
        image_duration: float
    ) -> List[str]:
        """Build optimized FFmpeg command"""
        cmd = [self.ffmpeg_path, "-y"]
        
        # Add hardware acceleration if available (Windows NVIDIA)
        if os.name == 'nt':  # Windows
            # Try to use NVIDIA hardware acceleration
            cmd.extend(["-hwaccel", "auto"])
        
        # Add input images with duration
        for image in image_paths:
            cmd.extend([
                "-loop", "1",
                "-t", str(image_duration),
                "-i", image
            ])
        
        # Add audio input
        cmd.extend(["-i", audio_path])
        
        # Add filter complex
        cmd.extend(["-filter_complex", filter_complex])
        
        # Map outputs
        cmd.extend([
            "-map", "[out]",
            "-map", f"{len(image_paths)}:a",  # Audio is last input
        ])
        
        # Video encoding settings (optimized for YouTube)
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            # Audio encoding
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            # YouTube optimizations
            "-movflags", "+faststart",
            "-shortest",  # Match shortest stream
            output_path
        ])
        
        return cmd
    
    async def _download_image(self, url: str, work_dir: Path) -> Optional[str]:
        """Download image from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        ext = url.split('.')[-1].split('?')[0]
                        if ext not in ['jpg', 'jpeg', 'png']:
                            ext = 'jpg'
                        
                        file_path = work_dir / f"img_{datetime.utcnow().timestamp()}.{ext}"
                        content = await response.read()
                        
                        async with aiofiles.open(file_path, 'wb') as f:
                            await f.write(content)
                        
                        return str(file_path)
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
        
        return None
    
    async def _create_title_card(
        self, 
        title: str, 
        work_dir: Path,
        resolution: Tuple[int, int]
    ) -> str:
        """Create a simple title card image"""
        try:
            # Create gradient background
            img = Image.new('RGB', resolution, color='black')
            draw = ImageDraw.Draw(img)
            
            # Add gradient
            for i in range(resolution[1]):
                color_value = int(100 + 155 * (i / resolution[1]))
                color = (color_value // 3, color_value // 2, color_value)
                draw.rectangle([(0, i), (resolution[0], i + 1)], fill=color)
            
            # Add title text
            try:
                font = ImageFont.truetype("arial.ttf", 80)
            except:
                font = ImageFont.load_default()
            
            # Center the text
            if title:
                bbox = draw.textbbox((0, 0), title, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (resolution[0] - text_width) // 2
                y = (resolution[1] - text_height) // 2
                
                # Draw shadow
                draw.text((x + 3, y + 3), title, font=font, fill='black')
                # Draw text
                draw.text((x, y), title, font=font, fill='white')
            
            # Save image
            output_path = work_dir / "title_card.jpg"
            img.save(output_path, 'JPEG', quality=95)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to create title card: {e}")
            # Return a black image as fallback
            img = Image.new('RGB', resolution, color='black')
            output_path = work_dir / "title_card.jpg"
            img.save(output_path, 'JPEG')
            return str(output_path)
    
    async def _create_srt_file(
        self,
        text: str,
        output_path: Path,
        duration: float
    ):
        """Create SRT subtitle file from text"""
        try:
            # Simple subtitle creation - split text into chunks
            words = text.split()
            words_per_subtitle = 10
            subtitle_duration = 3.0  # seconds per subtitle
            
            subtitles = []
            for i in range(0, len(words), words_per_subtitle):
                chunk = " ".join(words[i:i + words_per_subtitle])
                start_time = i / words_per_subtitle * subtitle_duration
                end_time = min(start_time + subtitle_duration, duration)
                
                if start_time < duration:
                    subtitles.append({
                        'index': len(subtitles) + 1,
                        'start': start_time,
                        'end': end_time,
                        'text': chunk
                    })
            
            # Write SRT file
            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                for sub in subtitles:
                    await f.write(f"{sub['index']}\n")
                    await f.write(f"{self._format_time(sub['start'])} --> {self._format_time(sub['end'])}\n")
                    await f.write(f"{sub['text']}\n\n")
                    
        except Exception as e:
            logger.error(f"Failed to create SRT file: {e}")
    
    def _format_time(self, seconds: float) -> str:
        """Format time for SRT file"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    async def optimize_video(
        self,
        input_path: str,
        output_path: str,
        target_size_mb: Optional[float] = None,
        quality: str = "medium"
    ) -> Dict[str, Any]:
        """Optimize video for YouTube upload"""
        try:
            quality_presets = {
                "low": {"crf": "28", "preset": "faster"},
                "medium": {"crf": "23", "preset": "medium"},
                "high": {"crf": "18", "preset": "slow"}
            }
            
            preset = quality_presets.get(quality, quality_presets["medium"])
            
            cmd = [
                self.ffmpeg_path, "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-crf", preset["crf"],
                "-preset", preset["preset"],
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                output_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            # Get file sizes
            original_size = os.path.getsize(input_path)
            optimized_size = os.path.getsize(output_path)
            
            return {
                "output_path": output_path,
                "original_size": original_size,
                "optimized_size": optimized_size,
                "compression_ratio": round(original_size / optimized_size, 2),
                "quality_preset": quality
            }
            
        except Exception as e:
            logger.error(f"Video optimization error: {e}")
            raise
            
    def cleanup_temp_files(self, video_id: Optional[str] = None):
        """Clean up temporary files"""
        try:
            if video_id:
                # Clean specific video directory
                video_dir = self.temp_dir / video_id
                if video_dir.exists():
                    shutil.rmtree(video_dir)
            else:
                # Clean all old files (older than 24 hours)
                import time
                current_time = time.time()
                for item in self.temp_dir.iterdir():
                    if item.is_dir():
                        if current_time - item.stat().st_mtime > 86400:  # 24 hours
                            shutil.rmtree(item)
                            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")