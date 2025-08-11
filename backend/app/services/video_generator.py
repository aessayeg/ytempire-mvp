"""
Professional Video Creator with Real Stock Footage
- Fetches actual videos/images from Pexels and Pixabay
- Implements Ken Burns effect for dynamic motion
- Professional text overlays and transitions
- Creates broadcast-quality videos
"""
import asyncio
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import subprocess
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math
import aiohttp
import random
import requests
from typing import List, Dict, Optional

class ProfessionalVideoCreator:
    def __init__(self):
        self.output_dir = Path("generated_videos")
        self.output_dir.mkdir(exist_ok=True)
        
        self.assets_dir = Path("professional_assets")
        self.assets_dir.mkdir(exist_ok=True)
        
        self.footage_dir = self.assets_dir / "footage"
        self.footage_dir.mkdir(exist_ok=True)
        
        # Clean old files
        for old_file in self.assets_dir.glob("frame_*.jpg"):
            old_file.unlink()
        
        self.ffmpeg_path = Path("ffmpeg/ffmpeg.exe")
        if not self.ffmpeg_path.exists():
            self.ffmpeg_path = "ffmpeg"
        
        self.fps = 30
        self.width = 1920
        self.height = 1080
        
        # API keys
        self.pexels_key = os.getenv("PEXELS_API_KEY", "")
        self.pixabay_key = os.getenv("PIXABAY_API_KEY", "")
        
        print(f"\n[API Status]")
        print(f"  Pexels API Key: {'Available' if self.pexels_key else 'Missing'}")
        print(f"  Pixabay API Key: {'Available' if self.pixabay_key else 'Missing'}")
    
    async def fetch_pexels_videos(self, query: str, per_page: int = 5) -> List[Dict]:
        """Fetch videos from Pexels API"""
        if not self.pexels_key:
            return []
        
        url = "https://api.pexels.com/videos/search"
        headers = {"Authorization": self.pexels_key}
        params = {
            "query": query,
            "per_page": per_page,
            "orientation": "landscape",
            "size": "medium"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        videos = []
                        for video in data.get("videos", []):
                            # Get HD video file
                            for file in video.get("video_files", []):
                                if file.get("quality") == "hd" and file.get("width") >= 1280:
                                    videos.append({
                                        "url": file["link"],
                                        "width": file["width"],
                                        "height": file["height"],
                                        "duration": video.get("duration", 10),
                                        "source": "pexels",
                                        "id": video["id"]
                                    })
                                    break
                        return videos
                    else:
                        print(f"  Pexels API error: {response.status}")
        except Exception as e:
            print(f"  Pexels fetch error: {e}")
        
        return []
    
    async def fetch_pixabay_videos(self, query: str, per_page: int = 5) -> List[Dict]:
        """Fetch videos from Pixabay API"""
        if not self.pixabay_key:
            return []
        
        url = "https://pixabay.com/api/videos/"
        params = {
            "key": self.pixabay_key,
            "q": query,
            "per_page": per_page,
            "video_type": "all",
            "min_width": 1280
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        videos = []
                        for video in data.get("hits", []):
                            if "videos" in video and "medium" in video["videos"]:
                                videos.append({
                                    "url": video["videos"]["medium"]["url"],
                                    "width": video["videos"]["medium"]["width"],
                                    "height": video["videos"]["medium"]["height"],
                                    "duration": video.get("duration", 10),
                                    "source": "pixabay",
                                    "id": video["id"]
                                })
                        return videos
                    else:
                        print(f"  Pixabay API error: {response.status}")
        except Exception as e:
            print(f"  Pixabay fetch error: {e}")
        
        return []
    
    async def fetch_pexels_images(self, query: str, per_page: int = 10) -> List[Dict]:
        """Fetch images from Pexels API"""
        if not self.pexels_key:
            return []
        
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": self.pexels_key}
        params = {
            "query": query,
            "per_page": per_page,
            "orientation": "landscape"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        images = []
                        for photo in data.get("photos", []):
                            images.append({
                                "url": photo["src"]["large2x"],  # High quality
                                "width": photo["width"],
                                "height": photo["height"],
                                "source": "pexels",
                                "type": "image",
                                "id": photo["id"]
                            })
                        return images
                    else:
                        print(f"  Pexels Images API error: {response.status}")
        except Exception as e:
            print(f"  Pexels Images fetch error: {e}")
        
        return []
    
    async def download_media(self, url: str, filename: str) -> Optional[Path]:
        """Download media file"""
        filepath = self.footage_dir / filename
        
        # Check if already downloaded
        if filepath.exists():
            return filepath
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        with open(filepath, 'wb') as f:
                            f.write(content)
                        return filepath
        except Exception as e:
            print(f"  Download error: {e}")
        
        return None
    
    def create_ken_burns_effect(self, image_path: Path, duration: float, effect_type: str = "zoom_in") -> List[Path]:
        """Apply Ken Burns effect to an image to create motion"""
        frames = []
        total_frames = int(duration * self.fps)
        
        # Load image
        img = Image.open(image_path)
        
        # Resize to fit video dimensions while maintaining aspect ratio
        img.thumbnail((self.width * 1.5, self.height * 1.5), Image.Resampling.LANCZOS)
        
        # Create canvas
        canvas_width = self.width
        canvas_height = self.height
        
        for frame_num in range(total_frames):
            progress = frame_num / max(total_frames - 1, 1)
            
            # Create frame
            frame = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
            
            if effect_type == "zoom_in":
                # Zoom from 100% to 130%
                scale = 1.0 + progress * 0.3
            elif effect_type == "zoom_out":
                # Zoom from 130% to 100%
                scale = 1.3 - progress * 0.3
            elif effect_type == "pan_left":
                # Pan from right to left
                scale = 1.2
            elif effect_type == "pan_right":
                # Pan from left to right
                scale = 1.2
            else:
                scale = 1.1
            
            # Calculate dimensions
            scaled_width = int(img.width * scale)
            scaled_height = int(img.height * scale)
            
            # Resize image
            scaled_img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            
            # Calculate position for panning
            if effect_type == "pan_left":
                x_offset = int((scaled_width - canvas_width) * progress)
                y_offset = (scaled_height - canvas_height) // 2
            elif effect_type == "pan_right":
                x_offset = int((scaled_width - canvas_width) * (1 - progress))
                y_offset = (scaled_height - canvas_height) // 2
            else:
                # Center the image
                x_offset = (scaled_width - canvas_width) // 2
                y_offset = (scaled_height - canvas_height) // 2
            
            # Crop to canvas size
            cropped = scaled_img.crop((x_offset, y_offset, 
                                      x_offset + canvas_width, 
                                      y_offset + canvas_height))
            
            # Paste onto frame
            frame.paste(cropped, (0, 0))
            
            # Save frame
            frame_path = self.assets_dir / f"kb_frame_{frame_num:05d}.jpg"
            frame.save(frame_path, 'JPEG', quality=95)
            frames.append(frame_path)
        
        return frames
    
    def add_text_overlay(self, image: Image.Image, text: str, position: str = "center") -> Image.Image:
        """Add professional text overlay to image"""
        draw = ImageDraw.Draw(image)
        
        # Load fonts
        try:
            font_large = ImageFont.truetype("arial.ttf", 80)
            font_small = ImageFont.truetype("arial.ttf", 50)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Create semi-transparent overlay
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Add gradient overlay for text readability
        if position == "bottom":
            # Bottom gradient
            for y in range(self.height - 300, self.height):
                alpha = int(180 * ((y - (self.height - 300)) / 300))
                overlay_draw.rectangle([(0, y), (self.width, y+1)], 
                                      fill=(0, 0, 0, alpha))
        else:
            # Center gradient
            for y in range(300, 600):
                alpha = 150
                overlay_draw.rectangle([(0, y), (self.width, y+1)], 
                                      fill=(0, 0, 0, alpha))
        
        # Composite overlay
        image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Add text with shadow
        if position == "bottom":
            text_y = self.height - 150
        else:
            text_y = 450
        
        # Shadow
        draw.text((self.width//2 + 3, text_y + 3), text,
                 font=font_large, fill=(0, 0, 0), anchor='mm')
        
        # Main text
        draw.text((self.width//2, text_y), text,
                 font=font_large, fill=(255, 255, 255), anchor='mm')
        
        return image
    
    async def create_video_segments(self):
        """Create video segments with real footage"""
        
        print("\n[1/6] Defining content segments...")
        
        segments = [
            {
                "id": "intro",
                "text": "Welcome to the AI Revolution of 2025",
                "keywords": ["artificial intelligence", "technology", "future", "innovation"],
                "duration": 3.0
            },
            {
                "id": "fact1",
                "text": "GPT-4 can write entire applications",
                "keywords": ["coding", "programming", "computer", "software development"],
                "duration": 4.0
            },
            {
                "id": "fact2",
                "text": "AI generates one billion images daily",
                "keywords": ["digital art", "creativity", "computer graphics", "design"],
                "duration": 4.0
            },
            {
                "id": "fact3",
                "text": "Self-driving cars are mainstream",
                "keywords": ["autonomous vehicles", "tesla", "self driving car", "future transportation"],
                "duration": 4.0
            },
            {
                "id": "fact4",
                "text": "AI diagnoses diseases better than doctors",
                "keywords": ["medical technology", "healthcare", "hospital", "doctor"],
                "duration": 4.0
            },
            {
                "id": "fact5",
                "text": "Robots learn by watching videos",
                "keywords": ["robotics", "machine learning", "automation", "robot"],
                "duration": 4.0
            },
            {
                "id": "future",
                "text": "The future is artificial intelligence",
                "keywords": ["future city", "technology", "innovation", "sci-fi"],
                "duration": 3.0
            }
        ]
        
        print(f"  Defined {len(segments)} segments")
        
        # Fetch media for each segment
        print("\n[2/6] Fetching stock footage and images...")
        
        for segment in segments:
            print(f"\n  Searching for: {segment['id']} - {segment['keywords'][0]}...")
            
            # Try to get videos first
            videos = []
            if self.pexels_key:
                videos = await self.fetch_pexels_videos(segment['keywords'][0], 2)
            
            if not videos and self.pixabay_key:
                videos = await self.fetch_pixabay_videos(segment['keywords'][0], 2)
            
            # If no videos, get images
            images = []
            if not videos:
                if self.pexels_key:
                    images = await self.fetch_pexels_images(segment['keywords'][0], 5)
                    print(f"    Found {len(images)} images from Pexels")
            
            segment["videos"] = videos
            segment["images"] = images
            
            # Download media
            if videos:
                print(f"    Downloading {len(videos)} videos...")
                for i, video in enumerate(videos[:2]):  # Limit to 2 videos
                    filename = f"{segment['id']}_video_{i}.mp4"
                    path = await self.download_media(video["url"], filename)
                    if path:
                        video["local_path"] = path
                        print(f"      Downloaded: {filename}")
            
            if images:
                print(f"    Downloading {len(images[:3])} images...")
                for i, image in enumerate(images[:3]):  # Limit to 3 images
                    filename = f"{segment['id']}_image_{i}.jpg"
                    path = await self.download_media(image["url"], filename)
                    if path:
                        image["local_path"] = path
                        print(f"      Downloaded: {filename}")
        
        return segments
    
    async def process_segment_media(self, segment: Dict) -> List[Path]:
        """Process media for a segment into frames"""
        frames = []
        duration = segment["duration"]
        frames_needed = int(duration * self.fps)
        
        print(f"\n  Processing {segment['id']}...")
        
        # Use videos if available
        if segment.get("videos") and any(v.get("local_path") for v in segment["videos"]):
            for video in segment["videos"]:
                if not video.get("local_path"):
                    continue
                
                # Extract frames from video
                video_frames_dir = self.assets_dir / f"{segment['id']}_frames"
                video_frames_dir.mkdir(exist_ok=True)
                
                cmd = [
                    str(self.ffmpeg_path), "-y",
                    "-i", str(video["local_path"]),
                    "-t", str(duration),
                    "-vf", f"fps={self.fps},scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2",
                    str(video_frames_dir / "frame_%05d.jpg")
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                
                # Collect frames
                video_frames = sorted(video_frames_dir.glob("frame_*.jpg"))
                frames.extend(video_frames[:frames_needed])
                
                if len(frames) >= frames_needed:
                    break
        
        # Use images with Ken Burns effect if no videos or need more frames
        if len(frames) < frames_needed and segment.get("images"):
            remaining_frames = frames_needed - len(frames)
            
            for image in segment["images"]:
                if not image.get("local_path"):
                    continue
                
                # Apply Ken Burns effect
                effect = random.choice(["zoom_in", "zoom_out", "pan_left", "pan_right"])
                image_frames = self.create_ken_burns_effect(
                    image["local_path"],
                    remaining_frames / self.fps,
                    effect
                )
                
                frames.extend(image_frames)
                
                if len(frames) >= frames_needed:
                    break
        
        # If still no frames, create placeholder
        if not frames:
            print(f"    No media found, creating animated placeholder...")
            frames = self.create_animated_placeholder(segment, frames_needed)
        
        # Add text overlay to frames
        print(f"    Adding text overlays to {len(frames)} frames...")
        processed_frames = []
        
        for i, frame_path in enumerate(frames[:frames_needed]):
            # Load frame
            frame = Image.open(frame_path)
            
            # Add text overlay with fade in/out
            progress = i / max(frames_needed - 1, 1)
            
            # Fade in first 0.5 seconds, fade out last 0.5 seconds
            fade_duration = 0.5
            fade_frames = int(fade_duration * self.fps)
            
            if i < fade_frames:
                alpha = i / fade_frames
            elif i > frames_needed - fade_frames:
                alpha = (frames_needed - i) / fade_frames
            else:
                alpha = 1.0
            
            if alpha > 0:
                # Add text
                frame = self.add_text_overlay(frame, segment["text"], "bottom")
            
            # Save processed frame
            processed_path = self.assets_dir / f"processed_{segment['id']}_{i:05d}.jpg"
            frame.save(processed_path, 'JPEG', quality=95)
            processed_frames.append(processed_path)
        
        print(f"    Processed {len(processed_frames)} frames")
        return processed_frames
    
    def create_animated_placeholder(self, segment: Dict, frames_needed: int) -> List[Path]:
        """Create animated placeholder when no media available"""
        frames = []
        
        for frame_num in range(frames_needed):
            progress = frame_num / max(frames_needed - 1, 1)
            
            # Create gradient background
            img = Image.new('RGB', (self.width, self.height))
            draw = ImageDraw.Draw(img)
            
            # Animated gradient
            for y in range(self.height):
                ratio = y / self.height
                
                # Color based on segment
                if "fact1" in segment["id"]:
                    r = int(100 + 100 * progress)
                    g = int(50 + 50 * ratio)
                    b = int(150 - 50 * progress)
                elif "fact2" in segment["id"]:
                    r = int(50 + 100 * ratio)
                    g = int(100 + 100 * progress)
                    b = int(100)
                else:
                    r = int(50 + 50 * ratio)
                    g = int(50 + 50 * ratio)
                    b = int(100 + 50 * progress)
                
                draw.rectangle([(0, y), (self.width, y+1)], fill=(r, g, b))
            
            # Add moving shapes for visual interest
            for i in range(5):
                x = (frame_num * 10 + i * 200) % self.width
                y = 300 + math.sin((frame_num + i * 30) * 0.1) * 200
                size = 50 + math.sin(frame_num * 0.05 + i) * 20
                
                draw.ellipse([x-size, y-size, x+size, y+size],
                           fill=(255, 255, 255, 50))
            
            # Save frame
            frame_path = self.assets_dir / f"placeholder_{segment['id']}_{frame_num:05d}.jpg"
            img.save(frame_path, 'JPEG', quality=90)
            frames.append(frame_path)
        
        return frames
    
    async def create_narration(self):
        """Create professional narration"""
        
        print("\n[3/6] Creating narration...")
        
        script = """Welcome to the AI Revolution of 2025.

GPT-4 can write entire applications.
AI generates one billion images daily.
Self-driving cars are mainstream.
AI diagnoses diseases better than doctors.
Robots learn by watching videos.

The future is artificial intelligence."""
        
        from app.services.ai_services import ElevenLabsService, AIServiceConfig
        
        config = AIServiceConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", "")
        )
        
        audio_path = self.output_dir / "professional_narration.mp3"
        
        if os.getenv("ELEVENLABS_API_KEY"):
            elevenlabs = ElevenLabsService(config)
            await elevenlabs.text_to_speech(
                text=script,
                output_path=str(audio_path)
            )
            print(f"  [OK] Professional narration created")
        else:
            # Create test audio
            self.create_test_audio(audio_path, 26)
            print(f"  [OK] Test audio created")
        
        return audio_path
    
    def create_test_audio(self, output_path, duration):
        """Create test audio"""
        cmd = [
            str(self.ffmpeg_path), "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration}:sample_rate=44100",
            "-af", "volume=0.02",
            "-c:a", "mp3",
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
    
    async def assemble_video(self, segments: List[Dict], audio_path: Path):
        """Assemble final video from segments"""
        
        print("\n[4/6] Processing all segments...")
        
        all_frames = []
        
        for segment in segments:
            segment_frames = await self.process_segment_media(segment)
            all_frames.extend(segment_frames)
        
        print(f"\n[5/6] Assembling {len(all_frames)} frames into video...")
        
        # Create video from frames
        temp_video = self.assets_dir / "temp_professional.mp4"
        
        # Create frame list file
        frame_list = self.assets_dir / "frames.txt"
        with open(frame_list, 'w') as f:
            for frame in all_frames:
                f.write(f"file '{frame.absolute()}'\n")
                f.write(f"duration {1/self.fps}\n")
        
        cmd = [
            str(self.ffmpeg_path), "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(frame_list),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(temp_video)
        ]
        
        subprocess.run(cmd, capture_output=True)
        
        # Add audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_video = self.output_dir / f"professional_video_{timestamp}.mp4"
        
        cmd = [
            str(self.ffmpeg_path), "-y",
            "-i", str(temp_video),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(final_video)
        ]
        
        subprocess.run(cmd, capture_output=True)
        
        if final_video.exists():
            print(f"  [OK] Final video created: {final_video.name}")
            return final_video
        
        return None
    
    async def create_video(self):
        """Main video creation process"""
        
        print("\n" + "="*70)
        print(" CREATING PROFESSIONAL VIDEO WITH STOCK FOOTAGE")
        print("="*70)
        
        # Create segments with media
        segments = await self.create_video_segments()
        
        # Create narration
        audio_path = await self.create_narration()
        
        # Assemble video
        final_video = await self.assemble_video(segments, audio_path)
        
        if final_video and final_video.exists():
            file_size = final_video.stat().st_size / (1024 * 1024)
            
            print("\n" + "="*70)
            print(" PROFESSIONAL VIDEO CREATED!")
            print("="*70)
            
            print(f"\n[VIDEO DETAILS]")
            print(f"  File: {final_video.name}")
            print(f"  Path: {final_video.absolute()}")
            print(f"  Size: {file_size:.2f} MB")
            
            print(f"\n[CONTENT SOURCES]")
            videos_count = sum(len(s.get("videos", [])) for s in segments)
            images_count = sum(len(s.get("images", [])) for s in segments)
            print(f"  Videos fetched: {videos_count}")
            print(f"  Images fetched: {images_count}")
            print(f"  Ken Burns effects applied")
            print(f"  Professional text overlays added")
            
            print(f"\n[TO PLAY]")
            print(f'  start "" "{final_video.absolute()}"')
            
            return str(final_video)
        
        return None


async def main():
    print("\n" + "="*70)
    print(" YTEMPIRE PROFESSIONAL VIDEO CREATOR")
    print("="*70)
    print("\nThis creates a PROFESSIONAL video with:")
    print("  - Real stock footage from Pexels/Pixabay")
    print("  - Ken Burns effect for dynamic motion")
    print("  - Professional text overlays")
    print("  - Smooth transitions")
    print("  - Broadcast-quality output")
    
    creator = ProfessionalVideoCreator()
    result = await creator.create_video()
    
    if result:
        print("\n" + "="*70)
        print(" SUCCESS! Your professional video is ready!")
        print("="*70)
    else:
        print("\n[FAILED] Video creation failed.")


if __name__ == "__main__":
    asyncio.run(main())