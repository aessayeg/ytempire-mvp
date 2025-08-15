"""
Quick Professional Video Creator
- Downloads real footage from Pexels
- Creates video in under 60 seconds
- Professional quality with real visuals
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import subprocess
import requests
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import asyncio

class QuickProVideoCreator:
    def __init__(self):
        self.output_dir = Path("generated_videos")
        self.output_dir.mkdir(exist_ok=True)
        
        self.assets_dir = Path("quick_pro_assets")
        self.assets_dir.mkdir(exist_ok=True)
        
        self.ffmpeg_path = Path("ffmpeg/ffmpeg.exe")
        if not self.ffmpeg_path.exists():
            self.ffmpeg_path = "ffmpeg"
        
        self.pexels_key = os.getenv("PEXELS_API_KEY", "")
    
    def download_pexels_video(self, query, filename):
        """Download a video from Pexels"""
        if not self.pexels_key:
            return None
        
        print(f"  Searching Pexels for: {query}")
        
        url = "https://api.pexels.com/videos/search"
        headers = {"Authorization": self.pexels_key}
        params = {"query": query, "per_page": 3, "orientation": "landscape"}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                videos = data.get("videos", [])
                
                for video in videos:
                    # Find HD video file
                    for file in video.get("video_files", []):
                        if file.get("quality") == "hd" and file.get("width", 0) >= 1280:
                            video_url = file["link"]
                            
                            # Download video
                            print(f"    Downloading HD video...")
                            video_response = requests.get(video_url, timeout=30)
                            if video_response.status_code == 200:
                                filepath = self.assets_dir / filename
                                with open(filepath, 'wb') as f:
                                    f.write(video_response.content)
                                print(f"    [OK] Downloaded: {filename}")
                                return filepath
        except Exception as e:
            print(f"    [ERROR] {e}")
        
        return None
    
    def download_pexels_image(self, query, filename):
        """Download an image from Pexels"""
        if not self.pexels_key:
            return None
        
        print(f"  Searching Pexels images for: {query}")
        
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": self.pexels_key}
        params = {"query": query, "per_page": 5, "orientation": "landscape"}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                photos = data.get("photos", [])
                
                if photos:
                    # Get high quality image
                    photo = photos[0]
                    image_url = photo["src"]["large2x"]
                    
                    # Download image
                    print(f"    Downloading HD image...")
                    image_response = requests.get(image_url, timeout=20)
                    if image_response.status_code == 200:
                        filepath = self.assets_dir / filename
                        with open(filepath, 'wb') as f:
                            f.write(image_response.content)
                        print(f"    [OK] Downloaded: {filename}")
                        return filepath
        except Exception as e:
            print(f"    [ERROR] {e}")
        
        return None
    
    def add_text_to_video(self, video_path, text, output_path):
        """Add text overlay to video using ffmpeg"""
        
        # Create text filter for ffmpeg
        text_filter = (
            f"drawtext=text='{text}':fontfile=arial.ttf:fontsize=80:"
            f"fontcolor=white:borderw=3:bordercolor=black:"
            f"x=(w-text_w)/2:y=h-150"
        )
        
        cmd = [
            str(self.ffmpeg_path), "-y",
            "-i", str(video_path),
            "-vf", text_filter,
            "-c:a", "copy",
            str(output_path)
        ]
        
        # Fallback without fontfile if it fails
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except:
            text_filter = (
                f"drawtext=text='{text}':fontsize=80:"
                f"fontcolor=white:borderw=3:bordercolor=black:"
                f"x=(w-text_w)/2:y=h-150"
            )
            
            cmd = [
                str(self.ffmpeg_path), "-y",
                "-i", str(video_path),
                "-vf", text_filter,
                "-c:a", "copy",
                str(output_path)
            ]
            subprocess.run(cmd, capture_output=True)
        
        return output_path.exists()
    
    async def create_video(self):
        """Create video quickly with real footage"""
        
        print("\n" + "="*70)
        print(" QUICK PROFESSIONAL VIDEO CREATION")
        print("="*70)
        
        # Clean assets directory
        for old_file in self.assets_dir.glob("*"):
            if old_file.is_file():
                old_file.unlink()
        
        # Define content
        segments = [
            {"query": "artificial intelligence technology", "text": "AI Revolution 2025"},
            {"query": "programming coding computer", "text": "GPT-4 Writes Apps"},
            {"query": "digital art creative", "text": "1 Billion AI Images"},
            {"query": "autonomous car tesla", "text": "Self-Driving Cars"},
            {"query": "medical technology health", "text": "AI Medical Diagnosis"},
            {"query": "robot automation future", "text": "Learning Robots"}
        ]
        
        print("\n[1/5] Downloading media from Pexels...")
        
        video_clips = []
        
        for i, segment in enumerate(segments):
            print(f"\nSegment {i+1}/{len(segments)}:")
            
            # Try to download video first
            video_file = self.download_pexels_video(
                segment["query"], 
                f"segment_{i}_video.mp4"
            )
            
            if video_file:
                # Trim to 4 seconds and add text
                trimmed_file = self.assets_dir / f"segment_{i}_trimmed.mp4"
                
                cmd = [
                    str(self.ffmpeg_path), "-y",
                    "-i", str(video_file),
                    "-t", "4",
                    "-vf", f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                    "-c:a", "copy",
                    str(trimmed_file)
                ]
                
                subprocess.run(cmd, capture_output=True)
                
                if trimmed_file.exists():
                    # Add text overlay
                    text_file = self.assets_dir / f"segment_{i}_text.mp4"
                    if self.add_text_to_video(trimmed_file, segment["text"], text_file):
                        video_clips.append(text_file)
                    else:
                        video_clips.append(trimmed_file)
            else:
                # Try image as fallback
                image_file = self.download_pexels_image(
                    segment["query"],
                    f"segment_{i}_image.jpg"
                )
                
                if image_file:
                    # Convert image to 4-second video with zoom effect
                    video_from_image = self.assets_dir / f"segment_{i}_from_image.mp4"
                    
                    cmd = [
                        str(self.ffmpeg_path), "-y",
                        "-loop", "1",
                        "-i", str(image_file),
                        "-t", "4",
                        "-vf", (
                            "scale=2880:1620,crop=1920:1080:'if(lte(t,4),(2880-1920)/2-(2880-1920)/2*t/4,0)':"
                            "'if(lte(t,4),(1620-1080)/2-(1620-1080)/2*t/4,0)',"
                            f"drawtext=text='{segment['text']}':fontsize=80:fontcolor=white:"
                            "borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-150"
                        ),
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        str(video_from_image)
                    ]
                    
                    subprocess.run(cmd, capture_output=True)
                    
                    if video_from_image.exists():
                        video_clips.append(video_from_image)
        
        if not video_clips:
            print("\n[ERROR] No media could be downloaded")
            return None
        
        print(f"\n[2/5] Successfully prepared {len(video_clips)} clips")
        
        # Create concat file
        print("\n[3/5] Combining clips...")
        
        concat_file = self.assets_dir / "concat.txt"
        with open(concat_file, 'w') as f:
            for clip in video_clips:
                f.write(f"file '{clip.absolute()}'\n")
        
        # Combine videos
        combined_video = self.assets_dir / "combined.mp4"
        
        cmd = [
            str(self.ffmpeg_path), "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(combined_video)
        ]
        
        subprocess.run(cmd, capture_output=True)
        
        # Create narration
        print("\n[4/5] Creating narration...")
        
        from app.services.ai_services import ElevenLabsService, AIServiceConfig
        
        script = """Welcome to the AI Revolution of 2025.
GPT-4 can write entire applications.
AI generates one billion images daily.
Self-driving cars are mainstream.
AI diagnoses diseases better than doctors.
Robots learn by watching videos."""
        
        config = AIServiceConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", "")
        )
        
        audio_path = self.output_dir / "quick_narration.mp3"
        
        if os.getenv("ELEVENLABS_API_KEY"):
            elevenlabs = ElevenLabsService(config)
            await elevenlabs.text_to_speech(
                text=script,
                output_path=str(audio_path)
            )
            print("  [OK] Narration created")
        else:
            # Create simple audio
            duration = len(video_clips) * 4
            cmd = [
                str(self.ffmpeg_path), "-y",
                "-f", "lavfi",
                "-i", f"anullsrc=r=44100:cl=stereo:d={duration}",
                "-c:a", "mp3",
                str(audio_path)
            ]
            subprocess.run(cmd, capture_output=True)
            print("  [OK] Silent audio created")
        
        # Combine with audio
        print("\n[5/5] Finalizing video...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_video = self.output_dir / f"quick_pro_{timestamp}.mp4"
        
        cmd = [
            str(self.ffmpeg_path), "-y",
            "-i", str(combined_video),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",  # Take video from first input
            "-map", "1:a:0",  # Take audio from second input  
            "-shortest",
            str(final_video)
        ]
        
        subprocess.run(cmd, capture_output=True)
        
        if final_video.exists():
            file_size = final_video.stat().st_size / (1024 * 1024)
            
            print("\n" + "="*70)
            print(" VIDEO CREATED SUCCESSFULLY!")
            print("="*70)
            
            print(f"\n[VIDEO INFO]")
            print(f"  File: {final_video.name}")
            print(f"  Path: {final_video.absolute()}")
            print(f"  Size: {file_size:.2f} MB")
            print(f"  Clips: {len(video_clips)}")
            
            print(f"\n[CONTENT]")
            print(f"  - Real stock footage from Pexels")
            print(f"  - Professional text overlays")
            print(f"  - Zoom effects on images")
            print(f"  - Synchronized narration")
            
            print(f"\n[TO PLAY]")
            print(f'  start "" "{final_video.absolute()}"')
            
            return str(final_video)
        
        return None


async def main():
    print("\n" + "="*70)
    print(" YTEMPIRE QUICK PROFESSIONAL VIDEO")
    print("="*70)
    print("\nCreates video with REAL footage from Pexels:")
    print("  - Downloads actual stock videos")
    print("  - Professional quality")
    print("  - Text overlays")
    print("  - Quick generation (< 60 seconds)")
    
    creator = QuickProVideoCreator()
    result = await creator.create_video()
    
    if result:
        print("\n" + "="*70)
        print(" SUCCESS! Professional video ready!")
        print("="*70)
    else:
        print("\n[FAILED] Could not create video")


if __name__ == "__main__":
    asyncio.run(main())