"""
Video Compilation Tasks
Owner: Data Pipeline Engineer #2

Video compilation and editing tasks for combining audio, visuals, and effects.
Integrates with FFmpeg, OpenCV, and other video processing tools.
"""

from celery import current_task
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import subprocess
import json
import os
from pathlib import Path
import tempfile

from app.core.celery_app import celery_app
from app.core.config import settings
from app.utils.cost_calculator import CostCalculator
from app.core.metrics import metrics

logger = logging.getLogger(__name__)


class VideoCompilationError(Exception):
    """Custom exception for video compilation errors."""
    pass


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 2, 'countdown': 300})
def compile_video_task(self, media_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compile final video from audio and visual components.
    
    Args:
        media_data: Combined results from audio synthesis and visual generation
        
    Returns:
        Video compilation results with file paths and metadata
    """
    try:
        video_id = media_data['id']
        logger.info(f"Starting video compilation for: {video_id}")
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'video_compilation', 'progress': 10, 'video_id': video_id}
        )
        
        # Validate required inputs
        validate_compilation_inputs(media_data)
        
        # Initialize compilation workspace
        workspace_dir = setup_compilation_workspace(video_id)
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'preparing_assets', 'progress': 20, 'video_id': video_id}
        )
        
        # Prepare video assets
        audio_path = media_data.get('audio_file_path')
        visuals = media_data.get('visuals', {})
        script = media_data.get('script', '')
        
        # Generate video timeline from script and visuals
        timeline = generate_video_timeline(script, visuals, audio_path)
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'processing_visuals', 'progress': 40, 'video_id': video_id}
        )
        
        # Process and prepare visual assets
        processed_visuals = process_visual_assets(visuals, workspace_dir, timeline)
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'creating_video_sequence', 'progress': 60, 'video_id': video_id}
        )
        
        # Create video sequence
        video_sequence_path = create_video_sequence(
            processed_visuals, timeline, workspace_dir, video_id
        )
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'adding_audio', 'progress': 75, 'video_id': video_id}
        )
        
        # Combine video with audio
        final_video_path = combine_audio_video(
            video_sequence_path, audio_path, workspace_dir, video_id
        )
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'post_processing', 'progress': 85, 'video_id': video_id}
        )
        
        # Apply post-processing effects
        final_video_path = apply_post_processing_effects(
            final_video_path, media_data, workspace_dir, video_id
        )
        
        # Generate thumbnail
        thumbnail_path = generate_video_thumbnail(final_video_path, workspace_dir, video_id)
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'quality_analysis', 'progress': 95, 'video_id': video_id}
        )
        
        # Analyze video quality
        video_metadata = analyze_compiled_video(final_video_path)
        
        # Calculate compilation costs
        compilation_cost = calculate_compilation_cost(video_metadata)
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'compilation_complete', 'progress': 100, 'video_id': video_id}
        )
        
        # Prepare result
        result = {
            **media_data,
            'video_file_path': final_video_path,
            'thumbnail_file_path': thumbnail_path,
            'video_duration': video_metadata['duration'],
            'video_resolution': video_metadata['resolution'],
            'video_file_size_mb': video_metadata['file_size_mb'],
            'compilation_cost': compilation_cost,
            'video_compilation_completed': True,
            'compilation_metadata': {
                'workspace_dir': str(workspace_dir),
                'timeline_segments': len(timeline),
                'visual_assets_used': len(processed_visuals),
                'compiled_at': datetime.utcnow().isoformat(),
                'quality_metrics': video_metadata
            }
        }
        
        # Record metrics
        metrics.record_api_cost('video_compilation', media_data['user_id'], compilation_cost)
        
        # Cleanup workspace (keep final files)
        cleanup_compilation_workspace(workspace_dir, keep_finals=True)
        
        logger.info(f"Video compilation completed for: {video_id}, cost: ${compilation_cost:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Video compilation failed for video {media_data.get('id')}: {str(e)}")
        raise


def validate_compilation_inputs(media_data: Dict[str, Any]) -> None:
    """Validate that all required inputs are available for compilation."""
    
    required_fields = ['id', 'user_id', 'script', 'audio_file_path']
    for field in required_fields:
        if field not in media_data:
            raise VideoCompilationError(f"Missing required field: {field}")
    
    # Check if audio file exists
    audio_path = media_data.get('audio_file_path')
    if not os.path.exists(audio_path):
        raise VideoCompilationError(f"Audio file not found: {audio_path}")
    
    # Check if visuals are available
    visuals = media_data.get('visuals', {})
    if not visuals.get('images') and not visuals.get('videos'):
        logger.warning("No visual assets found, will generate default visuals")


def setup_compilation_workspace(video_id: str) -> Path:
    """Setup workspace directory for video compilation."""
    
    workspace_dir = Path(f"/tmp/video_compilation/{video_id}")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (workspace_dir / "assets").mkdir(exist_ok=True)
    (workspace_dir / "sequences").mkdir(exist_ok=True)
    (workspace_dir / "temp").mkdir(exist_ok=True)
    (workspace_dir / "output").mkdir(exist_ok=True)
    
    return workspace_dir


def generate_video_timeline(script: str, visuals: Dict[str, Any], audio_path: str) -> List[Dict[str, Any]]:
    """Generate video timeline based on script and available visuals."""
    
    try:
        # Get audio duration
        audio_duration = get_audio_duration(audio_path)
        
        # Split script into segments
        script_segments = split_script_into_segments(script)
        
        # Calculate time per segment
        time_per_segment = audio_duration / len(script_segments) if script_segments else audio_duration
        
        timeline = []
        current_time = 0.0
        
        available_images = visuals.get('images', [])
        image_index = 0
        
        for i, segment in enumerate(script_segments):
            segment_duration = min(time_per_segment, audio_duration - current_time)
            
            # Assign visual asset to segment
            visual_asset = None
            if image_index < len(available_images):
                visual_asset = available_images[image_index]
                image_index += 1
            
            timeline_entry = {
                'segment_id': i,
                'start_time': current_time,
                'duration': segment_duration,
                'text': segment.strip(),
                'visual_asset': visual_asset,
                'transition': 'fade' if i > 0 else None,
                'transition_duration': 0.5
            }
            
            timeline.append(timeline_entry)
            current_time += segment_duration
            
            if current_time >= audio_duration:
                break
        
        logger.info(f"Generated timeline with {len(timeline)} segments, total duration: {audio_duration:.2f}s")
        return timeline
        
    except Exception as e:
        logger.error(f"Failed to generate video timeline: {str(e)}")
        raise VideoCompilationError(f"Timeline generation failed: {str(e)}")


def process_visual_assets(
    visuals: Dict[str, Any], 
    workspace_dir: Path, 
    timeline: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Process and prepare visual assets for video compilation."""
    
    processed_visuals = []
    
    try:
        # Process existing images
        images = visuals.get('images', [])
        for i, image_data in enumerate(images):
            processed_image = process_single_image(image_data, workspace_dir, i)
            processed_visuals.append(processed_image)
        
        # Generate additional visuals if needed
        needed_visuals = len(timeline) - len(processed_visuals)
        if needed_visuals > 0:
            generated_visuals = generate_default_visuals(needed_visuals, workspace_dir, len(processed_visuals))
            processed_visuals.extend(generated_visuals)
        
        return processed_visuals
        
    except Exception as e:
        logger.error(f"Failed to process visual assets: {str(e)}")
        raise VideoCompilationError(f"Visual asset processing failed: {str(e)}")


def process_single_image(image_data: Dict[str, Any], workspace_dir: Path, index: int) -> Dict[str, Any]:
    """Process a single image for video use."""
    
    try:
        from PIL import Image, ImageEnhance, ImageFilter
        
        # Load image
        image_path = image_data.get('file_path') or image_data.get('url')
        
        if not image_path:
            raise VideoCompilationError("No image path provided")
        
        if image_path.startswith('http'):
            # Download image
            import httpx
            with httpx.Client() as client:
                response = client.get(image_path)
                response.raise_for_status()
                
                temp_path = workspace_dir / "temp" / f"downloaded_{index}.jpg"
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                image_path = str(temp_path)
        
        # Open and process image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to video dimensions (1920x1080)
        target_size = (1920, 1080)
        
        # Calculate dimensions to maintain aspect ratio
        img_ratio = img.width / img.height
        target_ratio = target_size[0] / target_size[1]
        
        if img_ratio > target_ratio:
            # Image is wider than target
            new_width = int(target_size[1] * img_ratio)
            new_height = target_size[1]
        else:
            # Image is taller than target
            new_width = target_size[0]
            new_height = int(target_size[0] / img_ratio)
        
        # Resize image
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create final image with black bars if necessary
        final_img = Image.new('RGB', target_size, (0, 0, 0))
        
        # Center the image
        paste_x = (target_size[0] - new_width) // 2
        paste_y = (target_size[1] - new_height) // 2
        final_img.paste(img, (paste_x, paste_y))
        
        # Apply subtle enhancement
        enhancer = ImageEnhance.Color(final_img)
        final_img = enhancer.enhance(1.1)  # Slight color boost
        
        enhancer = ImageEnhance.Contrast(final_img)
        final_img = enhancer.enhance(1.05)  # Slight contrast boost
        
        # Save processed image
        output_path = workspace_dir / "assets" / f"processed_image_{index:03d}.jpg"
        final_img.save(output_path, 'JPEG', quality=95)
        
        return {
            'index': index,
            'file_path': str(output_path),
            'resolution': target_size,
            'original_path': image_path,
            'processed': True
        }
        
    except Exception as e:
        logger.error(f"Failed to process image {index}: {str(e)}")
        # Return a default visual instead of failing
        return generate_default_visual(workspace_dir, index)


def generate_default_visuals(count: int, workspace_dir: Path, start_index: int) -> List[Dict[str, Any]]:
    """Generate default visual assets when not enough are provided."""
    
    default_visuals = []
    
    for i in range(count):
        visual = generate_default_visual(workspace_dir, start_index + i)
        default_visuals.append(visual)
    
    return default_visuals


def generate_default_visual(workspace_dir: Path, index: int) -> Dict[str, Any]:
    """Generate a single default visual."""
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple gradient background
        img = Image.new('RGB', (1920, 1080), (20, 30, 50))
        draw = ImageDraw.Draw(img)
        
        # Create gradient
        for y in range(1080):
            color_intensity = int(20 + (y / 1080) * 30)
            color = (color_intensity, color_intensity + 10, color_intensity + 20)
            draw.line([(0, y), (1920, y)], fill=color)
        
        # Add subtle pattern
        for x in range(0, 1920, 100):
            for y in range(0, 1080, 100):
                draw.rectangle([x, y, x+50, y+50], outline=(40, 50, 70), width=1)
        
        # Save default image
        output_path = workspace_dir / "assets" / f"default_visual_{index:03d}.jpg"
        img.save(output_path, 'JPEG', quality=90)
        
        return {
            'index': index,
            'file_path': str(output_path),
            'resolution': (1920, 1080),
            'is_default': True,
            'processed': True
        }
        
    except Exception as e:
        logger.error(f"Failed to generate default visual: {str(e)}")
        raise VideoCompilationError(f"Default visual generation failed: {str(e)}")


def create_video_sequence(
    processed_visuals: List[Dict[str, Any]], 
    timeline: List[Dict[str, Any]], 
    workspace_dir: Path, 
    video_id: str
) -> str:
    """Create video sequence from processed visuals and timeline."""
    
    try:
        # Create individual video clips for each timeline segment
        clip_paths = []
        
        for i, segment in enumerate(timeline):
            visual_index = min(i, len(processed_visuals) - 1)
            visual = processed_visuals[visual_index]
            
            clip_path = create_video_clip(
                visual['file_path'],
                segment['duration'],
                workspace_dir / "sequences" / f"clip_{i:03d}.mp4"
            )
            clip_paths.append(clip_path)
        
        # Concatenate all clips
        sequence_path = workspace_dir / "output" / f"{video_id}_sequence.mp4"
        concatenate_video_clips(clip_paths, str(sequence_path))
        
        return str(sequence_path)
        
    except Exception as e:
        logger.error(f"Failed to create video sequence: {str(e)}")
        raise VideoCompilationError(f"Video sequence creation failed: {str(e)}")


def create_video_clip(image_path: str, duration: float, output_path: Path) -> str:
    """Create a video clip from a single image with specified duration."""
    
    try:
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-loop', '1',  # Loop input image
            '-i', image_path,
            '-t', str(duration),  # Duration
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-r', '30',  # Frame rate
            '-crf', '23',  # Quality
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise VideoCompilationError(f"FFmpeg error: {result.stderr}")
        
        return str(output_path)
        
    except subprocess.TimeoutExpired:
        raise VideoCompilationError("Video clip creation timeout")
    except Exception as e:
        raise VideoCompilationError(f"Failed to create video clip: {str(e)}")


def concatenate_video_clips(clip_paths: List[str], output_path: str) -> None:
    """Concatenate multiple video clips into one sequence."""
    
    try:
        # Create concat file
        concat_file = Path(output_path).parent / "concat_list.txt"
        
        with open(concat_file, 'w') as f:
            for clip_path in clip_paths:
                f.write(f"file '{clip_path}'\n")
        
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            raise VideoCompilationError(f"FFmpeg concatenation error: {result.stderr}")
        
        # Cleanup concat file
        concat_file.unlink()
        
    except subprocess.TimeoutExpired:
        raise VideoCompilationError("Video concatenation timeout")
    except Exception as e:
        raise VideoCompilationError(f"Failed to concatenate video clips: {str(e)}")


def combine_audio_video(video_path: str, audio_path: str, workspace_dir: Path, video_id: str) -> str:
    """Combine video sequence with audio track."""
    
    try:
        output_path = workspace_dir / "output" / f"{video_id}_final.mp4"
        
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',  # Copy video without re-encoding
            '-c:a', 'aac',   # Encode audio to AAC
            '-b:a', '192k',  # Audio bitrate
            '-shortest',     # Use shortest stream duration
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            raise VideoCompilationError(f"FFmpeg audio/video combination error: {result.stderr}")
        
        return str(output_path)
        
    except subprocess.TimeoutExpired:
        raise VideoCompilationError("Audio/video combination timeout")
    except Exception as e:
        raise VideoCompilationError(f"Failed to combine audio and video: {str(e)}")


def apply_post_processing_effects(
    video_path: str, 
    media_data: Dict[str, Any], 
    workspace_dir: Path, 
    video_id: str
) -> str:
    """Apply post-processing effects to the final video."""
    
    try:
        output_path = workspace_dir / "output" / f"{video_id}_processed.mp4"
        
        # Build filter chain
        filters = []
        
        # Add subtle color correction
        filters.append("eq=contrast=1.05:brightness=0.02:saturation=1.1")
        
        # Add slight sharpening
        filters.append("unsharp=5:5:0.8:3:3:0.4")
        
        # Combine filters
        filter_string = ",".join(filters)
        
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-i', video_path,
            '-vf', filter_string,
            '-c:v', 'libx264',
            '-crf', '20',  # High quality
            '-preset', 'medium',
            '-c:a', 'copy',  # Copy audio
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        
        if result.returncode != 0:
            # If post-processing fails, return original
            logger.warning(f"Post-processing failed, using original: {result.stderr}")
            return video_path
        
        return str(output_path)
        
    except Exception as e:
        logger.warning(f"Post-processing failed, using original video: {str(e)}")
        return video_path


def generate_video_thumbnail(video_path: str, workspace_dir: Path, video_id: str) -> str:
    """Generate thumbnail from video."""
    
    try:
        thumbnail_path = workspace_dir / "output" / f"{video_id}_thumbnail.jpg"
        
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-i', video_path,
            '-ss', '00:00:05',  # Take frame at 5 seconds
            '-vframes', '1',    # Take only 1 frame
            '-vf', 'scale=1280:720',  # YouTube thumbnail size
            '-q:v', '2',        # High quality
            str(thumbnail_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            logger.warning(f"Thumbnail generation failed: {result.stderr}")
            # Generate a default thumbnail
            return generate_default_thumbnail(workspace_dir, video_id)
        
        return str(thumbnail_path)
        
    except Exception as e:
        logger.warning(f"Thumbnail generation failed: {str(e)}")
        return generate_default_thumbnail(workspace_dir, video_id)


def generate_default_thumbnail(workspace_dir: Path, video_id: str) -> str:
    """Generate a default thumbnail."""
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create thumbnail image
        img = Image.new('RGB', (1280, 720), (30, 40, 60))
        draw = ImageDraw.Draw(img)
        
        # Add gradient background
        for y in range(720):
            color_intensity = int(30 + (y / 720) * 30)
            color = (color_intensity, color_intensity + 10, color_intensity + 20)
            draw.line([(0, y), (1280, y)], fill=color)
        
        # Add title text (if available)
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except:
            font = ImageFont.load_default()
        
        text = "YTEmpire Video"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = (1280 - text_width) // 2
        text_y = (720 - text_height) // 2
        
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
        
        # Save thumbnail
        thumbnail_path = workspace_dir / "output" / f"{video_id}_thumbnail.jpg"
        img.save(thumbnail_path, 'JPEG', quality=95)
        
        return str(thumbnail_path)
        
    except Exception as e:
        logger.error(f"Failed to generate default thumbnail: {str(e)}")
        raise VideoCompilationError("Thumbnail generation failed")


def analyze_compiled_video(video_path: str) -> Dict[str, Any]:
    """Analyze compiled video quality and metadata."""
    
    try:
        # Use ffprobe to get video metadata
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            raise VideoCompilationError("Failed to analyze video")
        
        metadata = json.loads(result.stdout)
        
        # Extract video stream info
        video_stream = None
        audio_stream = None
        
        for stream in metadata['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
            elif stream['codec_type'] == 'audio':
                audio_stream = stream
        
        if not video_stream:
            raise VideoCompilationError("No video stream found")
        
        # Calculate quality metrics
        format_info = metadata['format']
        
        analysis = {
            'duration': float(format_info.get('duration', 0)),
            'file_size_mb': float(format_info.get('size', 0)) / (1024 * 1024),
            'bitrate_kbps': int(format_info.get('bit_rate', 0)) // 1000,
            'resolution': f"{video_stream['width']}x{video_stream['height']}",
            'fps': eval(video_stream.get('r_frame_rate', '30/1')),
            'codec': video_stream.get('codec_name'),
            'pixel_format': video_stream.get('pix_fmt'),
            'quality_score': calculate_video_quality_score(metadata),
            'has_audio': audio_stream is not None,
            'audio_codec': audio_stream.get('codec_name') if audio_stream else None,
            'audio_bitrate_kbps': int(audio_stream.get('bit_rate', 0)) // 1000 if audio_stream else 0
        }
        
        return analysis
        
    except subprocess.TimeoutExpired:
        raise VideoCompilationError("Video analysis timeout")
    except json.JSONDecodeError:
        raise VideoCompilationError("Failed to parse video metadata")
    except Exception as e:
        logger.error(f"Video analysis failed: {str(e)}")
        return {
            'duration': 0,
            'file_size_mb': 0,
            'quality_score': 50,
            'resolution': '1920x1080',
            'error': str(e)
        }


def calculate_video_quality_score(metadata: Dict[str, Any]) -> float:
    """Calculate overall video quality score."""
    
    try:
        score = 70.0  # Base score
        
        format_info = metadata.get('format', {})
        video_stream = None
        
        for stream in metadata.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            return score
        
        # Resolution score
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        
        if width >= 1920 and height >= 1080:
            score += 15
        elif width >= 1280 and height >= 720:
            score += 10
        elif width < 720:
            score -= 10
        
        # Bitrate score
        bitrate = int(format_info.get('bit_rate', 0)) // 1000  # Convert to kbps
        
        if bitrate >= 5000:  # Good quality
            score += 10
        elif bitrate >= 2000:  # Acceptable quality
            score += 5
        elif bitrate < 1000:  # Low quality
            score -= 15
        
        # FPS score
        fps_str = video_stream.get('r_frame_rate', '30/1')
        try:
            fps = eval(fps_str)
            if fps >= 30:
                score += 5
            elif fps < 24:
                score -= 5
        except:
            pass
        
        return min(100, max(0, score))
        
    except Exception:
        return 70.0


def calculate_compilation_cost(video_metadata: Dict[str, Any]) -> float:
    """Calculate cost for video compilation based on processing complexity."""
    
    try:
        base_cost = 0.15  # Base compilation cost
        
        # Duration-based cost
        duration = video_metadata.get('duration', 0)
        duration_cost = (duration / 60) * 0.05  # $0.05 per minute
        
        # Resolution-based cost
        resolution = video_metadata.get('resolution', '1920x1080')
        width, height = map(int, resolution.split('x'))
        pixel_count = width * height
        
        if pixel_count >= 1920 * 1080:  # 1080p+
            resolution_cost = 0.10
        elif pixel_count >= 1280 * 720:  # 720p
            resolution_cost = 0.05
        else:
            resolution_cost = 0.02
        
        # File size penalty for very large files
        file_size_mb = video_metadata.get('file_size_mb', 0)
        size_penalty = max(0, (file_size_mb - 200) * 0.001)  # Small penalty for files > 200MB
        
        total_cost = base_cost + duration_cost + resolution_cost + size_penalty
        
        return round(total_cost, 4)
        
    except Exception:
        return 0.25  # Default cost


# Helper functions

def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            metadata = json.loads(result.stdout)
            return float(metadata['format']['duration'])
        else:
            # Fallback: estimate from file size (rough approximation)
            file_size = os.path.getsize(audio_path)
            estimated_duration = file_size / (192 * 1000 / 8)  # Assume 192kbps
            return estimated_duration
            
    except Exception as e:
        logger.warning(f"Failed to get audio duration: {str(e)}")
        return 300.0  # Default 5 minutes


def split_script_into_segments(script: str, max_segments: int = 20) -> List[str]:
    """Split script into segments for timeline generation."""
    
    # Split by sentences first
    import re
    sentences = re.split(r'(?<=[.!?])\s+', script)
    
    if len(sentences) <= max_segments:
        return sentences
    
    # Group sentences into segments
    sentences_per_segment = len(sentences) // max_segments
    segments = []
    
    for i in range(0, len(sentences), sentences_per_segment):
        segment = ' '.join(sentences[i:i + sentences_per_segment])
        segments.append(segment)
        
        if len(segments) >= max_segments:
            break
    
    return segments


def cleanup_compilation_workspace(workspace_dir: Path, keep_finals: bool = True) -> None:
    """Clean up compilation workspace, optionally keeping final files."""
    
    try:
        if keep_finals:
            # Only cleanup temp and intermediate files
            import shutil
            
            temp_dir = workspace_dir / "temp"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            sequences_dir = workspace_dir / "sequences"
            if sequences_dir.exists():
                shutil.rmtree(sequences_dir)
            
            assets_dir = workspace_dir / "assets"
            if assets_dir.exists():
                shutil.rmtree(assets_dir)
        else:
            # Remove entire workspace
            import shutil
            shutil.rmtree(workspace_dir)
            
    except Exception as e:
        logger.warning(f"Failed to cleanup workspace: {str(e)}")
        # Don't raise error for cleanup failures