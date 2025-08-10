"""
AI-Powered Thumbnail Generation System
Automated thumbnail creation using DALL-E, Stable Diffusion, and template-based approaches
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import openai
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import base64
import io
import os
from datetime import datetime
import redis.asyncio as redis
import logging
from prometheus_client import Histogram, Counter, Gauge

# Metrics
thumbnail_generation_time = Histogram('thumbnail_generation_duration', 'Time to generate thumbnail', ['method'])
thumbnails_generated = Counter('thumbnails_generated_total', 'Total thumbnails generated', ['style'])
generation_errors = Counter('thumbnail_generation_errors', 'Thumbnail generation errors', ['error_type'])
cache_hits = Counter('thumbnail_cache_hits', 'Thumbnail cache hits')

logger = logging.getLogger(__name__)

@dataclass
class ThumbnailRequest:
    """Thumbnail generation request"""
    video_id: str
    title: str
    description: str
    style: str  # 'realistic', 'cartoon', 'minimalist', 'dramatic', 'tech', 'gaming'
    color_scheme: Optional[List[str]] = None
    include_text: bool = True
    include_face: bool = False
    custom_prompt: Optional[str] = None
    resolution: Tuple[int, int] = (1280, 720)
    brand_overlay: bool = True

@dataclass
class GeneratedThumbnail:
    """Generated thumbnail with metadata"""
    image: Image.Image
    video_id: str
    style: str
    generation_method: str
    generation_time: float
    quality_score: float
    file_path: Optional[str] = None
    metadata: Dict = None

class ThumbnailGenerator:
    """Advanced AI-powered thumbnail generation system"""
    
    def __init__(self, 
                 openai_api_key: str = None,
                 stable_diffusion_model: str = "stabilityai/stable-diffusion-2-1",
                 redis_host: str = 'localhost',
                 redis_port: int = 6379):
        
        # Initialize OpenAI
        if openai_api_key:
            openai.api_key = openai_api_key
        
        # Initialize Stable Diffusion
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stable_diffusion = None
        self.sd_model_id = stable_diffusion_model
        
        # Redis for caching
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        
        # Template library
        self.templates = self._load_templates()
        
        # Font library for text overlay
        self.fonts = self._load_fonts()
        
        # Style configurations
        self.style_configs = {
            'realistic': {
                'prompt_prefix': 'photorealistic, high quality, professional photography',
                'negative_prompt': 'cartoon, anime, illustration, low quality',
                'guidance_scale': 7.5
            },
            'cartoon': {
                'prompt_prefix': 'cartoon style, animated, colorful, vibrant',
                'negative_prompt': 'realistic, photograph, dark, gloomy',
                'guidance_scale': 10
            },
            'minimalist': {
                'prompt_prefix': 'minimalist, clean, simple, modern design',
                'negative_prompt': 'cluttered, complex, busy, detailed',
                'guidance_scale': 8
            },
            'dramatic': {
                'prompt_prefix': 'dramatic lighting, cinematic, epic, intense',
                'negative_prompt': 'boring, flat, simple, mundane',
                'guidance_scale': 9
            },
            'tech': {
                'prompt_prefix': 'technology, futuristic, digital, cyber, neon',
                'negative_prompt': 'vintage, old, natural, organic',
                'guidance_scale': 8
            },
            'gaming': {
                'prompt_prefix': 'gaming, video game style, vibrant colors, action-packed',
                'negative_prompt': 'boring, realistic, mundane, simple',
                'guidance_scale': 9
            }
        }
    
    async def initialize(self):
        """Initialize async components"""
        # Initialize Redis
        self.redis_client = await redis.Redis(
            host=self.redis_host,
            port=self.redis_port
        )
        
        # Load Stable Diffusion model
        await self._load_stable_diffusion()
    
    async def _load_stable_diffusion(self):
        """Load Stable Diffusion model"""
        try:
            self.stable_diffusion = StableDiffusionPipeline.from_pretrained(
                self.sd_model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.stable_diffusion.scheduler = DPMSolverMultistepScheduler.from_config(
                self.stable_diffusion.scheduler.config
            )
            self.stable_diffusion = self.stable_diffusion.to(self.device)
            
            # Enable memory efficient attention
            if hasattr(self.stable_diffusion, 'enable_xformers_memory_efficient_attention'):
                self.stable_diffusion.enable_xformers_memory_efficient_attention()
                
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion: {e}")
            self.stable_diffusion = None
    
    async def generate_thumbnail(self, request: ThumbnailRequest) -> GeneratedThumbnail:
        """
        Generate thumbnail using the most appropriate method
        
        Args:
            request: ThumbnailRequest with generation parameters
            
        Returns:
            GeneratedThumbnail object with the generated image
        """
        start_time = datetime.utcnow()
        
        # Check cache first
        cached_thumbnail = await self._get_cached_thumbnail(request)
        if cached_thumbnail:
            cache_hits.inc()
            return cached_thumbnail
        
        # Choose generation method based on availability and request
        if request.custom_prompt and self.stable_diffusion:
            thumbnail = await self._generate_with_stable_diffusion(request)
        elif request.custom_prompt and openai.api_key:
            thumbnail = await self._generate_with_dalle(request)
        elif request.style in ['minimalist', 'tech']:
            thumbnail = await self._generate_template_based(request)
        else:
            thumbnail = await self._generate_hybrid(request)
        
        # Add text overlay if requested
        if request.include_text:
            thumbnail.image = await self._add_text_overlay(thumbnail.image, request)
        
        # Add brand overlay if requested
        if request.brand_overlay:
            thumbnail.image = await self._add_brand_overlay(thumbnail.image)
        
        # Apply post-processing
        thumbnail.image = await self._apply_post_processing(thumbnail.image, request.style)
        
        # Calculate quality score
        thumbnail.quality_score = await self._calculate_quality_score(thumbnail.image)
        
        # Calculate generation time
        thumbnail.generation_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Cache the result
        await self._cache_thumbnail(request, thumbnail)
        
        # Update metrics
        thumbnail_generation_time.labels(method=thumbnail.generation_method).observe(thumbnail.generation_time)
        thumbnails_generated.labels(style=request.style).inc()
        
        return thumbnail
    
    async def _generate_with_dalle(self, request: ThumbnailRequest) -> GeneratedThumbnail:
        """Generate thumbnail using DALL-E 3"""
        try:
            # Construct prompt
            prompt = self._construct_dalle_prompt(request)
            
            # Generate image
            response = await asyncio.to_thread(
                openai.Image.create,
                model="dall-e-3",
                prompt=prompt,
                size=f"{request.resolution[0]}x{request.resolution[1]}",
                quality="hd",
                n=1
            )
            
            # Download and convert image
            image_url = response['data'][0]['url']
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    image_data = await resp.read()
            
            image = Image.open(io.BytesIO(image_data))
            
            return GeneratedThumbnail(
                image=image,
                video_id=request.video_id,
                style=request.style,
                generation_method="dall-e-3",
                generation_time=0,
                quality_score=0,
                metadata={'prompt': prompt}
            )
            
        except Exception as e:
            logger.error(f"DALL-E generation failed: {e}")
            generation_errors.labels(error_type='dall-e').inc()
            # Fallback to template-based
            return await self._generate_template_based(request)
    
    async def _generate_with_stable_diffusion(self, request: ThumbnailRequest) -> GeneratedThumbnail:
        """Generate thumbnail using Stable Diffusion"""
        try:
            # Construct prompt
            prompt = self._construct_sd_prompt(request)
            style_config = self.style_configs.get(request.style, self.style_configs['realistic'])
            
            # Generate image
            with torch.no_grad():
                image = self.stable_diffusion(
                    prompt=prompt,
                    negative_prompt=style_config['negative_prompt'],
                    height=request.resolution[1],
                    width=request.resolution[0],
                    guidance_scale=style_config['guidance_scale'],
                    num_inference_steps=30
                ).images[0]
            
            return GeneratedThumbnail(
                image=image,
                video_id=request.video_id,
                style=request.style,
                generation_method="stable-diffusion",
                generation_time=0,
                quality_score=0,
                metadata={'prompt': prompt}
            )
            
        except Exception as e:
            logger.error(f"Stable Diffusion generation failed: {e}")
            generation_errors.labels(error_type='stable-diffusion').inc()
            # Fallback to template-based
            return await self._generate_template_based(request)
    
    async def _generate_template_based(self, request: ThumbnailRequest) -> GeneratedThumbnail:
        """Generate thumbnail using templates and procedural graphics"""
        # Create base image
        image = Image.new('RGB', request.resolution, color='white')
        draw = ImageDraw.Draw(image)
        
        # Apply style-specific template
        if request.style == 'minimalist':
            image = self._create_minimalist_template(image, draw, request)
        elif request.style == 'tech':
            image = self._create_tech_template(image, draw, request)
        elif request.style == 'gaming':
            image = self._create_gaming_template(image, draw, request)
        else:
            image = self._create_default_template(image, draw, request)
        
        return GeneratedThumbnail(
            image=image,
            video_id=request.video_id,
            style=request.style,
            generation_method="template",
            generation_time=0,
            quality_score=0,
            metadata={'template': request.style}
        )
    
    async def _generate_hybrid(self, request: ThumbnailRequest) -> GeneratedThumbnail:
        """Generate thumbnail using hybrid approach (AI + templates)"""
        # Generate base with AI if available
        if self.stable_diffusion:
            base_thumbnail = await self._generate_with_stable_diffusion(request)
        else:
            base_thumbnail = await self._generate_template_based(request)
        
        # Enhance with template elements
        image = base_thumbnail.image
        
        # Add gradient overlay
        gradient = self._create_gradient_overlay(image.size, request.color_scheme)
        image = Image.blend(image, gradient, alpha=0.3)
        
        # Add geometric shapes
        image = self._add_geometric_elements(image, request.style)
        
        base_thumbnail.image = image
        base_thumbnail.generation_method = "hybrid"
        
        return base_thumbnail
    
    def _construct_dalle_prompt(self, request: ThumbnailRequest) -> str:
        """Construct prompt for DALL-E"""
        if request.custom_prompt:
            base_prompt = request.custom_prompt
        else:
            base_prompt = f"YouTube thumbnail for video titled '{request.title}'"
        
        style_config = self.style_configs.get(request.style, {})
        prompt_prefix = style_config.get('prompt_prefix', '')
        
        # Add style modifiers
        prompt = f"{prompt_prefix}, {base_prompt}"
        
        # Add color scheme if specified
        if request.color_scheme:
            colors = ', '.join(request.color_scheme)
            prompt += f", color palette: {colors}"
        
        # Add composition hints
        prompt += ", centered composition, eye-catching, high contrast"
        
        if request.include_face:
            prompt += ", include human face with engaging expression"
        
        return prompt
    
    def _construct_sd_prompt(self, request: ThumbnailRequest) -> str:
        """Construct prompt for Stable Diffusion"""
        if request.custom_prompt:
            base_prompt = request.custom_prompt
        else:
            # Extract key concepts from title
            keywords = self._extract_keywords(request.title)
            base_prompt = f"thumbnail featuring {', '.join(keywords)}"
        
        style_config = self.style_configs.get(request.style, {})
        prompt_prefix = style_config.get('prompt_prefix', '')
        
        # Construct full prompt
        prompt = f"{prompt_prefix}, {base_prompt}, YouTube thumbnail style"
        
        if request.color_scheme:
            colors = ', '.join(request.color_scheme)
            prompt += f", {colors} color scheme"
        
        return prompt
    
    def _extract_keywords(self, title: str) -> List[str]:
        """Extract keywords from title for prompt generation"""
        # Simple keyword extraction
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = title.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return keywords[:5]  # Top 5 keywords
    
    def _create_minimalist_template(self, image: Image.Image, draw: ImageDraw.Draw, 
                                   request: ThumbnailRequest) -> Image.Image:
        """Create minimalist style template"""
        width, height = image.size
        
        # Background gradient
        colors = request.color_scheme or ['#ffffff', '#f0f0f0']
        for y in range(height):
            color_ratio = y / height
            r = int(self._hex_to_rgb(colors[0])[0] * (1 - color_ratio) + 
                   self._hex_to_rgb(colors[1])[0] * color_ratio)
            g = int(self._hex_to_rgb(colors[0])[1] * (1 - color_ratio) + 
                   self._hex_to_rgb(colors[1])[1] * color_ratio)
            b = int(self._hex_to_rgb(colors[0])[2] * (1 - color_ratio) + 
                   self._hex_to_rgb(colors[1])[2] * color_ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        # Add geometric shape
        shape_color = request.color_scheme[2] if len(request.color_scheme) > 2 else '#333333'
        draw.ellipse(
            [(width//4, height//4), (3*width//4, 3*height//4)],
            outline=self._hex_to_rgb(shape_color),
            width=5
        )
        
        return image
    
    def _create_tech_template(self, image: Image.Image, draw: ImageDraw.Draw,
                             request: ThumbnailRequest) -> Image.Image:
        """Create tech style template"""
        width, height = image.size
        
        # Dark background
        image = Image.new('RGB', (width, height), color='#0a0a0a')
        draw = ImageDraw.Draw(image)
        
        # Add grid pattern
        grid_color = '#1a1a1a'
        for x in range(0, width, 50):
            draw.line([(x, 0), (x, height)], fill=grid_color)
        for y in range(0, height, 50):
            draw.line([(0, y), (width, y)], fill=grid_color)
        
        # Add neon accent
        accent_color = request.color_scheme[0] if request.color_scheme else '#00ff88'
        draw.rectangle(
            [(50, height//3), (width-50, 2*height//3)],
            outline=accent_color,
            width=3
        )
        
        # Add glitch effect lines
        for _ in range(5):
            y = np.random.randint(0, height)
            width_offset = np.random.randint(-20, 20)
            draw.line(
                [(0, y), (width + width_offset, y)],
                fill=accent_color,
                width=1
            )
        
        return image
    
    def _create_gaming_template(self, image: Image.Image, draw: ImageDraw.Draw,
                               request: ThumbnailRequest) -> Image.Image:
        """Create gaming style template"""
        width, height = image.size
        
        # Gradient background
        colors = request.color_scheme or ['#ff0066', '#6600ff']
        gradient = self._create_gradient_overlay((width, height), colors)
        image = gradient
        draw = ImageDraw.Draw(image)
        
        # Add explosion/burst effect
        center_x, center_y = width // 2, height // 2
        for i in range(20):
            angle = (i * 18) * np.pi / 180
            length = np.random.randint(100, 200)
            end_x = center_x + int(length * np.cos(angle))
            end_y = center_y + int(length * np.sin(angle))
            draw.line(
                [(center_x, center_y), (end_x, end_y)],
                fill=(255, 255, 0, 128),
                width=3
            )
        
        return image
    
    def _create_default_template(self, image: Image.Image, draw: ImageDraw.Draw,
                                request: ThumbnailRequest) -> Image.Image:
        """Create default template"""
        width, height = image.size
        
        # Simple gradient background
        colors = request.color_scheme or ['#4a90e2', '#7b68ee']
        gradient = self._create_gradient_overlay((width, height), colors)
        
        return gradient
    
    def _create_gradient_overlay(self, size: Tuple[int, int], colors: List[str]) -> Image.Image:
        """Create gradient overlay"""
        width, height = size
        gradient = Image.new('RGBA', size)
        draw = ImageDraw.Draw(gradient)
        
        if len(colors) < 2:
            colors = [colors[0], colors[0]]
        
        color1 = self._hex_to_rgb(colors[0])
        color2 = self._hex_to_rgb(colors[1])
        
        for y in range(height):
            ratio = y / height
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b, 200))
        
        return gradient
    
    def _add_geometric_elements(self, image: Image.Image, style: str) -> Image.Image:
        """Add geometric elements based on style"""
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        width, height = image.size
        
        if style in ['tech', 'gaming']:
            # Add triangles
            points = [
                (width - 100, 50),
                (width - 50, 100),
                (width - 150, 100)
            ]
            draw.polygon(points, fill=(255, 255, 255, 100))
        
        if style == 'minimalist':
            # Add circles
            draw.ellipse(
                [(50, 50), (150, 150)],
                outline=(0, 0, 0, 150),
                width=3
            )
        
        # Composite overlay onto image
        image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        
        return image
    
    async def _add_text_overlay(self, image: Image.Image, request: ThumbnailRequest) -> Image.Image:
        """Add text overlay to thumbnail"""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Prepare text
        title_text = self._prepare_title_text(request.title)
        
        # Select font
        font_size = self._calculate_font_size(title_text, width, height)
        font = self._get_font(request.style, font_size)
        
        # Calculate text position
        text_bbox = draw.textbbox((0, 0), title_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Add text shadow
        shadow_offset = 3
        draw.text(
            (x + shadow_offset, y + shadow_offset),
            title_text,
            font=font,
            fill=(0, 0, 0, 128)
        )
        
        # Add main text
        text_color = self._get_text_color(image, x, y, text_width, text_height)
        draw.text((x, y), title_text, font=font, fill=text_color)
        
        # Add stroke/outline for better visibility
        for adj_x in [-1, 0, 1]:
            for adj_y in [-1, 0, 1]:
                if adj_x != 0 or adj_y != 0:
                    draw.text(
                        (x + adj_x, y + adj_y),
                        title_text,
                        font=font,
                        fill=(0, 0, 0, 255)
                    )
        draw.text((x, y), title_text, font=font, fill=text_color)
        
        return image
    
    async def _add_brand_overlay(self, image: Image.Image) -> Image.Image:
        """Add brand logo/watermark"""
        # Load brand logo (placeholder)
        try:
            logo_path = "assets/brand_logo.png"
            if os.path.exists(logo_path):
                logo = Image.open(logo_path).convert('RGBA')
                
                # Resize logo
                logo_size = (100, 50)
                logo = logo.resize(logo_size, Image.Resampling.LANCZOS)
                
                # Position logo
                width, height = image.size
                position = (width - logo_size[0] - 20, height - logo_size[1] - 20)
                
                # Composite logo onto image
                image = image.convert('RGBA')
                image.paste(logo, position, logo)
                image = image.convert('RGB')
        except Exception as e:
            logger.debug(f"Could not add brand overlay: {e}")
        
        return image
    
    async def _apply_post_processing(self, image: Image.Image, style: str) -> Image.Image:
        """Apply post-processing effects"""
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Enhance saturation for certain styles
        if style in ['gaming', 'cartoon']:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.3)
        
        # Add slight blur for dramatic effect
        if style == 'dramatic':
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
            
            # Add vignette
            image = self._add_vignette(image)
        
        # Sharpen for tech/minimalist
        if style in ['tech', 'minimalist']:
            image = image.filter(ImageFilter.SHARPEN)
        
        return image
    
    def _add_vignette(self, image: Image.Image) -> Image.Image:
        """Add vignette effect"""
        width, height = image.size
        
        # Create vignette mask
        vignette = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(vignette)
        
        # Draw gradient ellipse
        for i in range(min(width, height) // 2):
            alpha = int(255 * (1 - i / (min(width, height) / 2)) ** 2)
            draw.ellipse(
                [(i, i), (width - i, height - i)],
                outline=alpha
            )
        
        # Apply vignette
        black = Image.new('RGB', (width, height), (0, 0, 0))
        image = Image.composite(image, black, vignette)
        
        return image
    
    async def _calculate_quality_score(self, image: Image.Image) -> float:
        """Calculate thumbnail quality score"""
        score = 0.0
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Check resolution
        height, width = img_array.shape[:2]
        if width >= 1280 and height >= 720:
            score += 0.2
        
        # Check contrast
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        contrast = gray.std()
        if contrast > 50:
            score += 0.2
        
        # Check color distribution
        hist = cv2.calcHist([img_array], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = hist.flatten()
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        if entropy > 4:
            score += 0.2
        
        # Check edge density (detail)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / (width * height)
        if 0.05 <= edge_density <= 0.2:
            score += 0.2
        
        # Check for face (bonus)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            score += 0.2
        
        return min(1.0, score)
    
    def _prepare_title_text(self, title: str) -> str:
        """Prepare title text for overlay"""
        # Truncate if too long
        max_chars = 40
        if len(title) > max_chars:
            title = title[:max_chars-3] + "..."
        
        # Add line breaks for better layout
        words = title.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 20:
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def _calculate_font_size(self, text: str, width: int, height: int) -> int:
        """Calculate optimal font size"""
        # Base size on image dimensions
        base_size = min(width, height) // 10
        
        # Adjust based on text length
        text_length = len(text)
        if text_length > 30:
            base_size = int(base_size * 0.8)
        elif text_length < 15:
            base_size = int(base_size * 1.2)
        
        return max(24, min(base_size, 120))
    
    def _get_font(self, style: str, size: int) -> ImageFont.FreeTypeFont:
        """Get font based on style"""
        font_map = {
            'tech': 'fonts/tech.ttf',
            'gaming': 'fonts/gaming.ttf',
            'minimalist': 'fonts/minimal.ttf',
            'dramatic': 'fonts/bold.ttf'
        }
        
        font_path = font_map.get(style, 'fonts/default.ttf')
        
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
        except:
            pass
        
        # Fallback to default font
        return ImageFont.load_default()
    
    def _get_text_color(self, image: Image.Image, x: int, y: int, 
                       width: int, height: int) -> Tuple[int, int, int]:
        """Determine optimal text color based on background"""
        # Sample background area
        crop = image.crop((x, y, x + width, y + height))
        
        # Calculate average brightness
        gray = crop.convert('L')
        brightness = np.array(gray).mean()
        
        # Return white for dark backgrounds, black for light
        if brightness < 128:
            return (255, 255, 255)
        else:
            return (0, 0, 0)
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _load_templates(self) -> Dict:
        """Load template configurations"""
        # In production, load from file or database
        return {
            'default': {},
            'tech': {'grid': True, 'neon': True},
            'gaming': {'burst': True, 'vibrant': True},
            'minimalist': {'clean': True, 'geometric': True}
        }
    
    def _load_fonts(self) -> Dict:
        """Load font library"""
        # In production, load actual font files
        return {
            'default': 'Arial',
            'tech': 'Orbitron',
            'gaming': 'Press Start 2P',
            'minimalist': 'Helvetica Neue'
        }
    
    async def _get_cached_thumbnail(self, request: ThumbnailRequest) -> Optional[GeneratedThumbnail]:
        """Get cached thumbnail if available"""
        if not self.redis_client:
            return None
        
        cache_key = f"thumbnail:{request.video_id}:{request.style}"
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                # Decode base64 image
                image_data = base64.b64decode(cached_data)
                image = Image.open(io.BytesIO(image_data))
                
                return GeneratedThumbnail(
                    image=image,
                    video_id=request.video_id,
                    style=request.style,
                    generation_method="cached",
                    generation_time=0,
                    quality_score=1.0
                )
        except Exception as e:
            logger.error(f"Error retrieving cached thumbnail: {e}")
        
        return None
    
    async def _cache_thumbnail(self, request: ThumbnailRequest, thumbnail: GeneratedThumbnail):
        """Cache generated thumbnail"""
        if not self.redis_client:
            return
        
        cache_key = f"thumbnail:{request.video_id}:{request.style}"
        
        try:
            # Convert image to base64
            buffer = io.BytesIO()
            thumbnail.image.save(buffer, format='PNG')
            image_data = base64.b64encode(buffer.getvalue()).decode()
            
            # Cache for 24 hours
            await self.redis_client.setex(cache_key, 86400, image_data)
        except Exception as e:
            logger.error(f"Error caching thumbnail: {e}")
    
    async def save_thumbnail(self, thumbnail: GeneratedThumbnail, output_path: str) -> str:
        """Save thumbnail to file"""
        try:
            thumbnail.image.save(output_path, 'PNG', optimize=True)
            thumbnail.file_path = output_path
            return output_path
        except Exception as e:
            logger.error(f"Error saving thumbnail: {e}")
            raise

# Example usage
async def main():
    # Initialize generator
    generator = ThumbnailGenerator(
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    await generator.initialize()
    
    # Create thumbnail request
    request = ThumbnailRequest(
        video_id="test_001",
        title="10 Python Tips That Will Blow Your Mind",
        description="Amazing Python programming tips and tricks",
        style="tech",
        color_scheme=["#00ff88", "#0088ff", "#ff0088"],
        include_text=True,
        include_face=False
    )
    
    # Generate thumbnail
    thumbnail = await generator.generate_thumbnail(request)
    
    # Save thumbnail
    output_path = f"thumbnails/{request.video_id}.png"
    await generator.save_thumbnail(thumbnail, output_path)
    
    print(f"Thumbnail generated: {output_path}")
    print(f"Quality score: {thumbnail.quality_score:.2f}")
    print(f"Generation time: {thumbnail.generation_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())