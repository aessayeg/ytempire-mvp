"""
Thumbnail Generation Module for YouTube Videos
Uses DALL-E 3 and custom optimization for high CTR thumbnails
"""
import asyncio
import base64
import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import openai
import redis
import hashlib
import aiohttp
from io import BytesIO
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThumbnailStyle(Enum):
    """Thumbnail visual styles"""
    MODERN = "modern"
    MINIMALIST = "minimalist"
    BOLD = "bold"
    CINEMATIC = "cinematic"
    CARTOON = "cartoon"
    REALISTIC = "realistic"
    ABSTRACT = "abstract"
    VINTAGE = "vintage"

class TextPosition(Enum):
    """Text overlay positions"""
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER = "center"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"

@dataclass
class ThumbnailConfig:
    """Thumbnail generation configuration"""
    style: ThumbnailStyle
    title: str
    subtitle: Optional[str]
    color_scheme: List[str]
    include_face: bool
    include_text: bool
    text_position: TextPosition
    brand_logo: Optional[str]
    aspect_ratio: str = "16:9"
    resolution: Tuple[int, int] = (1280, 720)

@dataclass
class ThumbnailResult:
    """Generated thumbnail result"""
    image_path: str
    thumbnail_url: str
    variations: List[str]
    click_through_score: float
    generation_cost: float
    metadata: Dict[str, Any]

class ThumbnailGenerator:
    """
    Advanced thumbnail generation with CTR optimization
    """
    
    def __init__(
        self,
        openai_api_key: str,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        assets_dir: str = "assets/thumbnails"
    ):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.cache_ttl = 86400 * 7  # 7 days
        self.assets_dir = assets_dir
        
        # Create assets directory
        os.makedirs(assets_dir, exist_ok=True)
        os.makedirs(f"{assets_dir}/fonts", exist_ok=True)
        os.makedirs(f"{assets_dir}/templates", exist_ok=True)
        os.makedirs(f"{assets_dir}/generated", exist_ok=True)
        
        # CTR optimization patterns
        self.high_ctr_patterns = {
            'colors': {
                'gaming': ['#FF0000', '#000000', '#FFFFFF', '#00FF00'],
                'tech': ['#0080FF', '#000000', '#FFFFFF', '#FF6B00'],
                'education': ['#4CAF50', '#2196F3', '#FFFFFF', '#FFC107'],
                'entertainment': ['#E91E63', '#9C27B0', '#FFEB3B', '#00BCD4']
            },
            'elements': {
                'faces': True,  # Human faces increase CTR by 38%
                'arrows': True,  # Arrows and pointers increase CTR by 22%
                'numbers': True,  # Numbers increase CTR by 15%
                'contrast': True  # High contrast increases CTR by 27%
            }
        }
    
    async def generate_thumbnail(
        self,
        config: ThumbnailConfig,
        video_context: Dict[str, Any],
        quality_preset: str = 'balanced'
    ) -> ThumbnailResult:
        """
        Generate optimized thumbnail for video
        
        Args:
            config: Thumbnail configuration
            video_context: Video metadata and context
            quality_preset: Quality/cost tradeoff
        
        Returns:
            Generated thumbnail with metadata
        """
        # Check cache
        cache_key = self._generate_cache_key(config, video_context)
        cached_result = self._get_cached_thumbnail(cache_key)
        if cached_result:
            logger.info("Using cached thumbnail")
            return cached_result
        
        # Generate base image with DALL-E 3
        base_image = await self._generate_base_image(config, video_context, quality_preset)
        
        # Apply optimizations
        optimized_image = self._optimize_for_ctr(base_image, config, video_context)
        
        # Add text overlays
        if config.include_text:
            optimized_image = self._add_text_overlay(optimized_image, config)
        
        # Add brand elements
        if config.brand_logo:
            optimized_image = self._add_branding(optimized_image, config.brand_logo)
        
        # Generate variations for A/B testing
        variations = await self._generate_variations(optimized_image, config)
        
        # Calculate CTR score
        ctr_score = self._calculate_ctr_score(optimized_image, config, video_context)
        
        # Save images
        output_path = f"{self.assets_dir}/generated/{cache_key}.jpg"
        optimized_image.save(output_path, 'JPEG', quality=95, optimize=True)
        
        # Calculate cost
        generation_cost = self._calculate_cost(quality_preset, len(variations))
        
        # Create result
        result = ThumbnailResult(
            image_path=output_path,
            thumbnail_url=f"/thumbnails/{os.path.basename(output_path)}",
            variations=[f"/thumbnails/{os.path.basename(v)}" for v in variations],
            click_through_score=ctr_score,
            generation_cost=generation_cost,
            metadata={
                'style': config.style.value,
                'resolution': config.resolution,
                'has_face': config.include_face,
                'has_text': config.include_text,
                'quality_preset': quality_preset
            }
        )
        
        # Cache result
        self._cache_thumbnail(cache_key, result)
        
        return result
    
    async def _generate_base_image(
        self,
        config: ThumbnailConfig,
        video_context: Dict[str, Any],
        quality_preset: str
    ) -> Image.Image:
        """Generate base image using DALL-E 3"""
        
        # Build prompt for DALL-E
        prompt = self._build_dalle_prompt(config, video_context)
        
        try:
            # Use DALL-E 3 for high quality
            if quality_preset == 'quality':
                response = await asyncio.to_thread(
                    openai.Image.create,
                    model="dall-e-3",
                    prompt=prompt,
                    size="1792x1024",  # HD quality
                    quality="hd",
                    n=1
                )
            else:
                # Use DALL-E 2 for faster/cheaper generation
                response = await asyncio.to_thread(
                    openai.Image.create,
                    model="dall-e-2",
                    prompt=prompt,
                    size="1024x1024",
                    n=1
                )
            
            # Download and convert image
            image_url = response['data'][0]['url']
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    image_data = await resp.read()
                    image = Image.open(BytesIO(image_data))
            
            # Resize to target resolution
            image = image.resize(config.resolution, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"Error generating base image: {e}")
            # Fallback to template-based generation
            return self._generate_fallback_image(config)
    
    def _build_dalle_prompt(
        self,
        config: ThumbnailConfig,
        video_context: Dict[str, Any]
    ) -> str:
        """Build optimized DALL-E prompt"""
        
        style_descriptions = {
            ThumbnailStyle.MODERN: "modern, clean, professional",
            ThumbnailStyle.MINIMALIST: "minimalist, simple, elegant",
            ThumbnailStyle.BOLD: "bold, vibrant, eye-catching",
            ThumbnailStyle.CINEMATIC: "cinematic, dramatic, movie-poster style",
            ThumbnailStyle.CARTOON: "cartoon, animated, colorful",
            ThumbnailStyle.REALISTIC: "photorealistic, detailed, high-quality",
            ThumbnailStyle.ABSTRACT: "abstract, artistic, creative",
            ThumbnailStyle.VINTAGE: "vintage, retro, nostalgic"
        }
        
        # Base prompt
        prompt_parts = [
            f"YouTube thumbnail for video about {video_context.get('topic', 'content')}",
            style_descriptions.get(config.style, "professional"),
            f"Color scheme: {', '.join(config.color_scheme)}",
        ]
        
        # Add specific elements
        if config.include_face:
            prompt_parts.append("Include an expressive human face showing emotion")
        
        if video_context.get('category'):
            prompt_parts.append(f"Theme: {video_context['category']}")
        
        # Add CTR optimization elements
        prompt_parts.extend([
            "High contrast",
            "Eye-catching composition",
            "Professional quality",
            "Optimized for small display",
            "No text overlays"  # We'll add text separately
        ])
        
        return ", ".join(prompt_parts)
    
    def _generate_fallback_image(self, config: ThumbnailConfig) -> Image.Image:
        """Generate fallback template-based image"""
        
        # Create base image with gradient
        img = Image.new('RGB', config.resolution, color='white')
        draw = ImageDraw.Draw(img)
        
        # Create gradient background
        for i in range(config.resolution[1]):
            color_ratio = i / config.resolution[1]
            r = int(255 * (1 - color_ratio))
            g = int(100 * color_ratio)
            b = int(150)
            draw.rectangle([(0, i), (config.resolution[0], i+1)], fill=(r, g, b))
        
        return img
    
    def _optimize_for_ctr(
        self,
        image: Image.Image,
        config: ThumbnailConfig,
        video_context: Dict[str, Any]
    ) -> Image.Image:
        """Apply CTR optimization techniques"""
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)
        
        # Increase color saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.2)
        
        # Add subtle vignette effect
        image = self._add_vignette(image)
        
        # Add attention-grabbing elements
        if video_context.get('category') in self.high_ctr_patterns['colors']:
            image = self._apply_color_overlay(
                image,
                self.high_ctr_patterns['colors'][video_context['category']]
            )
        
        # Sharpen image for clarity
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
        
        return image
    
    def _add_vignette(self, image: Image.Image) -> Image.Image:
        """Add vignette effect to focus attention"""
        
        # Create vignette mask
        width, height = image.size
        mask = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(mask)
        
        # Draw gradient ellipse
        for i in range(min(width, height) // 2):
            alpha = int(255 * (1 - i / (min(width, height) / 2)) ** 2)
            draw.ellipse(
                [i, i, width-i, height-i],
                fill=alpha
            )
        
        # Apply vignette
        black = Image.new('RGB', (width, height), 'black')
        image = Image.composite(image, black, mask)
        
        return image
    
    def _apply_color_overlay(
        self,
        image: Image.Image,
        colors: List[str]
    ) -> Image.Image:
        """Apply strategic color overlay"""
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Create color overlay
        overlay = np.zeros_like(img_array)
        height, width = img_array.shape[:2]
        
        # Apply gradient with brand colors
        for i, color in enumerate(colors[:2]):
            # Convert hex to RGB
            rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            
            # Create gradient
            for y in range(height):
                alpha = (y / height) if i == 0 else (1 - y / height)
                overlay[y, :] += np.array(rgb) * alpha * 0.2
        
        # Blend with original
        result = img_array.astype(float) + overlay
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def _add_text_overlay(
        self,
        image: Image.Image,
        config: ThumbnailConfig
    ) -> Image.Image:
        """Add optimized text overlay"""
        
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Load or use default font
        try:
            title_font = ImageFont.truetype(f"{self.assets_dir}/fonts/bold.ttf", size=int(height * 0.08))
            subtitle_font = ImageFont.truetype(f"{self.assets_dir}/fonts/regular.ttf", size=int(height * 0.05))
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
        
        # Calculate text position
        text_positions = {
            TextPosition.TOP_LEFT: (width * 0.05, height * 0.05),
            TextPosition.TOP_CENTER: (width * 0.5, height * 0.05),
            TextPosition.TOP_RIGHT: (width * 0.95, height * 0.05),
            TextPosition.CENTER: (width * 0.5, height * 0.5),
            TextPosition.BOTTOM_LEFT: (width * 0.05, height * 0.85),
            TextPosition.BOTTOM_CENTER: (width * 0.5, height * 0.85),
            TextPosition.BOTTOM_RIGHT: (width * 0.95, height * 0.85)
        }
        
        position = text_positions[config.text_position]
        
        # Add text shadow for readability
        shadow_offset = 3
        
        # Draw title
        if config.title:
            # Shadow
            draw.text(
                (position[0] + shadow_offset, position[1] + shadow_offset),
                config.title,
                font=title_font,
                fill='black',
                anchor='mm' if 'center' in config.text_position.value else 'lt'
            )
            # Main text
            draw.text(
                position,
                config.title,
                font=title_font,
                fill='white',
                anchor='mm' if 'center' in config.text_position.value else 'lt',
                stroke_width=2,
                stroke_fill='black'
            )
        
        # Draw subtitle if present
        if config.subtitle:
            subtitle_pos = (position[0], position[1] + height * 0.1)
            # Shadow
            draw.text(
                (subtitle_pos[0] + shadow_offset, subtitle_pos[1] + shadow_offset),
                config.subtitle,
                font=subtitle_font,
                fill='black',
                anchor='mm' if 'center' in config.text_position.value else 'lt'
            )
            # Main text
            draw.text(
                subtitle_pos,
                config.subtitle,
                font=subtitle_font,
                fill='yellow',
                anchor='mm' if 'center' in config.text_position.value else 'lt'
            )
        
        return image
    
    def _add_branding(
        self,
        image: Image.Image,
        logo_path: str
    ) -> Image.Image:
        """Add brand logo to thumbnail"""
        
        try:
            # Load logo
            logo = Image.open(logo_path)
            
            # Resize logo to 10% of thumbnail width
            width, height = image.size
            logo_width = int(width * 0.1)
            logo_height = int(logo.height * (logo_width / logo.width))
            logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
            
            # Position in bottom right corner
            position = (width - logo_width - 20, height - logo_height - 20)
            
            # Paste logo with transparency
            if logo.mode == 'RGBA':
                image.paste(logo, position, logo)
            else:
                image.paste(logo, position)
            
        except Exception as e:
            logger.error(f"Error adding logo: {e}")
        
        return image
    
    async def _generate_variations(
        self,
        base_image: Image.Image,
        config: ThumbnailConfig
    ) -> List[str]:
        """Generate thumbnail variations for A/B testing"""
        
        variations = []
        
        # Variation 1: Different text position
        var1 = base_image.copy()
        alt_config = config
        alt_config.text_position = TextPosition.BOTTOM_CENTER
        var1 = self._add_text_overlay(var1, alt_config)
        var1_path = f"{self.assets_dir}/generated/var1_{hashlib.md5(str(config).encode()).hexdigest()}.jpg"
        var1.save(var1_path, 'JPEG', quality=95)
        variations.append(var1_path)
        
        # Variation 2: Higher contrast
        var2 = base_image.copy()
        enhancer = ImageEnhance.Contrast(var2)
        var2 = enhancer.enhance(1.5)
        var2_path = f"{self.assets_dir}/generated/var2_{hashlib.md5(str(config).encode()).hexdigest()}.jpg"
        var2.save(var2_path, 'JPEG', quality=95)
        variations.append(var2_path)
        
        # Variation 3: Different color scheme
        var3 = base_image.copy()
        enhancer = ImageEnhance.Color(var3)
        var3 = enhancer.enhance(0.8)
        var3_path = f"{self.assets_dir}/generated/var3_{hashlib.md5(str(config).encode()).hexdigest()}.jpg"
        var3.save(var3_path, 'JPEG', quality=95)
        variations.append(var3_path)
        
        return variations
    
    def _calculate_ctr_score(
        self,
        image: Image.Image,
        config: ThumbnailConfig,
        video_context: Dict[str, Any]
    ) -> float:
        """Calculate predicted CTR score (0-100)"""
        
        score = 50.0  # Base score
        
        # Face detection bonus
        if config.include_face:
            score += 15
        
        # High contrast bonus
        img_array = np.array(image.convert('L'))
        contrast = img_array.std()
        if contrast > 60:
            score += 10
        
        # Color vibrancy bonus
        img_array = np.array(image)
        saturation = np.std(img_array)
        if saturation > 70:
            score += 10
        
        # Text clarity bonus
        if config.include_text and config.title:
            if len(config.title) < 30:  # Short, punchy titles
                score += 5
        
        # Category optimization
        if video_context.get('category') in ['gaming', 'entertainment']:
            score += 5
        
        # Style bonus
        if config.style in [ThumbnailStyle.BOLD, ThumbnailStyle.CINEMATIC]:
            score += 5
        
        return min(score, 100.0)
    
    def _calculate_cost(self, quality_preset: str, num_variations: int) -> float:
        """Calculate generation cost"""
        
        base_costs = {
            'quality': 0.08,  # DALL-E 3 HD
            'balanced': 0.04,  # DALL-E 3 standard
            'fast': 0.02  # DALL-E 2
        }
        
        base_cost = base_costs.get(quality_preset, 0.04)
        variation_cost = 0.01 * num_variations  # Processing cost
        
        return base_cost + variation_cost
    
    def _generate_cache_key(
        self,
        config: ThumbnailConfig,
        video_context: Dict[str, Any]
    ) -> str:
        """Generate cache key for thumbnail"""
        
        key_data = {
            'title': config.title,
            'style': config.style.value,
            'topic': video_context.get('topic', ''),
            'category': video_context.get('category', '')
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_thumbnail(self, cache_key: str) -> Optional[ThumbnailResult]:
        """Retrieve cached thumbnail"""
        
        cached_data = self.redis_client.get(f"thumbnail:{cache_key}")
        if cached_data:
            try:
                data = json.loads(cached_data)
                return ThumbnailResult(**data)
            except Exception as e:
                logger.error(f"Error loading cached thumbnail: {e}")
        
        return None
    
    def _cache_thumbnail(self, cache_key: str, result: ThumbnailResult):
        """Cache thumbnail result"""
        
        try:
            result_dict = {
                'image_path': result.image_path,
                'thumbnail_url': result.thumbnail_url,
                'variations': result.variations,
                'click_through_score': result.click_through_score,
                'generation_cost': result.generation_cost,
                'metadata': result.metadata
            }
            
            self.redis_client.setex(
                f"thumbnail:{cache_key}",
                self.cache_ttl,
                json.dumps(result_dict)
            )
        except Exception as e:
            logger.error(f"Error caching thumbnail: {e}")


class ThumbnailAPI:
    """FastAPI integration for thumbnail generation"""
    
    def __init__(self, openai_api_key: str):
        self.generator = ThumbnailGenerator(openai_api_key)
    
    async def generate(
        self,
        title: str,
        style: str,
        video_topic: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate thumbnail via API"""
        
        config = ThumbnailConfig(
            style=ThumbnailStyle(style),
            title=title,
            subtitle=kwargs.get('subtitle'),
            color_scheme=kwargs.get('colors', ['#FF0000', '#000000', '#FFFFFF']),
            include_face=kwargs.get('include_face', True),
            include_text=kwargs.get('include_text', True),
            text_position=TextPosition(kwargs.get('text_position', 'bottom_center')),
            brand_logo=kwargs.get('logo_path')
        )
        
        video_context = {
            'topic': video_topic,
            'category': kwargs.get('category', 'general')
        }
        
        result = await self.generator.generate_thumbnail(
            config,
            video_context,
            kwargs.get('quality_preset', 'balanced')
        )
        
        return {
            'thumbnail_url': result.thumbnail_url,
            'variations': result.variations,
            'ctr_score': result.click_through_score,
            'cost': result.generation_cost,
            'metadata': result.metadata
        }


# Initialize global instance
thumbnail_api = ThumbnailAPI(openai_api_key="your-api-key-here")