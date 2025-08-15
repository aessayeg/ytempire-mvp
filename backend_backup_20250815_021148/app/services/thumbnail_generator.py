"""
Thumbnail Generator Service
Creates eye-catching thumbnails for YouTube videos
"""
import os
import asyncio
import aiohttp
from typing import Dict, Optional, Any
from pathlib import Path
import tempfile
import logging
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import io
import base64
from datetime import datetime

logger = logging.getLogger(__name__)


class ThumbnailGenerator:
    """Generate thumbnails using DALL-E or create programmatically"""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.temp_dir = Path(tempfile.gettempdir()) / "ytempire_thumbnails"
        self.temp_dir.mkdir(exist_ok=True)
        
        # YouTube thumbnail dimensions
        self.width = 1280
        self.height = 720
        
    async def generate(
        self,
        title: str,
        topic: str,
        style: str = "modern",
        use_ai: bool = True
    ) -> str:
        """
        Generate thumbnail for video
        Returns path to thumbnail file
        """
        if use_ai and self.openai_api_key:
            return await self._generate_with_dalle(title, topic, style)
        else:
            return await self._generate_programmatic(title, topic, style)
            
    async def _generate_with_dalle(self, title: str, topic: str, style: str) -> str:
        """Generate thumbnail using DALL-E 3"""
        try:
            prompt = self._create_dalle_prompt(title, topic, style)
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "dall-e-3",
                    "prompt": prompt,
                    "n": 1,
                    "size": "1792x1024",  # Closest to 16:9 ratio
                    "quality": "standard",
                    "response_format": "b64_json"
                }
                
                async with session.post(
                    "https://api.openai.com/v1/images/generations",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        image_b64 = result["data"][0]["b64_json"]
                        
                        # Save image
                        image_data = base64.b64decode(image_b64)
                        image = Image.open(io.BytesIO(image_data))
                        
                        # Resize to YouTube dimensions
                        image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
                        
                        # Save to file
                        output_path = self.temp_dir / f"thumbnail_{datetime.now().timestamp()}.jpg"
                        image.save(output_path, "JPEG", quality=95)
                        
                        logger.info(f"AI thumbnail generated: {output_path}")
                        return str(output_path)
                    else:
                        logger.error(f"DALL-E API error: {response.status}")
                        # Fall back to programmatic generation
                        return await self._generate_programmatic(title, topic, style)
                        
        except Exception as e:
            logger.error(f"DALL-E generation failed: {e}")
            # Fall back to programmatic generation
            return await self._generate_programmatic(title, topic, style)
            
    async def _generate_programmatic(self, title: str, topic: str, style: str) -> str:
        """Generate thumbnail programmatically using PIL"""
        # Create base image with gradient background
        image = Image.new('RGB', (self.width, self.height))
        draw = ImageDraw.Draw(image)
        
        # Create gradient background based on style
        colors = self._get_style_colors(style)
        self._draw_gradient(draw, colors["bg_start"], colors["bg_end"])
        
        # Add overlay pattern
        self._add_pattern_overlay(image, style)
        
        # Add title text
        self._add_title_text(image, title, colors)
        
        # Add decorative elements
        self._add_decorative_elements(image, style, colors)
        
        # Add subtle shadow/glow effects
        image = self._add_effects(image)
        
        # Save thumbnail
        output_path = self.temp_dir / f"thumbnail_{datetime.now().timestamp()}.jpg"
        image.save(output_path, "JPEG", quality=95)
        
        logger.info(f"Programmatic thumbnail generated: {output_path}")
        return str(output_path)
        
    def _create_dalle_prompt(self, title: str, topic: str, style: str) -> str:
        """Create optimized DALL-E prompt for thumbnail"""
        style_descriptions = {
            "modern": "modern, clean, minimalist design with bold colors",
            "tech": "futuristic tech style with neon accents and digital elements",
            "educational": "professional educational style with icons and infographics",
            "entertainment": "vibrant, eye-catching entertainment style with dynamic elements",
            "gaming": "gaming-inspired with RGB lighting effects and dynamic composition"
        }
        
        style_desc = style_descriptions.get(style, "modern professional")
        
        # Shorten title for prompt
        short_title = title[:50] if len(title) > 50 else title
        
        prompt = f"""Create a YouTube thumbnail image in {style_desc} style.
        Topic: {topic}
        Text overlay: "{short_title}"
        
        Requirements:
        - High contrast and eye-catching
        - Bold, readable text
        - Professional quality
        - 16:9 aspect ratio
        - Vibrant colors that stand out
        - Clear focal point
        - No blur or distortion
        """
        
        return prompt
        
    def _get_style_colors(self, style: str) -> Dict[str, tuple]:
        """Get color palette based on style"""
        palettes = {
            "modern": {
                "bg_start": (41, 128, 185),
                "bg_end": (109, 213, 250),
                "text": (255, 255, 255),
                "accent": (255, 195, 0)
            },
            "tech": {
                "bg_start": (15, 12, 41),
                "bg_end": (48, 43, 99),
                "text": (255, 255, 255),
                "accent": (0, 255, 255)
            },
            "educational": {
                "bg_start": (52, 73, 94),
                "bg_end": (44, 62, 80),
                "text": (255, 255, 255),
                "accent": (46, 204, 113)
            },
            "entertainment": {
                "bg_start": (255, 63, 52),
                "bg_end": (255, 175, 64),
                "text": (255, 255, 255),
                "accent": (155, 89, 182)
            },
            "gaming": {
                "bg_start": (136, 14, 79),
                "bg_end": (50, 16, 80),
                "text": (255, 255, 255),
                "accent": (138, 43, 226)
            }
        }
        
        return palettes.get(style, palettes["modern"])
        
    def _draw_gradient(self, draw: ImageDraw, start_color: tuple, end_color: tuple):
        """Draw gradient background"""
        for i in range(self.height):
            ratio = i / self.height
            r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
            g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
            b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
            draw.rectangle([(0, i), (self.width, i + 1)], fill=(r, g, b))
            
    def _add_pattern_overlay(self, image: Image, style: str):
        """Add subtle pattern overlay"""
        overlay = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        if style in ["tech", "gaming"]:
            # Add grid pattern
            for x in range(0, self.width, 50):
                draw.line([(x, 0), (x, self.height)], fill=(255, 255, 255, 20), width=1)
            for y in range(0, self.height, 50):
                draw.line([(0, y), (self.width, y)], fill=(255, 255, 255, 20), width=1)
                
        elif style == "modern":
            # Add circles
            for i in range(5):
                x = (i + 1) * self.width // 6
                y = self.height // 2
                radius = 100 + i * 20
                draw.ellipse(
                    [(x - radius, y - radius), (x + radius, y + radius)],
                    outline=(255, 255, 255, 30),
                    width=2
                )
                
        image.paste(overlay, (0, 0), overlay)
        
    def _add_title_text(self, image: Image, title: str, colors: Dict):
        """Add title text with proper formatting"""
        draw = ImageDraw.Draw(image)
        
        # Try to load a nice font, fall back to default
        try:
            # You would need to have fonts installed
            font_size = 80
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
            
        # Word wrap title if too long
        words = title.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            test_line = " ".join(current_line)
            # Simple width check (would be better with actual text measurement)
            if len(test_line) > 30:
                if len(current_line) > 1:
                    current_line.pop()
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    lines.append(test_line)
                    current_line = []
                    
        if current_line:
            lines.append(" ".join(current_line))
            
        # Draw text with shadow
        y_offset = (self.height - len(lines) * 100) // 2
        for i, line in enumerate(lines[:3]):  # Max 3 lines
            y = y_offset + i * 100
            
            # Shadow
            draw.text((52, y + 2), line, font=font, fill=(0, 0, 0, 128))
            # Main text
            draw.text((50, y), line, font=font, fill=colors["text"])
            
    def _add_decorative_elements(self, image: Image, style: str, colors: Dict):
        """Add style-specific decorative elements"""
        draw = ImageDraw.Draw(image)
        
        if style == "tech":
            # Add corner brackets
            bracket_size = 50
            bracket_width = 5
            # Top-left
            draw.line([(20, 20), (20 + bracket_size, 20)], fill=colors["accent"], width=bracket_width)
            draw.line([(20, 20), (20, 20 + bracket_size)], fill=colors["accent"], width=bracket_width)
            # Bottom-right
            draw.line([(self.width - 20 - bracket_size, self.height - 20), 
                      (self.width - 20, self.height - 20)], fill=colors["accent"], width=bracket_width)
            draw.line([(self.width - 20, self.height - 20 - bracket_size), 
                      (self.width - 20, self.height - 20)], fill=colors["accent"], width=bracket_width)
                      
        elif style == "modern":
            # Add accent bar
            draw.rectangle(
                [(0, self.height - 100), (self.width, self.height - 95)],
                fill=colors["accent"]
            )
            
    def _add_effects(self, image: Image) -> Image:
        """Add final effects like glow or shadow"""
        # Add slight blur to background while keeping text sharp
        # This would require more sophisticated masking in production
        
        # For now, just enhance the image slightly
        from PIL import ImageEnhance
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Increase color saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)
        
        return image