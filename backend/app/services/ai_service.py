"""
AI Service for Content Generation
Owner: VP of AI

Centralized AI service for all content generation needs.
Integrates with OpenAI, ElevenLabs, and other AI providers.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import httpx
import asyncio
from dataclasses import dataclass
from enum import Enum

from app.core.config import settings
from app.services.vault_service import VaultService
from app.utils.cost_calculator import CostCalculator, ServiceProvider
from app.core.metrics import metrics

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content that can be generated."""
    SCRIPT = "script"
    METADATA = "metadata"
    VISUAL_PROMPTS = "visual_prompts"
    TITLE_VARIATIONS = "title_variations"
    TAGS = "tags"
    THUMBNAIL_PROMPTS = "thumbnail_prompts"


@dataclass
class ContentGenerationRequest:
    """Request structure for content generation."""
    type: ContentType
    topic: str
    channel_context: Optional[Dict[str, Any]] = None
    target_duration: Optional[int] = None
    script: Optional[str] = None
    max_cost: float = 1.0
    style: str = "informative"
    audience: str = "general"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "type": self.type.value if isinstance(self.type, ContentType) else self.type,
            "topic": self.topic,
            "channel_context": self.channel_context or {},
            "target_duration": self.target_duration,
            "script": self.script,
            "max_cost": self.max_cost,
            "style": self.style,
            "audience": self.audience
        }


class AIServiceError(Exception):
    """Custom exception for AI service errors."""
    pass


class AIService:
    """Centralized AI service for all content generation."""
    
    def __init__(self):
        self.vault_service = VaultService()
        self.cost_calculator = CostCalculator()
        self.openai_client = None
        self._api_keys = {}
    
    async def initialize(self) -> None:
        """Initialize AI service with API keys."""
        try:
            # Get API keys from Vault
            self._api_keys['openai'] = await self.vault_service.get_secret("api-keys", "openai_api_key")
            self._api_keys['elevenlabs'] = await self.vault_service.get_secret("api-keys", "elevenlabs_api_key")
            
            if not self._api_keys.get('openai'):
                logger.warning("OpenAI API key not found, using environment variable")
                self._api_keys['openai'] = settings.OPENAI_API_KEY
            
            # Initialize OpenAI client
            if self._api_keys.get('openai'):
                import openai
                openai.api_key = self._api_keys['openai']
                self.openai_client = openai
                
        except Exception as e:
            logger.error(f"Failed to initialize AI service: {str(e)}")
            raise AIServiceError(f"AI service initialization failed: {str(e)}")
    
    def generate_content(self, request: ContentGenerationRequest) -> Dict[str, Any]:
        """Generate content based on request type."""
        
        try:
            # Ensure we have the API key
            if not self._api_keys.get('openai') and not settings.OPENAI_API_KEY:
                raise AIServiceError("OpenAI API key not available")
            
            if request.type == ContentType.SCRIPT or request.type == "script":
                return self.generate_script(request)
            elif request.type == ContentType.METADATA or request.type == "metadata":
                return self.generate_metadata(request)
            elif request.type == ContentType.VISUAL_PROMPTS or request.type == "visual_prompts":
                return self.generate_visual_prompts(request)
            elif request.type == ContentType.TITLE_VARIATIONS or request.type == "title_variations":
                return self.generate_title_variations(request)
            elif request.type == ContentType.TAGS or request.type == "tags":
                return self.generate_tags(request)
            else:
                raise AIServiceError(f"Unsupported content type: {request.type}")
                
        except Exception as e:
            logger.error(f"Content generation failed: {str(e)}")
            raise AIServiceError(f"Content generation failed: {str(e)}")
    
    def generate_script(self, request: ContentGenerationRequest) -> Dict[str, Any]:
        """Generate video script using OpenAI."""
        
        try:
            # Build prompt for script generation
            prompt = self._build_script_prompt(request)
            
            # Calculate target word count based on duration
            target_words = self._calculate_target_word_count(request.target_duration or 480)
            
            # Make OpenAI API call
            response = self._call_openai_chat(
                prompt=prompt,
                max_tokens=min(4000, target_words * 2),  # Allow for some overhead
                model="gpt-4-turbo",
                temperature=0.7
            )
            
            script_content = response['content']
            
            # Post-process script
            processed_script = self._post_process_script(script_content, request)
            
            return {
                'content': processed_script,
                'tokens_used': response['tokens_used'],
                'model': response['model'],
                'word_count': len(processed_script.split()),
                'estimated_duration': self._estimate_script_duration(processed_script),
                'generation_metadata': {
                    'prompt_type': 'script',
                    'target_duration': request.target_duration,
                    'style': request.style,
                    'generated_at': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Script generation failed: {str(e)}")
            raise AIServiceError(f"Script generation failed: {str(e)}")
    
    def generate_metadata(self, request: ContentGenerationRequest) -> Dict[str, Any]:
        """Generate title, description, and tags."""
        
        try:
            # Build prompt for metadata generation
            prompt = self._build_metadata_prompt(request)
            
            # Make OpenAI API call
            response = self._call_openai_chat(
                prompt=prompt,
                max_tokens=800,
                model="gpt-4-turbo",
                temperature=0.8,
                response_format="json"
            )
            
            # Parse JSON response
            import json
            metadata = json.loads(response['content'])
            
            # Validate and clean metadata
            cleaned_metadata = self._clean_metadata(metadata)
            
            return {
                'content': cleaned_metadata,
                'tokens_used': response['tokens_used'],
                'model': response['model'],
                'generation_metadata': {
                    'prompt_type': 'metadata',
                    'generated_at': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Metadata generation failed: {str(e)}")
            raise AIServiceError(f"Metadata generation failed: {str(e)}")
    
    def generate_visual_prompts(self, request: ContentGenerationRequest) -> Dict[str, Any]:
        """Generate visual prompts for image generation."""
        
        try:
            # Build prompt for visual prompts
            prompt = self._build_visual_prompts_prompt(request)
            
            # Make OpenAI API call
            response = self._call_openai_chat(
                prompt=prompt,
                max_tokens=1000,
                model="gpt-4-turbo",
                temperature=0.9,
                response_format="json"
            )
            
            # Parse JSON response
            import json
            visual_data = json.loads(response['content'])
            
            # Ensure we have a list of prompts
            if 'prompts' not in visual_data:
                visual_data = {'prompts': [visual_data.get('prompt', 'Abstract colorful background')]}
            
            return {
                'content': visual_data,
                'tokens_used': response['tokens_used'],
                'model': response['model'],
                'generation_metadata': {
                    'prompt_type': 'visual_prompts',
                    'generated_at': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Visual prompts generation failed: {str(e)}")
            raise AIServiceError(f"Visual prompts generation failed: {str(e)}")
    
    def generate_title_variations(self, request: ContentGenerationRequest) -> Dict[str, Any]:
        """Generate multiple title variations."""
        
        try:
            prompt = f"""Generate 5 engaging YouTube video titles for the topic: {request.topic}

Channel context: {request.channel_context}
Style: {request.style}
Audience: {request.audience}

Requirements:
- Each title should be 50-70 characters
- Include engaging words and phrases
- Consider SEO optimization
- Vary the approach (question, statement, list, etc.)

Return as JSON array: {{"titles": ["title1", "title2", ...]}}"""

            response = self._call_openai_chat(
                prompt=prompt,
                max_tokens=400,
                model="gpt-4-turbo",
                temperature=0.9,
                response_format="json"
            )
            
            import json
            titles_data = json.loads(response['content'])
            
            return {
                'content': titles_data,
                'tokens_used': response['tokens_used'],
                'model': response['model']
            }
            
        except Exception as e:
            logger.error(f"Title variations generation failed: {str(e)}")
            return {
                'content': {'titles': [f"Amazing {request.topic} You Need to Know"]},
                'tokens_used': 0,
                'model': 'fallback',
                'error': str(e)
            }
    
    def generate_tags(self, request: ContentGenerationRequest) -> Dict[str, Any]:
        """Generate relevant tags for the video."""
        
        try:
            prompt = f"""Generate 15 relevant YouTube tags for: {request.topic}

Consider:
- Main topic keywords
- Related subtopics
- Popular search terms
- Audience interests: {request.audience}
- Content style: {request.style}

Return as JSON: {{"tags": ["tag1", "tag2", ...]}}"""

            response = self._call_openai_chat(
                prompt=prompt,
                max_tokens=300,
                model="gpt-3.5-turbo",  # Cheaper for simple tasks
                temperature=0.7,
                response_format="json"
            )
            
            import json
            tags_data = json.loads(response['content'])
            
            return {
                'content': tags_data,
                'tokens_used': response['tokens_used'],
                'model': response['model']
            }
            
        except Exception as e:
            logger.error(f"Tags generation failed: {str(e)}")
            # Generate fallback tags
            fallback_tags = request.topic.split() + [request.style, '2025', 'trending']
            return {
                'content': {'tags': fallback_tags[:10]},
                'tokens_used': 0,
                'model': 'fallback',
                'error': str(e)
            }
    
    async def generate_images(
        self, 
        prompts: List[str], 
        video_id: str,
        style: str = "realistic",
        quality: str = "standard"
    ) -> Dict[str, Any]:
        """Generate images using DALL-E or other image generation services."""
        
        try:
            generated_images = []
            total_cost = 0.0
            
            for i, prompt in enumerate(prompts[:10]):  # Limit to 10 images
                try:
                    # Enhance prompt with style
                    enhanced_prompt = self._enhance_image_prompt(prompt, style)
                    
                    # Generate image using OpenAI DALL-E
                    image_result = await self._generate_single_image(
                        enhanced_prompt, f"{video_id}_{i}", quality
                    )
                    
                    generated_images.append(image_result)
                    total_cost += image_result.get('cost', 0)
                    
                    # Small delay to avoid rate limits
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate image {i}: {str(e)}")
                    # Continue with other images
                    continue
            
            return {
                'images': generated_images,
                'total_cost': total_cost,
                'generated_count': len(generated_images),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            return {
                'images': [],
                'total_cost': 0,
                'generated_count': 0,
                'success': False,
                'error': str(e)
            }
    
    def _call_openai_chat(
        self, 
        prompt: str, 
        max_tokens: int = 2000, 
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        response_format: str = "text"
    ) -> Dict[str, Any]:
        """Make OpenAI Chat API call with error handling."""
        
        try:
            import openai
            
            # Set API key
            api_key = self._api_keys.get('openai') or settings.OPENAI_API_KEY
            if not api_key:
                raise AIServiceError("OpenAI API key not available")
            
            # Prepare messages
            messages = [
                {"role": "system", "content": "You are an expert content creator for YouTube videos."},
                {"role": "user", "content": prompt}
            ]
            
            # Prepare request parameters
            request_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Add response format if specified
            if response_format == "json":
                request_params["response_format"] = {"type": "json_object"}
            
            # Make API call
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(**request_params)
            
            # Extract response data
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            return {
                'content': content,
                'tokens_used': tokens_used,
                'model': model,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise AIServiceError(f"OpenAI API call failed: {str(e)}")
    
    async def _generate_single_image(
        self, 
        prompt: str, 
        filename: str, 
        quality: str = "standard"
    ) -> Dict[str, Any]:
        """Generate a single image using DALL-E."""
        
        try:
            import openai
            from pathlib import Path
            import httpx
            
            # Set API key
            api_key = self._api_keys.get('openai') or settings.OPENAI_API_KEY
            client = openai.OpenAI(api_key=api_key)
            
            # Generate image
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality=quality,
                n=1
            )
            
            # Get image URL
            image_url = response.data[0].url
            
            # Download and save image
            async with httpx.AsyncClient() as http_client:
                image_response = await http_client.get(image_url)
                image_response.raise_for_status()
                
                # Save image
                image_dir = Path(f"/tmp/images/{filename.split('_')[0]}")
                image_dir.mkdir(parents=True, exist_ok=True)
                
                image_path = image_dir / f"{filename}.png"
                with open(image_path, 'wb') as f:
                    f.write(image_response.content)
            
            # Calculate cost
            cost = self.cost_calculator.calculate_image_generation_cost(
                ServiceProvider.OPENAI,
                f"dalle-3-{quality}",
                1
            )
            
            return {
                'file_path': str(image_path),
                'url': image_url,
                'prompt': prompt,
                'size': "1024x1024",
                'quality': quality,
                'cost': cost,
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Single image generation failed: {str(e)}")
            raise
    
    def _build_script_prompt(self, request: ContentGenerationRequest) -> str:
        """Build prompt for script generation."""
        
        channel_info = request.channel_context or {}
        target_words = self._calculate_target_word_count(request.target_duration or 480)
        
        prompt = f"""Create an engaging YouTube video script about: {request.topic}

Channel Context:
- Channel Name: {channel_info.get('name', 'YTEmpire Channel')}
- Category: {channel_info.get('category', 'Educational')}
- Target Audience: {channel_info.get('target_audience', request.audience)}
- Tone: {channel_info.get('tone', request.style)}

Requirements:
- Target length: ~{target_words} words ({request.target_duration//60} minutes)
- Style: {request.style}
- Include a compelling hook in the first 15 seconds
- Add clear transitions between sections
- Include a strong call-to-action at the end
- Make it engaging and informative
- Use natural, conversational language
- Include timestamps markers where appropriate

Structure:
1. Hook (0-15 seconds)
2. Introduction (15-45 seconds)
3. Main content (middle section)
4. Conclusion and CTA (last 30 seconds)

Write only the script content, no additional formatting."""

        return prompt
    
    def _build_metadata_prompt(self, request: ContentGenerationRequest) -> str:
        """Build prompt for metadata generation."""
        
        return f"""Generate YouTube video metadata for: {request.topic}

Script preview: {(request.script or '')[:500]}...

Generate a JSON response with:
{{
  "title": "Engaging title (50-70 characters)",
  "description": "Detailed description (100-500 words)",
  "tags": ["tag1", "tag2", "tag3", ...] (10-15 tags)
}}

Requirements:
- Title should be clickable and SEO-friendly
- Description should include video summary, key points, and call-to-action
- Tags should be relevant and help with discoverability
- Consider the channel context: {request.channel_context}"""
    
    def _build_visual_prompts_prompt(self, request: ContentGenerationRequest) -> str:
        """Build prompt for visual prompts generation."""
        
        script_preview = (request.script or '')[:1000]
        
        return f"""Generate visual prompts for image generation based on this video script:

Topic: {request.topic}
Script preview: {script_preview}...

Create 5-8 diverse visual prompts that would work well as background images for this video.

Return JSON format:
{{
  "prompts": [
    "Detailed visual prompt 1",
    "Detailed visual prompt 2",
    ...
  ]
}}

Requirements:
- Each prompt should be detailed and specific
- Consider different scenes/concepts from the script
- Make them suitable for stock-photo style images
- Avoid text or specific people
- Focus on conceptual, abstract, or technical imagery
- Style: professional, clean, modern"""
    
    def _enhance_image_prompt(self, base_prompt: str, style: str) -> str:
        """Enhance image prompt with style and quality modifiers."""
        
        style_modifiers = {
            "realistic": "photorealistic, high quality, professional photography",
            "artistic": "digital art, stylized, creative, artistic",
            "minimalist": "clean, simple, minimalist design, modern",
            "technical": "technical diagram, clean lines, professional",
            "abstract": "abstract art, conceptual, modern design"
        }
        
        quality_suffix = ", 4k resolution, professional quality, trending on artstation"
        style_suffix = style_modifiers.get(style, style_modifiers["realistic"])
        
        return f"{base_prompt}, {style_suffix}{quality_suffix}"
    
    def _calculate_target_word_count(self, duration_seconds: int) -> int:
        """Calculate target word count based on video duration."""
        # Average speaking pace: 150-180 words per minute
        words_per_second = 2.5  # 150 words/minute / 60 seconds
        return int(duration_seconds * words_per_second)
    
    def _estimate_script_duration(self, script: str) -> int:
        """Estimate video duration from script word count."""
        word_count = len(script.split())
        # 150 words per minute average
        return int((word_count / 150) * 60)
    
    def _post_process_script(self, script: str, request: ContentGenerationRequest) -> str:
        """Post-process generated script for better formatting."""
        
        # Clean up extra whitespace
        script = ' '.join(script.split())
        
        # Ensure proper paragraph breaks
        script = script.replace('. ', '.\n\n')
        
        # Add timing markers for longer scripts
        if len(script.split()) > 200:
            paragraphs = script.split('\n\n')
            processed_paragraphs = []
            
            for i, paragraph in enumerate(paragraphs):
                if i > 0 and i % 3 == 0:  # Every 3 paragraphs
                    timestamp = f"[{i//3 * 2}:00]"
                    processed_paragraphs.append(f"{timestamp} {paragraph}")
                else:
                    processed_paragraphs.append(paragraph)
            
            script = '\n\n'.join(processed_paragraphs)
        
        return script.strip()
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate metadata."""
        
        cleaned = {}
        
        # Clean title
        title = metadata.get('title', 'Untitled Video')
        if len(title) > 100:
            title = title[:97] + "..."
        cleaned['title'] = title
        
        # Clean description
        description = metadata.get('description', '')
        if len(description) > 5000:
            description = description[:4997] + "..."
        cleaned['description'] = description
        
        # Clean tags
        tags = metadata.get('tags', [])
        if isinstance(tags, list):
            # Ensure tags are strings and reasonable length
            cleaned_tags = []
            for tag in tags[:15]:  # Limit to 15 tags
                if isinstance(tag, str) and len(tag) <= 50:
                    cleaned_tags.append(tag.strip())
            cleaned['tags'] = cleaned_tags
        else:
            cleaned['tags'] = []
        
        return cleaned