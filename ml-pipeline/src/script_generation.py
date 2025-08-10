"""
Script Generation Module for YouTube Videos
Uses OpenAI GPT-4 for generating optimized video scripts
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import openai
import redis
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScriptStyle(Enum):
    """Video script styles"""
    INFORMATIVE = "informative"
    ENTERTAINING = "entertaining"
    TUTORIAL = "tutorial"
    REVIEW = "review"
    NEWS = "news"
    STORY = "story"
    EDUCATIONAL = "educational"
    PROMOTIONAL = "promotional"

class ScriptTone(Enum):
    """Script tone options"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    HUMOROUS = "humorous"
    SERIOUS = "serious"
    INSPIRATIONAL = "inspirational"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"

@dataclass
class ScriptRequest:
    """Script generation request"""
    topic: str
    style: ScriptStyle
    tone: ScriptTone
    duration_minutes: int
    target_audience: str
    keywords: List[str]
    language: str = "en"
    include_hook: bool = True
    include_cta: bool = True
    trending_context: Optional[Dict[str, Any]] = None
    channel_context: Optional[Dict[str, Any]] = None

@dataclass
class ScriptResponse:
    """Generated script response"""
    title: str
    hook: str
    introduction: str
    main_content: List[Dict[str, str]]  # List of sections
    conclusion: str
    call_to_action: str
    timestamps: List[Dict[str, str]]  # Timestamp markers
    keywords_used: List[str]
    estimated_duration: int  # seconds
    word_count: int
    cost: float
    metadata: Dict[str, Any]

class ScriptGenerator:
    """
    Advanced script generation using OpenAI GPT-4
    Optimized for engagement and cost efficiency
    """
    
    def __init__(self, api_key: str, redis_host: str = 'localhost', redis_port: int = 6379):
        self.api_key = api_key
        openai.api_key = api_key
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.cache_ttl = 86400  # 24 hours
        
        # Cost optimization settings
        self.model_config = {
            'fast': 'gpt-3.5-turbo-16k',
            'balanced': 'gpt-4',
            'quality': 'gpt-4-turbo-preview'
        }
        
        # Words per minute for different styles
        self.wpm_rates = {
            ScriptStyle.TUTORIAL: 140,
            ScriptStyle.NEWS: 160,
            ScriptStyle.STORY: 130,
            ScriptStyle.REVIEW: 150,
            ScriptStyle.INFORMATIVE: 145,
            ScriptStyle.ENTERTAINING: 155,
            ScriptStyle.EDUCATIONAL: 140,
            ScriptStyle.PROMOTIONAL: 160
        }
    
    async def generate_script(
        self,
        request: ScriptRequest,
        quality_preset: str = 'balanced'
    ) -> ScriptResponse:
        """
        Generate optimized video script
        
        Args:
            request: Script generation request
            quality_preset: Quality/cost tradeoff (fast, balanced, quality)
        
        Returns:
            Generated script with metadata
        """
        # Check cache first
        cache_key = self._generate_cache_key(request)
        cached_script = self._get_cached_script(cache_key)
        if cached_script:
            logger.info(f"Returning cached script for topic: {request.topic}")
            return cached_script
        
        # Calculate target word count
        target_words = self._calculate_word_count(request)
        
        # Build generation prompt
        prompt = self._build_prompt(request, target_words)
        
        # Generate script using OpenAI
        script_data = await self._generate_with_openai(prompt, quality_preset)
        
        # Parse and structure response
        script_response = self._parse_script_response(script_data, request)
        
        # Calculate cost
        script_response.cost = self._calculate_cost(
            len(prompt) + len(json.dumps(script_data)),
            quality_preset
        )
        
        # Cache the result
        self._cache_script(cache_key, script_response)
        
        return script_response
    
    def _calculate_word_count(self, request: ScriptRequest) -> int:
        """Calculate target word count based on duration and style"""
        wpm = self.wpm_rates.get(request.style, 150)
        return int(request.duration_minutes * wpm)
    
    def _build_prompt(self, request: ScriptRequest, target_words: int) -> str:
        """Build optimized prompt for script generation"""
        
        # Base prompt structure
        prompt_parts = [
            f"Generate a {request.duration_minutes}-minute YouTube video script.",
            f"Topic: {request.topic}",
            f"Style: {request.style.value}",
            f"Tone: {request.tone.value}",
            f"Target Audience: {request.target_audience}",
            f"Target Word Count: {target_words} words",
            f"Language: {request.language}",
        ]
        
        # Add keywords if provided
        if request.keywords:
            prompt_parts.append(f"Keywords to include: {', '.join(request.keywords)}")
        
        # Add trending context if available
        if request.trending_context:
            prompt_parts.append(f"Trending context: {json.dumps(request.trending_context)}")
        
        # Add channel context for consistency
        if request.channel_context:
            prompt_parts.append(f"Channel style: {json.dumps(request.channel_context)}")
        
        # Structure requirements
        prompt_parts.extend([
            "\nScript Structure Required:",
            "1. TITLE: Engaging, SEO-optimized title",
            "2. HOOK: Attention-grabbing opening (5-10 seconds)",
            "3. INTRODUCTION: Brief intro and what viewers will learn",
            "4. MAIN_CONTENT: Organized in clear sections with timestamps",
            "5. CONCLUSION: Summary of key points",
            "6. CALL_TO_ACTION: Encourage engagement (like, subscribe, comment)",
            "",
            "Format the response as JSON with the following structure:",
            "{",
            '  "title": "...",',
            '  "hook": "...",',
            '  "introduction": "...",',
            '  "main_content": [',
            '    {"section_title": "...", "content": "...", "duration_seconds": 60},',
            '    ...',
            '  ],',
            '  "conclusion": "...",',
            '  "call_to_action": "...",',
            '  "timestamps": [',
            '    {"time": "00:00", "label": "Introduction"},',
            '    ...',
            '  ],',
            '  "keywords_used": ["..."],',
            '}',
            "",
            "Optimization requirements:",
            "- Maximize viewer retention with engaging content",
            "- Include pattern interrupts every 30-45 seconds",
            "- Use power words and emotional triggers",
            "- Optimize for YouTube algorithm (watch time, engagement)",
            "- Natural keyword integration for SEO",
            f"- Ensure total script is approximately {target_words} words"
        ])
        
        return "\n".join(prompt_parts)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _generate_with_openai(self, prompt: str, quality_preset: str) -> Dict:
        """Generate script using OpenAI API with retry logic"""
        model = self.model_config[quality_preset]
        
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert YouTube scriptwriter who creates engaging, high-retention video scripts optimized for the YouTube algorithm."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            # Fallback to structured parsing
            return self._fallback_parse(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _fallback_parse(self, content: str) -> Dict:
        """Fallback parser if JSON parsing fails"""
        # Basic structure extraction
        script_data = {
            "title": self._extract_section(content, "TITLE"),
            "hook": self._extract_section(content, "HOOK"),
            "introduction": self._extract_section(content, "INTRODUCTION"),
            "main_content": [],
            "conclusion": self._extract_section(content, "CONCLUSION"),
            "call_to_action": self._extract_section(content, "CALL TO ACTION"),
            "timestamps": [],
            "keywords_used": []
        }
        
        # Extract main content sections
        main_content = self._extract_section(content, "MAIN CONTENT")
        if main_content:
            sections = main_content.split("\n\n")
            for i, section in enumerate(sections):
                if section.strip():
                    script_data["main_content"].append({
                        "section_title": f"Section {i+1}",
                        "content": section.strip(),
                        "duration_seconds": 60
                    })
        
        return script_data
    
    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a section from text content"""
        import re
        pattern = rf"{section_name}:?\s*(.*?)(?:\n[A-Z]+:|$)"
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _parse_script_response(
        self,
        script_data: Dict,
        request: ScriptRequest
    ) -> ScriptResponse:
        """Parse and structure the script response"""
        
        # Calculate total word count
        word_count = sum([
            len(script_data.get('hook', '').split()),
            len(script_data.get('introduction', '').split()),
            sum(len(section.get('content', '').split()) 
                for section in script_data.get('main_content', [])),
            len(script_data.get('conclusion', '').split()),
            len(script_data.get('call_to_action', '').split())
        ])
        
        # Calculate estimated duration
        wpm = self.wpm_rates.get(request.style, 150)
        estimated_duration = int((word_count / wpm) * 60)
        
        # Ensure timestamps exist
        if not script_data.get('timestamps'):
            script_data['timestamps'] = self._generate_timestamps(
                script_data.get('main_content', [])
            )
        
        return ScriptResponse(
            title=script_data.get('title', f"Video about {request.topic}"),
            hook=script_data.get('hook', ''),
            introduction=script_data.get('introduction', ''),
            main_content=script_data.get('main_content', []),
            conclusion=script_data.get('conclusion', ''),
            call_to_action=script_data.get('call_to_action', ''),
            timestamps=script_data.get('timestamps', []),
            keywords_used=script_data.get('keywords_used', request.keywords),
            estimated_duration=estimated_duration,
            word_count=word_count,
            cost=0,  # Will be calculated separately
            metadata={
                'style': request.style.value,
                'tone': request.tone.value,
                'target_audience': request.target_audience,
                'language': request.language,
                'generated_at': datetime.now().isoformat()
            }
        )
    
    def _generate_timestamps(self, main_content: List[Dict]) -> List[Dict[str, str]]:
        """Generate timestamps based on content sections"""
        timestamps = [{"time": "00:00", "label": "Introduction"}]
        
        current_time = 10  # Start after intro
        for section in main_content:
            minutes = current_time // 60
            seconds = current_time % 60
            timestamps.append({
                "time": f"{minutes:02d}:{seconds:02d}",
                "label": section.get('section_title', 'Section')
            })
            current_time += section.get('duration_seconds', 60)
        
        return timestamps
    
    def _calculate_cost(self, total_tokens: int, quality_preset: str) -> float:
        """Calculate generation cost based on tokens and model"""
        # Approximate cost per 1K tokens (as of 2024)
        cost_per_1k = {
            'gpt-3.5-turbo-16k': 0.003,
            'gpt-4': 0.03,
            'gpt-4-turbo-preview': 0.01
        }
        
        model = self.model_config[quality_preset]
        rate = cost_per_1k.get(model, 0.01)
        
        return (total_tokens / 1000) * rate
    
    def _generate_cache_key(self, request: ScriptRequest) -> str:
        """Generate cache key for script request"""
        key_data = {
            'topic': request.topic,
            'style': request.style.value,
            'tone': request.tone.value,
            'duration': request.duration_minutes,
            'audience': request.target_audience,
            'keywords': sorted(request.keywords),
            'language': request.language
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"script:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _get_cached_script(self, cache_key: str) -> Optional[ScriptResponse]:
        """Retrieve cached script if available"""
        cached_data = self.redis_client.get(cache_key)
        if cached_data:
            try:
                data = json.loads(cached_data)
                return ScriptResponse(**data)
            except Exception as e:
                logger.error(f"Error deserializing cached script: {e}")
        return None
    
    def _cache_script(self, cache_key: str, script: ScriptResponse):
        """Cache generated script"""
        try:
            script_dict = {
                'title': script.title,
                'hook': script.hook,
                'introduction': script.introduction,
                'main_content': script.main_content,
                'conclusion': script.conclusion,
                'call_to_action': script.call_to_action,
                'timestamps': script.timestamps,
                'keywords_used': script.keywords_used,
                'estimated_duration': script.estimated_duration,
                'word_count': script.word_count,
                'cost': script.cost,
                'metadata': script.metadata
            }
            
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(script_dict)
            )
        except Exception as e:
            logger.error(f"Error caching script: {e}")
    
    async def optimize_for_engagement(
        self,
        script: ScriptResponse,
        analytics_data: Optional[Dict[str, Any]] = None
    ) -> ScriptResponse:
        """
        Optimize script based on channel analytics
        
        Args:
            script: Original script
            analytics_data: Historical performance data
        
        Returns:
            Optimized script
        """
        if not analytics_data:
            return script
        
        # Analyze best performing content patterns
        optimization_prompt = self._build_optimization_prompt(script, analytics_data)
        
        # Get optimization suggestions
        optimized_data = await self._generate_with_openai(
            optimization_prompt,
            'fast'  # Use fast model for optimization
        )
        
        # Apply optimizations
        if optimized_data.get('hook'):
            script.hook = optimized_data['hook']
        
        if optimized_data.get('call_to_action'):
            script.call_to_action = optimized_data['call_to_action']
        
        script.metadata['optimized'] = True
        script.metadata['optimization_date'] = datetime.now().isoformat()
        
        return script
    
    def _build_optimization_prompt(
        self,
        script: ScriptResponse,
        analytics_data: Dict[str, Any]
    ) -> str:
        """Build prompt for script optimization"""
        prompt_parts = [
            "Optimize the following video script based on performance data:",
            f"Current Hook: {script.hook}",
            f"Current CTA: {script.call_to_action}",
            "",
            "Performance Data:",
            f"Average Retention: {analytics_data.get('avg_retention', 'N/A')}%",
            f"Best Performing Videos: {json.dumps(analytics_data.get('top_videos', []))}",
            f"Audience Retention Drops: {json.dumps(analytics_data.get('drop_points', []))}",
            "",
            "Provide optimized versions of:",
            "1. Hook (make it more engaging)",
            "2. Call to Action (increase conversion)",
            "",
            "Format as JSON:",
            '{"hook": "...", "call_to_action": "..."}'
        ]
        
        return "\n".join(prompt_parts)
    
    async def generate_variations(
        self,
        request: ScriptRequest,
        num_variations: int = 3
    ) -> List[ScriptResponse]:
        """
        Generate multiple script variations for A/B testing
        
        Args:
            request: Base script request
            num_variations: Number of variations to generate
        
        Returns:
            List of script variations
        """
        variations = []
        
        # Different tones and styles to try
        tone_variations = [
            ScriptTone.PROFESSIONAL,
            ScriptTone.CASUAL,
            ScriptTone.FRIENDLY
        ]
        
        for i in range(min(num_variations, len(tone_variations))):
            # Modify request for variation
            varied_request = ScriptRequest(
                topic=request.topic,
                style=request.style,
                tone=tone_variations[i],
                duration_minutes=request.duration_minutes,
                target_audience=request.target_audience,
                keywords=request.keywords,
                language=request.language,
                include_hook=request.include_hook,
                include_cta=request.include_cta,
                trending_context=request.trending_context,
                channel_context=request.channel_context
            )
            
            # Generate variation
            script = await self.generate_script(varied_request, 'fast')
            script.metadata['variation_id'] = i + 1
            script.metadata['variation_tone'] = tone_variations[i].value
            
            variations.append(script)
        
        return variations


# API Integration
class ScriptGenerationAPI:
    """FastAPI integration for script generation"""
    
    def __init__(self, openai_api_key: str):
        self.generator = ScriptGenerator(openai_api_key)
    
    async def generate(
        self,
        topic: str,
        style: str,
        duration: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate script via API
        
        Args:
            topic: Video topic
            style: Content style
            duration: Video duration (short, medium, long)
            **kwargs: Additional parameters
        
        Returns:
            Script data for API response
        """
        # Map duration to minutes
        duration_map = {
            'short': 3,
            'medium': 8,
            'long': 15
        }
        
        # Create request
        request = ScriptRequest(
            topic=topic,
            style=ScriptStyle(style),
            tone=ScriptTone(kwargs.get('tone', 'professional')),
            duration_minutes=duration_map.get(duration, 5),
            target_audience=kwargs.get('target_audience', 'general'),
            keywords=kwargs.get('keywords', []),
            language=kwargs.get('language', 'en'),
            trending_context=kwargs.get('trending_context'),
            channel_context=kwargs.get('channel_context')
        )
        
        # Generate script
        script = await self.generator.generate_script(
            request,
            kwargs.get('quality_preset', 'balanced')
        )
        
        # Convert to API response
        return {
            'title': script.title,
            'script': {
                'hook': script.hook,
                'introduction': script.introduction,
                'main_content': script.main_content,
                'conclusion': script.conclusion,
                'call_to_action': script.call_to_action
            },
            'timestamps': script.timestamps,
            'metadata': {
                'word_count': script.word_count,
                'estimated_duration': script.estimated_duration,
                'keywords_used': script.keywords_used,
                'cost': script.cost,
                **script.metadata
            }
        }


# Initialize global instance
script_api = ScriptGenerationAPI(openai_api_key="your-api-key-here")