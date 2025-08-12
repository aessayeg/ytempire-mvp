"""
AI Services Integration
Handles OpenAI, ElevenLabs, and Google TTS APIs
"""
import os
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass
import json
import base64
from datetime import datetime
import openai
from elevenlabs import generate, voices, set_api_key
try:
    from google.cloud import texttospeech
    from google.oauth2 import service_account
    GOOGLE_TTS_AVAILABLE = True
except ImportError:
    GOOGLE_TTS_AVAILABLE = False
import tempfile
import wave
from pydub import AudioSegment
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

@dataclass
class AIServiceConfig:
    """Configuration for AI services"""
    openai_api_key: str
    elevenlabs_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    openai_max_tokens: int = 4000
    openai_temperature: float = 0.7
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
    elevenlabs_model: str = "eleven_monolingual_v1"
    
    google_tts_credentials_path: str = "google_tts_credentials.json"
    google_tts_language_code: str = "en-US"
    google_tts_voice_name: str = "en-US-Neural2-J"
    
    max_retries: int = 3
    timeout: int = 60


class OpenAIService:
    """OpenAI API integration for text generation"""
    
    def __init__(self, config: AIServiceConfig):
        self.config = config
        openai.api_key = config.openai_api_key
        try:
            self.client = openai.AsyncOpenAI(api_key=config.openai_api_key)
        except TypeError as e:
            # Handle version compatibility issues
            logger.warning(f"OpenAI client initialization issue: {e}")
            # Fallback to basic initialization
            self.client = openai.AsyncOpenAI(api_key=config.openai_api_key)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_script(
        self,
        topic: str,
        style: str = "educational",
        length: str = "medium",
        target_audience: str = "general",
        keywords: List[str] = None
    ) -> Dict[str, Any]:
        """Generate a YouTube video script"""
        try:
            # Build the prompt
            prompt = self._build_script_prompt(topic, style, length, target_audience, keywords)
            
            # Generate script using GPT-4
            response = await self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are a professional YouTube script writer who creates engaging, informative, and well-structured video scripts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.openai_max_tokens,
                temperature=self.config.openai_temperature,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            script_data = json.loads(response.choices[0].message.content)
            
            # Calculate tokens and cost
            tokens_used = response.usage.total_tokens
            cost = self._calculate_cost(tokens_used, self.config.openai_model)
            
            return {
                "script": script_data,
                "tokens_used": tokens_used,
                "cost": cost,
                "model": self.config.openai_model,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"OpenAI script generation error: {e}")
            raise
            
    async def generate_title(self, topic: str, keywords: List[str] = None) -> Dict[str, Any]:
        """Generate optimized YouTube title"""
        try:
            prompt = f"""Generate 5 catchy, SEO-optimized YouTube video titles for a video about: {topic}
            Keywords to include: {', '.join(keywords) if keywords else 'none specified'}
            
            Requirements:
            - Maximum 60 characters
            - Include power words
            - Create curiosity
            - SEO friendly
            
            Return as JSON: {{"titles": ["title1", "title2", ...]}}"""
            
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a YouTube SEO expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.9,
                response_format={"type": "json_object"}
            )
            
            titles = json.loads(response.choices[0].message.content)
            
            return {
                "titles": titles["titles"],
                "tokens_used": response.usage.total_tokens,
                "cost": self._calculate_cost(response.usage.total_tokens, "gpt-3.5-turbo")
            }
            
        except Exception as e:
            logger.error(f"Title generation error: {e}")
            raise
            
    async def generate_description(
        self,
        title: str,
        script: str,
        keywords: List[str] = None
    ) -> Dict[str, Any]:
        """Generate SEO-optimized video description"""
        try:
            prompt = f"""Generate an SEO-optimized YouTube video description for:
            Title: {title}
            Script summary: {script[:500]}...
            Keywords: {', '.join(keywords) if keywords else 'none'}
            
            Include:
            - Compelling hook
            - Video overview
            - Timestamps (if applicable)
            - Call to action
            - Relevant hashtags
            
            Return as JSON: {{"description": "...", "hashtags": ["#tag1", "#tag2", ...]}}"""
            
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                "description": result["description"],
                "hashtags": result["hashtags"],
                "tokens_used": response.usage.total_tokens,
                "cost": self._calculate_cost(response.usage.total_tokens, "gpt-3.5-turbo")
            }
            
        except Exception as e:
            logger.error(f"Description generation error: {e}")
            raise
            
    async def generate_thumbnail_prompt(self, title: str, style: str = "vibrant") -> str:
        """Generate a prompt for thumbnail creation"""
        try:
            prompt = f"""Create a detailed prompt for generating a YouTube thumbnail for:
            Title: {title}
            Style: {style}
            
            Include specific details about:
            - Main visual elements
            - Color scheme
            - Text overlay style
            - Composition
            
            Return as a single detailed prompt string."""
            
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.8
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Thumbnail prompt generation error: {e}")
            raise
            
    def _build_script_prompt(
        self,
        topic: str,
        style: str,
        length: str,
        target_audience: str,
        keywords: List[str]
    ) -> str:
        """Build prompt for script generation"""
        word_counts = {
            "short": "300-500 words",
            "medium": "800-1200 words",
            "long": "1500-2000 words"
        }
        
        prompt = f"""Create a YouTube video script about: {topic}

Style: {style}
Length: {word_counts.get(length, "800-1200 words")}
Target Audience: {target_audience}
Keywords to include: {', '.join(keywords) if keywords else 'none specified'}

Structure the script as JSON with the following format:
{{
    "hook": "Opening hook to grab attention (10-15 seconds)",
    "introduction": "Brief introduction of the topic",
    "main_sections": [
        {{
            "title": "Section title",
            "content": "Section content",
            "duration_seconds": estimated_duration
        }}
    ],
    "conclusion": "Wrap-up and call to action",
    "call_to_action": "Subscribe, like, comment prompt",
    "estimated_total_duration": total_duration_in_seconds
}}

Make it engaging, informative, and optimized for retention."""
        
        return prompt
        
    def _calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate API cost based on tokens used"""
        # Pricing as of 2024
        pricing = {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},  # per 1K tokens
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }
        
        model_pricing = pricing.get(model, pricing["gpt-3.5-turbo"])
        # Simplified calculation (assuming 50/50 input/output split)
        avg_price = (model_pricing["input"] + model_pricing["output"]) / 2
        cost = (tokens / 1000) * avg_price
        
        return round(cost, 4)


class ElevenLabsService:
    """ElevenLabs API integration for voice synthesis"""
    
    def __init__(self, config: AIServiceConfig):
        self.config = config
        set_api_key(config.elevenlabs_api_key)
        self.api_key = config.elevenlabs_api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        
    async def text_to_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convert text to speech using ElevenLabs"""
        try:
            voice_id = voice_id or self.config.elevenlabs_voice_id
            model_id = model_id or self.config.elevenlabs_model
            
            # Generate audio
            audio = generate(
                text=text,
                voice=voice_id,
                model=model_id
            )
            
            # Save to file if path provided
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(audio)
            else:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    f.write(audio)
                    output_path = f.name
                    
            # Calculate cost (ElevenLabs charges per character)
            characters = len(text)
            cost = self._calculate_cost(characters)
            
            return {
                "audio_path": output_path,
                "duration_estimate": len(text) / 15,  # Rough estimate: 15 chars/second
                "characters": characters,
                "cost": cost,
                "voice_id": voice_id,
                "model": model_id
            }
            
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            raise
            
    async def get_voices(self) -> List[Dict]:
        """Get available voices"""
        try:
            available_voices = voices()
            
            voice_list = []
            for voice in available_voices:
                voice_list.append({
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "category": voice.category,
                    "description": voice.description
                })
                
            return voice_list
            
        except Exception as e:
            logger.error(f"Error fetching voices: {e}")
            raise
            
    async def clone_voice(
        self,
        name: str,
        audio_files: List[str],
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Clone a voice from audio samples"""
        try:
            # This would use the ElevenLabs voice cloning API
            # Requires professional tier subscription
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/voices/add"
                
                data = aiohttp.FormData()
                data.add_field("name", name)
                if description:
                    data.add_field("description", description)
                    
                for audio_file in audio_files:
                    with open(audio_file, "rb") as f:
                        data.add_field("files", f, filename=os.path.basename(audio_file))
                        
                headers = {"xi-api-key": self.api_key}
                
                async with session.post(url, data=data, headers=headers) as response:
                    result = await response.json()
                    
                    return {
                        "voice_id": result["voice_id"],
                        "name": name,
                        "status": "created"
                    }
                    
        except Exception as e:
            logger.error(f"Voice cloning error: {e}")
            raise
            
    def _calculate_cost(self, characters: int) -> float:
        """Calculate ElevenLabs API cost"""
        # ElevenLabs pricing (approximate)
        # Starter: $5/month for 30,000 characters
        # Creator: $22/month for 100,000 characters
        cost_per_char = 0.00022  # Based on Creator plan
        return round(characters * cost_per_char, 4)


class GoogleTTSService:
    """Google Text-to-Speech API integration"""
    
    def __init__(self, config: AIServiceConfig):
        self.config = config
        self.client = None
        
        # Initialize Google TTS client if available and credentials exist
        if GOOGLE_TTS_AVAILABLE:
            try:
                if os.path.exists(config.google_tts_credentials_path):
                    credentials = service_account.Credentials.from_service_account_file(
                        config.google_tts_credentials_path
                    )
                    self.client = texttospeech.TextToSpeechClient(credentials=credentials)
                else:
                    # Try to use default credentials (for GCP environments)
                    self.client = texttospeech.TextToSpeechClient()
            except Exception as e:
                logger.warning(f"Google TTS client not initialized: {e}")
                self.client = None
        else:
            logger.info("Google TTS library not installed, using mock TTS")
            
    async def text_to_speech(
        self,
        text: str,
        language_code: Optional[str] = None,
        voice_name: Optional[str] = None,
        output_path: Optional[str] = None,
        audio_encoding: str = "MP3"
    ) -> Dict[str, Any]:
        """Convert text to speech using Google TTS"""
        if self.client is None:
            # Return mock audio data for testing
            return await self._create_mock_audio(text, output_path)
        
        try:
            language_code = language_code or self.config.google_tts_language_code
            voice_name = voice_name or self.config.google_tts_voice_name
            
            # Set up synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Set up voice parameters
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name,
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
            
            # Set up audio config
            audio_config_map = {
                "MP3": texttospeech.AudioEncoding.MP3,
                "LINEAR16": texttospeech.AudioEncoding.LINEAR16,
                "OGG_OPUS": texttospeech.AudioEncoding.OGG_OPUS
            }
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=audio_config_map.get(audio_encoding, texttospeech.AudioEncoding.MP3),
                speaking_rate=1.0,
                pitch=0.0
            )
            
            # Perform synthesis
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Save audio to file
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(response.audio_content)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    f.write(response.audio_content)
                    output_path = f.name
                    
            # Calculate cost
            characters = len(text)
            cost = self._calculate_cost(characters, voice_name)
            
            return {
                "audio_path": output_path,
                "characters": characters,
                "cost": cost,
                "voice": voice_name,
                "language": language_code,
                "encoding": audio_encoding
            }
            
        except Exception as e:
            logger.error(f"Google TTS error: {e}")
            raise
            
    async def list_voices(self, language_code: Optional[str] = None) -> List[Dict]:
        """List available voices"""
        try:
            response = self.client.list_voices(language_code=language_code)
            
            voice_list = []
            for voice in response.voices:
                voice_list.append({
                    "name": voice.name,
                    "language_codes": voice.language_codes,
                    "ssml_gender": voice.ssml_gender.name,
                    "natural_sample_rate": voice.natural_sample_rate_hertz
                })
                
            return voice_list
            
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            raise
            
    def _calculate_cost(self, characters: int, voice_type: str) -> float:
        """Calculate Google TTS cost"""
        # Google TTS pricing per 1 million characters
        pricing = {
            "standard": 4.00,  # Standard voices
            "wavenet": 16.00,  # WaveNet voices
            "neural2": 16.00   # Neural2 voices
        }
        
        # Determine voice type
        if "Wavenet" in voice_type:
            rate = pricing["wavenet"]
        elif "Neural2" in voice_type:
            rate = pricing["neural2"]
        else:
            rate = pricing["standard"]
            
        cost = (characters / 1_000_000) * rate
        return round(cost, 4)
    
    async def _create_mock_audio(self, text: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Create mock audio file for testing when Google TTS is not available"""
        import json
        
        if not output_path:
            output_path = tempfile.mktemp(suffix=".mp3")
        
        # Create a simple mock audio file
        mock_audio_data = {
            "type": "mock_audio",
            "text": text[:500],
            "duration": len(text) / 15,  # Estimate 15 chars per second
            "format": "mp3"
        }
        
        # Write mock data as binary
        with open(output_path, 'wb') as f:
            f.write(b'ID3')  # MP3 header
            f.write(b'\x00' * 128)  # Some padding
            f.write(json.dumps(mock_audio_data).encode('utf-8'))
        
        return {
            "audio_path": output_path,
            "duration": mock_audio_data["duration"],
            "characters": len(text),
            "cost": 0.0,  # Free mock
            "is_mock": True
        }


class AIServiceOrchestrator:
    """Orchestrates multiple AI services for video generation"""
    
    def __init__(self, config: AIServiceConfig):
        self.config = config
        self.openai = OpenAIService(config)
        self.elevenlabs = ElevenLabsService(config)
        self.google_tts = GoogleTTSService(config)
        
    async def generate_complete_video_content(
        self,
        topic: str,
        style: str = "educational",
        length: str = "medium",
        voice_service: str = "elevenlabs",
        voice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate complete video content including script and audio"""
        try:
            total_cost = 0
            
            # 1. Generate script
            logger.info("Generating script...")
            script_result = await self.openai.generate_script(
                topic=topic,
                style=style,
                length=length
            )
            total_cost += script_result["cost"]
            
            # 2. Generate title
            logger.info("Generating title...")
            title_result = await self.openai.generate_title(topic)
            total_cost += title_result["cost"]
            
            # 3. Generate description
            logger.info("Generating description...")
            script_text = json.dumps(script_result["script"])
            description_result = await self.openai.generate_description(
                title=title_result["titles"][0],
                script=script_text
            )
            total_cost += description_result["cost"]
            
            # 4. Generate voice-over
            logger.info(f"Generating voice-over using {voice_service}...")
            
            # Extract script text for narration
            narration_text = self._extract_narration(script_result["script"])
            
            if voice_service == "elevenlabs":
                audio_result = await self.elevenlabs.text_to_speech(
                    text=narration_text,
                    voice_id=voice_id
                )
            else:  # google_tts
                audio_result = await self.google_tts.text_to_speech(
                    text=narration_text
                )
                
            total_cost += audio_result["cost"]
            
            # 5. Generate thumbnail prompt
            logger.info("Generating thumbnail prompt...")
            thumbnail_prompt = await self.openai.generate_thumbnail_prompt(
                title=title_result["titles"][0],
                style="vibrant"
            )
            
            return {
                "title": title_result["titles"][0],
                "alternative_titles": title_result["titles"][1:],
                "script": script_result["script"],
                "description": description_result["description"],
                "hashtags": description_result["hashtags"],
                "audio_path": audio_result["audio_path"],
                "thumbnail_prompt": thumbnail_prompt,
                "total_cost": round(total_cost, 2),
                "breakdown": {
                    "script_cost": script_result["cost"],
                    "title_cost": title_result["cost"],
                    "description_cost": description_result["cost"],
                    "audio_cost": audio_result["cost"]
                },
                "metadata": {
                    "topic": topic,
                    "style": style,
                    "length": length,
                    "voice_service": voice_service,
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Video content generation error: {e}")
            raise
            
    def _extract_narration(self, script: Dict) -> str:
        """Extract narration text from script structure"""
        parts = []
        
        if "hook" in script:
            parts.append(script["hook"])
            
        if "introduction" in script:
            parts.append(script["introduction"])
            
        if "main_sections" in script:
            for section in script["main_sections"]:
                parts.append(section.get("content", ""))
                
        if "conclusion" in script:
            parts.append(script["conclusion"])
            
        if "call_to_action" in script:
            parts.append(script["call_to_action"])
            
        return " ".join(parts)