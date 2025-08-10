"""
Voice Synthesis Module for YouTube Videos
Integrates ElevenLabs and Google TTS for voice generation
"""
import asyncio
import base64
import json
import logging
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import aiohttp
import redis
import hashlib
from google.cloud import texttospeech
from elevenlabs import generate, Voice, VoiceSettings, set_api_key
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceProvider(Enum):
    """Voice synthesis providers"""
    ELEVENLABS = "elevenlabs"
    GOOGLE_TTS = "google_tts"
    AZURE = "azure"
    AWS_POLLY = "aws_polly"

class VoiceStyle(Enum):
    """Voice style options"""
    NATURAL = "natural"
    ENERGETIC = "energetic"
    CALM = "calm"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    CONVERSATIONAL = "conversational"
    NARRATIVE = "narrative"

@dataclass
class VoiceConfig:
    """Voice synthesis configuration"""
    provider: VoiceProvider
    voice_id: str
    style: VoiceStyle
    language: str
    speed: float = 1.0
    pitch: float = 0.0
    stability: float = 0.5
    similarity_boost: float = 0.75
    use_speaker_boost: bool = True

@dataclass
class AudioSegmentInfo:
    """Information about an audio segment"""
    text: str
    duration: float
    file_path: str
    timestamp_start: float
    timestamp_end: float

class VoiceSynthesizer:
    """
    Multi-provider voice synthesis with cost optimization
    """
    
    def __init__(
        self,
        elevenlabs_api_key: Optional[str] = None,
        google_credentials_path: Optional[str] = None,
        redis_host: str = 'localhost',
        redis_port: int = 6379
    ):
        self.elevenlabs_api_key = elevenlabs_api_key
        self.google_credentials_path = google_credentials_path
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.cache_ttl = 604800  # 7 days
        
        # Initialize providers
        if elevenlabs_api_key:
            set_api_key(elevenlabs_api_key)
        
        if google_credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_path
            self.google_client = texttospeech.TextToSpeechClient()
        else:
            self.google_client = None
        
        # Voice mappings
        self.voice_mappings = {
            VoiceStyle.NATURAL: {
                VoiceProvider.ELEVENLABS: "21m00Tcm4TlvDq8ikWAM",  # Rachel
                VoiceProvider.GOOGLE_TTS: "en-US-Neural2-C"
            },
            VoiceStyle.ENERGETIC: {
                VoiceProvider.ELEVENLABS: "AZnzlk1XvdvUeBnXmlld",  # Domi
                VoiceProvider.GOOGLE_TTS: "en-US-Neural2-A"
            },
            VoiceStyle.PROFESSIONAL: {
                VoiceProvider.ELEVENLABS: "pNInz6obpgDQGcFmaJgB",  # Adam
                VoiceProvider.GOOGLE_TTS: "en-US-Neural2-D"
            },
            VoiceStyle.FRIENDLY: {
                VoiceProvider.ELEVENLABS: "ThT5KcBeYPX3keUQqHPh",  # Dorothy
                VoiceProvider.GOOGLE_TTS: "en-US-Neural2-F"
            }
        }
        
        # Cost per 1000 characters
        self.cost_per_1k_chars = {
            VoiceProvider.ELEVENLABS: 0.30,
            VoiceProvider.GOOGLE_TTS: 0.016,
            VoiceProvider.AZURE: 0.016,
            VoiceProvider.AWS_POLLY: 0.004
        }
    
    async def synthesize_script(
        self,
        script_text: str,
        voice_config: VoiceConfig,
        output_path: str,
        quality_preset: str = 'balanced'
    ) -> Dict[str, Any]:
        """
        Synthesize complete script to audio
        
        Args:
            script_text: Complete script text
            voice_config: Voice configuration
            output_path: Output audio file path
            quality_preset: Quality/cost tradeoff
        
        Returns:
            Synthesis result with metadata
        """
        # Check cache
        cache_key = self._generate_cache_key(script_text, voice_config)
        cached_audio = await self._get_cached_audio(cache_key)
        if cached_audio:
            logger.info("Using cached audio")
            return cached_audio
        
        # Split script into segments for better processing
        segments = self._split_script(script_text)
        
        # Synthesize each segment
        audio_segments = []
        total_cost = 0
        
        for i, segment in enumerate(segments):
            logger.info(f"Synthesizing segment {i+1}/{len(segments)}")
            
            # Select provider based on quality preset
            provider = self._select_provider(quality_preset, voice_config)
            
            # Synthesize segment
            audio_data = await self._synthesize_segment(
                segment,
                voice_config,
                provider
            )
            
            if audio_data:
                audio_segments.append(audio_data)
                total_cost += self._calculate_cost(len(segment), provider)
        
        # Combine audio segments
        final_audio = self._combine_audio_segments(audio_segments)
        
        # Apply audio processing
        processed_audio = self._process_audio(final_audio, quality_preset)
        
        # Save to file
        processed_audio.export(output_path, format="mp3", bitrate="192k")
        
        # Calculate metadata
        result = {
            'output_path': output_path,
            'duration_seconds': len(processed_audio) / 1000,
            'file_size_mb': os.path.getsize(output_path) / (1024 * 1024),
            'cost': total_cost,
            'provider': provider.value,
            'voice_style': voice_config.style.value,
            'segments_count': len(segments),
            'quality_preset': quality_preset
        }
        
        # Cache result
        await self._cache_audio(cache_key, result)
        
        return result
    
    def _split_script(self, script_text: str, max_chars: int = 1000) -> List[str]:
        """Split script into manageable segments"""
        # Split by paragraphs first
        paragraphs = script_text.split('\n\n')
        
        segments = []
        current_segment = ""
        
        for paragraph in paragraphs:
            # If paragraph is too long, split by sentences
            if len(paragraph) > max_chars:
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    if len(current_segment) + len(sentence) < max_chars:
                        current_segment += sentence + ". "
                    else:
                        if current_segment:
                            segments.append(current_segment.strip())
                        current_segment = sentence + ". "
            else:
                if len(current_segment) + len(paragraph) < max_chars:
                    current_segment += paragraph + "\n\n"
                else:
                    if current_segment:
                        segments.append(current_segment.strip())
                    current_segment = paragraph + "\n\n"
        
        # Add remaining segment
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments
    
    def _select_provider(
        self,
        quality_preset: str,
        voice_config: VoiceConfig
    ) -> VoiceProvider:
        """Select voice provider based on quality preset and availability"""
        if quality_preset == 'quality' and self.elevenlabs_api_key:
            return VoiceProvider.ELEVENLABS
        elif quality_preset == 'fast' and self.google_client:
            return VoiceProvider.GOOGLE_TTS
        else:
            # Balanced: use ElevenLabs for important parts, Google for rest
            if self.elevenlabs_api_key:
                return VoiceProvider.ELEVENLABS
            elif self.google_client:
                return VoiceProvider.GOOGLE_TTS
            else:
                raise ValueError("No voice synthesis provider configured")
    
    async def _synthesize_segment(
        self,
        text: str,
        voice_config: VoiceConfig,
        provider: VoiceProvider
    ) -> Optional[AudioSegment]:
        """Synthesize a single text segment"""
        try:
            if provider == VoiceProvider.ELEVENLABS:
                return await self._synthesize_elevenlabs(text, voice_config)
            elif provider == VoiceProvider.GOOGLE_TTS:
                return await self._synthesize_google(text, voice_config)
            else:
                logger.error(f"Provider {provider} not implemented")
                return None
        except Exception as e:
            logger.error(f"Error synthesizing segment: {e}")
            # Fallback to alternative provider
            if provider == VoiceProvider.ELEVENLABS and self.google_client:
                logger.info("Falling back to Google TTS")
                return await self._synthesize_google(text, voice_config)
            return None
    
    async def _synthesize_elevenlabs(
        self,
        text: str,
        voice_config: VoiceConfig
    ) -> AudioSegment:
        """Synthesize using ElevenLabs API"""
        voice_id = self.voice_mappings.get(
            voice_config.style, {}
        ).get(VoiceProvider.ELEVENLABS, "21m00Tcm4TlvDq8ikWAM")
        
        # Generate audio
        audio = generate(
            text=text,
            voice=Voice(
                voice_id=voice_id,
                settings=VoiceSettings(
                    stability=voice_config.stability,
                    similarity_boost=voice_config.similarity_boost,
                    style=0.5,
                    use_speaker_boost=voice_config.use_speaker_boost
                )
            ),
            model="eleven_monolingual_v1"
        )
        
        # Convert to AudioSegment
        return AudioSegment.from_file(
            audio,
            format="mp3"
        )
    
    async def _synthesize_google(
        self,
        text: str,
        voice_config: VoiceConfig
    ) -> AudioSegment:
        """Synthesize using Google Text-to-Speech"""
        if not self.google_client:
            raise ValueError("Google TTS not configured")
        
        voice_name = self.voice_mappings.get(
            voice_config.style, {}
        ).get(VoiceProvider.GOOGLE_TTS, "en-US-Neural2-C")
        
        # Set up synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Build voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code=voice_config.language[:2] + "-US",
            name=voice_name
        )
        
        # Select audio config
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=voice_config.speed,
            pitch=voice_config.pitch
        )
        
        # Perform synthesis
        response = self.google_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Convert to AudioSegment
        audio_data = response.audio_content
        return AudioSegment.from_file(
            io.BytesIO(audio_data),
            format="mp3"
        )
    
    def _combine_audio_segments(
        self,
        segments: List[AudioSegment]
    ) -> AudioSegment:
        """Combine multiple audio segments with crossfade"""
        if not segments:
            return AudioSegment.empty()
        
        combined = segments[0]
        
        for segment in segments[1:]:
            # Add small crossfade for smooth transitions
            combined = combined.append(segment, crossfade=50)
        
        return combined
    
    def _process_audio(
        self,
        audio: AudioSegment,
        quality_preset: str
    ) -> AudioSegment:
        """Apply audio processing and enhancement"""
        # Normalize audio levels
        processed = normalize(audio)
        
        # Apply compression for consistent volume
        if quality_preset in ['balanced', 'quality']:
            processed = compress_dynamic_range(
                processed,
                threshold=-20.0,
                ratio=4.0
            )
        
        # Add subtle fade in/out
        processed = processed.fade_in(100).fade_out(100)
        
        # Adjust overall volume
        processed = processed + 3  # Increase by 3dB
        
        return processed
    
    def _calculate_cost(self, text_length: int, provider: VoiceProvider) -> float:
        """Calculate synthesis cost"""
        cost_per_char = self.cost_per_1k_chars.get(provider, 0.01) / 1000
        return text_length * cost_per_char
    
    def _generate_cache_key(
        self,
        text: str,
        voice_config: VoiceConfig
    ) -> str:
        """Generate cache key for audio"""
        key_data = {
            'text_hash': hashlib.md5(text.encode()).hexdigest(),
            'provider': voice_config.provider.value,
            'voice_id': voice_config.voice_id,
            'style': voice_config.style.value,
            'speed': voice_config.speed,
            'pitch': voice_config.pitch
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"voice:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def _get_cached_audio(self, cache_key: str) -> Optional[Dict]:
        """Retrieve cached audio metadata"""
        cached_data = self.redis_client.get(cache_key)
        if cached_data:
            try:
                return json.loads(cached_data)
            except Exception as e:
                logger.error(f"Error loading cached audio: {e}")
        return None
    
    async def _cache_audio(self, cache_key: str, metadata: Dict):
        """Cache audio metadata"""
        try:
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(metadata)
            )
        except Exception as e:
            logger.error(f"Error caching audio: {e}")
    
    async def create_voice_clone(
        self,
        sample_audio_path: str,
        voice_name: str,
        voice_description: str
    ) -> str:
        """
        Create custom voice clone using ElevenLabs
        
        Args:
            sample_audio_path: Path to voice sample
            voice_name: Name for the cloned voice
            voice_description: Description of the voice
        
        Returns:
            Voice ID of the cloned voice
        """
        if not self.elevenlabs_api_key:
            raise ValueError("ElevenLabs API key required for voice cloning")
        
        # This would use ElevenLabs voice cloning API
        # Implementation depends on ElevenLabs SDK version
        logger.info(f"Creating voice clone: {voice_name}")
        
        # Placeholder for actual implementation
        return "cloned_voice_id"
    
    async def add_background_music(
        self,
        voice_audio_path: str,
        music_path: str,
        output_path: str,
        music_volume: float = -20
    ) -> str:
        """
        Add background music to voice audio
        
        Args:
            voice_audio_path: Path to voice audio
            music_path: Path to background music
            output_path: Output file path
            music_volume: Music volume in dB (negative for quieter)
        
        Returns:
            Path to output file
        """
        # Load audio files
        voice = AudioSegment.from_file(voice_audio_path)
        music = AudioSegment.from_file(music_path)
        
        # Adjust music length to match voice
        if len(music) < len(voice):
            # Loop music if too short
            loops_needed = (len(voice) // len(music)) + 1
            music = music * loops_needed
        
        # Trim music to match voice length
        music = music[:len(voice)]
        
        # Adjust music volume
        music = music + music_volume
        
        # Overlay music on voice
        combined = voice.overlay(music)
        
        # Export result
        combined.export(output_path, format="mp3", bitrate="192k")
        
        return output_path


class VoiceSynthesisAPI:
    """FastAPI integration for voice synthesis"""
    
    def __init__(
        self,
        elevenlabs_api_key: Optional[str] = None,
        google_credentials_path: Optional[str] = None
    ):
        self.synthesizer = VoiceSynthesizer(
            elevenlabs_api_key=elevenlabs_api_key,
            google_credentials_path=google_credentials_path
        )
    
    async def synthesize(
        self,
        text: str,
        voice_style: str,
        language: str = "en",
        quality_preset: str = "balanced",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synthesize text to speech via API
        
        Args:
            text: Text to synthesize
            voice_style: Voice style
            language: Language code
            quality_preset: Quality preset
            **kwargs: Additional parameters
        
        Returns:
            Synthesis result
        """
        # Create voice config
        voice_config = VoiceConfig(
            provider=VoiceProvider.ELEVENLABS if quality_preset == "quality" else VoiceProvider.GOOGLE_TTS,
            voice_id="default",
            style=VoiceStyle(voice_style),
            language=language,
            speed=kwargs.get('speed', 1.0),
            pitch=kwargs.get('pitch', 0.0)
        )
        
        # Generate output path
        output_path = f"temp/audio_{hashlib.md5(text.encode()).hexdigest()}.mp3"
        
        # Synthesize
        result = await self.synthesizer.synthesize_script(
            text,
            voice_config,
            output_path,
            quality_preset
        )
        
        return {
            'audio_url': f"/audio/{os.path.basename(output_path)}",
            'duration': result['duration_seconds'],
            'cost': result['cost'],
            'provider': result['provider']
        }


# Initialize global instance
voice_api = VoiceSynthesisAPI()