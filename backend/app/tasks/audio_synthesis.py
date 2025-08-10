"""
Audio Synthesis Tasks
Owner: Data Pipeline Engineer #1

AI-powered audio synthesis for video voiceovers.
Integrates with ElevenLabs, Azure TTS, and Google Cloud TTS.
"""

from celery import current_task
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import asyncio
import httpx
import os
from pathlib import Path

from app.core.celery_app import celery_app
from app.core.config import settings
from app.services.vault_service import VaultService
from app.utils.cost_calculator import CostCalculator
from app.core.metrics import metrics

logger = logging.getLogger(__name__)


class AudioSynthesisError(Exception):
    """Custom exception for audio synthesis errors."""
    pass


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 120})
def synthesize_audio_task(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesize audio from video script using AI voice services.
    
    Args:
        content_data: Content generation results from previous task
        
    Returns:
        Audio synthesis results with file paths and metadata
    """
    try:
        video_id = content_data['id']
        script = content_data.get('script', '')
        
        if not script:
            raise AudioSynthesisError("No script provided for audio synthesis")
        
        logger.info(f"Starting audio synthesis for video: {video_id}")
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'audio_synthesis', 'progress': 10, 'video_id': video_id}
        )
        
        # Initialize services
        vault_service = VaultService()
        cost_calculator = CostCalculator()
        
        # Get voice configuration
        voice_config = get_voice_configuration(content_data)
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'preparing_audio', 'progress': 20, 'video_id': video_id}
        )
        
        # Process script for better audio synthesis
        processed_script = preprocess_script_for_audio(script)
        
        # Split script into chunks for better processing
        script_chunks = split_script_into_chunks(processed_script)
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'synthesizing_chunks', 'progress': 30, 'video_id': video_id}
        )
        
        # Synthesize audio chunks
        audio_chunks = []
        total_cost = 0.0
        
        for i, chunk in enumerate(script_chunks):
            chunk_progress = 30 + (i / len(script_chunks)) * 50
            current_task.update_state(
                state='PROGRESS',
                meta={'stage': f'synthesizing_chunk_{i+1}', 'progress': chunk_progress, 'video_id': video_id}
            )
            
            chunk_result = await synthesize_chunk(
                chunk, voice_config, video_id, i, vault_service
            )
            
            audio_chunks.append(chunk_result)
            total_cost += chunk_result.get('cost', 0)
            
            # Check cost limits
            if total_cost > settings.MAX_AUDIO_COST:
                logger.warning(f"Audio synthesis cost exceeded limit: ${total_cost:.4f}")
                break
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'combining_audio', 'progress': 85, 'video_id': video_id}
        )
        
        # Combine audio chunks
        final_audio_path = combine_audio_chunks(audio_chunks, video_id)
        
        # Generate audio metadata
        audio_metadata = analyze_audio_quality(final_audio_path)
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'audio_complete', 'progress': 100, 'video_id': video_id}
        )
        
        # Prepare result
        result = {
            **content_data,
            'audio_file_path': final_audio_path,
            'audio_duration': audio_metadata['duration'],
            'audio_quality_score': audio_metadata['quality_score'],
            'audio_cost': total_cost,
            'voice_config': voice_config,
            'audio_synthesis_completed': True,
            'audio_metadata': {
                'chunks_processed': len(audio_chunks),
                'voice_service': voice_config.get('service'),
                'voice_id': voice_config.get('voice_id'),
                'synthesized_at': datetime.utcnow().isoformat(),
                'quality_metrics': audio_metadata
            }
        }
        
        # Record metrics
        metrics.record_api_cost(voice_config.get('service', 'unknown'), content_data['user_id'], total_cost)
        
        logger.info(f"Audio synthesis completed for video: {video_id}, cost: ${total_cost:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Audio synthesis failed for video {content_data.get('id')}: {str(e)}")
        raise


async def synthesize_chunk(
    text: str, 
    voice_config: Dict[str, Any], 
    video_id: str, 
    chunk_index: int,
    vault_service: VaultService
) -> Dict[str, Any]:
    """Synthesize a single audio chunk using the configured voice service."""
    
    service = voice_config.get('service', 'elevenlabs')
    
    try:
        if service == 'elevenlabs':
            return await synthesize_with_elevenlabs(text, voice_config, video_id, chunk_index, vault_service)
        elif service == 'azure':
            return await synthesize_with_azure(text, voice_config, video_id, chunk_index, vault_service)
        elif service == 'google':
            return await synthesize_with_google(text, voice_config, video_id, chunk_index, vault_service)
        else:
            raise AudioSynthesisError(f"Unsupported voice service: {service}")
            
    except Exception as e:
        logger.error(f"Failed to synthesize chunk {chunk_index}: {str(e)}")
        raise


async def synthesize_with_elevenlabs(
    text: str,
    voice_config: Dict[str, Any],
    video_id: str,
    chunk_index: int,
    vault_service: VaultService
) -> Dict[str, Any]:
    """Synthesize audio using ElevenLabs API."""
    
    try:
        # Get API key from Vault
        api_key = await vault_service.get_secret("api-keys", "elevenlabs_api_key")
        
        if not api_key:
            raise AudioSynthesisError("ElevenLabs API key not found")
        
        voice_id = voice_config.get('voice_id', 'default')
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        data = {
            "text": text,
            "model_id": voice_config.get('model_id', 'eleven_monolingual_v1'),
            "voice_settings": {
                "stability": voice_config.get('stability', 0.75),
                "similarity_boost": voice_config.get('similarity_boost', 0.75),
                "style": voice_config.get('style', 0.0),
                "use_speaker_boost": voice_config.get('speaker_boost', True)
            }
        }
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=data, headers=headers)
            
            if response.status_code != 200:
                raise AudioSynthesisError(f"ElevenLabs API error: {response.status_code} - {response.text}")
            
            # Save audio chunk
            audio_dir = Path(f"/tmp/audio/{video_id}")
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            chunk_path = audio_dir / f"chunk_{chunk_index:03d}.mp3"
            
            with open(chunk_path, "wb") as f:
                f.write(response.content)
            
            # Calculate cost (approximate based on character count)
            character_count = len(text)
            cost = (character_count / 1000) * 0.30  # $0.30 per 1k characters (approximate)
            
            return {
                'chunk_index': chunk_index,
                'file_path': str(chunk_path),
                'duration': estimate_audio_duration(text),
                'cost': cost,
                'character_count': character_count,
                'service': 'elevenlabs'
            }
            
    except Exception as e:
        logger.error(f"ElevenLabs synthesis failed for chunk {chunk_index}: {str(e)}")
        raise


async def synthesize_with_azure(
    text: str,
    voice_config: Dict[str, Any],
    video_id: str,
    chunk_index: int,
    vault_service: VaultService
) -> Dict[str, Any]:
    """Synthesize audio using Azure Cognitive Services Speech."""
    
    try:
        # Get API key from Vault
        api_key = await vault_service.get_secret("api-keys", "azure_speech_key")
        region = await vault_service.get_secret("api-keys", "azure_speech_region")
        
        if not api_key or not region:
            raise AudioSynthesisError("Azure Speech API key or region not found")
        
        url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        
        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-48khz-192kbitrate-mono-mp3"
        }
        
        voice_name = voice_config.get('voice_name', 'en-US-JennyNeural')
        
        ssml = f"""
        <speak version="1.0" xmlns="https://www.w3.org/2001/10/synthesis" xml:lang="en-US">
            <voice name="{voice_name}">
                <prosody rate="{voice_config.get('rate', 'medium')}" 
                         pitch="{voice_config.get('pitch', 'medium')}">
                    {text}
                </prosody>
            </voice>
        </speak>
        """
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, data=ssml.encode('utf-8'), headers=headers)
            
            if response.status_code != 200:
                raise AudioSynthesisError(f"Azure TTS API error: {response.status_code} - {response.text}")
            
            # Save audio chunk
            audio_dir = Path(f"/tmp/audio/{video_id}")
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            chunk_path = audio_dir / f"chunk_{chunk_index:03d}.mp3"
            
            with open(chunk_path, "wb") as f:
                f.write(response.content)
            
            # Calculate cost (approximate)
            character_count = len(text)
            cost = (character_count / 1000000) * 4.00  # $4 per 1M characters (approximate)
            
            return {
                'chunk_index': chunk_index,
                'file_path': str(chunk_path),
                'duration': estimate_audio_duration(text),
                'cost': cost,
                'character_count': character_count,
                'service': 'azure'
            }
            
    except Exception as e:
        logger.error(f"Azure TTS synthesis failed for chunk {chunk_index}: {str(e)}")
        raise


async def synthesize_with_google(
    text: str,
    voice_config: Dict[str, Any],
    video_id: str,
    chunk_index: int,
    vault_service: VaultService
) -> Dict[str, Any]:
    """Synthesize audio using Google Cloud Text-to-Speech."""
    
    try:
        from google.cloud import texttospeech
        
        # Initialize client with credentials from Vault
        credentials_json = await vault_service.get_secret("api-keys", "google_cloud_credentials")
        
        if not credentials_json:
            raise AudioSynthesisError("Google Cloud credentials not found")
        
        # Create client
        client = texttospeech.TextToSpeechClient.from_service_account_info(credentials_json)
        
        # Build synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Build voice selection
        voice = texttospeech.VoiceSelectionParams(
            language_code=voice_config.get('language_code', 'en-US'),
            name=voice_config.get('voice_name', 'en-US-Wavenet-D'),
            ssml_gender=texttospeech.SsmlVoiceGender[voice_config.get('gender', 'NEUTRAL')]
        )
        
        # Build audio config
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=voice_config.get('speaking_rate', 1.0),
            pitch=voice_config.get('pitch', 0.0)
        )
        
        # Perform synthesis
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Save audio chunk
        audio_dir = Path(f"/tmp/audio/{video_id}")
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        chunk_path = audio_dir / f"chunk_{chunk_index:03d}.mp3"
        
        with open(chunk_path, "wb") as f:
            f.write(response.audio_content)
        
        # Calculate cost (approximate)
        character_count = len(text)
        cost = (character_count / 1000000) * 4.00  # $4 per 1M characters (approximate)
        
        return {
            'chunk_index': chunk_index,
            'file_path': str(chunk_path),
            'duration': estimate_audio_duration(text),
            'cost': cost,
            'character_count': character_count,
            'service': 'google'
        }
        
    except Exception as e:
        logger.error(f"Google TTS synthesis failed for chunk {chunk_index}: {str(e)}")
        raise


def get_voice_configuration(content_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get voice configuration based on content and channel settings."""
    
    # Default configuration
    default_config = {
        'service': 'elevenlabs',  # Primary service
        'voice_id': 'default',
        'model_id': 'eleven_monolingual_v1',
        'stability': 0.75,
        'similarity_boost': 0.75,
        'style': 0.0,
        'speaker_boost': True
    }
    
    # Channel-specific voice settings
    channel_id = content_data.get('channel_id')
    if channel_id:
        # TODO: Get channel voice preferences from database
        # For now, return default
        pass
    
    # Content-based voice selection
    content_type = content_data.get('content_type', 'general')
    if content_type == 'educational':
        default_config.update({
            'stability': 0.85,
            'style': 0.2
        })
    elif content_type == 'entertainment':
        default_config.update({
            'stability': 0.65,
            'style': 0.8
        })
    
    return default_config


def preprocess_script_for_audio(script: str) -> str:
    """Preprocess script for better audio synthesis."""
    
    # Remove markdown formatting
    import re
    script = re.sub(r'\*\*(.*?)\*\*', r'\\1', script)  # Bold
    script = re.sub(r'\*(.*?)\*', r'\\1', script)      # Italic
    script = re.sub(r'`(.*?)`', r'\\1', script)        # Code
    
    # Add pauses for better pacing
    script = script.replace('.', '.')  # Short pause after sentences
    script = script.replace('!', '!')  # Emphasis
    script = script.replace('?', '?')  # Question intonation
    
    # Handle numbers and special characters
    script = re.sub(r'&', ' and ', script)
    script = re.sub(r'%', ' percent', script)
    script = re.sub(r'\$', ' dollars ', script)
    
    # Clean up extra whitespace
    script = re.sub(r'\s+', ' ', script)
    script = script.strip()
    
    return script


def split_script_into_chunks(script: str, max_chunk_size: int = 5000) -> List[str]:
    """Split script into smaller chunks for processing."""
    
    # Split by sentences first
    import re
    sentences = re.split(r'(?<=[.!?])\s+', script)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def combine_audio_chunks(audio_chunks: List[Dict[str, Any]], video_id: str) -> str:
    """Combine audio chunks into a single audio file."""
    
    try:
        from pydub import AudioSegment
        
        combined_audio = AudioSegment.empty()
        
        # Sort chunks by index
        sorted_chunks = sorted(audio_chunks, key=lambda x: x['chunk_index'])
        
        for chunk in sorted_chunks:
            chunk_audio = AudioSegment.from_mp3(chunk['file_path'])
            combined_audio += chunk_audio
            
            # Add small pause between chunks (100ms)
            pause = AudioSegment.silent(duration=100)
            combined_audio += pause
        
        # Export combined audio
        output_path = f"/tmp/audio/{video_id}/final_audio.mp3"
        combined_audio.export(output_path, format="mp3", bitrate="192k")
        
        # Clean up chunk files
        for chunk in sorted_chunks:
            try:
                os.remove(chunk['file_path'])
            except:
                pass
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to combine audio chunks: {str(e)}")
        raise


def analyze_audio_quality(audio_path: str) -> Dict[str, Any]:
    """Analyze audio quality metrics."""
    
    try:
        from pydub import AudioSegment
        
        audio = AudioSegment.from_mp3(audio_path)
        
        # Basic audio metrics
        duration = len(audio) / 1000.0  # Convert to seconds
        sample_rate = audio.frame_rate
        channels = audio.channels
        
        # Calculate RMS for volume analysis
        rms = audio.rms
        
        # Basic quality score based on technical parameters
        quality_score = 85.0  # Base score
        
        # Adjust based on sample rate
        if sample_rate >= 48000:
            quality_score += 5
        elif sample_rate < 22050:
            quality_score -= 10
        
        # Adjust based on RMS (volume consistency)
        if 1000 <= rms <= 5000:  # Good range
            quality_score += 5
        elif rms < 500:  # Too quiet
            quality_score -= 15
        elif rms > 8000:  # Too loud
            quality_score -= 10
        
        return {
            'duration': duration,
            'sample_rate': sample_rate,
            'channels': channels,
            'rms': rms,
            'quality_score': min(100, max(0, quality_score)),
            'file_size_mb': os.path.getsize(audio_path) / (1024 * 1024),
            'bitrate_kbps': audio.frame_rate * audio.frame_width * 8 * audio.channels / 1000
        }
        
    except Exception as e:
        logger.error(f"Audio quality analysis failed: {str(e)}")
        return {
            'duration': 0,
            'quality_score': 50,
            'error': str(e)
        }


def estimate_audio_duration(text: str, words_per_minute: int = 180) -> float:
    """Estimate audio duration based on text length."""
    
    word_count = len(text.split())
    duration_minutes = word_count / words_per_minute
    return duration_minutes * 60  # Convert to seconds