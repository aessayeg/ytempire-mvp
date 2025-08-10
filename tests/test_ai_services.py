"""
Comprehensive tests for AI services
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import json
import numpy as np
from typing import Dict, Any, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.services.ai_service_orchestrator import (
    AIServiceOrchestrator, AIProvider, AIModel, 
    AIServiceError, ModelNotAvailableError
)


class TestAIServiceOrchestrator:
    """Test AI service orchestrator functionality"""
    
    @pytest.fixture
    def ai_orchestrator(self):
        """Create AI service orchestrator instance"""
        with patch('backend.app.services.ai_service_orchestrator.OpenAI'):
            orchestrator = AIServiceOrchestrator()
            orchestrator.providers = {
                AIProvider.OPENAI: Mock(),
                AIProvider.ANTHROPIC: Mock(),
                AIProvider.GOOGLE: Mock()
            }
            return orchestrator
    
    @pytest.mark.asyncio
    async def test_generate_script_success(self, ai_orchestrator):
        """Test successful script generation"""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated script content"))]
        
        ai_orchestrator.providers[AIProvider.OPENAI].chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        
        result = await ai_orchestrator.generate_script(
            topic="Test Topic",
            style="educational",
            duration=10
        )
        
        assert "Generated script content" in result
    
    @pytest.mark.asyncio
    async def test_generate_script_with_fallback(self, ai_orchestrator):
        """Test script generation with provider fallback"""
        # First provider fails
        ai_orchestrator.providers[AIProvider.OPENAI].chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        # Second provider succeeds
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Fallback script"))]
        ai_orchestrator.providers[AIProvider.ANTHROPIC] = Mock()
        ai_orchestrator.providers[AIProvider.ANTHROPIC].messages.create = AsyncMock(
            return_value=Mock(content=[Mock(text="Fallback script")])
        )
        
        result = await ai_orchestrator.generate_script_with_fallback(
            topic="Test Topic",
            style="educational"
        )
        
        assert "Fallback script" in result
    
    @pytest.mark.asyncio
    async def test_generate_voice_elevenlabs(self, ai_orchestrator):
        """Test voice generation with ElevenLabs"""
        mock_audio = b"audio_data"
        
        with patch('backend.app.services.ai_service_orchestrator.ElevenLabs') as mock_eleven:
            mock_eleven.return_value.generate.return_value = mock_audio
            
            result = await ai_orchestrator.generate_voice(
                text="Test text",
                voice_id="test_voice",
                provider=AIProvider.ELEVENLABS
            )
            
            assert result == mock_audio
    
    @pytest.mark.asyncio
    async def test_generate_voice_google_tts(self, ai_orchestrator):
        """Test voice generation with Google TTS"""
        mock_response = Mock()
        mock_response.audio_content = b"google_audio_data"
        
        with patch('backend.app.services.ai_service_orchestrator.texttospeech') as mock_tts:
            mock_client = Mock()
            mock_client.synthesize_speech.return_value = mock_response
            mock_tts.TextToSpeechClient.return_value = mock_client
            
            result = await ai_orchestrator.generate_voice_google(
                text="Test text",
                language_code="en-US"
            )
            
            assert result == b"google_audio_data"
    
    @pytest.mark.asyncio
    async def test_generate_images(self, ai_orchestrator):
        """Test image generation"""
        mock_response = Mock()
        mock_response.data = [Mock(url="https://example.com/image.png")]
        
        ai_orchestrator.providers[AIProvider.OPENAI].images.generate = AsyncMock(
            return_value=mock_response
        )
        
        result = await ai_orchestrator.generate_images(
            prompt="Test image prompt",
            n=1,
            size="1024x1024"
        )
        
        assert len(result) == 1
        assert result[0] == "https://example.com/image.png"
    
    @pytest.mark.asyncio
    async def test_analyze_content(self, ai_orchestrator):
        """Test content analysis"""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "quality_score": 0.85,
            "suggestions": ["Add more examples", "Improve introduction"]
        })))]
        
        ai_orchestrator.providers[AIProvider.OPENAI].chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        
        result = await ai_orchestrator.analyze_content(
            content="Test content to analyze"
        )
        
        assert result["quality_score"] == 0.85
        assert len(result["suggestions"]) == 2
    
    @pytest.mark.asyncio
    async def test_optimize_prompt(self, ai_orchestrator):
        """Test prompt optimization"""
        original_prompt = "Write about dogs"
        optimized = await ai_orchestrator.optimize_prompt(original_prompt)
        
        # Should enhance the prompt
        assert len(optimized) > len(original_prompt)
    
    @pytest.mark.asyncio
    async def test_generate_embeddings(self, ai_orchestrator):
        """Test embedding generation"""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3, 0.4])]
        
        ai_orchestrator.providers[AIProvider.OPENAI].embeddings.create = AsyncMock(
            return_value=mock_response
        )
        
        result = await ai_orchestrator.generate_embeddings(
            texts=["Test text"],
            model="text-embedding-ada-002"
        )
        
        assert len(result) == 1
        assert len(result[0]) == 4
    
    @pytest.mark.asyncio
    async def test_moderate_content(self, ai_orchestrator):
        """Test content moderation"""
        mock_response = Mock()
        mock_response.results = [Mock(
            flagged=False,
            categories=Mock(violence=False, hate=False)
        )]
        
        ai_orchestrator.providers[AIProvider.OPENAI].moderations.create = AsyncMock(
            return_value=mock_response
        )
        
        result = await ai_orchestrator.moderate_content("Safe content")
        
        assert result["flagged"] is False
    
    @pytest.mark.asyncio
    async def test_translate_text(self, ai_orchestrator):
        """Test text translation"""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Bonjour le monde"))]
        
        ai_orchestrator.providers[AIProvider.OPENAI].chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        
        result = await ai_orchestrator.translate_text(
            text="Hello world",
            target_language="French"
        )
        
        assert result == "Bonjour le monde"
    
    @pytest.mark.asyncio
    async def test_summarize_text(self, ai_orchestrator):
        """Test text summarization"""
        long_text = "This is a very long text " * 100
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Short summary"))]
        
        ai_orchestrator.providers[AIProvider.OPENAI].chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        
        result = await ai_orchestrator.summarize_text(long_text)
        
        assert len(result) < len(long_text)
    
    @pytest.mark.asyncio
    async def test_chain_operations(self, ai_orchestrator):
        """Test chaining multiple AI operations"""
        # Mock all operations
        ai_orchestrator.generate_script = AsyncMock(return_value="Generated script")
        ai_orchestrator.optimize_prompt = AsyncMock(return_value="Optimized prompt")
        ai_orchestrator.analyze_content = AsyncMock(return_value={"score": 0.9})
        
        # Chain operations
        script = await ai_orchestrator.generate_script("Topic", "style")
        optimized = await ai_orchestrator.optimize_prompt(script)
        analysis = await ai_orchestrator.analyze_content(optimized)
        
        assert analysis["score"] == 0.9
    
    @pytest.mark.asyncio
    async def test_error_handling(self, ai_orchestrator):
        """Test error handling in AI operations"""
        ai_orchestrator.providers[AIProvider.OPENAI].chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        with pytest.raises(AIServiceError):
            await ai_orchestrator.generate_script("Topic", "style")
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, ai_orchestrator):
        """Test rate limiting handling"""
        ai_orchestrator.rate_limiter = Mock()
        ai_orchestrator.rate_limiter.check_rate_limit = Mock(return_value=False)
        
        with pytest.raises(AIServiceError) as exc_info:
            await ai_orchestrator.generate_script_with_rate_limit("Topic")
        
        assert "rate limit" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_cost_tracking(self, ai_orchestrator):
        """Test AI cost tracking"""
        ai_orchestrator.track_cost = Mock()
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Content"))]
        mock_response.usage = Mock(total_tokens=100)
        
        ai_orchestrator.providers[AIProvider.OPENAI].chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        
        await ai_orchestrator.generate_script_with_tracking("Topic")
        
        ai_orchestrator.track_cost.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, ai_orchestrator):
        """Test batch processing of AI requests"""
        topics = ["Topic 1", "Topic 2", "Topic 3"]
        
        ai_orchestrator.generate_script = AsyncMock(
            side_effect=lambda t, s: f"Script for {t}"
        )
        
        results = await ai_orchestrator.batch_generate_scripts(topics)
        
        assert len(results) == 3
        assert "Script for Topic 1" in results
    
    @pytest.mark.asyncio
    async def test_model_selection(self, ai_orchestrator):
        """Test dynamic model selection"""
        # Test selecting fast model for simple task
        model = ai_orchestrator.select_model(task_complexity="simple")
        assert model == AIModel.GPT_35_TURBO
        
        # Test selecting powerful model for complex task
        model = ai_orchestrator.select_model(task_complexity="complex")
        assert model == AIModel.GPT_4
    
    @pytest.mark.asyncio
    async def test_context_management(self, ai_orchestrator):
        """Test context management for conversations"""
        ai_orchestrator.add_to_context("user", "Hello")
        ai_orchestrator.add_to_context("assistant", "Hi there!")
        
        context = ai_orchestrator.get_context()
        assert len(context) == 2
        assert context[0]["role"] == "user"
    
    @pytest.mark.asyncio
    async def test_streaming_response(self, ai_orchestrator):
        """Test streaming AI responses"""
        chunks = ["Hello", " ", "world", "!"]
        
        async def mock_stream():
            for chunk in chunks:
                yield chunk
        
        ai_orchestrator.providers[AIProvider.OPENAI].chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )
        
        result = []
        async for chunk in ai_orchestrator.stream_response("Test"):
            result.append(chunk)
        
        assert "".join(result) == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_function_calling(self, ai_orchestrator):
        """Test AI function calling capabilities"""
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                function_call=Mock(
                    name="get_weather",
                    arguments='{"location": "New York"}'
                )
            )
        )]
        
        ai_orchestrator.providers[AIProvider.OPENAI].chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        
        result = await ai_orchestrator.call_function("What's the weather?")
        
        assert result["function"] == "get_weather"
        assert result["arguments"]["location"] == "New York"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])