"""
Mock data fixtures for testing
"""

from datetime import datetime, timedelta
from typing import Dict, Any
import uuid

def get_mock_user() -> Dict[str, Any]:
    """Get mock user data"""
    return {
        "id": str(uuid.uuid4()),
        "email": "test@example.com",
        "name": "Test User",
        "is_active": True,
        "is_superuser": False,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }

def get_mock_channel() -> Dict[str, Any]:
    """Get mock channel data"""
    return {
        "id": str(uuid.uuid4()),
        "youtube_channel_id": f"UC_{uuid.uuid4().hex[:20]}",
        "channel_name": "Test Channel",
        "channel_handle": "@testchannel",
        "niche": "technology",
        "target_audience": "developers",
        "subscriber_count": 1000,
        "video_count": 50,
        "created_at": datetime.now()
    }

def get_mock_video() -> Dict[str, Any]:
    """Get mock video data"""
    return {
        "id": str(uuid.uuid4()),
        "channel_id": str(uuid.uuid4()),
        "title": "Test Video Title",
        "description": "This is a test video description",
        "status": "completed",
        "youtube_video_id": f"v_{uuid.uuid4().hex[:11]}",
        "duration": 600,
        "views": 0,
        "likes": 0,
        "comments": 0,
        "created_at": datetime.now(),
        "published_at": datetime.now()
    }

def get_mock_script() -> Dict[str, Any]:
    """Get mock script data"""
    return {
        "id": str(uuid.uuid4()),
        "video_id": str(uuid.uuid4()),
        "content": "This is a test script content.",
        "hook": "Amazing hook to grab attention",
        "main_points": ["Point 1", "Point 2", "Point 3"],
        "call_to_action": "Subscribe for more content!",
        "word_count": 500,
        "created_at": datetime.now()
    }

def get_mock_cost_record() -> Dict[str, Any]:
    """Get mock cost record"""
    return {
        "id": str(uuid.uuid4()),
        "video_id": str(uuid.uuid4()),
        "service": "openai",
        "operation": "script_generation",
        "cost": 0.05,
        "tokens_used": 1500,
        "created_at": datetime.now()
    }

def get_mock_payment() -> Dict[str, Any]:
    """Get mock payment data"""
    return {
        "id": str(uuid.uuid4()),
        "user_id": str(uuid.uuid4()),
        "amount": 29.99,
        "currency": "USD",
        "status": "succeeded",
        "stripe_payment_id": f"pi_{uuid.uuid4().hex}",
        "created_at": datetime.now()
    }

def get_mock_analytics() -> Dict[str, Any]:
    """Get mock analytics data"""
    return {
        "channel_id": str(uuid.uuid4()),
        "date": datetime.now().date(),
        "views": 1000,
        "watch_time": 50000,
        "subscribers_gained": 10,
        "revenue": 5.50,
        "ctr": 0.05,
        "avg_view_duration": 50
    }

def get_mock_openai_response() -> Dict[str, Any]:
    """Get mock OpenAI API response"""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a mock response from OpenAI"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300
        }
    }

def get_mock_youtube_response() -> Dict[str, Any]:
    """Get mock YouTube API response"""
    return {
        "kind": "youtube#video",
        "etag": "mock_etag",
        "id": "mock_video_id",
        "snippet": {
            "publishedAt": datetime.now().isoformat(),
            "channelId": "mock_channel_id",
            "title": "Mock Video Title",
            "description": "Mock video description",
            "thumbnails": {
                "default": {"url": "https://example.com/thumb.jpg"}
            }
        },
        "statistics": {
            "viewCount": "1000",
            "likeCount": "50",
            "commentCount": "10"
        }
    }

def get_mock_elevenlabs_response() -> Dict[str, Any]:
    """Get mock ElevenLabs API response"""
    return {
        "audio_url": "https://example.com/audio.mp3",
        "characters_used": 500,
        "voice_id": "mock_voice_id",
        "model_id": "eleven_monolingual_v1"
    }

def get_mock_webhook_event() -> Dict[str, Any]:
    """Get mock webhook event"""
    return {
        "id": str(uuid.uuid4()),
        "event_type": "video.completed",
        "payload": {
            "video_id": str(uuid.uuid4()),
            "status": "completed",
            "url": "https://youtube.com/watch?v=mock"
        },
        "timestamp": datetime.now().isoformat()
    }