"""
End-to-End Video Generation Flow Tests
"""
import pytest
import requests
import time
import json
from datetime import datetime

class TestVideoGenerationFlow:
    """Test complete video generation pipeline"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.api_url = "http://localhost:8000/api/v1"
        self.n8n_url = "http://localhost:5678"
        self.ws_url = "ws://localhost:8000/ws"
        
        # Test user credentials
        self.test_user = {
            "email": "test@ytempire.com",
            "password": "TestPassword123!"
        }
        
        # Get auth token
        self.token = self._get_auth_token()
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    def _get_auth_token(self):
        """Helper to get authentication token"""
        response = requests.post(
            f"{self.api_url}/auth/login",
            json=self.test_user
        )
        if response.status_code == 200:
            return response.json()["access_token"]
        return None
    
    def test_create_channel(self):
        """Test channel creation"""
        channel_data = {
            "youtube_channel_id": "UC123456789",
            "channel_name": "Test Channel",
            "channel_handle": "@testchannel",
            "niche": "technology"
        }
        
        response = requests.post(
            f"{self.api_url}/channels",
            json=channel_data,
            headers=self.headers
        )
        
        assert response.status_code == 201
        channel = response.json()
        assert channel["channel_name"] == "Test Channel"
        return channel["id"]
    
    def test_generate_video(self):
        """Test video generation request"""
        # First create a channel
        channel_id = self.test_create_channel()
        
        # Request video generation
        video_data = {
            "channel_id": channel_id,
            "topic": "Top 10 Python Libraries for 2024",
            "style": "educational",
            "length": "medium",
            "target_audience": "developers"
        }
        
        response = requests.post(
            f"{self.api_url}/videos/generate",
            json=video_data,
            headers=self.headers
        )
        
        assert response.status_code == 202
        video = response.json()
        assert video["status"] == "pending"
        assert "id" in video
        return video["id"]
    
    def test_video_status_polling(self):
        """Test polling video generation status"""
        video_id = self.test_generate_video()
        
        # Poll status for up to 30 seconds
        max_attempts = 10
        for attempt in range(max_attempts):
            response = requests.get(
                f"{self.api_url}/videos/{video_id}/status",
                headers=self.headers
            )
            
            assert response.status_code == 200
            status = response.json()
            
            if status["status"] in ["completed", "failed"]:
                break
            
            time.sleep(3)
        
        # Check final status
        assert status["status"] in ["processing", "completed"]
        if status["status"] == "completed":
            assert "youtube_url" in status
            assert "total_cost" in status
    
    def test_video_queue(self):
        """Test video queue management"""
        # Generate multiple videos
        video_ids = []
        for i in range(3):
            video_data = {
                "channel_id": self.test_create_channel(),
                "topic": f"Test Video {i+1}",
                "style": "educational",
                "length": "short"
            }
            
            response = requests.post(
                f"{self.api_url}/videos/generate",
                json=video_data,
                headers=self.headers
            )
            
            if response.status_code == 202:
                video_ids.append(response.json()["id"])
        
        # Check queue
        response = requests.get(
            f"{self.api_url}/videos/queue",
            headers=self.headers
        )
        
        assert response.status_code == 200
        queue = response.json()
        assert "queued" in queue
        assert "processing" in queue
        assert "completed" in queue
    
    def test_cost_tracking(self):
        """Test cost tracking for generated videos"""
        video_id = self.test_generate_video()
        
        # Wait for video to process
        time.sleep(5)
        
        # Get cost breakdown
        response = requests.get(
            f"{self.api_url}/costs/videos/{video_id}",
            headers=self.headers
        )
        
        assert response.status_code == 200
        costs = response.json()
        assert "total" in costs
        assert "breakdown" in costs
        assert costs["total"] > 0
        assert costs["total"] < 5  # Should be under $5 per video
    
    def test_n8n_webhook_integration(self):
        """Test N8N workflow trigger"""
        webhook_data = {
            "channel_id": "test_channel",
            "topic": "Test Topic",
            "style": "educational",
            "webhook_callback_url": f"{self.api_url}/webhooks/callback"
        }
        
        # Trigger N8N workflow
        response = requests.post(
            f"{self.n8n_url}/webhook/video-generation",
            json=webhook_data
        )
        
        # N8N should accept the webhook
        assert response.status_code in [200, 202]
    
    def test_analytics_update(self):
        """Test analytics data update after video generation"""
        video_id = self.test_generate_video()
        
        # Wait for processing
        time.sleep(5)
        
        # Get analytics
        response = requests.get(
            f"{self.api_url}/analytics/videos/{video_id}",
            headers=self.headers
        )
        
        assert response.status_code == 200
        analytics = response.json()
        assert "video_id" in analytics
        assert "generation_time" in analytics
        assert "cost" in analytics