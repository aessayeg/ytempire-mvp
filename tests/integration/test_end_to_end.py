"""
End-to-End Integration Testing Suite for YTEmpire MVP
Tests complete user flows and system integration
"""
import pytest
import asyncio
from typing import Dict, Any
import httpx
from datetime import datetime
import json
import os
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import services
from backend.app.services.youtube_service import YouTubeService
from backend.app.services.ai_services import AIServiceOrchestrator
from backend.app.services.vector_database_deployed import VectorDatabase
from backend.app.services.metrics_pipeline_operational import MetricsPipeline
from data.feature_store.realtime_feature_store import RealtimeFeatureStore
from data.data_lake.data_lake_service import DataLakeService

class TestEndToEndIntegration:
    """Complete end-to-end integration tests"""
    
    @pytest.fixture
    async def api_client(self):
        """Create test API client"""
        async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
            yield client
            
    @pytest.fixture
    async def test_user(self):
        """Create test user data"""
        return {
            "email": "test@ytempire.com",
            "password": "Test123!@#",
            "name": "Test User",
            "role": "creator"
        }
        
    @pytest.fixture
    async def test_channel(self):
        """Create test channel data"""
        return {
            "name": "Test Channel",
            "description": "Automated test channel",
            "niche": "technology",
            "target_audience": "tech enthusiasts",
            "upload_frequency": "daily"
        }
        
    @pytest.mark.asyncio
    async def test_user_registration_and_authentication(self, api_client, test_user):
        """Test 1: User registration and authentication flow"""
        # Register user
        register_response = await api_client.post(
            "/api/v1/auth/register",
            json=test_user
        )
        assert register_response.status_code == 201
        user_data = register_response.json()
        assert "id" in user_data
        assert user_data["email"] == test_user["email"]
        
        # Login
        login_response = await api_client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user["email"],
                "password": test_user["password"]
            }
        )
        assert login_response.status_code == 200
        tokens = login_response.json()
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        
        # Verify token
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        profile_response = await api_client.get(
            "/api/v1/users/profile",
            headers=headers
        )
        assert profile_response.status_code == 200
        profile = profile_response.json()
        assert profile["email"] == test_user["email"]
        
        return tokens["access_token"]
        
    @pytest.mark.asyncio
    async def test_channel_creation_and_configuration(self, api_client, test_channel):
        """Test 2: Channel creation and configuration"""
        # Assume user is authenticated
        token = "test_token"  # Would come from previous test
        headers = {"Authorization": f"Bearer {token}"}
        
        # Create channel
        create_response = await api_client.post(
            "/api/v1/channels",
            json=test_channel,
            headers=headers
        )
        assert create_response.status_code == 201
        channel_data = create_response.json()
        assert "id" in channel_data
        assert channel_data["name"] == test_channel["name"]
        
        # Update channel settings
        update_data = {
            "upload_frequency": "weekly",
            "monetization_enabled": True
        }
        update_response = await api_client.patch(
            f"/api/v1/channels/{channel_data['id']}",
            json=update_data,
            headers=headers
        )
        assert update_response.status_code == 200
        
        # Get channel details
        get_response = await api_client.get(
            f"/api/v1/channels/{channel_data['id']}",
            headers=headers
        )
        assert get_response.status_code == 200
        channel = get_response.json()
        assert channel["upload_frequency"] == "weekly"
        
        return channel_data["id"]
        
    @pytest.mark.asyncio
    async def test_video_generation_request_mock(self, api_client):
        """Test 3: Video generation request (mocked)"""
        token = "test_token"
        channel_id = "test_channel_id"
        headers = {"Authorization": f"Bearer {token}"}
        
        # Create video generation request
        video_request = {
            "channel_id": channel_id,
            "topic": "Latest AI Trends in 2024",
            "style": "educational",
            "duration": "10-15 minutes",
            "voice_type": "professional",
            "include_music": True
        }
        
        with patch("backend.app.services.ai_services.AIServiceOrchestrator.generate_complete_video_content") as mock_generate:
            mock_generate.return_value = {
                "script": "AI is transforming the world...",
                "audio_url": "https://storage.example.com/audio/test.mp3",
                "thumbnail_url": "https://storage.example.com/thumbnails/test.jpg",
                "cost_breakdown": {
                    "script_generation": 0.50,
                    "voice_synthesis": 1.00,
                    "thumbnail_generation": 0.25,
                    "total": 1.75
                }
            }
            
            response = await api_client.post(
                "/api/v1/videos/generate",
                json=video_request,
                headers=headers
            )
            assert response.status_code == 202
            job_data = response.json()
            assert "job_id" in job_data
            assert "status" in job_data
            
            # Check job status
            status_response = await api_client.get(
                f"/api/v1/jobs/{job_data['job_id']}",
                headers=headers
            )
            assert status_response.status_code == 200
            
        return job_data["job_id"]
        
    @pytest.mark.asyncio
    async def test_cost_tracking_verification(self):
        """Test 4: Cost tracking verification"""
        # Initialize metrics pipeline
        metrics = MetricsPipeline()
        
        # Track video generation costs
        await metrics.track_metric(
            name="api_cost",
            value=0.50,
            tags={"service": "openai", "operation": "script_generation"}
        )
        await metrics.track_metric(
            name="api_cost",
            value=1.00,
            tags={"service": "elevenlabs", "operation": "voice_synthesis"}
        )
        await metrics.track_metric(
            name="api_cost",
            value=0.25,
            tags={"service": "openai", "operation": "thumbnail_generation"}
        )
        
        # Verify cost aggregation
        start_time = datetime.utcnow().replace(hour=0, minute=0, second=0)
        total_cost = await metrics._get_metric_sum("api_cost", start_time, datetime.utcnow())
        assert total_cost == 1.75
        
        # Verify cost breakdown by service
        dashboard_metrics = await metrics.get_dashboard_metrics()
        assert "total_cost_24h" in dashboard_metrics
        
    @pytest.mark.asyncio
    async def test_dashboard_data_display(self, api_client):
        """Test 5: Dashboard data display"""
        token = "test_token"
        headers = {"Authorization": f"Bearer {token}"}
        
        # Get dashboard metrics
        dashboard_response = await api_client.get(
            "/api/v1/dashboard/metrics",
            headers=headers
        )
        assert dashboard_response.status_code == 200
        metrics = dashboard_response.json()
        
        # Verify essential metrics
        assert "videos_generated_24h" in metrics
        assert "total_revenue_24h" in metrics
        assert "total_cost_24h" in metrics
        assert "active_channels" in metrics
        assert "queue_size" in metrics
        
        # Get analytics data
        analytics_response = await api_client.get(
            "/api/v1/analytics/overview",
            headers=headers
        )
        assert analytics_response.status_code == 200
        analytics = analytics_response.json()
        
        assert "weekly_trends" in analytics
        assert "top_performing_videos" in analytics
        assert "channel_growth" in analytics
        
    @pytest.mark.asyncio
    async def test_ai_services_integration(self):
        """Test AI services integration"""
        orchestrator = AIServiceOrchestrator()
        
        # Test script generation
        with patch("openai.ChatCompletion.create") as mock_openai:
            mock_openai.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Test script"))]
            )
            
            script = await orchestrator.openai_service.generate_script(
                topic="Test Topic",
                style="educational"
            )
            assert script == "Test script"
            
        # Test voice synthesis
        with patch("backend.app.services.ai_services.ElevenLabsService.synthesize_speech") as mock_voice:
            mock_voice.return_value = b"audio_data"
            
            audio = await orchestrator.elevenlabs_service.synthesize_speech(
                text="Test text",
                voice_id="test_voice"
            )
            assert audio == b"audio_data"
            
    @pytest.mark.asyncio
    async def test_vector_database_operations(self):
        """Test vector database operations"""
        vector_db = VectorDatabase()
        
        # Index test video
        success = await vector_db.index_video(
            video_id="test_video_1",
            title="Introduction to Machine Learning",
            description="Learn the basics of ML",
            script="Machine learning is a subset of AI...",
            tags=["AI", "ML", "Education"],
            metadata={"duration": 600, "views": 0}
        )
        assert success
        
        # Search similar videos
        results = await vector_db.search_similar_videos(
            query="artificial intelligence basics",
            limit=5
        )
        assert len(results) > 0
        assert results[0]["metadata"]["title"] == "Introduction to Machine Learning"
        
    @pytest.mark.asyncio
    async def test_feature_store_operations(self):
        """Test feature store operations"""
        feature_store = RealtimeFeatureStore()
        
        # Compute features
        features = await feature_store.compute_features(
            entity_id="test_video_1",
            feature_names=["video_view_velocity", "video_engagement_rate"],
            input_data={
                "views": 1000,
                "hours_since_upload": 24,
                "likes": 50,
                "comments": 10
            }
        )
        
        assert "video_view_velocity" in features
        assert features["video_view_velocity"] == 1000 / 24
        assert "video_engagement_rate" in features
        assert features["video_engagement_rate"] == (50 + 10) / 1000
        
    @pytest.mark.asyncio
    async def test_data_lake_operations(self):
        """Test data lake operations"""
        data_lake = DataLakeService()
        
        # Ingest test data
        test_data = {
            "video_id": ["v1", "v2", "v3"],
            "views": [1000, 2000, 3000],
            "likes": [50, 100, 150]
        }
        
        dataset_id = await data_lake.ingest_data(
            data=test_data,
            dataset_name="test_analytics",
            zone=data_lake.DataZone.BRONZE
        )
        
        assert dataset_id is not None
        
        # Read data back
        df = await data_lake.read_data(dataset_id)
        assert len(df) == 3
        assert df["views"].sum() == 6000
        
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete video generation workflow"""
        # 1. Initialize services
        ai_orchestrator = AIServiceOrchestrator()
        vector_db = VectorDatabase()
        metrics = MetricsPipeline()
        feature_store = RealtimeFeatureStore()
        
        # 2. Generate video content
        with patch.multiple(
            "backend.app.services.ai_services",
            generate_script=MagicMock(return_value="Test script"),
            synthesize_speech=MagicMock(return_value=b"audio"),
            generate_thumbnail=MagicMock(return_value="thumbnail_url")
        ):
            content = await ai_orchestrator.generate_complete_video_content(
                topic="Test Topic",
                style="educational"
            )
            
            assert content["script"] == "Test script"
            assert "cost_breakdown" in content
            
        # 3. Index in vector database
        await vector_db.index_video(
            video_id="workflow_test",
            title="Workflow Test Video",
            description="Testing complete workflow",
            script=content["script"],
            tags=["test"],
            metadata={"cost": content["cost_breakdown"]["total"]}
        )
        
        # 4. Track metrics
        await metrics.track_metric(
            name="videos_generated",
            value=1,
            tags={"channel_id": "test_channel", "status": "success"}
        )
        
        # 5. Store features
        await feature_store.store_features(
            entity_id="workflow_test",
            features={
                "generation_cost": content["cost_breakdown"]["total"],
                "script_length": len(content["script"])
            }
        )
        
        # Verify workflow completion
        search_results = await vector_db.search_similar_videos("Workflow Test", limit=1)
        assert len(search_results) > 0
        
        features = await feature_store.compute_features(
            entity_id="workflow_test",
            feature_names=["generation_cost"]
        )
        assert features["generation_cost"] > 0


class TestSystemIntegration:
    """System-level integration tests"""
    
    @pytest.mark.asyncio
    async def test_api_gateway_frontend_communication(self):
        """Test API Gateway ↔ Frontend communication"""
        # This would test CORS, authentication headers, WebSocket connections
        pass
        
    @pytest.mark.asyncio
    async def test_backend_ai_services_integration(self):
        """Test Backend ↔ AI Services integration"""
        # Test service discovery, retry logic, circuit breakers
        pass
        
    @pytest.mark.asyncio
    async def test_data_pipeline_metrics_collection(self):
        """Test Data pipeline ↔ Metrics collection"""
        # Test data flow from ingestion to metrics
        pass
        
    @pytest.mark.asyncio
    async def test_monitoring_alerting_systems(self):
        """Test Monitoring and alerting systems"""
        # Test Prometheus scraping, Grafana dashboards, alert rules
        pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])