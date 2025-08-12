"""
Beta User End-to-End Test Suite
Comprehensive automated testing for beta user scenarios
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import aiohttp
from faker import Faker
from playwright.async_api import async_playwright, Page, Browser
import smtplib
from email.mime.text import MIMEText
from dataclasses import dataclass

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
FRONTEND_URL = "http://localhost:3000"
fake = Faker()

@dataclass
class BetaUser:
    """Beta user test data"""
    email: str
    password: str
    name: str
    company: str
    token: str = None
    channels: List[Dict] = None


class BetaTestResults:
    """Track beta test results"""
    
    def __init__(self):
        self.results = {
            "scenarios_passed": 0,
            "scenarios_failed": 0,
            "performance_metrics": {},
            "user_feedback": [],
            "issues_found": [],
            "suggestions": []
        }
    
    def record_scenario(self, name: str, passed: bool, duration: float, notes: str = ""):
        """Record scenario result"""
        self.results["scenarios_passed" if passed else "scenarios_failed"] += 1
        self.results["performance_metrics"][name] = duration
        if notes:
            self.results["user_feedback"].append(f"{name}: {notes}")
    
    def add_issue(self, severity: str, description: str, scenario: str):
        """Add issue found during testing"""
        self.results["issues_found"].append({
            "severity": severity,
            "description": description,
            "scenario": scenario,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def generate_report(self) -> Dict:
        """Generate final test report"""
        total_scenarios = self.results["scenarios_passed"] + self.results["scenarios_failed"]
        return {
            **self.results,
            "success_rate": (self.results["scenarios_passed"] / total_scenarios) * 100 if total_scenarios > 0 else 0,
            "average_performance": sum(self.results["performance_metrics"].values()) / len(self.results["performance_metrics"]) if self.results["performance_metrics"] else 0
        }


class BetaUserTestSuite:
    """Main beta user test suite"""
    
    def __init__(self):
        self.results = BetaTestResults()
        self.test_users = []
        self.session = None
    
    async def setup(self):
        """Setup test environment"""
        self.session = aiohttp.ClientSession()
        
        # Create test users
        for i in range(5):
            user = BetaUser(
                email=f"beta_user_{i}@example.com",
                password="BetaTest123!",
                name=f"Beta User {i}",
                company=f"Test Company {i}"
            )
            self.test_users.append(user)
    
    async def cleanup(self):
        """Cleanup test environment"""
        if self.session:
            await self.session.close()
    
    async def register_beta_user(self, user: BetaUser) -> bool:
        """Register a beta user"""
        start_time = time.time()
        
        try:
            async with self.session.post(f"{API_BASE_URL}/auth/register", json={
                "email": user.email,
                "password": user.password,
                "name": user.name,
                "company": user.company,
                "beta_code": "BETA2024"
            }) as response:
                duration = time.time() - start_time
                
                if response.status == 201:
                    data = await response.json()
                    user.token = data.get("access_token")
                    self.results.record_scenario(
                        f"User Registration - {user.name}",
                        True,
                        duration,
                        "Registration completed successfully"
                    )
                    return True
                else:
                    self.results.record_scenario(
                        f"User Registration - {user.name}",
                        False,
                        duration,
                        f"Registration failed: {response.status}"
                    )
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            self.results.add_issue("HIGH", f"Registration exception: {str(e)}", "User Registration")
            return False


class TestBetaUserOnboarding:
    """Test beta user onboarding scenarios"""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.suite = BetaUserTestSuite()
        await self.suite.setup()
        yield
        await self.suite.cleanup()
    
    @pytest.mark.asyncio
    async def test_079_beta_user_registration_flow(self):
        """Test complete beta user registration flow"""
        for user in self.suite.test_users:
            success = await self.suite.register_beta_user(user)
            assert success, f"Registration failed for {user.name}"
    
    @pytest.mark.asyncio
    async def test_080_welcome_tour_completion(self):
        """Test welcome tour for new beta users"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Login as first beta user
            user = self.suite.test_users[0]
            await page.goto(f"{FRONTEND_URL}/login")
            await page.fill('input[name="email"]', user.email)
            await page.fill('input[name="password"]', user.password)
            await page.click('button[type="submit"]')
            
            # Check welcome tour starts
            await page.wait_for_selector('.welcome-tour')
            
            # Complete tour steps
            tour_steps = await page.query_selector_all('.tour-step')
            for step in tour_steps:
                await step.click()
                await page.wait_for_timeout(1000)
            
            # Verify tour completion
            await page.wait_for_selector('.tour-complete')
            
            await browser.close()
    
    @pytest.mark.asyncio
    async def test_081_youtube_channel_connection(self):
        """Test YouTube channel connection for beta users"""
        user = self.suite.test_users[0]
        headers = {"Authorization": f"Bearer {user.token}"}
        
        # Simulate OAuth connection (in real test, would use actual OAuth)
        mock_channels = [
            {
                "channel_id": "UC123456789",
                "title": "Test Gaming Channel",
                "subscriber_count": 1500,
                "niche": "gaming"
            },
            {
                "channel_id": "UC987654321",
                "title": "Tech Reviews",
                "subscriber_count": 850,
                "niche": "technology"
            }
        ]
        
        async with self.suite.session.post(f"{API_BASE_URL}/channels/connect",
            headers=headers,
            json={"channels": mock_channels}) as response:
            
            assert response.status == 200
            data = await response.json()
            assert len(data["connected_channels"]) == 2
            user.channels = data["connected_channels"]
    
    @pytest.mark.asyncio
    async def test_082_onboarding_preferences_setup(self):
        """Test user preferences setup during onboarding"""
        user = self.suite.test_users[0]
        headers = {"Authorization": f"Bearer {user.token}"}
        
        preferences = {
            "content_style": "educational",
            "upload_frequency": "daily",
            "voice_preference": "professional_female",
            "niche_focus": ["technology", "programming"],
            "budget_limit": 100,
            "quality_threshold": 85
        }
        
        async with self.suite.session.put(f"{API_BASE_URL}/user/preferences",
            headers=headers,
            json=preferences) as response:
            
            assert response.status == 200
            data = await response.json()
            assert data["preferences"]["content_style"] == "educational"


class TestBetaVideoGeneration:
    """Test video generation scenarios for beta users"""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.suite = BetaUserTestSuite()
        await self.suite.setup()
        # Register users for video generation tests
        for user in self.suite.test_users:
            await self.suite.register_beta_user(user)
        yield
        await self.suite.cleanup()
    
    @pytest.mark.asyncio
    async def test_083_single_video_generation(self):
        """Test single video generation with beta user limits"""
        user = self.suite.test_users[0]
        headers = {"Authorization": f"Bearer {user.token}"}
        
        start_time = time.time()
        
        async with self.suite.session.post(f"{API_BASE_URL}/videos/generate",
            headers=headers,
            json={
                "topic": "10 Python Tips for Beginners",
                "style": "educational",
                "duration": 8,
                "channel_id": "UC123456789",
                "beta_features": ["trending_optimization", "ai_thumbnail"]
            }) as response:
            
            assert response.status == 202
            data = await response.json()
            video_id = data["video_id"]
            
            # Monitor generation progress
            for attempt in range(30):  # 5 minute timeout
                await asyncio.sleep(10)
                
                async with self.suite.session.get(f"{API_BASE_URL}/videos/{video_id}/status",
                    headers=headers) as status_response:
                    
                    status_data = await status_response.json()
                    if status_data["status"] == "completed":
                        duration = time.time() - start_time
                        self.suite.results.record_scenario(
                            "Single Video Generation",
                            True,
                            duration,
                            f"Generated in {duration:.2f}s with cost ${status_data['cost']:.2f}"
                        )
                        assert status_data["cost"] < 3.0
                        break
                    elif status_data["status"] == "failed":
                        self.suite.results.add_issue(
                            "HIGH",
                            f"Video generation failed: {status_data.get('error')}",
                            "Single Video Generation"
                        )
                        assert False, "Video generation failed"
    
    @pytest.mark.asyncio
    async def test_084_batch_video_generation(self):
        """Test batch video generation for beta users"""
        user = self.suite.test_users[1]
        headers = {"Authorization": f"Bearer {user.token}"}
        
        topics = [
            "JavaScript Array Methods",
            "CSS Grid Layout Tutorial",
            "React Hooks Explained",
            "Node.js Best Practices",
            "Database Design Principles"
        ]
        
        # Submit batch request
        async with self.suite.session.post(f"{API_BASE_URL}/videos/batch",
            headers=headers,
            json={
                "topics": topics,
                "common_settings": {
                    "style": "educational",
                    "duration": 6,
                    "channel_id": "UC123456789"
                },
                "beta_optimization": True
            }) as response:
            
            assert response.status == 202
            data = await response.json()
            batch_id = data["batch_id"]
            
            # Monitor batch progress
            completed_videos = 0
            for attempt in range(60):  # 10 minute timeout
                await asyncio.sleep(10)
                
                async with self.suite.session.get(f"{API_BASE_URL}/batches/{batch_id}/status",
                    headers=headers) as status_response:
                    
                    batch_data = await status_response.json()
                    completed_videos = batch_data["completed_count"]
                    
                    if batch_data["status"] == "completed":
                        assert completed_videos == 5
                        assert batch_data["total_cost"] < 15.0  # $3 per video max
                        break
    
    @pytest.mark.asyncio
    async def test_085_trending_topic_optimization(self):
        """Test beta feature: trending topic optimization"""
        user = self.suite.test_users[2]
        headers = {"Authorization": f"Bearer {user.token}"}
        
        # Get trending topics
        async with self.suite.session.get(f"{API_BASE_URL}/trending/topics",
            headers=headers,
            params={"niche": "technology", "beta_insights": True}) as response:
            
            assert response.status == 200
            data = await response.json()
            assert len(data["topics"]) > 0
            
            trending_topic = data["topics"][0]
            
            # Generate video from trending topic
            async with self.suite.session.post(f"{API_BASE_URL}/videos/generate",
                headers=headers,
                json={
                    "trending_topic_id": trending_topic["id"],
                    "custom_angle": "beginner-friendly approach",
                    "seo_optimization": "aggressive",
                    "beta_features": ["trend_analysis", "competitor_insights"]
                }) as gen_response:
                
                assert gen_response.status == 202
                gen_data = await gen_response.json()
                assert gen_data["seo_score"] > 90


class TestBetaAnalytics:
    """Test analytics and monitoring for beta users"""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.suite = BetaUserTestSuite()
        await self.suite.setup()
        for user in self.suite.test_users:
            await self.suite.register_beta_user(user)
        yield
        await self.suite.cleanup()
    
    @pytest.mark.asyncio
    async def test_086_real_time_analytics_dashboard(self):
        """Test real-time analytics for beta users"""
        user = self.suite.test_users[0]
        headers = {"Authorization": f"Bearer {user.token}"}
        
        # Access real-time dashboard
        async with self.suite.session.get(f"{API_BASE_URL}/analytics/realtime",
            headers=headers,
            params={"beta_features": "advanced_insights"}) as response:
            
            assert response.status == 200
            data = await response.json()
            
            # Verify real-time data structure
            assert "current_viewers" in data
            assert "live_engagement" in data
            assert "revenue_today" in data
            assert "beta_insights" in data
            
            # Test beta-specific insights
            beta_insights = data["beta_insights"]
            assert "optimization_suggestions" in beta_insights
            assert "trend_predictions" in beta_insights
    
    @pytest.mark.asyncio
    async def test_087_performance_benchmarking(self):
        """Test performance benchmarking against other channels"""
        user = self.suite.test_users[1]
        headers = {"Authorization": f"Bearer {user.token}"}
        
        async with self.suite.session.get(f"{API_BASE_URL}/analytics/benchmark",
            headers=headers,
            params={
                "niche": "technology",
                "subscriber_range": "1k-10k",
                "beta_comparison": True
            }) as response:
            
            assert response.status == 200
            data = await response.json()
            
            assert "percentile_ranking" in data
            assert "improvement_areas" in data
            assert "top_performer_insights" in data
    
    @pytest.mark.asyncio
    async def test_088_custom_reporting(self):
        """Test custom report generation for beta users"""
        user = self.suite.test_users[2]
        headers = {"Authorization": f"Bearer {user.token}"}
        
        # Create custom report
        report_config = {
            "name": "Weekly Performance Report",
            "metrics": [
                "total_views",
                "engagement_rate",
                "revenue",
                "subscriber_growth",
                "top_videos"
            ],
            "date_range": "last_7_days",
            "format": "pdf",
            "schedule": "weekly",
            "beta_insights": True
        }
        
        async with self.suite.session.post(f"{API_BASE_URL}/reports/custom",
            headers=headers,
            json=report_config) as response:
            
            assert response.status == 201
            data = await response.json()
            report_id = data["report_id"]
            
            # Check report generation
            async with self.suite.session.get(f"{API_BASE_URL}/reports/{report_id}/status",
                headers=headers) as status_response:
                
                status_data = await status_response.json()
                assert status_data["status"] in ["generating", "completed"]


class TestBetaFeedback:
    """Collect and analyze beta user feedback"""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.suite = BetaUserTestSuite()
        await self.suite.setup()
        yield
        await self.suite.cleanup()
    
    @pytest.mark.asyncio
    async def test_089_feedback_collection_system(self):
        """Test feedback collection from beta users"""
        for user in self.suite.test_users:
            await self.suite.register_beta_user(user)
            
            # Simulate user feedback
            feedback = {
                "user_id": user.email,
                "experience_rating": fake.random_int(3, 5),
                "ease_of_use": fake.random_int(3, 5),
                "performance_rating": fake.random_int(3, 5),
                "feature_requests": [
                    "Better thumbnail editor",
                    "More voice options",
                    "Bulk operations"
                ],
                "pain_points": [
                    "Initial setup complexity",
                    "Video generation time"
                ],
                "would_recommend": True,
                "additional_comments": fake.text(max_nb_chars=200)
            }
            
            headers = {"Authorization": f"Bearer {user.token}"}
            async with self.suite.session.post(f"{API_BASE_URL}/feedback/submit",
                headers=headers,
                json=feedback) as response:
                
                assert response.status == 201
    
    @pytest.mark.asyncio
    async def test_090_usage_analytics_collection(self):
        """Test collection of usage analytics from beta users"""
        user = self.suite.test_users[0]
        headers = {"Authorization": f"Bearer {user.token}"}
        
        # Simulate user actions
        actions = [
            {"action": "video_generation_started", "timestamp": datetime.utcnow().isoformat()},
            {"action": "dashboard_viewed", "duration": 45},
            {"action": "analytics_accessed", "feature": "real_time_dashboard"},
            {"action": "settings_modified", "changes": ["voice_preference", "quality_threshold"]},
            {"action": "video_generation_completed", "duration": 420, "cost": 2.85}
        ]
        
        for action in actions:
            async with self.suite.session.post(f"{API_BASE_URL}/analytics/track",
                headers=headers,
                json=action) as response:
                assert response.status == 200


# Test execution and reporting
@pytest.mark.asyncio
async def test_091_comprehensive_beta_test_execution():
    """Execute all beta test scenarios and generate report"""
    suite = BetaUserTestSuite()
    await suite.setup()
    
    try:
        # Run all test scenarios
        await suite.register_beta_user(suite.test_users[0])
        
        # Generate final report
        report = suite.results.generate_report()
        
        # Save report
        with open("tests/beta_testing/beta_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Verify success criteria
        assert report["success_rate"] >= 80, f"Success rate too low: {report['success_rate']}%"
        assert len(report["issues_found"]) == 0, f"Critical issues found: {report['issues_found']}"
        
        print(f"Beta testing completed with {report['success_rate']}% success rate")
        
    finally:
        await suite.cleanup()


if __name__ == "__main__":
    # Run beta test suite
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--html=tests/beta_testing/beta_test_report.html",
        "--self-contained-html"
    ])