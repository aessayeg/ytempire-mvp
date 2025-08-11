#!/usr/bin/env python3
"""
End-to-End Video Generation Pipeline Test Suite
Tests the complete video generation flow with cost verification
"""

import asyncio
import time
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import requests
import websocket
import psycopg2
import redis
from decimal import Decimal

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """Test configuration"""
    api_url: str = "http://localhost:8000/api/v1"
    ws_url: str = "ws://localhost:8000/ws"
    n8n_url: str = "http://localhost:5678"
    db_url: str = "postgresql://ytempire:admin@localhost:5432/ytempire_db"
    redis_url: str = "redis://localhost:6379/0"
    
    # Test parameters
    max_video_generation_time: int = 600  # 10 minutes
    max_cost_per_video: float = 3.0
    min_quality_score: float = 0.85
    concurrent_videos: int = 5
    
    # Test user
    test_email: str = "test@ytempire.com"
    test_password: str = "TestPassword123!"

@dataclass
class VideoGenerationResult:
    """Result of video generation test"""
    video_id: str
    channel_id: str
    status: str
    generation_time: float
    total_cost: float
    quality_score: float
    errors: List[str]
    metadata: Dict[str, Any]

class VideoPipelineTest:
    """End-to-end video pipeline testing"""
    
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.token = None
        self.headers = {}
        self.test_results = []
        self.ws_client = None
        
    def setup(self):
        """Setup test environment"""
        logger.info("Setting up test environment...")
        
        # Authenticate
        self.authenticate()
        
        # Verify services are running
        self.verify_services()
        
        # Setup WebSocket connection
        self.setup_websocket()
        
        logger.info("Test environment ready")
    
    def authenticate(self):
        """Authenticate and get token"""
        response = requests.post(
            f"{self.config.api_url}/auth/login",
            json={
                "email": self.config.test_email,
                "password": self.config.test_password
            }
        )
        
        if response.status_code != 200:
            # Try to register first
            register_response = requests.post(
                f"{self.config.api_url}/auth/register",
                json={
                    "email": self.config.test_email,
                    "password": self.config.test_password,
                    "name": "Test User"
                }
            )
            if register_response.status_code == 201:
                # Retry login
                response = requests.post(
                    f"{self.config.api_url}/auth/login",
                    json={
                        "email": self.config.test_email,
                        "password": self.config.test_password
                    }
                )
        
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
            logger.info("Authentication successful")
        else:
            raise Exception(f"Authentication failed: {response.text}")
    
    def verify_services(self):
        """Verify all required services are running"""
        services = [
            ("Backend API", f"{self.config.api_url}/health"),
            ("N8N Workflows", f"{self.config.n8n_url}/healthz"),
            ("Flower", "http://localhost:5555/api/workers"),
            ("Prometheus", "http://localhost:9090/-/healthy"),
            ("Grafana", "http://localhost:3001/api/health")
        ]
        
        for service_name, url in services:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✓ {service_name} is running")
                else:
                    logger.warning(f"✗ {service_name} returned status {response.status_code}")
            except Exception as e:
                logger.warning(f"✗ {service_name} is not accessible: {e}")
    
    def setup_websocket(self):
        """Setup WebSocket connection for real-time updates"""
        try:
            ws_url = f"{self.config.ws_url}/test_client?token={self.token}"
            self.ws_client = websocket.WebSocket()
            self.ws_client.connect(ws_url)
            logger.info("WebSocket connected")
        except Exception as e:
            logger.warning(f"WebSocket connection failed: {e}")
    
    def create_test_channel(self) -> str:
        """Create a test channel"""
        channel_data = {
            "youtube_channel_id": f"UC_TEST_{int(time.time())}",
            "channel_name": "Test Channel",
            "channel_handle": f"@testchannel_{int(time.time())}",
            "niche": "technology",
            "target_audience": "developers",
            "upload_schedule": {
                "frequency": "daily",
                "preferred_times": ["09:00", "15:00", "20:00"]
            }
        }
        
        response = requests.post(
            f"{self.config.api_url}/channels",
            json=channel_data,
            headers=self.headers
        )
        
        if response.status_code == 201:
            channel = response.json()
            logger.info(f"Created test channel: {channel['id']}")
            return channel["id"]
        else:
            raise Exception(f"Failed to create channel: {response.text}")
    
    def test_video_generation(self, channel_id: str, topic: str = None) -> VideoGenerationResult:
        """Test single video generation"""
        start_time = time.time()
        errors = []
        
        # Generate video request
        video_data = {
            "channel_id": channel_id,
            "topic": topic or "Top 10 Python Tips for 2024",
            "style": "educational",
            "length": "medium",
            "target_audience": "developers",
            "quality_settings": {
                "resolution": "1080p",
                "fps": 30,
                "bitrate": "5000k"
            },
            "cost_limit": self.config.max_cost_per_video
        }
        
        logger.info(f"Starting video generation: {video_data['topic']}")
        
        # Submit video generation request
        response = requests.post(
            f"{self.config.api_url}/videos/generate",
            json=video_data,
            headers=self.headers
        )
        
        if response.status_code != 202:
            errors.append(f"Failed to start generation: {response.text}")
            return VideoGenerationResult(
                video_id="",
                channel_id=channel_id,
                status="failed",
                generation_time=0,
                total_cost=0,
                quality_score=0,
                errors=errors,
                metadata={}
            )
        
        video = response.json()
        video_id = video["id"]
        logger.info(f"Video generation started: {video_id}")
        
        # Poll for completion
        status = "pending"
        metadata = {}
        
        while time.time() - start_time < self.config.max_video_generation_time:
            # Check status
            status_response = requests.get(
                f"{self.config.api_url}/videos/{video_id}/status",
                headers=self.headers
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data["status"]
                metadata = status_data.get("metadata", {})
                
                logger.info(f"Video {video_id} status: {status} - {status_data.get('progress', 0)}%")
                
                if status in ["completed", "failed", "error"]:
                    break
                
                # Check WebSocket for real-time updates
                if self.ws_client:
                    try:
                        ws_message = self.ws_client.recv()
                        ws_data = json.loads(ws_message)
                        logger.debug(f"WebSocket update: {ws_data}")
                    except:
                        pass
            
            time.sleep(5)
        
        generation_time = time.time() - start_time
        
        # Get final video details
        video_response = requests.get(
            f"{self.config.api_url}/videos/{video_id}",
            headers=self.headers
        )
        
        if video_response.status_code == 200:
            video_data = video_response.json()
            total_cost = video_data.get("total_cost", 0)
            quality_score = video_data.get("quality_score", 0)
            
            # Verify cost tracking
            cost_response = requests.get(
                f"{self.config.api_url}/costs/video/{video_id}",
                headers=self.headers
            )
            if cost_response.status_code == 200:
                cost_breakdown = cost_response.json()
                logger.info(f"Cost breakdown: {json.dumps(cost_breakdown, indent=2)}")
        else:
            total_cost = 0
            quality_score = 0
            errors.append(f"Failed to get video details: {video_response.text}")
        
        # Check if generation exceeded time limit
        if generation_time >= self.config.max_video_generation_time:
            errors.append(f"Generation exceeded time limit: {generation_time:.2f}s")
            status = "timeout"
        
        # Check cost limit
        if total_cost > self.config.max_cost_per_video:
            errors.append(f"Cost exceeded limit: ${total_cost:.2f} > ${self.config.max_cost_per_video}")
        
        # Check quality threshold
        if quality_score < self.config.min_quality_score:
            errors.append(f"Quality below threshold: {quality_score:.2f} < {self.config.min_quality_score}")
        
        result = VideoGenerationResult(
            video_id=video_id,
            channel_id=channel_id,
            status=status,
            generation_time=generation_time,
            total_cost=total_cost,
            quality_score=quality_score,
            errors=errors,
            metadata=metadata
        )
        
        self.test_results.append(result)
        return result
    
    def test_concurrent_generation(self, channel_id: str, num_videos: int = None):
        """Test concurrent video generation"""
        num_videos = num_videos or self.config.concurrent_videos
        topics = [
            "Python vs JavaScript in 2024",
            "10 AI Tools Every Developer Needs",
            "Building Scalable Microservices",
            "DevOps Best Practices",
            "Cloud Computing Fundamentals",
            "Machine Learning for Beginners",
            "Web3 Development Guide",
            "Mobile App Development Trends",
            "Cybersecurity Essentials",
            "Database Optimization Tips"
        ]
        
        logger.info(f"Starting concurrent generation of {num_videos} videos")
        
        # Start all video generations
        generation_tasks = []
        for i in range(num_videos):
            topic = topics[i % len(topics)]
            # Use threading or async for true concurrent testing
            result = self.test_video_generation(channel_id, topic)
            generation_tasks.append(result)
        
        # Analyze results
        successful = [r for r in generation_tasks if r.status == "completed"]
        failed = [r for r in generation_tasks if r.status != "completed"]
        
        logger.info(f"Concurrent generation results:")
        logger.info(f"  Successful: {len(successful)}/{num_videos}")
        logger.info(f"  Failed: {len(failed)}/{num_videos}")
        
        if successful:
            avg_time = sum(r.generation_time for r in successful) / len(successful)
            avg_cost = sum(r.total_cost for r in successful) / len(successful)
            avg_quality = sum(r.quality_score for r in successful) / len(successful)
            
            logger.info(f"  Average generation time: {avg_time:.2f}s")
            logger.info(f"  Average cost: ${avg_cost:.2f}")
            logger.info(f"  Average quality score: {avg_quality:.2f}")
    
    def test_multi_account_rotation(self):
        """Test YouTube multi-account rotation"""
        logger.info("Testing multi-account rotation...")
        
        # Get account status
        response = requests.get(
            f"{self.config.api_url}/youtube/accounts/status",
            headers=self.headers
        )
        
        if response.status_code == 200:
            accounts = response.json()
            logger.info(f"Active YouTube accounts: {len(accounts.get('active', []))}")
            logger.info(f"Account health scores: {json.dumps(accounts.get('health_scores', {}), indent=2)}")
            
            # Test account rotation by making multiple requests
            for i in range(5):
                rotation_response = requests.post(
                    f"{self.config.api_url}/youtube/accounts/rotate",
                    headers=self.headers
                )
                if rotation_response.status_code == 200:
                    selected_account = rotation_response.json()
                    logger.info(f"Rotation {i+1}: Selected account {selected_account['account_id']}")
    
    def test_quality_scoring(self, video_id: str):
        """Test video quality scoring"""
        logger.info(f"Testing quality scoring for video {video_id}")
        
        response = requests.post(
            f"{self.config.api_url}/videos/{video_id}/score",
            headers=self.headers
        )
        
        if response.status_code == 200:
            score_data = response.json()
            logger.info(f"Quality scores:")
            logger.info(f"  Overall: {score_data.get('overall_score', 0):.2f}")
            logger.info(f"  Content: {score_data.get('content_score', 0):.2f}")
            logger.info(f"  Technical: {score_data.get('technical_score', 0):.2f}")
            logger.info(f"  Engagement: {score_data.get('engagement_score', 0):.2f}")
    
    def test_cost_optimization(self):
        """Test cost optimization features"""
        logger.info("Testing cost optimization...")
        
        # Test progressive model fallback
        fallback_test = {
            "task": "script_generation",
            "preferred_model": "gpt-4",
            "content": "Generate a script about Python programming"
        }
        
        response = requests.post(
            f"{self.config.api_url}/ai/generate",
            json=fallback_test,
            headers=self.headers
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Model used: {result.get('model_used')}")
            logger.info(f"Cost: ${result.get('cost', 0):.4f}")
            logger.info(f"Fallback triggered: {result.get('fallback_used', False)}")
    
    def verify_database_state(self):
        """Verify database state after tests"""
        try:
            conn = psycopg2.connect(self.config.db_url)
            cursor = conn.cursor()
            
            # Check video count
            cursor.execute("SELECT COUNT(*) FROM videos WHERE created_at > NOW() - INTERVAL '1 hour'")
            video_count = cursor.fetchone()[0]
            logger.info(f"Videos created in last hour: {video_count}")
            
            # Check cost tracking
            cursor.execute("""
                SELECT 
                    SUM(total_cost) as total_cost,
                    AVG(total_cost) as avg_cost,
                    MIN(total_cost) as min_cost,
                    MAX(total_cost) as max_cost
                FROM costs 
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            cost_stats = cursor.fetchone()
            if cost_stats[0]:
                logger.info(f"Cost statistics:")
                logger.info(f"  Total: ${cost_stats[0]:.2f}")
                logger.info(f"  Average: ${cost_stats[1]:.2f}")
                logger.info(f"  Min: ${cost_stats[2]:.2f}")
                logger.info(f"  Max: ${cost_stats[3]:.2f}")
            
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Database verification failed: {e}")
    
    def verify_redis_state(self):
        """Verify Redis cache state"""
        try:
            r = redis.from_url(self.config.redis_url)
            
            # Check cache keys
            cache_keys = r.keys("cache:*")
            logger.info(f"Cache entries: {len(cache_keys)}")
            
            # Check queue sizes
            celery_queues = ["celery", "video_processing", "thumbnail_generation"]
            for queue in celery_queues:
                queue_size = r.llen(queue)
                logger.info(f"Queue '{queue}' size: {queue_size}")
        except Exception as e:
            logger.error(f"Redis verification failed: {e}")
    
    def generate_report(self):
        """Generate test report"""
        logger.info("\n" + "="*60)
        logger.info("VIDEO PIPELINE TEST REPORT")
        logger.info("="*60)
        
        if not self.test_results:
            logger.info("No test results to report")
            return
        
        # Overall statistics
        total_tests = len(self.test_results)
        successful = [r for r in self.test_results if r.status == "completed"]
        failed = [r for r in self.test_results if r.status != "completed"]
        
        logger.info(f"\nTest Summary:")
        logger.info(f"  Total tests: {total_tests}")
        logger.info(f"  Successful: {len(successful)} ({len(successful)/total_tests*100:.1f}%)")
        logger.info(f"  Failed: {len(failed)} ({len(failed)/total_tests*100:.1f}%)")
        
        if successful:
            # Time statistics
            times = [r.generation_time for r in successful]
            logger.info(f"\nGeneration Time:")
            logger.info(f"  Average: {sum(times)/len(times):.2f}s")
            logger.info(f"  Min: {min(times):.2f}s")
            logger.info(f"  Max: {max(times):.2f}s")
            
            # Cost statistics
            costs = [r.total_cost for r in successful]
            logger.info(f"\nCost per Video:")
            logger.info(f"  Average: ${sum(costs)/len(costs):.2f}")
            logger.info(f"  Min: ${min(costs):.2f}")
            logger.info(f"  Max: ${max(costs):.2f}")
            logger.info(f"  Under $3: {len([c for c in costs if c < 3.0])}/{len(costs)}")
            
            # Quality statistics
            qualities = [r.quality_score for r in successful]
            logger.info(f"\nQuality Scores:")
            logger.info(f"  Average: {sum(qualities)/len(qualities):.2f}")
            logger.info(f"  Min: {min(qualities):.2f}")
            logger.info(f"  Max: {max(qualities):.2f}")
            logger.info(f"  Above 0.85: {len([q for q in qualities if q >= 0.85])}/{len(qualities)}")
        
        # Errors summary
        if failed:
            logger.info(f"\nErrors Encountered:")
            all_errors = []
            for result in failed:
                all_errors.extend(result.errors)
            
            from collections import Counter
            error_counts = Counter(all_errors)
            for error, count in error_counts.most_common(5):
                logger.info(f"  {error}: {count} occurrences")
        
        logger.info("\n" + "="*60)
        
        # Save detailed report
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump([asdict(r) for r in self.test_results], f, indent=2, default=str)
        logger.info(f"Detailed report saved to: {report_file}")
    
    def cleanup(self):
        """Cleanup test resources"""
        logger.info("Cleaning up test resources...")
        
        if self.ws_client:
            self.ws_client.close()
        
        # Optionally delete test data
        # This would include test channels, videos, etc.
    
    def run_all_tests(self):
        """Run complete test suite"""
        try:
            # Setup
            self.setup()
            
            # Create test channel
            channel_id = self.create_test_channel()
            
            # Test 1: Single video generation
            logger.info("\n--- Test 1: Single Video Generation ---")
            result = self.test_video_generation(channel_id)
            if result.status == "completed":
                logger.info(f"✓ Single video generated successfully in {result.generation_time:.2f}s")
                logger.info(f"  Cost: ${result.total_cost:.2f}")
                logger.info(f"  Quality: {result.quality_score:.2f}")
            else:
                logger.error(f"✗ Single video generation failed: {result.errors}")
            
            # Test 2: Concurrent generation
            logger.info("\n--- Test 2: Concurrent Video Generation ---")
            self.test_concurrent_generation(channel_id)
            
            # Test 3: Multi-account rotation
            logger.info("\n--- Test 3: Multi-Account Rotation ---")
            self.test_multi_account_rotation()
            
            # Test 4: Quality scoring
            if result.video_id:
                logger.info("\n--- Test 4: Quality Scoring ---")
                self.test_quality_scoring(result.video_id)
            
            # Test 5: Cost optimization
            logger.info("\n--- Test 5: Cost Optimization ---")
            self.test_cost_optimization()
            
            # Verify system state
            logger.info("\n--- System State Verification ---")
            self.verify_database_state()
            self.verify_redis_state()
            
            # Generate report
            self.generate_report()
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            raise
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YTEmpire Video Pipeline Test Suite")
    parser.add_argument("--api-url", default="http://localhost:8000/api/v1", help="API URL")
    parser.add_argument("--concurrent", type=int, default=5, help="Number of concurrent videos")
    parser.add_argument("--max-cost", type=float, default=3.0, help="Maximum cost per video")
    parser.add_argument("--min-quality", type=float, default=0.85, help="Minimum quality score")
    args = parser.parse_args()
    
    config = TestConfig(
        api_url=args.api_url,
        concurrent_videos=args.concurrent,
        max_cost_per_video=args.max_cost,
        min_quality_score=args.min_quality
    )
    
    tester = VideoPipelineTest(config)
    tester.run_all_tests()

if __name__ == "__main__":
    main()