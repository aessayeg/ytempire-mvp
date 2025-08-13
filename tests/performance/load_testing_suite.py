#!/usr/bin/env python3
"""
Comprehensive Load Testing Suite for YTEmpire
Tests with 100+ concurrent users and identifies bottlenecks
"""

import os
import sys
import json
import asyncio
import aiohttp
import logging
import time
import random
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import websockets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    """Load test configuration"""
    base_url: str = "http://localhost:8000"
    ws_url: str = "ws://localhost:8000"
    concurrent_users: int = 100
    test_duration_seconds: int = 600  # 10 minutes
    ramp_up_time: int = 120  # 2 minutes
    think_time_min: float = 1.0
    think_time_max: float = 5.0
    
@dataclass
class TestUser:
    """Test user information"""
    id: int
    email: str
    password: str
    token: Optional[str] = None
    
@dataclass
class RequestMetrics:
    """Request performance metrics"""
    endpoint: str
    method: str
    response_time_ms: float
    status_code: int
    success: bool
    timestamp: datetime
    user_id: int
    error_message: Optional[str] = None

@dataclass
class TestResults:
    """Aggregated test results"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0
    p50_response_time_ms: float = 0
    p95_response_time_ms: float = 0
    p99_response_time_ms: float = 0
    requests_per_second: float = 0
    error_rate: float = 0
    bottlenecks: List[str] = field(default_factory=list)

class TestScenario(Enum):
    """Load test scenarios"""
    BROWSE = "browse"
    VIDEO_GENERATION = "video_generation"
    ANALYTICS = "analytics"
    MIXED = "mixed"
    STRESS = "stress"

class LoadTestRunner:
    """Main load testing orchestrator"""
    
    def __init__(self, config: LoadTestConfig = None):
        self.config = config or LoadTestConfig()
        self.test_users: List[TestUser] = []
        self.metrics: List[RequestMetrics] = []
        self.active_users = 0
        self.test_start_time = None
        self.test_end_time = None
        self.session = None
        
    async def run_complete_load_test(self) -> Dict[str, Any]:
        """Run complete load testing campaign"""
        logger.info("Starting comprehensive load testing campaign")
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "concurrent_users": self.config.concurrent_users,
                "test_duration": self.config.test_duration_seconds,
                "base_url": self.config.base_url
            },
            "scenarios": {},
            "bottlenecks": [],
            "recommendations": []
        }
        
        try:
            # Setup test environment
            await self.setup_test_environment()
            
            # Run different test scenarios
            logger.info("Running test scenarios...")
            
            # Scenario 1: Normal browsing load
            test_results["scenarios"]["browse"] = await self.run_scenario(
                TestScenario.BROWSE,
                users=50,
                duration=300
            )
            
            # Scenario 2: Video generation load
            test_results["scenarios"]["video_generation"] = await self.run_scenario(
                TestScenario.VIDEO_GENERATION,
                users=20,
                duration=300
            )
            
            # Scenario 3: Mixed workload
            test_results["scenarios"]["mixed"] = await self.run_scenario(
                TestScenario.MIXED,
                users=100,
                duration=600
            )
            
            # Scenario 4: Stress test
            test_results["scenarios"]["stress"] = await self.run_scenario(
                TestScenario.STRESS,
                users=200,
                duration=300
            )
            
            # Identify bottlenecks
            test_results["bottlenecks"] = await self.identify_bottlenecks()
            
            # Generate recommendations
            test_results["recommendations"] = self.generate_recommendations(test_results)
            
            # Generate performance report
            await self.generate_performance_report(test_results)
            
        except Exception as e:
            logger.error(f"Load test failed: {e}")
            test_results["error"] = str(e)
        finally:
            await self.cleanup_test_environment()
            
        return test_results
    
    async def setup_test_environment(self):
        """Setup test environment and create test users"""
        logger.info("Setting up test environment...")
        
        # Create aiohttp session
        self.session = aiohttp.ClientSession()
        
        # Create test users
        for i in range(self.config.concurrent_users * 2):  # Create extra users
            user = TestUser(
                id=i,
                email=f"loadtest_{i}@example.com",
                password="TestPass123!"
            )
            
            # Register user
            try:
                async with self.session.post(
                    f"{self.config.base_url}/api/v1/auth/register",
                    json={
                        "email": user.email,
                        "password": user.password,
                        "name": f"Load Test User {i}"
                    }
                ) as response:
                    if response.status in [201, 409]:  # Created or already exists
                        self.test_users.append(user)
            except Exception as e:
                logger.warning(f"Failed to create user {i}: {e}")
        
        logger.info(f"Created {len(self.test_users)} test users")
    
    async def cleanup_test_environment(self):
        """Cleanup test environment"""
        logger.info("Cleaning up test environment...")
        
        if self.session:
            await self.session.close()
    
    async def run_scenario(
        self,
        scenario: TestScenario,
        users: int,
        duration: int
    ) -> TestResults:
        """Run a specific test scenario"""
        logger.info(f"Running {scenario.value} scenario with {users} users for {duration}s")
        
        self.metrics = []
        self.test_start_time = datetime.now()
        
        # Create tasks for concurrent users
        tasks = []
        
        # Ramp up users gradually
        ramp_up_delay = self.config.ramp_up_time / users if users > 0 else 0
        
        for i in range(min(users, len(self.test_users))):
            user = self.test_users[i]
            task = asyncio.create_task(
                self.simulate_user(user, scenario, duration, i * ramp_up_delay)
            )
            tasks.append(task)
        
        # Wait for all users to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.test_end_time = datetime.now()
        
        # Calculate results
        return self.calculate_test_results()
    
    async def simulate_user(
        self,
        user: TestUser,
        scenario: TestScenario,
        duration: int,
        initial_delay: float
    ):
        """Simulate a single user's behavior"""
        # Initial delay for ramp-up
        await asyncio.sleep(initial_delay)
        
        # Authenticate user
        await self.authenticate_user(user)
        
        if not user.token:
            logger.warning(f"User {user.id} failed to authenticate")
            return
        
        self.active_users += 1
        start_time = datetime.now()
        
        try:
            while (datetime.now() - start_time).total_seconds() < duration:
                # Execute scenario actions
                if scenario == TestScenario.BROWSE:
                    await self.browse_scenario(user)
                elif scenario == TestScenario.VIDEO_GENERATION:
                    await self.video_generation_scenario(user)
                elif scenario == TestScenario.ANALYTICS:
                    await self.analytics_scenario(user)
                elif scenario == TestScenario.MIXED:
                    await self.mixed_scenario(user)
                elif scenario == TestScenario.STRESS:
                    await self.stress_scenario(user)
                
                # Think time between actions
                think_time = random.uniform(
                    self.config.think_time_min,
                    self.config.think_time_max
                )
                await asyncio.sleep(think_time)
                
        finally:
            self.active_users -= 1
    
    async def authenticate_user(self, user: TestUser):
        """Authenticate a test user"""
        try:
            start_time = time.time()
            
            async with self.session.post(
                f"{self.config.base_url}/api/v1/auth/login",
                json={
                    "email": user.email,
                    "password": user.password
                }
            ) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    user.token = data.get("access_token")
                    success = True
                else:
                    success = False
                
                self.record_metric(
                    endpoint="/api/v1/auth/login",
                    method="POST",
                    response_time_ms=response_time,
                    status_code=response.status,
                    success=success,
                    user_id=user.id
                )
                
        except Exception as e:
            logger.error(f"Authentication failed for user {user.id}: {e}")
            self.record_metric(
                endpoint="/api/v1/auth/login",
                method="POST",
                response_time_ms=0,
                status_code=0,
                success=False,
                user_id=user.id,
                error_message=str(e)
            )
    
    async def browse_scenario(self, user: TestUser):
        """Simulate browsing behavior"""
        headers = {"Authorization": f"Bearer {user.token}"}
        
        # Get channels
        await self.make_request(
            "GET",
            "/api/v1/channels",
            headers,
            user.id
        )
        
        # Get videos
        await self.make_request(
            "GET",
            "/api/v1/videos?limit=20",
            headers,
            user.id
        )
        
        # Get specific video
        video_id = random.randint(1, 100)
        await self.make_request(
            "GET",
            f"/api/v1/videos/{video_id}",
            headers,
            user.id
        )
        
        # Get analytics
        await self.make_request(
            "GET",
            "/api/v1/analytics/dashboard",
            headers,
            user.id
        )
    
    async def video_generation_scenario(self, user: TestUser):
        """Simulate video generation"""
        headers = {"Authorization": f"Bearer {user.token}"}
        
        # Get trending topics
        await self.make_request(
            "GET",
            "/api/v1/trends",
            headers,
            user.id
        )
        
        # Generate video
        video_data = {
            "title": f"Test Video {random.randint(1000, 9999)}",
            "topic": "Technology",
            "style": "educational",
            "duration": "short",
            "channel_id": "test_channel_001"
        }
        
        response = await self.make_request(
            "POST",
            "/api/v1/videos/generate",
            headers,
            user.id,
            json_data=video_data
        )
        
        # Poll for completion if task started
        if response and response.get("task_id"):
            task_id = response["task_id"]
            
            for _ in range(10):  # Poll up to 10 times
                await asyncio.sleep(3)
                
                status_response = await self.make_request(
                    "GET",
                    f"/api/v1/videos/status/{task_id}",
                    headers,
                    user.id
                )
                
                if status_response and status_response.get("status") in ["completed", "failed"]:
                    break
    
    async def analytics_scenario(self, user: TestUser):
        """Simulate analytics queries"""
        headers = {"Authorization": f"Bearer {user.token}"}
        
        # Dashboard metrics
        await self.make_request(
            "GET",
            "/api/v1/analytics/dashboard",
            headers,
            user.id
        )
        
        # Channel analytics
        await self.make_request(
            "GET",
            "/api/v1/analytics/channels",
            headers,
            user.id
        )
        
        # Video performance
        await self.make_request(
            "GET",
            "/api/v1/analytics/videos/performance",
            headers,
            user.id
        )
        
        # Revenue analytics
        await self.make_request(
            "GET",
            "/api/v1/analytics/revenue",
            headers,
            user.id
        )
    
    async def mixed_scenario(self, user: TestUser):
        """Simulate mixed workload"""
        scenario_choice = random.random()
        
        if scenario_choice < 0.6:
            await self.browse_scenario(user)
        elif scenario_choice < 0.8:
            await self.analytics_scenario(user)
        else:
            await self.video_generation_scenario(user)
    
    async def stress_scenario(self, user: TestUser):
        """Simulate stress test with rapid requests"""
        headers = {"Authorization": f"Bearer {user.token}"}
        
        # Rapid fire requests
        endpoints = [
            "/api/v1/channels",
            "/api/v1/videos",
            "/api/v1/analytics/dashboard",
            "/api/v1/trends",
            "/api/v1/users/me"
        ]
        
        tasks = []
        for _ in range(5):
            endpoint = random.choice(endpoints)
            task = self.make_request("GET", endpoint, headers, user.id)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def make_request(
        self,
        method: str,
        endpoint: str,
        headers: Dict[str, str],
        user_id: int,
        json_data: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make HTTP request and record metrics"""
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            start_time = time.time()
            
            async with self.session.request(
                method,
                url,
                headers=headers,
                json=json_data
            ) as response:
                response_time = (time.time() - start_time) * 1000
                
                success = 200 <= response.status < 400
                
                self.record_metric(
                    endpoint=endpoint,
                    method=method,
                    response_time_ms=response_time,
                    status_code=response.status,
                    success=success,
                    user_id=user_id
                )
                
                if response.status == 200:
                    return await response.json()
                
                return None
                
        except Exception as e:
            logger.error(f"Request failed: {method} {endpoint} - {e}")
            self.record_metric(
                endpoint=endpoint,
                method=method,
                response_time_ms=0,
                status_code=0,
                success=False,
                user_id=user_id,
                error_message=str(e)
            )
            return None
    
    def record_metric(
        self,
        endpoint: str,
        method: str,
        response_time_ms: float,
        status_code: int,
        success: bool,
        user_id: int,
        error_message: Optional[str] = None
    ):
        """Record request metrics"""
        metric = RequestMetrics(
            endpoint=endpoint,
            method=method,
            response_time_ms=response_time_ms,
            status_code=status_code,
            success=success,
            timestamp=datetime.now(),
            user_id=user_id,
            error_message=error_message
        )
        
        self.metrics.append(metric)
    
    def calculate_test_results(self) -> TestResults:
        """Calculate aggregated test results"""
        if not self.metrics:
            return TestResults()
        
        results = TestResults()
        
        # Basic counts
        results.total_requests = len(self.metrics)
        results.successful_requests = sum(1 for m in self.metrics if m.success)
        results.failed_requests = results.total_requests - results.successful_requests
        
        # Response times (only for successful requests)
        response_times = [m.response_time_ms for m in self.metrics if m.success and m.response_time_ms > 0]
        
        if response_times:
            results.avg_response_time_ms = statistics.mean(response_times)
            results.p50_response_time_ms = statistics.median(response_times)
            results.p95_response_time_ms = np.percentile(response_times, 95)
            results.p99_response_time_ms = np.percentile(response_times, 99)
        
        # Throughput
        if self.test_start_time and self.test_end_time:
            duration_seconds = (self.test_end_time - self.test_start_time).total_seconds()
            if duration_seconds > 0:
                results.requests_per_second = results.total_requests / duration_seconds
        
        # Error rate
        if results.total_requests > 0:
            results.error_rate = results.failed_requests / results.total_requests
        
        return results
    
    async def identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        if not self.metrics:
            return bottlenecks
        
        # Group metrics by endpoint
        endpoint_metrics = {}
        for metric in self.metrics:
            key = f"{metric.method} {metric.endpoint}"
            if key not in endpoint_metrics:
                endpoint_metrics[key] = []
            if metric.success and metric.response_time_ms > 0:
                endpoint_metrics[key].append(metric.response_time_ms)
        
        # Identify slow endpoints
        for endpoint, times in endpoint_metrics.items():
            if times:
                avg_time = statistics.mean(times)
                p95_time = np.percentile(times, 95)
                
                if p95_time > 1000:  # > 1 second
                    bottlenecks.append(
                        f"{endpoint}: P95 response time {p95_time:.0f}ms (avg: {avg_time:.0f}ms)"
                    )
        
        # Check error rates by endpoint
        endpoint_errors = {}
        endpoint_totals = {}
        
        for metric in self.metrics:
            key = f"{metric.method} {metric.endpoint}"
            if key not in endpoint_totals:
                endpoint_totals[key] = 0
                endpoint_errors[key] = 0
            
            endpoint_totals[key] += 1
            if not metric.success:
                endpoint_errors[key] += 1
        
        for endpoint, total in endpoint_totals.items():
            if total > 10:  # Only consider endpoints with sufficient requests
                error_rate = endpoint_errors[endpoint] / total
                if error_rate > 0.05:  # > 5% error rate
                    bottlenecks.append(
                        f"{endpoint}: High error rate {error_rate:.1%} ({endpoint_errors[endpoint]}/{total} requests failed)"
                    )
        
        # Check for connection errors
        connection_errors = sum(
            1 for m in self.metrics 
            if m.error_message and "connection" in m.error_message.lower()
        )
        
        if connection_errors > 10:
            bottlenecks.append(
                f"Connection pool exhaustion: {connection_errors} connection errors"
            )
        
        # Check for timeout errors
        timeout_errors = sum(
            1 for m in self.metrics 
            if m.error_message and "timeout" in m.error_message.lower()
        )
        
        if timeout_errors > 10:
            bottlenecks.append(
                f"Timeout issues: {timeout_errors} timeout errors"
            )
        
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > 80:
            bottlenecks.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        if memory_percent > 80:
            bottlenecks.append(f"High memory usage: {memory_percent:.1f}%")
        
        return bottlenecks
    
    def generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Analyze each scenario
        for scenario_name, scenario_result in test_results.get("scenarios", {}).items():
            if isinstance(scenario_result, TestResults):
                # High response times
                if scenario_result.p95_response_time_ms > 1000:
                    recommendations.append(
                        f"Optimize {scenario_name} scenario - P95 response time exceeds 1s"
                    )
                
                # High error rate
                if scenario_result.error_rate > 0.05:
                    recommendations.append(
                        f"Investigate errors in {scenario_name} scenario - {scenario_result.error_rate:.1%} error rate"
                    )
                
                # Low throughput
                if scenario_result.requests_per_second < 10:
                    recommendations.append(
                        f"Improve throughput for {scenario_name} - only {scenario_result.requests_per_second:.1f} req/s"
                    )
        
        # Analyze bottlenecks
        bottlenecks = test_results.get("bottlenecks", [])
        
        if any("connection pool" in b.lower() for b in bottlenecks):
            recommendations.append("Increase database connection pool size")
            recommendations.append("Implement connection pooling for Redis")
        
        if any("timeout" in b.lower() for b in bottlenecks):
            recommendations.append("Increase timeout values for long-running operations")
            recommendations.append("Implement request queuing for video generation")
        
        if any("cpu" in b.lower() for b in bottlenecks):
            recommendations.append("Scale horizontally - add more application servers")
            recommendations.append("Optimize CPU-intensive operations")
        
        if any("memory" in b.lower() for b in bottlenecks):
            recommendations.append("Increase memory allocation")
            recommendations.append("Implement better caching strategies")
        
        # General recommendations
        if not recommendations:
            recommendations.append("System performing well under load")
        
        recommendations.extend([
            "Implement response caching for frequently accessed endpoints",
            "Use CDN for static content delivery",
            "Implement database query optimization",
            "Consider implementing rate limiting",
            "Set up auto-scaling based on load metrics"
        ])
        
        return recommendations
    
    async def generate_performance_report(self, test_results: Dict[str, Any]):
        """Generate comprehensive performance report"""
        report_path = Path(f"load_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        report_path.mkdir(exist_ok=True)
        
        # Generate markdown report
        with open(report_path / "report.md", 'w') as f:
            f.write("# YTEmpire Load Testing Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration
            f.write("## Test Configuration\n\n")
            config = test_results.get("configuration", {})
            f.write(f"- **Concurrent Users**: {config.get('concurrent_users', 'N/A')}\n")
            f.write(f"- **Test Duration**: {config.get('test_duration', 'N/A')}s\n")
            f.write(f"- **Target URL**: {config.get('base_url', 'N/A')}\n\n")
            
            # Results by scenario
            f.write("## Test Results by Scenario\n\n")
            
            for scenario_name, scenario_result in test_results.get("scenarios", {}).items():
                if isinstance(scenario_result, TestResults):
                    f.write(f"### {scenario_name.upper()}\n\n")
                    f.write(f"- **Total Requests**: {scenario_result.total_requests}\n")
                    f.write(f"- **Successful**: {scenario_result.successful_requests}\n")
                    f.write(f"- **Failed**: {scenario_result.failed_requests}\n")
                    f.write(f"- **Error Rate**: {scenario_result.error_rate:.2%}\n")
                    f.write(f"- **Throughput**: {scenario_result.requests_per_second:.1f} req/s\n\n")
                    
                    f.write("**Response Times**:\n")
                    f.write(f"- Average: {scenario_result.avg_response_time_ms:.0f}ms\n")
                    f.write(f"- P50: {scenario_result.p50_response_time_ms:.0f}ms\n")
                    f.write(f"- P95: {scenario_result.p95_response_time_ms:.0f}ms\n")
                    f.write(f"- P99: {scenario_result.p99_response_time_ms:.0f}ms\n\n")
            
            # Bottlenecks
            f.write("## Identified Bottlenecks\n\n")
            for bottleneck in test_results.get("bottlenecks", []):
                f.write(f"- {bottleneck}\n")
            
            if not test_results.get("bottlenecks"):
                f.write("No significant bottlenecks identified.\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for i, rec in enumerate(test_results.get("recommendations", []), 1):
                f.write(f"{i}. {rec}\n")
            
            f.write("\n")
            
            # Performance targets
            f.write("## Performance Targets\n\n")
            f.write("| Metric | Target | Actual | Status |\n")
            f.write("|--------|--------|--------|--------|\n")
            
            # Check against targets
            mixed_scenario = test_results.get("scenarios", {}).get("mixed")
            if isinstance(mixed_scenario, TestResults):
                f.write(f"| Response Time (P95) | <500ms | {mixed_scenario.p95_response_time_ms:.0f}ms | {'✅' if mixed_scenario.p95_response_time_ms < 500 else '❌'} |\n")
                f.write(f"| Error Rate | <5% | {mixed_scenario.error_rate:.1%} | {'✅' if mixed_scenario.error_rate < 0.05 else '❌'} |\n")
                f.write(f"| Throughput | >100 req/s | {mixed_scenario.requests_per_second:.1f} req/s | {'✅' if mixed_scenario.requests_per_second > 100 else '❌'} |\n")
        
        # Generate visualizations
        await self.generate_performance_charts(test_results, report_path)
        
        logger.info(f"Performance report generated in {report_path}")
    
    async def generate_performance_charts(self, test_results: Dict[str, Any], report_path: Path):
        """Generate performance visualization charts"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style("whitegrid")
            
            # Response time comparison chart
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Chart 1: Response times by scenario
            scenarios = []
            p50_times = []
            p95_times = []
            p99_times = []
            
            for name, result in test_results.get("scenarios", {}).items():
                if isinstance(result, TestResults):
                    scenarios.append(name)
                    p50_times.append(result.p50_response_time_ms)
                    p95_times.append(result.p95_response_time_ms)
                    p99_times.append(result.p99_response_time_ms)
            
            x = np.arange(len(scenarios))
            width = 0.25
            
            axes[0, 0].bar(x - width, p50_times, width, label='P50', color='green')
            axes[0, 0].bar(x, p95_times, width, label='P95', color='orange')
            axes[0, 0].bar(x + width, p99_times, width, label='P99', color='red')
            axes[0, 0].set_xlabel('Scenario')
            axes[0, 0].set_ylabel('Response Time (ms)')
            axes[0, 0].set_title('Response Times by Scenario')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(scenarios)
            axes[0, 0].legend()
            axes[0, 0].axhline(y=500, color='r', linestyle='--', label='Target (500ms)')
            
            # Chart 2: Throughput comparison
            throughputs = []
            for name, result in test_results.get("scenarios", {}).items():
                if isinstance(result, TestResults):
                    throughputs.append(result.requests_per_second)
            
            axes[0, 1].bar(scenarios, throughputs, color='steelblue')
            axes[0, 1].set_xlabel('Scenario')
            axes[0, 1].set_ylabel('Requests per Second')
            axes[0, 1].set_title('Throughput by Scenario')
            axes[0, 1].axhline(y=100, color='g', linestyle='--', label='Target (100 req/s)')
            axes[0, 1].legend()
            
            # Chart 3: Error rates
            error_rates = []
            for name, result in test_results.get("scenarios", {}).items():
                if isinstance(result, TestResults):
                    error_rates.append(result.error_rate * 100)
            
            axes[1, 0].bar(scenarios, error_rates, color=['green' if e < 5 else 'red' for e in error_rates])
            axes[1, 0].set_xlabel('Scenario')
            axes[1, 0].set_ylabel('Error Rate (%)')
            axes[1, 0].set_title('Error Rates by Scenario')
            axes[1, 0].axhline(y=5, color='r', linestyle='--', label='Threshold (5%)')
            axes[1, 0].legend()
            
            # Chart 4: Response time distribution (if we have detailed metrics)
            if self.metrics:
                response_times = [m.response_time_ms for m in self.metrics if m.success and m.response_time_ms > 0]
                if response_times:
                    axes[1, 1].hist(response_times, bins=50, color='skyblue', edgecolor='black')
                    axes[1, 1].set_xlabel('Response Time (ms)')
                    axes[1, 1].set_ylabel('Frequency')
                    axes[1, 1].set_title('Response Time Distribution')
                    axes[1, 1].axvline(x=np.percentile(response_times, 95), color='r', linestyle='--', label='P95')
                    axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig(report_path / "performance_charts.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate timeline chart if we have metrics
            if self.metrics:
                fig, ax = plt.subplots(figsize=(15, 6))
                
                # Group metrics by minute
                metrics_df = pd.DataFrame([
                    {
                        'timestamp': m.timestamp,
                        'response_time': m.response_time_ms,
                        'success': m.success
                    }
                    for m in self.metrics if m.response_time_ms > 0
                ])
                
                if not metrics_df.empty:
                    metrics_df['minute'] = metrics_df['timestamp'].dt.floor('1min')
                    timeline = metrics_df.groupby('minute').agg({
                        'response_time': ['mean', lambda x: np.percentile(x, 95)],
                        'success': 'mean'
                    })
                    
                    timeline.columns = ['avg_response_time', 'p95_response_time', 'success_rate']
                    
                    ax.plot(timeline.index, timeline['avg_response_time'], label='Average', color='blue')
                    ax.plot(timeline.index, timeline['p95_response_time'], label='P95', color='red')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Response Time (ms)')
                    ax.set_title('Response Time Over Time')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(report_path / "timeline_chart.png", dpi=300, bbox_inches='tight')
                    plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping chart generation")

class WebSocketLoadTest:
    """WebSocket-specific load testing"""
    
    def __init__(self, ws_url: str, num_connections: int = 100):
        self.ws_url = ws_url
        self.num_connections = num_connections
        self.connections = []
        self.metrics = []
    
    async def run_websocket_test(self, duration: int = 60) -> Dict[str, Any]:
        """Run WebSocket load test"""
        logger.info(f"Starting WebSocket load test with {self.num_connections} connections")
        
        results = {
            "connections_attempted": self.num_connections,
            "connections_successful": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "avg_latency_ms": 0,
            "errors": []
        }
        
        # Create connections
        tasks = []
        for i in range(self.num_connections):
            task = asyncio.create_task(self.websocket_client(i, duration))
            tasks.append(task)
            await asyncio.sleep(0.1)  # Stagger connections
        
        # Wait for all clients to complete
        connection_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for result in connection_results:
            if isinstance(result, dict):
                results["connections_successful"] += 1
                results["messages_sent"] += result.get("messages_sent", 0)
                results["messages_received"] += result.get("messages_received", 0)
            elif isinstance(result, Exception):
                results["errors"].append(str(result))
        
        # Calculate average latency
        if self.metrics:
            latencies = [m["latency_ms"] for m in self.metrics if "latency_ms" in m]
            if latencies:
                results["avg_latency_ms"] = statistics.mean(latencies)
        
        return results
    
    async def websocket_client(self, client_id: int, duration: int) -> Dict[str, Any]:
        """Simulate a WebSocket client"""
        client_results = {
            "messages_sent": 0,
            "messages_received": 0
        }
        
        try:
            async with websockets.connect(f"{self.ws_url}/ws/{client_id}") as websocket:
                start_time = datetime.now()
                
                while (datetime.now() - start_time).total_seconds() < duration:
                    # Send message
                    message = {
                        "type": "ping",
                        "timestamp": datetime.now().isoformat(),
                        "client_id": client_id
                    }
                    
                    send_time = time.time()
                    await websocket.send(json.dumps(message))
                    client_results["messages_sent"] += 1
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        receive_time = time.time()
                        client_results["messages_received"] += 1
                        
                        # Record latency
                        latency_ms = (receive_time - send_time) * 1000
                        self.metrics.append({
                            "client_id": client_id,
                            "latency_ms": latency_ms,
                            "timestamp": datetime.now()
                        })
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"WebSocket client {client_id}: Response timeout")
                    
                    # Wait before next message
                    await asyncio.sleep(random.uniform(1, 3))
                
        except Exception as e:
            logger.error(f"WebSocket client {client_id} error: {e}")
            raise
        
        return client_results

async def main():
    """Main execution function"""
    logger.info("Starting YTEmpire Load Testing Campaign")
    
    # Run HTTP load tests
    config = LoadTestConfig(
        base_url="http://localhost:8000",
        concurrent_users=100,
        test_duration_seconds=600
    )
    
    runner = LoadTestRunner(config)
    http_results = await runner.run_complete_load_test()
    
    # Run WebSocket load tests
    ws_test = WebSocketLoadTest(
        ws_url="ws://localhost:8000",
        num_connections=50
    )
    ws_results = await ws_test.run_websocket_test(duration=60)
    
    # Print summary
    print("\n" + "="*60)
    print("LOAD TESTING CAMPAIGN COMPLETE")
    print("="*60)
    
    print("\n## HTTP Load Test Results")
    for scenario_name, scenario_result in http_results.get("scenarios", {}).items():
        if isinstance(scenario_result, TestResults):
            print(f"\n### {scenario_name.upper()}")
            print(f"  Total Requests: {scenario_result.total_requests}")
            print(f"  Success Rate: {(1 - scenario_result.error_rate):.1%}")
            print(f"  Throughput: {scenario_result.requests_per_second:.1f} req/s")
            print(f"  P95 Response Time: {scenario_result.p95_response_time_ms:.0f}ms")
    
    print("\n## WebSocket Test Results")
    print(f"  Successful Connections: {ws_results['connections_successful']}/{ws_results['connections_attempted']}")
    print(f"  Messages Sent: {ws_results['messages_sent']}")
    print(f"  Messages Received: {ws_results['messages_received']}")
    print(f"  Average Latency: {ws_results['avg_latency_ms']:.1f}ms")
    
    print("\n## Bottlenecks Identified")
    for bottleneck in http_results.get("bottlenecks", []):
        print(f"  - {bottleneck}")
    
    print("\n## Recommendations")
    for i, rec in enumerate(http_results.get("recommendations", [])[:5], 1):
        print(f"  {i}. {rec}")
    
    print("\nDetailed report generated in: load_test_report_*/")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())