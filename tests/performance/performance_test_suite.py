"""
Comprehensive Performance Testing Suite
Load testing, stress testing, and performance benchmarking
"""
import asyncio
import aiohttp
import time
import json
import statistics
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
import threading
import multiprocessing
import psutil
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from locust import HttpUser, task, between, events
import websockets
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest

logger = logging.getLogger(__name__)

class TestType(Enum):
    """Types of performance tests"""
    LOAD = "load"            # Normal expected load
    STRESS = "stress"        # Beyond normal capacity
    SPIKE = "spike"          # Sudden load increase
    SOAK = "soak"            # Extended duration
    SCALABILITY = "scalability"  # Scaling behavior
    BREAKPOINT = "breakpoint"    # Find breaking point

@dataclass
class PerformanceMetrics:
    """Performance test metrics"""
    test_type: TestType
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    response_times: List[float]
    throughput_rps: float
    error_rate: float
    percentiles: Dict[str, float]
    resource_usage: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TestScenario:
    """Performance test scenario configuration"""
    name: str
    test_type: TestType
    target_rps: int
    duration_seconds: int
    ramp_up_seconds: int
    concurrent_users: int
    endpoints: List[Dict[str, Any]]
    think_time: Tuple[float, float] = (1, 3)
    success_criteria: Dict[str, Any] = field(default_factory=dict)

class PerformanceTestSuite:
    """Comprehensive performance testing framework"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.metrics_history: List[PerformanceMetrics] = []
        self.resource_monitor = ResourceMonitor()
        self.report_generator = PerformanceReportGenerator()
        self._setup_metrics()
        
    def _setup_metrics(self):
        """Setup Prometheus metrics"""
        self.request_counter = Counter(
            'perf_test_requests_total',
            'Total requests made',
            ['endpoint', 'method', 'status']
        )
        self.response_histogram = Histogram(
            'perf_test_response_time_seconds',
            'Response time distribution',
            ['endpoint', 'method']
        )
        self.active_users = Gauge(
            'perf_test_active_users',
            'Number of active users'
        )
        
    async def run_test_suite(self) -> Dict[str, Any]:
        """Run complete performance test suite"""
        results = {}
        
        # Define test scenarios
        scenarios = [
            self._create_load_test_scenario(),
            self._create_stress_test_scenario(),
            self._create_spike_test_scenario(),
            self._create_api_endpoint_scenario(),
            self._create_websocket_scenario(),
            self._create_database_scenario()
        ]
        
        for scenario in scenarios:
            logger.info(f"Running scenario: {scenario.name}")
            metrics = await self.run_scenario(scenario)
            results[scenario.name] = metrics
            
            # Check success criteria
            if not self._check_success_criteria(metrics, scenario.success_criteria):
                logger.warning(f"Scenario {scenario.name} failed success criteria")
                
        # Generate comprehensive report
        report = self.report_generator.generate_report(results)
        
        return report
        
    def _create_load_test_scenario(self) -> TestScenario:
        """Create load test scenario"""
        return TestScenario(
            name="Normal Load Test",
            test_type=TestType.LOAD,
            target_rps=100,
            duration_seconds=300,
            ramp_up_seconds=30,
            concurrent_users=50,
            endpoints=[
                {"path": "/api/v1/health", "method": "GET", "weight": 0.1},
                {"path": "/api/v1/auth/login", "method": "POST", "weight": 0.2},
                {"path": "/api/v1/channels", "method": "GET", "weight": 0.3},
                {"path": "/api/v1/videos/generate", "method": "POST", "weight": 0.2},
                {"path": "/api/v1/dashboard/metrics", "method": "GET", "weight": 0.2}
            ],
            success_criteria={
                "max_response_time_p95": 1.0,  # 1 second
                "max_error_rate": 0.01,  # 1%
                "min_throughput": 90  # 90 RPS
            }
        )
        
    def _create_stress_test_scenario(self) -> TestScenario:
        """Create stress test scenario"""
        return TestScenario(
            name="Stress Test",
            test_type=TestType.STRESS,
            target_rps=500,
            duration_seconds=600,
            ramp_up_seconds=60,
            concurrent_users=200,
            endpoints=[
                {"path": "/api/v1/videos/generate", "method": "POST", "weight": 0.5},
                {"path": "/api/v1/ai/generate-script", "method": "POST", "weight": 0.3},
                {"path": "/api/v1/analytics/compute", "method": "POST", "weight": 0.2}
            ],
            success_criteria={
                "max_response_time_p99": 5.0,
                "max_error_rate": 0.05,
                "system_stability": True
            }
        )
        
    def _create_spike_test_scenario(self) -> TestScenario:
        """Create spike test scenario"""
        return TestScenario(
            name="Spike Test",
            test_type=TestType.SPIKE,
            target_rps=1000,
            duration_seconds=120,
            ramp_up_seconds=5,  # Very quick ramp-up
            concurrent_users=500,
            endpoints=[
                {"path": "/api/v1/health", "method": "GET", "weight": 1.0}
            ],
            success_criteria={
                "recovery_time": 30,  # Seconds to recover
                "max_error_rate_during_spike": 0.1
            }
        )
        
    def _create_api_endpoint_scenario(self) -> TestScenario:
        """Create API endpoint test scenario"""
        return TestScenario(
            name="API Endpoint Performance",
            test_type=TestType.LOAD,
            target_rps=200,
            duration_seconds=180,
            ramp_up_seconds=20,
            concurrent_users=100,
            endpoints=[
                {"path": "/api/v1/channels", "method": "POST", "weight": 0.1},
                {"path": "/api/v1/channels/{id}", "method": "GET", "weight": 0.2},
                {"path": "/api/v1/channels/{id}", "method": "PUT", "weight": 0.1},
                {"path": "/api/v1/videos", "method": "GET", "weight": 0.3},
                {"path": "/api/v1/videos/{id}", "method": "GET", "weight": 0.2},
                {"path": "/api/v1/analytics/overview", "method": "GET", "weight": 0.1}
            ],
            success_criteria={
                "max_response_time_p50": 0.2,
                "max_response_time_p95": 0.5,
                "max_response_time_p99": 1.0
            }
        )
        
    def _create_websocket_scenario(self) -> TestScenario:
        """Create WebSocket test scenario"""
        return TestScenario(
            name="WebSocket Performance",
            test_type=TestType.LOAD,
            target_rps=50,
            duration_seconds=300,
            ramp_up_seconds=30,
            concurrent_users=100,
            endpoints=[
                {"path": "/ws", "method": "WEBSOCKET", "weight": 1.0}
            ],
            success_criteria={
                "max_connection_time": 1.0,
                "max_message_latency": 0.1,
                "connection_stability": 0.99
            }
        )
        
    def _create_database_scenario(self) -> TestScenario:
        """Create database stress test scenario"""
        return TestScenario(
            name="Database Performance",
            test_type=TestType.LOAD,
            target_rps=150,
            duration_seconds=240,
            ramp_up_seconds=30,
            concurrent_users=75,
            endpoints=[
                {"path": "/api/v1/analytics/aggregate", "method": "POST", "weight": 0.3},
                {"path": "/api/v1/search", "method": "POST", "weight": 0.3},
                {"path": "/api/v1/reports/generate", "method": "POST", "weight": 0.2},
                {"path": "/api/v1/bulk/import", "method": "POST", "weight": 0.2}
            ],
            success_criteria={
                "max_query_time": 2.0,
                "connection_pool_exhaustion": False,
                "deadlock_rate": 0.001
            }
        )
        
    async def run_scenario(self, scenario: TestScenario) -> PerformanceMetrics:
        """Run a specific test scenario"""
        logger.info(f"Starting scenario: {scenario.name}")
        
        # Start resource monitoring
        self.resource_monitor.start()
        
        # Initialize metrics
        response_times = []
        errors = []
        start_time = time.time()
        
        # Create user pool
        users = []
        for i in range(scenario.concurrent_users):
            user = VirtualUser(
                user_id=i,
                base_url=self.base_url,
                endpoints=scenario.endpoints,
                think_time=scenario.think_time
            )
            users.append(user)
            
        # Ramp up users
        ramp_up_task = asyncio.create_task(
            self._ramp_up_users(users, scenario.ramp_up_seconds)
        )
        
        # Run test
        test_task = asyncio.create_task(
            self._run_users(
                users,
                scenario.duration_seconds,
                response_times,
                errors
            )
        )
        
        # Wait for completion
        await ramp_up_task
        await test_task
        
        # Stop resource monitoring
        resource_usage = self.resource_monitor.stop()
        
        # Calculate metrics
        end_time = time.time()
        duration = end_time - start_time
        
        metrics = PerformanceMetrics(
            test_type=scenario.test_type,
            duration_seconds=duration,
            total_requests=len(response_times) + len(errors),
            successful_requests=len(response_times),
            failed_requests=len(errors),
            response_times=response_times,
            throughput_rps=len(response_times) / duration if duration > 0 else 0,
            error_rate=len(errors) / (len(response_times) + len(errors)) if response_times or errors else 0,
            percentiles=self._calculate_percentiles(response_times),
            resource_usage=resource_usage
        )
        
        self.metrics_history.append(metrics)
        
        return metrics
        
    async def _ramp_up_users(self, users: List['VirtualUser'], ramp_up_seconds: int):
        """Gradually ramp up virtual users"""
        if not users or ramp_up_seconds <= 0:
            return
            
        delay = ramp_up_seconds / len(users)
        
        for user in users:
            user.active = True
            await asyncio.sleep(delay)
            
    async def _run_users(
        self,
        users: List['VirtualUser'],
        duration_seconds: int,
        response_times: List[float],
        errors: List[Dict]
    ):
        """Run virtual users for specified duration"""
        end_time = time.time() + duration_seconds
        tasks = []
        
        for user in users:
            task = asyncio.create_task(
                user.run_until(end_time, response_times, errors)
            )
            tasks.append(task)
            
        await asyncio.gather(*tasks)
        
    def _calculate_percentiles(self, response_times: List[float]) -> Dict[str, float]:
        """Calculate response time percentiles"""
        if not response_times:
            return {}
            
        return {
            "p50": np.percentile(response_times, 50),
            "p75": np.percentile(response_times, 75),
            "p90": np.percentile(response_times, 90),
            "p95": np.percentile(response_times, 95),
            "p99": np.percentile(response_times, 99),
            "min": min(response_times),
            "max": max(response_times),
            "mean": statistics.mean(response_times),
            "stdev": statistics.stdev(response_times) if len(response_times) > 1 else 0
        }
        
    def _check_success_criteria(
        self,
        metrics: PerformanceMetrics,
        criteria: Dict[str, Any]
    ) -> bool:
        """Check if test meets success criteria"""
        for criterion, threshold in criteria.items():
            if criterion == "max_response_time_p95":
                if metrics.percentiles.get("p95", float('inf')) > threshold:
                    return False
            elif criterion == "max_response_time_p99":
                if metrics.percentiles.get("p99", float('inf')) > threshold:
                    return False
            elif criterion == "max_error_rate":
                if metrics.error_rate > threshold:
                    return False
            elif criterion == "min_throughput":
                if metrics.throughput_rps < threshold:
                    return False
                    
        return True
        
    async def run_breakpoint_test(self) -> Dict[str, Any]:
        """Find system breaking point"""
        logger.info("Starting breakpoint test")
        
        initial_users = 10
        increment = 10
        max_users = 1000
        current_users = initial_users
        breaking_point = None
        
        while current_users <= max_users:
            scenario = TestScenario(
                name=f"Breakpoint Test - {current_users} users",
                test_type=TestType.BREAKPOINT,
                target_rps=current_users * 2,
                duration_seconds=60,
                ramp_up_seconds=10,
                concurrent_users=current_users,
                endpoints=[
                    {"path": "/api/v1/health", "method": "GET", "weight": 0.5},
                    {"path": "/api/v1/videos/generate", "method": "POST", "weight": 0.5}
                ]
            )
            
            metrics = await self.run_scenario(scenario)
            
            # Check for breaking point
            if metrics.error_rate > 0.1 or metrics.percentiles.get("p95", 0) > 5.0:
                breaking_point = current_users
                logger.info(f"Breaking point found at {current_users} users")
                break
                
            current_users += increment
            
        return {
            "breaking_point_users": breaking_point,
            "max_sustained_users": breaking_point - increment if breaking_point else max_users,
            "metrics_at_breaking_point": metrics.__dict__ if breaking_point else None
        }


class VirtualUser:
    """Virtual user for load testing"""
    
    def __init__(
        self,
        user_id: int,
        base_url: str,
        endpoints: List[Dict[str, Any]],
        think_time: Tuple[float, float]
    ):
        self.user_id = user_id
        self.base_url = base_url
        self.endpoints = endpoints
        self.think_time = think_time
        self.active = False
        self.session = None
        
    async def run_until(
        self,
        end_time: float,
        response_times: List[float],
        errors: List[Dict]
    ):
        """Run user actions until end time"""
        async with aiohttp.ClientSession() as self.session:
            while time.time() < end_time and self.active:
                # Select endpoint based on weights
                endpoint = self._select_endpoint()
                
                # Make request
                try:
                    start = time.time()
                    
                    if endpoint["method"] == "GET":
                        await self._make_get_request(endpoint["path"])
                    elif endpoint["method"] == "POST":
                        await self._make_post_request(endpoint["path"])
                    elif endpoint["method"] == "WEBSOCKET":
                        await self._test_websocket(endpoint["path"])
                        
                    response_time = time.time() - start
                    response_times.append(response_time)
                    
                except Exception as e:
                    errors.append({
                        "user_id": self.user_id,
                        "endpoint": endpoint["path"],
                        "error": str(e),
                        "timestamp": datetime.utcnow()
                    })
                    
                # Think time
                await asyncio.sleep(np.random.uniform(*self.think_time))
                
    def _select_endpoint(self) -> Dict[str, Any]:
        """Select endpoint based on weights"""
        weights = [e.get("weight", 1.0) for e in self.endpoints]
        return np.random.choice(self.endpoints, p=np.array(weights)/sum(weights))
        
    async def _make_get_request(self, path: str):
        """Make GET request"""
        url = f"{self.base_url}{path}"
        async with self.session.get(url) as response:
            await response.text()
            
    async def _make_post_request(self, path: str):
        """Make POST request"""
        url = f"{self.base_url}{path}"
        data = self._generate_test_data(path)
        async with self.session.post(url, json=data) as response:
            await response.text()
            
    def _generate_test_data(self, path: str) -> Dict[str, Any]:
        """Generate test data for POST requests"""
        if "login" in path:
            return {
                "email": f"user{self.user_id}@test.com",
                "password": "testpass123"
            }
        elif "generate" in path:
            return {
                "topic": f"Test Topic {self.user_id}",
                "style": "educational",
                "duration": 10
            }
        else:
            return {"test": "data"}
            
    async def _test_websocket(self, path: str):
        """Test WebSocket connection"""
        url = f"ws://localhost:8000{path}"
        async with websockets.connect(url) as websocket:
            # Send test message
            await websocket.send(json.dumps({"type": "heartbeat"}))
            # Wait for response
            response = await websocket.recv()


class ResourceMonitor:
    """Monitor system resources during tests"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        
    def start(self):
        """Start monitoring resources"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
        if not self.metrics:
            return {}
            
        # Calculate aggregates
        return {
            "cpu_percent": {
                "mean": statistics.mean([m["cpu"] for m in self.metrics]),
                "max": max([m["cpu"] for m in self.metrics])
            },
            "memory_percent": {
                "mean": statistics.mean([m["memory"] for m in self.metrics]),
                "max": max([m["memory"] for m in self.metrics])
            },
            "disk_io": {
                "read_mb": sum([m["disk_read"] for m in self.metrics]) / 1024 / 1024,
                "write_mb": sum([m["disk_write"] for m in self.metrics]) / 1024 / 1024
            },
            "network_io": {
                "sent_mb": sum([m["net_sent"] for m in self.metrics]) / 1024 / 1024,
                "recv_mb": sum([m["net_recv"] for m in self.metrics]) / 1024 / 1024
            }
        }
        
    def _monitor_loop(self):
        """Monitor loop running in separate thread"""
        while self.monitoring:
            metrics = {
                "timestamp": time.time(),
                "cpu": psutil.cpu_percent(interval=1),
                "memory": psutil.virtual_memory().percent,
                "disk_read": psutil.disk_io_counters().read_bytes,
                "disk_write": psutil.disk_io_counters().write_bytes,
                "net_sent": psutil.net_io_counters().bytes_sent,
                "net_recv": psutil.net_io_counters().bytes_recv
            }
            self.metrics.append(metrics)
            time.sleep(1)


class PerformanceReportGenerator:
    """Generate performance test reports"""
    
    def generate_report(self, results: Dict[str, PerformanceMetrics]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "summary": self._generate_summary(results),
            "detailed_results": {},
            "recommendations": [],
            "graphs": {}
        }
        
        for scenario_name, metrics in results.items():
            report["detailed_results"][scenario_name] = {
                "duration": metrics.duration_seconds,
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "throughput_rps": metrics.throughput_rps,
                "error_rate": metrics.error_rate,
                "response_times": metrics.percentiles,
                "resource_usage": metrics.resource_usage
            }
            
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(results)
        
        # Generate graphs (paths to saved graphs)
        report["graphs"] = self._generate_graphs(results)
        
        return report
        
    def _generate_summary(self, results: Dict[str, PerformanceMetrics]) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            "total_scenarios": len(results),
            "total_requests": sum(m.total_requests for m in results.values()),
            "overall_success_rate": 1 - (sum(m.failed_requests for m in results.values()) / 
                                         sum(m.total_requests for m in results.values())),
            "average_throughput": statistics.mean([m.throughput_rps for m in results.values()]),
            "test_duration": sum(m.duration_seconds for m in results.values())
        }
        
    def _generate_recommendations(self, results: Dict[str, PerformanceMetrics]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        for scenario_name, metrics in results.items():
            if metrics.error_rate > 0.05:
                recommendations.append(
                    f"High error rate ({metrics.error_rate:.2%}) in {scenario_name}. "
                    "Consider increasing resources or optimizing code."
                )
                
            if metrics.percentiles.get("p95", 0) > 2.0:
                recommendations.append(
                    f"High P95 latency ({metrics.percentiles['p95']:.2f}s) in {scenario_name}. "
                    "Consider caching, query optimization, or horizontal scaling."
                )
                
            if metrics.resource_usage.get("cpu_percent", {}).get("max", 0) > 80:
                recommendations.append(
                    f"High CPU usage in {scenario_name}. "
                    "Consider CPU optimization or vertical scaling."
                )
                
        return recommendations
        
    def _generate_graphs(self, results: Dict[str, PerformanceMetrics]) -> Dict[str, str]:
        """Generate performance graphs"""
        graphs = {}
        
        # Response time distribution
        plt.figure(figsize=(10, 6))
        for scenario_name, metrics in results.items():
            if metrics.response_times:
                plt.hist(metrics.response_times, alpha=0.5, label=scenario_name, bins=50)
        plt.xlabel("Response Time (s)")
        plt.ylabel("Frequency")
        plt.title("Response Time Distribution")
        plt.legend()
        graph_path = "reports/response_time_distribution.png"
        plt.savefig(graph_path)
        plt.close()
        graphs["response_time_distribution"] = graph_path
        
        # Throughput comparison
        plt.figure(figsize=(10, 6))
        scenarios = list(results.keys())
        throughputs = [results[s].throughput_rps for s in scenarios]
        plt.bar(scenarios, throughputs)
        plt.xlabel("Scenario")
        plt.ylabel("Throughput (RPS)")
        plt.title("Throughput Comparison")
        plt.xticks(rotation=45)
        graph_path = "reports/throughput_comparison.png"
        plt.savefig(graph_path)
        plt.close()
        graphs["throughput_comparison"] = graph_path
        
        return graphs


# Locust user for distributed testing
class YTEmpireUser(HttpUser):
    """Locust user for distributed load testing"""
    wait_time = between(1, 3)
    
    @task(3)
    def view_dashboard(self):
        """View dashboard"""
        self.client.get("/api/v1/dashboard/metrics")
        
    @task(2)
    def generate_video(self):
        """Generate video"""
        self.client.post("/api/v1/videos/generate", json={
            "topic": "Test Topic",
            "style": "educational"
        })
        
    @task(1)
    def view_analytics(self):
        """View analytics"""
        self.client.get("/api/v1/analytics/overview")
        
    def on_start(self):
        """Login on start"""
        response = self.client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "testpass123"
        })
        if response.status_code == 200:
            self.token = response.json().get("access_token")
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})


# Main execution
if __name__ == "__main__":
    async def main():
        suite = PerformanceTestSuite()
        results = await suite.run_test_suite()
        print(json.dumps(results, indent=2, default=str))
        
    asyncio.run(main())