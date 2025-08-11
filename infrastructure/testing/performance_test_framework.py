#!/usr/bin/env python3
"""
Performance Testing Framework for YTEmpire MVP
Comprehensive load testing, stress testing, and performance monitoring
"""

import asyncio
import aiohttp
import time
import json
import statistics
import logging
import os
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import concurrent.futures
import threading
import queue
import random
import string
import psycopg2
import redis
import websockets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

logger = logging.getLogger(__name__)

class TestType(Enum):
    LOAD = "load"
    STRESS = "stress"
    SPIKE = "spike"
    ENDURANCE = "endurance"
    VOLUME = "volume"

class Protocol(Enum):
    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "websocket"
    DATABASE = "database"
    REDIS = "redis"

@dataclass
class TestConfig:
    """Test configuration parameters"""
    name: str
    test_type: TestType
    protocol: Protocol
    target_url: str
    duration_seconds: int
    concurrent_users: int
    ramp_up_seconds: int
    ramp_down_seconds: int
    max_rps: Optional[int] = None
    headers: Dict[str, str] = None
    payload: Dict[str, Any] = None
    assertions: List[Dict[str, Any]] = None
    scenarios: List[Dict[str, Any]] = None

@dataclass
class TestResult:
    """Individual test result"""
    timestamp: float
    response_time: float
    status_code: int
    success: bool
    error_message: Optional[str] = None
    response_size: int = 0
    request_size: int = 0

@dataclass
class TestSummary:
    """Test execution summary"""
    test_name: str
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    
    # Response time statistics
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p90_response_time: float
    p95_response_time: float
    p99_response_time: float
    
    # Throughput statistics
    requests_per_second: float
    peak_rps: float
    
    # Resource utilization
    cpu_usage: List[float]
    memory_usage: List[float]
    
    # Errors breakdown
    error_breakdown: Dict[str, int]
    
    # Additional metrics
    total_bytes_sent: int
    total_bytes_received: int

class MetricsCollector:
    """Collects system metrics during testing"""
    
    def __init__(self):
        self.collecting = False
        self.metrics_thread = None
        self.cpu_samples = []
        self.memory_samples = []
        self.network_samples = []
        
    def start_collection(self):
        """Start collecting system metrics"""
        self.collecting = True
        self.metrics_thread = threading.Thread(target=self._collect_metrics)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()
    
    def stop_collection(self):
        """Stop collecting system metrics"""
        self.collecting = False
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5)
    
    def _collect_metrics(self):
        """Collect system metrics in background"""
        import psutil
        
        while self.collecting:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_samples.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_samples.append(memory.percent)
                
                # Network I/O
                network = psutil.net_io_counters()
                self.network_samples.append({
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                })
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(1)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics"""
        return {
            'cpu_usage': self.cpu_samples.copy(),
            'memory_usage': self.memory_samples.copy(),
            'network_samples': self.network_samples.copy()
        }

class HTTPLoadTester:
    """HTTP/HTTPS load testing implementation"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.results_lock = threading.Lock()
        
    async def run_single_request(self, session: aiohttp.ClientSession, 
                                request_data: Dict[str, Any]) -> TestResult:
        """Execute a single HTTP request"""
        start_time = time.time()
        
        try:
            method = request_data.get('method', 'GET').upper()
            url = request_data.get('url', self.config.target_url)
            headers = {**(self.config.headers or {}), **request_data.get('headers', {})}
            payload = request_data.get('payload', self.config.payload)
            
            request_size = len(json.dumps(payload).encode()) if payload else 0
            
            if method == 'GET':
                async with session.get(url, headers=headers) as response:
                    response_text = await response.text()
                    response_size = len(response_text.encode())
                    status_code = response.status
            elif method == 'POST':
                async with session.post(url, headers=headers, json=payload) as response:
                    response_text = await response.text()
                    response_size = len(response_text.encode())
                    status_code = response.status
            elif method == 'PUT':
                async with session.put(url, headers=headers, json=payload) as response:
                    response_text = await response.text()
                    response_size = len(response_text.encode())
                    status_code = response.status
            elif method == 'DELETE':
                async with session.delete(url, headers=headers) as response:
                    response_text = await response.text()
                    response_size = len(response_text.encode())
                    status_code = response.status
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response_time = time.time() - start_time
            success = 200 <= status_code < 400
            
            # Validate assertions
            if self.config.assertions:
                for assertion in self.config.assertions:
                    assertion_type = assertion.get('type')
                    expected = assertion.get('expected')
                    
                    if assertion_type == 'status_code' and status_code != expected:
                        success = False
                    elif assertion_type == 'response_time' and response_time > expected:
                        success = False
                    elif assertion_type == 'contains' and expected not in response_text:
                        success = False
            
            return TestResult(
                timestamp=start_time,
                response_time=response_time,
                status_code=status_code,
                success=success,
                response_size=response_size,
                request_size=request_size
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                timestamp=start_time,
                response_time=response_time,
                status_code=0,
                success=False,
                error_message=str(e)
            )
    
    async def run_user_session(self, user_id: int, scenarios: List[Dict[str, Any]]):
        """Run a complete user session with multiple requests"""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout,
            headers={'User-Agent': f'YTEmpire-LoadTest-User-{user_id}'}
        ) as session:
            
            for scenario in scenarios:
                # Add think time between requests
                think_time = scenario.get('think_time', 0)
                if think_time > 0:
                    await asyncio.sleep(think_time)
                
                result = await self.run_single_request(session, scenario)
                
                with self.results_lock:
                    self.results.append(result)
                
                # Rate limiting
                if self.config.max_rps:
                    await asyncio.sleep(1.0 / self.config.max_rps)
    
    async def run_load_test(self) -> List[TestResult]:
        """Execute the load test"""
        logger.info(f"Starting HTTP load test: {self.config.name}")
        
        # Default scenario if none provided
        scenarios = self.config.scenarios or [{'method': 'GET', 'url': self.config.target_url}]
        
        # Calculate user ramp-up
        users_per_second = self.config.concurrent_users / max(self.config.ramp_up_seconds, 1)
        
        tasks = []
        start_time = time.time()
        
        # Ramp up users
        for user_id in range(self.config.concurrent_users):
            delay = user_id / users_per_second
            task = asyncio.create_task(
                self._delayed_user_session(user_id, scenarios, delay)
            )
            tasks.append(task)
        
        # Wait for test duration
        await asyncio.sleep(self.config.duration_seconds)
        
        # Cancel remaining tasks for ramp-down
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete gracefully
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"HTTP load test completed: {len(self.results)} requests executed")
        return self.results.copy()
    
    async def _delayed_user_session(self, user_id: int, scenarios: List[Dict[str, Any]], delay: float):
        """Start user session after delay"""
        await asyncio.sleep(delay)
        
        end_time = time.time() + self.config.duration_seconds - delay
        while time.time() < end_time:
            await self.run_user_session(user_id, scenarios)

class WebSocketLoadTester:
    """WebSocket load testing implementation"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.results_lock = threading.Lock()
    
    async def run_websocket_session(self, user_id: int):
        """Run WebSocket session for a single user"""
        try:
            uri = self.config.target_url.replace('http', 'ws')
            
            async with websockets.connect(uri) as websocket:
                session_start = time.time()
                session_end = session_start + self.config.duration_seconds
                
                while time.time() < session_end:
                    # Send message
                    start_time = time.time()
                    message = json.dumps({
                        'user_id': user_id,
                        'timestamp': start_time,
                        'data': self.config.payload or {'test': 'message'}
                    })
                    
                    try:
                        await websocket.send(message)
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        
                        response_time = time.time() - start_time
                        
                        result = TestResult(
                            timestamp=start_time,
                            response_time=response_time,
                            status_code=200,
                            success=True,
                            response_size=len(response.encode()),
                            request_size=len(message.encode())
                        )
                        
                    except asyncio.TimeoutError:
                        result = TestResult(
                            timestamp=start_time,
                            response_time=time.time() - start_time,
                            status_code=0,
                            success=False,
                            error_message="WebSocket timeout"
                        )
                    except Exception as e:
                        result = TestResult(
                            timestamp=start_time,
                            response_time=time.time() - start_time,
                            status_code=0,
                            success=False,
                            error_message=str(e)
                        )
                    
                    with self.results_lock:
                        self.results.append(result)
                    
                    # Wait before next message
                    await asyncio.sleep(1.0)
                    
        except Exception as e:
            logger.error(f"WebSocket session error for user {user_id}: {e}")
    
    async def run_load_test(self) -> List[TestResult]:
        """Execute WebSocket load test"""
        logger.info(f"Starting WebSocket load test: {self.config.name}")
        
        tasks = []
        for user_id in range(self.config.concurrent_users):
            delay = user_id * (self.config.ramp_up_seconds / self.config.concurrent_users)
            task = asyncio.create_task(
                self._delayed_websocket_session(user_id, delay)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"WebSocket load test completed: {len(self.results)} messages exchanged")
        return self.results.copy()
    
    async def _delayed_websocket_session(self, user_id: int, delay: float):
        """Start WebSocket session after delay"""
        await asyncio.sleep(delay)
        await self.run_websocket_session(user_id)

class DatabaseLoadTester:
    """Database load testing implementation"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.results_lock = threading.Lock()
        
        # Parse database URL
        self.db_config = self._parse_db_url(config.target_url)
    
    def _parse_db_url(self, url: str) -> Dict[str, str]:
        """Parse database URL"""
        # Simple URL parsing - in production use proper URL parsing
        if url.startswith('postgresql://'):
            # postgresql://user:password@host:port/database
            parts = url.replace('postgresql://', '').split('/')
            auth_host = parts[0]
            database = parts[1] if len(parts) > 1 else 'postgres'
            
            if '@' in auth_host:
                auth, host = auth_host.split('@')
                user, password = auth.split(':')
            else:
                user, password, host = 'postgres', '', auth_host
            
            host_port = host.split(':')
            host = host_port[0]
            port = int(host_port[1]) if len(host_port) > 1 else 5432
            
            return {
                'host': host,
                'port': port,
                'user': user,
                'password': password,
                'database': database
            }
        
        return {}
    
    def run_single_query(self, query: str, params: Tuple = None) -> TestResult:
        """Execute a single database query"""
        start_time = time.time()
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute(query, params)
            
            if cursor.description:  # SELECT query
                results = cursor.fetchall()
                response_size = len(str(results).encode())
            else:  # INSERT/UPDATE/DELETE
                conn.commit()
                response_size = 0
            
            cursor.close()
            conn.close()
            
            response_time = time.time() - start_time
            
            return TestResult(
                timestamp=start_time,
                response_time=response_time,
                status_code=200,
                success=True,
                response_size=response_size,
                request_size=len(query.encode())
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                timestamp=start_time,
                response_time=response_time,
                status_code=0,
                success=False,
                error_message=str(e)
            )
    
    def run_user_queries(self, user_id: int):
        """Run queries for a single user"""
        queries = self.config.scenarios or [
            {'query': 'SELECT 1'},
            {'query': 'SELECT count(*) FROM pg_stat_activity'},
            {'query': 'SELECT version()'}
        ]
        
        session_end = time.time() + self.config.duration_seconds
        
        while time.time() < session_end:
            for scenario in queries:
                query = scenario['query']
                params = scenario.get('params')
                
                result = self.run_single_query(query, params)
                
                with self.results_lock:
                    self.results.append(result)
                
                # Think time between queries
                think_time = scenario.get('think_time', 0.1)
                time.sleep(think_time)
                
                if time.time() >= session_end:
                    break
    
    def run_load_test(self) -> List[TestResult]:
        """Execute database load test"""
        logger.info(f"Starting database load test: {self.config.name}")
        
        with ThreadPoolExecutor(max_workers=self.config.concurrent_users) as executor:
            futures = []
            
            for user_id in range(self.config.concurrent_users):
                future = executor.submit(self.run_user_queries, user_id)
                futures.append(future)
                
                # Ramp up delay
                time.sleep(self.config.ramp_up_seconds / self.config.concurrent_users)
            
            # Wait for all users to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Database test user error: {e}")
        
        logger.info(f"Database load test completed: {len(self.results)} queries executed")
        return self.results.copy()

class RedisLoadTester:
    """Redis load testing implementation"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.results_lock = threading.Lock()
        
        # Parse Redis URL
        self.redis_config = self._parse_redis_url(config.target_url)
    
    def _parse_redis_url(self, url: str) -> Dict[str, Any]:
        """Parse Redis URL"""
        if url.startswith('redis://'):
            # redis://[:password@]host:port[/database]
            parts = url.replace('redis://', '').split('/')
            host_part = parts[0]
            db = int(parts[1]) if len(parts) > 1 else 0
            
            if '@' in host_part:
                password, host_port = host_part.split('@')
                password = password[1:] if password.startswith(':') else password
            else:
                password = None
                host_port = host_part
            
            host_port_parts = host_port.split(':')
            host = host_port_parts[0]
            port = int(host_port_parts[1]) if len(host_port_parts) > 1 else 6379
            
            return {
                'host': host,
                'port': port,
                'password': password,
                'db': db,
                'decode_responses': True
            }
        
        return {'host': 'localhost', 'port': 6379, 'decode_responses': True}
    
    def run_single_operation(self, operation: Dict[str, Any]) -> TestResult:
        """Execute a single Redis operation"""
        start_time = time.time()
        
        try:
            r = redis.Redis(**self.redis_config)
            
            op_type = operation.get('type', 'GET')
            key = operation.get('key', f'test_key_{random.randint(1, 1000)}')
            value = operation.get('value', 'test_value')
            
            request_size = len(str(key).encode()) + len(str(value).encode())
            
            if op_type == 'SET':
                result = r.set(key, value)
                response_size = len(str(result).encode())
            elif op_type == 'GET':
                result = r.get(key)
                response_size = len(str(result).encode()) if result else 0
            elif op_type == 'INCR':
                result = r.incr(key)
                response_size = len(str(result).encode())
            elif op_type == 'LPUSH':
                result = r.lpush(key, value)
                response_size = len(str(result).encode())
            elif op_type == 'LPOP':
                result = r.lpop(key)
                response_size = len(str(result).encode()) if result else 0
            elif op_type == 'HSET':
                field = operation.get('field', 'test_field')
                result = r.hset(key, field, value)
                response_size = len(str(result).encode())
            elif op_type == 'HGET':
                field = operation.get('field', 'test_field')
                result = r.hget(key, field)
                response_size = len(str(result).encode()) if result else 0
            else:
                # Ping operation
                result = r.ping()
                response_size = len(str(result).encode())
            
            response_time = time.time() - start_time
            
            return TestResult(
                timestamp=start_time,
                response_time=response_time,
                status_code=200,
                success=True,
                response_size=response_size,
                request_size=request_size
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                timestamp=start_time,
                response_time=response_time,
                status_code=0,
                success=False,
                error_message=str(e)
            )
    
    def run_user_operations(self, user_id: int):
        """Run Redis operations for a single user"""
        operations = self.config.scenarios or [
            {'type': 'SET', 'key': f'user_{user_id}_key', 'value': f'user_{user_id}_value'},
            {'type': 'GET', 'key': f'user_{user_id}_key'},
            {'type': 'INCR', 'key': f'counter_{user_id}'},
            {'type': 'PING'}
        ]
        
        session_end = time.time() + self.config.duration_seconds
        
        while time.time() < session_end:
            for operation in operations:
                result = self.run_single_operation(operation)
                
                with self.results_lock:
                    self.results.append(result)
                
                # Think time between operations
                think_time = operation.get('think_time', 0.01)
                time.sleep(think_time)
                
                if time.time() >= session_end:
                    break
    
    def run_load_test(self) -> List[TestResult]:
        """Execute Redis load test"""
        logger.info(f"Starting Redis load test: {self.config.name}")
        
        with ThreadPoolExecutor(max_workers=self.config.concurrent_users) as executor:
            futures = []
            
            for user_id in range(self.config.concurrent_users):
                future = executor.submit(self.run_user_operations, user_id)
                futures.append(future)
                
                # Ramp up delay
                time.sleep(self.config.ramp_up_seconds / self.config.concurrent_users)
            
            # Wait for all users to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Redis test user error: {e}")
        
        logger.info(f"Redis load test completed: {len(self.results)} operations executed")
        return self.results.copy()

class ResultAnalyzer:
    """Analyzes and generates reports from test results"""
    
    @staticmethod
    def analyze_results(results: List[TestResult], test_config: TestConfig, 
                       metrics: Dict[str, Any]) -> TestSummary:
        """Analyze test results and generate summary"""
        if not results:
            return TestSummary(
                test_name=test_config.name,
                start_time=datetime.now(),
                end_time=datetime.now(),
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                error_rate=0,
                avg_response_time=0,
                min_response_time=0,
                max_response_time=0,
                p50_response_time=0,
                p90_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                peak_rps=0,
                cpu_usage=metrics.get('cpu_usage', []),
                memory_usage=metrics.get('memory_usage', []),
                error_breakdown={},
                total_bytes_sent=0,
                total_bytes_received=0
            )
        
        # Basic statistics
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r.success)
        failed_requests = total_requests - successful_requests
        error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0
        
        # Response time statistics
        response_times = [r.response_time for r in results]
        avg_response_time = statistics.mean(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        # Percentiles
        sorted_times = sorted(response_times)
        p50_response_time = ResultAnalyzer._percentile(sorted_times, 50)
        p90_response_time = ResultAnalyzer._percentile(sorted_times, 90)
        p95_response_time = ResultAnalyzer._percentile(sorted_times, 95)
        p99_response_time = ResultAnalyzer._percentile(sorted_times, 99)
        
        # Throughput calculations
        start_time = datetime.fromtimestamp(min(r.timestamp for r in results))
        end_time = datetime.fromtimestamp(max(r.timestamp for r in results))
        duration = (end_time - start_time).total_seconds()
        requests_per_second = total_requests / duration if duration > 0 else 0
        
        # Peak RPS calculation (1-second windows)
        peak_rps = ResultAnalyzer._calculate_peak_rps(results)
        
        # Error breakdown
        error_breakdown = {}
        for result in results:
            if not result.success:
                error_key = result.error_message or f"HTTP_{result.status_code}"
                error_breakdown[error_key] = error_breakdown.get(error_key, 0) + 1
        
        # Data transfer statistics
        total_bytes_sent = sum(r.request_size for r in results)
        total_bytes_received = sum(r.response_size for r in results)
        
        return TestSummary(
            test_name=test_config.name,
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            error_rate=error_rate,
            avg_response_time=avg_response_time * 1000,  # Convert to milliseconds
            min_response_time=min_response_time * 1000,
            max_response_time=max_response_time * 1000,
            p50_response_time=p50_response_time * 1000,
            p90_response_time=p90_response_time * 1000,
            p95_response_time=p95_response_time * 1000,
            p99_response_time=p99_response_time * 1000,
            requests_per_second=requests_per_second,
            peak_rps=peak_rps,
            cpu_usage=metrics.get('cpu_usage', []),
            memory_usage=metrics.get('memory_usage', []),
            error_breakdown=error_breakdown,
            total_bytes_sent=total_bytes_sent,
            total_bytes_received=total_bytes_received
        )
    
    @staticmethod
    def _percentile(sorted_data: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not sorted_data:
            return 0
        
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    @staticmethod
    def _calculate_peak_rps(results: List[TestResult]) -> float:
        """Calculate peak requests per second in 1-second windows"""
        if not results:
            return 0
        
        # Group results by second
        second_counts = {}
        for result in results:
            second = int(result.timestamp)
            second_counts[second] = second_counts.get(second, 0) + 1
        
        return max(second_counts.values()) if second_counts else 0
    
    @staticmethod
    def generate_report(summary: TestSummary, output_path: str = None) -> str:
        """Generate detailed test report"""
        report = f"""
# Performance Test Report

## Test Summary
- **Test Name**: {summary.test_name}
- **Start Time**: {summary.start_time}
- **End Time**: {summary.end_time}
- **Duration**: {(summary.end_time - summary.start_time).total_seconds():.2f} seconds

## Request Statistics
- **Total Requests**: {summary.total_requests:,}
- **Successful Requests**: {summary.successful_requests:,}
- **Failed Requests**: {summary.failed_requests:,}
- **Error Rate**: {summary.error_rate:.2f}%

## Response Time Statistics (milliseconds)
- **Average**: {summary.avg_response_time:.2f} ms
- **Minimum**: {summary.min_response_time:.2f} ms
- **Maximum**: {summary.max_response_time:.2f} ms
- **50th Percentile**: {summary.p50_response_time:.2f} ms
- **90th Percentile**: {summary.p90_response_time:.2f} ms
- **95th Percentile**: {summary.p95_response_time:.2f} ms
- **99th Percentile**: {summary.p99_response_time:.2f} ms

## Throughput Statistics
- **Average RPS**: {summary.requests_per_second:.2f} req/sec
- **Peak RPS**: {summary.peak_rps:.2f} req/sec

## Data Transfer
- **Total Bytes Sent**: {summary.total_bytes_sent:,} bytes ({summary.total_bytes_sent/1024/1024:.2f} MB)
- **Total Bytes Received**: {summary.total_bytes_received:,} bytes ({summary.total_bytes_received/1024/1024:.2f} MB)

## System Resource Usage
- **Average CPU Usage**: {statistics.mean(summary.cpu_usage):.2f}% (Peak: {max(summary.cpu_usage) if summary.cpu_usage else 0:.2f}%)
- **Average Memory Usage**: {statistics.mean(summary.memory_usage):.2f}% (Peak: {max(summary.memory_usage) if summary.memory_usage else 0:.2f}%)

## Error Breakdown
"""
        
        if summary.error_breakdown:
            for error, count in summary.error_breakdown.items():
                report += f"- **{error}**: {count:,} ({(count/summary.total_requests)*100:.2f}%)\n"
        else:
            report += "- No errors detected\n"
        
        report += f"""

## Performance Assessment
"""
        
        # Performance assessment
        if summary.error_rate < 1:
            report += "- ✅ **Error Rate**: Excellent (< 1%)\n"
        elif summary.error_rate < 5:
            report += "- ⚠️ **Error Rate**: Acceptable (1-5%)\n"
        else:
            report += "- ❌ **Error Rate**: Poor (> 5%)\n"
        
        if summary.p95_response_time < 500:
            report += "- ✅ **Response Time**: Excellent (P95 < 500ms)\n"
        elif summary.p95_response_time < 1000:
            report += "- ⚠️ **Response Time**: Acceptable (P95 < 1s)\n"
        else:
            report += "- ❌ **Response Time**: Poor (P95 > 1s)\n"
        
        avg_cpu = statistics.mean(summary.cpu_usage) if summary.cpu_usage else 0
        if avg_cpu < 70:
            report += "- ✅ **CPU Usage**: Normal (< 70%)\n"
        elif avg_cpu < 90:
            report += "- ⚠️ **CPU Usage**: High (70-90%)\n"
        else:
            report += "- ❌ **CPU Usage**: Critical (> 90%)\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report
    
    @staticmethod
    def generate_charts(summary: TestSummary, results: List[TestResult], output_dir: str):
        """Generate performance charts"""
        if not results:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Response time over time
        plt.figure(figsize=(12, 6))
        timestamps = [r.timestamp for r in results]
        response_times = [r.response_time * 1000 for r in results]  # Convert to ms
        
        plt.scatter(timestamps, response_times, alpha=0.6, s=1)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Response Time (ms)')
        plt.title(f'Response Time Over Time - {summary.test_name}')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/response_time_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Response time distribution
        plt.figure(figsize=(10, 6))
        plt.hist(response_times, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Response Time (ms)')
        plt.ylabel('Frequency')
        plt.title(f'Response Time Distribution - {summary.test_name}')
        plt.axvline(summary.avg_response_time, color='red', linestyle='--', label=f'Average: {summary.avg_response_time:.1f}ms')
        plt.axvline(summary.p95_response_time, color='orange', linestyle='--', label=f'95th Percentile: {summary.p95_response_time:.1f}ms')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/response_time_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Throughput over time
        if len(results) > 10:
            plt.figure(figsize=(12, 6))
            
            # Calculate RPS in 1-second windows
            second_counts = {}
            for result in results:
                second = int(result.timestamp)
                second_counts[second] = second_counts.get(second, 0) + 1
            
            seconds = sorted(second_counts.keys())
            rps_values = [second_counts[s] for s in seconds]
            
            plt.plot(seconds, rps_values, linewidth=2)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Requests per Second')
            plt.title(f'Throughput Over Time - {summary.test_name}')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{output_dir}/throughput_over_time.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # System resource usage
        if summary.cpu_usage and summary.memory_usage:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            time_points = list(range(len(summary.cpu_usage)))
            
            ax1.plot(time_points, summary.cpu_usage, color='blue', linewidth=2)
            ax1.set_ylabel('CPU Usage (%)')
            ax1.set_title(f'System Resource Usage - {summary.test_name}')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(time_points, summary.memory_usage, color='green', linewidth=2)
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Memory Usage (%)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/system_resources.png', dpi=150, bbox_inches='tight')
            plt.close()

class PerformanceTestFramework:
    """Main performance testing framework"""
    
    def __init__(self):
        self.test_configs: List[TestConfig] = []
        self.test_results: Dict[str, List[TestResult]] = {}
        self.test_summaries: Dict[str, TestSummary] = {}
    
    def load_config_from_file(self, config_path: str):
        """Load test configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        for test_data in config_data.get('tests', []):
            config = TestConfig(
                name=test_data['name'],
                test_type=TestType(test_data['test_type']),
                protocol=Protocol(test_data['protocol']),
                target_url=test_data['target_url'],
                duration_seconds=test_data['duration_seconds'],
                concurrent_users=test_data['concurrent_users'],
                ramp_up_seconds=test_data.get('ramp_up_seconds', 10),
                ramp_down_seconds=test_data.get('ramp_down_seconds', 10),
                max_rps=test_data.get('max_rps'),
                headers=test_data.get('headers'),
                payload=test_data.get('payload'),
                assertions=test_data.get('assertions'),
                scenarios=test_data.get('scenarios')
            )
            self.test_configs.append(config)
    
    async def run_test(self, config: TestConfig) -> Tuple[List[TestResult], Dict[str, Any]]:
        """Run a single performance test"""
        logger.info(f"Starting test: {config.name}")
        
        # Start metrics collection
        metrics_collector = MetricsCollector()
        metrics_collector.start_collection()
        
        try:
            # Select appropriate tester
            if config.protocol == Protocol.HTTP or config.protocol == Protocol.HTTPS:
                tester = HTTPLoadTester(config)
                results = await tester.run_load_test()
            elif config.protocol == Protocol.WEBSOCKET:
                tester = WebSocketLoadTester(config)
                results = await tester.run_load_test()
            elif config.protocol == Protocol.DATABASE:
                tester = DatabaseLoadTester(config)
                results = tester.run_load_test()
            elif config.protocol == Protocol.REDIS:
                tester = RedisLoadTester(config)
                results = tester.run_load_test()
            else:
                raise ValueError(f"Unsupported protocol: {config.protocol}")
            
            return results, metrics_collector.get_metrics()
            
        finally:
            metrics_collector.stop_collection()
    
    async def run_all_tests(self) -> Dict[str, TestSummary]:
        """Run all configured tests"""
        logger.info(f"Running {len(self.test_configs)} performance tests")
        
        for config in self.test_configs:
            results, metrics = await self.run_test(config)
            
            # Store results
            self.test_results[config.name] = results
            
            # Analyze results
            summary = ResultAnalyzer.analyze_results(results, config, metrics)
            self.test_summaries[config.name] = summary
            
            logger.info(f"Test completed: {config.name} - "
                       f"{summary.total_requests} requests, "
                       f"{summary.error_rate:.1f}% error rate, "
                       f"{summary.avg_response_time:.1f}ms avg response time")
        
        return self.test_summaries
    
    def generate_reports(self, output_dir: str = "performance_reports"):
        """Generate comprehensive test reports"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate individual test reports
        for test_name, summary in self.test_summaries.items():
            report_path = f"{output_dir}/{test_name}_report.md"
            ResultAnalyzer.generate_report(summary, report_path)
            
            # Generate charts
            charts_dir = f"{output_dir}/{test_name}_charts"
            results = self.test_results[test_name]
            ResultAnalyzer.generate_charts(summary, results, charts_dir)
        
        # Generate consolidated report
        self._generate_consolidated_report(output_dir)
        
        logger.info(f"Performance test reports generated in {output_dir}")
    
    def _generate_consolidated_report(self, output_dir: str):
        """Generate consolidated report for all tests"""
        report_path = f"{output_dir}/consolidated_report.md"
        
        report = "# Consolidated Performance Test Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## Test Summary Overview\n\n"
        report += "| Test Name | Total Requests | Error Rate | Avg Response Time | P95 Response Time | RPS |\n"
        report += "|-----------|----------------|------------|-------------------|-------------------|-----|\n"
        
        for test_name, summary in self.test_summaries.items():
            report += f"| {test_name} | {summary.total_requests:,} | {summary.error_rate:.1f}% | {summary.avg_response_time:.1f}ms | {summary.p95_response_time:.1f}ms | {summary.requests_per_second:.1f} |\n"
        
        report += "\n## Performance Benchmarks\n\n"
        
        best_response_time = min(s.avg_response_time for s in self.test_summaries.values())
        best_throughput = max(s.requests_per_second for s in self.test_summaries.values())
        lowest_error_rate = min(s.error_rate for s in self.test_summaries.values())
        
        report += f"- **Best Average Response Time**: {best_response_time:.1f}ms\n"
        report += f"- **Highest Throughput**: {best_throughput:.1f} RPS\n"
        report += f"- **Lowest Error Rate**: {lowest_error_rate:.1f}%\n"
        
        report += "\n## Recommendations\n\n"
        
        # Add performance recommendations based on results
        for test_name, summary in self.test_summaries.items():
            if summary.error_rate > 5:
                report += f"- ❌ **{test_name}**: High error rate ({summary.error_rate:.1f}%) - investigate application stability\n"
            if summary.p95_response_time > 1000:
                report += f"- ⚠️ **{test_name}**: Slow P95 response time ({summary.p95_response_time:.1f}ms) - optimize performance\n"
            if summary.cpu_usage and max(summary.cpu_usage) > 90:
                report += f"- ⚠️ **{test_name}**: High CPU usage (peak {max(summary.cpu_usage):.1f}%) - consider scaling\n"
        
        with open(report_path, 'w') as f:
            f.write(report)

# CLI Interface
async def main():
    """Command-line interface for performance testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YTEmpire Performance Test Framework")
    parser.add_argument("--config", "-c", required=True, help="Test configuration file path")
    parser.add_argument("--output", "-o", default="performance_reports", help="Output directory for reports")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize framework
    framework = PerformanceTestFramework()
    
    # Load configuration
    framework.load_config_from_file(args.config)
    
    # Run tests
    summaries = await framework.run_all_tests()
    
    # Generate reports
    framework.generate_reports(args.output)
    
    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE TEST SUMMARY")
    print("="*80)
    
    for test_name, summary in summaries.items():
        print(f"\n{test_name}:")
        print(f"  Total Requests: {summary.total_requests:,}")
        print(f"  Error Rate: {summary.error_rate:.1f}%")
        print(f"  Avg Response Time: {summary.avg_response_time:.1f}ms")
        print(f"  P95 Response Time: {summary.p95_response_time:.1f}ms")
        print(f"  Throughput: {summary.requests_per_second:.1f} RPS")

if __name__ == "__main__":
    asyncio.run(main())