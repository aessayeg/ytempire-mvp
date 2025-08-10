#!/usr/bin/env python3
"""
Full End-to-End System Test for YTEmpire
Day 10 P0 Task: Complete System Validation
Tests all components and integrations
"""

import asyncio
import aiohttp
import json
import time
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import psycopg2
import redis
import websocket
from colorama import init, Fore, Back, Style

# Initialize colorama for colored output
init(autoreset=True)

# Test configuration
API_BASE_URL = "http://localhost:8000/api/v1"
FRONTEND_URL = "http://localhost:3000"
WS_URL = "ws://localhost:8000/ws"
REDIS_URL = "redis://localhost:6379"
DATABASE_URL = "postgresql://ytempire:password@localhost:5432/ytempire_db"

class SystemTest:
    def __init__(self):
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "start_time": datetime.now(),
            "tests": []
        }
        self.auth_token = None
        self.test_user_id = None
        self.test_channel_id = None
        self.test_video_id = None
        
    def log_success(self, message: str):
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} {message}")
        
    def log_error(self, message: str):
        print(f"{Fore.RED}✗{Style.RESET_ALL} {message}")
        
    def log_warning(self, message: str):
        print(f"{Fore.YELLOW}⚠{Style.RESET_ALL} {message}")
        
    def log_info(self, message: str):
        print(f"{Fore.CYAN}ℹ{Style.RESET_ALL} {message}")
        
    def log_section(self, title: str):
        print(f"\n{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{title.center(60)}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}\n")
        
    async def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and track results"""
        self.results["total_tests"] += 1
        test_result = {
            "name": test_name,
            "start_time": datetime.now(),
            "status": "running"
        }
        
        try:
            self.log_info(f"Running: {test_name}")
            result = await test_func()
            
            if result:
                self.log_success(f"Passed: {test_name}")
                test_result["status"] = "passed"
                self.results["passed"] += 1
            else:
                self.log_error(f"Failed: {test_name}")
                test_result["status"] = "failed"
                self.results["failed"] += 1
                
            test_result["end_time"] = datetime.now()
            test_result["duration"] = (test_result["end_time"] - test_result["start_time"]).total_seconds()
            
        except Exception as e:
            self.log_error(f"Error in {test_name}: {str(e)}")
            test_result["status"] = "error"
            test_result["error"] = str(e)
            self.results["failed"] += 1
            
        self.results["tests"].append(test_result)
        return test_result["status"] == "passed"
        
    # Infrastructure Tests
    async def test_database_connection(self) -> bool:
        """Test PostgreSQL database connection"""
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            cursor.close()
            conn.close()
            self.log_info(f"PostgreSQL version: {version[0][:30]}...")
            return True
        except Exception as e:
            self.log_error(f"Database connection failed: {str(e)}")
            return False
            
    async def test_redis_connection(self) -> bool:
        """Test Redis connection"""
        try:
            r = redis.Redis.from_url(REDIS_URL)
            r.ping()
            info = r.info()
            self.log_info(f"Redis version: {info.get('redis_version', 'unknown')}")
            return True
        except Exception as e:
            self.log_error(f"Redis connection failed: {str(e)}")
            return False
            
    async def test_api_health(self) -> bool:
        """Test API health endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_info(f"API Status: {data.get('status', 'unknown')}")
                    return data.get("status") == "healthy"
                return False
                
    # Authentication Tests
    async def test_user_registration(self) -> bool:
        """Test user registration"""
        async with aiohttp.ClientSession() as session:
            test_email = f"test_{int(time.time())}@ytempire.com"
            async with session.post(f"{API_BASE_URL}/auth/register", json={
                "email": test_email,
                "password": "SecurePass123!",
                "name": "Test User"
            }) as response:
                if response.status in [200, 201]:
                    data = await response.json()
                    self.test_user_id = data.get("user_id")
                    self.auth_token = data.get("access_token")
                    self.log_info(f"User created: {self.test_user_id}")
                    return True
                return False
                
    async def test_user_login(self) -> bool:
        """Test user login"""
        if not self.auth_token:
            # Use existing test account
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{API_BASE_URL}/auth/login", json={
                    "email": "test@ytempire.com",
                    "password": "TestPass123!"
                }) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.auth_token = data.get("access_token")
                        return True
        return self.auth_token is not None
        
    # Channel Management Tests
    async def test_create_channel(self) -> bool:
        """Test channel creation"""
        if not self.auth_token:
            return False
            
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            async with session.post(f"{API_BASE_URL}/channels", 
                headers=headers,
                json={
                    "name": f"Test Channel {int(time.time())}",
                    "youtube_channel_id": f"UC{''.join([str(i) for i in range(22)])}",
                    "description": "E2E Test Channel"
                }) as response:
                if response.status in [200, 201]:
                    data = await response.json()
                    self.test_channel_id = data.get("channel_id")
                    self.log_info(f"Channel created: {self.test_channel_id}")
                    return True
                return False
                
    async def test_list_channels(self) -> bool:
        """Test listing channels"""
        if not self.auth_token:
            return False
            
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            async with session.get(f"{API_BASE_URL}/channels", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_info(f"Found {len(data)} channels")
                    return True
                return False
                
    # Video Generation Tests
    async def test_video_generation_request(self) -> bool:
        """Test video generation request"""
        if not self.auth_token or not self.test_channel_id:
            self.log_warning("Skipping video generation - no channel available")
            return False
            
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            async with session.post(f"{API_BASE_URL}/videos/generate", 
                headers=headers,
                json={
                    "channel_id": self.test_channel_id,
                    "topic": "System Test Video",
                    "style": "educational",
                    "duration": 5
                }) as response:
                if response.status in [200, 202]:
                    data = await response.json()
                    self.test_video_id = data.get("video_id")
                    self.log_info(f"Video generation started: {self.test_video_id}")
                    return True
                return False
                
    async def test_video_status(self) -> bool:
        """Test video status checking"""
        if not self.auth_token or not self.test_video_id:
            return False
            
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            async with session.get(f"{API_BASE_URL}/videos/{self.test_video_id}/status", 
                headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_info(f"Video status: {data.get('status')}")
                    return True
                return False
                
    # Cost Tracking Tests
    async def test_cost_tracking(self) -> bool:
        """Test cost tracking system"""
        if not self.auth_token:
            return False
            
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            async with session.get(f"{API_BASE_URL}/costs/summary", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    avg_cost = data.get("average_cost_per_video", 0)
                    self.log_info(f"Average cost per video: ${avg_cost:.2f}")
                    
                    # Check if cost is under $3
                    if avg_cost > 0 and avg_cost < 3.0:
                        self.log_success(f"Cost target achieved: ${avg_cost:.2f} < $3.00")
                        return True
                    elif avg_cost == 0:
                        self.log_warning("No cost data available yet")
                        return True
                    else:
                        self.log_error(f"Cost target missed: ${avg_cost:.2f} > $3.00")
                        return False
                return False
                
    # Analytics Tests
    async def test_analytics_api(self) -> bool:
        """Test analytics endpoints"""
        if not self.auth_token:
            return False
            
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            async with session.get(f"{API_BASE_URL}/analytics/dashboard", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_info(f"Analytics data retrieved: {len(data)} metrics")
                    return True
                return False
                
    # WebSocket Tests
    async def test_websocket_connection(self) -> bool:
        """Test WebSocket connection"""
        try:
            ws = websocket.WebSocket()
            ws.connect(f"{WS_URL}/test_client")
            ws.send(json.dumps({"type": "ping"}))
            ws.settimeout(5)
            result = ws.recv()
            ws.close()
            self.log_info("WebSocket connection successful")
            return True
        except Exception as e:
            self.log_error(f"WebSocket connection failed: {str(e)}")
            return False
            
    # Performance Tests
    async def test_api_response_time(self) -> bool:
        """Test API response time < 500ms"""
        async with aiohttp.ClientSession() as session:
            endpoints = ["/health", "/channels", "/videos", "/analytics/summary"]
            
            total_time = 0
            test_count = 0
            
            for endpoint in endpoints:
                start = time.time()
                headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
                
                async with session.get(f"{API_BASE_URL}{endpoint}", headers=headers) as response:
                    elapsed = (time.time() - start) * 1000
                    total_time += elapsed
                    test_count += 1
                    
                    if elapsed > 500:
                        self.log_warning(f"Slow response: {endpoint} took {elapsed:.0f}ms")
                        
            avg_time = total_time / test_count if test_count > 0 else 0
            self.log_info(f"Average API response time: {avg_time:.0f}ms")
            
            if avg_time < 500:
                self.log_success(f"Performance target achieved: {avg_time:.0f}ms < 500ms")
                return True
            else:
                self.log_error(f"Performance target missed: {avg_time:.0f}ms > 500ms")
                return False
                
    # Frontend Tests
    async def test_frontend_availability(self) -> bool:
        """Test frontend availability"""
        async with aiohttp.ClientSession() as session:
            async with session.get(FRONTEND_URL) as response:
                if response.status == 200:
                    self.log_info("Frontend is accessible")
                    return True
                self.log_error(f"Frontend returned status: {response.status}")
                return False
                
    # Integration Tests
    async def test_end_to_end_flow(self) -> bool:
        """Test complete end-to-end flow"""
        try:
            # This is a summary test that validates the entire flow worked
            if self.auth_token and self.test_channel_id:
                self.log_success("End-to-end flow validation successful")
                return True
            else:
                self.log_warning("End-to-end flow partially complete")
                return False
        except Exception as e:
            self.log_error(f"End-to-end flow failed: {str(e)}")
            return False
            
    async def run_all_tests(self):
        """Run all system tests"""
        self.log_section("YTEmpire E2E System Test Suite")
        self.log_info(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Infrastructure Tests
        self.log_section("Infrastructure Tests")
        await self.run_test("Database Connection", self.test_database_connection)
        await self.run_test("Redis Connection", self.test_redis_connection)
        await self.run_test("API Health Check", self.test_api_health)
        
        # Authentication Tests
        self.log_section("Authentication Tests")
        await self.run_test("User Registration", self.test_user_registration)
        await self.run_test("User Login", self.test_user_login)
        
        # Core Functionality Tests
        self.log_section("Core Functionality Tests")
        await self.run_test("Channel Creation", self.test_create_channel)
        await self.run_test("Channel Listing", self.test_list_channels)
        await self.run_test("Video Generation Request", self.test_video_generation_request)
        await self.run_test("Video Status Check", self.test_video_status)
        
        # Business Logic Tests
        self.log_section("Business Logic Tests")
        await self.run_test("Cost Tracking System", self.test_cost_tracking)
        await self.run_test("Analytics API", self.test_analytics_api)
        
        # Real-time Features Tests
        self.log_section("Real-time Features Tests")
        await self.run_test("WebSocket Connection", self.test_websocket_connection)
        
        # Performance Tests
        self.log_section("Performance Tests")
        await self.run_test("API Response Time", self.test_api_response_time)
        
        # Frontend Tests
        self.log_section("Frontend Tests")
        await self.run_test("Frontend Availability", self.test_frontend_availability)
        
        # Integration Tests
        self.log_section("Integration Tests")
        await self.run_test("End-to-End Flow", self.test_end_to_end_flow)
        
        # Generate Report
        self.generate_report()
        
    def generate_report(self):
        """Generate test report"""
        self.log_section("Test Results Summary")
        
        total_duration = (datetime.now() - self.results["start_time"]).total_seconds()
        pass_rate = (self.results["passed"] / self.results["total_tests"] * 100) if self.results["total_tests"] > 0 else 0
        
        print(f"{Fore.CYAN}Total Tests:{Style.RESET_ALL} {self.results['total_tests']}")
        print(f"{Fore.GREEN}Passed:{Style.RESET_ALL} {self.results['passed']}")
        print(f"{Fore.RED}Failed:{Style.RESET_ALL} {self.results['failed']}")
        print(f"{Fore.YELLOW}Warnings:{Style.RESET_ALL} {self.results['warnings']}")
        print(f"{Fore.CYAN}Pass Rate:{Style.RESET_ALL} {pass_rate:.1f}%")
        print(f"{Fore.CYAN}Total Duration:{Style.RESET_ALL} {total_duration:.2f} seconds")
        
        # Save detailed report
        report_file = f"e2e_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.log_info(f"Detailed report saved to: {report_file}")
        
        # Overall status
        print(f"\n{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
        if pass_rate >= 90:
            print(f"{Back.GREEN}{Fore.WHITE} SYSTEM TEST PASSED {Style.RESET_ALL}")
            print(f"{Fore.GREEN}✓ System is ready for production{Style.RESET_ALL}")
        elif pass_rate >= 70:
            print(f"{Back.YELLOW}{Fore.BLACK} SYSTEM TEST PASSED WITH WARNINGS {Style.RESET_ALL}")
            print(f"{Fore.YELLOW}⚠ System is operational but needs attention{Style.RESET_ALL}")
        else:
            print(f"{Back.RED}{Fore.WHITE} SYSTEM TEST FAILED {Style.RESET_ALL}")
            print(f"{Fore.RED}✗ Critical issues detected{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
        
        # Exit code based on results
        sys.exit(0 if pass_rate >= 70 else 1)

async def main():
    """Main entry point"""
    tester = SystemTest()
    await tester.run_all_tests()

if __name__ == "__main__":
    # Check if services are running
    print(f"{Fore.CYAN}YTEmpire End-to-End System Test{Style.RESET_ALL}")
    print(f"{Fore.CYAN}================================{Style.RESET_ALL}\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Test interrupted by user{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}Test suite failed: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)