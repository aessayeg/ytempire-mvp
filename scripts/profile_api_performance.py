"""
API Performance Profiling Script
Measures and reports API endpoint performance metrics
"""
import asyncio
import time
import statistics
import json
from typing import List, Dict, Any
import aiohttp
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

class APIPerformanceProfiler:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
        self.results = []
        self.token = None
        
    async def login(self, email: str = "test@example.com", password: str = "testpass123"):
        """Get authentication token"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/auth/login",
                json={"email": email, "password": password}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.token = data.get("access_token")
                    return True
        return False
    
    async def measure_endpoint(
        self, 
        method: str, 
        endpoint: str, 
        payload: Dict = None,
        iterations: int = 10
    ) -> Dict[str, Any]:
        """Measure single endpoint performance"""
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        times = []
        sizes = []
        statuses = []
        
        async with aiohttp.ClientSession() as session:
            for _ in range(iterations):
                start = time.perf_counter()
                
                try:
                    if method == "GET":
                        async with session.get(
                            f"{self.api_url}{endpoint}",
                            headers=headers
                        ) as response:
                            data = await response.read()
                            elapsed = (time.perf_counter() - start) * 1000  # ms
                            times.append(elapsed)
                            sizes.append(len(data))
                            statuses.append(response.status)
                    
                    elif method == "POST":
                        async with session.post(
                            f"{self.api_url}{endpoint}",
                            json=payload or {},
                            headers=headers
                        ) as response:
                            data = await response.read()
                            elapsed = (time.perf_counter() - start) * 1000
                            times.append(elapsed)
                            sizes.append(len(data))
                            statuses.append(response.status)
                            
                except Exception as e:
                    print(f"Error testing {endpoint}: {e}")
                    times.append(999999)  # Error indicator
                    sizes.append(0)
                    statuses.append(500)
                
                # Small delay between requests
                await asyncio.sleep(0.1)
        
        # Calculate statistics
        result = {
            "endpoint": endpoint,
            "method": method,
            "iterations": iterations,
            "avg_time_ms": statistics.mean(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "p50_time_ms": statistics.median(times),
            "p95_time_ms": sorted(times)[int(len(times) * 0.95)] if times else 0,
            "p99_time_ms": sorted(times)[int(len(times) * 0.99)] if times else 0,
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
            "avg_size_bytes": statistics.mean(sizes) if sizes else 0,
            "success_rate": sum(1 for s in statuses if s < 400) / len(statuses) * 100,
            "raw_times": times
        }
        
        return result
    
    async def profile_all_endpoints(self):
        """Profile all main API endpoints"""
        endpoints = [
            # Health checks
            ("GET", "/health", None),
            
            # Authentication
            ("POST", "/auth/login", {"email": "test@example.com", "password": "test123"}),
            ("POST", "/auth/register", {"email": f"test{int(time.time())}@example.com", "password": "test123"}),
            
            # User endpoints
            ("GET", "/users/me", None),
            
            # Channel endpoints
            ("GET", "/channels", None),
            ("POST", "/channels", {"name": f"Test Channel {int(time.time())}", "description": "Test"}),
            
            # Video endpoints
            ("GET", "/videos", None),
            ("POST", "/videos/generate", {"topic": "Test topic", "style": "educational"}),
            
            # Analytics
            ("GET", "/analytics/overview", None),
            ("GET", "/dashboard/stats", None),
            
            # Optimized endpoints
            ("GET", "/channels/optimized/dashboard", None),
            ("GET", "/channels/optimized/channels", None),
        ]
        
        print("\n" + "="*60)
        print("  API Performance Profiling")
        print("="*60)
        
        for method, endpoint, payload in endpoints:
            print(f"\nTesting {method} {endpoint}...")
            result = await self.measure_endpoint(method, endpoint, payload, iterations=10)
            self.results.append(result)
            
            # Print summary
            print(f"  Avg: {result['avg_time_ms']:.2f}ms")
            print(f"  P95: {result['p95_time_ms']:.2f}ms")
            print(f"  Size: {result['avg_size_bytes']:.0f} bytes")
            print(f"  Success: {result['success_rate']:.0f}%")
    
    def generate_report(self):
        """Generate performance report"""
        print("\n" + "="*60)
        print("  Performance Report")
        print("="*60)
        
        # Sort by average time
        sorted_results = sorted(self.results, key=lambda x: x['avg_time_ms'])
        
        # Create DataFrame for better visualization
        df = pd.DataFrame(sorted_results)
        
        print("\n📊 Performance Summary:")
        print("-" * 60)
        print(f"{'Endpoint':<40} {'Avg (ms)':<10} {'P95 (ms)':<10} {'Status':<10}")
        print("-" * 60)
        
        for result in sorted_results:
            endpoint = f"{result['method']} {result['endpoint']}"[:40]
            avg_time = result['avg_time_ms']
            p95_time = result['p95_time_ms']
            
            # Status indicator
            if avg_time < 100:
                status = "✅ Excellent"
            elif avg_time < 500:
                status = "✓ Good"
            elif avg_time < 1000:
                status = "⚠ Slow"
            else:
                status = "❌ Critical"
            
            print(f"{endpoint:<40} {avg_time:>8.2f}   {p95_time:>8.2f}   {status}")
        
        # Overall statistics
        print("\n📈 Overall Statistics:")
        print("-" * 60)
        all_times = [t for r in self.results for t in r['raw_times']]
        print(f"Total requests: {len(all_times)}")
        print(f"Average response time: {statistics.mean(all_times):.2f}ms")
        print(f"Median response time: {statistics.median(all_times):.2f}ms")
        print(f"P95 response time: {sorted(all_times)[int(len(all_times) * 0.95)]:.2f}ms")
        print(f"P99 response time: {sorted(all_times)[int(len(all_times) * 0.99)]:.2f}ms")
        
        # Identify slow endpoints
        slow_endpoints = [r for r in self.results if r['p95_time_ms'] > 500]
        if slow_endpoints:
            print("\n⚠️  Slow Endpoints (P95 > 500ms):")
            for endpoint in slow_endpoints:
                print(f"  - {endpoint['method']} {endpoint['endpoint']}: {endpoint['p95_time_ms']:.2f}ms")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_report_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Detailed report saved to: {filename}")
        
        return df
    
    def plot_performance(self):
        """Create performance visualization"""
        if not self.results:
            print("No results to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('API Performance Analysis', fontsize=16)
        
        # Prepare data
        endpoints = [f"{r['method']} {r['endpoint'][:20]}" for r in self.results]
        avg_times = [r['avg_time_ms'] for r in self.results]
        p95_times = [r['p95_time_ms'] for r in self.results]
        
        # Plot 1: Average response times
        axes[0, 0].barh(endpoints, avg_times)
        axes[0, 0].set_xlabel('Average Response Time (ms)')
        axes[0, 0].set_title('Average Response Times by Endpoint')
        axes[0, 0].axvline(x=500, color='r', linestyle='--', label='Target (500ms)')
        
        # Plot 2: P95 response times
        axes[0, 1].barh(endpoints, p95_times)
        axes[0, 1].set_xlabel('P95 Response Time (ms)')
        axes[0, 1].set_title('P95 Response Times by Endpoint')
        axes[0, 1].axvline(x=500, color='r', linestyle='--', label='Target (500ms)')
        
        # Plot 3: Response time distribution
        all_times = [t for r in self.results for t in r['raw_times'] if t < 5000]
        axes[1, 0].hist(all_times, bins=50, edgecolor='black')
        axes[1, 0].set_xlabel('Response Time (ms)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Response Time Distribution')
        axes[1, 0].axvline(x=500, color='r', linestyle='--', label='Target')
        
        # Plot 4: Success rates
        success_rates = [r['success_rate'] for r in self.results]
        axes[1, 1].bar(range(len(endpoints)), success_rates)
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].set_title('Success Rates by Endpoint')
        axes[1, 1].set_xticks(range(len(endpoints)))
        axes[1, 1].set_xticklabels(endpoints, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_plot_{timestamp}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"📊 Performance plot saved to: {filename}")
        
        plt.show()

async def main():
    """Run performance profiling"""
    profiler = APIPerformanceProfiler()
    
    # Login first (if needed)
    # await profiler.login()
    
    # Profile endpoints
    await profiler.profile_all_endpoints()
    
    # Generate report
    df = profiler.generate_report()
    
    # Create visualizations
    try:
        profiler.plot_performance()
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    return profiler.results

if __name__ == "__main__":
    results = asyncio.run(main())