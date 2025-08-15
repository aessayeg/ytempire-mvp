"""
Scaling Optimizer Service
Handles 10x volume scaling for real-time processing
"""
import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from collections import defaultdict, deque
import numpy as np
import psutil
import aiohttp
import weakref
import gc
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)


class ScalingMode(Enum):
    """Scaling modes for different load levels"""
    NORMAL = "normal"          # Standard processing
    HIGH_LOAD = "high_load"    # 2-5x normal load
    PEAK_LOAD = "peak_load"    # 5-10x normal load
    EMERGENCY = "emergency"    # >10x normal load


@dataclass
class SystemMetrics:
    """Current system performance metrics"""
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_usage: float
    network_io: Dict[str, float]
    queue_depths: Dict[str, int]
    processing_rates: Dict[str, float]
    error_rates: Dict[str, float]
    response_times: Dict[str, float]
    active_connections: int
    timestamp: datetime


@dataclass
class ScalingConfiguration:
    """Configuration for scaling behavior"""
    mode: ScalingMode
    max_concurrent_tasks: int
    batch_size: int
    flush_interval: int  # seconds
    memory_threshold: float  # percentage
    cpu_threshold: float  # percentage
    queue_threshold: int  # max queue depth
    circuit_breaker_threshold: float  # error rate
    cache_ttl: int  # seconds
    worker_pool_size: int
    async_workers: int


class CircuitBreaker:
    """Circuit breaker for service protection"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if datetime.utcnow().timestamp() - self.last_failure_time < self.timeout:
                raise Exception("Circuit breaker is OPEN")
            else:
                self.state = "half_open"
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow().timestamp()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e


class AdaptiveCache:
    """Adaptive cache with intelligent TTL and eviction"""
    
    def __init__(self, redis_client: redis.Redis, max_memory_mb: int = 512):
        self.redis_client = redis_client
        self.max_memory_mb = max_memory_mb
        self.cache_stats = defaultdict(int)
        self.access_patterns = defaultdict(list)
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value with access tracking"""
        try:
            value = await self.redis_client.get(f"cache:{key}")
            if value:
                self.cache_stats['hits'] += 1
                self.access_patterns[key].append(datetime.utcnow())
                return json.loads(value)
            else:
                self.cache_stats['misses'] += 1
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with adaptive TTL"""
        try:
            # Calculate adaptive TTL based on access patterns
            if ttl is None:
                ttl = self._calculate_adaptive_ttl(key)
            
            await self.redis_client.setex(
                f"cache:{key}", 
                ttl, 
                json.dumps(value, default=str)
            )
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def _calculate_adaptive_ttl(self, key: str) -> int:
        """Calculate TTL based on access patterns"""
        accesses = self.access_patterns.get(key, [])
        
        if len(accesses) < 2:
            return 300  # Default 5 minutes
        
        # Calculate access frequency
        recent_accesses = [a for a in accesses if a > datetime.utcnow() - timedelta(hours=1)]
        frequency = len(recent_accesses)
        
        # High frequency = longer TTL
        if frequency > 10:
            return 1800  # 30 minutes
        elif frequency > 5:
            return 900   # 15 minutes
        else:
            return 300   # 5 minutes
    
    async def cleanup_expired(self):
        """Clean up expired and least accessed keys"""
        try:
            # Get cache memory usage
            info = await self.redis_client.info('memory')
            used_memory_mb = info.get('used_memory', 0) / 1024 / 1024
            
            if used_memory_mb > self.max_memory_mb:
                # Find least recently used keys
                now = datetime.utcnow()
                key_scores = []
                
                for key, accesses in self.access_patterns.items():
                    if accesses:
                        last_access = max(accesses)
                        score = (now - last_access).total_seconds()
                        key_scores.append((score, key))
                
                # Remove oldest 20% of keys
                key_scores.sort(reverse=True)
                keys_to_remove = key_scores[:len(key_scores) // 5]
                
                for _, key in keys_to_remove:
                    await self.redis_client.delete(f"cache:{key}")
                    del self.access_patterns[key]
                
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")


class ScalingOptimizer:
    """Main scaling optimizer for handling high load"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        
        # System monitoring
        self.current_metrics: Optional[SystemMetrics] = None
        self.metrics_history: deque = deque(maxlen=100)
        
        # Scaling configuration
        self.scaling_configs = {
            ScalingMode.NORMAL: ScalingConfiguration(
                mode=ScalingMode.NORMAL,
                max_concurrent_tasks=100,
                batch_size=50,
                flush_interval=30,
                memory_threshold=70.0,
                cpu_threshold=70.0,
                queue_threshold=1000,
                circuit_breaker_threshold=0.1,
                cache_ttl=300,
                worker_pool_size=4,
                async_workers=10
            ),
            ScalingMode.HIGH_LOAD: ScalingConfiguration(
                mode=ScalingMode.HIGH_LOAD,
                max_concurrent_tasks=500,
                batch_size=100,
                flush_interval=15,
                memory_threshold=80.0,
                cpu_threshold=80.0,
                queue_threshold=5000,
                circuit_breaker_threshold=0.15,
                cache_ttl=600,
                worker_pool_size=8,
                async_workers=25
            ),
            ScalingMode.PEAK_LOAD: ScalingConfiguration(
                mode=ScalingMode.PEAK_LOAD,
                max_concurrent_tasks=1000,
                batch_size=200,
                flush_interval=10,
                memory_threshold=85.0,
                cpu_threshold=85.0,
                queue_threshold=10000,
                circuit_breaker_threshold=0.2,
                cache_ttl=900,
                worker_pool_size=16,
                async_workers=50
            ),
            ScalingMode.EMERGENCY: ScalingConfiguration(
                mode=ScalingMode.EMERGENCY,
                max_concurrent_tasks=2000,
                batch_size=500,
                flush_interval=5,
                memory_threshold=90.0,
                cpu_threshold=90.0,
                queue_threshold=20000,
                circuit_breaker_threshold=0.3,
                cache_ttl=1800,
                worker_pool_size=32,
                async_workers=100
            )
        }
        
        self.current_mode = ScalingMode.NORMAL
        self.current_config = self.scaling_configs[ScalingMode.NORMAL]
        
        # Processing components
        self.adaptive_cache: Optional[AdaptiveCache] = None
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.worker_pool: Optional[ThreadPoolExecutor] = None
        self.processing_semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # Task queues for different priorities
        self.priority_queues = {
            'high': asyncio.Queue(maxsize=10000),
            'normal': asyncio.Queue(maxsize=50000),
            'low': asyncio.Queue(maxsize=100000)
        }
        
        # Processing state
        self.processing_active = False
        self.current_load = 0.0
        self.performance_scores = deque(maxlen=50)
        
        # Weak references for automatic cleanup
        self.active_tasks = weakref.WeakSet()
    
    async def initialize(self):
        """Initialize the scaling optimizer"""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            
            # Initialize adaptive cache
            self.adaptive_cache = AdaptiveCache(self.redis_client)
            
            # Initialize circuit breakers for different services
            self.circuit_breakers = {
                'analytics': CircuitBreaker(failure_threshold=3, timeout=30),
                'video_processing': CircuitBreaker(failure_threshold=5, timeout=60),
                'ai_services': CircuitBreaker(failure_threshold=2, timeout=45),
                'database': CircuitBreaker(failure_threshold=10, timeout=120)
            }
            
            # Initialize thread pool
            self.worker_pool = ThreadPoolExecutor(
                max_workers=self.current_config.worker_pool_size
            )
            
            # Initialize semaphores
            self._initialize_semaphores()
            
            # Start background tasks
            asyncio.create_task(self._system_monitor())
            asyncio.create_task(self._load_balancer())
            asyncio.create_task(self._auto_scaler())
            asyncio.create_task(self._performance_optimizer())
            asyncio.create_task(self._queue_processor())
            asyncio.create_task(self._cache_manager())
            asyncio.create_task(self._garbage_collector())
            
            self.processing_active = True
            logger.info("Scaling optimizer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize scaling optimizer: {e}")
            raise
    
    def _initialize_semaphores(self):
        """Initialize processing semaphores"""
        self.processing_semaphores = {
            'analytics': asyncio.Semaphore(self.current_config.async_workers),
            'video_processing': asyncio.Semaphore(max(1, self.current_config.async_workers // 4)),
            'ai_services': asyncio.Semaphore(max(1, self.current_config.async_workers // 2)),
            'database': asyncio.Semaphore(self.current_config.async_workers * 2),
            'general': asyncio.Semaphore(self.current_config.max_concurrent_tasks)
        }
    
    async def _system_monitor(self):
        """Monitor system performance metrics"""
        while self.processing_active:
            try:
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                
                # Get queue depths
                queue_depths = {
                    name: queue.qsize() 
                    for name, queue in self.priority_queues.items()
                }
                
                # Get processing rates (mock calculation)
                processing_rates = await self._calculate_processing_rates()
                error_rates = await self._calculate_error_rates()
                response_times = await self._calculate_response_times()
                
                # Create metrics object
                self.current_metrics = SystemMetrics(
                    cpu_usage=cpu_percent,
                    memory_usage=memory.percent,
                    memory_available=memory.available / 1024 / 1024 / 1024,  # GB
                    disk_usage=disk.percent,
                    network_io={
                        'bytes_sent': network.bytes_sent,
                        'bytes_recv': network.bytes_recv
                    },
                    queue_depths=queue_depths,
                    processing_rates=processing_rates,
                    error_rates=error_rates,
                    response_times=response_times,
                    active_connections=len(self.active_tasks),
                    timestamp=datetime.utcnow()
                )
                
                # Add to history
                self.metrics_history.append(self.current_metrics)
                
                # Calculate current load
                self._calculate_current_load()
                
                # Store metrics in Redis for monitoring
                await self._store_metrics()
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _calculate_processing_rates(self) -> Dict[str, float]:
        """Calculate processing rates for different components"""
        # This would calculate actual processing rates
        # For now, return mock data
        return {
            'analytics': 100.0,
            'video_processing': 5.0,
            'ai_services': 20.0,
            'database': 500.0
        }
    
    async def _calculate_error_rates(self) -> Dict[str, float]:
        """Calculate error rates for different components"""
        error_rates = {}
        for service, breaker in self.circuit_breakers.items():
            if breaker.state == "open":
                error_rates[service] = 1.0
            else:
                error_rates[service] = max(0.0, breaker.failure_count / 10)
        return error_rates
    
    async def _calculate_response_times(self) -> Dict[str, float]:
        """Calculate average response times"""
        # Mock response times - in production, track actual times
        return {
            'analytics': 50.0,
            'video_processing': 2000.0,
            'ai_services': 800.0,
            'database': 25.0
        }
    
    def _calculate_current_load(self):
        """Calculate current system load factor"""
        if not self.current_metrics:
            self.current_load = 0.0
            return
        
        # Weighted load calculation
        cpu_load = self.current_metrics.cpu_usage / 100
        memory_load = self.current_metrics.memory_usage / 100
        queue_load = sum(self.current_metrics.queue_depths.values()) / 10000
        
        # Weighted average
        self.current_load = (cpu_load * 0.4 + memory_load * 0.4 + queue_load * 0.2)
        
        # Store performance score
        performance_score = max(0.0, 1.0 - self.current_load)
        self.performance_scores.append(performance_score)
    
    async def _store_metrics(self):
        """Store metrics in Redis for monitoring"""
        if not self.current_metrics:
            return
        
        metrics_key = f"scaling:metrics:{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        metrics_data = asdict(self.current_metrics)
        
        await self.redis_client.setex(
            metrics_key, 3600, json.dumps(metrics_data, default=str)
        )
    
    async def _auto_scaler(self):
        """Automatically adjust scaling mode based on load"""
        while self.processing_active:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                if not self.current_metrics:
                    continue
                
                # Determine required scaling mode
                new_mode = self._determine_scaling_mode()
                
                if new_mode != self.current_mode:
                    await self._switch_scaling_mode(new_mode)
                
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(60)
    
    def _determine_scaling_mode(self) -> ScalingMode:
        """Determine required scaling mode based on current metrics"""
        if not self.current_metrics:
            return ScalingMode.NORMAL
        
        # Check for emergency conditions
        if (self.current_metrics.cpu_usage > 95 or 
            self.current_metrics.memory_usage > 95 or
            sum(self.current_metrics.queue_depths.values()) > 50000):
            return ScalingMode.EMERGENCY
        
        # Check for peak load
        if (self.current_metrics.cpu_usage > 85 or
            self.current_metrics.memory_usage > 85 or
            sum(self.current_metrics.queue_depths.values()) > 20000):
            return ScalingMode.PEAK_LOAD
        
        # Check for high load
        if (self.current_metrics.cpu_usage > 70 or
            self.current_metrics.memory_usage > 70 or
            sum(self.current_metrics.queue_depths.values()) > 5000):
            return ScalingMode.HIGH_LOAD
        
        return ScalingMode.NORMAL
    
    async def _switch_scaling_mode(self, new_mode: ScalingMode):
        """Switch to a new scaling mode"""
        logger.info(f"Switching scaling mode: {self.current_mode.value} -> {new_mode.value}")
        
        old_config = self.current_config
        self.current_mode = new_mode
        self.current_config = self.scaling_configs[new_mode]
        
        # Reconfigure components
        await self._reconfigure_for_scaling(old_config, self.current_config)
        
        logger.info(f"Scaling mode switched to {new_mode.value}")
    
    async def _reconfigure_for_scaling(self, old_config: ScalingConfiguration, new_config: ScalingConfiguration):
        """Reconfigure components for new scaling mode"""
        try:
            # Update semaphores
            self._initialize_semaphores()
            
            # Resize thread pool if significantly different
            if abs(new_config.worker_pool_size - old_config.worker_pool_size) > 2:
                if self.worker_pool:
                    self.worker_pool.shutdown(wait=False)
                
                self.worker_pool = ThreadPoolExecutor(
                    max_workers=new_config.worker_pool_size
                )
            
            # Update cache configuration
            if self.adaptive_cache:
                self.adaptive_cache.max_memory_mb = min(512, new_config.cache_ttl // 2)
            
            logger.info(f"Reconfigured for {new_config.mode.value}: "
                       f"tasks={new_config.max_concurrent_tasks}, "
                       f"batch={new_config.batch_size}, "
                       f"workers={new_config.worker_pool_size}")
            
        except Exception as e:
            logger.error(f"Reconfiguration error: {e}")
    
    async def _load_balancer(self):
        """Balance load across different processing queues"""
        while self.processing_active:
            try:
                await asyncio.sleep(5)  # Balance every 5 seconds
                
                # Check queue imbalances
                queue_sizes = {
                    name: queue.qsize() 
                    for name, queue in self.priority_queues.items()
                }
                
                # If normal queue is overloaded, promote some tasks to high priority
                if queue_sizes['normal'] > self.current_config.queue_threshold // 2:
                    await self._rebalance_queues()
                
            except Exception as e:
                logger.error(f"Load balancing error: {e}")
                await asyncio.sleep(30)
    
    async def _rebalance_queues(self):
        """Rebalance tasks across priority queues"""
        try:
            # Move some normal priority tasks to high priority
            normal_queue = self.priority_queues['normal']
            high_queue = self.priority_queues['high']
            
            moved_count = 0
            max_moves = min(100, normal_queue.qsize() // 10)
            
            for _ in range(max_moves):
                if normal_queue.empty():
                    break
                
                try:
                    task = normal_queue.get_nowait()
                    await high_queue.put(task)
                    moved_count += 1
                except asyncio.QueueEmpty:
                    break
                except asyncio.QueueFull:
                    # Put task back if high queue is full
                    await normal_queue.put(task)
                    break
            
            if moved_count > 0:
                logger.info(f"Rebalanced {moved_count} tasks to high priority queue")
                
        except Exception as e:
            logger.error(f"Queue rebalancing error: {e}")
    
    async def _queue_processor(self):
        """Process tasks from priority queues"""
        while self.processing_active:
            try:
                # Process in priority order
                for priority in ['high', 'normal', 'low']:
                    queue = self.priority_queues[priority]
                    
                    if not queue.empty():
                        # Process batch of tasks
                        batch = []
                        batch_size = min(self.current_config.batch_size, queue.qsize())
                        
                        for _ in range(batch_size):
                            try:
                                task = queue.get_nowait()
                                batch.append(task)
                            except asyncio.QueueEmpty:
                                break
                        
                        if batch:
                            await self._process_task_batch(batch, priority)
                
                # Short sleep to prevent CPU spinning
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(5)
    
    async def _process_task_batch(self, tasks: List[Any], priority: str):
        """Process a batch of tasks"""
        try:
            # Use appropriate semaphore
            semaphore = self.processing_semaphores.get('general')
            
            async with semaphore:
                # Process tasks concurrently
                task_coroutines = []
                for task in tasks:
                    coroutine = self._process_single_task(task, priority)
                    task_coroutines.append(coroutine)
                
                # Execute with timeout based on priority
                timeout = 30 if priority == 'high' else 60 if priority == 'normal' else 120
                
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*task_coroutines, return_exceptions=True),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Batch processing timeout for {priority} priority tasks")
                
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
    
    async def _process_single_task(self, task: Any, priority: str):
        """Process a single task"""
        try:
            # Mock task processing - in production this would route to appropriate handlers
            task_type = task.get('type', 'unknown')
            
            # Use circuit breaker if available
            circuit_breaker = self.circuit_breakers.get(task_type)
            
            if circuit_breaker:
                await circuit_breaker.call(self._execute_task, task)
            else:
                await self._execute_task(task)
                
        except Exception as e:
            logger.error(f"Task processing error: {e}")
    
    async def _execute_task(self, task: Any):
        """Execute a task (mock implementation)"""
        # Mock task execution
        task_type = task.get('type', 'unknown')
        processing_time = {
            'analytics': 0.1,
            'video_processing': 2.0,
            'ai_services': 1.0,
            'database': 0.05
        }.get(task_type, 0.5)
        
        await asyncio.sleep(processing_time)
    
    async def _performance_optimizer(self):
        """Optimize performance based on metrics"""
        while self.processing_active:
            try:
                await asyncio.sleep(60)  # Optimize every minute
                
                if len(self.performance_scores) < 10:
                    continue
                
                # Calculate average performance
                avg_performance = np.mean(list(self.performance_scores)[-10:])
                
                # If performance is consistently low, apply optimizations
                if avg_performance < 0.7:
                    await self._apply_performance_optimizations()
                
                # If performance is good, we can relax some constraints
                elif avg_performance > 0.9:
                    await self._relax_performance_constraints()
                
            except Exception as e:
                logger.error(f"Performance optimization error: {e}")
                await asyncio.sleep(300)
    
    async def _apply_performance_optimizations(self):
        """Apply performance optimizations during high load"""
        logger.info("Applying performance optimizations")
        
        # Increase cache TTL
        if self.adaptive_cache:
            for key in list(self.adaptive_cache.access_patterns.keys())[:100]:
                # Extend TTL for frequently accessed keys
                accesses = self.adaptive_cache.access_patterns[key]
                if len(accesses) > 5:
                    await self.redis_client.expire(f"cache:{key}", 1800)  # 30 minutes
        
        # Trigger garbage collection
        gc.collect()
        
        # Reduce batch sizes temporarily
        if self.current_config.batch_size > 20:
            self.current_config.batch_size = max(20, self.current_config.batch_size // 2)
    
    async def _relax_performance_constraints(self):
        """Relax constraints when performance is good"""
        logger.info("Relaxing performance constraints")
        
        # Restore normal batch sizes
        normal_config = self.scaling_configs[self.current_mode]
        self.current_config.batch_size = normal_config.batch_size
    
    async def _cache_manager(self):
        """Manage cache lifecycle and optimization"""
        while self.processing_active:
            try:
                await asyncio.sleep(300)  # Manage every 5 minutes
                
                if self.adaptive_cache:
                    await self.adaptive_cache.cleanup_expired()
                
            except Exception as e:
                logger.error(f"Cache management error: {e}")
                await asyncio.sleep(600)
    
    async def _garbage_collector(self):
        """Periodic garbage collection"""
        while self.processing_active:
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                
                # Force garbage collection during high memory usage
                if self.current_metrics and self.current_metrics.memory_usage > 80:
                    gc.collect()
                    logger.info("Performed garbage collection due to high memory usage")
                
            except Exception as e:
                logger.error(f"Garbage collection error: {e}")
                await asyncio.sleep(1200)
    
    # Public API methods
    
    async def submit_task(self, task: Dict[str, Any], priority: str = 'normal') -> bool:
        """Submit a task for processing"""
        try:
            queue = self.priority_queues.get(priority, self.priority_queues['normal'])
            
            # Add task with timeout to prevent blocking
            await asyncio.wait_for(queue.put(task), timeout=5)
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"Failed to submit task - {priority} queue full")
            return False
        except Exception as e:
            logger.error(f"Task submission error: {e}")
            return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'scaling_mode': self.current_mode.value,
            'current_load': self.current_load,
            'performance_score': np.mean(list(self.performance_scores)[-10:]) if self.performance_scores else 0.0,
            'queue_depths': {
                name: queue.qsize() 
                for name, queue in self.priority_queues.items()
            },
            'circuit_breakers': {
                name: breaker.state 
                for name, breaker in self.circuit_breakers.items()
            },
            'metrics': asdict(self.current_metrics) if self.current_metrics else {},
            'active_tasks': len(self.active_tasks),
            'processing_active': self.processing_active
        }
    
    async def force_scaling_mode(self, mode: ScalingMode):
        """Force a specific scaling mode"""
        if mode in self.scaling_configs:
            await self._switch_scaling_mode(mode)
            logger.info(f"Forced scaling mode to {mode.value}")
        else:
            raise ValueError(f"Invalid scaling mode: {mode}")
    
    async def get_performance_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get performance metrics for the specified time period"""
        try:
            pattern = "scaling:metrics:*"
            cursor = 0
            metrics = []
            
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            while True:
                cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        try:
                            metric_data = json.loads(data)
                            timestamp = datetime.fromisoformat(metric_data['timestamp'])
                            
                            if timestamp > cutoff_time:
                                metrics.append(metric_data)
                        except:
                            continue
                
                if cursor == 0:
                    break
            
            return sorted(metrics, key=lambda x: x['timestamp'])
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return []
    
    async def shutdown(self):
        """Shutdown the scaling optimizer gracefully"""
        self.processing_active = False
        
        # Shutdown thread pool
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Scaling optimizer shut down")


# Global instance
scaling_optimizer = ScalingOptimizer()