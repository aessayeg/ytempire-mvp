#!/usr/bin/env python3
"""
Auto-Scaling Implementation for YTEmpire MVP
Comprehensive resource scaling based on metrics and demand
"""

import asyncio
import aiohttp
import logging
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import docker
import subprocess
import psutil
from redis import Redis
import psycopg2

logger = logging.getLogger(__name__)

class ScaleDirection(Enum):
    UP = "up"
    DOWN = "down"
    NONE = "none"

class ScaleAction(Enum):
    HORIZONTAL = "horizontal"  # Add/remove containers
    VERTICAL = "vertical"     # Adjust CPU/memory limits
    HYBRID = "hybrid"         # Both horizontal and vertical

@dataclass
class ScalingMetrics:
    """Current system metrics for scaling decisions"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    request_rate: float
    response_time_p95: float
    queue_depth: int
    active_connections: int
    error_rate: float
    
    # Video processing specific
    video_queue_length: int
    processing_rate: float
    gpu_utilization: float
    storage_usage: float

@dataclass
class ScalingRule:
    """Scaling rule configuration"""
    name: str
    metric: str
    threshold_up: float
    threshold_down: float
    action: ScaleAction
    cooldown_seconds: int
    min_instances: int
    max_instances: int
    scale_factor: float
    enabled: bool = True

@dataclass
class ServiceConfig:
    """Service scaling configuration"""
    name: str
    current_instances: int
    min_instances: int
    max_instances: int
    cpu_request: str
    cpu_limit: str
    memory_request: str
    memory_limit: str
    rules: List[ScalingRule]
    last_scaled: Optional[datetime] = None

@dataclass
class ScalingEvent:
    """Record of scaling action"""
    timestamp: datetime
    service: str
    action: str
    old_instances: int
    new_instances: int
    trigger_metric: str
    trigger_value: float
    success: bool
    error_message: Optional[str] = None

class MetricsCollector:
    """Collects system and application metrics"""
    
    def __init__(self):
        self.redis = Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            password=os.getenv('REDIS_PASSWORD') or None,
            decode_responses=True
        )
        
        self.db_config = {
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'port': int(os.getenv('DATABASE_PORT', '5432')),
            'database': os.getenv('DATABASE_NAME', 'ytempire'),
            'user': os.getenv('DATABASE_USER', 'postgres'),
            'password': os.getenv('DATABASE_PASSWORD', 'password')
        }
        
        self.api_base_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
    
    async def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            storage_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # GPU utilization (if available)
            gpu_utilization = await self._get_gpu_utilization()
            
            return {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'storage_usage': storage_usage,
                'network_bytes_sent': net_io.bytes_sent,
                'network_bytes_recv': net_io.bytes_recv,
                'gpu_utilization': gpu_utilization
            }
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}
    
    async def collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics"""
        try:
            metrics = {}
            
            # API metrics
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"{self.api_base_url}/metrics") as response:
                        if response.status == 200:
                            api_metrics = await response.json()
                            metrics.update(api_metrics)
                except Exception as e:
                    logger.warning(f"Failed to collect API metrics: {e}")
            
            # Database metrics
            try:
                conn = psycopg2.connect(**self.db_config)
                cursor = conn.cursor()
                
                # Active connections
                cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
                active_connections = cursor.fetchone()[0]
                
                # Queue depth (video processing queue)
                cursor.execute("SELECT count(*) FROM video_queue WHERE status = 'pending'")
                queue_depth = cursor.fetchone()[0]
                
                # Video queue length
                cursor.execute("SELECT count(*) FROM video_queue WHERE status IN ('pending', 'processing')")
                video_queue_length = cursor.fetchone()[0]
                
                cursor.close()
                conn.close()
                
                metrics.update({
                    'active_connections': active_connections,
                    'queue_depth': queue_depth,
                    'video_queue_length': video_queue_length
                })
                
            except Exception as e:
                logger.warning(f"Failed to collect database metrics: {e}")
            
            # Redis metrics
            try:
                redis_info = self.redis.info()
                metrics.update({
                    'redis_connected_clients': redis_info.get('connected_clients', 0),
                    'redis_used_memory': redis_info.get('used_memory', 0),
                    'redis_operations_per_sec': redis_info.get('instantaneous_ops_per_sec', 0)
                })
            except Exception as e:
                logger.warning(f"Failed to collect Redis metrics: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            return {}
    
    async def _get_gpu_utilization(self) -> float:
        """Get GPU utilization if available"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return 0.0
    
    async def collect_all_metrics(self) -> ScalingMetrics:
        """Collect comprehensive metrics for scaling decisions"""
        system_metrics = await self.collect_system_metrics()
        app_metrics = await self.collect_application_metrics()
        
        return ScalingMetrics(
            timestamp=datetime.now(),
            cpu_usage=system_metrics.get('cpu_usage', 0),
            memory_usage=system_metrics.get('memory_usage', 0),
            request_rate=app_metrics.get('request_rate', 0),
            response_time_p95=app_metrics.get('response_time_p95', 0),
            queue_depth=app_metrics.get('queue_depth', 0),
            active_connections=app_metrics.get('active_connections', 0),
            error_rate=app_metrics.get('error_rate', 0),
            video_queue_length=app_metrics.get('video_queue_length', 0),
            processing_rate=app_metrics.get('processing_rate', 0),
            gpu_utilization=system_metrics.get('gpu_utilization', 0),
            storage_usage=system_metrics.get('storage_usage', 0)
        )

class DockerScaler:
    """Docker-based container scaling"""
    
    def __init__(self):
        self.client = docker.from_env()
    
    async def scale_service(self, service_name: str, target_replicas: int) -> bool:
        """Scale Docker service to target replicas"""
        try:
            # For Docker Compose services
            result = subprocess.run([
                'docker-compose', 'up', '--scale', f'{service_name}={target_replicas}', '-d'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Scaled {service_name} to {target_replicas} replicas")
                return True
            else:
                logger.error(f"Failed to scale {service_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Docker scaling error for {service_name}: {e}")
            return False
    
    async def get_service_replicas(self, service_name: str) -> int:
        """Get current number of replicas for a service"""
        try:
            containers = self.client.containers.list(
                filters={'label': f'com.docker.compose.service={service_name}'}
            )
            return len(containers)
        except Exception as e:
            logger.error(f"Failed to get replicas for {service_name}: {e}")
            return 0
    
    async def update_resource_limits(self, service_name: str, cpu_limit: str, memory_limit: str) -> bool:
        """Update resource limits for containers"""
        try:
            containers = self.client.containers.list(
                filters={'label': f'com.docker.compose.service={service_name}'}
            )
            
            success = True
            for container in containers:
                try:
                    container.update(
                        cpu_quota=int(float(cpu_limit) * 100000),  # Convert to CPU quota
                        mem_limit=memory_limit
                    )
                    logger.info(f"Updated resources for {container.name}")
                except Exception as e:
                    logger.error(f"Failed to update {container.name}: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update resources for {service_name}: {e}")
            return False

class KubernetesScaler:
    """Kubernetes-based scaling (for future use)"""
    
    def __init__(self):
        self.api_client = None  # Would initialize k8s client
    
    async def scale_deployment(self, name: str, namespace: str, replicas: int) -> bool:
        """Scale Kubernetes deployment"""
        # Placeholder for Kubernetes implementation
        logger.info(f"K8s scaling: {name} to {replicas} replicas in {namespace}")
        return True
    
    async def update_hpa(self, name: str, namespace: str, min_replicas: int, max_replicas: int) -> bool:
        """Update Horizontal Pod Autoscaler"""
        # Placeholder for HPA implementation
        logger.info(f"K8s HPA update: {name} min={min_replicas}, max={max_replicas}")
        return True

class AutoScaler:
    """Main auto-scaling controller"""
    
    def __init__(self, config_path: str = None):
        self.metrics_collector = MetricsCollector()
        self.docker_scaler = DockerScaler()
        self.k8s_scaler = KubernetesScaler()
        
        self.services = self._load_service_configs(config_path)
        self.scaling_events: List[ScalingEvent] = []
        self.metrics_history: List[ScalingMetrics] = []
        
        # Configuration
        self.evaluation_interval = 30  # seconds
        self.metrics_retention_hours = 24
        
    def _load_service_configs(self, config_path: str = None) -> Dict[str, ServiceConfig]:
        """Load service scaling configurations"""
        default_config = {
            'backend': ServiceConfig(
                name='backend',
                current_instances=2,
                min_instances=1,
                max_instances=10,
                cpu_request='500m',
                cpu_limit='2',
                memory_request='512Mi',
                memory_limit='2Gi',
                rules=[
                    ScalingRule(
                        name='cpu_usage',
                        metric='cpu_usage',
                        threshold_up=80.0,
                        threshold_down=30.0,
                        action=ScaleAction.HORIZONTAL,
                        cooldown_seconds=300,
                        min_instances=1,
                        max_instances=10,
                        scale_factor=1.5
                    ),
                    ScalingRule(
                        name='response_time',
                        metric='response_time_p95',
                        threshold_up=2000.0,  # 2 seconds
                        threshold_down=500.0,  # 500ms
                        action=ScaleAction.HORIZONTAL,
                        cooldown_seconds=180,
                        min_instances=1,
                        max_instances=8,
                        scale_factor=1.3
                    ),
                    ScalingRule(
                        name='queue_depth',
                        metric='queue_depth',
                        threshold_up=50,
                        threshold_down=10,
                        action=ScaleAction.HORIZONTAL,
                        cooldown_seconds=120,
                        min_instances=1,
                        max_instances=15,
                        scale_factor=2.0
                    )
                ]
            ),
            'celery_worker': ServiceConfig(
                name='celery_worker',
                current_instances=3,
                min_instances=1,
                max_instances=20,
                cpu_request='1',
                cpu_limit='4',
                memory_request='1Gi',
                memory_limit='8Gi',
                rules=[
                    ScalingRule(
                        name='video_queue',
                        metric='video_queue_length',
                        threshold_up=20,
                        threshold_down=5,
                        action=ScaleAction.HORIZONTAL,
                        cooldown_seconds=180,
                        min_instances=1,
                        max_instances=20,
                        scale_factor=1.5
                    ),
                    ScalingRule(
                        name='gpu_utilization',
                        metric='gpu_utilization',
                        threshold_up=85.0,
                        threshold_down=40.0,
                        action=ScaleAction.HORIZONTAL,
                        cooldown_seconds=300,
                        min_instances=1,
                        max_instances=10,
                        scale_factor=1.2
                    )
                ]
            ),
            'redis': ServiceConfig(
                name='redis',
                current_instances=1,
                min_instances=1,
                max_instances=3,
                cpu_request='250m',
                cpu_limit='1',
                memory_request='256Mi',
                memory_limit='1Gi',
                rules=[
                    ScalingRule(
                        name='memory_usage',
                        metric='memory_usage',
                        threshold_up=80.0,
                        threshold_down=50.0,
                        action=ScaleAction.VERTICAL,
                        cooldown_seconds=600,
                        min_instances=1,
                        max_instances=3,
                        scale_factor=1.5
                    )
                ]
            )
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with default config
                for service_name, config in loaded_config.items():
                    if service_name in default_config:
                        # Update existing config
                        default_config[service_name].__dict__.update(config)
                    else:
                        # Add new service config
                        default_config[service_name] = ServiceConfig(**config)
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    async def evaluate_scaling_decision(self, metrics: ScalingMetrics) -> List[Tuple[str, ScaleDirection, int, str]]:
        """Evaluate scaling decisions for all services"""
        decisions = []
        
        for service_name, config in self.services.items():
            for rule in config.rules:
                if not rule.enabled:
                    continue
                
                # Check cooldown period
                if (config.last_scaled and 
                    (datetime.now() - config.last_scaled).total_seconds() < rule.cooldown_seconds):
                    continue
                
                # Get metric value
                metric_value = getattr(metrics, rule.metric, 0)
                
                # Determine scale direction
                direction = ScaleDirection.NONE
                target_instances = config.current_instances
                
                if metric_value > rule.threshold_up:
                    if config.current_instances < rule.max_instances:
                        direction = ScaleDirection.UP
                        target_instances = min(
                            int(config.current_instances * rule.scale_factor),
                            rule.max_instances
                        )
                
                elif metric_value < rule.threshold_down:
                    if config.current_instances > rule.min_instances:
                        direction = ScaleDirection.DOWN
                        target_instances = max(
                            int(config.current_instances / rule.scale_factor),
                            rule.min_instances
                        )
                
                if direction != ScaleDirection.NONE:
                    decisions.append((service_name, direction, target_instances, rule.name))
                    break  # Only one scaling action per service per evaluation
        
        return decisions
    
    async def execute_scaling_action(self, service_name: str, target_instances: int, trigger: str) -> bool:
        """Execute scaling action for a service"""
        config = self.services[service_name]
        old_instances = config.current_instances
        
        success = False
        error_message = None
        
        try:
            # Execute scaling based on orchestrator
            if os.getenv('ORCHESTRATOR', 'docker') == 'kubernetes':
                success = await self.k8s_scaler.scale_deployment(
                    service_name, 'default', target_instances
                )
            else:
                success = await self.docker_scaler.scale_service(service_name, target_instances)
            
            if success:
                config.current_instances = target_instances
                config.last_scaled = datetime.now()
                logger.info(f"Successfully scaled {service_name} from {old_instances} to {target_instances}")
            else:
                error_message = "Scaling command failed"
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"Failed to scale {service_name}: {e}")
        
        # Record scaling event
        event = ScalingEvent(
            timestamp=datetime.now(),
            service=service_name,
            action=f"scale_{target_instances}",
            old_instances=old_instances,
            new_instances=target_instances if success else old_instances,
            trigger_metric=trigger,
            trigger_value=0,  # Would be filled with actual metric value
            success=success,
            error_message=error_message
        )
        
        self.scaling_events.append(event)
        
        return success
    
    async def predict_scaling_needs(self, metrics_window: List[ScalingMetrics]) -> Dict[str, Any]:
        """Predict future scaling needs based on historical data"""
        if len(metrics_window) < 10:
            return {}
        
        predictions = {}
        
        # Simple trend analysis
        for service_name, config in self.services.items():
            for rule in config.rules:
                metric_name = rule.metric
                values = [getattr(m, metric_name, 0) for m in metrics_window[-10:]]
                
                # Calculate trend
                if len(values) >= 3:
                    trend = (values[-1] - values[-3]) / 2  # Simple slope
                    
                    # Predict next value
                    predicted_value = values[-1] + trend
                    
                    # Determine if scaling action might be needed
                    action_needed = None
                    if predicted_value > rule.threshold_up:
                        action_needed = 'scale_up'
                    elif predicted_value < rule.threshold_down:
                        action_needed = 'scale_down'
                    
                    if action_needed:
                        predictions[f"{service_name}_{rule.name}"] = {
                            'service': service_name,
                            'metric': metric_name,
                            'current_value': values[-1],
                            'predicted_value': predicted_value,
                            'action_needed': action_needed,
                            'confidence': min(abs(trend) * 10, 1.0)  # Simple confidence score
                        }
        
        return predictions
    
    async def cleanup_old_data(self):
        """Clean up old metrics and events"""
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        
        # Clean metrics history
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        # Clean scaling events (keep more history)
        event_cutoff = datetime.now() - timedelta(days=7)
        self.scaling_events = [
            e for e in self.scaling_events
            if e.timestamp > event_cutoff
        ]
    
    async def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and recent events"""
        return {
            'timestamp': datetime.now().isoformat(),
            'services': {
                name: {
                    'current_instances': config.current_instances,
                    'min_instances': config.min_instances,
                    'max_instances': config.max_instances,
                    'last_scaled': config.last_scaled.isoformat() if config.last_scaled else None,
                    'rules_count': len(config.rules)
                }
                for name, config in self.services.items()
            },
            'recent_events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'service': event.service,
                    'action': event.action,
                    'old_instances': event.old_instances,
                    'new_instances': event.new_instances,
                    'success': event.success
                }
                for event in self.scaling_events[-10:]
            ]
        }
    
    async def run_scaling_loop(self):
        """Main scaling loop"""
        logger.info("Starting auto-scaling loop")
        
        while True:
            try:
                # Collect current metrics
                metrics = await self.metrics_collector.collect_all_metrics()
                self.metrics_history.append(metrics)
                
                logger.debug(f"Collected metrics: CPU={metrics.cpu_usage}%, "
                           f"Memory={metrics.memory_usage}%, Queue={metrics.queue_depth}")
                
                # Evaluate scaling decisions
                decisions = await self.evaluate_scaling_decision(metrics)
                
                # Execute scaling actions
                for service_name, direction, target_instances, trigger in decisions:
                    logger.info(f"Scaling decision: {service_name} {direction.value} to {target_instances} "
                              f"(trigger: {trigger})")
                    
                    success = await self.execute_scaling_action(service_name, target_instances, trigger)
                    
                    if success:
                        # Wait a bit before next evaluation to allow scaling to take effect
                        await asyncio.sleep(30)
                
                # Predictive scaling (experimental)
                if len(self.metrics_history) >= 10:
                    predictions = await self.predict_scaling_needs(self.metrics_history)
                    if predictions:
                        logger.info(f"Scaling predictions: {len(predictions)} potential actions")
                
                # Cleanup old data periodically
                if len(self.metrics_history) % 100 == 0:
                    await self.cleanup_old_data()
                
                # Wait for next evaluation
                await asyncio.sleep(self.evaluation_interval)
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

# CLI Interface
async def main():
    """Command-line interface for auto-scaler"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YTEmpire Auto-Scaler")
    parser.add_argument("action", choices=["start", "status", "test", "config"])
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--service", "-s", help="Service name for specific operations")
    parser.add_argument("--instances", "-i", type=int, help="Target instances for manual scaling")
    
    args = parser.parse_args()
    
    scaler = AutoScaler(args.config)
    
    if args.action == "start":
        logger.info("Starting auto-scaler daemon")
        await scaler.run_scaling_loop()
        
    elif args.action == "status":
        status = await scaler.get_scaling_status()
        print(json.dumps(status, indent=2))
        
    elif args.action == "test":
        logger.info("Running scaling test")
        metrics = await scaler.metrics_collector.collect_all_metrics()
        decisions = await scaler.evaluate_scaling_decision(metrics)
        
        print(f"Current metrics: {asdict(metrics)}")
        print(f"Scaling decisions: {decisions}")
        
    elif args.action == "config":
        print("Current service configurations:")
        for name, config in scaler.services.items():
            print(f"\n{name}:")
            print(f"  Instances: {config.current_instances} (min: {config.min_instances}, max: {config.max_instances})")
            print(f"  Rules: {len(config.rules)}")
            for rule in config.rules:
                print(f"    - {rule.name}: {rule.metric} up={rule.threshold_up} down={rule.threshold_down}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())