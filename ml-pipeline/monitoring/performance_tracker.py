"""
ML Model Performance Tracker
Monitors model performance, tracks metrics, and manages model health
"""
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import asyncio
import aiofiles
import pickle

# Metrics libraries
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available")

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    model_id: str
    model_type: str
    timestamp: datetime
    latency_ms: float
    accuracy: float
    throughput: int
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float]
    cache_hit_rate: float
    cost_per_inference: float
    total_inferences: int
    failed_inferences: int
    metadata: Dict[str, Any]


@dataclass
class PerformanceAlert:
    """Performance alert data"""
    alert_id: str
    model_id: str
    alert_type: str  # latency, error_rate, accuracy_drop, resource_usage
    severity: str  # low, medium, high, critical
    message: str
    threshold_violated: float
    current_value: float
    timestamp: datetime
    auto_resolved: bool = False


class PerformanceTracker:
    """
    Tracks and monitors ML model performance metrics
    Provides alerting and optimization recommendations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize performance tracker with configuration"""
        self.config = config or {}
        
        # Metrics storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=10000))
        self.current_metrics = {}
        self.alerts = deque(maxlen=1000)
        
        # Performance thresholds
        self.thresholds = {
            'latency_ms': self.config.get('max_latency_ms', 1000),
            'error_rate': self.config.get('max_error_rate', 0.05),
            'min_accuracy': self.config.get('min_accuracy', 0.85),
            'max_memory_mb': self.config.get('max_memory_mb', 4096),
            'max_cpu_percent': self.config.get('max_cpu_percent', 80),
            'max_gpu_percent': self.config.get('max_gpu_percent', 90)
        }
        
        # Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
        
        # Model-specific trackers
        self.model_trackers = {}
        
        # Optimization recommendations
        self.optimization_history = defaultdict(list)
        
        logger.info("Performance tracker initialized")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics collectors"""
        self.prom_inference_counter = Counter(
            'ml_model_inferences_total',
            'Total number of model inferences',
            ['model_id', 'model_type', 'status']
        )
        
        self.prom_latency_histogram = Histogram(
            'ml_model_latency_ms',
            'Model inference latency in milliseconds',
            ['model_id', 'model_type'],
            buckets=(10, 50, 100, 250, 500, 1000, 2500, 5000)
        )
        
        self.prom_accuracy_gauge = Gauge(
            'ml_model_accuracy',
            'Current model accuracy',
            ['model_id', 'model_type']
        )
        
        self.prom_error_rate_gauge = Gauge(
            'ml_model_error_rate',
            'Current model error rate',
            ['model_id', 'model_type']
        )
        
        self.prom_resource_usage = Gauge(
            'ml_model_resource_usage',
            'Model resource usage',
            ['model_id', 'resource_type']
        )
    
    async def track_inference(
        self,
        model_id: str,
        model_type: str,
        latency_ms: float,
        success: bool,
        input_size: Optional[int] = None,
        output_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track a single model inference
        
        Args:
            model_id: Model identifier
            model_type: Type of model (trend, script, voice, etc.)
            latency_ms: Inference latency in milliseconds
            success: Whether inference was successful
            input_size: Size of input data
            output_size: Size of output data
            metadata: Additional metadata
        """
        try:
            # Update Prometheus metrics if available
            if PROMETHEUS_AVAILABLE:
                status = "success" if success else "failure"
                self.prom_inference_counter.labels(
                    model_id=model_id,
                    model_type=model_type,
                    status=status
                ).inc()
                
                if success:
                    self.prom_latency_histogram.labels(
                        model_id=model_id,
                        model_type=model_type
                    ).observe(latency_ms)
            
            # Update internal tracking
            tracker_key = f"{model_id}:{model_type}"
            if tracker_key not in self.model_trackers:
                self.model_trackers[tracker_key] = {
                    'total_inferences': 0,
                    'successful_inferences': 0,
                    'failed_inferences': 0,
                    'total_latency': 0,
                    'latency_samples': deque(maxlen=1000),
                    'error_samples': deque(maxlen=100)
                }
            
            tracker = self.model_trackers[tracker_key]
            tracker['total_inferences'] += 1
            
            if success:
                tracker['successful_inferences'] += 1
                tracker['total_latency'] += latency_ms
                tracker['latency_samples'].append(latency_ms)
            else:
                tracker['failed_inferences'] += 1
                tracker['error_samples'].append({
                    'timestamp': datetime.now(),
                    'metadata': metadata
                })
            
            # Check for performance issues
            await self._check_performance_thresholds(
                model_id, model_type, latency_ms, success
            )
            
        except Exception as e:
            logger.error(f"Error tracking inference: {e}")
    
    async def track_batch_inference(
        self,
        model_id: str,
        model_type: str,
        batch_size: int,
        total_latency_ms: float,
        successful_count: int,
        failed_count: int
    ):
        """
        Track batch inference performance
        
        Args:
            model_id: Model identifier
            model_type: Type of model
            batch_size: Size of batch
            total_latency_ms: Total batch processing time
            successful_count: Number of successful inferences
            failed_count: Number of failed inferences
        """
        avg_latency = total_latency_ms / batch_size if batch_size > 0 else 0
        success_rate = successful_count / batch_size if batch_size > 0 else 0
        
        # Track individual metrics
        for _ in range(successful_count):
            await self.track_inference(
                model_id, model_type, avg_latency, True
            )
        
        for _ in range(failed_count):
            await self.track_inference(
                model_id, model_type, avg_latency, False
            )
        
        # Log batch performance
        logger.info(
            f"Batch inference - Model: {model_id}, "
            f"Batch size: {batch_size}, Success rate: {success_rate:.2%}, "
            f"Avg latency: {avg_latency:.2f}ms"
        )
    
    async def update_model_metrics(
        self,
        model_id: str,
        model_type: str,
        metrics: Dict[str, Any]
    ):
        """
        Update comprehensive model metrics
        
        Args:
            model_id: Model identifier
            model_type: Type of model
            metrics: Dictionary of metrics to update
        """
        try:
            # Create metrics object
            model_metrics = ModelMetrics(
                model_id=model_id,
                model_type=model_type,
                timestamp=datetime.now(),
                latency_ms=metrics.get('latency_ms', 0),
                accuracy=metrics.get('accuracy', 0),
                throughput=metrics.get('throughput', 0),
                error_rate=metrics.get('error_rate', 0),
                memory_usage_mb=metrics.get('memory_usage_mb', 0),
                cpu_usage_percent=metrics.get('cpu_usage_percent', 0),
                gpu_usage_percent=metrics.get('gpu_usage_percent'),
                cache_hit_rate=metrics.get('cache_hit_rate', 0),
                cost_per_inference=metrics.get('cost_per_inference', 0),
                total_inferences=metrics.get('total_inferences', 0),
                failed_inferences=metrics.get('failed_inferences', 0),
                metadata=metrics.get('metadata', {})
            )
            
            # Store metrics
            self.current_metrics[model_id] = model_metrics
            self.metrics_history[model_id].append(model_metrics)
            
            # Update Prometheus gauges if available
            if PROMETHEUS_AVAILABLE:
                self.prom_accuracy_gauge.labels(
                    model_id=model_id,
                    model_type=model_type
                ).set(model_metrics.accuracy)
                
                self.prom_error_rate_gauge.labels(
                    model_id=model_id,
                    model_type=model_type
                ).set(model_metrics.error_rate)
                
                self.prom_resource_usage.labels(
                    model_id=model_id,
                    resource_type="memory_mb"
                ).set(model_metrics.memory_usage_mb)
                
                self.prom_resource_usage.labels(
                    model_id=model_id,
                    resource_type="cpu_percent"
                ).set(model_metrics.cpu_usage_percent)
                
                if model_metrics.gpu_usage_percent is not None:
                    self.prom_resource_usage.labels(
                        model_id=model_id,
                        resource_type="gpu_percent"
                    ).set(model_metrics.gpu_usage_percent)
            
            # Check for anomalies
            await self._detect_anomalies(model_id, model_metrics)
            
        except Exception as e:
            logger.error(f"Error updating model metrics: {e}")
    
    async def _check_performance_thresholds(
        self,
        model_id: str,
        model_type: str,
        latency_ms: float,
        success: bool
    ):
        """Check if performance thresholds are violated"""
        
        # Check latency threshold
        if latency_ms > self.thresholds['latency_ms']:
            await self._create_alert(
                model_id=model_id,
                alert_type="latency",
                severity="high" if latency_ms > self.thresholds['latency_ms'] * 2 else "medium",
                message=f"High latency detected: {latency_ms:.2f}ms",
                threshold_violated=self.thresholds['latency_ms'],
                current_value=latency_ms
            )
        
        # Calculate error rate
        tracker_key = f"{model_id}:{model_type}"
        if tracker_key in self.model_trackers:
            tracker = self.model_trackers[tracker_key]
            if tracker['total_inferences'] > 100:  # Need sufficient samples
                error_rate = tracker['failed_inferences'] / tracker['total_inferences']
                
                if error_rate > self.thresholds['error_rate']:
                    await self._create_alert(
                        model_id=model_id,
                        alert_type="error_rate",
                        severity="critical" if error_rate > 0.1 else "high",
                        message=f"High error rate: {error_rate:.2%}",
                        threshold_violated=self.thresholds['error_rate'],
                        current_value=error_rate
                    )
    
    async def _detect_anomalies(self, model_id: str, metrics: ModelMetrics):
        """Detect anomalies in model performance"""
        
        # Get historical metrics
        history = list(self.metrics_history[model_id])
        if len(history) < 10:
            return  # Not enough history for anomaly detection
        
        # Calculate statistics from recent history
        recent_metrics = history[-100:]  # Last 100 samples
        latencies = [m.latency_ms for m in recent_metrics]
        accuracies = [m.accuracy for m in recent_metrics if m.accuracy > 0]
        
        if latencies:
            mean_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            
            # Check for latency anomaly (3 sigma rule)
            if metrics.latency_ms > mean_latency + 3 * std_latency:
                await self._create_alert(
                    model_id=model_id,
                    alert_type="latency_anomaly",
                    severity="medium",
                    message=f"Latency anomaly detected: {metrics.latency_ms:.2f}ms (mean: {mean_latency:.2f}ms)",
                    threshold_violated=mean_latency + 3 * std_latency,
                    current_value=metrics.latency_ms
                )
        
        if accuracies and metrics.accuracy > 0:
            mean_accuracy = np.mean(accuracies)
            
            # Check for accuracy drop
            if metrics.accuracy < mean_accuracy * 0.9:  # 10% drop
                await self._create_alert(
                    model_id=model_id,
                    alert_type="accuracy_drop",
                    severity="high",
                    message=f"Accuracy drop detected: {metrics.accuracy:.2%} (mean: {mean_accuracy:.2%})",
                    threshold_violated=mean_accuracy * 0.9,
                    current_value=metrics.accuracy
                )
    
    async def _create_alert(
        self,
        model_id: str,
        alert_type: str,
        severity: str,
        message: str,
        threshold_violated: float,
        current_value: float
    ):
        """Create and store performance alert"""
        
        alert = PerformanceAlert(
            alert_id=f"{model_id}:{alert_type}:{int(time.time())}",
            model_id=model_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            threshold_violated=threshold_violated,
            current_value=current_value,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        
        # Log alert
        log_level = logging.CRITICAL if severity == "critical" else logging.WARNING
        logger.log(log_level, f"Performance Alert - {message}")
        
        # Trigger any configured alert handlers
        # This could send notifications, trigger auto-scaling, etc.
    
    async def get_model_performance_summary(
        self,
        model_id: str,
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Get performance summary for a model
        
        Args:
            model_id: Model identifier
            time_window_minutes: Time window for statistics
            
        Returns:
            Performance summary dictionary
        """
        # Get recent metrics
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_metrics = [
            m for m in self.metrics_history[model_id]
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {
                'model_id': model_id,
                'status': 'no_data',
                'message': 'No recent metrics available'
            }
        
        # Calculate statistics
        latencies = [m.latency_ms for m in recent_metrics]
        accuracies = [m.accuracy for m in recent_metrics if m.accuracy > 0]
        error_rates = [m.error_rate for m in recent_metrics]
        
        summary = {
            'model_id': model_id,
            'time_window_minutes': time_window_minutes,
            'total_inferences': len(recent_metrics),
            'latency': {
                'mean': np.mean(latencies),
                'median': np.median(latencies),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'min': np.min(latencies),
                'max': np.max(latencies)
            },
            'accuracy': {
                'mean': np.mean(accuracies) if accuracies else 0,
                'min': np.min(accuracies) if accuracies else 0,
                'max': np.max(accuracies) if accuracies else 0
            },
            'error_rate': {
                'mean': np.mean(error_rates),
                'max': np.max(error_rates)
            },
            'recent_alerts': [
                asdict(a) for a in self.alerts
                if a.model_id == model_id and a.timestamp > cutoff_time
            ]
        }
        
        # Add optimization recommendations
        summary['recommendations'] = await self._generate_recommendations(
            model_id, recent_metrics
        )
        
        return summary
    
    async def _generate_recommendations(
        self,
        model_id: str,
        metrics: List[ModelMetrics]
    ) -> List[str]:
        """Generate optimization recommendations based on metrics"""
        
        recommendations = []
        
        if not metrics:
            return recommendations
        
        # Analyze latency
        latencies = [m.latency_ms for m in metrics]
        mean_latency = np.mean(latencies)
        
        if mean_latency > self.thresholds['latency_ms']:
            recommendations.append(
                f"Consider model optimization: average latency {mean_latency:.2f}ms exceeds threshold"
            )
            recommendations.append("Enable batch processing to improve throughput")
        
        # Analyze error rate
        error_rates = [m.error_rate for m in metrics]
        mean_error_rate = np.mean(error_rates)
        
        if mean_error_rate > self.thresholds['error_rate']:
            recommendations.append(
                f"High error rate ({mean_error_rate:.2%}) - consider retraining or input validation"
            )
        
        # Analyze resource usage
        memory_usage = [m.memory_usage_mb for m in metrics if m.memory_usage_mb > 0]
        if memory_usage:
            mean_memory = np.mean(memory_usage)
            if mean_memory > self.thresholds['max_memory_mb'] * 0.8:
                recommendations.append(
                    f"High memory usage ({mean_memory:.0f}MB) - consider model quantization or pruning"
                )
        
        # Check cache effectiveness
        cache_rates = [m.cache_hit_rate for m in metrics if m.cache_hit_rate >= 0]
        if cache_rates:
            mean_cache_rate = np.mean(cache_rates)
            if mean_cache_rate < 0.5:
                recommendations.append(
                    f"Low cache hit rate ({mean_cache_rate:.2%}) - review caching strategy"
                )
        
        return recommendations
    
    async def export_metrics(
        self,
        model_id: Optional[str] = None,
        format: str = "json",
        filepath: Optional[str] = None
    ) -> Optional[str]:
        """
        Export metrics data
        
        Args:
            model_id: Specific model or all if None
            format: Export format (json, csv, pickle)
            filepath: Output file path
            
        Returns:
            Exported data as string if no filepath provided
        """
        try:
            # Gather metrics
            if model_id:
                metrics_data = {
                    model_id: [asdict(m) for m in self.metrics_history[model_id]]
                }
            else:
                metrics_data = {
                    mid: [asdict(m) for m in history]
                    for mid, history in self.metrics_history.items()
                }
            
            # Format data
            if format == "json":
                # Convert datetime objects to strings
                for model_metrics in metrics_data.values():
                    for metric in model_metrics:
                        metric['timestamp'] = metric['timestamp'].isoformat()
                
                output = json.dumps(metrics_data, indent=2)
                
            elif format == "csv":
                # Flatten to DataFrame
                rows = []
                for model_id, metrics_list in metrics_data.items():
                    for metric in metrics_list:
                        metric['model_id'] = model_id
                        rows.append(metric)
                
                df = pd.DataFrame(rows)
                output = df.to_csv(index=False)
                
            elif format == "pickle":
                output = pickle.dumps(metrics_data)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Save or return
            if filepath:
                async with aiofiles.open(filepath, 'wb' if format == "pickle" else 'w') as f:
                    await f.write(output)
                logger.info(f"Metrics exported to {filepath}")
                return None
            else:
                return output
                
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return None
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[PerformanceAlert]:
        """Get active (non-resolved) alerts"""
        alerts = [a for a in self.alerts if not a.auto_resolved]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts


# Global instance for easy access
performance_tracker = PerformanceTracker()


# Convenience functions
async def track_model_inference(
    model_id: str,
    model_type: str,
    latency_ms: float,
    success: bool,
    **kwargs
):
    """Track a model inference"""
    await performance_tracker.track_inference(
        model_id, model_type, latency_ms, success, **kwargs
    )


async def get_model_performance(model_id: str) -> Dict[str, Any]:
    """Get model performance summary"""
    return await performance_tracker.get_model_performance_summary(model_id)