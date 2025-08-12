"""
Analytics Data Pipeline Service
Real-time analytics processing and aggregation for YTEmpire
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from enum import Enum
import redis.asyncio as redis
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, text
import httpx
import hashlib
import uuid
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import io
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics to track"""
    VIEWS = "views"
    WATCH_TIME = "watch_time"
    LIKES = "likes"
    COMMENTS = "comments"
    SHARES = "shares"
    SUBSCRIBERS = "subscribers"
    REVENUE = "revenue"
    CTR = "ctr"
    AVG_VIEW_DURATION = "avg_view_duration"
    IMPRESSIONS = "impressions"
    ENGAGEMENT_RATE = "engagement_rate"
    COST = "cost"
    ROI = "roi"
    CONVERSION_RATE = "conversion_rate"


class AggregationLevel(str, Enum):
    """Data aggregation levels"""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class DataSource(str, Enum):
    """Analytics data sources"""
    YOUTUBE = "youtube"
    YOUTUBE_ANALYTICS = "youtube_analytics"
    INTERNAL = "internal"
    STRIPE = "stripe"
    OPENAI = "openai"
    CUSTOM = "custom"


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    metric_type: MetricType
    value: float
    source: DataSource
    channel_id: Optional[str] = None
    video_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedMetric:
    """Aggregated metric data"""
    period_start: datetime
    period_end: datetime
    aggregation_level: AggregationLevel
    metric_type: MetricType
    count: int
    sum: float
    avg: float
    min: float
    max: float
    std_dev: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_95: float
    channel_id: Optional[str] = None
    video_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsReport:
    """Analytics report structure"""
    report_id: str
    report_type: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    metrics: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    charts: Dict[str, Any]
    gdpr_compliant: bool = True
    data_retention_days: int = 365
    anonymized: bool = False


@dataclass
class PredictiveModel:
    """Predictive analytics model"""
    model_id: str
    model_type: str  # "linear_regression", "anomaly_detection", "trend_prediction"
    trained_at: datetime
    accuracy: float
    features: List[str]
    target: str
    model_data: bytes  # Serialized model
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GDPRDataRequest:
    """GDPR compliance data request"""
    request_id: str
    user_id: str
    request_type: str  # "access", "delete", "rectify", "portability"
    requested_at: datetime
    processed_at: Optional[datetime] = None
    status: str = "pending"  # "pending", "processing", "completed", "rejected"
    data_categories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    

class AdvancedAnalyticsPipeline:
    """
    Advanced real-time analytics pipeline with GDPR compliance and predictive models
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        batch_size: int = 1000,
        flush_interval: int = 30,
        enable_real_time_streaming: bool = True,
        enable_predictive_models: bool = True,
        gdpr_compliance_enabled: bool = True
    ):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.enable_real_time_streaming = enable_real_time_streaming
        self.enable_predictive_models = enable_predictive_models
        self.gdpr_compliance_enabled = gdpr_compliance_enabled
        
        # In-memory buffers
        self.metric_buffer: deque = deque(maxlen=50000)  # Increased for real-time streaming
        self.streaming_buffer: deque = deque(maxlen=10000)  # Real-time streaming buffer
        self.aggregation_buffer: Dict[str, List[MetricPoint]] = defaultdict(list)
        
        # Processing state
        self.processing = False
        self.streaming_active = False
        self.last_flush = datetime.utcnow()
        
        # Predictive models
        self.predictive_models: Dict[str, PredictiveModel] = {}
        self.model_training_queue = deque()
        
        # GDPR compliance
        self.encryption_key = self._generate_encryption_key() if gdpr_compliance_enabled else None
        self.gdpr_requests: Dict[str, GDPRDataRequest] = {}
        self.data_retention_policies = {
            "raw_metrics": 90,  # days
            "aggregated_metrics": 365,
            "reports": 1095,  # 3 years
            "user_data": 365
        }
        
        # Real-time streaming
        self.streaming_subscribers: Dict[str, set] = defaultdict(set)
        self.stream_processors = {
            "anomaly_detection": self._detect_anomalies,
            "trend_analysis": self._analyze_trends,
            "performance_monitoring": self._monitor_performance
        }
        
        # Aggregation windows
        self.aggregation_windows = {
            AggregationLevel.MINUTE: timedelta(minutes=1),
            AggregationLevel.HOUR: timedelta(hours=1),
            AggregationLevel.DAY: timedelta(days=1),
            AggregationLevel.WEEK: timedelta(weeks=1),
            AggregationLevel.MONTH: timedelta(days=30),
        }
        
        # Enhanced alert thresholds
        self.alert_thresholds = {
            "low_engagement": 0.02,  # 2% engagement rate
            "high_cost": 100.0,  # $100 daily cost
            "low_roi": 0.5,  # 50% ROI
            "view_drop": 0.3,  # 30% view drop
            "anomaly_score": 0.8,  # Anomaly detection threshold
            "trend_change": 0.2,  # 20% trend change
            "performance_degradation": 0.4  # 40% performance drop
        }
        
    async def initialize(self):
        """Initialize the advanced analytics pipeline"""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            
            # Start background tasks
            asyncio.create_task(self.process_metrics())
            asyncio.create_task(self.aggregate_metrics())
            asyncio.create_task(self.generate_reports())
            asyncio.create_task(self.monitor_alerts())
            
            # Start advanced features
            if self.enable_real_time_streaming:
                asyncio.create_task(self.real_time_streaming_processor())
                asyncio.create_task(self.stream_analytics())
            
            if self.enable_predictive_models:
                asyncio.create_task(self.predictive_model_trainer())
                asyncio.create_task(self.run_predictions())
            
            if self.gdpr_compliance_enabled:
                asyncio.create_task(self.gdpr_compliance_processor())
                asyncio.create_task(self.data_retention_cleanup())
            
            # Load existing predictive models
            await self.load_predictive_models()
            
            logger.info("Advanced analytics pipeline initialized with all features")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced analytics pipeline: {e}")
            raise
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for GDPR compliance"""
        return Fernet.generate_key()
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data for GDPR compliance"""
        if not self.gdpr_compliance_enabled or not self.encryption_key:
            return data
        
        f = Fernet(self.encryption_key)
        encrypted = f.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not self.gdpr_compliance_enabled or not self.encryption_key:
            return encrypted_data
        
        try:
            f = Fernet(self.encryption_key)
            decoded = base64.b64decode(encrypted_data.encode())
            decrypted = f.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            return encrypted_data
    
    async def real_time_streaming_processor(self):
        """Process real-time streaming analytics"""
        self.streaming_active = True
        logger.info("Real-time streaming processor started")
        
        while self.streaming_active:
            try:
                await asyncio.sleep(1)  # Process every second
                
                if not self.streaming_buffer:
                    continue
                
                # Process streaming metrics
                batch = []
                while self.streaming_buffer and len(batch) < 100:
                    batch.append(self.streaming_buffer.popleft())
                
                if batch:
                    # Run real-time processors
                    for processor_name, processor_func in self.stream_processors.items():
                        try:
                            results = await processor_func(batch)
                            
                            # Publish results to subscribers
                            if results:
                                await self._publish_streaming_results(processor_name, results)
                                
                        except Exception as e:
                            logger.error(f"Stream processor {processor_name} failed: {e}")
                
            except Exception as e:
                logger.error(f"Real-time streaming error: {e}")
    
    async def stream_analytics(self):
        """Continuous stream analytics processing"""
        while self.streaming_active:
            try:
                await asyncio.sleep(5)  # Process every 5 seconds
                
                # Get recent metrics for stream analysis
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(minutes=5)
                
                # Analyze streaming data for patterns
                await self._analyze_streaming_patterns(start_time, end_time)
                
            except Exception as e:
                logger.error(f"Stream analytics error: {e}")
    
    async def _detect_anomalies(self, metrics: List[MetricPoint]) -> Dict[str, Any]:
        """Detect anomalies in real-time metrics"""
        if not metrics or len(metrics) < 10:
            return {}
        
        try:
            # Extract values for anomaly detection
            values = np.array([m.value for m in metrics]).reshape(-1, 1)
            
            # Use Isolation Forest for anomaly detection
            if len(values) >= 10:
                detector = IsolationForest(contamination=0.1, random_state=42)
                anomaly_scores = detector.fit_predict(values)
                
                anomalies = []
                for i, score in enumerate(anomaly_scores):
                    if score == -1:  # Anomaly detected
                        anomalies.append({
                            "timestamp": metrics[i].timestamp.isoformat(),
                            "metric_type": metrics[i].metric_type.value,
                            "value": metrics[i].value,
                            "expected_range": self._get_expected_range(metrics[i].metric_type),
                            "anomaly_score": float(detector.score_samples([values[i]])[0])
                        })
                
                if anomalies:
                    return {
                        "type": "anomalies_detected",
                        "count": len(anomalies),
                        "anomalies": anomalies,
                        "detected_at": datetime.utcnow().isoformat()
                    }
        
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
        
        return {}
    
    async def _analyze_trends(self, metrics: List[MetricPoint]) -> Dict[str, Any]:
        """Analyze trends in real-time metrics"""
        if not metrics or len(metrics) < 5:
            return {}
        
        try:
            # Group by metric type
            by_type = defaultdict(list)
            for m in metrics:
                by_type[m.metric_type.value].append(m.value)
            
            trends = {}
            for metric_type, values in by_type.items():
                if len(values) >= 3:
                    # Simple trend analysis
                    trend_direction = self._calculate_trend_direction(values)
                    trend_strength = self._calculate_trend_strength(values)
                    
                    trends[metric_type] = {
                        "direction": trend_direction,
                        "strength": trend_strength,
                        "current_value": values[-1],
                        "change_percent": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                    }
            
            if trends:
                return {
                    "type": "trend_analysis",
                    "trends": trends,
                    "analyzed_at": datetime.utcnow().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
        
        return {}
    
    async def _monitor_performance(self, metrics: List[MetricPoint]) -> Dict[str, Any]:
        """Monitor performance metrics in real-time"""
        if not metrics:
            return {}
        
        try:
            performance_metrics = {}
            
            # Calculate key performance indicators
            by_type = defaultdict(list)
            for m in metrics:
                by_type[m.metric_type.value].append(m.value)
            
            # Check for performance issues
            alerts = []
            for metric_type, values in by_type.items():
                if values:
                    avg_value = np.mean(values)
                    recent_value = values[-1]
                    
                    # Check for significant drops
                    if avg_value > 0 and recent_value < avg_value * (1 - self.alert_thresholds["performance_degradation"]):
                        alerts.append({
                            "type": "performance_degradation",
                            "metric": metric_type,
                            "current": recent_value,
                            "average": avg_value,
                            "drop_percent": ((avg_value - recent_value) / avg_value) * 100
                        })
                    
                    performance_metrics[metric_type] = {
                        "current": recent_value,
                        "average": avg_value,
                        "min": min(values),
                        "max": max(values)
                    }
            
            return {
                "type": "performance_monitoring",
                "metrics": performance_metrics,
                "alerts": alerts,
                "monitored_at": datetime.utcnow().isoformat()
            } if performance_metrics else {}
        
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
        
        return {}
    
    async def predictive_model_trainer(self):
        """Background task to train predictive models"""
        while True:
            try:
                await asyncio.sleep(3600)  # Train models every hour
                
                # Train trend prediction model
                await self._train_trend_prediction_model()
                
                # Train anomaly detection model
                await self._train_anomaly_detection_model()
                
                # Train engagement prediction model
                await self._train_engagement_prediction_model()
                
            except Exception as e:
                logger.error(f"Model training error: {e}")
    
    async def _train_trend_prediction_model(self):
        """Train trend prediction model"""
        try:
            # Get historical data for training
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)
            
            # Collect training data
            training_data = []
            for metric_type in [MetricType.VIEWS, MetricType.ENGAGEMENT_RATE, MetricType.REVENUE]:
                metrics = await self.query_metrics(
                    metric_type=metric_type,
                    start_time=start_time,
                    end_time=end_time,
                    aggregation_level=AggregationLevel.HOUR
                )
                
                if len(metrics) >= 24:  # Need at least 24 hours of data
                    values = [m.get("avg", m.get("value", 0)) for m in metrics[-168:]]  # Last 7 days
                    
                    # Create features (sliding windows)
                    for i in range(24, len(values)):
                        features = values[i-24:i]  # 24-hour lookback
                        target = values[i]  # Next hour prediction
                        
                        training_data.append({
                            "features": features,
                            "target": target,
                            "metric_type": metric_type.value
                        })
            
            if len(training_data) >= 50:  # Minimum training data
                # Train separate models for each metric type
                for metric_type in [MetricType.VIEWS, MetricType.ENGAGEMENT_RATE, MetricType.REVENUE]:
                    type_data = [d for d in training_data if d["metric_type"] == metric_type.value]
                    
                    if len(type_data) >= 20:
                        X = np.array([d["features"] for d in type_data])
                        y = np.array([d["target"] for d in type_data])
                        
                        # Train model
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        # Calculate accuracy (R²)
                        accuracy = model.score(X, y)
                        
                        # Serialize model
                        model_data = io.BytesIO()
                        joblib.dump(model, model_data)
                        
                        # Store model
                        predictive_model = PredictiveModel(
                            model_id=f"trend_prediction_{metric_type.value}",
                            model_type="linear_regression",
                            trained_at=datetime.utcnow(),
                            accuracy=accuracy,
                            features=[f"hour_{i}" for i in range(24)],
                            target=metric_type.value,
                            model_data=model_data.getvalue(),
                            metadata={
                                "training_samples": len(type_data),
                                "lookback_hours": 24
                            }
                        )
                        
                        self.predictive_models[predictive_model.model_id] = predictive_model
                        await self._store_predictive_model(predictive_model)
                        
                        logger.info(f"Trained trend prediction model for {metric_type.value} with accuracy {accuracy:.3f}")
        
        except Exception as e:
            logger.error(f"Trend prediction model training failed: {e}")
    
    async def run_predictions(self):
        """Run predictions using trained models"""
        while True:
            try:
                await asyncio.sleep(900)  # Run predictions every 15 minutes
                
                for model_id, model in self.predictive_models.items():
                    if model.model_type == "linear_regression":
                        prediction = await self._make_trend_prediction(model)
                        if prediction:
                            await self._store_prediction(model_id, prediction)
                            await self._publish_streaming_results("predictions", prediction)
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
    
    async def gdpr_compliance_processor(self):
        """Process GDPR compliance requests"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Process pending GDPR requests
                for request_id, request in list(self.gdpr_requests.items()):
                    if request.status == "pending":
                        await self._process_gdpr_request(request)
                
            except Exception as e:
                logger.error(f"GDPR compliance error: {e}")
    
    async def data_retention_cleanup(self):
        """Clean up data based on retention policies"""
        while True:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                now = datetime.utcnow()
                
                # Clean up expired data
                for data_type, retention_days in self.data_retention_policies.items():
                    cutoff_date = now - timedelta(days=retention_days)
                    await self._cleanup_expired_data(data_type, cutoff_date)
                
            except Exception as e:
                logger.error(f"Data retention cleanup error: {e}")
    
    # GDPR Compliance Methods
    
    async def handle_gdpr_request(self, user_id: str, request_type: str, data_categories: List[str] = None) -> str:
        """Handle GDPR data request"""
        request_id = str(uuid.uuid4())
        
        gdpr_request = GDPRDataRequest(
            request_id=request_id,
            user_id=user_id,
            request_type=request_type,
            requested_at=datetime.utcnow(),
            data_categories=data_categories or ["all"]
        )
        
        self.gdpr_requests[request_id] = gdpr_request
        
        # Store request for persistence
        await self.redis_client.setex(
            f"gdpr:request:{request_id}",
            86400 * 30,  # 30 days
            json.dumps(asdict(gdpr_request), default=str)
        )
        
        logger.info(f"GDPR {request_type} request {request_id} created for user {user_id}")
        return request_id
    
    async def _process_gdpr_request(self, request: GDPRDataRequest):
        """Process a GDPR request"""
        try:
            request.status = "processing"
            
            if request.request_type == "access":
                data = await self._extract_user_data(request.user_id, request.data_categories)
                # Store extracted data for user access
                await self.redis_client.setex(
                    f"gdpr:data:{request.request_id}",
                    86400 * 7,  # 7 days
                    json.dumps(data, default=str)
                )
            
            elif request.request_type == "delete":
                await self._delete_user_data(request.user_id, request.data_categories)
            
            elif request.request_type == "portability":
                data = await self._export_user_data(request.user_id, request.data_categories)
                # Store exported data
                await self.redis_client.setex(
                    f"gdpr:export:{request.request_id}",
                    86400 * 30,  # 30 days
                    json.dumps(data, default=str)
                )
            
            request.status = "completed"
            request.processed_at = datetime.utcnow()
            
        except Exception as e:
            request.status = "rejected"
            request.metadata["error"] = str(e)
            logger.error(f"GDPR request {request.request_id} failed: {e}")
    
    # Enhanced helper methods
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction with improved accuracy"""
        if len(values) < 3:
            return "stable"
        
        # Use linear regression for trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            
            # Calculate relative slope
            avg_value = np.mean(values)
            relative_slope = slope / avg_value if avg_value != 0 else 0
            
            if relative_slope > 0.05:  # 5% increase
                return "increasing"
            elif relative_slope < -0.05:  # 5% decrease
                return "decreasing"
            else:
                return "stable"
        except Exception:
            return "stable"
    
    def _calculate_trend_strength(self, values: List[float]) -> float:
        """Calculate trend strength (0-1)"""
        if len(values) < 3:
            return 0.0
        
        try:
            x = np.arange(len(values))
            y = np.array(values)
            
            # Calculate R-squared
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            return max(0.0, min(1.0, r_squared))
            
        except Exception:
            return 0.0
    
    async def _publish_streaming_results(self, stream_type: str, results: Dict[str, Any]):
        """Publish streaming results to subscribers"""
        message = {
            "stream_type": stream_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": results
        }
        
        # Publish to Redis streams
        await self.redis_client.xadd(
            f"stream:{stream_type}",
            message,
            maxlen=1000  # Keep last 1000 messages
        )
        
        # Notify WebSocket subscribers
        for subscriber in self.streaming_subscribers.get(stream_type, set()):
            try:
                # This would integrate with WebSocket manager
                pass
            except Exception as e:
                logger.error(f"Failed to notify subscriber {subscriber}: {e}")
    
    def _get_expected_range(self, metric_type: MetricType) -> Tuple[float, float]:
        """Get expected range for metric type"""
        ranges = {
            MetricType.VIEWS: (0, 10000),
            MetricType.ENGAGEMENT_RATE: (0, 1),
            MetricType.CTR: (0, 1),
            MetricType.REVENUE: (0, 1000),
            MetricType.COST: (0, 100)
        }
        return ranges.get(metric_type, (0, 1000))


# Maintain backward compatibility
class AnalyticsPipeline(AdvancedAnalyticsPipeline):
    """Backward compatible AnalyticsPipeline"""
    
    def __init__(self, *args, **kwargs):
        # Set advanced features to False for backward compatibility
        kwargs.setdefault('enable_real_time_streaming', False)
        kwargs.setdefault('enable_predictive_models', False)
        kwargs.setdefault('gdpr_compliance_enabled', False)
        super().__init__(*args, **kwargs)
            
    async def ingest_metric(self, metric: MetricPoint):
        """
        Ingest a single metric into the pipeline
        """
        try:
            # Add to buffer
            self.metric_buffer.append(metric)
            
            # Store in Redis for real-time access
            key = f"metric:{metric.metric_type.value}:{metric.timestamp.timestamp()}"
            await self.redis_client.setex(
                key,
                3600,  # 1 hour TTL
                json.dumps(asdict(metric), default=str)
            )
            
            # Check if batch processing needed
            if len(self.metric_buffer) >= self.batch_size:
                await self.flush_metrics()
                
        except Exception as e:
            logger.error(f"Failed to ingest metric: {e}")
            
    async def ingest_batch(self, metrics: List[MetricPoint]):
        """
        Ingest a batch of metrics
        """
        for metric in metrics:
            await self.ingest_metric(metric)
            
    async def process_metrics(self):
        """
        Background task to process buffered metrics
        """
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                
                if self.metric_buffer:
                    await self.flush_metrics()
                    
            except Exception as e:
                logger.error(f"Metric processing error: {e}")
                
    async def flush_metrics(self):
        """
        Flush buffered metrics to storage and aggregation
        """
        if not self.metric_buffer:
            return
            
        try:
            # Copy buffer
            metrics = list(self.metric_buffer)
            self.metric_buffer.clear()
            
            # Group by type and source
            grouped = defaultdict(list)
            for metric in metrics:
                key = f"{metric.metric_type.value}:{metric.source.value}"
                grouped[key].append(metric)
            
            # Store grouped metrics
            for key, group_metrics in grouped.items():
                await self.store_metrics(key, group_metrics)
                
                # Add to aggregation buffer
                self.aggregation_buffer[key].extend(group_metrics)
            
            self.last_flush = datetime.utcnow()
            logger.info(f"Flushed {len(metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
            
    async def store_metrics(self, key: str, metrics: List[MetricPoint]):
        """
        Store metrics in time-series format
        """
        # Convert to DataFrame for efficient storage
        df_data = []
        for metric in metrics:
            df_data.append({
                "timestamp": metric.timestamp,
                "value": metric.value,
                "channel_id": metric.channel_id,
                "video_id": metric.video_id,
                "user_id": metric.user_id,
                **metric.metadata
            })
        
        df = pd.DataFrame(df_data)
        
        # Store in Redis as compressed JSON
        redis_key = f"timeseries:{key}:{datetime.utcnow().strftime('%Y%m%d%H')}"
        compressed_data = df.to_json(orient='records', date_format='iso')
        
        await self.redis_client.setex(
            redis_key,
            86400 * 7,  # 7 days retention
            compressed_data
        )
        
    async def aggregate_metrics(self):
        """
        Background task to aggregate metrics at different levels
        """
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                for key, metrics in self.aggregation_buffer.items():
                    if metrics:
                        for level in AggregationLevel:
                            await self.aggregate_at_level(key, metrics, level)
                
                # Clear processed metrics
                self.aggregation_buffer.clear()
                
            except Exception as e:
                logger.error(f"Aggregation error: {e}")
                
    async def aggregate_at_level(
        self,
        key: str,
        metrics: List[MetricPoint],
        level: AggregationLevel
    ):
        """
        Aggregate metrics at a specific level
        """
        if not metrics:
            return
            
        window = self.aggregation_windows.get(level)
        if not window:
            return
        
        # Group metrics by window
        windowed_groups = defaultdict(list)
        for metric in metrics:
            window_start = self.get_window_start(metric.timestamp, level)
            windowed_groups[window_start].append(metric.value)
        
        # Calculate aggregations
        for window_start, values in windowed_groups.items():
            if not values:
                continue
                
            values_array = np.array(values)
            
            aggregated = AggregatedMetric(
                period_start=window_start,
                period_end=window_start + window,
                aggregation_level=level,
                metric_type=metrics[0].metric_type,
                count=len(values),
                sum=float(np.sum(values_array)),
                avg=float(np.mean(values_array)),
                min=float(np.min(values_array)),
                max=float(np.max(values_array)),
                std_dev=float(np.std(values_array)),
                percentile_25=float(np.percentile(values_array, 25)),
                percentile_50=float(np.percentile(values_array, 50)),
                percentile_75=float(np.percentile(values_array, 75)),
                percentile_95=float(np.percentile(values_array, 95)),
                channel_id=metrics[0].channel_id,
                video_id=metrics[0].video_id
            )
            
            # Store aggregated metric
            await self.store_aggregated_metric(aggregated)
            
    async def store_aggregated_metric(self, aggregated: AggregatedMetric):
        """
        Store aggregated metric
        """
        key = f"aggregated:{aggregated.metric_type.value}:{aggregated.aggregation_level.value}:{aggregated.period_start.timestamp()}"
        
        await self.redis_client.setex(
            key,
            86400 * 30,  # 30 days retention
            json.dumps(asdict(aggregated), default=str)
        )
        
    async def query_metrics(
        self,
        metric_type: MetricType,
        start_time: datetime,
        end_time: datetime,
        aggregation_level: Optional[AggregationLevel] = None,
        channel_id: Optional[str] = None,
        video_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query metrics for a time range
        """
        results = []
        
        if aggregation_level:
            # Query aggregated metrics
            pattern = f"aggregated:{metric_type.value}:{aggregation_level.value}:*"
        else:
            # Query raw metrics
            pattern = f"timeseries:{metric_type.value}:*"
        
        # Scan Redis for matching keys
        cursor = 0
        while True:
            cursor, keys = await self.redis_client.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            
            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    metrics = json.loads(data)
                    
                    # Filter by time range and IDs
                    for metric in metrics if isinstance(metrics, list) else [metrics]:
                        metric_time = datetime.fromisoformat(metric.get('timestamp') or metric.get('period_start'))
                        
                        if start_time <= metric_time <= end_time:
                            if channel_id and metric.get('channel_id') != channel_id:
                                continue
                            if video_id and metric.get('video_id') != video_id:
                                continue
                            
                            results.append(metric)
            
            if cursor == 0:
                break
        
        return results
        
    async def get_channel_analytics(
        self,
        channel_id: str,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """
        Get comprehensive analytics for a channel
        """
        start_time = datetime.combine(start_date, datetime.min.time())
        end_time = datetime.combine(end_date, datetime.max.time())
        
        analytics = {
            "channel_id": channel_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "metrics": {},
            "trends": {},
            "top_videos": [],
            "engagement": {}
        }
        
        # Fetch all metric types
        for metric_type in MetricType:
            metrics = await self.query_metrics(
                metric_type=metric_type,
                start_time=start_time,
                end_time=end_time,
                channel_id=channel_id,
                aggregation_level=AggregationLevel.DAY
            )
            
            if metrics:
                # Calculate summary stats
                values = [m.get('avg', m.get('value', 0)) for m in metrics]
                analytics["metrics"][metric_type.value] = {
                    "total": sum(values),
                    "average": np.mean(values) if values else 0,
                    "trend": self.calculate_trend(values)
                }
        
        # Calculate engagement metrics
        if analytics["metrics"].get("views") and analytics["metrics"].get("likes"):
            views = analytics["metrics"]["views"]["total"]
            likes = analytics["metrics"]["likes"]["total"]
            comments = analytics["metrics"].get("comments", {}).get("total", 0)
            
            analytics["engagement"] = {
                "rate": (likes + comments) / views if views > 0 else 0,
                "likes_per_view": likes / views if views > 0 else 0,
                "comments_per_view": comments / views if views > 0 else 0
            }
        
        return analytics
        
    async def get_video_performance(
        self,
        video_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get performance metrics for a specific video
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        performance = {
            "video_id": video_id,
            "period_days": days,
            "metrics": {},
            "hourly_views": [],
            "retention": {},
            "traffic_sources": {}
        }
        
        # Get view metrics
        views = await self.query_metrics(
            metric_type=MetricType.VIEWS,
            start_time=start_time,
            end_time=end_time,
            video_id=video_id,
            aggregation_level=AggregationLevel.HOUR
        )
        
        if views:
            performance["hourly_views"] = [
                {
                    "hour": v.get("period_start"),
                    "views": v.get("sum", 0)
                }
                for v in views
            ]
            
            performance["metrics"]["total_views"] = sum(v.get("sum", 0) for v in views)
            performance["metrics"]["peak_hour_views"] = max(v.get("sum", 0) for v in views)
        
        # Get watch time
        watch_time = await self.query_metrics(
            metric_type=MetricType.WATCH_TIME,
            start_time=start_time,
            end_time=end_time,
            video_id=video_id
        )
        
        if watch_time:
            total_watch_minutes = sum(w.get("value", 0) for w in watch_time)
            performance["metrics"]["total_watch_hours"] = total_watch_minutes / 60
            
            if performance["metrics"].get("total_views"):
                performance["metrics"]["avg_view_duration"] = (
                    total_watch_minutes / performance["metrics"]["total_views"]
                )
        
        return performance
        
    async def generate_reports(self):
        """
        Background task to generate periodic reports
        """
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Generate daily report at midnight
                now = datetime.utcnow()
                if now.hour == 0:
                    await self.generate_daily_report()
                
                # Generate weekly report on Mondays
                if now.weekday() == 0 and now.hour == 0:
                    await self.generate_weekly_report()
                    
            except Exception as e:
                logger.error(f"Report generation error: {e}")
                
    async def generate_daily_report(self) -> AnalyticsReport:
        """
        Generate daily analytics report
        """
        yesterday = date.today() - timedelta(days=1)
        
        report = AnalyticsReport(
            report_id=f"daily_{yesterday.isoformat()}",
            report_type="daily",
            period_start=datetime.combine(yesterday, datetime.min.time()),
            period_end=datetime.combine(yesterday, datetime.max.time()),
            generated_at=datetime.utcnow(),
            metrics={},
            insights=[],
            recommendations=[],
            charts={}
        )
        
        # Collect metrics for all channels
        # This would typically query from database
        
        # Generate insights
        report.insights = await self.generate_insights(report.metrics)
        
        # Generate recommendations
        report.recommendations = await self.generate_recommendations(report.metrics)
        
        # Store report
        await self.store_report(report)
        
        return report
        
    async def generate_weekly_report(self) -> AnalyticsReport:
        """
        Generate weekly analytics report
        """
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=6)
        
        report = AnalyticsReport(
            report_id=f"weekly_{start_date.isoformat()}_{end_date.isoformat()}",
            report_type="weekly",
            period_start=datetime.combine(start_date, datetime.min.time()),
            period_end=datetime.combine(end_date, datetime.max.time()),
            generated_at=datetime.utcnow(),
            metrics={},
            insights=[],
            recommendations=[],
            charts={}
        )
        
        # Generate comprehensive weekly metrics
        # This would aggregate daily reports
        
        return report
        
    async def monitor_alerts(self):
        """
        Monitor metrics for alert conditions
        """
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Check engagement rate
                await self.check_engagement_alerts()
                
                # Check cost alerts
                await self.check_cost_alerts()
                
                # Check performance alerts
                await self.check_performance_alerts()
                
            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                
    async def check_engagement_alerts(self):
        """
        Check for low engagement alerts
        """
        # Query recent engagement metrics
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        engagement_metrics = await self.query_metrics(
            metric_type=MetricType.ENGAGEMENT_RATE,
            start_time=start_time,
            end_time=end_time,
            aggregation_level=AggregationLevel.HOUR
        )
        
        for metric in engagement_metrics:
            if metric.get("avg", 1) < self.alert_thresholds["low_engagement"]:
                await self.send_alert(
                    "low_engagement",
                    f"Low engagement rate detected: {metric.get('avg', 0):.2%}",
                    metric
                )
                
    async def check_cost_alerts(self):
        """
        Check for high cost alerts
        """
        today = date.today()
        start_time = datetime.combine(today, datetime.min.time())
        end_time = datetime.utcnow()
        
        cost_metrics = await self.query_metrics(
            metric_type=MetricType.COST,
            start_time=start_time,
            end_time=end_time
        )
        
        total_cost = sum(m.get("value", 0) for m in cost_metrics)
        
        if total_cost > self.alert_thresholds["high_cost"]:
            await self.send_alert(
                "high_cost",
                f"Daily cost exceeded threshold: ${total_cost:.2f}",
                {"total_cost": total_cost, "threshold": self.alert_thresholds["high_cost"]}
            )
            
    async def check_performance_alerts(self):
        """
        Check for performance degradation alerts
        """
        # Compare current hour to previous hour
        end_time = datetime.utcnow()
        current_hour_start = end_time.replace(minute=0, second=0, microsecond=0)
        previous_hour_start = current_hour_start - timedelta(hours=1)
        
        # Get current hour views
        current_views = await self.query_metrics(
            metric_type=MetricType.VIEWS,
            start_time=current_hour_start,
            end_time=end_time
        )
        
        # Get previous hour views
        previous_views = await self.query_metrics(
            metric_type=MetricType.VIEWS,
            start_time=previous_hour_start,
            end_time=current_hour_start
        )
        
        if current_views and previous_views:
            current_total = sum(v.get("value", 0) for v in current_views)
            previous_total = sum(v.get("value", 0) for v in previous_views)
            
            if previous_total > 0:
                drop_rate = 1 - (current_total / previous_total)
                
                if drop_rate > self.alert_thresholds["view_drop"]:
                    await self.send_alert(
                        "view_drop",
                        f"Significant view drop detected: {drop_rate:.1%}",
                        {
                            "current_views": current_total,
                            "previous_views": previous_total,
                            "drop_rate": drop_rate
                        }
                    )
                    
    async def send_alert(self, alert_type: str, message: str, data: Dict[str, Any]):
        """
        Send alert notification
        """
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        # Store alert in Redis
        await self.redis_client.rpush(
            "alerts:queue",
            json.dumps(alert)
        )
        
        # Log alert
        logger.warning(f"Alert: {alert_type} - {message}")
        
    async def store_report(self, report: AnalyticsReport):
        """
        Store analytics report
        """
        key = f"report:{report.report_type}:{report.report_id}"
        
        await self.redis_client.setex(
            key,
            86400 * 90,  # 90 days retention
            json.dumps(asdict(report), default=str)
        )
        
    async def generate_insights(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate insights from metrics
        """
        insights = []
        
        # Example insight generation logic
        if metrics.get("views", {}).get("trend") == "increasing":
            insights.append("View count is trending upward")
            
        if metrics.get("engagement", {}).get("rate", 0) > 0.05:
            insights.append("Engagement rate is above average")
            
        return insights
        
    async def generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on metrics
        """
        recommendations = []
        
        # Example recommendation logic
        if metrics.get("watch_time", {}).get("average", 0) < 2:
            recommendations.append("Consider creating more engaging content to increase watch time")
            
        if metrics.get("cost", {}).get("total", 0) > 50:
            recommendations.append("Review cost optimization strategies")
            
        return recommendations
        
    def calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend direction from values
        """
        if len(values) < 2:
            return "stable"
            
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
            
    def get_window_start(self, timestamp: datetime, level: AggregationLevel) -> datetime:
        """
        Get the start of the aggregation window for a timestamp
        """
        if level == AggregationLevel.MINUTE:
            return timestamp.replace(second=0, microsecond=0)
        elif level == AggregationLevel.HOUR:
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif level == AggregationLevel.DAY:
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif level == AggregationLevel.WEEK:
            days_since_monday = timestamp.weekday()
            week_start = timestamp - timedelta(days=days_since_monday)
            return week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif level == AggregationLevel.MONTH:
            return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return timestamp


# Global instances
analytics_pipeline = AnalyticsPipeline()  # Backward compatible
advanced_analytics_pipeline = AdvancedAnalyticsPipeline(
    enable_real_time_streaming=True,
    enable_predictive_models=True,
    gdpr_compliance_enabled=True
)