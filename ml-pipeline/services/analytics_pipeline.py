"""
Analytics Data Pipeline
Real-time and batch analytics processing for YTEmpire
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import select, func, and_, or_
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Metrics for monitoring
events_processed = Counter('analytics_events_processed_total', 'Total events processed')
events_failed = Counter('analytics_events_failed_total', 'Total events failed')
processing_time = Histogram('analytics_processing_seconds', 'Time spent processing events')
queue_depth = Gauge('analytics_queue_depth', 'Current queue depth')
data_quality_score = Gauge('analytics_data_quality_score', 'Data quality score')


class EventType(Enum):
    """Analytics event types"""
    VIDEO_VIEW = "video_view"
    VIDEO_PUBLISH = "video_publish"
    CHANNEL_UPDATE = "channel_update"
    USER_ACTION = "user_action"
    COST_INCURRED = "cost_incurred"
    REVENUE_EARNED = "revenue_earned"
    QUALITY_SCORE = "quality_score"
    MODEL_PREDICTION = "model_prediction"
    SYSTEM_METRIC = "system_metric"


@dataclass
class AnalyticsEvent:
    """Base analytics event structure"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    user_id: Optional[str]
    channel_id: Optional[str]
    video_id: Optional[str]
    data: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class AggregatedMetric:
    """Aggregated metric result"""
    metric_name: str
    period: str  # hourly, daily, weekly, monthly
    timestamp: datetime
    dimensions: Dict[str, str]
    values: Dict[str, float]
    confidence: float


class DataQualityChecker:
    """Data quality validation and monitoring"""
    
    def __init__(self):
        self.quality_rules = {
            'completeness': self._check_completeness,
            'consistency': self._check_consistency,
            'timeliness': self._check_timeliness,
            'accuracy': self._check_accuracy,
            'uniqueness': self._check_uniqueness
        }
        self.quality_thresholds = {
            'completeness': 0.95,
            'consistency': 0.98,
            'timeliness': 0.99,
            'accuracy': 0.97,
            'uniqueness': 0.99
        }
    
    async def validate_event(self, event: AnalyticsEvent) -> Tuple[bool, Dict[str, float]]:
        """Validate event data quality"""
        scores = {}
        
        for rule_name, rule_func in self.quality_rules.items():
            score = await rule_func(event)
            scores[rule_name] = score
        
        # Calculate overall quality score
        overall_score = np.mean(list(scores.values()))
        
        # Check if meets minimum thresholds
        is_valid = all(
            scores.get(rule, 0) >= threshold 
            for rule, threshold in self.quality_thresholds.items()
        )
        
        return is_valid, scores
    
    async def _check_completeness(self, event: AnalyticsEvent) -> float:
        """Check if all required fields are present"""
        required_fields = ['event_id', 'event_type', 'timestamp', 'data']
        present_fields = sum(1 for field in required_fields if getattr(event, field) is not None)
        return present_fields / len(required_fields)
    
    async def _check_consistency(self, event: AnalyticsEvent) -> float:
        """Check data consistency"""
        # Check if data types are correct
        try:
            if not isinstance(event.timestamp, datetime):
                return 0.0
            if not isinstance(event.data, dict):
                return 0.0
            return 1.0
        except:
            return 0.0
    
    async def _check_timeliness(self, event: AnalyticsEvent) -> float:
        """Check if event is recent"""
        age = datetime.utcnow() - event.timestamp
        if age > timedelta(hours=24):
            return 0.5  # Old events get lower score
        if age > timedelta(hours=1):
            return 0.9
        return 1.0
    
    async def _check_accuracy(self, event: AnalyticsEvent) -> float:
        """Check data accuracy (simplified)"""
        # Check for obviously wrong values
        if event.event_type == EventType.VIDEO_VIEW:
            views = event.data.get('view_count', 0)
            if views < 0 or views > 1000000000:  # Unrealistic view count
                return 0.0
        return 1.0
    
    async def _check_uniqueness(self, event: AnalyticsEvent) -> float:
        """Check for duplicate events (simplified)"""
        # In production, would check against recent events in cache
        return 1.0 if event.event_id else 0.0


class EventStreamProcessor:
    """Real-time event stream processing"""
    
    def __init__(self, redis_url: str):
        self.redis_client = None
        self.redis_url = redis_url
        self.stream_key = "analytics:events:stream"
        self.consumer_group = "analytics_processors"
        self.quality_checker = DataQualityChecker()
        self.processing = False
        
    async def connect(self):
        """Connect to Redis streams"""
        self.redis_client = await redis.from_url(self.redis_url)
        
        # Create consumer group
        try:
            await self.redis_client.xgroup_create(
                self.stream_key,
                self.consumer_group,
                id='0'
            )
        except:
            pass  # Group already exists
    
    async def publish_event(self, event: AnalyticsEvent) -> str:
        """Publish event to stream"""
        if not self.redis_client:
            await self.connect()
        
        # Validate event quality
        is_valid, quality_scores = await self.quality_checker.validate_event(event)
        
        if not is_valid:
            logger.warning(f"Event {event.event_id} failed quality checks: {quality_scores}")
            events_failed.inc()
            return None
        
        # Serialize event
        event_data = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'timestamp': event.timestamp.isoformat(),
            'user_id': event.user_id or '',
            'channel_id': event.channel_id or '',
            'video_id': event.video_id or '',
            'data': json.dumps(event.data),
            'metadata': json.dumps(event.metadata),
            'quality_score': np.mean(list(quality_scores.values()))
        }
        
        # Add to stream
        message_id = await self.redis_client.xadd(
            self.stream_key,
            event_data
        )
        
        events_processed.inc()
        queue_depth.inc()
        
        return message_id
    
    async def process_stream(self, batch_size: int = 10):
        """Process events from stream"""
        if not self.redis_client:
            await self.connect()
        
        self.processing = True
        consumer_name = f"consumer_{datetime.utcnow().timestamp()}"
        
        while self.processing:
            try:
                # Read from stream
                messages = await self.redis_client.xreadgroup(
                    self.consumer_group,
                    consumer_name,
                    {self.stream_key: '>'},
                    count=batch_size,
                    block=1000  # Block for 1 second
                )
                
                if messages:
                    for stream_name, stream_messages in messages:
                        for message_id, data in stream_messages:
                            await self._process_message(message_id, data)
                            
                            # Acknowledge message
                            await self.redis_client.xack(
                                self.stream_key,
                                self.consumer_group,
                                message_id
                            )
                            queue_depth.dec()
                
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                events_failed.inc()
                await asyncio.sleep(1)
    
    async def _process_message(self, message_id: str, data: Dict[str, str]):
        """Process individual message"""
        with processing_time.time():
            try:
                # Reconstruct event
                event = AnalyticsEvent(
                    event_id=data['event_id'],
                    event_type=EventType(data['event_type']),
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    user_id=data.get('user_id') or None,
                    channel_id=data.get('channel_id') or None,
                    video_id=data.get('video_id') or None,
                    data=json.loads(data['data']),
                    metadata=json.loads(data['metadata'])
                )
                
                # Route to appropriate processor
                await self._route_event(event)
                
            except Exception as e:
                logger.error(f"Message processing failed: {e}")
                events_failed.inc()
    
    async def _route_event(self, event: AnalyticsEvent):
        """Route event to appropriate processor"""
        if event.event_type in [EventType.VIDEO_VIEW, EventType.VIDEO_PUBLISH]:
            await self._process_video_event(event)
        elif event.event_type in [EventType.COST_INCURRED, EventType.REVENUE_EARNED]:
            await self._process_financial_event(event)
        elif event.event_type == EventType.QUALITY_SCORE:
            await self._process_quality_event(event)
        else:
            await self._process_generic_event(event)
    
    async def _process_video_event(self, event: AnalyticsEvent):
        """Process video-related events"""
        # Store in time-series format for aggregation
        key = f"analytics:video:{event.video_id}:{event.event_type.value}"
        await self.redis_client.zadd(
            key,
            {json.dumps(event.data): event.timestamp.timestamp()}
        )
        await self.redis_client.expire(key, 86400 * 30)  # 30 days retention
    
    async def _process_financial_event(self, event: AnalyticsEvent):
        """Process financial events"""
        # Update running totals
        if event.event_type == EventType.COST_INCURRED:
            key = f"analytics:costs:{event.channel_id}:{datetime.utcnow().strftime('%Y%m%d')}"
            await self.redis_client.hincrbyfloat(key, event.data.get('service', 'unknown'), event.data.get('amount', 0))
        elif event.event_type == EventType.REVENUE_EARNED:
            key = f"analytics:revenue:{event.channel_id}:{datetime.utcnow().strftime('%Y%m%d')}"
            await self.redis_client.hincrbyfloat(key, 'total', event.data.get('amount', 0))
    
    async def _process_quality_event(self, event: AnalyticsEvent):
        """Process quality score events"""
        score = event.data.get('score', 0)
        data_quality_score.set(score)
    
    async def _process_generic_event(self, event: AnalyticsEvent):
        """Process generic events"""
        # Store in general purpose format
        key = f"analytics:events:{event.event_type.value}:{datetime.utcnow().strftime('%Y%m%d')}"
        await self.redis_client.lpush(key, json.dumps(asdict(event)))
        await self.redis_client.expire(key, 86400 * 7)  # 7 days retention


class DataTransformationEngine:
    """Transform raw data into analytics-ready format"""
    
    def __init__(self):
        self.transformers = {
            'normalize': self._normalize_data,
            'aggregate': self._aggregate_data,
            'enrich': self._enrich_data,
            'derive': self._derive_metrics
        }
    
    async def transform(
        self,
        data: pd.DataFrame,
        transformations: List[str]
    ) -> pd.DataFrame:
        """Apply transformations to data"""
        result = data.copy()
        
        for transform_name in transformations:
            if transform_name in self.transformers:
                result = await self.transformers[transform_name](result)
        
        return result
    
    async def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical columns"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].std() > 0:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        return df
    
    async def _aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by time periods"""
        if 'timestamp' not in df.columns:
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.date
        df['week'] = df['timestamp'].dt.isocalendar().week
        df['month'] = df['timestamp'].dt.month
        
        return df
    
    async def _enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich data with additional context"""
        # Add derived fields
        if 'view_count' in df.columns and 'like_count' in df.columns:
            df['engagement_rate'] = df['like_count'] / df['view_count'].replace(0, 1)
        
        if 'revenue' in df.columns and 'cost' in df.columns:
            df['profit'] = df['revenue'] - df['cost']
            df['profit_margin'] = df['profit'] / df['revenue'].replace(0, 1)
        
        return df
    
    async def _derive_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive business metrics"""
        # Calculate moving averages
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if len(df) > 7:
                df[f'{col}_ma7'] = df[col].rolling(window=7, min_periods=1).mean()
            if len(df) > 30:
                df[f'{col}_ma30'] = df[col].rolling(window=30, min_periods=1).mean()
        
        return df


class AggregationPipeline:
    """Data aggregation pipeline for analytics"""
    
    def __init__(self, redis_url: str):
        self.redis_client = None
        self.redis_url = redis_url
        self.transformation_engine = DataTransformationEngine()
    
    async def connect(self):
        """Connect to data stores"""
        self.redis_client = await redis.from_url(self.redis_url)
    
    async def aggregate_metrics(
        self,
        metric_type: str,
        period: str,
        dimensions: Dict[str, str]
    ) -> AggregatedMetric:
        """Aggregate metrics by period and dimensions"""
        if not self.redis_client:
            await self.connect()
        
        # Build key pattern
        key_pattern = f"analytics:{metric_type}:*"
        if dimensions:
            for dim, value in dimensions.items():
                key_pattern = f"analytics:{metric_type}:{value}:*"
        
        # Get matching keys
        keys = []
        cursor = 0
        while True:
            cursor, batch_keys = await self.redis_client.scan(
                cursor,
                match=key_pattern,
                count=100
            )
            keys.extend(batch_keys)
            if cursor == 0:
                break
        
        # Aggregate data
        aggregated_values = defaultdict(float)
        count = 0
        
        for key in keys:
            key_type = await self.redis_client.type(key)
            
            if key_type == 'hash':
                data = await self.redis_client.hgetall(key)
                for field, value in data.items():
                    try:
                        aggregated_values[field] += float(value)
                        count += 1
                    except:
                        pass
            elif key_type == 'zset':
                data = await self.redis_client.zrange(key, 0, -1, withscores=True)
                for member, score in data:
                    aggregated_values['total'] += score
                    count += 1
        
        # Calculate confidence based on data availability
        confidence = min(1.0, count / 100)  # More data = higher confidence
        
        return AggregatedMetric(
            metric_name=metric_type,
            period=period,
            timestamp=datetime.utcnow(),
            dimensions=dimensions,
            values=dict(aggregated_values),
            confidence=confidence
        )
    
    async def create_rollups(
        self,
        source_period: str,
        target_period: str
    ) -> int:
        """Create data rollups for different time periods"""
        if not self.redis_client:
            await self.connect()
        
        rollup_count = 0
        
        # Define rollup rules
        rollup_rules = {
            ('hourly', 'daily'): 24,
            ('daily', 'weekly'): 7,
            ('daily', 'monthly'): 30,
            ('weekly', 'monthly'): 4
        }
        
        if (source_period, target_period) not in rollup_rules:
            logger.warning(f"No rollup rule for {source_period} -> {target_period}")
            return 0
        
        aggregation_factor = rollup_rules[(source_period, target_period)]
        
        # Get source keys
        pattern = f"analytics:*:{source_period}:*"
        keys = []
        cursor = 0
        while True:
            cursor, batch_keys = await self.redis_client.scan(
                cursor,
                match=pattern,
                count=100
            )
            keys.extend(batch_keys)
            if cursor == 0:
                break
        
        # Group keys by metric type
        grouped_keys = defaultdict(list)
        for key in keys:
            parts = key.split(':')
            if len(parts) >= 3:
                metric_type = parts[1]
                grouped_keys[metric_type].append(key)
        
        # Create rollups
        for metric_type, metric_keys in grouped_keys.items():
            if len(metric_keys) >= aggregation_factor:
                # Aggregate and store
                target_key = f"analytics:{metric_type}:{target_period}:{datetime.utcnow().strftime('%Y%m%d')}"
                
                for i in range(0, len(metric_keys), aggregation_factor):
                    batch = metric_keys[i:i+aggregation_factor]
                    
                    # Aggregate batch
                    aggregated = defaultdict(float)
                    for key in batch:
                        key_type = await self.redis_client.type(key)
                        if key_type == 'hash':
                            data = await self.redis_client.hgetall(key)
                            for field, value in data.items():
                                try:
                                    aggregated[field] += float(value)
                                except:
                                    pass
                    
                    # Store rollup
                    if aggregated:
                        for field, value in aggregated.items():
                            await self.redis_client.hincrbyfloat(target_key, field, value)
                        rollup_count += 1
        
        return rollup_count


class AnalyticsPipeline:
    """Main analytics pipeline orchestrator"""
    
    def __init__(self, redis_url: str, db_url: str):
        self.redis_url = redis_url
        self.db_url = db_url
        self.stream_processor = EventStreamProcessor(redis_url)
        self.aggregation_pipeline = AggregationPipeline(redis_url)
        self.quality_checker = DataQualityChecker()
        self.transformation_engine = DataTransformationEngine()
        self.engine = None
    
    async def initialize(self):
        """Initialize pipeline components"""
        await self.stream_processor.connect()
        await self.aggregation_pipeline.connect()
        self.engine = create_async_engine(self.db_url)
        
        logger.info("Analytics pipeline initialized")
    
    async def ingest_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        **kwargs
    ) -> str:
        """Ingest analytics event"""
        event = AnalyticsEvent(
            event_id=f"{event_type.value}_{datetime.utcnow().timestamp()}",
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=kwargs.get('user_id'),
            channel_id=kwargs.get('channel_id'),
            video_id=kwargs.get('video_id'),
            data=data,
            metadata=kwargs.get('metadata', {})
        )
        
        return await self.stream_processor.publish_event(event)
    
    async def run_stream_processing(self):
        """Run continuous stream processing"""
        await self.stream_processor.process_stream()
    
    async def run_batch_aggregations(self):
        """Run batch aggregation jobs"""
        while True:
            try:
                # Run hourly aggregations
                await self.aggregation_pipeline.create_rollups('hourly', 'daily')
                
                # Run daily aggregations (once per day)
                if datetime.utcnow().hour == 0:
                    await self.aggregation_pipeline.create_rollups('daily', 'weekly')
                    await self.aggregation_pipeline.create_rollups('daily', 'monthly')
                
                # Sleep for an hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Batch aggregation failed: {e}")
                await asyncio.sleep(60)
    
    async def get_metrics(
        self,
        metric_type: str,
        period: str = 'daily',
        dimensions: Optional[Dict[str, str]] = None
    ) -> AggregatedMetric:
        """Get aggregated metrics"""
        return await self.aggregation_pipeline.aggregate_metrics(
            metric_type,
            period,
            dimensions or {}
        )
    
    async def run_data_quality_checks(self) -> Dict[str, float]:
        """Run data quality checks on recent data"""
        quality_results = {
            'stream_health': 1.0,
            'data_completeness': 0.0,
            'data_freshness': 0.0,
            'processing_lag': 0.0
        }
        
        # Check stream health
        info = await self.stream_processor.redis_client.xinfo_stream(self.stream_processor.stream_key)
        if info:
            quality_results['stream_health'] = 1.0
            
            # Check data freshness
            if info.get('last-generated-id'):
                last_id = info['last-generated-id']
                timestamp = int(last_id.split('-')[0]) / 1000
                age = datetime.utcnow().timestamp() - timestamp
                quality_results['data_freshness'] = max(0, 1 - (age / 3600))  # Decay over 1 hour
        
        # Check queue depth
        current_queue_depth = queue_depth._value.get()
        quality_results['processing_lag'] = max(0, 1 - (current_queue_depth / 1000))  # Good if < 1000 events
        
        # Overall data quality
        quality_results['overall'] = np.mean(list(quality_results.values()))
        data_quality_score.set(quality_results['overall'])
        
        return quality_results
    
    async def shutdown(self):
        """Shutdown pipeline gracefully"""
        self.stream_processor.processing = False
        await self.stream_processor.redis_client.close()
        await self.aggregation_pipeline.redis_client.close()
        if self.engine:
            await self.engine.dispose()
        
        logger.info("Analytics pipeline shut down")


# Global instance
analytics_pipeline = None

async def get_analytics_pipeline(redis_url: str, db_url: str) -> AnalyticsPipeline:
    """Get or create analytics pipeline instance"""
    global analytics_pipeline
    
    if analytics_pipeline is None:
        analytics_pipeline = AnalyticsPipeline(redis_url, db_url)
        await analytics_pipeline.initialize()
    
    return analytics_pipeline