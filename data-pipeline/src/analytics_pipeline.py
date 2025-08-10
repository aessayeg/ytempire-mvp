"""
Real-time Analytics Pipeline for YTEmpire
Kafka-based streaming analytics with real-time dashboards
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import redis
import pandas as pd
import numpy as np
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
import psycopg2
from psycopg2.extras import RealDictCursor
import websocket
from collections import defaultdict
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventType(Enum):
    """Analytics event types"""
    PAGE_VIEW = "page_view"
    VIDEO_VIEW = "video_view"
    VIDEO_COMPLETE = "video_complete"
    CHANNEL_SUBSCRIBE = "channel_subscribe"
    VIDEO_LIKE = "video_like"
    VIDEO_COMMENT = "video_comment"
    VIDEO_SHARE = "video_share"
    GENERATION_START = "generation_start"
    GENERATION_COMPLETE = "generation_complete"
    GENERATION_FAILED = "generation_failed"
    USER_SIGNUP = "user_signup"
    USER_LOGIN = "user_login"
    PAYMENT = "payment"

@dataclass
class AnalyticsEvent:
    """Analytics event structure"""
    event_id: str
    event_type: EventType
    user_id: Optional[str]
    session_id: str
    timestamp: datetime
    properties: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class AggregatedMetric:
    """Aggregated analytics metric"""
    metric_name: str
    value: float
    period: str
    dimensions: Dict[str, str]
    timestamp: datetime

class AnalyticsPipeline:
    """
    Real-time analytics pipeline with Kafka streaming
    """
    
    def __init__(
        self,
        kafka_bootstrap_servers: str = 'localhost:9092',
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        postgres_config: Optional[Dict[str, str]] = None
    ):
        self.kafka_servers = kafka_bootstrap_servers
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # PostgreSQL connection
        self.postgres_config = postgres_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'ytempire_analytics',
            'user': 'analytics_user',
            'password': 'analytics_pass'
        }
        
        # Kafka setup
        self.producer = None
        self.consumer = None
        self.admin_client = None
        
        # Stream processing state
        self.aggregation_windows = {
            '1min': 60,
            '5min': 300,
            '1hour': 3600,
            '1day': 86400
        }
        
        self.metric_buffers = defaultdict(list)
        self.websocket_connections = []
    
    async def initialize(self):
        """Initialize analytics pipeline components"""
        
        # Create Kafka topics
        await self._create_kafka_topics()
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        
        # Initialize database tables
        await self._initialize_database()
        
        # Start stream processors
        asyncio.create_task(self._process_event_stream())
        asyncio.create_task(self._process_aggregations())
        asyncio.create_task(self._process_alerts())
        
        logger.info("Analytics pipeline initialized")
    
    async def _create_kafka_topics(self):
        """Create required Kafka topics"""
        
        self.admin_client = KafkaAdminClient(
            bootstrap_servers=self.kafka_servers,
            client_id='analytics_admin'
        )
        
        topics = [
            NewTopic(name='analytics_events', num_partitions=3, replication_factor=1),
            NewTopic(name='analytics_metrics', num_partitions=2, replication_factor=1),
            NewTopic(name='analytics_alerts', num_partitions=1, replication_factor=1)
        ]
        
        try:
            self.admin_client.create_topics(new_topics=topics, validate_only=False)
            logger.info("Kafka topics created successfully")
        except Exception as e:
            logger.warning(f"Topics may already exist: {e}")
    
    async def _initialize_database(self):
        """Initialize analytics database tables"""
        
        create_tables_sql = """
        -- Events table for raw events
        CREATE TABLE IF NOT EXISTS analytics_events (
            event_id VARCHAR(64) PRIMARY KEY,
            event_type VARCHAR(50) NOT NULL,
            user_id VARCHAR(64),
            session_id VARCHAR(64) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            properties JSONB,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Metrics table for aggregated data
        CREATE TABLE IF NOT EXISTS analytics_metrics (
            id SERIAL PRIMARY KEY,
            metric_name VARCHAR(100) NOT NULL,
            value FLOAT NOT NULL,
            period VARCHAR(20) NOT NULL,
            dimensions JSONB,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- User sessions table
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id VARCHAR(64) PRIMARY KEY,
            user_id VARCHAR(64),
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            duration_seconds INTEGER,
            page_views INTEGER DEFAULT 0,
            events_count INTEGER DEFAULT 0,
            metadata JSONB
        );
        
        -- Video analytics table
        CREATE TABLE IF NOT EXISTS video_analytics (
            video_id VARCHAR(64) NOT NULL,
            date DATE NOT NULL,
            views INTEGER DEFAULT 0,
            watch_time_seconds INTEGER DEFAULT 0,
            likes INTEGER DEFAULT 0,
            comments INTEGER DEFAULT 0,
            shares INTEGER DEFAULT 0,
            revenue DECIMAL(10, 2) DEFAULT 0,
            cost DECIMAL(10, 2) DEFAULT 0,
            PRIMARY KEY (video_id, date)
        );
        
        -- Channel analytics table
        CREATE TABLE IF NOT EXISTS channel_analytics (
            channel_id VARCHAR(64) NOT NULL,
            date DATE NOT NULL,
            views INTEGER DEFAULT 0,
            subscribers_gained INTEGER DEFAULT 0,
            subscribers_lost INTEGER DEFAULT 0,
            revenue DECIMAL(10, 2) DEFAULT 0,
            videos_published INTEGER DEFAULT 0,
            engagement_rate FLOAT,
            PRIMARY KEY (channel_id, date)
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_events_timestamp ON analytics_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_events_user ON analytics_events(user_id);
        CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON analytics_metrics(timestamp);
        CREATE INDEX IF NOT EXISTS idx_metrics_name ON analytics_metrics(metric_name);
        """
        
        try:
            conn = psycopg2.connect(**self.postgres_config)
            cur = conn.cursor()
            cur.execute(create_tables_sql)
            conn.commit()
            cur.close()
            conn.close()
            logger.info("Analytics database tables created")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
    
    async def track_event(
        self,
        event_type: EventType,
        user_id: Optional[str],
        session_id: str,
        properties: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track analytics event
        
        Args:
            event_type: Type of event
            user_id: User identifier
            session_id: Session identifier
            properties: Event properties
            metadata: Additional metadata
        """
        
        event = AnalyticsEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now(),
            properties=properties,
            metadata=metadata or {}
        )
        
        # Send to Kafka
        await self._send_to_kafka('analytics_events', event)
        
        # Update real-time counters
        await self._update_realtime_counters(event)
        
        # Store in database
        await self._store_event(event)
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        data = f"{datetime.now().isoformat()}{np.random.random()}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    async def _send_to_kafka(self, topic: str, event: AnalyticsEvent):
        """Send event to Kafka topic"""
        
        if self.producer:
            try:
                # Serialize event
                event_dict = {
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'user_id': event.user_id,
                    'session_id': event.session_id,
                    'timestamp': event.timestamp.isoformat(),
                    'properties': event.properties,
                    'metadata': event.metadata
                }
                
                # Send to Kafka
                self.producer.send(
                    topic,
                    key=event.user_id,
                    value=event_dict
                )
                
            except Exception as e:
                logger.error(f"Error sending to Kafka: {e}")
    
    async def _update_realtime_counters(self, event: AnalyticsEvent):
        """Update real-time Redis counters"""
        
        try:
            # Update global counters
            self.redis_client.hincrby('analytics:counters:global', event.event_type.value, 1)
            
            # Update hourly counters
            hour_key = datetime.now().strftime('%Y%m%d%H')
            self.redis_client.hincrby(f'analytics:counters:hourly:{hour_key}', event.event_type.value, 1)
            self.redis_client.expire(f'analytics:counters:hourly:{hour_key}', 86400)  # 24 hours
            
            # Update user counters
            if event.user_id:
                self.redis_client.hincrby(f'analytics:user:{event.user_id}', event.event_type.value, 1)
            
            # Update session activity
            self.redis_client.zadd(
                'analytics:active_sessions',
                {event.session_id: datetime.now().timestamp()}
            )
            
        except Exception as e:
            logger.error(f"Error updating Redis counters: {e}")
    
    async def _store_event(self, event: AnalyticsEvent):
        """Store event in PostgreSQL"""
        
        try:
            conn = psycopg2.connect(**self.postgres_config)
            cur = conn.cursor()
            
            insert_sql = """
            INSERT INTO analytics_events 
            (event_id, event_type, user_id, session_id, timestamp, properties, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (event_id) DO NOTHING
            """
            
            cur.execute(insert_sql, (
                event.event_id,
                event.event_type.value,
                event.user_id,
                event.session_id,
                event.timestamp,
                json.dumps(event.properties),
                json.dumps(event.metadata)
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing event: {e}")
    
    async def _process_event_stream(self):
        """Process incoming event stream from Kafka"""
        
        self.consumer = KafkaConsumer(
            'analytics_events',
            bootstrap_servers=self.kafka_servers,
            auto_offset_reset='latest',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        while True:
            try:
                # Poll for messages
                messages = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, records in messages.items():
                    for record in records:
                        event_data = record.value
                        
                        # Process event
                        await self._process_single_event(event_data)
                        
                        # Update aggregations
                        self._buffer_for_aggregation(event_data)
                        
                        # Send to WebSocket clients
                        await self._broadcast_to_websockets(event_data)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing event stream: {e}")
                await asyncio.sleep(5)
    
    async def _process_single_event(self, event_data: Dict[str, Any]):
        """Process individual event"""
        
        event_type = EventType(event_data['event_type'])
        
        # Update specific analytics based on event type
        if event_type == EventType.VIDEO_VIEW:
            await self._update_video_analytics(event_data)
        elif event_type == EventType.CHANNEL_SUBSCRIBE:
            await self._update_channel_analytics(event_data)
        elif event_type == EventType.GENERATION_COMPLETE:
            await self._update_generation_analytics(event_data)
    
    async def _update_video_analytics(self, event_data: Dict[str, Any]):
        """Update video-specific analytics"""
        
        video_id = event_data['properties'].get('video_id')
        if not video_id:
            return
        
        try:
            conn = psycopg2.connect(**self.postgres_config)
            cur = conn.cursor()
            
            update_sql = """
            INSERT INTO video_analytics (video_id, date, views)
            VALUES (%s, %s, 1)
            ON CONFLICT (video_id, date)
            DO UPDATE SET views = video_analytics.views + 1
            """
            
            cur.execute(update_sql, (
                video_id,
                datetime.now().date()
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating video analytics: {e}")
    
    async def _update_channel_analytics(self, event_data: Dict[str, Any]):
        """Update channel-specific analytics"""
        
        channel_id = event_data['properties'].get('channel_id')
        if not channel_id:
            return
        
        try:
            conn = psycopg2.connect(**self.postgres_config)
            cur = conn.cursor()
            
            update_sql = """
            INSERT INTO channel_analytics (channel_id, date, subscribers_gained)
            VALUES (%s, %s, 1)
            ON CONFLICT (channel_id, date)
            DO UPDATE SET subscribers_gained = channel_analytics.subscribers_gained + 1
            """
            
            cur.execute(update_sql, (
                channel_id,
                datetime.now().date()
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating channel analytics: {e}")
    
    async def _update_generation_analytics(self, event_data: Dict[str, Any]):
        """Update generation analytics"""
        
        # Track generation costs and success rates
        cost = event_data['properties'].get('cost', 0)
        success = event_data['event_type'] == 'generation_complete'
        
        # Update daily generation metrics
        date_key = datetime.now().strftime('%Y%m%d')
        
        if success:
            self.redis_client.hincrby(f'analytics:generation:{date_key}', 'success', 1)
            self.redis_client.hincrbyfloat(f'analytics:generation:{date_key}', 'total_cost', cost)
        else:
            self.redis_client.hincrby(f'analytics:generation:{date_key}', 'failed', 1)
    
    def _buffer_for_aggregation(self, event_data: Dict[str, Any]):
        """Buffer events for aggregation"""
        
        # Add to appropriate buffers
        for window_name, window_seconds in self.aggregation_windows.items():
            buffer_key = f"{window_name}:{int(datetime.now().timestamp() / window_seconds)}"
            self.metric_buffers[buffer_key].append(event_data)
    
    async def _process_aggregations(self):
        """Process aggregated metrics"""
        
        while True:
            try:
                current_time = datetime.now()
                
                # Process each aggregation window
                for window_name, window_seconds in self.aggregation_windows.items():
                    # Check if window is complete
                    window_key = f"{window_name}:{int((current_time.timestamp() - window_seconds) / window_seconds)}"
                    
                    if window_key in self.metric_buffers:
                        events = self.metric_buffers.pop(window_key)
                        
                        if events:
                            # Calculate aggregated metrics
                            metrics = self._calculate_aggregated_metrics(events, window_name)
                            
                            # Store metrics
                            for metric in metrics:
                                await self._store_metric(metric)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error processing aggregations: {e}")
                await asyncio.sleep(30)
    
    def _calculate_aggregated_metrics(
        self,
        events: List[Dict[str, Any]],
        window: str
    ) -> List[AggregatedMetric]:
        """Calculate aggregated metrics from events"""
        
        metrics = []
        
        # Count events by type
        event_counts = defaultdict(int)
        for event in events:
            event_counts[event['event_type']] += 1
        
        # Create metrics
        for event_type, count in event_counts.items():
            metric = AggregatedMetric(
                metric_name=f"event_count_{event_type}",
                value=count,
                period=window,
                dimensions={'event_type': event_type},
                timestamp=datetime.now()
            )
            metrics.append(metric)
        
        # Calculate unique users
        unique_users = len(set(e['user_id'] for e in events if e.get('user_id')))
        metrics.append(AggregatedMetric(
            metric_name="unique_users",
            value=unique_users,
            period=window,
            dimensions={},
            timestamp=datetime.now()
        ))
        
        return metrics
    
    async def _store_metric(self, metric: AggregatedMetric):
        """Store aggregated metric"""
        
        try:
            conn = psycopg2.connect(**self.postgres_config)
            cur = conn.cursor()
            
            insert_sql = """
            INSERT INTO analytics_metrics 
            (metric_name, value, period, dimensions, timestamp)
            VALUES (%s, %s, %s, %s, %s)
            """
            
            cur.execute(insert_sql, (
                metric.metric_name,
                metric.value,
                metric.period,
                json.dumps(metric.dimensions),
                metric.timestamp
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing metric: {e}")
    
    async def _process_alerts(self):
        """Process analytics alerts"""
        
        while True:
            try:
                # Check for alert conditions
                alerts = await self._check_alert_conditions()
                
                # Send alerts
                for alert in alerts:
                    await self._send_alert(alert)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error processing alerts: {e}")
                await asyncio.sleep(60)
    
    async def _check_alert_conditions(self) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        
        alerts = []
        
        # Check error rate
        error_count = int(self.redis_client.hget('analytics:counters:global', 'generation_failed') or 0)
        success_count = int(self.redis_client.hget('analytics:counters:global', 'generation_complete') or 0)
        
        if success_count > 0:
            error_rate = error_count / (error_count + success_count)
            if error_rate > 0.1:  # 10% error rate threshold
                alerts.append({
                    'type': 'high_error_rate',
                    'severity': 'warning',
                    'message': f'High error rate detected: {error_rate:.2%}',
                    'value': error_rate
                })
        
        # Check cost threshold
        date_key = datetime.now().strftime('%Y%m%d')
        daily_cost = float(self.redis_client.hget(f'analytics:generation:{date_key}', 'total_cost') or 0)
        
        if daily_cost > 100:  # $100 daily cost threshold
            alerts.append({
                'type': 'high_cost',
                'severity': 'warning',
                'message': f'High daily cost: ${daily_cost:.2f}',
                'value': daily_cost
            })
        
        return alerts
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """Send alert notification"""
        
        # Send to Kafka alerts topic
        if self.producer:
            self.producer.send('analytics_alerts', value=alert)
        
        # Log alert
        logger.warning(f"Alert: {alert['message']}")
        
        # Could also send to email, Slack, etc.
    
    async def _broadcast_to_websockets(self, event_data: Dict[str, Any]):
        """Broadcast event to WebSocket clients"""
        
        message = json.dumps({
            'type': 'analytics_event',
            'data': event_data
        })
        
        # Send to all connected clients
        for ws in self.websocket_connections:
            try:
                await ws.send(message)
            except:
                # Remove disconnected clients
                self.websocket_connections.remove(ws)
    
    async def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time analytics metrics"""
        
        # Get current counters
        global_counters = self.redis_client.hgetall('analytics:counters:global')
        
        # Get active sessions
        active_sessions = self.redis_client.zcount(
            'analytics:active_sessions',
            datetime.now().timestamp() - 300,  # Last 5 minutes
            datetime.now().timestamp()
        )
        
        # Get recent metrics from database
        conn = psycopg2.connect(**self.postgres_config)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get today's video stats
        cur.execute("""
            SELECT 
                COUNT(*) as videos_today,
                SUM(views) as views_today,
                SUM(revenue) as revenue_today,
                SUM(cost) as cost_today
            FROM video_analytics
            WHERE date = CURRENT_DATE
        """)
        today_stats = cur.fetchone()
        
        cur.close()
        conn.close()
        
        return {
            'realtime': {
                'active_sessions': active_sessions,
                'events_total': sum(int(v) for v in global_counters.values())
            },
            'today': today_stats,
            'counters': global_counters
        }


# Initialize global instance
analytics_pipeline = AnalyticsPipeline()