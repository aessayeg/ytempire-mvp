"""
Apache Flink Streaming Analytics Setup
Real-time stream processing for YTEmpire analytics
"""
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import redis.asyncio as redis
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.table.descriptors import Schema, Kafka, Json, Rowtime
from pyflink.table.window import Tumble, Slide, Session
from pyflink.table.expressions import col, lit, current_timestamp
from pyflink.table.udf import udf
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer

from app.core.config import settings

logger = logging.getLogger(__name__)


class StreamType(str, Enum):
    """Types of data streams"""
    VIDEO_EVENTS = "video_events"
    USER_ACTIONS = "user_actions"
    COST_EVENTS = "cost_events"
    REVENUE_EVENTS = "revenue_events"
    QUALITY_METRICS = "quality_metrics"
    SYSTEM_METRICS = "system_metrics"


@dataclass
class StreamingJob:
    """Streaming job configuration"""
    job_id: str
    name: str
    stream_type: StreamType
    source_topic: str
    sink_topic: str
    window_size: int  # seconds
    aggregations: List[str]
    filters: Dict[str, Any]
    is_active: bool
    created_at: datetime


class FlinkStreamingAnalytics:
    """Apache Flink streaming analytics service"""
    
    def __init__(self):
        self.env = StreamExecutionEnvironment.get_execution_environment()
        self.t_env = StreamTableEnvironment.create(self.env)
        self.redis_client: Optional[redis.Redis] = None
        self.jobs: Dict[str, StreamingJob] = {}
        
        # Configure Flink environment
        self.env.set_parallelism(4)
        self.env.enable_checkpointing(60000)  # Checkpoint every minute
        
        # Kafka configuration
        self.kafka_bootstrap_servers = settings.KAFKA_BOOTSTRAP_SERVERS or "localhost:9092"
        
    async def initialize(self):
        """Initialize streaming analytics"""
        try:
            # Connect to Redis for metadata
            self.redis_client = await redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Configure table environment
            self._configure_table_environment()
            
            # Register UDFs
            self._register_udfs()
            
            # Create default streaming jobs
            await self._create_default_jobs()
            
            logger.info("Flink streaming analytics initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize streaming analytics: {e}")
            raise
            
    def _configure_table_environment(self):
        """Configure Flink table environment"""
        # Add Kafka connector
        self.t_env.get_config().get_configuration().set_string(
            "table.exec.mini-batch.enabled", "true"
        )
        self.t_env.get_config().get_configuration().set_string(
            "table.exec.mini-batch.allow-latency", "5 s"
        )
        self.t_env.get_config().get_configuration().set_string(
            "table.exec.mini-batch.size", "5000"
        )
        
    def _register_udfs(self):
        """Register user-defined functions"""
        # Quality score calculation UDF
        @udf(result_type=Types.FLOAT())
        def calculate_quality_score(views, likes, comments, watch_time):
            if views == 0:
                return 0.0
            engagement_rate = (likes + comments) / views
            avg_watch_time = watch_time / views if views > 0 else 0
            score = (engagement_rate * 0.4 + min(avg_watch_time / 300, 1) * 0.6) * 100
            return float(min(score, 100))
            
        self.t_env.create_temporary_function("calculate_quality_score", calculate_quality_score)
        
        # ROI calculation UDF
        @udf(result_type=Types.FLOAT())
        def calculate_roi(revenue, cost):
            if cost == 0:
                return 100.0 if revenue > 0 else 0.0
            return float(((revenue - cost) / cost) * 100)
            
        self.t_env.create_temporary_function("calculate_roi", calculate_roi)
        
    async def create_video_events_stream(self):
        """Create video events streaming job"""
        # Define source table (Kafka)
        self.t_env.execute_sql("""
            CREATE TABLE video_events (
                event_id STRING,
                video_id STRING,
                channel_id STRING,
                event_type STRING,
                views BIGINT,
                likes BIGINT,
                comments BIGINT,
                watch_time_seconds BIGINT,
                event_time TIMESTAMP(3),
                WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
            ) WITH (
                'connector' = 'kafka',
                'topic' = 'video-events',
                'properties.bootstrap.servers' = '{servers}',
                'properties.group.id' = 'flink-video-analytics',
                'format' = 'json',
                'scan.startup.mode' = 'latest-offset'
            )
        """.format(servers=self.kafka_bootstrap_servers))
        
        # Define sink table (Kafka)
        self.t_env.execute_sql("""
            CREATE TABLE video_analytics (
                window_start TIMESTAMP(3),
                window_end TIMESTAMP(3),
                channel_id STRING,
                total_views BIGINT,
                total_likes BIGINT,
                total_comments BIGINT,
                avg_watch_time FLOAT,
                engagement_rate FLOAT,
                quality_score FLOAT,
                video_count BIGINT
            ) WITH (
                'connector' = 'kafka',
                'topic' = 'video-analytics',
                'properties.bootstrap.servers' = '{servers}',
                'format' = 'json'
            )
        """.format(servers=self.kafka_bootstrap_servers))
        
        # Create streaming query with windowed aggregation
        self.t_env.execute_sql("""
            INSERT INTO video_analytics
            SELECT 
                TUMBLE_START(event_time, INTERVAL '5' MINUTE) as window_start,
                TUMBLE_END(event_time, INTERVAL '5' MINUTE) as window_end,
                channel_id,
                SUM(views) as total_views,
                SUM(likes) as total_likes,
                SUM(comments) as total_comments,
                AVG(watch_time_seconds) as avg_watch_time,
                CAST(SUM(likes + comments) AS FLOAT) / CAST(SUM(views) AS FLOAT) as engagement_rate,
                calculate_quality_score(
                    SUM(views), 
                    SUM(likes), 
                    SUM(comments), 
                    SUM(watch_time_seconds)
                ) as quality_score,
                COUNT(DISTINCT video_id) as video_count
            FROM video_events
            GROUP BY 
                channel_id,
                TUMBLE(event_time, INTERVAL '5' MINUTE)
        """)
        
    async def create_revenue_stream(self):
        """Create revenue analytics streaming job"""
        # Define revenue events table
        self.t_env.execute_sql("""
            CREATE TABLE revenue_events (
                event_id STRING,
                channel_id STRING,
                video_id STRING,
                revenue FLOAT,
                cost FLOAT,
                source STRING,
                event_time TIMESTAMP(3),
                WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
            ) WITH (
                'connector' = 'kafka',
                'topic' = 'revenue-events',
                'properties.bootstrap.servers' = '{servers}',
                'properties.group.id' = 'flink-revenue-analytics',
                'format' = 'json',
                'scan.startup.mode' = 'latest-offset'
            )
        """.format(servers=self.kafka_bootstrap_servers))
        
        # Define revenue analytics sink
        self.t_env.execute_sql("""
            CREATE TABLE revenue_analytics (
                window_start TIMESTAMP(3),
                window_end TIMESTAMP(3),
                channel_id STRING,
                total_revenue FLOAT,
                total_cost FLOAT,
                profit FLOAT,
                roi FLOAT,
                avg_revenue_per_video FLOAT,
                video_count BIGINT
            ) WITH (
                'connector' = 'kafka',
                'topic' = 'revenue-analytics',
                'properties.bootstrap.servers' = '{servers}',
                'format' = 'json'
            )
        """.format(servers=self.kafka_bootstrap_servers))
        
        # Revenue aggregation with ROI calculation
        self.t_env.execute_sql("""
            INSERT INTO revenue_analytics
            SELECT 
                TUMBLE_START(event_time, INTERVAL '1' HOUR) as window_start,
                TUMBLE_END(event_time, INTERVAL '1' HOUR) as window_end,
                channel_id,
                SUM(revenue) as total_revenue,
                SUM(cost) as total_cost,
                SUM(revenue) - SUM(cost) as profit,
                calculate_roi(SUM(revenue), SUM(cost)) as roi,
                AVG(revenue) as avg_revenue_per_video,
                COUNT(DISTINCT video_id) as video_count
            FROM revenue_events
            GROUP BY 
                channel_id,
                TUMBLE(event_time, INTERVAL '1' HOUR)
        """)
        
    async def create_anomaly_detection_stream(self):
        """Create anomaly detection streaming job"""
        # Define metrics table for anomaly detection
        self.t_env.execute_sql("""
            CREATE TABLE system_metrics (
                metric_name STRING,
                metric_value FLOAT,
                threshold_min FLOAT,
                threshold_max FLOAT,
                service STRING,
                event_time TIMESTAMP(3),
                WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
            ) WITH (
                'connector' = 'kafka',
                'topic' = 'system-metrics',
                'properties.bootstrap.servers' = '{servers}',
                'properties.group.id' = 'flink-anomaly-detection',
                'format' = 'json',
                'scan.startup.mode' = 'latest-offset'
            )
        """.format(servers=self.kafka_bootstrap_servers))
        
        # Define anomaly alerts sink
        self.t_env.execute_sql("""
            CREATE TABLE anomaly_alerts (
                alert_time TIMESTAMP(3),
                metric_name STRING,
                metric_value FLOAT,
                threshold_violated STRING,
                service STRING,
                severity STRING
            ) WITH (
                'connector' = 'kafka',
                'topic' = 'anomaly-alerts',
                'properties.bootstrap.servers' = '{servers}',
                'format' = 'json'
            )
        """.format(servers=self.kafka_bootstrap_servers))
        
        # Detect anomalies based on thresholds
        self.t_env.execute_sql("""
            INSERT INTO anomaly_alerts
            SELECT 
                event_time as alert_time,
                metric_name,
                metric_value,
                CASE 
                    WHEN metric_value < threshold_min THEN 'MIN_THRESHOLD'
                    WHEN metric_value > threshold_max THEN 'MAX_THRESHOLD'
                    ELSE 'UNKNOWN'
                END as threshold_violated,
                service,
                CASE 
                    WHEN ABS(metric_value - threshold_max) > threshold_max * 0.5 
                        OR ABS(metric_value - threshold_min) > threshold_min * 0.5
                    THEN 'HIGH'
                    ELSE 'MEDIUM'
                END as severity
            FROM system_metrics
            WHERE metric_value < threshold_min OR metric_value > threshold_max
        """)
        
    async def create_user_behavior_stream(self):
        """Create user behavior analytics stream"""
        # Define user action events
        self.t_env.execute_sql("""
            CREATE TABLE user_actions (
                user_id STRING,
                action_type STRING,
                channel_id STRING,
                video_id STRING,
                session_id STRING,
                duration_seconds INT,
                event_time TIMESTAMP(3),
                WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
            ) WITH (
                'connector' = 'kafka',
                'topic' = 'user-actions',
                'properties.bootstrap.servers' = '{servers}',
                'properties.group.id' = 'flink-user-behavior',
                'format' = 'json',
                'scan.startup.mode' = 'latest-offset'
            )
        """.format(servers=self.kafka_bootstrap_servers))
        
        # Define user behavior analytics sink
        self.t_env.execute_sql("""
            CREATE TABLE user_behavior_analytics (
                window_start TIMESTAMP(3),
                window_end TIMESTAMP(3),
                user_id STRING,
                total_actions INT,
                unique_videos INT,
                unique_channels INT,
                total_watch_time INT,
                avg_session_duration FLOAT,
                most_frequent_action STRING
            ) WITH (
                'connector' = 'kafka',
                'topic' = 'user-behavior-analytics',
                'properties.bootstrap.servers' = '{servers}',
                'format' = 'json'
            )
        """.format(servers=self.kafka_bootstrap_servers))
        
        # User behavior aggregation
        self.t_env.execute_sql("""
            INSERT INTO user_behavior_analytics
            SELECT 
                SESSION_START(event_time, INTERVAL '30' MINUTE) as window_start,
                SESSION_END(event_time, INTERVAL '30' MINUTE) as window_end,
                user_id,
                COUNT(*) as total_actions,
                COUNT(DISTINCT video_id) as unique_videos,
                COUNT(DISTINCT channel_id) as unique_channels,
                SUM(duration_seconds) as total_watch_time,
                AVG(duration_seconds) as avg_session_duration,
                MODE() WITHIN GROUP (ORDER BY action_type) as most_frequent_action
            FROM user_actions
            GROUP BY 
                user_id,
                SESSION(event_time, INTERVAL '30' MINUTE)
        """)
        
    async def create_cost_optimization_stream(self):
        """Create cost optimization analytics stream"""
        # Define cost events
        self.t_env.execute_sql("""
            CREATE TABLE cost_events (
                service STRING,
                operation STRING,
                cost FLOAT,
                video_id STRING,
                channel_id STRING,
                model_type STRING,
                event_time TIMESTAMP(3),
                WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
            ) WITH (
                'connector' = 'kafka',
                'topic' = 'cost-events',
                'properties.bootstrap.servers' = '{servers}',
                'properties.group.id' = 'flink-cost-optimization',
                'format' = 'json',
                'scan.startup.mode' = 'latest-offset'
            )
        """.format(servers=self.kafka_bootstrap_servers))
        
        # Define cost analytics sink
        self.t_env.execute_sql("""
            CREATE TABLE cost_analytics (
                window_start TIMESTAMP(3),
                window_end TIMESTAMP(3),
                service STRING,
                total_cost FLOAT,
                avg_cost_per_operation FLOAT,
                most_expensive_operation STRING,
                operation_count BIGINT,
                cost_trend STRING
            ) WITH (
                'connector' = 'kafka',
                'topic' = 'cost-analytics',
                'properties.bootstrap.servers' = '{servers}',
                'format' = 'json'
            )
        """.format(servers=self.kafka_bootstrap_servers))
        
        # Cost optimization analytics
        self.t_env.execute_sql("""
            INSERT INTO cost_analytics
            SELECT 
                TUMBLE_START(event_time, INTERVAL '15' MINUTE) as window_start,
                TUMBLE_END(event_time, INTERVAL '15' MINUTE) as window_end,
                service,
                SUM(cost) as total_cost,
                AVG(cost) as avg_cost_per_operation,
                FIRST_VALUE(operation) as most_expensive_operation,
                COUNT(*) as operation_count,
                CASE 
                    WHEN SUM(cost) > LAG(SUM(cost)) OVER (PARTITION BY service ORDER BY TUMBLE_START(event_time, INTERVAL '15' MINUTE))
                    THEN 'INCREASING'
                    ELSE 'DECREASING'
                END as cost_trend
            FROM cost_events
            GROUP BY 
                service,
                TUMBLE(event_time, INTERVAL '15' MINUTE)
        """)
        
    async def start_all_streams(self):
        """Start all streaming jobs"""
        try:
            # Create all streaming jobs
            await self.create_video_events_stream()
            await self.create_revenue_stream()
            await self.create_anomaly_detection_stream()
            await self.create_user_behavior_stream()
            await self.create_cost_optimization_stream()
            
            # Execute all jobs
            self.env.execute("YTEmpire Streaming Analytics")
            
            logger.info("All streaming jobs started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start streaming jobs: {e}")
            raise
            
    async def get_stream_statistics(self) -> Dict[str, Any]:
        """Get streaming job statistics"""
        stats = {
            "active_jobs": len(self.jobs),
            "total_events_processed": 0,
            "average_latency_ms": 0,
            "checkpoints_completed": 0,
            "last_checkpoint": None,
            "job_details": []
        }
        
        for job_id, job in self.jobs.items():
            job_stats = {
                "job_id": job_id,
                "name": job.name,
                "stream_type": job.stream_type.value,
                "status": "running" if job.is_active else "stopped",
                "window_size": job.window_size,
                "created_at": job.created_at.isoformat()
            }
            stats["job_details"].append(job_stats)
            
        return stats
        
    async def _create_default_jobs(self):
        """Create default streaming jobs"""
        default_jobs = [
            StreamingJob(
                job_id="job_video_analytics",
                name="Video Analytics Stream",
                stream_type=StreamType.VIDEO_EVENTS,
                source_topic="video-events",
                sink_topic="video-analytics",
                window_size=300,  # 5 minutes
                aggregations=["sum", "avg", "count"],
                filters={},
                is_active=True,
                created_at=datetime.utcnow()
            ),
            StreamingJob(
                job_id="job_revenue_analytics",
                name="Revenue Analytics Stream",
                stream_type=StreamType.REVENUE_EVENTS,
                source_topic="revenue-events",
                sink_topic="revenue-analytics",
                window_size=3600,  # 1 hour
                aggregations=["sum", "avg"],
                filters={},
                is_active=True,
                created_at=datetime.utcnow()
            ),
            StreamingJob(
                job_id="job_cost_optimization",
                name="Cost Optimization Stream",
                stream_type=StreamType.COST_EVENTS,
                source_topic="cost-events",
                sink_topic="cost-analytics",
                window_size=900,  # 15 minutes
                aggregations=["sum", "avg", "max"],
                filters={},
                is_active=True,
                created_at=datetime.utcnow()
            )
        ]
        
        for job in default_jobs:
            self.jobs[job.job_id] = job
            
            # Store job metadata in Redis
            await self.redis_client.setex(
                f"streaming:job:{job.job_id}",
                86400 * 30,  # 30 days retention
                json.dumps({
                    "job_id": job.job_id,
                    "name": job.name,
                    "stream_type": job.stream_type.value,
                    "source_topic": job.source_topic,
                    "sink_topic": job.sink_topic,
                    "window_size": job.window_size,
                    "is_active": job.is_active,
                    "created_at": job.created_at.isoformat()
                })
            )


# Singleton instance
flink_streaming = FlinkStreamingAnalytics()


# Docker Compose configuration for Flink deployment
DOCKER_COMPOSE_FLINK = """
version: '3.8'

services:
  jobmanager:
    image: flink:1.17
    ports:
      - "8081:8081"
    command: jobmanager
    environment:
      - JOB_MANAGER_RPC_ADDRESS=jobmanager
    volumes:
      - ./flink-checkpoints:/tmp/flink-checkpoints
      - ./flink-savepoints:/tmp/flink-savepoints

  taskmanager:
    image: flink:1.17
    depends_on:
      - jobmanager
    command: taskmanager
    scale: 2
    environment:
      - JOB_MANAGER_RPC_ADDRESS=jobmanager
    volumes:
      - ./flink-checkpoints:/tmp/flink-checkpoints
      - ./flink-savepoints:/tmp/flink-savepoints

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    ports:
      - "2181:2181"
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
"""


# Save Docker Compose configuration
def save_docker_compose():
    """Save Docker Compose configuration for Flink"""
    import os
    
    streaming_dir = Path(__file__).parent
    docker_compose_path = streaming_dir / "docker-compose.flink.yml"
    
    with open(docker_compose_path, 'w') as f:
        f.write(DOCKER_COMPOSE_FLINK)
        
    print(f"Docker Compose configuration saved to {docker_compose_path}")
    print("To start Flink cluster: docker-compose -f docker-compose.flink.yml up -d")


if __name__ == "__main__":
    # Save Docker Compose configuration
    save_docker_compose()
    
    # Initialize and start streaming
    async def main():
        await flink_streaming.initialize()
        await flink_streaming.start_all_streams()
        
    asyncio.run(main())