# 4.1 DATA ENGINEERING - Implementation Guide

## Pipeline Implementation

### Core Data Pipeline Architecture

```python
class YTEmpireDataPipeline:
    """
    Main data pipeline orchestrator for YTEMPIRE platform
    Handles 250 channels, 500+ videos/day, 10GB+ daily data
    """
    
    def __init__(self):
        self.postgres_conn = psycopg2.connect(
            host="localhost",
            database="ytempire",
            user="ytempire_user",
            password=os.environ["DB_PASSWORD"]
        )
        self.redis_client = redis.Redis(
            host="localhost",
            port=6379,
            decode_responses=True
        )
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    
    async def process_video_generation_event(self, event: dict):
        """
        Process video generation completion events
        """
        try:
            # 1. Validate event data
            validated_data = self.validate_event(event)
            
            # 2. Store in operational database
            video_id = await self.store_video_metadata(validated_data)
            
            # 3. Send to streaming pipeline
            self.kafka_producer.send('video-events', {
                'event_type': 'video_generated',
                'video_id': video_id,
                'channel_id': validated_data['channel_id'],
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # 4. Update cache
            cache_key = f"channel:{validated_data['channel_id']}:latest_video"
            self.redis_client.setex(cache_key, 3600, video_id)
            
            # 5. Trigger analytics update
            await self.update_channel_analytics(validated_data['channel_id'])
            
            return video_id
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            await self.handle_pipeline_error(event, e)
            raise
    
    def validate_event(self, event: dict) -> dict:
        """
        Validate and sanitize event data
        """
        required_fields = ['channel_id', 'video_data', 'timestamp']
        for field in required_fields:
            if field not in event:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate data types and ranges
        if not isinstance(event['channel_id'], str):
            raise TypeError("channel_id must be string")
        
        return event
    
    async def store_video_metadata(self, data: dict) -> str:
        """
        Store video metadata in PostgreSQL
        """
        video_id = str(uuid.uuid4())
        
        query = """
            INSERT INTO ytempire.videos 
            (video_id, channel_id, title, description, status, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING video_id
        """
        
        cursor = self.postgres_conn.cursor()
        cursor.execute(query, (
            video_id,
            data['channel_id'],
            data.get('title', ''),
            data.get('description', ''),
            'processing',
            datetime.utcnow()
        ))
        self.postgres_conn.commit()
        
        return video_id
    
    async def update_channel_analytics(self, channel_id: str):
        """
        Update channel analytics in real-time
        """
        # Increment video count
        query = """
            UPDATE ytempire.channels 
            SET video_count = video_count + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE channel_id = %s
        """
        
        cursor = self.postgres_conn.cursor()
        cursor.execute(query, (channel_id,))
        self.postgres_conn.commit()
        
        # Clear cache to force refresh
        self.redis_client.delete(f"channel:{channel_id}:stats")
    
    async def handle_pipeline_error(self, event: dict, error: Exception):
        """
        Handle pipeline errors with retry logic
        """
        error_record = {
            'event': event,
            'error': str(error),
            'timestamp': datetime.utcnow().isoformat(),
            'retry_count': event.get('retry_count', 0)
        }
        
        # Log to error queue for manual review
        self.kafka_producer.send('error-queue', error_record)
        
        # Send alert if critical
        if error_record['retry_count'] > 3:
            await self.send_critical_alert(error_record)
```

### YouTube Data Collection Pipeline

```python
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import asyncio

class YouTubeDataCollector:
    """
    Manages YouTube API data collection with quota optimization
    """
    
    def __init__(self):
        self.youtube = build('youtube', 'v3', developerKey=os.environ['YOUTUBE_API_KEY'])
        self.quota_tracker = QuotaTracker(daily_limit=10000)
        self.cache_ttl = 3600  # 1 hour cache
        
    async def collect_channel_analytics(self, channel_id: str) -> dict:
        """
        Collect comprehensive analytics for a channel
        """
        # Check quota availability
        if not self.quota_tracker.can_make_request(cost=3):
            logger.warning(f"Quota exceeded for channel {channel_id}")
            return await self.get_cached_data(channel_id)
        
        try:
            # Fetch channel statistics
            channel_response = self.youtube.channels().list(
                part='statistics,snippet,contentDetails',
                id=channel_id
            ).execute()
            
            # Fetch recent videos
            videos_response = self.youtube.search().list(
                part='id,snippet',
                channelId=channel_id,
                maxResults=10,
                order='date',
                type='video'
            ).execute()
            
            # Get video statistics
            video_ids = ','.join([v['id']['videoId'] for v in videos_response['items']])
            video_stats = self.youtube.videos().list(
                part='statistics,contentDetails',
                id=video_ids
            ).execute()
            
            # Process and structure data
            analytics_data = {
                'channel_id': channel_id,
                'timestamp': datetime.utcnow(),
                'subscribers': int(channel_response['items'][0]['statistics']['subscriberCount']),
                'total_views': int(channel_response['items'][0]['statistics']['viewCount']),
                'video_count': int(channel_response['items'][0]['statistics']['videoCount']),
                'recent_videos': self.process_video_stats(video_stats['items'])
            }
            
            # Store in database
            await self.store_analytics(analytics_data)
            
            # Cache the data
            await self.cache_data(channel_id, analytics_data)
            
            # Update quota usage
            self.quota_tracker.use_quota(3)
            
            return analytics_data
            
        except HttpError as e:
            if e.resp.status == 403:  # Quota exceeded
                return await self.handle_quota_exceeded(channel_id)
            raise
    
    def process_video_stats(self, video_items: list) -> list:
        """
        Process raw video statistics
        """
        processed_videos = []
        
        for item in video_items:
            processed = {
                'video_id': item['id'],
                'title': item['snippet']['title'],
                'views': int(item['statistics'].get('viewCount', 0)),
                'likes': int(item['statistics'].get('likeCount', 0)),
                'comments': int(item['statistics'].get('commentCount', 0)),
                'duration': self.parse_duration(item['contentDetails']['duration'])
            }
            processed_videos.append(processed)
        
        return processed_videos
    
    def parse_duration(self, duration_str: str) -> int:
        """
        Parse ISO 8601 duration to seconds
        """
        import isodate
        duration = isodate.parse_duration(duration_str)
        return int(duration.total_seconds())
    
    async def get_cached_data(self, channel_id: str) -> dict:
        """
        Retrieve cached channel data
        """
        cache_key = f"channel_analytics:{channel_id}"
        cached = self.redis_client.get(cache_key)
        
        if cached:
            return json.loads(cached)
        
        # Return minimal data if no cache
        return {
            'channel_id': channel_id,
            'cached': True,
            'timestamp': datetime.utcnow()
        }
    
    async def cache_data(self, channel_id: str, data: dict):
        """
        Cache channel analytics data
        """
        cache_key = f"channel_analytics:{channel_id}"
        self.redis_client.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(data, default=str)
        )
    
    async def handle_quota_exceeded(self, channel_id: str) -> dict:
        """
        Handle YouTube API quota exceeded scenario
        """
        logger.error(f"YouTube quota exceeded for channel {channel_id}")
        
        # Try to get from cache first
        cached = await self.get_cached_data(channel_id)
        if cached and not cached.get('cached'):
            return cached
        
        # Use YouTube Reporting API as fallback (zero quota cost)
        return await self.get_reporting_api_data(channel_id)

class QuotaTracker:
    """
    Track and manage API quota usage
    """
    
    def __init__(self, daily_limit: int):
        self.daily_limit = daily_limit
        self.redis_client = redis.Redis(host='localhost', port=6379)
        
    def can_make_request(self, cost: int) -> bool:
        """
        Check if request can be made within quota
        """
        today = datetime.utcnow().strftime('%Y-%m-%d')
        key = f"youtube_quota:{today}"
        
        current_usage = int(self.redis_client.get(key) or 0)
        return (current_usage + cost) <= self.daily_limit
    
    def use_quota(self, cost: int):
        """
        Record quota usage
        """
        today = datetime.utcnow().strftime('%Y-%m-%d')
        key = f"youtube_quota:{today}"
        
        self.redis_client.incrby(key, cost)
        self.redis_client.expire(key, 86400)  # Expire after 24 hours
```

### Real-time Streaming Pipeline

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

class StreamingAnalyticsPipeline:
    """
    Real-time analytics processing using Spark Structured Streaming
    """
    
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("YTEmpire-Streaming") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.streaming.stateStore.stateSchemaCheck", "false") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
    
    def create_streaming_job(self):
        """
        Create and start the streaming analytics job
        """
        # Define schema for incoming events
        event_schema = StructType([
            StructField("event_type", StringType(), True),
            StructField("channel_id", StringType(), True),
            StructField("video_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("metrics", MapType(StringType(), DoubleType()), True)
        ])
        
        # Read from Kafka stream
        df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("subscribe", "video-events,channel-events") \
            .option("startingOffsets", "latest") \
            .option("maxOffsetsPerTrigger", 10000) \
            .load()
        
        # Parse JSON data
        parsed_df = df.select(
            from_json(col("value").cast("string"), event_schema).alias("data")
        ).select("data.*")
        
        # Calculate windowed aggregations
        windowed_stats = parsed_df \
            .withWatermark("timestamp", "1 minute") \
            .groupBy(
                window(col("timestamp"), "5 minutes", "1 minute"),
                col("channel_id")
            ) \
            .agg(
                count("*").alias("event_count"),
                avg("metrics.views").alias("avg_views"),
                sum("metrics.revenue").alias("total_revenue"),
                max("timestamp").alias("last_event_time")
            )
        
        # Write to multiple sinks
        # Sink 1: PostgreSQL for persistence
        query1 = windowed_stats.writeStream \
            .outputMode("append") \
            .foreachBatch(self.write_to_postgres) \
            .trigger(processingTime='30 seconds') \
            .option("checkpointLocation", "/tmp/checkpoint/postgres") \
            .start()
        
        # Sink 2: In-memory table for real-time queries
        query2 = windowed_stats.writeStream \
            .outputMode("complete") \
            .format("memory") \
            .queryName("real_time_stats") \
            .trigger(processingTime='10 seconds') \
            .start()
        
        return [query1, query2]
    
    def write_to_postgres(self, batch_df, batch_id):
        """
        Write streaming batch to PostgreSQL
        """
        try:
            # Add batch metadata
            enriched_df = batch_df.withColumn("batch_id", lit(batch_id)) \
                                  .withColumn("processed_at", current_timestamp())
            
            # Write to PostgreSQL
            enriched_df.write \
                .format("jdbc") \
                .option("url", "jdbc:postgresql://localhost:5432/ytempire") \
                .option("dbtable", "streaming_analytics") \
                .option("user", "ytempire_user") \
                .option("password", os.environ["DB_PASSWORD"]) \
                .option("driver", "org.postgresql.Driver") \
                .mode("append") \
                .save()
            
            logger.info(f"Batch {batch_id} written to PostgreSQL: {batch_df.count()} records")
            
        except Exception as e:
            logger.error(f"Error writing batch {batch_id}: {e}")
            raise
    
    def query_real_time_stats(self, channel_id: str = None):
        """
        Query real-time statistics from memory table
        """
        query = "SELECT * FROM real_time_stats"
        
        if channel_id:
            query += f" WHERE channel_id = '{channel_id}'"
        
        query += " ORDER BY window.start DESC"
        
        return self.spark.sql(query).toPandas()
```

## ETL Processes

### Daily ETL Pipeline

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import pandas as pd

# DAG Configuration
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['data-team@ytempire.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'ytempire_daily_etl',
    default_args=default_args,
    description='Daily ETL for YouTube channel analytics',
    schedule_interval='0 2 * * *',  # 2 AM daily
    catchup=False,
    max_active_runs=1,
    tags=['production', 'daily', 'etl']
)

def extract_youtube_data(**context):
    """
    Extract data from YouTube API
    """
    collector = YouTubeDataCollector()
    channels = get_active_channels()
    
    extracted_count = 0
    failed_channels = []
    
    for channel in channels:
        try:
            # Collect channel analytics
            analytics = await collector.collect_channel_analytics(channel['youtube_channel_id'])
            
            # Store raw data
            store_raw_data(analytics, context['ds'])
            
            # Send to processing queue
            send_to_processing_queue(analytics)
            
            extracted_count += 1
            
        except Exception as e:
            logger.error(f"Failed to extract data for channel {channel['channel_id']}: {e}")
            failed_channels.append(channel['channel_id'])
            continue
    
    # Store extraction metadata
    context['task_instance'].xcom_push(
        key='extraction_stats',
        value={
            'channels_processed': extracted_count,
            'failed_channels': failed_channels,
            'total_channels': len(channels)
        }
    )
    
    return {"channels_processed": extracted_count}

def transform_analytics_data(**context):
    """
    Transform raw analytics data
    """
    # Get extraction stats
    extraction_stats = context['task_instance'].xcom_pull(
        task_ids='extract_youtube_data',
        key='extraction_stats'
    )
    
    # Load raw data
    raw_data = load_raw_data(context['ds'])
    
    # Initialize transformer
    transformer = DataTransformer()
    
    # Apply transformations
    transformed_data = []
    for record in raw_data:
        try:
            transformed = {
                'channel_id': record['channel_id'],
                'date': context['ds'],
                'subscribers': int(record['subscribers']),
                'views_daily': transformer.calculate_daily_views(record),
                'engagement_rate': transformer.calculate_engagement_rate(record),
                'revenue_estimate': transformer.estimate_revenue(record),
                'growth_rate': transformer.calculate_growth_rate(record, get_historical_data(record['channel_id']))
            }
            transformed_data.append(transformed)
        except Exception as e:
            logger.error(f"Transformation error for channel {record['channel_id']}: {e}")
            continue
    
    # Validate transformed data
    validated_data = validate_transformations(transformed_data)
    
    # Store transformed data
    store_transformed_data(validated_data, context['ds'])
    
    return {"records_transformed": len(validated_data)}

def load_to_warehouse(**context):
    """
    Load transformed data to data warehouse
    """
    from psycopg2.extras import execute_values
    
    # Load transformed data
    data = load_transformed_data(context['ds'])
    
    if not data:
        logger.warning("No data to load to warehouse")
        return {"records_loaded": 0}
    
    # Bulk insert to PostgreSQL
    conn = psycopg2.connect(
        host="localhost",
        database="ytempire",
        user="ytempire_user",
        password=os.environ["DB_PASSWORD"]
    )
    
    cursor = conn.cursor()
    
    try:
        # Prepare data for bulk insert
        values = [
            (
                d['channel_id'],
                d['date'],
                d['subscribers'],
                d['views_daily'],
                d['engagement_rate'],
                d['revenue_estimate'],
                d['growth_rate']
            )
            for d in data
        ]
        
        # Prepare bulk insert query
        insert_query = """
            INSERT INTO analytics.channel_daily_metrics 
            (channel_id, date, subscribers, views_daily, engagement_rate, revenue_estimate, growth_rate)
            VALUES %s
            ON CONFLICT (channel_id, date) 
            DO UPDATE SET
                subscribers = EXCLUDED.subscribers,
                views_daily = EXCLUDED.views_daily,
                engagement_rate = EXCLUDED.engagement_rate,
                revenue_estimate = EXCLUDED.revenue_estimate,
                growth_rate = EXCLUDED.growth_rate,
                updated_at = CURRENT_TIMESTAMP
        """
        
        # Execute bulk insert
        execute_values(cursor, insert_query, values)
        conn.commit()
        
        logger.info(f"Successfully loaded {len(data)} records to warehouse")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to load data to warehouse: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
    
    return {"records_loaded": len(data)}

def send_daily_report(**context):
    """
    Send daily ETL report
    """
    stats = context['task_instance'].xcom_pull(task_ids='extract_youtube_data', key='extraction_stats')
    
    report = f"""
    Daily ETL Report - {context['ds']}
    
    Extraction Stats:
    - Total Channels: {stats['total_channels']}
    - Processed: {stats['channels_processed']}
    - Failed: {len(stats['failed_channels'])}
    
    Failed Channels: {', '.join(stats['failed_channels']) if stats['failed_channels'] else 'None'}
    
    Please check the logs for more details.
    """
    
    return report

# Define tasks
extract_task = PythonOperator(
    task_id='extract_youtube_data',
    python_callable=extract_youtube_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_analytics_data',
    python_callable=transform_analytics_data,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_to_warehouse',
    python_callable=load_to_warehouse,
    dag=dag
)

# Create materialized views
update_views = PostgresOperator(
    task_id='update_materialized_views',
    postgres_conn_id='postgres_default',
    sql="""
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_channel_performance;
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_video_analytics;
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_revenue_summary;
    """,
    dag=dag
)

# Send report
report_task = PythonOperator(
    task_id='send_daily_report',
    python_callable=send_daily_report,
    dag=dag,
    trigger_rule='all_done'  # Run even if upstream tasks fail
)

# Set task dependencies
extract_task >> transform_task >> load_task >> update_views >> report_task
```

### Data Transformation Functions

```python
class DataTransformer:
    """
    Data transformation utilities for YTEMPIRE
    """
    
    @staticmethod
    def calculate_engagement_rate(video_data: dict) -> float:
        """
        Calculate engagement rate from video metrics
        """
        views = video_data.get('views', 0)
        if views == 0:
            return 0.0
        
        engagements = (
            video_data.get('likes', 0) +
            video_data.get('comments', 0) +
            video_data.get('shares', 0)
        )
        
        return round(engagements / views * 100, 2)
    
    @staticmethod
    def estimate_revenue(channel_data: dict) -> float:
        """
        Estimate revenue based on views and engagement
        """
        views = channel_data.get('views_daily', 0)
        engagement_rate = channel_data.get('engagement_rate', 0)
        
        # Base CPM rates (Cost Per Mille - per 1000 views)
        base_cpm = 2.0  # $2 per 1000 views baseline
        
        # Adjust CPM based on engagement rate
        if engagement_rate > 5:
            cpm = base_cpm * 1.5  # 50% premium for high engagement
        elif engagement_rate > 2:
            cpm = base_cpm * 1.2  # 20% premium for moderate engagement
        else:
            cpm = base_cpm
        
        # Calculate ad revenue
        ad_revenue = (views / 1000) * cpm
        
        # Add affiliate revenue estimate (10% of ad revenue)
        affiliate_revenue = ad_revenue * 0.1
        
        # Add sponsorship estimate (for channels with high views)
        sponsorship_revenue = 0
        if views > 10000:
            sponsorship_revenue = views * 0.001  # $0.001 per view for sponsorships
        
        total_revenue = ad_revenue + affiliate_revenue + sponsorship_revenue
        
        return round(total_revenue, 2)
    
    @staticmethod
    def calculate_growth_rate(channel_data: dict, historical_data: dict) -> float:
        """
        Calculate channel growth rate
        """
        if not historical_data:
            return 0.0
            
        current_subs = channel_data.get('subscribers', 0)
        previous_subs = historical_data.get('subscribers', 0)
        
        if previous_subs == 0:
            return 100.0 if current_subs > 0 else 0.0
        
        growth = ((current_subs - previous_subs) / previous_subs) * 100
        return round(growth, 2)
    
    @staticmethod
    def calculate_daily_views(channel_data: dict) -> int:
        """
        Calculate daily views from channel data
        """
        # If we have recent videos data, sum their views
        if 'recent_videos' in channel_data:
            daily_views = sum(video.get('views', 0) for video in channel_data['recent_videos'])
            return daily_views
        
        # Otherwise estimate from total views and video count
        total_views = channel_data.get('total_views', 0)
        video_count = channel_data.get('video_count', 1)
        
        # Rough estimate: average views per video / 30 days
        if video_count > 0:
            avg_views_per_video = total_views / video_count
            daily_estimate = int(avg_views_per_video / 30)
            return daily_estimate
        
        return 0
    
    @staticmethod
    def normalize_metrics(metrics: dict) -> dict:
        """
        Normalize metrics for consistency
        """
        normalized = {}
        
        for key, value in metrics.items():
            if value is None:
                normalized[key] = 0
            elif isinstance(value, str) and value.isdigit():
                normalized[key] = int(value)
            elif isinstance(value, float):
                normalized[key] = round(value, 2)
            else:
                normalized[key] = value
        
        return normalized
```

## Streaming Architecture

### N8N Workflow Implementation

```javascript
// N8N Workflow Configuration for Real-time Processing
{
  "name": "YTEmpire Real-time Analytics",
  "nodes": [
    {
      "name": "Video Upload Trigger",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "video-uploaded",
        "method": "POST",
        "responseMode": "onReceived"
      },
      "position": [250, 300]
    },
    {
      "name": "Validate Webhook Data",
      "type": "n8n-nodes-base.if",
      "parameters": {
        "conditions": {
          "string": [
            {
              "value1": "={{$json[\"video_id\"]}}",
              "operation": "isNotEmpty"
            }
          ]
        }
      },
      "position": [450, 300]
    },
    {
      "name": "Fetch Video Metrics",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "https://www.googleapis.com/youtube/v3/videos",
        "method": "GET",
        "queryParameters": {
          "parameters": [
            {
              "name": "id",
              "value": "={{$json[\"video_id\"]}}"
            },
            {
              "name": "part",
              "value": "statistics,contentDetails"
            },
            {
              "name": "key",
              "value": "{{$env.YOUTUBE_API_KEY}}"
            }
          ]
        }
      },
      "position": [650, 300]
    },
    {
      "name": "Store in PostgreSQL",
      "type": "n8n-nodes-base.postgres",
      "parameters": {
        "operation": "insert",
        "schema": "ytempire",
        "table": "video_metrics",
        "columns": "video_id,views,likes,comments,timestamp",
        "returnFields": "video_id"
      },
      "position": [850, 300]
    },
    {
      "name": "Update Redis Cache",
      "type": "n8n-nodes-base.redis",
      "parameters": {
        "operation": "set",
        "key": "video:{{$json[\"video_id\"]}}:metrics",
        "value": "={{JSON.stringify($json)}}",
        "expire": true,
        "ttl": 3600
      },
      "position": [1050, 300]
    },
    {
      "name": "Send to Analytics Queue",
      "type": "n8n-nodes-base.kafka",
      "parameters": {
        "topic": "video-analytics",
        "message": "={{JSON.stringify($json)}}",
        "options": {
          "compression": "gzip"
        }
      },
      "position": [1250, 300]
    }
  ],
  "connections": {
    "Video Upload Trigger": {
      "main": [
        [
          {
            "node": "Validate Webhook Data",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Validate Webhook Data": {
      "main": [
        [
          {
            "node": "Fetch Video Metrics",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Fetch Video Metrics": {
      "main": [
        [
          {
            "node": "Store in PostgreSQL",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Store in PostgreSQL": {
      "main": [
        [
          {
            "node": "Update Redis Cache",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Update Redis Cache": {
      "main": [
        [
          {
            "node": "Send to Analytics Queue",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

### Event Streaming Pipeline

```python
from kafka import KafkaConsumer, KafkaProducer
import json
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EventStreamProcessor:
    """
    Process real-time events from multiple sources
    """
    
    def __init__(self):
        self.kafka_consumer = KafkaConsumer(
            'video-events',
            'channel-events',
            'user-events',
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='latest',
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='event-processor-group',
            enable_auto_commit=False
        )
        
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.event_handlers = {
            'video_uploaded': self.handle_video_upload,
            'video_published': self.handle_video_publish,
            'metrics_update': self.handle_metrics_update,
            'quality_check': self.handle_quality_check
        }
        
        self.redis_client = redis.Redis(host='localhost', port=6379)
        self.postgres_conn = psycopg2.connect(
            host="localhost",
            database="ytempire",
            user="ytempire_user",
            password=os.environ["DB_PASSWORD"]
        )
    
    async def process_events(self):
        """
        Main event processing loop
        """
        logger.info("Starting event stream processor")
        
        for message in self.kafka_consumer:
            try:
                event = message.value
                event_type = event.get('event_type')
                
                logger.debug(f"Processing event: {event_type}")
                
                if event_type in self.event_handlers:
                    await self.event_handlers[event_type](event)
                else:
                    logger.warning(f"Unknown event type: {event_type}")
                
                # Commit offset after successful processing
                self.kafka_consumer.commit()
                
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                await self.handle_error(message, e)
    
    async def handle_video_upload(self, event: dict):
        """
        Process video upload events
        """
        video_id = event['video_id']
        channel_id = event['channel_id']
        
        logger.info(f"Processing video upload: {video_id}")
        
        # Update real-time dashboard
        await self.update_dashboard_metrics(channel_id, {
            'latest_video': video_id,
            'upload_time': event['timestamp'],
            'status': 'processing'
        })
        
        # Trigger quality check
        await self.trigger_quality_check(video_id)
        
        # Update channel statistics
        await self.increment_channel_video_count(channel_id)
        
        # Send notification
        self.kafka_producer.send('notifications', {
            'type': 'video_uploaded',
            'channel_id': channel_id,
            'video_id': video_id,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def handle_video_publish(self, event: dict):
        """
        Process video publish events
        """
        video_id = event['video_id']
        youtube_id = event['youtube_video_id']
        
        # Update video status
        query = """
            UPDATE ytempire.videos 
            SET status = 'published',
                youtube_video_id = %s,
                published_at = %s
            WHERE video_id = %s
        """
        
        cursor = self.postgres_conn.cursor()
        cursor.execute(query, (youtube_id, datetime.utcnow(), video_id))
        self.postgres_conn.commit()
        
        logger.info(f"Video published: {video_id} -> {youtube_id}")
    
    async def handle_metrics_update(self, event: dict):
        """
        Process metrics update events
        """
        metrics = event['metrics']
        
        # Store in time-series database
        await self.store_timeseries_data({
            'timestamp': event['timestamp'],
            'video_id': event['video_id'],
            'views': metrics.get('views', 0),
            'likes': metrics.get('likes', 0),
            'comments': metrics.get('comments', 0),
            'watch_time': metrics.get('watch_time_minutes', 0)
        })
        
        # Update aggregations
        await self.update_channel_aggregations(event['channel_id'], metrics)
        
        # Check for viral indicators
        if metrics.get('views', 0) > 10000:
            await self.trigger_viral_alert(event['video_id'], metrics)
    
    async def handle_quality_check(self, event: dict):
        """
        Process quality check events
        """
        video_id = event['video_id']
        quality_score = event['quality_score']
        
        # Update quality score
        query = """
            UPDATE ytempire.videos 
            SET quality_score = %s
            WHERE video_id = %s
        """
        
        cursor = self.postgres_conn.cursor()
        cursor.execute(query, (quality_score, video_id))
        self.postgres_conn.commit()
        
        # Take action based on score
        if quality_score < 85:
            logger.warning(f"Low quality score for video {video_id}: {quality_score}")
            await self.trigger_quality_improvement(video_id)
    
    async def update_dashboard_metrics(self, channel_id: str, metrics: dict):
        """
        Update real-time dashboard metrics in Redis
        """
        key = f"dashboard:channel:{channel_id}"
        self.redis_client.hset(key, mapping=metrics)
        self.redis_client.expire(key, 3600)  # 1 hour TTL
    
    async def store_timeseries_data(self, data: dict):
        """
        Store time-series data in PostgreSQL/TimescaleDB
        """
        query = """
            INSERT INTO ytempire.video_metrics 
            (time, video_id, views, likes, comments, watch_time_minutes)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        cursor = self.postgres_conn.cursor()
        cursor.execute(query, (
            data['timestamp'],
            data['video_id'],
            data['views'],
            data['likes'],
            data['comments'],
            data['watch_time']
        ))
        self.postgres_conn.commit()
    
    async def handle_error(self, message, error: Exception):
        """
        Handle processing errors
        """
        error_record = {
            'message': str(message.value),
            'error': str(error),
            'timestamp': datetime.utcnow().isoformat(),
            'topic': message.topic,
            'partition': message.partition,
            'offset': message.offset
        }
        
        # Send to error queue
        self.kafka_producer.send('error-queue', error_record)
        
        # Log error
        logger.error(f"Error processing message: {error_record}")
        
        # Skip message and continue
        self.kafka_consumer.commit()
```

## Data Quality Framework

### Data Validation Pipeline

```python
from great_expectations import DataContext
from great_expectations.core.batch import RuntimeBatchRequest
import pandas as pd
from typing import Dict, List

class DataQualityValidator:
    """
    Comprehensive data quality validation using Great Expectations
    """
    
    def __init__(self):
        self.context = DataContext()
        self.setup_expectations()
        self.redis_client = redis.Redis(host='localhost', port=6379)
    
    def setup_expectations(self):
        """
        Define data quality expectations
        """
        # Create expectation suite for video data
        video_suite = self.context.create_expectation_suite(
            expectation_suite_name="video_data_quality",
            overwrite_existing=True
        )
        
        # Add expectations
        video_expectations = [
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "video_id"}
            },
            {
                "expectation_type": "expect_column_values_to_be_unique",
                "kwargs": {"column": "video_id"}
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "duration_seconds",
                    "min_value": 30,
                    "max_value": 3600
                }
            },
            {
                "expectation_type": "expect_column_values_to_match_regex",
                "kwargs": {
                    "column": "youtube_video_id",
                    "regex": "^[a-zA-Z0-9_-]{11}$"
                }
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "quality_score",
                    "min_value": 0,
                    "max_value": 100
                }
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "generation_cost",
                    "min_value": 0,
                    "max_value": 3.0  # Max $3 per video
                }
            }
        ]
        
        for expectation in video_expectations:
            video_suite.add_expectation(expectation_configuration=expectation)
        
        self.context.save_expectation_suite(video_suite)
        
        # Create channel data expectations
        channel_suite = self.context.create_expectation_suite(
            expectation_suite_name="channel_data_quality",
            overwrite_existing=True
        )
        
        channel_expectations = [
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "channel_id"}
            },
            {
                "expectation_type": "expect_column_values_to_match_regex",
                "kwargs": {
                    "column": "youtube_channel_id",
                    "regex": "^UC[a-zA-Z0-9_-]{22}$"
                }
            },
            {
                "expectation_type": "expect_column_values_to_be_in_set",
                "kwargs": {
                    "column": "status",
                    "value_set": ["active", "paused", "archived"]
                }
            }
        ]
        
        for expectation in channel_expectations:
            channel_suite.add_expectation(expectation_configuration=expectation)
        
        self.context.save_expectation_suite(channel_suite)
    
    def validate_batch(self, df: pd.DataFrame, suite_name: str) -> dict:
        """
        Validate a batch of data against expectations
        """
        # Create batch request
        batch_request = RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="runtime_data_connector",
            data_asset_name="df",
            runtime_parameters={"batch_data": df},
            batch_identifiers={
                "run_id": f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
        )
        
        # Run validation
        validator = self.context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name
        )
        
        validation_result = validator.validate()
        
        # Process results
        if not validation_result.success:
            self.handle_validation_failure(validation_result)
        
        # Store validation metrics
        self.store_validation_metrics(validation_result)
        
        return {
            "success": validation_result.success,
            "statistics": validation_result.statistics,
            "failed_expectations": len([r for r in validation_result.results if not r.success]),
            "total_expectations": len(validation_result.results)
        }
    
    def handle_validation_failure(self, validation_result):
        """
        Handle data quality validation failures
        """
        failed_expectations = [
            r for r in validation_result.results if not r.success
        ]
        
        for failure in failed_expectations:
            logger.error(f"Validation failed: {failure.expectation_config}")
            
            # Send alert
            self.send_quality_alert({
                "expectation": failure.expectation_config.expectation_type,
                "kwargs": failure.expectation_config.kwargs,
                "result": failure.result
            })
    
    def store_validation_metrics(self, validation_result):
        """
        Store validation metrics for monitoring
        """
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'success': validation_result.success,
            'total_expectations': len(validation_result.results),
            'failed_expectations': len([r for r in validation_result.results if not r.success])
        }
        
        # Store in Redis for real-time monitoring
        key = f"validation:metrics:{datetime.utcnow().strftime('%Y%m%d')}"
        self.redis_client.lpush(key, json.dumps(metrics))
        self.redis_client.expire(key, 86400 * 7)  # Keep for 7 days
    
    def send_quality_alert(self, alert_data: dict):
        """
        Send data quality alerts
        """
        alert = {
            'type': 'data_quality_failure',
            'severity': 'high',
            'details': alert_data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to alert queue
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        producer.send('alerts', alert)
        producer.close()
```

### Data Quality Monitoring

```python
from prometheus_client import Gauge, Counter
import asyncio

class DataQualityMonitor:
    """
    Real-time data quality monitoring
    """
    
    def __init__(self):
        self.metrics = {
            'completeness': Gauge('data_completeness', 'Percentage of non-null values'),
            'accuracy': Gauge('data_accuracy', 'Percentage of accurate values'),
            'consistency': Gauge('data_consistency', 'Data consistency score'),
            'timeliness': Gauge('data_timeliness', 'Data freshness in minutes'),
            'uniqueness': Gauge('data_uniqueness', 'Percentage of unique records'),
            'validity': Gauge('data_validity', 'Percentage of valid records')
        }
        
        self.postgres_conn = psycopg2.connect(
            host="localhost",
            database="ytempire",
            user="ytempire_user",
            password=os.environ["DB_PASSWORD"]
        )
    
    async def calculate_quality_scores(self):
        """
        Calculate and update data quality scores
        """
        while True:
            try:
                # Calculate completeness
                completeness_score = await self.calculate_completeness()
                self.metrics['completeness'].set(completeness_score)
                
                # Calculate accuracy
                accuracy_score = await self.calculate_accuracy()
                self.metrics['accuracy'].set(accuracy_score)
                
                # Calculate consistency
                consistency_score = await self.calculate_consistency()
                self.metrics['consistency'].set(consistency_score)
                
                # Calculate timeliness
                timeliness_score = await self.calculate_timeliness()
                self.metrics['timeliness'].set(timeliness_score)
                
                # Calculate uniqueness
                uniqueness_score = await self.calculate_uniqueness()
                self.metrics['uniqueness'].set(uniqueness_score)
                
                # Calculate validity
                validity_score = await self.calculate_validity()
                self.metrics['validity'].set(validity_score)
                
                # Overall quality score (weighted average)
                overall_score = (
                    completeness_score * 0.20 +
                    accuracy_score * 0.25 +
                    consistency_score * 0.20 +
                    timeliness_score * 0.15 +
                    uniqueness_score * 0.10 +
                    validity_score * 0.10
                )
                
                logger.info(f"Overall data quality score: {overall_score:.2f}%")
                
                # Alert if quality drops below threshold
                if overall_score < 85:
                    await self.trigger_quality_alert(overall_score)
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error calculating quality scores: {e}")
                await asyncio.sleep(60)
    
    async def calculate_completeness(self) -> float:
        """
        Calculate data completeness score
        """
        query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(video_id) as non_null_video_id,
                COUNT(channel_id) as non_null_channel_id,
                COUNT(title) as non_null_title,
                COUNT(youtube_video_id) as non_null_youtube_id
            FROM ytempire.videos
            WHERE created_at >= NOW() - INTERVAL '1 day'
        """
        
        cursor = self.postgres_conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        
        if result[0] == 0:  # No records
            return 100.0
        
        # Calculate percentage of non-null values
        non_null_count = sum(result[1:])
        total_possible = result[0] * 4  # 4 required fields
        
        completeness = (non_null_count / total_possible) * 100
        
        return round(completeness, 2)
    
    async def calculate_accuracy(self) -> float:
        """
        Calculate data accuracy score
        """
        query = """
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN quality_score BETWEEN 0 AND 100 THEN 1 END) as valid_scores,
                COUNT(CASE WHEN generation_cost BETWEEN 0 AND 3 THEN 1 END) as valid_costs,
                COUNT(CASE WHEN duration_seconds BETWEEN 30 AND 3600 THEN 1 END) as valid_duration
            FROM ytempire.videos
            WHERE created_at >= NOW() - INTERVAL '1 day'
        """
        
        cursor = self.postgres_conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        
        if result[0] == 0:
            return 100.0
        
        # Calculate accuracy percentage
        accurate_count = sum(result[1:])
        total_checks = result[0] * 3
        
        accuracy = (accurate_count / total_checks) * 100
        
        return round(accuracy, 2)
    
    async def calculate_timeliness(self) -> float:
        """
        Calculate data timeliness (freshness)
        """
        query = """
            SELECT 
                MAX(updated_at) as last_update,
                NOW() as current_time
            FROM ytempire.video_metrics
        """
        
        cursor = self.postgres_conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        
        if not result[0]:
            return 0.0
        
        # Calculate minutes since last update
        time_diff = (result[1] - result[0]).total_seconds() / 60
        
        # Score based on freshness (100% if < 5 min, 0% if > 60 min)
        if time_diff <= 5:
            return 100.0
        elif time_diff >= 60:
            return 0.0
        else:
            return round(100 - ((time_diff - 5) / 55 * 100), 2)
    
    async def trigger_quality_alert(self, score: float):
        """
        Trigger alert for low data quality
        """
        alert = {
            'type': 'low_data_quality',
            'severity': 'high' if score < 70 else 'medium',
            'score': score,
            'timestamp': datetime.utcnow().isoformat(),
            'message': f"Data quality score dropped to {score:.2f}%"
        }
        
        logger.warning(f"Data quality alert: {alert}")
        
        # Send to monitoring system
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        producer.send('alerts', alert)
        producer.close()
```

### Data Anomaly Detection

```python
from scipy import stats
import numpy as np
from collections import deque

class AnomalyDetector:
    """
    Detect anomalies in data pipelines
    """
    
    def __init__(self):
        self.zscore_threshold = 3
        self.historical_data = {}
        self.window_size = 100
        self.min_samples = 30
    
    def detect_anomalies(self, metric_name: str, value: float) -> bool:
        """
        Detect if a metric value is anomalous using Z-score method
        """
        # Initialize history if needed
        if metric_name not in self.historical_data:
            self.historical_data[metric_name] = deque(maxlen=self.window_size)
        
        history = self.historical_data[metric_name]
        
        # Need minimum samples for reliable detection
        if len(history) < self.min_samples:
            history.append(value)
            return False
        
        # Calculate statistics
        history_array = np.array(history)
        mean = np.mean(history_array)
        std = np.std(history_array)
        
        if std == 0:
            # No variation in historical data
            is_anomaly = value != mean
        else:
            # Calculate z-score
            z_score = abs((value - mean) / std)
            is_anomaly = z_score > self.zscore_threshold
        
        if is_anomaly:
            logger.warning(f"Anomaly detected for {metric_name}: value={value}, mean={mean:.2f}, std={std:.2f}")
            self.handle_anomaly(metric_name, value, mean, std)
        
        # Add to history
        history.append(value)
        
        return is_anomaly
    
    def detect_pattern_anomaly(self, metric_name: str, values: List[float]) -> bool:
        """
        Detect anomalies in patterns or sequences
        """
        if len(values) < 10:
            return False
        
        # Check for sudden spikes or drops
        diffs = np.diff(values)
        mean_diff = np.mean(np.abs(diffs))
        max_diff = np.max(np.abs(diffs))
        
        # Anomaly if max difference is >5x the mean difference
        if max_diff > 5 * mean_diff:
            logger.warning(f"Pattern anomaly detected for {metric_name}")
            return True
        
        # Check for trend reversal
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        trend_first = np.polyfit(range(len(first_half)), first_half, 1)[0]
        trend_second = np.polyfit(range(len(second_half)), second_half, 1)[0]
        
        # Anomaly if trend reverses significantly
        if np.sign(trend_first) != np.sign(trend_second) and abs(trend_first - trend_second) > mean_diff:
            logger.warning(f"Trend reversal detected for {metric_name}")
            return True
        
        return False
    
    def handle_anomaly(self, metric_name: str, value: float, mean: float, std: float):
        """
        Handle detected anomalies
        """
        alert = {
            'type': 'data_anomaly',
            'metric': metric_name,
            'value': value,
            'expected_range': f"{mean - 2*std:.2f} to {mean + 2*std:.2f}",
            'timestamp': datetime.utcnow().isoformat(),
            'severity': 'high' if abs(value - mean) > 4 * std else 'medium'
        }
        
        # Send alert
        self.send_alert(alert)
        
        # Log to monitoring system
        logger.error(f"ANOMALY: {alert}")
        
        # Store for analysis
        self.store_anomaly_record(alert)
    
    def send_alert(self, alert: dict):
        """
        Send anomaly alert to monitoring system
        """
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        producer.send('anomaly-alerts', alert)
        producer.close()
    
    def store_anomaly_record(self, alert: dict):
        """
        Store anomaly for future analysis
        """
        conn = psycopg2.connect(
            host="localhost",
            database="ytempire",
            user="ytempire_user",
            password=os.environ["DB_PASSWORD"]
        )
        
        cursor = conn.cursor()
        query = """
            INSERT INTO analytics.anomalies 
            (metric_name, value, severity, details, detected_at)
            VALUES (%s, %s, %s, %s, %s)
        """
        
        cursor.execute(query, (
            alert['metric'],
            alert['value'],
            alert['severity'],
            json.dumps(alert),
            datetime.utcnow()
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
```