"""
Real-time Streaming Setup
Apache Kafka, Flink, and real-time data processing pipeline
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import NewTopic, ConfigResource, ConfigResourceType
from kafka.errors import KafkaError
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
from apache_beam.io import ReadFromKafka, WriteToKafka
from apache_beam.transforms.window import FixedWindows, SlidingWindows, Sessions
from apache_beam.transforms.trigger import AfterWatermark, AfterProcessingTime, AccumulationMode
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge
import websocket
import threading
from confluent_kafka import Producer, Consumer, TopicPartition
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.serialization import StringSerializer, SerializationContext, MessageField
from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
import avro.schema

# Metrics
events_processed = Counter('streaming_events_processed', 'Total events processed', ['stream', 'event_type'])
event_latency = Histogram('streaming_event_latency', 'Event processing latency', ['stream'])
stream_lag = Gauge('streaming_lag_seconds', 'Stream processing lag', ['stream', 'partition'])
error_count = Counter('streaming_errors', 'Streaming errors', ['stream', 'error_type'])

logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    """Stream configuration"""
    name: str
    topics: List[str]
    consumer_group: str
    batch_size: int = 100
    batch_timeout_ms: int = 1000
    max_retries: int = 3
    schema: Optional[Dict] = None
    processors: List[Callable] = field(default_factory=list)

@dataclass
class StreamEvent:
    """Stream event"""
    event_id: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

class StreamingPipeline:
    """Real-time streaming data pipeline"""
    
    def __init__(self,
                 kafka_brokers: str = 'localhost:9092',
                 schema_registry_url: str = 'http://localhost:8081',
                 redis_host: str = 'localhost',
                 redis_port: int = 6379):
        
        self.kafka_brokers = kafka_brokers
        self.schema_registry_url = schema_registry_url
        
        # Kafka clients
        self.admin_client = KafkaAdminClient(
            bootstrap_servers=kafka_brokers,
            client_id='streaming_pipeline_admin'
        )
        
        # Schema registry
        self.schema_registry = SchemaRegistryClient({'url': schema_registry_url})
        
        # Redis for state management
        self.redis_client = None
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # Stream configurations
        self.streams = {}
        self._register_streams()
        
        # Processing threads
        self.processing_threads = {}
        self.running = False
    
    async def initialize(self):
        """Initialize async components"""
        self.redis_client = await redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=True
        )
        
        # Create topics
        await self._create_topics()
        
        # Register schemas
        await self._register_schemas()
    
    def _register_streams(self):
        """Register all data streams"""
        
        # User activity stream
        self.streams['user_activity'] = StreamConfig(
            name='user_activity',
            topics=['user_events', 'user_interactions'],
            consumer_group='activity_processor',
            batch_size=100,
            processors=[
                self._process_user_activity,
                self._update_user_profile,
                self._detect_anomalies
            ],
            schema={
                'type': 'record',
                'name': 'UserActivity',
                'fields': [
                    {'name': 'user_id', 'type': 'string'},
                    {'name': 'action', 'type': 'string'},
                    {'name': 'timestamp', 'type': 'long'},
                    {'name': 'metadata', 'type': ['null', 'string'], 'default': None}
                ]
            }
        )
        
        # Video analytics stream
        self.streams['video_analytics'] = StreamConfig(
            name='video_analytics',
            topics=['video_views', 'video_engagement'],
            consumer_group='analytics_processor',
            batch_size=200,
            processors=[
                self._process_video_metrics,
                self._calculate_trending,
                self._update_recommendations
            ],
            schema={
                'type': 'record',
                'name': 'VideoAnalytics',
                'fields': [
                    {'name': 'video_id', 'type': 'string'},
                    {'name': 'event_type', 'type': 'string'},
                    {'name': 'value', 'type': 'float'},
                    {'name': 'timestamp', 'type': 'long'}
                ]
            }
        )
        
        # System metrics stream
        self.streams['system_metrics'] = StreamConfig(
            name='system_metrics',
            topics=['system_performance', 'application_logs'],
            consumer_group='metrics_processor',
            batch_size=500,
            batch_timeout_ms=5000,
            processors=[
                self._process_system_metrics,
                self._detect_system_anomalies,
                self._trigger_auto_scaling
            ]
        )
        
        # Financial transactions stream
        self.streams['transactions'] = StreamConfig(
            name='transactions',
            topics=['payments', 'subscriptions'],
            consumer_group='transaction_processor',
            batch_size=50,
            processors=[
                self._process_transaction,
                self._update_revenue_metrics,
                self._detect_fraud
            ],
            schema={
                'type': 'record',
                'name': 'Transaction',
                'fields': [
                    {'name': 'transaction_id', 'type': 'string'},
                    {'name': 'user_id', 'type': 'string'},
                    {'name': 'amount', 'type': 'float'},
                    {'name': 'currency', 'type': 'string'},
                    {'name': 'timestamp', 'type': 'long'}
                ]
            }
        )
        
        # ML model predictions stream
        self.streams['ml_predictions'] = StreamConfig(
            name='ml_predictions',
            topics=['model_predictions', 'model_feedback'],
            consumer_group='ml_processor',
            batch_size=100,
            processors=[
                self._process_predictions,
                self._update_model_metrics,
                self._trigger_retraining
            ]
        )
    
    async def _create_topics(self):
        """Create Kafka topics"""
        topics_to_create = []
        
        for stream_config in self.streams.values():
            for topic_name in stream_config.topics:
                topic = NewTopic(
                    name=topic_name,
                    num_partitions=3,
                    replication_factor=1,
                    topic_configs={
                        'retention.ms': '604800000',  # 7 days
                        'compression.type': 'snappy',
                        'max.message.bytes': '1048576'  # 1MB
                    }
                )
                topics_to_create.append(topic)
        
        try:
            self.admin_client.create_topics(topics_to_create, validate_only=False)
            logger.info(f"Created {len(topics_to_create)} topics")
        except Exception as e:
            logger.warning(f"Error creating topics: {e}")
    
    async def _register_schemas(self):
        """Register Avro schemas"""
        for stream_name, stream_config in self.streams.items():
            if stream_config.schema:
                schema_str = json.dumps(stream_config.schema)
                
                try:
                    # Register schema for each topic
                    for topic in stream_config.topics:
                        self.schema_registry.register_schema(
                            f"{topic}-value",
                            avro.schema.parse(schema_str)
                        )
                        logger.info(f"Registered schema for topic {topic}")
                except Exception as e:
                    logger.error(f"Error registering schema: {e}")
    
    def start_streaming(self):
        """Start all streaming pipelines"""
        self.running = True
        
        for stream_name, stream_config in self.streams.items():
            # Start consumer thread for each stream
            thread = threading.Thread(
                target=self._consume_stream,
                args=(stream_config,),
                name=f"stream_{stream_name}"
            )
            thread.daemon = True
            thread.start()
            self.processing_threads[stream_name] = thread
            
            logger.info(f"Started streaming pipeline: {stream_name}")
    
    def stop_streaming(self):
        """Stop all streaming pipelines"""
        self.running = False
        
        # Wait for threads to complete
        for thread in self.processing_threads.values():
            thread.join(timeout=5)
        
        logger.info("Stopped all streaming pipelines")
    
    def _consume_stream(self, stream_config: StreamConfig):
        """Consume and process stream"""
        consumer = KafkaConsumer(
            *stream_config.topics,
            bootstrap_servers=self.kafka_brokers,
            group_id=stream_config.consumer_group,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=False,
            max_poll_records=stream_config.batch_size
        )
        
        batch = []
        last_batch_time = datetime.now()
        
        while self.running:
            try:
                # Poll for messages
                messages = consumer.poll(timeout_ms=stream_config.batch_timeout_ms)
                
                for topic_partition, records in messages.items():
                    for record in records:
                        # Create event
                        event = StreamEvent(
                            event_id=f"{record.topic}_{record.partition}_{record.offset}",
                            event_type=record.topic,
                            timestamp=datetime.fromtimestamp(record.timestamp / 1000),
                            data=record.value,
                            metadata={
                                'partition': record.partition,
                                'offset': record.offset
                            }
                        )
                        
                        batch.append(event)
                        
                        # Update metrics
                        events_processed.labels(
                            stream=stream_config.name,
                            event_type=record.topic
                        ).inc()
                        
                        # Calculate lag
                        lag = (datetime.now() - event.timestamp).total_seconds()
                        stream_lag.labels(
                            stream=stream_config.name,
                            partition=record.partition
                        ).set(lag)
                
                # Process batch if full or timeout
                if len(batch) >= stream_config.batch_size or \
                   (datetime.now() - last_batch_time).total_seconds() > stream_config.batch_timeout_ms / 1000:
                    
                    if batch:
                        asyncio.run(self._process_batch(batch, stream_config))
                        consumer.commit()
                        batch = []
                        last_batch_time = datetime.now()
                        
            except Exception as e:
                logger.error(f"Error consuming stream {stream_config.name}: {e}")
                error_count.labels(
                    stream=stream_config.name,
                    error_type=type(e).__name__
                ).inc()
    
    async def _process_batch(self, batch: List[StreamEvent], stream_config: StreamConfig):
        """Process batch of events"""
        start_time = datetime.now()
        
        try:
            # Run processors in sequence
            for processor in stream_config.processors:
                if asyncio.iscoroutinefunction(processor):
                    await processor(batch)
                else:
                    processor(batch)
            
            # Record latency
            latency = (datetime.now() - start_time).total_seconds()
            event_latency.labels(stream=stream_config.name).observe(latency)
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            error_count.labels(
                stream=stream_config.name,
                error_type='processing_error'
            ).inc()
    
    async def publish_event(self, topic: str, event: Dict[str, Any]):
        """Publish event to stream"""
        producer = KafkaProducer(
            bootstrap_servers=self.kafka_brokers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        try:
            future = producer.send(topic, value=event)
            record_metadata = future.get(timeout=10)
            
            logger.debug(f"Published event to {topic}: partition={record_metadata.partition}, offset={record_metadata.offset}")
            
        except KafkaError as e:
            logger.error(f"Error publishing event: {e}")
            raise
        finally:
            producer.close()
    
    # Stream processors
    async def _process_user_activity(self, events: List[StreamEvent]):
        """Process user activity events"""
        for event in events:
            user_id = event.data.get('user_id')
            action = event.data.get('action')
            
            # Update user activity in Redis
            if self.redis_client:
                key = f"user_activity:{user_id}"
                await self.redis_client.hincrby(key, action, 1)
                await self.redis_client.expire(key, 86400)  # 24 hours
            
            # Track specific actions
            if action in ['video_upload', 'subscription_start']:
                await self._trigger_notification(user_id, action)
    
    async def _update_user_profile(self, events: List[StreamEvent]):
        """Update user profiles based on activity"""
        user_updates = {}
        
        for event in events:
            user_id = event.data.get('user_id')
            if user_id not in user_updates:
                user_updates[user_id] = {
                    'last_active': event.timestamp,
                    'activity_count': 0
                }
            
            user_updates[user_id]['activity_count'] += 1
            user_updates[user_id]['last_active'] = max(
                user_updates[user_id]['last_active'],
                event.timestamp
            )
        
        # Batch update profiles
        for user_id, updates in user_updates.items():
            if self.redis_client:
                key = f"user_profile:{user_id}"
                await self.redis_client.hset(key, mapping={
                    'last_active': updates['last_active'].isoformat(),
                    'activity_count': updates['activity_count']
                })
    
    async def _detect_anomalies(self, events: List[StreamEvent]):
        """Detect anomalous user behavior"""
        for event in events:
            # Simple anomaly detection based on rate
            user_id = event.data.get('user_id')
            
            if self.redis_client:
                key = f"user_rate:{user_id}"
                count = await self.redis_client.incr(key)
                
                if count == 1:
                    await self.redis_client.expire(key, 60)  # 1 minute window
                
                if count > 100:  # More than 100 actions per minute
                    logger.warning(f"Anomaly detected for user {user_id}: high activity rate")
                    await self._trigger_alert('user_anomaly', {
                        'user_id': user_id,
                        'rate': count,
                        'timestamp': event.timestamp
                    })
    
    async def _process_video_metrics(self, events: List[StreamEvent]):
        """Process video analytics events"""
        video_metrics = {}
        
        for event in events:
            video_id = event.data.get('video_id')
            metric_type = event.data.get('event_type')
            value = event.data.get('value', 1)
            
            if video_id not in video_metrics:
                video_metrics[video_id] = {}
            
            if metric_type not in video_metrics[video_id]:
                video_metrics[video_id][metric_type] = 0
            
            video_metrics[video_id][metric_type] += value
        
        # Update metrics in Redis
        if self.redis_client:
            for video_id, metrics in video_metrics.items():
                key = f"video_metrics:{video_id}"
                await self.redis_client.hset(key, mapping=metrics)
                await self.redis_client.expire(key, 3600)  # 1 hour
    
    async def _calculate_trending(self, events: List[StreamEvent]):
        """Calculate trending videos"""
        trending_scores = {}
        
        for event in events:
            video_id = event.data.get('video_id')
            event_type = event.data.get('event_type')
            
            # Weight different events
            weight = {
                'view': 1,
                'like': 5,
                'comment': 10,
                'share': 20
            }.get(event_type, 1)
            
            if video_id not in trending_scores:
                trending_scores[video_id] = 0
            
            # Time decay factor
            age_hours = (datetime.now() - event.timestamp).total_seconds() / 3600
            decay = np.exp(-age_hours / 24)  # 24-hour half-life
            
            trending_scores[video_id] += weight * decay
        
        # Update trending list
        if self.redis_client and trending_scores:
            # Sort by score
            sorted_videos = sorted(trending_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Update Redis sorted set
            for video_id, score in sorted_videos[:100]:  # Top 100
                await self.redis_client.zadd('trending_videos', {video_id: score})
    
    async def _update_recommendations(self, events: List[StreamEvent]):
        """Update recommendation engine with new data"""
        # Collect interaction data
        interactions = []
        
        for event in events:
            if event.data.get('event_type') in ['view', 'like', 'share']:
                interactions.append({
                    'user_id': event.data.get('user_id'),
                    'video_id': event.data.get('video_id'),
                    'interaction_type': event.data.get('event_type'),
                    'timestamp': event.timestamp
                })
        
        # Send to recommendation service
        if interactions:
            # This would call the recommendation service API
            pass
    
    async def _process_system_metrics(self, events: List[StreamEvent]):
        """Process system performance metrics"""
        for event in events:
            metric_name = event.data.get('metric')
            value = event.data.get('value')
            
            # Update Prometheus metrics
            if metric_name == 'cpu_usage':
                # Update CPU gauge
                pass
            elif metric_name == 'memory_usage':
                # Update memory gauge
                pass
            elif metric_name == 'request_latency':
                # Update latency histogram
                pass
    
    async def _detect_system_anomalies(self, events: List[StreamEvent]):
        """Detect system anomalies"""
        for event in events:
            metric_name = event.data.get('metric')
            value = event.data.get('value')
            
            # Simple threshold-based detection
            thresholds = {
                'cpu_usage': 80,
                'memory_usage': 90,
                'error_rate': 5
            }
            
            if metric_name in thresholds and value > thresholds[metric_name]:
                await self._trigger_alert('system_anomaly', {
                    'metric': metric_name,
                    'value': value,
                    'threshold': thresholds[metric_name],
                    'timestamp': event.timestamp
                })
    
    async def _trigger_auto_scaling(self, events: List[StreamEvent]):
        """Trigger auto-scaling based on metrics"""
        # Aggregate metrics
        cpu_values = []
        memory_values = []
        
        for event in events:
            if event.data.get('metric') == 'cpu_usage':
                cpu_values.append(event.data.get('value'))
            elif event.data.get('metric') == 'memory_usage':
                memory_values.append(event.data.get('value'))
        
        # Check if scaling needed
        if cpu_values and np.mean(cpu_values) > 70:
            # Trigger scale-up
            await self._scale_service('up', 'high_cpu')
        elif memory_values and np.mean(memory_values) > 80:
            # Trigger scale-up
            await self._scale_service('up', 'high_memory')
    
    async def _process_transaction(self, events: List[StreamEvent]):
        """Process financial transactions"""
        for event in events:
            transaction_id = event.data.get('transaction_id')
            amount = event.data.get('amount')
            currency = event.data.get('currency')
            
            # Validate transaction
            if amount <= 0:
                logger.warning(f"Invalid transaction amount: {transaction_id}")
                continue
            
            # Update revenue metrics
            if self.redis_client:
                key = f"revenue:{datetime.now().strftime('%Y%m%d')}"
                await self.redis_client.hincrbyfloat(key, currency, amount)
    
    async def _update_revenue_metrics(self, events: List[StreamEvent]):
        """Update revenue metrics"""
        daily_revenue = {}
        
        for event in events:
            date_key = event.timestamp.strftime('%Y%m%d')
            amount = event.data.get('amount', 0)
            
            if date_key not in daily_revenue:
                daily_revenue[date_key] = 0
            
            daily_revenue[date_key] += amount
        
        # Update metrics
        if self.redis_client:
            for date_key, revenue in daily_revenue.items():
                await self.redis_client.hset('daily_revenue', date_key, revenue)
    
    async def _detect_fraud(self, events: List[StreamEvent]):
        """Detect fraudulent transactions"""
        for event in events:
            user_id = event.data.get('user_id')
            amount = event.data.get('amount')
            
            # Simple fraud detection rules
            if amount > 10000:  # Large transaction
                await self._trigger_alert('large_transaction', event.data)
            
            # Check velocity
            if self.redis_client:
                key = f"transaction_velocity:{user_id}"
                count = await self.redis_client.incr(key)
                
                if count == 1:
                    await self.redis_client.expire(key, 3600)  # 1 hour
                
                if count > 10:  # More than 10 transactions per hour
                    await self._trigger_alert('high_velocity', {
                        'user_id': user_id,
                        'count': count
                    })
    
    async def _process_predictions(self, events: List[StreamEvent]):
        """Process ML model predictions"""
        for event in events:
            model_name = event.data.get('model')
            prediction = event.data.get('prediction')
            confidence = event.data.get('confidence')
            
            # Store predictions for analysis
            if self.redis_client:
                key = f"predictions:{model_name}:{datetime.now().strftime('%Y%m%d%H')}"
                await self.redis_client.rpush(key, json.dumps({
                    'prediction': prediction,
                    'confidence': confidence,
                    'timestamp': event.timestamp.isoformat()
                }))
                await self.redis_client.expire(key, 86400)
    
    async def _update_model_metrics(self, events: List[StreamEvent]):
        """Update model performance metrics"""
        model_metrics = {}
        
        for event in events:
            if event.data.get('feedback'):  # Has ground truth
                model_name = event.data.get('model')
                prediction = event.data.get('prediction')
                actual = event.data.get('feedback')
                
                if model_name not in model_metrics:
                    model_metrics[model_name] = {
                        'correct': 0,
                        'total': 0
                    }
                
                model_metrics[model_name]['total'] += 1
                if prediction == actual:
                    model_metrics[model_name]['correct'] += 1
        
        # Update accuracy metrics
        if self.redis_client:
            for model_name, metrics in model_metrics.items():
                accuracy = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
                await self.redis_client.hset('model_accuracy', model_name, accuracy)
    
    async def _trigger_retraining(self, events: List[StreamEvent]):
        """Trigger model retraining if needed"""
        if self.redis_client:
            # Check model accuracy
            accuracies = await self.redis_client.hgetall('model_accuracy')
            
            for model_name, accuracy in accuracies.items():
                if float(accuracy) < 0.8:  # Below threshold
                    logger.info(f"Triggering retraining for model {model_name}")
                    # Send retraining request
                    await self.publish_event('model_retraining', {
                        'model': model_name,
                        'reason': 'low_accuracy',
                        'current_accuracy': float(accuracy),
                        'timestamp': datetime.now().isoformat()
                    })
    
    async def _trigger_notification(self, user_id: str, action: str):
        """Trigger user notification"""
        # Send to notification service
        await self.publish_event('notifications', {
            'user_id': user_id,
            'type': 'activity',
            'action': action,
            'timestamp': datetime.now().isoformat()
        })
    
    async def _trigger_alert(self, alert_type: str, data: Dict):
        """Trigger system alert"""
        await self.publish_event('alerts', {
            'alert_type': alert_type,
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'severity': 'high' if alert_type in ['fraud', 'system_anomaly'] else 'medium'
        })
    
    async def _scale_service(self, direction: str, reason: str):
        """Scale service up or down"""
        await self.publish_event('scaling_events', {
            'direction': direction,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })

# Apache Beam pipeline for batch processing
class BeamStreamProcessor:
    """Apache Beam stream processor for complex transformations"""
    
    def __init__(self, kafka_config: Dict):
        self.kafka_config = kafka_config
    
    def create_pipeline(self):
        """Create Beam pipeline"""
        options = PipelineOptions([
            '--runner=DirectRunner',
            '--streaming'
        ])
        options.view_as(StandardOptions).streaming = True
        
        pipeline = beam.Pipeline(options=options)
        
        # Read from Kafka
        events = (
            pipeline
            | 'ReadFromKafka' >> ReadFromKafka(
                consumer_config=self.kafka_config,
                topics=['user_events'],
                with_metadata=True
            )
            | 'ParseEvents' >> beam.Map(self.parse_event)
        )
        
        # Window aggregations
        windowed_events = (
            events
            | 'Window' >> beam.WindowInto(
                FixedWindows(60),  # 1-minute windows
                trigger=AfterWatermark(early=AfterProcessingTime(10)),
                accumulation_mode=AccumulationMode.DISCARDING
            )
        )
        
        # Compute aggregations
        user_stats = (
            windowed_events
            | 'ExtractUser' >> beam.Map(lambda x: (x['user_id'], 1))
            | 'CountPerUser' >> beam.CombinePerKey(sum)
        )
        
        # Write results
        (
            user_stats
            | 'FormatOutput' >> beam.Map(self.format_output)
            | 'WriteToKafka' >> WriteToKafka(
                producer_config=self.kafka_config,
                topic='user_stats'
            )
        )
        
        return pipeline
    
    def parse_event(self, kafka_record):
        """Parse Kafka record"""
        return json.loads(kafka_record.value.decode('utf-8'))
    
    def format_output(self, user_stat):
        """Format output for Kafka"""
        user_id, count = user_stat
        return json.dumps({
            'user_id': user_id,
            'event_count': count,
            'timestamp': datetime.now().isoformat()
        }).encode('utf-8')

# WebSocket streaming for real-time UI updates
class WebSocketStreamer:
    """WebSocket streamer for real-time UI updates"""
    
    def __init__(self, websocket_url: str):
        self.websocket_url = websocket_url
        self.ws = None
        self.running = False
    
    def connect(self):
        """Connect to WebSocket"""
        self.ws = websocket.WebSocketApp(
            self.websocket_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Run in separate thread
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
    
    def on_open(self, ws):
        """Handle connection open"""
        logger.info("WebSocket connection opened")
        self.running = True
    
    def on_message(self, ws, message):
        """Handle incoming message"""
        try:
            data = json.loads(message)
            # Process real-time update
            asyncio.run(self.process_realtime_update(data))
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def on_error(self, ws, error):
        """Handle error"""
        logger.error(f"WebSocket error: {error}")
    
    def on_close(self, ws):
        """Handle connection close"""
        logger.info("WebSocket connection closed")
        self.running = False
    
    async def process_realtime_update(self, data: Dict):
        """Process real-time update"""
        # Send to appropriate stream
        update_type = data.get('type')
        
        if update_type == 'user_action':
            # Publish to user activity stream
            pass
        elif update_type == 'video_event':
            # Publish to video analytics stream
            pass
    
    def send_update(self, data: Dict):
        """Send update to WebSocket clients"""
        if self.ws and self.running:
            self.ws.send(json.dumps(data))

# Example usage
async def main():
    # Initialize streaming pipeline
    pipeline = StreamingPipeline(
        kafka_brokers='localhost:9092',
        schema_registry_url='http://localhost:8081'
    )
    
    await pipeline.initialize()
    
    # Start streaming
    pipeline.start_streaming()
    
    # Publish test event
    await pipeline.publish_event('user_events', {
        'user_id': 'user_123',
        'action': 'video_view',
        'video_id': 'video_456',
        'timestamp': datetime.now().isoformat()
    })
    
    # Let it run for a while
    await asyncio.sleep(60)
    
    # Stop streaming
    pipeline.stop_streaming()

if __name__ == "__main__":
    asyncio.run(main())