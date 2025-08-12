"""
Real-time Analytics Service
Enhanced streaming analytics for beta user tracking and live dashboards
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
import websockets
import uuid
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class RealtimeMetric:
    """Real-time metric data point"""
    metric_name: str
    value: float
    timestamp: datetime
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    video_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class UserBehaviorEvent:
    """User behavior event for real-time tracking"""
    event_id: str
    user_id: str
    session_id: str
    event_type: str
    timestamp: datetime
    page_url: Optional[str] = None
    referrer: Optional[str] = None
    user_agent: Optional[str] = None
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


@dataclass
class BetaUserMetrics:
    """Beta user success metrics"""
    user_id: str
    onboarding_completion: float  # 0-100%
    time_to_first_video: Optional[int]  # seconds
    videos_generated_today: int
    engagement_score: float  # 0-10
    feature_adoption_rate: float  # 0-100%
    daily_active_time: int  # seconds
    success_indicators: Dict[str, bool]
    churn_risk_score: float  # 0-100%


class RealtimeAnalyticsService:
    """Enhanced real-time analytics service for streaming data"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        
        # Streaming buffers
        self.metrics_buffer = deque(maxlen=10000)
        self.events_buffer = deque(maxlen=10000)
        self.beta_metrics_buffer = deque(maxlen=1000)
        
        # WebSocket connections for live dashboards
        self.websocket_connections: Set[websockets.WebSocketServerProtocol] = set()
        
        # Processing state
        self.streaming_active = False
        self.last_processed = datetime.utcnow()
        
        # Beta user tracking
        self.beta_users: Set[str] = set()
        self.user_sessions: Dict[str, Dict] = {}
        
        # Success metrics configuration
        self.success_thresholds = {
            'time_to_first_video': 900,  # 15 minutes
            'min_videos_per_day': 3,
            'min_engagement_score': 7.0,
            'min_feature_adoption': 60.0,
            'max_churn_risk': 30.0
        }
        
        # Feature pipeline configuration
        self.feature_windows = {
            'engagement': timedelta(hours=1),
            'usage_patterns': timedelta(hours=6), 
            'performance': timedelta(minutes=15),
            'costs': timedelta(minutes=30)
        }
    
    async def initialize(self):
        """Initialize the real-time analytics service"""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            
            # Load beta users list
            await self._load_beta_users()
            
            # Start background tasks
            asyncio.create_task(self._stream_processor())
            asyncio.create_task(self._beta_user_tracker())
            asyncio.create_task(self._feature_pipeline())
            asyncio.create_task(self._live_dashboard_updater())
            asyncio.create_task(self._performance_monitor())
            
            self.streaming_active = True
            logger.info("Real-time analytics service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize real-time analytics: {e}")
            raise
    
    async def _load_beta_users(self):
        """Load current beta users from Redis"""
        try:
            beta_users_data = await self.redis_client.get("beta_users:list")
            if beta_users_data:
                self.beta_users = set(json.loads(beta_users_data))
            logger.info(f"Loaded {len(self.beta_users)} beta users for tracking")
        except Exception as e:
            logger.error(f"Failed to load beta users: {e}")
    
    async def add_beta_user(self, user_id: str):
        """Add a user to beta tracking"""
        self.beta_users.add(user_id)
        await self.redis_client.set(
            "beta_users:list", 
            json.dumps(list(self.beta_users))
        )
        
        # Initialize beta user metrics
        metrics = BetaUserMetrics(
            user_id=user_id,
            onboarding_completion=0.0,
            time_to_first_video=None,
            videos_generated_today=0,
            engagement_score=0.0,
            feature_adoption_rate=0.0,
            daily_active_time=0,
            success_indicators={
                'profile_completed': False,
                'channel_connected': False,
                'first_video_generated': False,
                'first_video_published': False,
                'engaged_with_analytics': False
            },
            churn_risk_score=50.0
        )
        
        await self._store_beta_metrics(metrics)
        logger.info(f"Added beta user {user_id} to tracking")
    
    async def track_event(self, event: UserBehaviorEvent):
        """Track a user behavior event"""
        self.events_buffer.append(event)
        
        # Immediate processing for beta users
        if event.user_id in self.beta_users:
            await self._process_beta_event(event)
        
        # Store in Redis for persistence
        key = f"events:stream:{datetime.utcnow().strftime('%Y%m%d%H')}"
        await self.redis_client.lpush(key, json.dumps(asdict(event), default=str))
        await self.redis_client.expire(key, 86400)  # 24 hour retention
    
    async def track_metric(self, metric: RealtimeMetric):
        """Track a real-time metric"""
        self.metrics_buffer.append(metric)
        
        # Store in time-series format
        key = f"metrics:{metric.metric_name}:{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        data = {
            'value': metric.value,
            'timestamp': metric.timestamp.isoformat(),
            'user_id': metric.user_id,
            'channel_id': metric.channel_id,
            'video_id': metric.video_id,
            **metric.metadata
        }
        
        await self.redis_client.setex(key, 3600, json.dumps(data))  # 1 hour retention
    
    async def _stream_processor(self):
        """Main streaming processor"""
        while self.streaming_active:
            try:
                await asyncio.sleep(5)  # Process every 5 seconds
                
                # Process metrics buffer
                if self.metrics_buffer:
                    await self._process_metrics_batch()
                
                # Process events buffer
                if self.events_buffer:
                    await self._process_events_batch()
                
                # Update last processed timestamp
                self.last_processed = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                await asyncio.sleep(10)
    
    async def _process_metrics_batch(self):
        """Process a batch of metrics"""
        batch = []
        while self.metrics_buffer and len(batch) < 100:
            batch.append(self.metrics_buffer.popleft())
        
        if not batch:
            return
        
        # Group by metric type
        by_metric = defaultdict(list)
        for metric in batch:
            by_metric[metric.metric_name].append(metric)
        
        # Calculate aggregations
        aggregations = {}
        for metric_name, metrics in by_metric.items():
            values = [m.value for m in metrics]
            aggregations[metric_name] = {
                'count': len(values),
                'sum': sum(values),
                'avg': np.mean(values),
                'min': min(values),
                'max': max(values),
                'latest': values[-1],
                'timestamp': datetime.utcnow().isoformat()
            }
        
        # Store aggregations
        agg_key = f"aggregations:{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        await self.redis_client.setex(
            agg_key, 300, json.dumps(aggregations)
        )
        
        # Broadcast to live dashboards
        await self._broadcast_to_dashboards({
            'type': 'metrics_update',
            'data': aggregations
        })
    
    async def _process_events_batch(self):
        """Process a batch of events"""
        batch = []
        while self.events_buffer and len(batch) < 100:
            batch.append(self.events_buffer.popleft())
        
        if not batch:
            return
        
        # Event analytics
        event_stats = defaultdict(int)
        user_activity = defaultdict(int)
        session_activity = defaultdict(set)
        
        for event in batch:
            event_stats[event.event_type] += 1
            user_activity[event.user_id] += 1
            session_activity[event.session_id].add(event.event_type)
        
        # Store real-time event analytics
        analytics = {
            'total_events': len(batch),
            'unique_users': len(user_activity),
            'unique_sessions': len(session_activity),
            'event_breakdown': dict(event_stats),
            'avg_events_per_user': np.mean(list(user_activity.values())) if user_activity else 0,
            'avg_events_per_session': np.mean([len(events) for events in session_activity.values()]) if session_activity else 0,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        analytics_key = f"event_analytics:{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        await self.redis_client.setex(
            analytics_key, 300, json.dumps(analytics)
        )
        
        # Broadcast to dashboards
        await self._broadcast_to_dashboards({
            'type': 'event_analytics',
            'data': analytics
        })
    
    async def _beta_user_tracker(self):
        """Track beta user specific metrics"""
        while self.streaming_active:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                for user_id in self.beta_users:
                    metrics = await self._calculate_beta_metrics(user_id)
                    if metrics:
                        await self._store_beta_metrics(metrics)
                        await self._check_success_criteria(metrics)
                
            except Exception as e:
                logger.error(f"Beta user tracking error: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_beta_metrics(self, user_id: str) -> Optional[BetaUserMetrics]:
        """Calculate comprehensive beta user metrics"""
        try:
            # Get recent events for this user
            today = datetime.utcnow().strftime('%Y%m%d')
            events_key = f"events:user:{user_id}:{today}"
            events_data = await self.redis_client.lrange(events_key, 0, -1)
            
            events = []
            for event_data in events_data:
                try:
                    event = json.loads(event_data)
                    events.append(event)
                except:
                    continue
            
            if not events:
                return None
            
            # Calculate onboarding completion
            onboarding_steps = ['signup', 'profile_setup', 'channel_connect', 'first_video']
            completed_steps = set(e.get('event_type') for e in events)
            onboarding_completion = (len(completed_steps & set(onboarding_steps)) / len(onboarding_steps)) * 100
            
            # Calculate time to first video
            signup_time = None
            first_video_time = None
            
            for event in sorted(events, key=lambda x: x.get('timestamp', '')):
                if event.get('event_type') == 'signup' and not signup_time:
                    signup_time = datetime.fromisoformat(event['timestamp'])
                elif event.get('event_type') == 'video_generate' and not first_video_time:
                    first_video_time = datetime.fromisoformat(event['timestamp'])
                    break
            
            time_to_first_video = None
            if signup_time and first_video_time:
                time_to_first_video = int((first_video_time - signup_time).total_seconds())
            
            # Count videos generated today
            videos_today = len([e for e in events if e.get('event_type') == 'video_generate'])
            
            # Calculate engagement score (0-10)
            engagement_events = ['video_generate', 'dashboard_view', 'analytics_view', 'settings_update']
            engagement_count = len([e for e in events if e.get('event_type') in engagement_events])
            engagement_score = min(10.0, engagement_count * 0.5)
            
            # Feature adoption rate
            available_features = ['video_generate', 'analytics_view', 'channel_manage', 'cost_tracking', 'auto_publish']
            used_features = set(e.get('data', {}).get('feature_name') for e in events 
                              if e.get('event_type') == 'feature_use')
            feature_adoption_rate = (len(used_features & set(available_features)) / len(available_features)) * 100
            
            # Daily active time (estimate from event frequency)
            if events:
                first_event = min(events, key=lambda x: x.get('timestamp', ''))['timestamp']
                last_event = max(events, key=lambda x: x.get('timestamp', ''))['timestamp']
                daily_active_time = int((
                    datetime.fromisoformat(last_event) - datetime.fromisoformat(first_event)
                ).total_seconds())
            else:
                daily_active_time = 0
            
            # Success indicators
            success_indicators = {
                'profile_completed': 'profile_setup' in completed_steps,
                'channel_connected': 'channel_connect' in completed_steps,
                'first_video_generated': 'video_generate' in completed_steps,
                'first_video_published': 'video_publish' in completed_steps,
                'engaged_with_analytics': 'analytics_view' in completed_steps
            }
            
            # Churn risk score
            churn_factors = 0
            if onboarding_completion < 50:
                churn_factors += 20
            if videos_today == 0:
                churn_factors += 30
            if engagement_score < 3:
                churn_factors += 25
            if feature_adoption_rate < 30:
                churn_factors += 15
            if time_to_first_video and time_to_first_video > 1800:  # 30 minutes
                churn_factors += 10
            
            churn_risk_score = min(100.0, churn_factors)
            
            return BetaUserMetrics(
                user_id=user_id,
                onboarding_completion=onboarding_completion,
                time_to_first_video=time_to_first_video,
                videos_generated_today=videos_today,
                engagement_score=engagement_score,
                feature_adoption_rate=feature_adoption_rate,
                daily_active_time=daily_active_time,
                success_indicators=success_indicators,
                churn_risk_score=churn_risk_score
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate beta metrics for {user_id}: {e}")
            return None
    
    async def _store_beta_metrics(self, metrics: BetaUserMetrics):
        """Store beta user metrics"""
        key = f"beta_metrics:{metrics.user_id}:{datetime.utcnow().strftime('%Y%m%d')}"
        await self.redis_client.setex(
            key, 86400 * 7, json.dumps(asdict(metrics), default=str)
        )
        
        # Add to buffer for aggregation
        self.beta_metrics_buffer.append(metrics)
    
    async def _check_success_criteria(self, metrics: BetaUserMetrics):
        """Check if user meets success criteria and generate alerts"""
        alerts = []
        
        # Check success thresholds
        if metrics.time_to_first_video and metrics.time_to_first_video > self.success_thresholds['time_to_first_video']:
            alerts.append({
                'type': 'slow_onboarding',
                'user_id': metrics.user_id,
                'message': f'User taking {metrics.time_to_first_video//60} minutes to first video'
            })
        
        if metrics.videos_generated_today < self.success_thresholds['min_videos_per_day']:
            alerts.append({
                'type': 'low_usage',
                'user_id': metrics.user_id,
                'message': f'Only {metrics.videos_generated_today} videos generated today'
            })
        
        if metrics.engagement_score < self.success_thresholds['min_engagement_score']:
            alerts.append({
                'type': 'low_engagement',
                'user_id': metrics.user_id,
                'message': f'Low engagement score: {metrics.engagement_score}'
            })
        
        if metrics.churn_risk_score > self.success_thresholds['max_churn_risk']:
            alerts.append({
                'type': 'churn_risk',
                'user_id': metrics.user_id,
                'message': f'High churn risk: {metrics.churn_risk_score}%'
            })
        
        # Store alerts
        if alerts:
            alerts_key = f"alerts:beta_users:{datetime.utcnow().strftime('%Y%m%d%H')}"
            await self.redis_client.lpush(alerts_key, *[json.dumps(alert) for alert in alerts])
            await self.redis_client.expire(alerts_key, 86400)
    
    async def _feature_pipeline(self):
        """Real-time feature processing pipeline"""
        while self.streaming_active:
            try:
                await asyncio.sleep(60)  # Process every minute
                
                # Process different feature windows
                for feature_type, window in self.feature_windows.items():
                    await self._process_feature_window(feature_type, window)
                
            except Exception as e:
                logger.error(f"Feature pipeline error: {e}")
                await asyncio.sleep(120)
    
    async def _process_feature_window(self, feature_type: str, window: timedelta):
        """Process features within a time window"""
        end_time = datetime.utcnow()
        start_time = end_time - window
        
        # Get metrics within window
        pattern = f"metrics:{feature_type}:*"
        cursor = 0
        metrics = []
        
        while True:
            cursor, keys = await self.redis_client.scan(
                cursor, match=pattern, count=100
            )
            
            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    try:
                        metric_data = json.loads(data)
                        timestamp = datetime.fromisoformat(metric_data['timestamp'])
                        if start_time <= timestamp <= end_time:
                            metrics.append(metric_data)
                    except:
                        continue
            
            if cursor == 0:
                break
        
        if metrics:
            # Calculate derived features
            features = self._calculate_derived_features(feature_type, metrics)
            
            # Store features
            feature_key = f"features:{feature_type}:{end_time.strftime('%Y%m%d%H%M')}"
            await self.redis_client.setex(
                feature_key, int(window.total_seconds()), json.dumps(features)
            )
    
    def _calculate_derived_features(self, feature_type: str, metrics: List[Dict]) -> Dict[str, Any]:
        """Calculate derived features from raw metrics"""
        if not metrics:
            return {}
        
        values = [m.get('value', 0) for m in metrics]
        
        base_features = {
            'count': len(metrics),
            'sum': sum(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values),
            'median': np.median(values),
            'q75': np.percentile(values, 75),
            'q25': np.percentile(values, 25)
        }
        
        # Feature-specific calculations
        if feature_type == 'engagement':
            # Calculate engagement velocity and momentum
            if len(values) > 1:
                diffs = np.diff(values)
                base_features.update({
                    'velocity': np.mean(diffs),
                    'acceleration': np.mean(np.diff(diffs)) if len(diffs) > 1 else 0,
                    'trend': 'increasing' if np.mean(diffs) > 0 else 'decreasing'
                })
        
        elif feature_type == 'performance':
            # Calculate performance stability and efficiency
            if len(values) > 5:
                base_features.update({
                    'stability': 1 - (np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0,
                    'efficiency': np.mean(values[-5:]) / np.mean(values[:5]) if len(values) >= 10 else 1
                })
        
        return base_features
    
    async def _live_dashboard_updater(self):
        """Update live dashboards via WebSocket"""
        while self.streaming_active:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds
                
                # Collect real-time data
                dashboard_data = await self._collect_dashboard_data()
                
                # Broadcast to connected clients
                await self._broadcast_to_dashboards({
                    'type': 'dashboard_update',
                    'data': dashboard_data,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_dashboard_data(self) -> Dict[str, Any]:
        """Collect data for live dashboard updates"""
        # Get recent aggregations
        pattern = "aggregations:*"
        cursor = 0
        recent_aggs = []
        
        while True:
            cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=10)
            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    try:
                        agg_data = json.loads(data)
                        recent_aggs.append(agg_data)
                    except:
                        continue
            if cursor == 0:
                break
        
        # Get beta user summary
        beta_summary = await self._get_beta_user_summary()
        
        # Get current alerts
        alerts = await self._get_current_alerts()
        
        return {
            'metrics': recent_aggs[-1] if recent_aggs else {},
            'beta_users': beta_summary,
            'alerts': alerts,
            'system_status': {
                'streaming_active': self.streaming_active,
                'last_processed': self.last_processed.isoformat(),
                'buffer_sizes': {
                    'metrics': len(self.metrics_buffer),
                    'events': len(self.events_buffer),
                    'beta_metrics': len(self.beta_metrics_buffer)
                }
            }
        }
    
    async def _get_beta_user_summary(self) -> Dict[str, Any]:
        """Get summary of beta user metrics"""
        if not self.beta_metrics_buffer:
            return {}
        
        # Aggregate beta metrics
        total_users = len(self.beta_users)
        active_users = len([m for m in self.beta_metrics_buffer if m.videos_generated_today > 0])
        avg_engagement = np.mean([m.engagement_score for m in self.beta_metrics_buffer])
        high_risk_users = len([m for m in self.beta_metrics_buffer if m.churn_risk_score > 50])
        
        return {
            'total_beta_users': total_users,
            'active_today': active_users,
            'average_engagement': avg_engagement,
            'high_risk_count': high_risk_users,
            'success_rate': (active_users / total_users * 100) if total_users > 0 else 0
        }
    
    async def _get_current_alerts(self) -> List[Dict[str, Any]]:
        """Get current alerts"""
        alerts_key = f"alerts:beta_users:{datetime.utcnow().strftime('%Y%m%d%H')}"
        alert_data = await self.redis_client.lrange(alerts_key, 0, 10)
        
        alerts = []
        for alert_json in alert_data:
            try:
                alert = json.loads(alert_json)
                alerts.append(alert)
            except:
                continue
        
        return alerts
    
    async def _broadcast_to_dashboards(self, message: Dict[str, Any]):
        """Broadcast message to all connected dashboard clients"""
        if not self.websocket_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = set()
        
        for ws in self.websocket_connections:
            try:
                await ws.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(ws)
            except Exception as e:
                logger.error(f"WebSocket broadcast error: {e}")
                disconnected.add(ws)
        
        # Remove disconnected clients
        self.websocket_connections -= disconnected
    
    async def _performance_monitor(self):
        """Monitor system performance and scale processing"""
        while self.streaming_active:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Monitor buffer sizes
                metrics_buffer_size = len(self.metrics_buffer)
                events_buffer_size = len(self.events_buffer)
                
                # Scale processing if buffers are getting full
                if metrics_buffer_size > 5000 or events_buffer_size > 5000:
                    logger.warning(f"High buffer usage - metrics: {metrics_buffer_size}, events: {events_buffer_size}")
                    # Trigger additional processing
                    await self._process_metrics_batch()
                    await self._process_events_batch()
                
                # Monitor memory usage and cleanup old data
                await self._cleanup_old_data()
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(120)
    
    async def _cleanup_old_data(self):
        """Clean up old data to manage memory usage"""
        # Clean up old aggregations (keep last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        pattern = "aggregations:*"
        
        cursor = 0
        while True:
            cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
            
            for key in keys:
                # Extract timestamp from key
                parts = key.split(':')
                if len(parts) > 1:
                    try:
                        key_time = datetime.strptime(parts[1], '%Y%m%d%H%M')
                        if key_time < cutoff:
                            await self.redis_client.delete(key)
                    except:
                        continue
            
            if cursor == 0:
                break
    
    # Public API methods
    
    async def get_realtime_metrics(self, metric_names: List[str] = None) -> Dict[str, Any]:
        """Get current real-time metrics"""
        # Get latest aggregations
        pattern = "aggregations:*"
        cursor = 0
        latest_data = {}
        latest_time = None
        
        while True:
            cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=10)
            
            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    try:
                        agg_data = json.loads(data)
                        timestamp = datetime.fromisoformat(agg_data.get('timestamp', ''))
                        
                        if latest_time is None or timestamp > latest_time:
                            latest_time = timestamp
                            latest_data = agg_data
                    except:
                        continue
            
            if cursor == 0:
                break
        
        # Filter by requested metrics if specified
        if metric_names:
            filtered_data = {k: v for k, v in latest_data.items() 
                           if k in metric_names or k == 'timestamp'}
            return filtered_data
        
        return latest_data
    
    async def get_beta_user_metrics(self, user_id: str = None) -> Dict[str, Any]:
        """Get beta user metrics"""
        if user_id:
            # Get specific user metrics
            key = f"beta_metrics:{user_id}:{datetime.utcnow().strftime('%Y%m%d')}"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return {}
        else:
            # Get summary of all beta users
            return await self._get_beta_user_summary()
    
    async def register_websocket(self, websocket: websockets.WebSocketServerProtocol):
        """Register a WebSocket connection for live updates"""
        self.websocket_connections.add(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.websocket_connections)}")
    
    async def unregister_websocket(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a WebSocket connection"""
        self.websocket_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.websocket_connections)}")
    
    async def shutdown(self):
        """Shutdown the service gracefully"""
        self.streaming_active = False
        
        # Close all WebSocket connections
        for ws in self.websocket_connections:
            try:
                await ws.close()
            except:
                pass
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Real-time analytics service shut down")


# Global instance
realtime_analytics_service = RealtimeAnalyticsService()