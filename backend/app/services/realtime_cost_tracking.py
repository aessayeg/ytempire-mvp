"""
Realtime Cost Tracking Service
Provides real-time cost monitoring, budget alerts, and usage tracking
"""

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from collections import defaultdict
import uuid

from redis import asyncio as aioredis
from sqlalchemy import select, func, and_
from app.db.session import AsyncSessionLocal
from app.models.cost import Cost
from app.models.user import User
from app.core.config import settings
from app.services.notification_service import notification_service
from app.services.websocket_manager import ConnectionManager

logger = logging.getLogger(__name__)


class RealtimeCostTracker:
    """
    Real-time cost tracking and monitoring service
    """
    
    def __init__(self):
        self.redis_client: Optional[aioredis.Redis] = None
        self.connection_manager = ConnectionManager()
        self.budget_alerts: Dict[str, Dict] = {}
        self.cost_thresholds = {
            'video_generation': Decimal('3.00'),  # $3 per video target
            'daily_openai': Decimal('50.00'),     # $50 daily OpenAI limit
            'daily_elevenlabs': Decimal('20.00'),  # $20 daily ElevenLabs limit
            'daily_google': Decimal('10.00')       # $10 daily Google limit
        }
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = aioredis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
                password=settings.REDIS_PASSWORD,
                decode_responses=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {str(e)}")
    
    async def track_cost_realtime(
        self,
        user_id: str,
        service: str,
        operation: str,
        amount: Decimal,
        video_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Track cost in real-time with immediate updates
        
        Args:
            user_id: User ID
            service: Service name (openai, elevenlabs, dalle, etc.)
            operation: Operation type (script_generation, voice_synthesis, etc.)
            amount: Cost amount
            video_id: Associated video ID
            channel_id: Associated channel ID
            metadata: Additional metadata
        
        Returns:
            Tracking result with current totals
        """
        try:
            # Store in database
            async with AsyncSessionLocal() as db:
                cost = Cost(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    service=service,
                    operation=operation,
                    amount=amount,
                    video_id=video_id,
                    channel_id=channel_id,
                    metadata=json.dumps(metadata or {})
                )
                db.add(cost)
                await db.commit()
            
            # Update Redis counters for real-time tracking
            if self.redis_client:
                # Daily service counter
                daily_key = f"cost:daily:{service}:{datetime.utcnow().strftime('%Y%m%d')}"
                await self.redis_client.incrbyfloat(daily_key, float(amount))
                await self.redis_client.expire(daily_key, 86400)  # 24 hour TTL
                
                # User daily counter
                user_daily_key = f"cost:user:{user_id}:daily:{datetime.utcnow().strftime('%Y%m%d')}"
                await self.redis_client.incrbyfloat(user_daily_key, float(amount))
                await self.redis_client.expire(user_daily_key, 86400)
                
                # Video cost counter
                if video_id:
                    video_key = f"cost:video:{video_id}"
                    await self.redis_client.incrbyfloat(video_key, float(amount))
                    await self.redis_client.expire(video_key, 604800)  # 7 day TTL
            
            # Get current totals
            totals = await self.get_current_totals(user_id, service)
            
            # Check budget alerts
            await self.check_budget_alerts(user_id, service, totals)
            
            # Send real-time update via WebSocket
            await self.send_cost_update(user_id, {
                'type': 'cost_update',
                'service': service,
                'operation': operation,
                'amount': float(amount),
                'totals': totals,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            return {
                'success': True,
                'cost_id': cost.id,
                'amount': float(amount),
                'totals': totals
            }
        
        except Exception as e:
            logger.error(f"Real-time cost tracking failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_current_totals(
        self,
        user_id: str,
        service: Optional[str] = None
    ) -> Dict[str, float]:
        """Get current cost totals"""
        totals = {}
        
        if self.redis_client:
            today = datetime.utcnow().strftime('%Y%m%d')
            
            # Get service daily total
            if service:
                daily_key = f"cost:daily:{service}:{today}"
                daily_total = await self.redis_client.get(daily_key)
                totals[f'{service}_daily'] = float(daily_total or 0)
            
            # Get user daily total
            user_daily_key = f"cost:user:{user_id}:daily:{today}"
            user_daily = await self.redis_client.get(user_daily_key)
            totals['user_daily'] = float(user_daily or 0)
        
        # Get database totals if Redis unavailable
        if not totals:
            async with AsyncSessionLocal() as db:
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
                
                if service:
                    result = await db.execute(
                        select(func.sum(Cost.amount)).where(
                            and_(
                                Cost.service == service,
                                Cost.created_at >= today_start
                            )
                        )
                    )
                    totals[f'{service}_daily'] = float(result.scalar() or 0)
                
                result = await db.execute(
                    select(func.sum(Cost.amount)).where(
                        and_(
                            Cost.user_id == user_id,
                            Cost.created_at >= today_start
                        )
                    )
                )
                totals['user_daily'] = float(result.scalar() or 0)
        
        return totals
    
    async def check_budget_alerts(
        self,
        user_id: str,
        service: str,
        totals: Dict[str, float]
    ):
        """Check and send budget alerts"""
        alerts_to_send = []
        
        # Check service daily limits
        service_daily = totals.get(f'{service}_daily', 0)
        
        if service == 'openai' and service_daily > float(self.cost_thresholds['daily_openai']):
            alerts_to_send.append({
                'type': 'budget_exceeded',
                'service': 'OpenAI',
                'limit': float(self.cost_thresholds['daily_openai']),
                'current': service_daily
            })
        
        elif service == 'elevenlabs' and service_daily > float(self.cost_thresholds['daily_elevenlabs']):
            alerts_to_send.append({
                'type': 'budget_exceeded',
                'service': 'ElevenLabs',
                'limit': float(self.cost_thresholds['daily_elevenlabs']),
                'current': service_daily
            })
        
        # Check warnings at 80% threshold
        for threshold_service, limit in [
            ('openai', self.cost_thresholds['daily_openai']),
            ('elevenlabs', self.cost_thresholds['daily_elevenlabs'])
        ]:
            if service == threshold_service:
                if service_daily > float(limit) * 0.8 and service_daily <= float(limit):
                    alerts_to_send.append({
                        'type': 'budget_warning',
                        'service': threshold_service.title(),
                        'limit': float(limit),
                        'current': service_daily,
                        'percentage': (service_daily / float(limit)) * 100
                    })
        
        # Send alerts
        for alert in alerts_to_send:
            await self.send_budget_alert(user_id, alert)
    
    async def send_budget_alert(self, user_id: str, alert: Dict[str, Any]):
        """Send budget alert to user"""
        # Check if we've already sent this alert today
        alert_key = f"alert:{user_id}:{alert['type']}:{alert['service']}:{datetime.utcnow().strftime('%Y%m%d')}"
        
        if self.redis_client:
            already_sent = await self.redis_client.get(alert_key)
            if already_sent:
                return
            
            # Mark as sent
            await self.redis_client.setex(alert_key, 86400, "1")
        
        # Send notification
        if alert['type'] == 'budget_exceeded':
            message = f"⚠️ Budget Exceeded: {alert['service']} daily limit of ${alert['limit']:.2f} exceeded. Current: ${alert['current']:.2f}"
            notification_type = "error"
        else:
            message = f"⚡ Budget Warning: {alert['service']} at {alert['percentage']:.0f}% of daily limit (${alert['current']:.2f} of ${alert['limit']:.2f})"
            notification_type = "warning"
        
        await notification_service.send_notification(
            user_id=user_id,
            title="Budget Alert",
            message=message,
            type=notification_type
        )
        
        # Send WebSocket update
        await self.send_cost_update(user_id, {
            'type': 'budget_alert',
            'alert': alert,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def send_cost_update(self, user_id: str, update: Dict[str, Any]):
        """Send cost update via WebSocket"""
        await self.connection_manager.send_to_user(user_id, update)
    
    async def get_video_cost_breakdown(self, video_id: str) -> Dict[str, Any]:
        """Get detailed cost breakdown for a video"""
        async with AsyncSessionLocal() as db:
            costs = await db.execute(
                select(Cost).where(Cost.video_id == video_id)
            )
            
            breakdown = defaultdict(float)
            details = []
            
            for cost in costs.scalars():
                breakdown[cost.service] += float(cost.amount)
                details.append({
                    'service': cost.service,
                    'operation': cost.operation,
                    'amount': float(cost.amount),
                    'timestamp': cost.created_at.isoformat()
                })
            
            total = sum(breakdown.values())
            
            return {
                'video_id': video_id,
                'total_cost': total,
                'breakdown': dict(breakdown),
                'details': details,
                'within_target': total <= float(self.cost_thresholds['video_generation'])
            }
    
    async def get_realtime_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get real-time cost dashboard data"""
        today = datetime.utcnow().strftime('%Y%m%d')
        dashboard = {
            'timestamp': datetime.utcnow().isoformat(),
            'daily_totals': {},
            'service_breakdown': {},
            'recent_costs': [],
            'budget_status': {}
        }
        
        # Get daily totals from Redis
        if self.redis_client:
            # Service totals
            for service in ['openai', 'elevenlabs', 'dalle', 'google']:
                key = f"cost:daily:{service}:{today}"
                value = await self.redis_client.get(key)
                dashboard['daily_totals'][service] = float(value or 0)
            
            # User total
            user_key = f"cost:user:{user_id}:daily:{today}"
            user_total = await self.redis_client.get(user_key)
            dashboard['daily_totals']['total'] = float(user_total or 0)
        
        # Get recent costs from database
        async with AsyncSessionLocal() as db:
            recent = await db.execute(
                select(Cost)
                .where(Cost.user_id == user_id)
                .order_by(Cost.created_at.desc())
                .limit(10)
            )
            
            for cost in recent.scalars():
                dashboard['recent_costs'].append({
                    'service': cost.service,
                    'operation': cost.operation,
                    'amount': float(cost.amount),
                    'timestamp': cost.created_at.isoformat()
                })
        
        # Calculate budget status
        for service, limit in [
            ('openai', self.cost_thresholds['daily_openai']),
            ('elevenlabs', self.cost_thresholds['daily_elevenlabs']),
            ('google', self.cost_thresholds['daily_google'])
        ]:
            current = dashboard['daily_totals'].get(service, 0)
            dashboard['budget_status'][service] = {
                'current': current,
                'limit': float(limit),
                'percentage': (current / float(limit)) * 100 if limit > 0 else 0,
                'status': 'ok' if current < float(limit) * 0.8 else 'warning' if current < float(limit) else 'exceeded'
            }
        
        return dashboard
    
    async def start_monitoring(self):
        """Start real-time monitoring tasks"""
        asyncio.create_task(self._monitor_costs())
    
    async def _monitor_costs(self):
        """Background task to monitor costs"""
        while True:
            try:
                # Check all active users for budget alerts
                async with AsyncSessionLocal() as db:
                    # Get users with recent activity
                    recent_users = await db.execute(
                        select(Cost.user_id).distinct()
                        .where(Cost.created_at >= datetime.utcnow() - timedelta(hours=1))
                    )
                    
                    for user_id in recent_users.scalars():
                        totals = await self.get_current_totals(user_id)
                        
                        # Check each service
                        for service in ['openai', 'elevenlabs', 'google']:
                            service_total = totals.get(f'{service}_daily', 0)
                            limit = self.cost_thresholds.get(f'daily_{service}')
                            
                            if limit and service_total > float(limit) * 0.9:
                                await self.check_budget_alerts(user_id, service, totals)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Cost monitoring error: {str(e)}")
                await asyncio.sleep(60)
    
    async def export_cost_report(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Export cost report for date range"""
        async with AsyncSessionLocal() as db:
            costs = await db.execute(
                select(Cost).where(
                    and_(
                        Cost.user_id == user_id,
                        Cost.created_at >= start_date,
                        Cost.created_at <= end_date
                    )
                )
            )
            
            report = {
                'user_id': user_id,
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'summary': defaultdict(lambda: {'count': 0, 'total': 0}),
                'daily_breakdown': defaultdict(lambda: defaultdict(float)),
                'top_operations': defaultdict(float)
            }
            
            for cost in costs.scalars():
                date_key = cost.created_at.strftime('%Y-%m-%d')
                
                # Summary by service
                report['summary'][cost.service]['count'] += 1
                report['summary'][cost.service]['total'] += float(cost.amount)
                
                # Daily breakdown
                report['daily_breakdown'][date_key][cost.service] += float(cost.amount)
                
                # Top operations
                report['top_operations'][cost.operation] += float(cost.amount)
            
            # Convert defaultdicts to regular dicts
            report['summary'] = dict(report['summary'])
            report['daily_breakdown'] = {k: dict(v) for k, v in report['daily_breakdown'].items()}
            report['top_operations'] = dict(sorted(
                report['top_operations'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
            
            # Calculate totals
            report['total_cost'] = sum(s['total'] for s in report['summary'].values())
            report['total_operations'] = sum(s['count'] for s in report['summary'].values())
            
            return report


# Singleton instance
realtime_cost_tracker = RealtimeCostTracker()