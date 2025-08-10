"""
Cost Tracking System for YTEmpire
Real-time cost calculation and tracking for all AI/ML operations
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from decimal import Decimal
import json
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from prometheus_client import Counter, Histogram, Gauge
import redis.asyncio as redis
from pydantic import BaseModel

from app.core.database import get_db
from app.core.config import settings
from app.models.cost import CostRecord, CostThreshold, CostAggregation
from app.core.exceptions import ThresholdExceededException


# Pricing configuration (USD)
API_COSTS = {
    "openai": {
        "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "dall-e-3": {"standard": 0.04, "hd": 0.08},  # per image
        "whisper": 0.006,  # per minute
    },
    "elevenlabs": {
        "standard": 0.30,  # per 1K characters
        "premium": 0.50,
    },
    "google": {
        "translate": 20.0,  # per million characters
        "tts": 4.0,  # per million characters
        "vision": 1.50,  # per 1K images
    },
    "youtube": {
        "data_api": 0.0,  # Free within quota
        "quota_unit": 1,  # Units per request
    },
    "aws": {
        "s3_storage": 0.023,  # per GB per month
        "s3_transfer": 0.09,  # per GB
        "ec2_gpu": 3.06,  # per hour (p3.2xlarge)
    },
}

# Infrastructure costs
INFRASTRUCTURE_COSTS = {
    "compute": {
        "cpu_hour": 0.05,
        "gpu_hour": 0.50,
        "memory_gb_hour": 0.01,
    },
    "storage": {
        "gb_month": 0.10,
        "transfer_gb": 0.05,
    },
    "database": {
        "query": 0.0001,
        "storage_gb": 0.25,
    },
}


class CostMetrics(BaseModel):
    """Cost metrics data model"""
    total_cost: Decimal
    api_costs: Dict[str, Decimal]
    infrastructure_costs: Dict[str, Decimal]
    per_video_cost: Optional[Decimal]
    daily_cost: Decimal
    monthly_projection: Decimal
    threshold_status: Dict[str, Any]


class CostTracker:
    """Main cost tracking service"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.cost_cache: Dict[str, Any] = {}
        
        # Prometheus metrics
        self.cost_counter = Counter(
            'ytempire_api_cost_total',
            'Total API costs',
            ['service', 'operation']
        )
        self.cost_histogram = Histogram(
            'ytempire_operation_cost',
            'Cost per operation',
            ['operation_type']
        )
        self.cost_gauge = Gauge(
            'ytempire_current_daily_cost',
            'Current daily cost'
        )
        
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        
    async def track_api_call(
        self,
        service: str,
        operation: str,
        units: float,
        metadata: Optional[Dict] = None,
        db: Optional[AsyncSession] = None
    ) -> Decimal:
        """Track API call cost"""
        # Calculate cost
        cost = self._calculate_api_cost(service, operation, units)
        
        # Update metrics
        self.cost_counter.labels(service=service, operation=operation).inc(float(cost))
        self.cost_histogram.labels(operation_type=f"{service}_{operation}").observe(float(cost))
        
        # Store in database
        if db:
            cost_record = CostRecord(
                service=service,
                operation=operation,
                units=units,
                unit_cost=self._get_unit_cost(service, operation),
                total_cost=cost,
                metadata=metadata or {},
                timestamp=datetime.utcnow()
            )
            db.add(cost_record)
            await db.commit()
        
        # Update Redis cache
        await self._update_cache(service, operation, cost)
        
        # Check thresholds
        await self._check_thresholds(service, cost, db)
        
        return cost
        
    async def track_infrastructure_usage(
        self,
        resource_type: str,
        resource_name: str,
        usage: float,
        duration_hours: float = 1.0,
        db: Optional[AsyncSession] = None
    ) -> Decimal:
        """Track infrastructure resource usage"""
        # Calculate cost
        if resource_type in INFRASTRUCTURE_COSTS:
            if resource_name in INFRASTRUCTURE_COSTS[resource_type]:
                unit_cost = INFRASTRUCTURE_COSTS[resource_type][resource_name]
                cost = Decimal(str(usage * unit_cost * duration_hours))
            else:
                cost = Decimal("0")
        else:
            cost = Decimal("0")
            
        # Store in database
        if db:
            cost_record = CostRecord(
                service="infrastructure",
                operation=f"{resource_type}_{resource_name}",
                units=usage,
                unit_cost=float(unit_cost) if 'unit_cost' in locals() else 0,
                total_cost=cost,
                metadata={"duration_hours": duration_hours},
                timestamp=datetime.utcnow()
            )
            db.add(cost_record)
            await db.commit()
            
        # Update cache
        await self._update_cache("infrastructure", resource_type, cost)
        
        return cost
        
    async def get_video_cost(
        self,
        video_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Calculate total cost for a video"""
        # Get all costs associated with video
        result = await db.execute(
            select(CostRecord).where(
                CostRecord.metadata["video_id"].astext == video_id
            )
        )
        records = result.scalars().all()
        
        # Aggregate costs
        total_cost = sum(record.total_cost for record in records)
        cost_breakdown = {}
        
        for record in records:
            service_key = f"{record.service}_{record.operation}"
            if service_key not in cost_breakdown:
                cost_breakdown[service_key] = Decimal("0")
            cost_breakdown[service_key] += record.total_cost
            
        return {
            "video_id": video_id,
            "total_cost": float(total_cost),
            "breakdown": {k: float(v) for k, v in cost_breakdown.items()},
            "timestamp": datetime.utcnow().isoformat()
        }
        
    async def get_real_time_costs(self) -> CostMetrics:
        """Get real-time cost metrics"""
        if not self.redis_client:
            await self.initialize()
            
        # Get cached values
        daily_cost = await self.redis_client.get("cost:daily:total")
        daily_cost = Decimal(daily_cost) if daily_cost else Decimal("0")
        
        # Get API costs
        api_costs = {}
        api_keys = await self.redis_client.keys("cost:api:*")
        for key in api_keys:
            service = key.split(":")[2]
            cost = await self.redis_client.get(key)
            api_costs[service] = Decimal(cost) if cost else Decimal("0")
            
        # Get infrastructure costs
        infra_costs = {}
        infra_keys = await self.redis_client.keys("cost:infrastructure:*")
        for key in infra_keys:
            resource = key.split(":")[2]
            cost = await self.redis_client.get(key)
            infra_costs[resource] = Decimal(cost) if cost else Decimal("0")
            
        # Calculate projections
        hours_passed = (datetime.utcnow() - datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )).total_seconds() / 3600
        
        if hours_passed > 0:
            hourly_rate = daily_cost / Decimal(str(hours_passed))
            daily_projection = hourly_rate * Decimal("24")
            monthly_projection = daily_projection * Decimal("30")
        else:
            daily_projection = Decimal("0")
            monthly_projection = Decimal("0")
            
        # Get video count for per-video cost
        video_count = await self.redis_client.get("stats:videos:processed:today")
        video_count = int(video_count) if video_count else 1
        per_video_cost = daily_cost / Decimal(str(video_count)) if video_count > 0 else None
        
        # Check threshold status
        threshold_status = await self._get_threshold_status(daily_cost)
        
        # Update Prometheus gauge
        self.cost_gauge.set(float(daily_cost))
        
        return CostMetrics(
            total_cost=daily_cost,
            api_costs=api_costs,
            infrastructure_costs=infra_costs,
            per_video_cost=per_video_cost,
            daily_cost=daily_cost,
            monthly_projection=monthly_projection,
            threshold_status=threshold_status
        )
        
    async def get_cost_aggregations(
        self,
        start_date: datetime,
        end_date: datetime,
        granularity: str,  # 'hour', 'day', 'week', 'month'
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Get aggregated cost data"""
        # Determine grouping
        if granularity == 'hour':
            date_trunc = func.date_trunc('hour', CostRecord.timestamp)
        elif granularity == 'day':
            date_trunc = func.date_trunc('day', CostRecord.timestamp)
        elif granularity == 'week':
            date_trunc = func.date_trunc('week', CostRecord.timestamp)
        else:  # month
            date_trunc = func.date_trunc('month', CostRecord.timestamp)
            
        # Query aggregated data
        result = await db.execute(
            select(
                date_trunc.label('period'),
                CostRecord.service,
                func.sum(CostRecord.total_cost).label('total_cost'),
                func.count(CostRecord.id).label('operation_count'),
                func.avg(CostRecord.total_cost).label('avg_cost')
            ).where(
                and_(
                    CostRecord.timestamp >= start_date,
                    CostRecord.timestamp <= end_date
                )
            ).group_by(date_trunc, CostRecord.service)
            .order_by(date_trunc)
        )
        
        aggregations = []
        for row in result:
            aggregations.append({
                "period": row.period.isoformat(),
                "service": row.service,
                "total_cost": float(row.total_cost),
                "operation_count": row.operation_count,
                "average_cost": float(row.avg_cost)
            })
            
        return aggregations
        
    async def set_threshold(
        self,
        threshold_type: str,  # 'daily', 'monthly', 'per_video', 'service'
        value: Decimal,
        service: Optional[str] = None,
        alert_email: Optional[str] = None,
        db: Optional[AsyncSession] = None
    ):
        """Set cost threshold"""
        if db:
            threshold = CostThreshold(
                threshold_type=threshold_type,
                service=service,
                value=value,
                alert_email=alert_email,
                is_active=True,
                created_at=datetime.utcnow()
            )
            db.add(threshold)
            await db.commit()
            
        # Cache threshold
        threshold_key = f"threshold:{threshold_type}"
        if service:
            threshold_key += f":{service}"
        await self.redis_client.set(threshold_key, str(value))
        
    def _calculate_api_cost(self, service: str, operation: str, units: float) -> Decimal:
        """Calculate API cost based on service and operation"""
        if service not in API_COSTS:
            return Decimal("0")
            
        service_costs = API_COSTS[service]
        
        if service == "openai":
            if "gpt" in operation:
                model = operation.split("_")[0]
                token_type = operation.split("_")[1]  # 'input' or 'output'
                if model in service_costs and token_type in service_costs[model]:
                    # Convert tokens to thousands
                    return Decimal(str(units / 1000 * service_costs[model][token_type]))
            elif "dall-e" in operation:
                quality = operation.split("_")[1] if "_" in operation else "standard"
                return Decimal(str(units * service_costs.get("dall-e-3", {}).get(quality, 0)))
            elif "whisper" in operation:
                return Decimal(str(units * service_costs.get("whisper", 0)))
                
        elif service == "elevenlabs":
            tier = operation if operation in service_costs else "standard"
            # Convert characters to thousands
            return Decimal(str(units / 1000 * service_costs[tier]))
            
        elif service == "google":
            if operation in service_costs:
                # Convert to millions for Google services
                return Decimal(str(units / 1000000 * service_costs[operation]))
                
        elif service == "aws":
            if operation in service_costs:
                return Decimal(str(units * service_costs[operation]))
                
        return Decimal("0")
        
    def _get_unit_cost(self, service: str, operation: str) -> float:
        """Get unit cost for service operation"""
        if service not in API_COSTS:
            return 0.0
            
        service_costs = API_COSTS[service]
        
        if isinstance(service_costs, dict):
            for key in service_costs:
                if key in operation:
                    if isinstance(service_costs[key], dict):
                        for subkey in service_costs[key]:
                            if subkey in operation:
                                return service_costs[key][subkey]
                    else:
                        return service_costs[key]
                        
        return 0.0
        
    async def _update_cache(self, service: str, operation: str, cost: Decimal):
        """Update Redis cache with cost data"""
        if not self.redis_client:
            return
            
        # Update service total
        service_key = f"cost:api:{service}"
        await self.redis_client.incrbyfloat(service_key, float(cost))
        
        # Update daily total
        daily_key = "cost:daily:total"
        await self.redis_client.incrbyfloat(daily_key, float(cost))
        
        # Set expiry for daily key (reset at midnight)
        ttl = await self.redis_client.ttl(daily_key)
        if ttl == -1:  # No expiry set
            midnight = datetime.utcnow().replace(
                hour=23, minute=59, second=59, microsecond=999999
            )
            seconds_until_midnight = (midnight - datetime.utcnow()).total_seconds()
            await self.redis_client.expire(daily_key, int(seconds_until_midnight))
            
    async def _check_thresholds(
        self,
        service: str,
        cost: Decimal,
        db: Optional[AsyncSession]
    ):
        """Check if cost exceeds thresholds"""
        if not self.redis_client:
            return
            
        # Check service threshold
        service_threshold = await self.redis_client.get(f"threshold:service:{service}")
        if service_threshold:
            service_total = await self.redis_client.get(f"cost:api:{service}")
            if service_total and Decimal(service_total) > Decimal(service_threshold):
                await self._trigger_threshold_alert(
                    "service",
                    service,
                    Decimal(service_total),
                    Decimal(service_threshold)
                )
                
        # Check daily threshold
        daily_threshold = await self.redis_client.get("threshold:daily")
        if daily_threshold:
            daily_total = await self.redis_client.get("cost:daily:total")
            if daily_total and Decimal(daily_total) > Decimal(daily_threshold):
                await self._trigger_threshold_alert(
                    "daily",
                    None,
                    Decimal(daily_total),
                    Decimal(daily_threshold)
                )
                
    async def _trigger_threshold_alert(
        self,
        threshold_type: str,
        service: Optional[str],
        current_value: Decimal,
        threshold_value: Decimal
    ):
        """Trigger threshold exceeded alert"""
        alert_key = f"alert:{threshold_type}"
        if service:
            alert_key += f":{service}"
            
        # Check if alert already sent today
        alert_sent = await self.redis_client.get(alert_key)
        if not alert_sent:
            # Send alert (implement notification service)
            alert_data = {
                "type": threshold_type,
                "service": service,
                "current_value": float(current_value),
                "threshold": float(threshold_value),
                "exceeded_by": float(current_value - threshold_value),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Mark alert as sent (expires at midnight)
            await self.redis_client.set(alert_key, "1", ex=86400)
            
            # Log alert
            print(f"COST THRESHOLD ALERT: {json.dumps(alert_data, indent=2)}")
            
    async def _get_threshold_status(self, current_cost: Decimal) -> Dict[str, Any]:
        """Get threshold status"""
        if not self.redis_client:
            return {}
            
        status = {
            "daily": {
                "current": float(current_cost),
                "threshold": None,
                "percentage": 0,
                "status": "ok"
            }
        }
        
        # Check daily threshold
        daily_threshold = await self.redis_client.get("threshold:daily")
        if daily_threshold:
            threshold = Decimal(daily_threshold)
            status["daily"]["threshold"] = float(threshold)
            percentage = (current_cost / threshold * 100) if threshold > 0 else 0
            status["daily"]["percentage"] = float(percentage)
            
            if percentage >= 100:
                status["daily"]["status"] = "exceeded"
            elif percentage >= 80:
                status["daily"]["status"] = "warning"
            else:
                status["daily"]["status"] = "ok"
                
        return status


# Global instance
cost_tracker = CostTracker()