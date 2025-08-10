"""
Cost Aggregation Pipeline Service
Tracks and aggregates costs across all services for YTEmpire
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from enum import Enum
from decimal import Decimal
import redis.asyncio as redis
from collections import defaultdict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CostCategory(str, Enum):
    """Cost categories"""
    OPENAI_API = "openai_api"
    YOUTUBE_API = "youtube_api"
    ELEVENLABS_TTS = "elevenlabs_tts"
    GOOGLE_TTS = "google_tts"
    STORAGE = "storage"
    COMPUTE = "compute"
    BANDWIDTH = "bandwidth"
    DATABASE = "database"
    THIRD_PARTY = "third_party"
    OTHER = "other"


class BillingPeriod(str, Enum):
    """Billing period types"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class AlertType(str, Enum):
    """Cost alert types"""
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    UNUSUAL_SPIKE = "unusual_spike"
    BUDGET_WARNING = "budget_warning"
    BUDGET_CRITICAL = "budget_critical"
    FORECAST_OVERRUN = "forecast_overrun"


@dataclass
class CostItem:
    """Individual cost item"""
    item_id: str
    timestamp: datetime
    category: CostCategory
    service: str
    amount: Decimal
    currency: str = "USD"
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    video_id: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostAggregate:
    """Aggregated cost data"""
    period_start: datetime
    period_end: datetime
    billing_period: BillingPeriod
    category: Optional[CostCategory] = None
    total_amount: Decimal = Decimal("0")
    item_count: int = 0
    avg_cost: Decimal = Decimal("0")
    min_cost: Decimal = Decimal("0")
    max_cost: Decimal = Decimal("0")
    user_breakdown: Dict[str, Decimal] = field(default_factory=dict)
    channel_breakdown: Dict[str, Decimal] = field(default_factory=dict)
    service_breakdown: Dict[str, Decimal] = field(default_factory=dict)


@dataclass
class Budget:
    """Budget configuration"""
    budget_id: str
    name: str
    amount: Decimal
    period: BillingPeriod
    category: Optional[CostCategory] = None
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    alert_thresholds: List[int] = field(default_factory=lambda: [50, 75, 90, 100])
    created_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True


@dataclass
class CostForecast:
    """Cost forecast data"""
    forecast_date: date
    period: BillingPeriod
    predicted_cost: Decimal
    confidence_interval_low: Decimal
    confidence_interval_high: Decimal
    trend: str  # increasing, decreasing, stable
    seasonality_factor: float
    

class CostAggregationPipeline:
    """
    Pipeline for aggregating and analyzing costs across all services
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        aggregation_interval: int = 300  # 5 minutes
    ):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.aggregation_interval = aggregation_interval
        
        # Cost tracking
        self.cost_buffer: List[CostItem] = []
        self.aggregates: Dict[str, CostAggregate] = {}
        self.budgets: Dict[str, Budget] = {}
        
        # Pricing configuration
        self.pricing = {
            CostCategory.OPENAI_API: {
                "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
                "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
                "dall-e-3": {"hd": 0.08, "standard": 0.04},  # per image
            },
            CostCategory.YOUTUBE_API: {
                "search": 100,  # quota units
                "upload": 1600,
                "update": 50,
                "list": 1,
            },
            CostCategory.ELEVENLABS_TTS: {
                "standard": 0.18,  # per 1K characters
                "premium": 0.30,
            },
            CostCategory.STORAGE: {
                "s3_standard": 0.023,  # per GB/month
                "s3_infrequent": 0.0125,
            },
            CostCategory.COMPUTE: {
                "gpu_hour": 0.90,  # per hour
                "cpu_hour": 0.10,
            }
        }
        
        # Alert settings
        self.alert_enabled = True
        self.alert_cooldown = 3600  # 1 hour between same alerts
        self.last_alerts: Dict[str, datetime] = {}
        
    async def initialize(self):
        """Initialize the cost aggregation pipeline"""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            
            # Load existing budgets
            await self.load_budgets()
            
            # Start background tasks
            asyncio.create_task(self.aggregate_costs())
            asyncio.create_task(self.monitor_budgets())
            asyncio.create_task(self.generate_forecasts())
            asyncio.create_task(self.cleanup_old_data())
            
            logger.info("Cost aggregation pipeline initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize cost pipeline: {e}")
            raise
            
    async def track_cost(
        self,
        category: CostCategory,
        service: str,
        amount: float,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        video_id: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track a cost item
        """
        try:
            cost_item = CostItem(
                item_id=f"{category.value}_{datetime.utcnow().timestamp()}",
                timestamp=datetime.utcnow(),
                category=category,
                service=service,
                amount=Decimal(str(amount)),
                user_id=user_id,
                channel_id=channel_id,
                video_id=video_id,
                description=description,
                metadata=metadata or {}
            )
            
            # Add to buffer
            self.cost_buffer.append(cost_item)
            
            # Store in Redis for immediate access
            await self.store_cost_item(cost_item)
            
            # Check for immediate alerts
            await self.check_cost_alerts(cost_item)
            
            return cost_item.item_id
            
        except Exception as e:
            logger.error(f"Failed to track cost: {e}")
            raise
            
    async def store_cost_item(self, cost_item: CostItem):
        """Store cost item in Redis"""
        key = f"cost:item:{cost_item.item_id}"
        data = asdict(cost_item)
        data['amount'] = str(cost_item.amount)
        data['timestamp'] = cost_item.timestamp.isoformat()
        
        await self.redis_client.setex(
            key,
            86400 * 30,  # 30 days retention
            json.dumps(data)
        )
        
        # Add to daily set for quick retrieval
        daily_key = f"cost:daily:{cost_item.timestamp.date()}"
        await self.redis_client.sadd(daily_key, cost_item.item_id)
        await self.redis_client.expire(daily_key, 86400 * 30)
        
    async def calculate_api_cost(
        self,
        service: str,
        usage: Dict[str, Any]
    ) -> Decimal:
        """
        Calculate cost based on API usage
        """
        total_cost = Decimal("0")
        
        if service == "openai":
            model = usage.get("model", "gpt-3.5-turbo")
            if "tokens" in usage:
                input_tokens = usage["tokens"].get("input", 0)
                output_tokens = usage["tokens"].get("output", 0)
                
                pricing = self.pricing[CostCategory.OPENAI_API].get(model, {})
                input_cost = Decimal(str(input_tokens / 1000 * pricing.get("input", 0)))
                output_cost = Decimal(str(output_tokens / 1000 * pricing.get("output", 0)))
                total_cost = input_cost + output_cost
                
        elif service == "youtube":
            operations = usage.get("operations", {})
            quota_pricing = self.pricing[CostCategory.YOUTUBE_API]
            
            for op, count in operations.items():
                quota_units = quota_pricing.get(op, 1) * count
                # YouTube has 10,000 daily quota, estimate cost
                total_cost += Decimal(str(quota_units * 0.0001))  # $1 per 10K units
                
        elif service == "elevenlabs":
            characters = usage.get("characters", 0)
            tier = usage.get("tier", "standard")
            pricing = self.pricing[CostCategory.ELEVENLABS_TTS]
            
            char_cost = pricing.get(tier, 0.18)
            total_cost = Decimal(str(characters / 1000 * char_cost))
            
        return total_cost
        
    async def aggregate_costs(self):
        """
        Background task to aggregate costs periodically
        """
        while True:
            try:
                await asyncio.sleep(self.aggregation_interval)
                
                if self.cost_buffer:
                    await self.process_cost_buffer()
                    
            except Exception as e:
                logger.error(f"Cost aggregation error: {e}")
                
    async def process_cost_buffer(self):
        """Process buffered cost items"""
        if not self.cost_buffer:
            return
            
        # Copy and clear buffer
        items = self.cost_buffer.copy()
        self.cost_buffer.clear()
        
        # Aggregate by different dimensions
        for period in BillingPeriod:
            await self.aggregate_by_period(items, period)
            
    async def aggregate_by_period(
        self,
        items: List[CostItem],
        period: BillingPeriod
    ):
        """Aggregate costs for a specific period"""
        period_groups = defaultdict(list)
        
        for item in items:
            period_key = self.get_period_key(item.timestamp, period)
            period_groups[period_key].append(item)
        
        for period_key, period_items in period_groups.items():
            aggregate = await self.create_aggregate(period_key, period, period_items)
            
            # Store aggregate
            await self.store_aggregate(aggregate)
            
            # Update running totals
            self.aggregates[f"{period.value}:{period_key}"] = aggregate
            
    async def create_aggregate(
        self,
        period_key: str,
        period: BillingPeriod,
        items: List[CostItem]
    ) -> CostAggregate:
        """Create cost aggregate from items"""
        if not items:
            return None
            
        amounts = [item.amount for item in items]
        
        aggregate = CostAggregate(
            period_start=self.parse_period_key(period_key, period),
            period_end=self.get_period_end(period_key, period),
            billing_period=period,
            total_amount=sum(amounts),
            item_count=len(items),
            avg_cost=sum(amounts) / len(amounts),
            min_cost=min(amounts),
            max_cost=max(amounts)
        )
        
        # Calculate breakdowns
        for item in items:
            if item.user_id:
                aggregate.user_breakdown[item.user_id] = (
                    aggregate.user_breakdown.get(item.user_id, Decimal("0")) + item.amount
                )
            if item.channel_id:
                aggregate.channel_breakdown[item.channel_id] = (
                    aggregate.channel_breakdown.get(item.channel_id, Decimal("0")) + item.amount
                )
            if item.service:
                aggregate.service_breakdown[item.service] = (
                    aggregate.service_breakdown.get(item.service, Decimal("0")) + item.amount
                )
        
        return aggregate
        
    async def store_aggregate(self, aggregate: CostAggregate):
        """Store cost aggregate in Redis"""
        key = f"cost:aggregate:{aggregate.billing_period.value}:{aggregate.period_start.timestamp()}"
        
        data = {
            "period_start": aggregate.period_start.isoformat(),
            "period_end": aggregate.period_end.isoformat(),
            "billing_period": aggregate.billing_period.value,
            "total_amount": str(aggregate.total_amount),
            "item_count": aggregate.item_count,
            "avg_cost": str(aggregate.avg_cost),
            "min_cost": str(aggregate.min_cost),
            "max_cost": str(aggregate.max_cost),
            "user_breakdown": {k: str(v) for k, v in aggregate.user_breakdown.items()},
            "channel_breakdown": {k: str(v) for k, v in aggregate.channel_breakdown.items()},
            "service_breakdown": {k: str(v) for k, v in aggregate.service_breakdown.items()}
        }
        
        await self.redis_client.setex(
            key,
            86400 * 90,  # 90 days retention
            json.dumps(data)
        )
        
    async def get_cost_summary(
        self,
        start_date: date,
        end_date: date,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get cost summary for a date range
        """
        summary = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_cost": Decimal("0"),
            "daily_average": Decimal("0"),
            "category_breakdown": {},
            "service_breakdown": {},
            "top_costs": [],
            "trend": "stable"
        }
        
        # Fetch daily aggregates
        current = start_date
        daily_costs = []
        
        while current <= end_date:
            daily_key = f"cost:daily:{current}"
            item_ids = await self.redis_client.smembers(daily_key)
            
            daily_total = Decimal("0")
            for item_id in item_ids:
                item_data = await self.redis_client.get(f"cost:item:{item_id}")
                if item_data:
                    item = json.loads(item_data)
                    
                    # Apply filters
                    if user_id and item.get("user_id") != user_id:
                        continue
                    if channel_id and item.get("channel_id") != channel_id:
                        continue
                    
                    amount = Decimal(item["amount"])
                    daily_total += amount
                    
                    # Update category breakdown
                    category = item.get("category", "other")
                    summary["category_breakdown"][category] = (
                        summary["category_breakdown"].get(category, Decimal("0")) + amount
                    )
                    
                    # Update service breakdown
                    service = item.get("service", "unknown")
                    summary["service_breakdown"][service] = (
                        summary["service_breakdown"].get(service, Decimal("0")) + amount
                    )
            
            daily_costs.append(float(daily_total))
            summary["total_cost"] += daily_total
            current += timedelta(days=1)
        
        # Calculate averages and trends
        if daily_costs:
            summary["daily_average"] = summary["total_cost"] / len(daily_costs)
            
            # Calculate trend (simple linear regression)
            if len(daily_costs) > 1:
                x = np.arange(len(daily_costs))
                y = np.array(daily_costs)
                slope = np.polyfit(x, y, 1)[0]
                
                if slope > 0.1:
                    summary["trend"] = "increasing"
                elif slope < -0.1:
                    summary["trend"] = "decreasing"
        
        # Convert Decimal to float for JSON serialization
        summary["total_cost"] = float(summary["total_cost"])
        summary["daily_average"] = float(summary["daily_average"])
        summary["category_breakdown"] = {
            k: float(v) for k, v in summary["category_breakdown"].items()
        }
        summary["service_breakdown"] = {
            k: float(v) for k, v in summary["service_breakdown"].items()
        }
        
        return summary
        
    async def create_budget(
        self,
        name: str,
        amount: float,
        period: BillingPeriod,
        category: Optional[CostCategory] = None,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        alert_thresholds: Optional[List[int]] = None
    ) -> str:
        """
        Create a new budget
        """
        budget = Budget(
            budget_id=f"budget_{datetime.utcnow().timestamp()}",
            name=name,
            amount=Decimal(str(amount)),
            period=period,
            category=category,
            user_id=user_id,
            channel_id=channel_id,
            alert_thresholds=alert_thresholds or [50, 75, 90, 100]
        )
        
        self.budgets[budget.budget_id] = budget
        
        # Store in Redis
        await self.store_budget(budget)
        
        return budget.budget_id
        
    async def store_budget(self, budget: Budget):
        """Store budget in Redis"""
        key = f"budget:{budget.budget_id}"
        data = asdict(budget)
        data["amount"] = str(budget.amount)
        data["created_at"] = budget.created_at.isoformat()
        
        await self.redis_client.set(key, json.dumps(data))
        
    async def load_budgets(self):
        """Load budgets from Redis"""
        pattern = "budget:*"
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
                    budget_data = json.loads(data)
                    budget = Budget(
                        budget_id=budget_data["budget_id"],
                        name=budget_data["name"],
                        amount=Decimal(budget_data["amount"]),
                        period=BillingPeriod(budget_data["period"]),
                        category=CostCategory(budget_data["category"]) if budget_data.get("category") else None,
                        user_id=budget_data.get("user_id"),
                        channel_id=budget_data.get("channel_id"),
                        alert_thresholds=budget_data.get("alert_thresholds", [50, 75, 90, 100]),
                        created_at=datetime.fromisoformat(budget_data["created_at"]),
                        active=budget_data.get("active", True)
                    )
                    self.budgets[budget.budget_id] = budget
            
            if cursor == 0:
                break
                
    async def monitor_budgets(self):
        """
        Background task to monitor budget usage
        """
        while True:
            try:
                await asyncio.sleep(600)  # Check every 10 minutes
                
                for budget in self.budgets.values():
                    if budget.active:
                        await self.check_budget_status(budget)
                        
            except Exception as e:
                logger.error(f"Budget monitoring error: {e}")
                
    async def check_budget_status(self, budget: Budget):
        """Check budget usage and send alerts"""
        # Get current period dates
        period_start, period_end = self.get_budget_period(budget)
        
        # Get cost summary for the period
        summary = await self.get_cost_summary(
            start_date=period_start.date(),
            end_date=period_end.date(),
            user_id=budget.user_id,
            channel_id=budget.channel_id
        )
        
        current_usage = Decimal(str(summary["total_cost"]))
        usage_percentage = (current_usage / budget.amount * 100) if budget.amount > 0 else 0
        
        # Check alert thresholds
        for threshold in budget.alert_thresholds:
            if usage_percentage >= threshold:
                await self.send_budget_alert(
                    budget,
                    current_usage,
                    usage_percentage,
                    threshold
                )
                break  # Only send highest threshold alert
                
    async def send_budget_alert(
        self,
        budget: Budget,
        current_usage: Decimal,
        usage_percentage: float,
        threshold: int
    ):
        """Send budget alert"""
        alert_key = f"{budget.budget_id}:{threshold}"
        
        # Check cooldown
        if alert_key in self.last_alerts:
            if datetime.utcnow() - self.last_alerts[alert_key] < timedelta(seconds=self.alert_cooldown):
                return
        
        alert_type = AlertType.BUDGET_WARNING
        if threshold >= 100:
            alert_type = AlertType.BUDGET_CRITICAL
        
        alert = {
            "type": alert_type.value,
            "budget_id": budget.budget_id,
            "budget_name": budget.name,
            "threshold": threshold,
            "current_usage": float(current_usage),
            "budget_amount": float(budget.amount),
            "usage_percentage": usage_percentage,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store alert
        await self.redis_client.rpush("alerts:budget", json.dumps(alert))
        
        # Update last alert time
        self.last_alerts[alert_key] = datetime.utcnow()
        
        logger.warning(f"Budget alert: {budget.name} at {usage_percentage:.1f}% (${current_usage:.2f}/${budget.amount:.2f})")
        
    async def check_cost_alerts(self, cost_item: CostItem):
        """Check for immediate cost alerts"""
        # Check for unusual spikes
        if cost_item.amount > Decimal("100"):  # High single cost
            alert = {
                "type": AlertType.UNUSUAL_SPIKE.value,
                "cost_item_id": cost_item.item_id,
                "amount": float(cost_item.amount),
                "category": cost_item.category.value,
                "service": cost_item.service,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.rpush("alerts:cost", json.dumps(alert))
            logger.warning(f"High cost alert: ${cost_item.amount:.2f} for {cost_item.service}")
            
    async def generate_forecasts(self):
        """
        Background task to generate cost forecasts
        """
        while True:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                # Generate forecasts for different periods
                for period in [BillingPeriod.DAILY, BillingPeriod.WEEKLY, BillingPeriod.MONTHLY]:
                    await self.generate_forecast(period)
                    
            except Exception as e:
                logger.error(f"Forecast generation error: {e}")
                
    async def generate_forecast(self, period: BillingPeriod) -> CostForecast:
        """
        Generate cost forecast for a period
        """
        # Get historical data (last 30 days)
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        
        summary = await self.get_cost_summary(start_date, end_date)
        
        # Simple forecast based on average and trend
        daily_avg = Decimal(str(summary["daily_average"]))
        
        if period == BillingPeriod.DAILY:
            predicted = daily_avg
        elif period == BillingPeriod.WEEKLY:
            predicted = daily_avg * 7
        elif period == BillingPeriod.MONTHLY:
            predicted = daily_avg * 30
        else:
            predicted = daily_avg
        
        # Add variance for confidence intervals
        variance = predicted * Decimal("0.2")  # 20% variance
        
        forecast = CostForecast(
            forecast_date=date.today() + timedelta(days=1),
            period=period,
            predicted_cost=predicted,
            confidence_interval_low=predicted - variance,
            confidence_interval_high=predicted + variance,
            trend=summary["trend"],
            seasonality_factor=1.0  # Simplified, no seasonality
        )
        
        # Store forecast
        await self.store_forecast(forecast)
        
        return forecast
        
    async def store_forecast(self, forecast: CostForecast):
        """Store cost forecast"""
        key = f"forecast:{forecast.period.value}:{forecast.forecast_date}"
        
        data = {
            "forecast_date": forecast.forecast_date.isoformat(),
            "period": forecast.period.value,
            "predicted_cost": str(forecast.predicted_cost),
            "confidence_interval_low": str(forecast.confidence_interval_low),
            "confidence_interval_high": str(forecast.confidence_interval_high),
            "trend": forecast.trend,
            "seasonality_factor": forecast.seasonality_factor
        }
        
        await self.redis_client.setex(
            key,
            86400 * 30,  # 30 days retention
            json.dumps(data)
        )
        
    async def cleanup_old_data(self):
        """
        Background task to cleanup old cost data
        """
        while True:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                # Remove data older than retention period
                cutoff_date = date.today() - timedelta(days=90)
                
                # Clean up daily sets
                current = cutoff_date - timedelta(days=30)
                while current < cutoff_date:
                    daily_key = f"cost:daily:{current}"
                    await self.redis_client.delete(daily_key)
                    current += timedelta(days=1)
                    
                logger.info(f"Cleaned up cost data older than {cutoff_date}")
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                
    def get_period_key(self, timestamp: datetime, period: BillingPeriod) -> str:
        """Get period key for a timestamp"""
        if period == BillingPeriod.HOURLY:
            return timestamp.strftime("%Y%m%d%H")
        elif period == BillingPeriod.DAILY:
            return timestamp.strftime("%Y%m%d")
        elif period == BillingPeriod.WEEKLY:
            return timestamp.strftime("%Y%W")
        elif period == BillingPeriod.MONTHLY:
            return timestamp.strftime("%Y%m")
        elif period == BillingPeriod.QUARTERLY:
            quarter = (timestamp.month - 1) // 3 + 1
            return f"{timestamp.year}Q{quarter}"
        elif period == BillingPeriod.YEARLY:
            return str(timestamp.year)
        return timestamp.strftime("%Y%m%d")
        
    def parse_period_key(self, period_key: str, period: BillingPeriod) -> datetime:
        """Parse period key to datetime"""
        if period == BillingPeriod.HOURLY:
            return datetime.strptime(period_key, "%Y%m%d%H")
        elif period == BillingPeriod.DAILY:
            return datetime.strptime(period_key, "%Y%m%d")
        elif period == BillingPeriod.WEEKLY:
            return datetime.strptime(period_key + "1", "%Y%W%w")
        elif period == BillingPeriod.MONTHLY:
            return datetime.strptime(period_key, "%Y%m")
        elif period == BillingPeriod.YEARLY:
            return datetime.strptime(period_key, "%Y")
        return datetime.utcnow()
        
    def get_period_end(self, period_key: str, period: BillingPeriod) -> datetime:
        """Get end of period"""
        start = self.parse_period_key(period_key, period)
        
        if period == BillingPeriod.HOURLY:
            return start + timedelta(hours=1)
        elif period == BillingPeriod.DAILY:
            return start + timedelta(days=1)
        elif period == BillingPeriod.WEEKLY:
            return start + timedelta(weeks=1)
        elif period == BillingPeriod.MONTHLY:
            return start + timedelta(days=30)
        elif period == BillingPeriod.YEARLY:
            return start + timedelta(days=365)
        return start
        
    def get_budget_period(self, budget: Budget) -> Tuple[datetime, datetime]:
        """Get current budget period dates"""
        now = datetime.utcnow()
        
        if budget.period == BillingPeriod.DAILY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif budget.period == BillingPeriod.WEEKLY:
            days_since_monday = now.weekday()
            start = now - timedelta(days=days_since_monday)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(weeks=1)
        elif budget.period == BillingPeriod.MONTHLY:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # Next month
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=now.month + 1)
        else:
            start = now
            end = now + timedelta(days=1)
            
        return start, end


# Global instance
cost_pipeline = CostAggregationPipeline()