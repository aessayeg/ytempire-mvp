"""
Cost Analytics Implementation
Comprehensive cost tracking, analysis, and optimization system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import redis.asyncio as redis
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import boto3
from google.cloud import billing_v1
import logging
from prometheus_client import Counter, Histogram, Gauge
import json

# Metrics
cost_tracked = Counter('costs_tracked_total', 'Total costs tracked', ['service', 'category'])
cost_anomaly_detected = Counter('cost_anomalies_detected', 'Cost anomalies detected', ['service'])
cost_optimization_savings = Gauge('cost_optimization_savings', 'Estimated savings from optimizations', ['optimization_type'])
budget_utilization = Gauge('budget_utilization_percent', 'Budget utilization percentage', ['budget_name'])

logger = logging.getLogger(__name__)

Base = declarative_base()

@dataclass
class CostItem:
    """Individual cost item"""
    service: str
    resource_id: str
    resource_type: str
    amount: Decimal
    currency: str
    timestamp: datetime
    usage_quantity: float
    usage_unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CostAllocation:
    """Cost allocation configuration"""
    name: str
    rules: List[Dict[str, Any]]
    allocation_method: str  # 'proportional', 'fixed', 'usage_based'
    target_dimensions: List[str]  # ['department', 'project', 'team']

@dataclass
class Budget:
    """Budget configuration"""
    name: str
    amount: Decimal
    period: str  # 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
    category: Optional[str] = None
    service: Optional[str] = None
    alert_thresholds: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.0])
    owner: Optional[str] = None

class CostRecord(Base):
    """Cost record database model"""
    __tablename__ = 'cost_records'
    
    id = Column(Integer, primary_key=True)
    service = Column(String(100), index=True)
    resource_id = Column(String(255))
    resource_type = Column(String(100))
    amount = Column(Float)
    currency = Column(String(10))
    timestamp = Column(DateTime, index=True)
    usage_quantity = Column(Float)
    usage_unit = Column(String(50))
    tags = Column(JSON)
    metadata = Column(JSON)
    allocated_to = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

class CostOptimization(Base):
    """Cost optimization recommendations"""
    __tablename__ = 'cost_optimizations'
    
    id = Column(Integer, primary_key=True)
    optimization_type = Column(String(100))
    resource_id = Column(String(255))
    current_cost = Column(Float)
    estimated_savings = Column(Float)
    recommendation = Column(String(1000))
    priority = Column(String(20))  # 'high', 'medium', 'low'
    status = Column(String(20))  # 'pending', 'implemented', 'dismissed'
    created_at = Column(DateTime, default=datetime.utcnow)
    implemented_at = Column(DateTime, nullable=True)

class CostAnalytics:
    """Comprehensive cost analytics and optimization system"""
    
    def __init__(self,
                 db_url: str,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 aws_region: str = 'us-east-1',
                 gcp_project_id: str = None):
        
        # Database setup
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Redis for caching
        self.redis_client = None
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # Cloud provider clients
        self.aws_ce_client = boto3.client('ce', region_name=aws_region)  # Cost Explorer
        self.aws_cw_client = boto3.client('cloudwatch', region_name=aws_region)
        
        if gcp_project_id:
            self.gcp_billing_client = billing_v1.CloudBillingClient()
            self.gcp_project_id = gcp_project_id
        else:
            self.gcp_billing_client = None
        
        # Cost configurations
        self.budgets = {}
        self.allocations = {}
        self._initialize_configurations()
    
    async def initialize(self):
        """Initialize async components"""
        self.redis_client = await redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=True
        )
    
    def _initialize_configurations(self):
        """Initialize cost tracking configurations"""
        
        # Define budgets
        self.budgets = {
            'infrastructure': Budget(
                name='infrastructure',
                amount=Decimal('10000'),
                period='monthly',
                category='infrastructure',
                alert_thresholds=[0.7, 0.85, 0.95, 1.0]
            ),
            'ai_compute': Budget(
                name='ai_compute',
                amount=Decimal('5000'),
                period='monthly',
                service='gpu_instances',
                alert_thresholds=[0.8, 0.9, 1.0]
            ),
            'data_storage': Budget(
                name='data_storage',
                amount=Decimal('2000'),
                period='monthly',
                category='storage',
                alert_thresholds=[0.75, 0.9]
            ),
            'api_calls': Budget(
                name='api_calls',
                amount=Decimal('3000'),
                period='monthly',
                service='openai',
                alert_thresholds=[0.8, 0.95]
            )
        }
        
        # Define allocation rules
        self.allocations = {
            'department': CostAllocation(
                name='department',
                allocation_method='usage_based',
                target_dimensions=['engineering', 'data_science', 'operations'],
                rules=[
                    {'pattern': 'gpu-*', 'allocate_to': 'data_science'},
                    {'pattern': 'api-*', 'allocate_to': 'engineering'},
                    {'pattern': 'db-*', 'allocate_to': 'operations'}
                ]
            ),
            'project': CostAllocation(
                name='project',
                allocation_method='proportional',
                target_dimensions=['youtube_automation', 'analytics', 'infrastructure'],
                rules=[
                    {'service': 'openai', 'allocate_to': 'youtube_automation', 'weight': 0.8},
                    {'service': 's3', 'allocate_to': 'analytics', 'weight': 0.6}
                ]
            )
        }
    
    async def track_cost(self, cost_item: CostItem) -> Dict[str, Any]:
        """
        Track a cost item
        
        Args:
            cost_item: Cost item to track
            
        Returns:
            Tracking result with allocation and budget status
        """
        # Allocate cost
        allocation = self._allocate_cost(cost_item)
        
        # Store in database
        cost_record = CostRecord(
            service=cost_item.service,
            resource_id=cost_item.resource_id,
            resource_type=cost_item.resource_type,
            amount=float(cost_item.amount),
            currency=cost_item.currency,
            timestamp=cost_item.timestamp,
            usage_quantity=cost_item.usage_quantity,
            usage_unit=cost_item.usage_unit,
            tags=cost_item.tags,
            metadata=cost_item.metadata,
            allocated_to=allocation
        )
        
        self.session.add(cost_record)
        self.session.commit()
        
        # Update metrics
        cost_tracked.labels(
            service=cost_item.service,
            category=cost_item.resource_type
        ).inc(float(cost_item.amount))
        
        # Check budget
        budget_status = await self._check_budget_impact(cost_item)
        
        # Detect anomalies
        is_anomaly = await self._detect_cost_anomaly(cost_item)
        
        if is_anomaly:
            cost_anomaly_detected.labels(service=cost_item.service).inc()
        
        return {
            'cost_id': cost_record.id,
            'allocated_to': allocation,
            'budget_status': budget_status,
            'is_anomaly': is_anomaly
        }
    
    async def get_cost_breakdown(self,
                                start_date: datetime,
                                end_date: datetime,
                                group_by: List[str] = None) -> pd.DataFrame:
        """
        Get cost breakdown for a period
        
        Args:
            start_date: Start date
            end_date: End date
            group_by: Dimensions to group by
            
        Returns:
            DataFrame with cost breakdown
        """
        # Query database
        query = self.session.query(CostRecord).filter(
            CostRecord.timestamp >= start_date,
            CostRecord.timestamp <= end_date
        )
        
        records = query.all()
        
        # Convert to DataFrame
        data = []
        for record in records:
            data.append({
                'date': record.timestamp.date(),
                'service': record.service,
                'resource_type': record.resource_type,
                'amount': record.amount,
                'allocated_to': record.allocated_to,
                'usage_quantity': record.usage_quantity,
                'usage_unit': record.usage_unit
            })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            return df
        
        # Group by specified dimensions
        if group_by:
            grouped = df.groupby(group_by).agg({
                'amount': 'sum',
                'usage_quantity': 'sum'
            }).reset_index()
            return grouped
        
        return df
    
    async def analyze_cost_trends(self,
                                 service: Optional[str] = None,
                                 lookback_days: int = 30) -> Dict[str, Any]:
        """
        Analyze cost trends
        
        Args:
            service: Optional service filter
            lookback_days: Number of days to analyze
            
        Returns:
            Cost trend analysis
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Get historical data
        query = self.session.query(CostRecord).filter(
            CostRecord.timestamp >= start_date,
            CostRecord.timestamp <= end_date
        )
        
        if service:
            query = query.filter(CostRecord.service == service)
        
        records = query.all()
        
        # Convert to DataFrame for analysis
        data = []
        for record in records:
            data.append({
                'date': record.timestamp.date(),
                'amount': record.amount,
                'service': record.service
            })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            return {'error': 'No data available'}
        
        # Daily aggregation
        daily_costs = df.groupby('date')['amount'].sum().reset_index()
        
        # Calculate trends
        analysis = {
            'total_cost': float(df['amount'].sum()),
            'daily_average': float(daily_costs['amount'].mean()),
            'daily_std': float(daily_costs['amount'].std()),
            'trend': self._calculate_trend(daily_costs),
            'forecast': self._forecast_costs(daily_costs),
            'top_services': df.groupby('service')['amount'].sum().nlargest(5).to_dict()
        }
        
        # Identify cost spikes
        spikes = self._identify_spikes(daily_costs)
        if spikes:
            analysis['spikes'] = spikes
        
        return analysis
    
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get cost optimization recommendations
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Analyze AWS costs
        aws_recommendations = await self._analyze_aws_costs()
        recommendations.extend(aws_recommendations)
        
        # Analyze GCP costs
        if self.gcp_billing_client:
            gcp_recommendations = await self._analyze_gcp_costs()
            recommendations.extend(gcp_recommendations)
        
        # Analyze internal metrics
        internal_recommendations = await self._analyze_internal_costs()
        recommendations.extend(internal_recommendations)
        
        # Store recommendations
        for rec in recommendations:
            optimization = CostOptimization(
                optimization_type=rec['type'],
                resource_id=rec['resource_id'],
                current_cost=rec['current_cost'],
                estimated_savings=rec['estimated_savings'],
                recommendation=rec['recommendation'],
                priority=rec['priority'],
                status='pending'
            )
            self.session.add(optimization)
        
        self.session.commit()
        
        # Update metrics
        total_savings = sum(r['estimated_savings'] for r in recommendations)
        cost_optimization_savings.labels(optimization_type='all').set(total_savings)
        
        return recommendations
    
    async def _analyze_aws_costs(self) -> List[Dict[str, Any]]:
        """Analyze AWS costs for optimization"""
        recommendations = []
        
        try:
            # Get cost and usage data
            response = self.aws_ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                    'End': datetime.now().strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['UnblendedCost', 'UsageQuantity'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                    {'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}
                ]
            )
            
            # Analyze EC2 instances
            ec2_recommendations = self._analyze_ec2_rightsizing()
            recommendations.extend(ec2_recommendations)
            
            # Analyze Reserved Instance coverage
            ri_recommendations = self._analyze_ri_coverage()
            recommendations.extend(ri_recommendations)
            
            # Analyze S3 storage classes
            s3_recommendations = self._analyze_s3_optimization()
            recommendations.extend(s3_recommendations)
            
        except Exception as e:
            logger.error(f"Error analyzing AWS costs: {e}")
        
        return recommendations
    
    def _analyze_ec2_rightsizing(self) -> List[Dict[str, Any]]:
        """Analyze EC2 instances for rightsizing"""
        recommendations = []
        
        try:
            # Get EC2 utilization metrics
            response = self.aws_cw_client.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[],
                StartTime=datetime.now() - timedelta(days=7),
                EndTime=datetime.now(),
                Period=3600,
                Statistics=['Average']
            )
            
            # Analyze utilization
            if response['Datapoints']:
                avg_utilization = np.mean([dp['Average'] for dp in response['Datapoints']])
                
                if avg_utilization < 20:
                    recommendations.append({
                        'type': 'ec2_rightsizing',
                        'resource_id': 'ec2_instances',
                        'current_cost': 1000,  # Placeholder
                        'estimated_savings': 300,
                        'recommendation': 'Consider downsizing underutilized EC2 instances',
                        'priority': 'high'
                    })
        except Exception as e:
            logger.error(f"Error analyzing EC2: {e}")
        
        return recommendations
    
    def _analyze_ri_coverage(self) -> List[Dict[str, Any]]:
        """Analyze Reserved Instance coverage"""
        recommendations = []
        
        try:
            response = self.aws_ce_client.get_reservation_coverage(
                TimePeriod={
                    'Start': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    'End': datetime.now().strftime('%Y-%m-%d')
                }
            )
            
            coverage = float(response['Total']['CoverageHours']['CoverageHoursPercentage'])
            
            if coverage < 70:
                recommendations.append({
                    'type': 'reserved_instances',
                    'resource_id': 'ec2_ri',
                    'current_cost': 5000,  # Placeholder
                    'estimated_savings': 1500,
                    'recommendation': f'Increase Reserved Instance coverage (current: {coverage:.1f}%)',
                    'priority': 'medium'
                })
        except Exception as e:
            logger.error(f"Error analyzing RI coverage: {e}")
        
        return recommendations
    
    def _analyze_s3_optimization(self) -> List[Dict[str, Any]]:
        """Analyze S3 storage optimization"""
        recommendations = []
        
        # This would analyze S3 storage classes and lifecycle policies
        recommendations.append({
            'type': 's3_lifecycle',
            'resource_id': 's3_buckets',
            'current_cost': 500,
            'estimated_savings': 200,
            'recommendation': 'Implement S3 lifecycle policies for infrequent access data',
            'priority': 'medium'
        })
        
        return recommendations
    
    async def _analyze_gcp_costs(self) -> List[Dict[str, Any]]:
        """Analyze GCP costs for optimization"""
        recommendations = []
        
        # Placeholder for GCP analysis
        recommendations.append({
            'type': 'gcp_committed_use',
            'resource_id': 'gcp_compute',
            'current_cost': 3000,
            'estimated_savings': 900,
            'recommendation': 'Consider GCP Committed Use Discounts',
            'priority': 'medium'
        })
        
        return recommendations
    
    async def _analyze_internal_costs(self) -> List[Dict[str, Any]]:
        """Analyze internal service costs"""
        recommendations = []
        
        # Analyze OpenAI API usage
        openai_costs = await self._analyze_openai_usage()
        if openai_costs:
            recommendations.extend(openai_costs)
        
        # Analyze GPU utilization
        gpu_costs = await self._analyze_gpu_utilization()
        if gpu_costs:
            recommendations.extend(gpu_costs)
        
        return recommendations
    
    async def _analyze_openai_usage(self) -> List[Dict[str, Any]]:
        """Analyze OpenAI API usage for optimization"""
        recommendations = []
        
        # Get recent OpenAI costs
        query = self.session.query(CostRecord).filter(
            CostRecord.service == 'openai',
            CostRecord.timestamp >= datetime.now() - timedelta(days=7)
        )
        
        records = query.all()
        
        if records:
            total_cost = sum(r.amount for r in records)
            total_tokens = sum(r.usage_quantity for r in records)
            
            # Check for optimization opportunities
            if total_tokens > 1000000:  # More than 1M tokens
                recommendations.append({
                    'type': 'openai_optimization',
                    'resource_id': 'openai_api',
                    'current_cost': float(total_cost),
                    'estimated_savings': float(total_cost * 0.2),
                    'recommendation': 'Consider caching frequent prompts or using smaller models',
                    'priority': 'high'
                })
        
        return recommendations
    
    async def _analyze_gpu_utilization(self) -> List[Dict[str, Any]]:
        """Analyze GPU utilization for optimization"""
        recommendations = []
        
        # Check GPU utilization from metrics
        if self.redis_client:
            utilization = await self.redis_client.get('gpu_utilization_avg')
            
            if utilization and float(utilization) < 50:
                recommendations.append({
                    'type': 'gpu_optimization',
                    'resource_id': 'gpu_instances',
                    'current_cost': 2000,
                    'estimated_savings': 800,
                    'recommendation': 'GPU underutilized - consider batch processing or smaller instances',
                    'priority': 'high'
                })
        
        return recommendations
    
    def _allocate_cost(self, cost_item: CostItem) -> str:
        """Allocate cost based on rules"""
        # Try department allocation first
        dept_allocation = self.allocations.get('department')
        if dept_allocation:
            for rule in dept_allocation.rules:
                if 'pattern' in rule:
                    import re
                    if re.match(rule['pattern'], cost_item.resource_id):
                        return rule['allocate_to']
        
        # Try project allocation
        proj_allocation = self.allocations.get('project')
        if proj_allocation:
            for rule in proj_allocation.rules:
                if rule.get('service') == cost_item.service:
                    return rule['allocate_to']
        
        return 'unallocated'
    
    async def _check_budget_impact(self, cost_item: CostItem) -> Dict[str, Any]:
        """Check impact on budgets"""
        budget_status = {}
        
        for budget_name, budget in self.budgets.items():
            # Check if cost applies to this budget
            if budget.service and budget.service != cost_item.service:
                continue
            if budget.category and budget.category != cost_item.resource_type:
                continue
            
            # Get current spend
            current_spend = await self._get_budget_spend(budget_name)
            new_spend = current_spend + float(cost_item.amount)
            utilization = (new_spend / float(budget.amount)) * 100
            
            # Update metric
            budget_utilization.labels(budget_name=budget_name).set(utilization)
            
            # Check thresholds
            for threshold in budget.alert_thresholds:
                if utilization >= threshold * 100 and (current_spend / float(budget.amount)) < threshold:
                    await self._trigger_budget_alert(budget_name, utilization, threshold)
            
            budget_status[budget_name] = {
                'utilization': utilization,
                'remaining': float(budget.amount) - new_spend,
                'exceeded': new_spend > float(budget.amount)
            }
        
        return budget_status
    
    async def _get_budget_spend(self, budget_name: str) -> float:
        """Get current budget spend"""
        budget = self.budgets[budget_name]
        
        # Determine period
        if budget.period == 'daily':
            start_date = datetime.now().replace(hour=0, minute=0, second=0)
        elif budget.period == 'weekly':
            start_date = datetime.now() - timedelta(days=datetime.now().weekday())
        elif budget.period == 'monthly':
            start_date = datetime.now().replace(day=1)
        else:
            start_date = datetime.now() - timedelta(days=30)
        
        # Query spend
        query = self.session.query(CostRecord).filter(
            CostRecord.timestamp >= start_date
        )
        
        if budget.service:
            query = query.filter(CostRecord.service == budget.service)
        if budget.category:
            query = query.filter(CostRecord.resource_type == budget.category)
        
        records = query.all()
        return sum(r.amount for r in records)
    
    async def _trigger_budget_alert(self, budget_name: str, utilization: float, threshold: float):
        """Trigger budget alert"""
        logger.warning(f"Budget alert: {budget_name} at {utilization:.1f}% (threshold: {threshold*100}%)")
        
        # Send notification (implement notification service)
        # await notification_service.send_alert(...)
    
    async def _detect_cost_anomaly(self, cost_item: CostItem) -> bool:
        """Detect cost anomalies"""
        # Get historical average for this service
        avg_cost = await self._get_average_cost(cost_item.service, cost_item.resource_type)
        
        if avg_cost == 0:
            return False
        
        # Check if current cost is anomalous (> 3 std deviations)
        std_dev = await self._get_cost_std_dev(cost_item.service, cost_item.resource_type)
        
        if abs(float(cost_item.amount) - avg_cost) > 3 * std_dev:
            return True
        
        return False
    
    async def _get_average_cost(self, service: str, resource_type: str) -> float:
        """Get average cost for service/resource"""
        # Check cache
        if self.redis_client:
            cached = await self.redis_client.get(f"avg_cost:{service}:{resource_type}")
            if cached:
                return float(cached)
        
        # Query database
        query = self.session.query(CostRecord).filter(
            CostRecord.service == service,
            CostRecord.resource_type == resource_type,
            CostRecord.timestamp >= datetime.now() - timedelta(days=30)
        )
        
        records = query.all()
        
        if not records:
            return 0
        
        avg_cost = np.mean([r.amount for r in records])
        
        # Cache result
        if self.redis_client:
            await self.redis_client.setex(
                f"avg_cost:{service}:{resource_type}",
                3600,
                str(avg_cost)
            )
        
        return avg_cost
    
    async def _get_cost_std_dev(self, service: str, resource_type: str) -> float:
        """Get cost standard deviation"""
        query = self.session.query(CostRecord).filter(
            CostRecord.service == service,
            CostRecord.resource_type == resource_type,
            CostRecord.timestamp >= datetime.now() - timedelta(days=30)
        )
        
        records = query.all()
        
        if len(records) < 2:
            return 0
        
        return np.std([r.amount for r in records])
    
    def _calculate_trend(self, daily_costs: pd.DataFrame) -> str:
        """Calculate cost trend"""
        if len(daily_costs) < 3:
            return 'insufficient_data'
        
        # Simple linear regression
        x = np.arange(len(daily_costs))
        y = daily_costs['amount'].values
        
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _forecast_costs(self, daily_costs: pd.DataFrame, days_ahead: int = 7) -> List[float]:
        """Forecast future costs"""
        if len(daily_costs) < 7:
            return []
        
        # Simple moving average forecast
        ma_window = min(7, len(daily_costs))
        recent_avg = daily_costs['amount'].tail(ma_window).mean()
        
        # Add trend component
        x = np.arange(len(daily_costs))
        y = daily_costs['amount'].values
        slope = np.polyfit(x, y, 1)[0]
        
        forecast = []
        for i in range(days_ahead):
            forecast_value = recent_avg + slope * i
            forecast.append(max(0, forecast_value))
        
        return forecast
    
    def _identify_spikes(self, daily_costs: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify cost spikes"""
        if len(daily_costs) < 3:
            return []
        
        spikes = []
        mean_cost = daily_costs['amount'].mean()
        std_cost = daily_costs['amount'].std()
        
        for idx, row in daily_costs.iterrows():
            if row['amount'] > mean_cost + 2 * std_cost:
                spikes.append({
                    'date': str(row['date']),
                    'amount': float(row['amount']),
                    'deviation': float((row['amount'] - mean_cost) / std_cost)
                })
        
        return spikes

# Example usage
async def main():
    # Initialize cost analytics
    cost_analytics = CostAnalytics(
        db_url='postgresql://user:pass@localhost/costs',
        aws_region='us-east-1',
        gcp_project_id='ytempire-project'
    )
    
    await cost_analytics.initialize()
    
    # Track a cost item
    cost_item = CostItem(
        service='openai',
        resource_id='api_call_123',
        resource_type='api',
        amount=Decimal('1.50'),
        currency='USD',
        timestamp=datetime.now(),
        usage_quantity=1000,
        usage_unit='tokens'
    )
    
    result = await cost_analytics.track_cost(cost_item)
    print(f"Cost tracked: {result}")
    
    # Get cost breakdown
    breakdown = await cost_analytics.get_cost_breakdown(
        datetime.now() - timedelta(days=7),
        datetime.now(),
        group_by=['service', 'date']
    )
    print(f"Cost breakdown shape: {breakdown.shape}")
    
    # Analyze trends
    trends = await cost_analytics.analyze_cost_trends(lookback_days=30)
    print(f"Cost trends: {trends}")
    
    # Get optimization recommendations
    recommendations = await cost_analytics.get_optimization_recommendations()
    print(f"Found {len(recommendations)} optimization recommendations")

if __name__ == "__main__":
    asyncio.run(main())