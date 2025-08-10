"""
Business Dashboard Data Pipeline
ETL and data aggregation for business intelligence dashboards
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
import asyncio
import aiohttp
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.sql import select, func, and_, or_
from sqlalchemy.orm import sessionmaker
import redis.asyncio as redis
import logging
from prometheus_client import Counter, Histogram, Gauge
import json
import pyarrow as pa
import pyarrow.parquet as pq
from decimal import Decimal

# Metrics
etl_pipeline_duration = Histogram('etl_pipeline_duration', 'ETL pipeline execution time', ['pipeline'])
data_rows_processed = Counter('dashboard_data_rows_processed', 'Total rows processed', ['table'])
data_quality_score = Gauge('data_quality_score', 'Data quality score', ['dataset'])
aggregation_time = Histogram('aggregation_time', 'Time to compute aggregations', ['metric'])

logger = logging.getLogger(__name__)

@dataclass
class DashboardMetric:
    """Dashboard metric definition"""
    name: str
    display_name: str
    calculation: str  # SQL or function name
    aggregation: str  # 'sum', 'avg', 'count', 'max', 'min'
    dimensions: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    time_grain: str = 'daily'  # 'hourly', 'daily', 'weekly', 'monthly'
    cache_ttl: int = 3600

@dataclass
class DataPipelineConfig:
    """Data pipeline configuration"""
    name: str
    source_tables: List[str]
    target_table: str
    schedule: str  # cron expression
    transformations: List[callable] = field(default_factory=list)
    quality_checks: List[callable] = field(default_factory=list)

class BusinessDashboardData:
    """Business dashboard data pipeline and aggregation system"""
    
    def __init__(self,
                 db_url: str,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 data_warehouse_url: str = None):
        
        # Database connections
        self.db_engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.db_engine)
        
        # Data warehouse connection (if separate)
        if data_warehouse_url:
            self.dw_engine = create_engine(data_warehouse_url)
        else:
            self.dw_engine = self.db_engine
        
        # Redis for caching
        self.redis_client = None
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # Metric definitions
        self.metrics = {}
        self._register_metrics()
        
        # Pipeline configurations
        self.pipelines = {}
        self._register_pipelines()
    
    async def initialize(self):
        """Initialize async components"""
        self.redis_client = await redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=True
        )
        
        # Create aggregation tables
        await self._create_aggregation_tables()
    
    def _register_metrics(self):
        """Register all dashboard metrics"""
        
        # Revenue metrics
        self.metrics['total_revenue'] = DashboardMetric(
            name='total_revenue',
            display_name='Total Revenue',
            calculation='SELECT SUM(amount) FROM transactions WHERE status = "completed"',
            aggregation='sum',
            dimensions=['date', 'product', 'region'],
            time_grain='daily'
        )
        
        self.metrics['mrr'] = DashboardMetric(
            name='mrr',
            display_name='Monthly Recurring Revenue',
            calculation='SELECT SUM(amount) FROM subscriptions WHERE status = "active"',
            aggregation='sum',
            dimensions=['date', 'plan_type'],
            time_grain='monthly'
        )
        
        self.metrics['arpu'] = DashboardMetric(
            name='arpu',
            display_name='Average Revenue Per User',
            calculation='SELECT SUM(amount) / COUNT(DISTINCT user_id) FROM transactions',
            aggregation='avg',
            dimensions=['date', 'user_segment'],
            time_grain='monthly'
        )
        
        # User metrics
        self.metrics['dau'] = DashboardMetric(
            name='dau',
            display_name='Daily Active Users',
            calculation='SELECT COUNT(DISTINCT user_id) FROM user_activity',
            aggregation='count',
            dimensions=['date', 'platform'],
            time_grain='daily'
        )
        
        self.metrics['mau'] = DashboardMetric(
            name='mau',
            display_name='Monthly Active Users',
            calculation='SELECT COUNT(DISTINCT user_id) FROM user_activity WHERE date >= DATE_SUB(NOW(), INTERVAL 30 DAY)',
            aggregation='count',
            dimensions=['month', 'platform'],
            time_grain='monthly'
        )
        
        self.metrics['user_retention'] = DashboardMetric(
            name='user_retention',
            display_name='User Retention Rate',
            calculation=self._calculate_retention,
            aggregation='avg',
            dimensions=['cohort_month', 'days_since_signup'],
            time_grain='daily'
        )
        
        # Content metrics
        self.metrics['videos_created'] = DashboardMetric(
            name='videos_created',
            display_name='Videos Created',
            calculation='SELECT COUNT(*) FROM videos',
            aggregation='count',
            dimensions=['date', 'category', 'creator_tier'],
            time_grain='daily'
        )
        
        self.metrics['avg_video_performance'] = DashboardMetric(
            name='avg_video_performance',
            display_name='Average Video Performance',
            calculation='SELECT AVG(views * engagement_rate) FROM video_analytics',
            aggregation='avg',
            dimensions=['date', 'category'],
            time_grain='daily'
        )
        
        # Engagement metrics
        self.metrics['engagement_rate'] = DashboardMetric(
            name='engagement_rate',
            display_name='Engagement Rate',
            calculation='SELECT (SUM(likes) + SUM(comments) + SUM(shares)) / SUM(views) FROM video_analytics',
            aggregation='avg',
            dimensions=['date', 'content_type'],
            time_grain='daily'
        )
        
        # Cost metrics
        self.metrics['total_costs'] = DashboardMetric(
            name='total_costs',
            display_name='Total Costs',
            calculation='SELECT SUM(amount) FROM costs',
            aggregation='sum',
            dimensions=['date', 'cost_category', 'department'],
            time_grain='daily'
        )
        
        self.metrics['cost_per_user'] = DashboardMetric(
            name='cost_per_user',
            display_name='Cost Per User',
            calculation=self._calculate_cost_per_user,
            aggregation='avg',
            dimensions=['date', 'user_segment'],
            time_grain='monthly'
        )
        
        # Performance metrics
        self.metrics['api_latency'] = DashboardMetric(
            name='api_latency',
            display_name='API Latency',
            calculation='SELECT AVG(response_time) FROM api_logs',
            aggregation='avg',
            dimensions=['date', 'endpoint', 'method'],
            time_grain='hourly'
        )
        
        self.metrics['error_rate'] = DashboardMetric(
            name='error_rate',
            display_name='Error Rate',
            calculation='SELECT COUNT(CASE WHEN status >= 400 THEN 1 END) / COUNT(*) FROM api_logs',
            aggregation='avg',
            dimensions=['date', 'service'],
            time_grain='hourly'
        )
    
    def _register_pipelines(self):
        """Register data pipelines"""
        
        # Revenue pipeline
        self.pipelines['revenue'] = DataPipelineConfig(
            name='revenue',
            source_tables=['transactions', 'subscriptions', 'refunds'],
            target_table='revenue_summary',
            schedule='0 * * * *',  # Hourly
            transformations=[
                self._transform_revenue_data,
                self._calculate_net_revenue
            ],
            quality_checks=[
                self._check_revenue_consistency,
                self._check_negative_amounts
            ]
        )
        
        # User analytics pipeline
        self.pipelines['user_analytics'] = DataPipelineConfig(
            name='user_analytics',
            source_tables=['user_activity', 'users', 'sessions'],
            target_table='user_analytics_summary',
            schedule='0 */4 * * *',  # Every 4 hours
            transformations=[
                self._transform_user_data,
                self._calculate_user_segments
            ],
            quality_checks=[
                self._check_user_data_completeness
            ]
        )
        
        # Content performance pipeline
        self.pipelines['content_performance'] = DataPipelineConfig(
            name='content_performance',
            source_tables=['videos', 'video_analytics', 'channel_stats'],
            target_table='content_performance_summary',
            schedule='0 */2 * * *',  # Every 2 hours
            transformations=[
                self._transform_content_data,
                self._calculate_performance_scores
            ]
        )
    
    async def run_pipeline(self, pipeline_name: str) -> Dict[str, Any]:
        """
        Run a specific data pipeline
        
        Args:
            pipeline_name: Name of the pipeline to run
            
        Returns:
            Pipeline execution results
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
        
        pipeline = self.pipelines[pipeline_name]
        start_time = datetime.now()
        
        try:
            # Extract data
            source_data = await self._extract_data(pipeline.source_tables)
            
            # Transform data
            transformed_data = source_data
            for transformation in pipeline.transformations:
                if asyncio.iscoroutinefunction(transformation):
                    transformed_data = await transformation(transformed_data)
                else:
                    transformed_data = transformation(transformed_data)
            
            # Run quality checks
            for check in pipeline.quality_checks:
                if asyncio.iscoroutinefunction(check):
                    result = await check(transformed_data)
                else:
                    result = check(transformed_data)
                
                if not result:
                    raise ValueError(f"Quality check failed: {check.__name__}")
            
            # Load data
            rows_loaded = await self._load_data(transformed_data, pipeline.target_table)
            
            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            etl_pipeline_duration.labels(pipeline=pipeline_name).observe(execution_time)
            data_rows_processed.labels(table=pipeline.target_table).inc(rows_loaded)
            
            return {
                'status': 'success',
                'pipeline': pipeline_name,
                'rows_processed': rows_loaded,
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.error(f"Pipeline {pipeline_name} failed: {e}")
            return {
                'status': 'failed',
                'pipeline': pipeline_name,
                'error': str(e)
            }
    
    async def _extract_data(self, tables: List[str]) -> Dict[str, pd.DataFrame]:
        """Extract data from source tables"""
        data = {}
        
        for table in tables:
            query = f"SELECT * FROM {table} WHERE updated_at >= NOW() - INTERVAL 1 DAY"
            
            with self.db_engine.connect() as conn:
                df = pd.read_sql(query, conn)
                data[table] = df
                
                logger.info(f"Extracted {len(df)} rows from {table}")
        
        return data
    
    async def _load_data(self, data: pd.DataFrame, target_table: str) -> int:
        """Load data into target table"""
        rows_before = 0
        
        # Get current row count
        with self.dw_engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {target_table}"))
            rows_before = result.scalar() or 0
        
        # Load data
        data.to_sql(target_table, self.dw_engine, if_exists='append', index=False)
        
        # Get new row count
        with self.dw_engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {target_table}"))
            rows_after = result.scalar() or 0
        
        return rows_after - rows_before
    
    async def get_metric(self,
                        metric_name: str,
                        start_date: datetime,
                        end_date: datetime,
                        dimensions: Optional[List[str]] = None,
                        filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Get metric data for dashboard
        
        Args:
            metric_name: Name of the metric
            start_date: Start date for data
            end_date: End date for data
            dimensions: Dimensions to group by
            filters: Additional filters
            
        Returns:
            DataFrame with metric data
        """
        if metric_name not in self.metrics:
            raise ValueError(f"Metric {metric_name} not found")
        
        metric = self.metrics[metric_name]
        
        # Check cache
        cache_key = self._get_cache_key(metric_name, start_date, end_date, dimensions, filters)
        cached_data = await self._get_cached_data(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        # Calculate metric
        start_time = datetime.now()
        
        if isinstance(metric.calculation, str):
            # SQL query
            data = await self._execute_metric_query(metric, start_date, end_date, dimensions, filters)
        else:
            # Function
            data = await metric.calculation(start_date, end_date, dimensions, filters)
        
        # Record metrics
        calculation_time = (datetime.now() - start_time).total_seconds()
        aggregation_time.labels(metric=metric_name).observe(calculation_time)
        
        # Cache result
        await self._cache_data(cache_key, data, metric.cache_ttl)
        
        return data
    
    async def _execute_metric_query(self,
                                   metric: DashboardMetric,
                                   start_date: datetime,
                                   end_date: datetime,
                                   dimensions: Optional[List[str]],
                                   filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Execute metric SQL query"""
        query = metric.calculation
        
        # Add date filters
        date_filter = f" AND date >= '{start_date}' AND date <= '{end_date}'"
        if 'WHERE' in query:
            query = query.replace('WHERE', f'WHERE{date_filter} AND')
        else:
            query = query.replace('FROM', f'FROM WHERE{date_filter}')
        
        # Add custom filters
        if filters:
            for key, value in filters.items():
                query += f" AND {key} = '{value}'"
        
        # Add GROUP BY for dimensions
        if dimensions:
            query += f" GROUP BY {', '.join(dimensions)}"
        
        with self.dw_engine.connect() as conn:
            return pd.read_sql(query, conn)
    
    async def get_dashboard_data(self,
                                dashboard_id: str,
                                time_range: str = '7d') -> Dict[str, Any]:
        """
        Get all data for a specific dashboard
        
        Args:
            dashboard_id: Dashboard identifier
            time_range: Time range ('24h', '7d', '30d', '90d')
            
        Returns:
            Dictionary with all dashboard data
        """
        # Define dashboard configurations
        dashboards = {
            'executive': ['total_revenue', 'mrr', 'dau', 'mau', 'user_retention'],
            'revenue': ['total_revenue', 'mrr', 'arpu', 'cost_per_user'],
            'engagement': ['dau', 'mau', 'engagement_rate', 'avg_video_performance'],
            'operations': ['api_latency', 'error_rate', 'total_costs']
        }
        
        if dashboard_id not in dashboards:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        # Calculate date range
        end_date = datetime.now()
        if time_range == '24h':
            start_date = end_date - timedelta(hours=24)
        elif time_range == '7d':
            start_date = end_date - timedelta(days=7)
        elif time_range == '30d':
            start_date = end_date - timedelta(days=30)
        elif time_range == '90d':
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=7)
        
        # Fetch all metrics for dashboard
        dashboard_data = {
            'dashboard_id': dashboard_id,
            'time_range': time_range,
            'generated_at': datetime.now().isoformat(),
            'metrics': {}
        }
        
        for metric_name in dashboards[dashboard_id]:
            try:
                metric_data = await self.get_metric(
                    metric_name,
                    start_date,
                    end_date
                )
                
                dashboard_data['metrics'][metric_name] = {
                    'data': metric_data.to_dict('records'),
                    'summary': {
                        'current': metric_data.iloc[-1].to_dict() if not metric_data.empty else {},
                        'previous': metric_data.iloc[-2].to_dict() if len(metric_data) > 1 else {},
                        'change': self._calculate_change(metric_data)
                    }
                }
            except Exception as e:
                logger.error(f"Error fetching metric {metric_name}: {e}")
                dashboard_data['metrics'][metric_name] = {'error': str(e)}
        
        return dashboard_data
    
    def _calculate_change(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate period-over-period change"""
        if len(data) < 2:
            return {'absolute': 0, 'percentage': 0}
        
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Assume first numeric column is the metric value
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if not numeric_cols.any():
            return {'absolute': 0, 'percentage': 0}
        
        value_col = numeric_cols[0]
        current_value = current[value_col]
        previous_value = previous[value_col]
        
        absolute_change = current_value - previous_value
        percentage_change = (absolute_change / previous_value * 100) if previous_value != 0 else 0
        
        return {
            'absolute': float(absolute_change),
            'percentage': float(percentage_change)
        }
    
    # Transformation functions
    def _transform_revenue_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Transform revenue data"""
        transactions = data.get('transactions', pd.DataFrame())
        subscriptions = data.get('subscriptions', pd.DataFrame())
        refunds = data.get('refunds', pd.DataFrame())
        
        # Combine all revenue sources
        revenue_data = []
        
        # Process transactions
        if not transactions.empty:
            transactions['revenue_type'] = 'transaction'
            transactions['net_amount'] = transactions['amount']
            revenue_data.append(transactions[['date', 'user_id', 'net_amount', 'revenue_type']])
        
        # Process subscriptions
        if not subscriptions.empty:
            subscriptions['revenue_type'] = 'subscription'
            subscriptions['net_amount'] = subscriptions['amount']
            revenue_data.append(subscriptions[['date', 'user_id', 'net_amount', 'revenue_type']])
        
        # Process refunds (negative revenue)
        if not refunds.empty:
            refunds['revenue_type'] = 'refund'
            refunds['net_amount'] = -refunds['amount']
            revenue_data.append(refunds[['date', 'user_id', 'net_amount', 'revenue_type']])
        
        if revenue_data:
            return pd.concat(revenue_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _calculate_net_revenue(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate net revenue after fees and taxes"""
        if data.empty:
            return data
        
        # Apply processing fees (2.9% + $0.30 for payments)
        data['processing_fee'] = data['net_amount'] * 0.029 + 0.30
        
        # Apply platform fees (if applicable)
        data['platform_fee'] = data['net_amount'] * 0.10  # 10% platform fee
        
        # Calculate net revenue
        data['net_revenue'] = data['net_amount'] - data['processing_fee'] - data['platform_fee']
        
        return data
    
    def _transform_user_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Transform user data"""
        activity = data.get('user_activity', pd.DataFrame())
        users = data.get('users', pd.DataFrame())
        sessions = data.get('sessions', pd.DataFrame())
        
        if activity.empty or users.empty:
            return pd.DataFrame()
        
        # Merge user information
        user_data = activity.merge(users[['user_id', 'created_at', 'tier']], on='user_id', how='left')
        
        # Calculate user age
        user_data['user_age_days'] = (pd.to_datetime('now') - pd.to_datetime(user_data['created_at'])).dt.days
        
        # Add session information
        if not sessions.empty:
            session_stats = sessions.groupby('user_id').agg({
                'duration': 'mean',
                'page_views': 'sum'
            }).rename(columns={'duration': 'avg_session_duration', 'page_views': 'total_page_views'})
            
            user_data = user_data.merge(session_stats, on='user_id', how='left')
        
        return user_data
    
    def _calculate_user_segments(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate user segments"""
        if data.empty:
            return data
        
        # Define segments based on activity
        def segment_user(row):
            if row['user_age_days'] <= 7:
                return 'new'
            elif row.get('total_page_views', 0) > 100:
                return 'power'
            elif row['user_age_days'] > 30 and row.get('total_page_views', 0) < 10:
                return 'at_risk'
            else:
                return 'regular'
        
        data['segment'] = data.apply(segment_user, axis=1)
        
        return data
    
    def _transform_content_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Transform content performance data"""
        videos = data.get('videos', pd.DataFrame())
        analytics = data.get('video_analytics', pd.DataFrame())
        
        if videos.empty or analytics.empty:
            return pd.DataFrame()
        
        # Merge video and analytics data
        content_data = videos.merge(analytics, on='video_id', how='left')
        
        # Calculate engagement metrics
        content_data['engagement_score'] = (
            content_data['likes'] * 1 +
            content_data['comments'] * 2 +
            content_data['shares'] * 3
        ) / content_data['views']
        
        # Calculate virality score
        content_data['virality_score'] = np.log1p(content_data['views']) * content_data['engagement_score']
        
        return content_data
    
    def _calculate_performance_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate content performance scores"""
        if data.empty:
            return data
        
        # Normalize scores to 0-100 scale
        for score_col in ['engagement_score', 'virality_score']:
            if score_col in data.columns:
                min_val = data[score_col].min()
                max_val = data[score_col].max()
                if max_val > min_val:
                    data[f'{score_col}_normalized'] = ((data[score_col] - min_val) / (max_val - min_val)) * 100
        
        return data
    
    # Quality check functions
    def _check_revenue_consistency(self, data: pd.DataFrame) -> bool:
        """Check revenue data consistency"""
        if data.empty:
            return True
        
        # Check for duplicate transactions
        if 'transaction_id' in data.columns:
            duplicates = data['transaction_id'].duplicated().sum()
            if duplicates > 0:
                logger.warning(f"Found {duplicates} duplicate transactions")
                return False
        
        # Check for negative amounts (except refunds)
        non_refund_data = data[data['revenue_type'] != 'refund']
        if (non_refund_data['net_amount'] < 0).any():
            logger.warning("Found negative amounts in non-refund transactions")
            return False
        
        return True
    
    def _check_negative_amounts(self, data: pd.DataFrame) -> bool:
        """Check for unexpected negative amounts"""
        if data.empty:
            return True
        
        # Net revenue should not be negative for normal transactions
        if 'net_revenue' in data.columns:
            negative_revenue = data[(data['revenue_type'] != 'refund') & (data['net_revenue'] < 0)]
            if not negative_revenue.empty:
                logger.warning(f"Found {len(negative_revenue)} transactions with negative net revenue")
                return False
        
        return True
    
    def _check_user_data_completeness(self, data: pd.DataFrame) -> bool:
        """Check user data completeness"""
        if data.empty:
            return True
        
        # Check for missing user IDs
        missing_users = data['user_id'].isna().sum()
        if missing_users > 0:
            logger.warning(f"Found {missing_users} records with missing user IDs")
            return False
        
        # Check data quality score
        completeness = 1 - (data.isna().sum().sum() / (len(data) * len(data.columns)))
        data_quality_score.labels(dataset='user_data').set(completeness)
        
        return completeness > 0.8  # 80% threshold
    
    # Custom metric calculations
    async def _calculate_retention(self,
                                  start_date: datetime,
                                  end_date: datetime,
                                  dimensions: Optional[List[str]],
                                  filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Calculate user retention"""
        query = """
        WITH cohorts AS (
            SELECT 
                user_id,
                DATE(created_at) as cohort_date,
                DATE(activity_date) as activity_date,
                DATEDIFF(activity_date, created_at) as days_since_signup
            FROM users u
            JOIN user_activity a ON u.user_id = a.user_id
            WHERE u.created_at >= :start_date AND u.created_at <= :end_date
        )
        SELECT 
            cohort_date,
            days_since_signup,
            COUNT(DISTINCT user_id) as users,
            COUNT(DISTINCT user_id) / FIRST_VALUE(COUNT(DISTINCT user_id)) 
                OVER (PARTITION BY cohort_date ORDER BY days_since_signup) as retention_rate
        FROM cohorts
        GROUP BY cohort_date, days_since_signup
        ORDER BY cohort_date, days_since_signup
        """
        
        with self.db_engine.connect() as conn:
            return pd.read_sql(query, conn, params={'start_date': start_date, 'end_date': end_date})
    
    async def _calculate_cost_per_user(self,
                                      start_date: datetime,
                                      end_date: datetime,
                                      dimensions: Optional[List[str]],
                                      filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Calculate cost per user"""
        # Get total costs
        costs_query = """
        SELECT DATE(date) as date, SUM(amount) as total_cost
        FROM costs
        WHERE date >= :start_date AND date <= :end_date
        GROUP BY DATE(date)
        """
        
        # Get active users
        users_query = """
        SELECT DATE(activity_date) as date, COUNT(DISTINCT user_id) as active_users
        FROM user_activity
        WHERE activity_date >= :start_date AND activity_date <= :end_date
        GROUP BY DATE(activity_date)
        """
        
        with self.db_engine.connect() as conn:
            costs_df = pd.read_sql(costs_query, conn, params={'start_date': start_date, 'end_date': end_date})
            users_df = pd.read_sql(users_query, conn, params={'start_date': start_date, 'end_date': end_date})
        
        # Merge and calculate
        result = costs_df.merge(users_df, on='date', how='outer')
        result['cost_per_user'] = result['total_cost'] / result['active_users']
        
        return result
    
    async def _create_aggregation_tables(self):
        """Create aggregation tables for dashboard data"""
        metadata = MetaData()
        
        # Revenue summary table
        revenue_summary = Table(
            'revenue_summary',
            metadata,
            Column('id', Integer, primary_key=True),
            Column('date', DateTime),
            Column('revenue_type', String(50)),
            Column('user_id', String(100)),
            Column('net_amount', Float),
            Column('processing_fee', Float),
            Column('platform_fee', Float),
            Column('net_revenue', Float),
            Column('created_at', DateTime, default=datetime.utcnow)
        )
        
        # User analytics summary table
        user_analytics_summary = Table(
            'user_analytics_summary',
            metadata,
            Column('id', Integer, primary_key=True),
            Column('date', DateTime),
            Column('user_id', String(100)),
            Column('segment', String(50)),
            Column('user_age_days', Integer),
            Column('avg_session_duration', Float),
            Column('total_page_views', Integer),
            Column('created_at', DateTime, default=datetime.utcnow)
        )
        
        # Content performance summary table
        content_performance_summary = Table(
            'content_performance_summary',
            metadata,
            Column('id', Integer, primary_key=True),
            Column('date', DateTime),
            Column('video_id', String(100)),
            Column('views', Integer),
            Column('likes', Integer),
            Column('comments', Integer),
            Column('shares', Integer),
            Column('engagement_score', Float),
            Column('virality_score', Float),
            Column('created_at', DateTime, default=datetime.utcnow)
        )
        
        # Create tables if they don't exist
        metadata.create_all(self.dw_engine)
    
    def _get_cache_key(self,
                      metric_name: str,
                      start_date: datetime,
                      end_date: datetime,
                      dimensions: Optional[List[str]],
                      filters: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for metric data"""
        key_parts = [
            metric_name,
            start_date.strftime('%Y%m%d'),
            end_date.strftime('%Y%m%d')
        ]
        
        if dimensions:
            key_parts.append('_'.join(sorted(dimensions)))
        
        if filters:
            filter_str = '_'.join([f"{k}_{v}" for k, v in sorted(filters.items())])
            key_parts.append(filter_str)
        
        return ':'.join(key_parts)
    
    async def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached data"""
        if not self.redis_client:
            return None
        
        try:
            cached = await self.redis_client.get(f"dashboard:{cache_key}")
            if cached:
                return pd.read_json(cached)
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
        
        return None
    
    async def _cache_data(self, cache_key: str, data: pd.DataFrame, ttl: int):
        """Cache data"""
        if not self.redis_client or data.empty:
            return
        
        try:
            json_data = data.to_json(date_format='iso')
            await self.redis_client.setex(f"dashboard:{cache_key}", ttl, json_data)
        except Exception as e:
            logger.error(f"Error caching data: {e}")

# Example usage
async def main():
    # Initialize dashboard data pipeline
    dashboard_data = BusinessDashboardData(
        db_url='postgresql://user:pass@localhost/ytempire',
        data_warehouse_url='postgresql://user:pass@localhost/ytempire_dw'
    )
    
    await dashboard_data.initialize()
    
    # Run revenue pipeline
    result = await dashboard_data.run_pipeline('revenue')
    print(f"Revenue pipeline result: {result}")
    
    # Get dashboard data
    exec_dashboard = await dashboard_data.get_dashboard_data('executive', '7d')
    print(f"Executive dashboard metrics: {list(exec_dashboard['metrics'].keys())}")
    
    # Get specific metric
    revenue_data = await dashboard_data.get_metric(
        'total_revenue',
        datetime.now() - timedelta(days=30),
        datetime.now(),
        dimensions=['date', 'product']
    )
    print(f"Revenue data shape: {revenue_data.shape}")

if __name__ == "__main__":
    asyncio.run(main())