"""
Business Metrics Dashboard for AI/ML Models
Real-time monitoring and visualization of business KPIs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import aiohttp
import redis.asyncio as redis
from prometheus_client import Histogram, Counter, Gauge, Summary
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import logging
import json
from sqlalchemy import create_engine, text
import mlflow

# Metrics
dashboard_load_time = Histogram('dashboard_load_duration', 'Dashboard load time', ['page'])
metric_calculation_time = Histogram('metric_calculation_duration', 'Metric calculation time', ['metric'])
alert_triggered = Counter('business_alerts_triggered', 'Business alerts triggered', ['alert_type'])
active_users = Gauge('dashboard_active_users', 'Active dashboard users')

logger = logging.getLogger(__name__)

@dataclass
class BusinessMetric:
    """Container for business metrics"""
    name: str
    value: float
    unit: str
    trend: str  # 'up', 'down', 'stable'
    change_percent: float
    timestamp: datetime
    category: str
    target: Optional[float] = None
    status: str = 'normal'  # 'normal', 'warning', 'critical'

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    refresh_interval_seconds: int = 30
    retention_days: int = 90
    alert_thresholds: Dict = field(default_factory=dict)
    metrics_to_track: List[str] = field(default_factory=list)
    comparison_periods: List[str] = field(default_factory=lambda: ['day', 'week', 'month'])

class BusinessMetricsDashboard:
    """Advanced business metrics dashboard for YouTube automation platform"""
    
    def __init__(self, 
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 db_url: str = None,
                 mlflow_uri: str = None):
        
        self.redis_client = None
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # Database connection
        self.db_engine = create_engine(db_url) if db_url else None
        
        # MLflow tracking
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Metric definitions
        self.metric_definitions = {
            'revenue': {
                'name': 'Total Revenue',
                'unit': '$',
                'category': 'financial',
                'query': 'SELECT SUM(amount) FROM revenue WHERE date >= :start_date',
                'target': 100000,
                'alert_threshold': 0.8
            },
            'user_growth': {
                'name': 'User Growth Rate',
                'unit': '%',
                'category': 'growth',
                'calculation': self._calculate_user_growth,
                'target': 10,
                'alert_threshold': 5
            },
            'video_performance': {
                'name': 'Average Video Performance',
                'unit': 'views',
                'category': 'content',
                'calculation': self._calculate_video_performance,
                'target': 10000,
                'alert_threshold': 5000
            },
            'model_accuracy': {
                'name': 'Model Accuracy',
                'unit': '%',
                'category': 'ml',
                'source': 'mlflow',
                'metric_name': 'accuracy',
                'target': 0.85,
                'alert_threshold': 0.80
            },
            'api_latency': {
                'name': 'API Latency',
                'unit': 'ms',
                'category': 'performance',
                'source': 'prometheus',
                'query': 'avg(api_request_duration_seconds)',
                'target': 100,
                'alert_threshold': 200
            },
            'cost_per_video': {
                'name': 'Cost per Video',
                'unit': '$',
                'category': 'financial',
                'calculation': self._calculate_cost_per_video,
                'target': 5,
                'alert_threshold': 10
            },
            'engagement_rate': {
                'name': 'Engagement Rate',
                'unit': '%',
                'category': 'content',
                'calculation': self._calculate_engagement_rate,
                'target': 5,
                'alert_threshold': 2
            },
            'churn_rate': {
                'name': 'Churn Rate',
                'unit': '%',
                'category': 'growth',
                'calculation': self._calculate_churn_rate,
                'target': 5,
                'alert_threshold': 10
            },
            'ltv': {
                'name': 'Customer Lifetime Value',
                'unit': '$',
                'category': 'financial',
                'calculation': self._calculate_ltv,
                'target': 1000,
                'alert_threshold': 500
            },
            'cac': {
                'name': 'Customer Acquisition Cost',
                'unit': '$',
                'category': 'financial',
                'calculation': self._calculate_cac,
                'target': 100,
                'alert_threshold': 200
            }
        }
        
        # Setup dashboard layout
        self._setup_dashboard_layout()
        
        # Setup callbacks
        self._setup_callbacks()
    
    async def initialize(self):
        """Initialize async components"""
        self.redis_client = await redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=True
        )
    
    def _setup_dashboard_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("YTEmpire Business Metrics Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # KPI Cards Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Revenue", className="card-title"),
                            html.H2(id="revenue-value", children="$0"),
                            html.P(id="revenue-change", className="text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Active Users", className="card-title"),
                            html.H2(id="users-value", children="0"),
                            html.P(id="users-change", className="text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Videos Generated", className="card-title"),
                            html.H2(id="videos-value", children="0"),
                            html.P(id="videos-change", className="text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Model Accuracy", className="card-title"),
                            html.H2(id="accuracy-value", children="0%"),
                            html.P(id="accuracy-status", className="text-muted")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Charts Row 1
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="revenue-chart")
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(id="user-growth-chart")
                ], width=6)
            ], className="mb-4"),
            
            # Charts Row 2
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="video-performance-chart")
                ], width=4),
                
                dbc.Col([
                    dcc.Graph(id="engagement-chart")
                ], width=4),
                
                dbc.Col([
                    dcc.Graph(id="cost-analysis-chart")
                ], width=4)
            ], className="mb-4"),
            
            # Model Performance Section
            dbc.Row([
                dbc.Col([
                    html.H3("ML Model Performance"),
                    dcc.Graph(id="model-metrics-chart")
                ], width=12)
            ], className="mb-4"),
            
            # Real-time Metrics Section
            dbc.Row([
                dbc.Col([
                    html.H3("Real-time Metrics"),
                    dcc.Graph(id="realtime-metrics")
                ], width=12)
            ], className="mb-4"),
            
            # Alerts Section
            dbc.Row([
                dbc.Col([
                    html.H3("Active Alerts"),
                    html.Div(id="alerts-container")
                ], width=12)
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # 30 seconds
                n_intervals=0
            )
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('revenue-value', 'children'),
             Output('revenue-change', 'children'),
             Output('users-value', 'children'),
             Output('users-change', 'children'),
             Output('videos-value', 'children'),
             Output('videos-change', 'children'),
             Output('accuracy-value', 'children'),
             Output('accuracy-status', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_kpi_cards(n):
            """Update KPI cards"""
            metrics = asyncio.run(self.fetch_current_metrics())
            
            return (
                f"${metrics['revenue']['value']:,.0f}",
                f"{metrics['revenue']['change']:+.1f}% vs last period",
                f"{metrics['users']['value']:,}",
                f"{metrics['users']['change']:+.1f}% growth",
                f"{metrics['videos']['value']:,}",
                f"{metrics['videos']['change']:+.1f}% vs yesterday",
                f"{metrics['accuracy']['value']:.1%}",
                f"Target: {metrics['accuracy']['target']:.1%}"
            )
        
        @self.app.callback(
            Output('revenue-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_revenue_chart(n):
            """Update revenue chart"""
            data = asyncio.run(self.fetch_revenue_data())
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=data['revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='#2E7D32', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=data['target'],
                mode='lines',
                name='Target',
                line=dict(color='#FFA726', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Revenue Trend",
                xaxis_title="Date",
                yaxis_title="Revenue ($)",
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output('user-growth-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_user_growth_chart(n):
            """Update user growth chart"""
            data = asyncio.run(self.fetch_user_growth_data())
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=data['period'],
                y=data['new_users'],
                name='New Users',
                marker_color='#1976D2'
            ))
            
            fig.add_trace(go.Bar(
                x=data['period'],
                y=data['active_users'],
                name='Active Users',
                marker_color='#42A5F5'
            ))
            
            fig.update_layout(
                title="User Growth",
                xaxis_title="Period",
                yaxis_title="Users",
                barmode='group'
            )
            
            return fig
        
        @self.app.callback(
            Output('video-performance-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_video_performance(n):
            """Update video performance chart"""
            data = asyncio.run(self.fetch_video_performance_data())
            
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=data['views'],
                name='Views',
                marker_color='#7B1FA2'
            ))
            
            fig.add_trace(go.Box(
                y=data['engagement'],
                name='Engagement',
                marker_color='#BA68C8'
            ))
            
            fig.update_layout(
                title="Video Performance Distribution",
                yaxis_title="Count"
            )
            
            return fig
        
        @self.app.callback(
            Output('model-metrics-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_model_metrics(n):
            """Update model metrics chart"""
            data = asyncio.run(self.fetch_model_metrics())
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy Over Time', 'Loss Over Time', 
                              'Inference Latency', 'Model Versions')
            )
            
            # Accuracy
            fig.add_trace(
                go.Scatter(x=data['time'], y=data['accuracy'], name='Accuracy'),
                row=1, col=1
            )
            
            # Loss
            fig.add_trace(
                go.Scatter(x=data['time'], y=data['loss'], name='Loss'),
                row=1, col=2
            )
            
            # Latency
            fig.add_trace(
                go.Histogram(x=data['latency'], name='Latency'),
                row=2, col=1
            )
            
            # Model versions
            fig.add_trace(
                go.Bar(x=data['versions'], y=data['version_accuracy'], name='Version Accuracy'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            
            return fig
        
        @self.app.callback(
            Output('realtime-metrics', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_realtime_metrics(n):
            """Update real-time metrics"""
            data = asyncio.run(self.fetch_realtime_metrics())
            
            fig = go.Figure()
            
            # Add traces for different metrics
            for metric in data['metrics']:
                fig.add_trace(go.Scatter(
                    x=data['timestamps'],
                    y=data[metric],
                    mode='lines',
                    name=metric
                ))
            
            fig.update_layout(
                title="Real-time System Metrics",
                xaxis_title="Time",
                yaxis_title="Value",
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output('alerts-container', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_alerts(n):
            """Update alerts section"""
            alerts = asyncio.run(self.fetch_active_alerts())
            
            alert_components = []
            for alert in alerts:
                color = 'danger' if alert['severity'] == 'critical' else 'warning'
                alert_components.append(
                    dbc.Alert(
                        [
                            html.H5(alert['title'], className="alert-heading"),
                            html.P(alert['message']),
                            html.Hr(),
                            html.P(f"Time: {alert['timestamp']}", className="mb-0")
                        ],
                        color=color,
                        dismissable=True
                    )
                )
            
            return alert_components if alert_components else [html.P("No active alerts")]
    
    async def fetch_current_metrics(self) -> Dict:
        """Fetch current metric values"""
        metrics = {}
        
        # Revenue
        revenue_current = await self._fetch_metric_value('revenue', 'current')
        revenue_previous = await self._fetch_metric_value('revenue', 'previous')
        metrics['revenue'] = {
            'value': revenue_current,
            'change': ((revenue_current - revenue_previous) / revenue_previous * 100) if revenue_previous else 0
        }
        
        # Users
        users_current = await self._fetch_metric_value('active_users', 'current')
        users_previous = await self._fetch_metric_value('active_users', 'previous')
        metrics['users'] = {
            'value': users_current,
            'change': ((users_current - users_previous) / users_previous * 100) if users_previous else 0
        }
        
        # Videos
        videos_current = await self._fetch_metric_value('videos_generated', 'current')
        videos_previous = await self._fetch_metric_value('videos_generated', 'previous')
        metrics['videos'] = {
            'value': videos_current,
            'change': ((videos_current - videos_previous) / videos_previous * 100) if videos_previous else 0
        }
        
        # Model Accuracy
        accuracy = await self._fetch_model_accuracy()
        metrics['accuracy'] = {
            'value': accuracy,
            'target': 0.85
        }
        
        return metrics
    
    async def _fetch_metric_value(self, metric_name: str, period: str) -> float:
        """Fetch single metric value"""
        # Check cache first
        if self.redis_client:
            cached = await self.redis_client.get(f"metric:{metric_name}:{period}")
            if cached:
                return float(cached)
        
        # Fetch from database or calculate
        value = 0
        if self.db_engine and metric_name in self.metric_definitions:
            metric_def = self.metric_definitions[metric_name]
            
            if 'query' in metric_def:
                # Execute SQL query
                with self.db_engine.connect() as conn:
                    if period == 'current':
                        start_date = datetime.now() - timedelta(days=1)
                    else:
                        start_date = datetime.now() - timedelta(days=2)
                        
                    result = conn.execute(
                        text(metric_def['query']),
                        {'start_date': start_date}
                    ).scalar()
                    value = result or 0
            
            elif 'calculation' in metric_def:
                # Use calculation function
                value = await metric_def['calculation'](period)
        
        # Cache the result
        if self.redis_client:
            await self.redis_client.setex(
                f"metric:{metric_name}:{period}",
                300,  # 5 minutes cache
                str(value)
            )
        
        return value
    
    async def _fetch_model_accuracy(self) -> float:
        """Fetch latest model accuracy from MLflow"""
        try:
            # Get latest model version
            client = mlflow.tracking.MlflowClient()
            model_name = "content_generator"
            latest_version = client.get_latest_versions(model_name)[0]
            
            # Get run metrics
            run = client.get_run(latest_version.run_id)
            accuracy = run.data.metrics.get('accuracy', 0.0)
            
            return accuracy
        except:
            # Fallback to cached or default value
            if self.redis_client:
                cached = await self.redis_client.get("metric:model_accuracy")
                if cached:
                    return float(cached)
            return 0.85
    
    async def fetch_revenue_data(self) -> pd.DataFrame:
        """Fetch revenue data for chart"""
        # Generate sample data (replace with actual database query)
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        revenue = np.random.normal(10000, 2000, 30).cumsum()
        target = [10000] * 30
        
        return pd.DataFrame({
            'date': dates,
            'revenue': revenue,
            'target': target
        })
    
    async def fetch_user_growth_data(self) -> pd.DataFrame:
        """Fetch user growth data"""
        # Generate sample data
        periods = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
        new_users = [150, 200, 180, 250]
        active_users = [1000, 1150, 1300, 1500]
        
        return pd.DataFrame({
            'period': periods,
            'new_users': new_users,
            'active_users': active_users
        })
    
    async def fetch_video_performance_data(self) -> pd.DataFrame:
        """Fetch video performance data"""
        # Generate sample data
        views = np.random.lognormal(8, 2, 100)
        engagement = np.random.beta(2, 5, 100) * 100
        
        return pd.DataFrame({
            'views': views,
            'engagement': engagement
        })
    
    async def fetch_model_metrics(self) -> Dict:
        """Fetch model metrics for visualization"""
        # Generate sample data
        time_points = pd.date_range(end=datetime.now(), periods=24, freq='H')
        
        return {
            'time': time_points,
            'accuracy': np.random.normal(0.85, 0.02, 24),
            'loss': np.random.exponential(0.1, 24),
            'latency': np.random.gamma(2, 2, 1000),
            'versions': ['v1.0', 'v1.1', 'v1.2', 'v2.0'],
            'version_accuracy': [0.82, 0.84, 0.85, 0.87]
        }
    
    async def fetch_realtime_metrics(self) -> Dict:
        """Fetch real-time metrics"""
        # Generate sample real-time data
        timestamps = pd.date_range(end=datetime.now(), periods=60, freq='min')
        
        return {
            'timestamps': timestamps,
            'metrics': ['cpu_usage', 'memory_usage', 'request_rate'],
            'cpu_usage': np.random.beta(2, 5, 60) * 100,
            'memory_usage': np.random.beta(3, 2, 60) * 100,
            'request_rate': np.random.poisson(100, 60)
        }
    
    async def fetch_active_alerts(self) -> List[Dict]:
        """Fetch active alerts"""
        alerts = []
        
        # Check metric thresholds
        for metric_name, metric_def in self.metric_definitions.items():
            current_value = await self._fetch_metric_value(metric_name, 'current')
            
            if 'alert_threshold' in metric_def:
                threshold = metric_def['alert_threshold']
                target = metric_def.get('target', threshold)
                
                # Check if threshold is violated
                if metric_def['category'] == 'performance':
                    # Lower is better for performance metrics
                    if current_value > threshold:
                        alerts.append({
                            'title': f"{metric_def['name']} Alert",
                            'message': f"Current value ({current_value:.2f}) exceeds threshold ({threshold})",
                            'severity': 'warning' if current_value < threshold * 1.5 else 'critical',
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                else:
                    # Higher is better for most metrics
                    if current_value < threshold:
                        alerts.append({
                            'title': f"{metric_def['name']} Alert",
                            'message': f"Current value ({current_value:.2f}) below threshold ({threshold})",
                            'severity': 'warning' if current_value > threshold * 0.5 else 'critical',
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
        
        # Update alert counter
        if alerts:
            alert_triggered.labels(alert_type='threshold').inc(len(alerts))
        
        return alerts
    
    async def _calculate_user_growth(self, period: str) -> float:
        """Calculate user growth rate"""
        # Placeholder calculation
        return np.random.uniform(5, 15)
    
    async def _calculate_video_performance(self, period: str) -> float:
        """Calculate average video performance"""
        # Placeholder calculation
        return np.random.uniform(8000, 12000)
    
    async def _calculate_cost_per_video(self, period: str) -> float:
        """Calculate cost per video"""
        # Placeholder calculation
        return np.random.uniform(3, 7)
    
    async def _calculate_engagement_rate(self, period: str) -> float:
        """Calculate engagement rate"""
        # Placeholder calculation
        return np.random.uniform(3, 7)
    
    async def _calculate_churn_rate(self, period: str) -> float:
        """Calculate churn rate"""
        # Placeholder calculation
        return np.random.uniform(3, 7)
    
    async def _calculate_ltv(self, period: str) -> float:
        """Calculate customer lifetime value"""
        # Placeholder calculation
        return np.random.uniform(800, 1200)
    
    async def _calculate_cac(self, period: str) -> float:
        """Calculate customer acquisition cost"""
        # Placeholder calculation
        return np.random.uniform(80, 120)
    
    def run_dashboard(self, debug: bool = False, port: int = 8050):
        """Run the dashboard server"""
        active_users.inc()
        self.app.run_server(debug=debug, port=port)

# Additional components for advanced analytics

class BusinessIntelligence:
    """Business intelligence and analytics engine"""
    
    def __init__(self, dashboard: BusinessMetricsDashboard):
        self.dashboard = dashboard
        
    async def generate_insights(self) -> List[Dict]:
        """Generate business insights from metrics"""
        insights = []
        
        # Analyze revenue trends
        revenue_data = await self.dashboard.fetch_revenue_data()
        if len(revenue_data) > 7:
            trend = np.polyfit(range(len(revenue_data)), revenue_data['revenue'], 1)[0]
            if trend > 0:
                insights.append({
                    'type': 'positive',
                    'title': 'Revenue Growth',
                    'description': f'Revenue is growing at ${trend:.2f} per day'
                })
            else:
                insights.append({
                    'type': 'negative',
                    'title': 'Revenue Decline',
                    'description': f'Revenue is declining at ${abs(trend):.2f} per day',
                    'action': 'Review pricing and marketing strategies'
                })
        
        # Analyze user retention
        metrics = await self.dashboard.fetch_current_metrics()
        if metrics['users']['change'] < -5:
            insights.append({
                'type': 'warning',
                'title': 'User Retention Issue',
                'description': 'User growth is declining',
                'action': 'Implement retention campaigns'
            })
        
        # Model performance insights
        if metrics['accuracy']['value'] < metrics['accuracy']['target']:
            insights.append({
                'type': 'warning',
                'title': 'Model Performance',
                'description': f"Model accuracy ({metrics['accuracy']['value']:.2%}) below target",
                'action': 'Retrain model with more recent data'
            })
        
        return insights
    
    async def forecast_metrics(self, metric_name: str, days_ahead: int = 30) -> pd.DataFrame:
        """Forecast future metric values"""
        # Get historical data
        historical_data = []
        for i in range(30):
            date = datetime.now() - timedelta(days=i)
            value = await self.dashboard._fetch_metric_value(metric_name, f'day_{i}')
            historical_data.append({'date': date, 'value': value})
        
        df = pd.DataFrame(historical_data)
        
        # Simple linear regression forecast
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['value'].values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate forecast
        future_X = np.arange(len(df), len(df) + days_ahead).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        forecast_dates = pd.date_range(start=datetime.now(), periods=days_ahead, freq='D')
        
        return pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast,
            'lower_bound': forecast * 0.9,
            'upper_bound': forecast * 1.1
        })

# Example usage
async def main():
    # Initialize dashboard
    dashboard = BusinessMetricsDashboard(
        redis_host='localhost',
        redis_port=6379,
        db_url='postgresql://user:pass@localhost/ytempire',
        mlflow_uri='http://localhost:5000'
    )
    
    await dashboard.initialize()
    
    # Run dashboard
    dashboard.run_dashboard(debug=True, port=8050)

if __name__ == "__main__":
    asyncio.run(main())