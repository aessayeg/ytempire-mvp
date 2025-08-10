#!/usr/bin/env python3
"""
YTEmpire Capacity Planning Tool
P2 Enhancement - Advanced capacity planning and resource forecasting
"""

import os
import sys
import json
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import requests
from dataclasses import dataclass, asdict
import argparse

# Machine learning imports
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import joblib
except ImportError:
    print("Warning: sklearn not available, ML forecasting disabled")
    
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Warning: plotting libraries not available")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('capacity_planner')

@dataclass
class ResourceMetric:
    """Resource utilization metric"""
    name: str
    current_value: float
    threshold: float
    unit: str
    trend: str  # 'increasing', 'decreasing', 'stable'
    forecast_7d: float
    forecast_30d: float
    recommendation: str

@dataclass
class CapacityPlanningReport:
    """Capacity planning report"""
    timestamp: datetime
    current_utilization: Dict[str, ResourceMetric]
    growth_predictions: Dict[str, float]
    scaling_recommendations: List[str]
    cost_projections: Dict[str, float]
    risk_assessment: Dict[str, str]
    action_items: List[str]

class PrometheusClient:
    """Simple Prometheus client for metrics collection"""
    
    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def query(self, query: str, time: Optional[datetime] = None) -> Dict[str, Any]:
        """Execute Prometheus query"""
        try:
            params = {'query': query}
            if time:
                params['time'] = time.timestamp()
            
            response = self.session.get(f"{self.base_url}/api/v1/query", params=params)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Prometheus query failed: {e}")
            return {"data": {"result": []}}
    
    def query_range(self, query: str, start: datetime, end: datetime, step: str = "1h") -> Dict[str, Any]:
        """Execute Prometheus range query"""
        try:
            params = {
                'query': query,
                'start': start.timestamp(),
                'end': end.timestamp(), 
                'step': step
            }
            
            response = self.session.get(f"{self.base_url}/api/v1/query_range", params=params)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Prometheus range query failed: {e}")
            return {"data": {"result": []}}

class CapacityPlanner:
    """Main capacity planning engine"""
    
    def __init__(self, config_path: str = "capacity_config.yaml"):
        self.config = self._load_config(config_path)
        self.prometheus = PrometheusClient(self.config.get('prometheus_url', 'http://localhost:9090'))
        self.models = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            'prometheus_url': 'http://localhost:9090',
            'metrics': {
                'cpu': {
                    'query': 'rate(container_cpu_usage_seconds_total[5m]) * 100',
                    'threshold': 80,
                    'unit': '%'
                },
                'memory': {
                    'query': '(container_memory_usage_bytes / container_spec_memory_limit_bytes) * 100',
                    'threshold': 85,
                    'unit': '%'
                },
                'disk': {
                    'query': '(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100',
                    'threshold': 90,
                    'unit': '%'
                },
                'network': {
                    'query': 'rate(container_network_transmit_bytes_total[5m]) + rate(container_network_receive_bytes_total[5m])',
                    'threshold': 800000000,
                    'unit': 'bytes/s'
                },
                'video_generation': {
                    'query': 'rate(ytempire_videos_generated_total[1h])',
                    'threshold': 50,
                    'unit': 'videos/hour'
                },
                'api_requests': {
                    'query': 'rate(http_requests_total[5m])',
                    'threshold': 1000,
                    'unit': 'requests/5min'
                }
            },
            'forecasting': {
                'lookback_days': 30,
                'forecast_days': [7, 30],
                'model_type': 'linear_regression'
            },
            'scaling': {
                'cpu_scale_up_threshold': 75,
                'cpu_scale_down_threshold': 30,
                'memory_scale_up_threshold': 80,
                'memory_scale_down_threshold': 40
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def collect_metrics(self) -> Dict[str, float]:
        """Collect current metrics from Prometheus"""
        metrics = {}
        
        for metric_name, metric_config in self.config['metrics'].items():
            try:
                result = self.prometheus.query(metric_config['query'])
                
                if result['data']['result']:
                    # Take the first result or average if multiple
                    values = [float(item['value'][1]) for item in result['data']['result']]
                    metrics[metric_name] = np.mean(values) if values else 0.0
                else:
                    metrics[metric_name] = 0.0
                    
            except Exception as e:
                logger.error(f"Failed to collect metric {metric_name}: {e}")
                metrics[metric_name] = 0.0
        
        return metrics
    
    def collect_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Collect historical data for forecasting"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        all_data = []
        
        for metric_name, metric_config in self.config['metrics'].items():
            try:
                result = self.prometheus.query_range(
                    metric_config['query'],
                    start_time,
                    end_time,
                    '1h'
                )
                
                for series in result['data']['result']:
                    for timestamp, value in series['values']:
                        all_data.append({
                            'timestamp': datetime.fromtimestamp(timestamp),
                            'metric': metric_name,
                            'value': float(value)
                        })
                        
            except Exception as e:
                logger.error(f"Failed to collect historical data for {metric_name}: {e}")
        
        if not all_data:
            logger.warning("No historical data collected, using mock data")
            return self._generate_mock_data(days)
        
        return pd.DataFrame(all_data)
    
    def _generate_mock_data(self, days: int) -> pd.DataFrame:
        """Generate mock data for testing"""
        data = []
        end_time = datetime.now()
        
        for metric_name in self.config['metrics'].keys():
            for hour in range(days * 24):
                timestamp = end_time - timedelta(hours=hour)
                
                # Generate trending data with some noise
                base_value = 50 + (hour / (days * 24)) * 30  # Trending upward
                noise = np.random.normal(0, 5)
                value = max(0, base_value + noise)
                
                data.append({
                    'timestamp': timestamp,
                    'metric': metric_name,
                    'value': value
                })
        
        return pd.DataFrame(data)
    
    def analyze_trends(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze trends in historical data"""
        trends = {}
        
        for metric in df['metric'].unique():
            metric_data = df[df['metric'] == metric].sort_values('timestamp')
            
            if len(metric_data) < 2:
                trends[metric] = 'stable'
                continue
                
            # Simple linear trend analysis
            x = np.arange(len(metric_data))
            y = metric_data['value'].values
            
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.1:
                trends[metric] = 'increasing'
            elif slope < -0.1:
                trends[metric] = 'decreasing'
            else:
                trends[metric] = 'stable'
        
        return trends
    
    def forecast_metrics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Forecast metrics using machine learning"""
        forecasts = {}
        
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("Sklearn not available, using simple forecasting")
            return self._simple_forecast(df)
        
        for metric in df['metric'].unique():
            metric_data = df[df['metric'] == metric].sort_values('timestamp')
            
            if len(metric_data) < 10:
                forecasts[metric] = {'7d': 0.0, '30d': 0.0}
                continue
            
            # Prepare features
            metric_data['hour'] = metric_data['timestamp'].dt.hour
            metric_data['day_of_week'] = metric_data['timestamp'].dt.dayofweek
            metric_data['day_of_month'] = metric_data['timestamp'].dt.day
            
            X = metric_data[['hour', 'day_of_week', 'day_of_month']].values
            y = metric_data['value'].values
            
            # Train model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Make predictions
            future_7d = self._generate_future_features(7)
            future_30d = self._generate_future_features(30)
            
            pred_7d = model.predict(scaler.transform(future_7d))
            pred_30d = model.predict(scaler.transform(future_30d))
            
            forecasts[metric] = {
                '7d': float(np.mean(pred_7d)),
                '30d': float(np.mean(pred_30d))
            }
            
            # Store model for later use
            self.models[metric] = {
                'model': model,
                'scaler': scaler
            }
        
        return forecasts
    
    def _simple_forecast(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Simple forecasting without ML libraries"""
        forecasts = {}
        
        for metric in df['metric'].unique():
            metric_data = df[df['metric'] == metric].sort_values('timestamp')
            
            if len(metric_data) < 2:
                forecasts[metric] = {'7d': 0.0, '30d': 0.0}
                continue
            
            # Simple linear extrapolation
            recent_avg = metric_data['value'].tail(24).mean()  # Last 24 hours
            overall_avg = metric_data['value'].mean()
            
            # Assume trend continues
            trend_factor = recent_avg / overall_avg if overall_avg > 0 else 1.0
            
            forecasts[metric] = {
                '7d': recent_avg * trend_factor,
                '30d': recent_avg * (trend_factor ** 2)
            }
        
        return forecasts
    
    def _generate_future_features(self, days: int) -> np.ndarray:
        """Generate feature matrix for future dates"""
        features = []
        start_time = datetime.now()
        
        for hour in range(days * 24):
            future_time = start_time + timedelta(hours=hour)
            features.append([
                future_time.hour,
                future_time.weekday(),
                future_time.day
            ])
        
        return np.array(features)
    
    def generate_scaling_recommendations(self, current_metrics: Dict[str, float], forecasts: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate scaling recommendations"""
        recommendations = []
        scaling_config = self.config['scaling']
        
        for metric_name, current_value in current_metrics.items():
            if metric_name not in self.config['metrics']:
                continue
                
            threshold = self.config['metrics'][metric_name]['threshold']
            forecast_7d = forecasts.get(metric_name, {}).get('7d', current_value)
            
            # CPU scaling recommendations
            if metric_name == 'cpu':
                if current_value > scaling_config['cpu_scale_up_threshold']:
                    recommendations.append(f"🔴 URGENT: Scale up CPU resources (current: {current_value:.1f}%)")
                elif forecast_7d > scaling_config['cpu_scale_up_threshold']:
                    recommendations.append(f"🟡 WARNING: CPU may need scaling within 7 days (forecast: {forecast_7d:.1f}%)")
                elif current_value < scaling_config['cpu_scale_down_threshold']:
                    recommendations.append(f"🟢 OPTIMIZATION: Consider scaling down CPU resources (current: {current_value:.1f}%)")
            
            # Memory scaling recommendations
            elif metric_name == 'memory':
                if current_value > scaling_config['memory_scale_up_threshold']:
                    recommendations.append(f"🔴 URGENT: Scale up memory resources (current: {current_value:.1f}%)")
                elif forecast_7d > scaling_config['memory_scale_up_threshold']:
                    recommendations.append(f"🟡 WARNING: Memory may need scaling within 7 days (forecast: {forecast_7d:.1f}%)")
                elif current_value < scaling_config['memory_scale_down_threshold']:
                    recommendations.append(f"🟢 OPTIMIZATION: Consider scaling down memory resources (current: {current_value:.1f}%)")
            
            # Disk scaling recommendations  
            elif metric_name == 'disk':
                if current_value > 85:
                    recommendations.append(f"🔴 URGENT: Disk space critically low (current: {current_value:.1f}%)")
                elif forecast_7d > 90:
                    recommendations.append(f"🟡 WARNING: Disk may be full within 7 days (forecast: {forecast_7d:.1f}%)")
            
            # Network scaling recommendations
            elif metric_name == 'network':
                if current_value > threshold * 0.8:
                    recommendations.append(f"🟡 WARNING: Network utilization high (current: {current_value/1e6:.1f} MB/s)")
            
            # Video generation capacity
            elif metric_name == 'video_generation':
                if current_value > threshold * 0.9:
                    recommendations.append(f"🔴 URGENT: Video generation at capacity (current: {current_value:.1f}/hour)")
                elif forecast_7d > threshold:
                    recommendations.append(f"🟡 WARNING: Video generation may hit capacity within 7 days")
        
        return recommendations
    
    def calculate_cost_projections(self, current_metrics: Dict[str, float], forecasts: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate cost projections based on resource usage"""
        # AWS pricing estimates (simplified)
        pricing = {
            'cpu_hour': 0.05,      # Per vCPU hour
            'memory_gb_hour': 0.01, # Per GB hour
            'storage_gb_month': 0.10, # Per GB month
            'network_gb': 0.09      # Per GB transfer
        }
        
        current_costs = {
            'cpu': current_metrics.get('cpu', 0) * pricing['cpu_hour'] * 24 * 30 / 100,
            'memory': current_metrics.get('memory', 0) * pricing['memory_gb_hour'] * 24 * 30 / 100,
            'storage': current_metrics.get('disk', 0) * pricing['storage_gb_month'] / 100,
            'network': current_metrics.get('network', 0) * pricing['network_gb'] / 1e9
        }
        
        forecast_costs = {}
        for period in ['7d', '30d']:
            forecast_costs[period] = 0
            for metric_name, forecast_value in forecasts.items():
                if period in forecast_value:
                    if metric_name == 'cpu':
                        days = 7 if period == '7d' else 30
                        forecast_costs[period] += forecast_value[period] * pricing['cpu_hour'] * 24 * days / 100
                    elif metric_name == 'memory':
                        days = 7 if period == '7d' else 30
                        forecast_costs[period] += forecast_value[period] * pricing['memory_gb_hour'] * 24 * days / 100
        
        return {
            'current_monthly': sum(current_costs.values()),
            'forecast_7d': forecast_costs.get('7d', 0),
            'forecast_30d': forecast_costs.get('30d', 0),
            'breakdown': current_costs
        }
    
    def assess_risks(self, current_metrics: Dict[str, float], forecasts: Dict[str, Dict[str, float]], trends: Dict[str, str]) -> Dict[str, str]:
        """Assess capacity risks"""
        risks = {}
        
        for metric_name, current_value in current_metrics.items():
            if metric_name not in self.config['metrics']:
                continue
                
            threshold = self.config['metrics'][metric_name]['threshold']
            forecast_7d = forecasts.get(metric_name, {}).get('7d', current_value)
            trend = trends.get(metric_name, 'stable')
            
            risk_level = 'low'
            
            if current_value > threshold:
                risk_level = 'critical'
            elif forecast_7d > threshold:
                risk_level = 'high'
            elif trend == 'increasing' and current_value > threshold * 0.7:
                risk_level = 'medium'
            
            risks[metric_name] = risk_level
        
        return risks
    
    def generate_report(self) -> CapacityPlanningReport:
        """Generate comprehensive capacity planning report"""
        logger.info("Generating capacity planning report...")
        
        # Collect current metrics
        current_metrics = self.collect_metrics()
        logger.info(f"Collected metrics: {current_metrics}")
        
        # Collect historical data
        historical_data = self.collect_historical_data(self.config['forecasting']['lookback_days'])
        logger.info(f"Collected {len(historical_data)} historical data points")
        
        # Analyze trends
        trends = self.analyze_trends(historical_data)
        logger.info(f"Analyzed trends: {trends}")
        
        # Generate forecasts
        forecasts = self.forecast_metrics(historical_data)
        logger.info(f"Generated forecasts: {forecasts}")
        
        # Build resource metrics
        resource_metrics = {}
        for metric_name, current_value in current_metrics.items():
            if metric_name not in self.config['metrics']:
                continue
                
            metric_config = self.config['metrics'][metric_name]
            
            resource_metrics[metric_name] = ResourceMetric(
                name=metric_name,
                current_value=current_value,
                threshold=metric_config['threshold'],
                unit=metric_config['unit'],
                trend=trends.get(metric_name, 'stable'),
                forecast_7d=forecasts.get(metric_name, {}).get('7d', current_value),
                forecast_30d=forecasts.get(metric_name, {}).get('30d', current_value),
                recommendation=self._get_metric_recommendation(metric_name, current_value, forecasts.get(metric_name, {}))
            )
        
        # Generate scaling recommendations
        scaling_recommendations = self.generate_scaling_recommendations(current_metrics, forecasts)
        
        # Calculate cost projections
        cost_projections = self.calculate_cost_projections(current_metrics, forecasts)
        
        # Assess risks
        risk_assessment = self.assess_risks(current_metrics, forecasts, trends)
        
        # Generate action items
        action_items = self._generate_action_items(resource_metrics, scaling_recommendations, risk_assessment)
        
        return CapacityPlanningReport(
            timestamp=datetime.now(),
            current_utilization=resource_metrics,
            growth_predictions=forecasts,
            scaling_recommendations=scaling_recommendations,
            cost_projections=cost_projections,
            risk_assessment=risk_assessment,
            action_items=action_items
        )
    
    def _get_metric_recommendation(self, metric_name: str, current_value: float, forecast: Dict[str, float]) -> str:
        """Get recommendation for a specific metric"""
        if metric_name not in self.config['metrics']:
            return "No recommendation"
            
        threshold = self.config['metrics'][metric_name]['threshold']
        forecast_7d = forecast.get('7d', current_value)
        
        if current_value > threshold:
            return f"CRITICAL: Immediate action required (current: {current_value:.1f})"
        elif forecast_7d > threshold:
            return f"WARNING: Will exceed threshold within 7 days (forecast: {forecast_7d:.1f})"
        elif current_value > threshold * 0.8:
            return f"CAUTION: Approaching threshold (current: {current_value:.1f})"
        else:
            return f"OK: Within normal range (current: {current_value:.1f})"
    
    def _generate_action_items(self, resource_metrics: Dict[str, ResourceMetric], 
                             scaling_recommendations: List[str], 
                             risk_assessment: Dict[str, str]) -> List[str]:
        """Generate prioritized action items"""
        action_items = []
        
        # Critical actions first
        for metric_name, risk_level in risk_assessment.items():
            if risk_level == 'critical':
                action_items.append(f"🔴 IMMEDIATE: Address {metric_name} capacity issue")
        
        # High risk actions
        for metric_name, risk_level in risk_assessment.items():
            if risk_level == 'high':
                action_items.append(f"🟡 URGENT: Plan {metric_name} capacity expansion within 7 days")
        
        # Optimization opportunities
        for metric_name, metric in resource_metrics.items():
            if metric.current_value < metric.threshold * 0.3:
                action_items.append(f"🟢 OPTIMIZE: Consider reducing {metric_name} allocation to save costs")
        
        # Add scaling recommendations as action items
        action_items.extend(scaling_recommendations[:3])  # Top 3 recommendations
        
        return action_items[:10]  # Limit to top 10 action items
    
    def save_report(self, report: CapacityPlanningReport, output_path: str = "capacity_report.json"):
        """Save report to file"""
        report_dict = asdict(report)
        
        # Convert datetime to string for JSON serialization
        report_dict['timestamp'] = report.timestamp.isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Report saved to {output_path}")
    
    def print_report(self, report: CapacityPlanningReport):
        """Print formatted report to console"""
        print("\n" + "="*80)
        print("🏗️  YTEMPIRE CAPACITY PLANNING REPORT")
        print("="*80)
        print(f"📅 Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()
        
        print("📊 CURRENT RESOURCE UTILIZATION")
        print("-" * 50)
        for metric_name, metric in report.current_utilization.items():
            status_icon = "🔴" if metric.current_value > metric.threshold else "🟢"
            print(f"{status_icon} {metric_name.upper():15} {metric.current_value:8.1f} {metric.unit:12} (threshold: {metric.threshold})")
        print()
        
        print("📈 GROWTH PREDICTIONS")
        print("-" * 50)
        for metric_name, forecasts in report.growth_predictions.items():
            print(f"📊 {metric_name.upper():15} 7-day: {forecasts.get('7d', 0):8.1f}  30-day: {forecasts.get('30d', 0):8.1f}")
        print()
        
        print("⚡ SCALING RECOMMENDATIONS")
        print("-" * 50)
        for recommendation in report.scaling_recommendations[:5]:
            print(f"  {recommendation}")
        print()
        
        print("💰 COST PROJECTIONS")
        print("-" * 50)
        print(f"💵 Current Monthly: ${report.cost_projections['current_monthly']:,.2f}")
        print(f"📊 7-day Forecast: ${report.cost_projections['forecast_7d']:,.2f}")
        print(f"📊 30-day Forecast: ${report.cost_projections['forecast_30d']:,.2f}")
        print()
        
        print("⚠️  RISK ASSESSMENT")
        print("-" * 50)
        for metric_name, risk_level in report.risk_assessment.items():
            risk_icon = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(risk_level, "⚫")
            print(f"{risk_icon} {metric_name.upper():15} {risk_level.upper()}")
        print()
        
        print("✅ ACTION ITEMS")
        print("-" * 50)
        for i, action in enumerate(report.action_items[:5], 1):
            print(f"{i:2d}. {action}")
        print()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="YTEmpire Capacity Planning Tool")
    parser.add_argument('--config', default='capacity_config.yaml', help='Configuration file path')
    parser.add_argument('--output', default='capacity_report.json', help='Output report file')
    parser.add_argument('--format', choices=['json', 'console', 'both'], default='both', help='Output format')
    parser.add_argument('--days', type=int, default=30, help='Historical data days to analyze')
    
    args = parser.parse_args()
    
    try:
        planner = CapacityPlanner(args.config)
        report = planner.generate_report()
        
        if args.format in ['json', 'both']:
            planner.save_report(report, args.output)
        
        if args.format in ['console', 'both']:
            planner.print_report(report)
    
    except Exception as e:
        logger.error(f"Capacity planning failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()