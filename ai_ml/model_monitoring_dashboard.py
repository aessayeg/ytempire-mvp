"""
Model Monitoring Dashboard
Real-time monitoring of ML model health, drift, and performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import redis.asyncio as redis
from prometheus_client import Histogram, Counter, Gauge
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st
import mlflow
from scipy import stats
import logging

# Metrics
model_inference_count = Counter('model_inference_total', 'Total model inferences', ['model_name', 'version'])
model_error_count = Counter('model_errors_total', 'Total model errors', ['model_name', 'error_type'])
drift_score = Gauge('model_drift_score', 'Model drift score', ['model_name', 'drift_type'])
model_latency_histogram = Histogram('model_inference_latency', 'Model inference latency', ['model_name'])

logger = logging.getLogger(__name__)

@dataclass
class ModelHealth:
    """Model health status"""
    model_name: str
    version: str
    status: str  # 'healthy', 'degraded', 'critical'
    accuracy: float
    latency_p50: float
    latency_p95: float
    error_rate: float
    drift_detected: bool
    last_updated: datetime

class ModelMonitoringDashboard:
    """Real-time model monitoring dashboard"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = None
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # Model registry
        self.monitored_models = {
            'content_generator': {
                'baseline_accuracy': 0.85,
                'latency_threshold': 100,
                'error_threshold': 0.05
            },
            'thumbnail_generator': {
                'baseline_accuracy': 0.80,
                'latency_threshold': 500,
                'error_threshold': 0.10
            },
            'quality_scorer': {
                'baseline_accuracy': 0.90,
                'latency_threshold': 50,
                'error_threshold': 0.03
            }
        }
    
    async def initialize(self):
        """Initialize components"""
        self.redis_client = await redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=True
        )
    
    def create_dashboard(self):
        """Create Streamlit dashboard"""
        st.set_page_config(
            page_title="Model Monitoring Dashboard",
            page_icon="🤖",
            layout="wide"
        )
        
        st.title("🤖 ML Model Monitoring Dashboard")
        st.markdown("---")
        
        # Sidebar
        with st.sidebar:
            st.header("Configuration")
            selected_model = st.selectbox(
                "Select Model",
                list(self.monitored_models.keys())
            )
            
            refresh_interval = st.slider(
                "Refresh Interval (seconds)",
                min_value=5,
                max_value=60,
                value=10
            )
            
            st.markdown("---")
            st.header("Alerts Configuration")
            
            accuracy_threshold = st.slider(
                "Accuracy Alert Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.05
            )
            
            latency_threshold = st.slider(
                "Latency Alert (ms)",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )
        
        # Main content
        asyncio.run(self._render_dashboard(selected_model))
        
        # Auto-refresh
        if st.button("🔄 Refresh"):
            st.experimental_rerun()
    
    async def _render_dashboard(self, model_name: str):
        """Render dashboard components"""
        
        # Model Health Overview
        col1, col2, col3, col4 = st.columns(4)
        
        health = await self.get_model_health(model_name)
        
        with col1:
            self._render_metric_card(
                "Model Status",
                health.status,
                self._get_status_color(health.status)
            )
        
        with col2:
            self._render_metric_card(
                "Accuracy",
                f"{health.accuracy:.2%}",
                "green" if health.accuracy > 0.8 else "orange"
            )
        
        with col3:
            self._render_metric_card(
                "P95 Latency",
                f"{health.latency_p95:.1f}ms",
                "green" if health.latency_p95 < 100 else "orange"
            )
        
        with col4:
            self._render_metric_card(
                "Error Rate",
                f"{health.error_rate:.2%}",
                "green" if health.error_rate < 0.05 else "red"
            )
        
        st.markdown("---")
        
        # Performance Metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Performance Metrics")
            perf_fig = await self._create_performance_chart(model_name)
            st.plotly_chart(perf_fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Prediction Distribution")
            dist_fig = await self._create_distribution_chart(model_name)
            st.plotly_chart(dist_fig, use_container_width=True)
        
        # Drift Detection
        st.markdown("---")
        st.subheader("🎯 Drift Detection")
        
        col1, col2, col3 = st.columns(3)
        
        drift_metrics = await self.detect_drift(model_name)
        
        with col1:
            st.metric(
                "Data Drift",
                f"{drift_metrics['data_drift']:.3f}",
                delta=f"{drift_metrics['data_drift_change']:.3f}"
            )
        
        with col2:
            st.metric(
                "Concept Drift",
                f"{drift_metrics['concept_drift']:.3f}",
                delta=f"{drift_metrics['concept_drift_change']:.3f}"
            )
        
        with col3:
            st.metric(
                "Prediction Drift",
                f"{drift_metrics['prediction_drift']:.3f}",
                delta=f"{drift_metrics['prediction_drift_change']:.3f}"
            )
        
        # Drift visualization
        drift_fig = await self._create_drift_chart(model_name)
        st.plotly_chart(drift_fig, use_container_width=True)
        
        # Feature Importance
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Feature Importance")
            importance_fig = await self._create_feature_importance_chart(model_name)
            st.plotly_chart(importance_fig, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Anomaly Detection")
            anomaly_fig = await self._create_anomaly_chart(model_name)
            st.plotly_chart(anomaly_fig, use_container_width=True)
        
        # Model Comparison
        st.markdown("---")
        st.subheader("📊 Model Version Comparison")
        comparison_fig = await self._create_model_comparison_chart(model_name)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Alerts Section
        st.markdown("---")
        st.subheader("🚨 Active Alerts")
        alerts = await self.get_active_alerts(model_name)
        self._render_alerts(alerts)
    
    def _render_metric_card(self, title: str, value: str, color: str):
        """Render metric card"""
        st.markdown(
            f"""
            <div style="padding: 1rem; background-color: {color}15; border-radius: 0.5rem; border-left: 4px solid {color};">
                <h4 style="margin: 0; color: #666;">{title}</h4>
                <h2 style="margin: 0; color: {color};">{value}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def _get_status_color(self, status: str) -> str:
        """Get color for status"""
        colors = {
            'healthy': '#4CAF50',
            'degraded': '#FF9800',
            'critical': '#F44336'
        }
        return colors.get(status, '#9E9E9E')
    
    async def get_model_health(self, model_name: str) -> ModelHealth:
        """Get model health status"""
        # Fetch metrics from cache/monitoring system
        metrics = await self._fetch_model_metrics(model_name)
        
        # Determine health status
        status = 'healthy'
        if metrics['accuracy'] < 0.7 or metrics['error_rate'] > 0.1:
            status = 'critical'
        elif metrics['accuracy'] < 0.8 or metrics['error_rate'] > 0.05:
            status = 'degraded'
        
        return ModelHealth(
            model_name=model_name,
            version=metrics.get('version', '1.0'),
            status=status,
            accuracy=metrics['accuracy'],
            latency_p50=metrics['latency_p50'],
            latency_p95=metrics['latency_p95'],
            error_rate=metrics['error_rate'],
            drift_detected=metrics.get('drift_detected', False),
            last_updated=datetime.now()
        )
    
    async def _fetch_model_metrics(self, model_name: str) -> Dict:
        """Fetch model metrics from monitoring system"""
        # Check cache
        if self.redis_client:
            cached = await self.redis_client.get(f"model_metrics:{model_name}")
            if cached:
                import json
                return json.loads(cached)
        
        # Generate sample metrics (replace with actual monitoring)
        metrics = {
            'accuracy': np.random.normal(0.85, 0.05),
            'latency_p50': np.random.gamma(2, 20),
            'latency_p95': np.random.gamma(3, 30),
            'error_rate': np.random.beta(1, 20),
            'version': '1.2.3',
            'drift_detected': np.random.random() > 0.8
        }
        
        # Cache metrics
        if self.redis_client:
            import json
            await self.redis_client.setex(
                f"model_metrics:{model_name}",
                60,
                json.dumps(metrics)
            )
        
        return metrics
    
    async def detect_drift(self, model_name: str) -> Dict:
        """Detect model drift"""
        # Fetch recent predictions and ground truth
        recent_data = await self._fetch_recent_predictions(model_name)
        baseline_data = await self._fetch_baseline_data(model_name)
        
        # Calculate drift metrics
        data_drift = self._calculate_data_drift(recent_data, baseline_data)
        concept_drift = self._calculate_concept_drift(recent_data, baseline_data)
        prediction_drift = self._calculate_prediction_drift(recent_data, baseline_data)
        
        # Calculate changes
        prev_drift = await self._fetch_previous_drift(model_name)
        
        result = {
            'data_drift': data_drift,
            'data_drift_change': data_drift - prev_drift.get('data_drift', data_drift),
            'concept_drift': concept_drift,
            'concept_drift_change': concept_drift - prev_drift.get('concept_drift', concept_drift),
            'prediction_drift': prediction_drift,
            'prediction_drift_change': prediction_drift - prev_drift.get('prediction_drift', prediction_drift)
        }
        
        # Update drift metrics
        drift_score.labels(model_name=model_name, drift_type='data').set(data_drift)
        drift_score.labels(model_name=model_name, drift_type='concept').set(concept_drift)
        drift_score.labels(model_name=model_name, drift_type='prediction').set(prediction_drift)
        
        # Cache current drift
        await self._cache_drift_metrics(model_name, result)
        
        return result
    
    def _calculate_data_drift(self, recent: pd.DataFrame, baseline: pd.DataFrame) -> float:
        """Calculate data drift using KS statistic"""
        if recent.empty or baseline.empty:
            return 0.0
        
        drift_scores = []
        for column in recent.select_dtypes(include=[np.number]).columns:
            if column in baseline.columns:
                statistic, _ = stats.ks_2samp(recent[column], baseline[column])
                drift_scores.append(statistic)
        
        return np.mean(drift_scores) if drift_scores else 0.0
    
    def _calculate_concept_drift(self, recent: pd.DataFrame, baseline: pd.DataFrame) -> float:
        """Calculate concept drift"""
        # Simplified: compare prediction distributions
        if 'prediction' not in recent.columns or 'prediction' not in baseline.columns:
            return 0.0
        
        statistic, _ = stats.ks_2samp(recent['prediction'], baseline['prediction'])
        return statistic
    
    def _calculate_prediction_drift(self, recent: pd.DataFrame, baseline: pd.DataFrame) -> float:
        """Calculate prediction drift"""
        if 'prediction' not in recent.columns or 'prediction' not in baseline.columns:
            return 0.0
        
        # Compare prediction distributions
        recent_mean = recent['prediction'].mean()
        baseline_mean = baseline['prediction'].mean()
        
        return abs(recent_mean - baseline_mean) / (baseline_mean + 1e-10)
    
    async def _create_performance_chart(self, model_name: str) -> go.Figure:
        """Create performance metrics chart"""
        # Fetch historical metrics
        timestamps = pd.date_range(end=datetime.now(), periods=24, freq='H')
        accuracy = np.random.normal(0.85, 0.02, 24)
        latency = np.random.gamma(2, 30, 24)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Accuracy Over Time', 'Latency Over Time'),
            vertical_spacing=0.15
        )
        
        # Accuracy
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=accuracy,
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='#2E7D32')
            ),
            row=1, col=1
        )
        
        # Add threshold line
        fig.add_hline(
            y=0.8,
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold",
            row=1, col=1
        )
        
        # Latency
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=latency,
                mode='lines+markers',
                name='Latency (ms)',
                line=dict(color='#1976D2')
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=500, showlegend=False)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=2, col=1)
        
        return fig
    
    async def _create_distribution_chart(self, model_name: str) -> go.Figure:
        """Create prediction distribution chart"""
        # Generate sample predictions
        predictions = np.random.beta(2, 2, 1000)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=predictions,
            nbinsx=30,
            name='Predictions',
            marker_color='#7B1FA2'
        ))
        
        fig.update_layout(
            title="Prediction Distribution",
            xaxis_title="Prediction Value",
            yaxis_title="Count",
            height=400
        )
        
        return fig
    
    async def _create_drift_chart(self, model_name: str) -> go.Figure:
        """Create drift visualization"""
        timestamps = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        # Generate sample drift scores
        data_drift = np.random.beta(2, 8, 30)
        concept_drift = np.random.beta(2, 10, 30)
        prediction_drift = np.random.beta(2, 12, 30)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=data_drift,
            mode='lines',
            name='Data Drift',
            line=dict(color='#FF5722')
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=concept_drift,
            mode='lines',
            name='Concept Drift',
            line=dict(color='#FFC107')
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=prediction_drift,
            mode='lines',
            name='Prediction Drift',
            line=dict(color='#9C27B0')
        ))
        
        # Add danger zone
        fig.add_hrect(
            y0=0.5, y1=1,
            fillcolor="red", opacity=0.1,
            annotation_text="Danger Zone"
        )
        
        fig.update_layout(
            title="Drift Metrics Over Time",
            xaxis_title="Date",
            yaxis_title="Drift Score",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    async def _create_feature_importance_chart(self, model_name: str) -> go.Figure:
        """Create feature importance chart"""
        # Sample feature importance
        features = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        importance = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='#4CAF50'
        ))
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=400
        )
        
        return fig
    
    async def _create_anomaly_chart(self, model_name: str) -> go.Figure:
        """Create anomaly detection chart"""
        timestamps = pd.date_range(end=datetime.now(), periods=100, freq='15min')
        values = np.random.normal(100, 10, 100)
        
        # Add some anomalies
        anomaly_indices = [20, 45, 78]
        for idx in anomaly_indices:
            values[idx] = values[idx] * 2
        
        fig = go.Figure()
        
        # Normal points
        normal_mask = np.ones(len(values), dtype=bool)
        normal_mask[anomaly_indices] = False
        
        fig.add_trace(go.Scatter(
            x=timestamps[normal_mask],
            y=values[normal_mask],
            mode='markers',
            name='Normal',
            marker=dict(color='#2196F3', size=6)
        ))
        
        # Anomalies
        fig.add_trace(go.Scatter(
            x=timestamps[anomaly_indices],
            y=values[anomaly_indices],
            mode='markers',
            name='Anomaly',
            marker=dict(color='#F44336', size=12, symbol='x')
        ))
        
        fig.update_layout(
            title="Anomaly Detection",
            xaxis_title="Time",
            yaxis_title="Metric Value",
            height=400
        )
        
        return fig
    
    async def _create_model_comparison_chart(self, model_name: str) -> go.Figure:
        """Create model version comparison chart"""
        versions = ['v1.0', 'v1.1', 'v1.2', 'v2.0', 'v2.1']
        metrics_names = ['Accuracy', 'Latency', 'Memory', 'Throughput']
        
        # Generate sample data
        data = np.random.rand(len(versions), len(metrics_names))
        
        fig = go.Figure()
        
        for i, version in enumerate(versions):
            fig.add_trace(go.Scatterpolar(
                r=data[i],
                theta=metrics_names,
                fill='toself',
                name=version
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Model Version Comparison",
            height=500,
            showlegend=True
        )
        
        return fig
    
    async def get_active_alerts(self, model_name: str) -> List[Dict]:
        """Get active alerts for model"""
        alerts = []
        
        # Check model health
        health = await self.get_model_health(model_name)
        
        if health.status == 'critical':
            alerts.append({
                'severity': 'critical',
                'title': 'Model Health Critical',
                'message': f'Model {model_name} is in critical state',
                'timestamp': datetime.now()
            })
        
        if health.accuracy < 0.8:
            alerts.append({
                'severity': 'warning',
                'title': 'Low Accuracy',
                'message': f'Accuracy ({health.accuracy:.2%}) below threshold',
                'timestamp': datetime.now()
            })
        
        if health.latency_p95 > 200:
            alerts.append({
                'severity': 'warning',
                'title': 'High Latency',
                'message': f'P95 latency ({health.latency_p95:.1f}ms) above threshold',
                'timestamp': datetime.now()
            })
        
        if health.drift_detected:
            alerts.append({
                'severity': 'warning',
                'title': 'Drift Detected',
                'message': 'Significant drift detected in model predictions',
                'timestamp': datetime.now()
            })
        
        return alerts
    
    def _render_alerts(self, alerts: List[Dict]):
        """Render alerts"""
        if not alerts:
            st.success("✅ No active alerts")
        else:
            for alert in alerts:
                if alert['severity'] == 'critical':
                    st.error(f"🔴 **{alert['title']}**: {alert['message']}")
                else:
                    st.warning(f"🟡 **{alert['title']}**: {alert['message']}")
    
    async def _fetch_recent_predictions(self, model_name: str) -> pd.DataFrame:
        """Fetch recent predictions"""
        # Generate sample data
        return pd.DataFrame({
            'prediction': np.random.beta(2, 2, 100),
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(0, 1, 100)
        })
    
    async def _fetch_baseline_data(self, model_name: str) -> pd.DataFrame:
        """Fetch baseline data"""
        # Generate sample baseline
        return pd.DataFrame({
            'prediction': np.random.beta(2.1, 2, 100),
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(0.1, 1, 100)
        })
    
    async def _fetch_previous_drift(self, model_name: str) -> Dict:
        """Fetch previous drift metrics"""
        if self.redis_client:
            cached = await self.redis_client.get(f"drift_metrics:{model_name}")
            if cached:
                import json
                return json.loads(cached)
        return {}
    
    async def _cache_drift_metrics(self, model_name: str, metrics: Dict):
        """Cache drift metrics"""
        if self.redis_client:
            import json
            await self.redis_client.setex(
                f"drift_metrics:{model_name}",
                3600,
                json.dumps(metrics)
            )

# Run dashboard
def main():
    dashboard = ModelMonitoringDashboard()
    asyncio.run(dashboard.initialize())
    dashboard.create_dashboard()

if __name__ == "__main__":
    main()