#!/usr/bin/env python3
"""
Advanced Monitoring Dashboards for YTEmpire
Creates comprehensive Grafana dashboards for business metrics, performance, and operations
"""

import json
import yaml
import logging
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from grafana_api import GrafanaApi
    GRAFANA_API_AVAILABLE = True
except ImportError:
    GRAFANA_API_AVAILABLE = False
    logger.warning("Grafana API not available. Install with: pip install grafana-api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:/Users/Hp/projects/ytempire-mvp/logs/platform_ops/dashboard_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedDashboardManager:
    """
    Manages advanced Grafana dashboards for YTEmpire monitoring
    """
    
    def __init__(self, grafana_url: str = "http://localhost:3000", 
                 grafana_token: Optional[str] = None,
                 grafana_user: str = "admin",
                 grafana_password: str = "admin"):
        """
        Initialize dashboard manager
        """
        self.grafana_url = grafana_url
        self.grafana_token = grafana_token
        self.grafana_user = grafana_user
        self.grafana_password = grafana_password
        
        # Initialize Grafana API client
        if GRAFANA_API_AVAILABLE:
            if grafana_token:
                self.grafana = GrafanaApi.from_url(
                    url=grafana_url,
                    credential=grafana_token
                )
            else:
                self.grafana = GrafanaApi.from_url(
                    url=grafana_url,
                    credential=(grafana_user, grafana_password)
                )
        else:
            self.grafana = None
            logger.warning("Grafana API not available - dashboard manager will work in offline mode")
    
    def create_all_dashboards(self) -> Dict[str, Any]:
        """
        Create all advanced dashboards
        """
        logger.info("Creating all advanced monitoring dashboards")
        
        results = {
            'created_dashboards': [],
            'errors': [],
            'timestamp': datetime.now().isoformat()
        }
        
        dashboards = [
            ('business_metrics', self._create_business_dashboard),
            ('operational_overview', self._create_operational_dashboard),
            ('ai_ml_pipeline', self._create_ai_ml_dashboard),
            ('cost_optimization', self._create_cost_dashboard),
            ('security_monitoring', self._create_security_dashboard),
            ('performance_analytics', self._create_performance_dashboard),
            ('video_pipeline_monitoring', self._create_video_pipeline_dashboard),
            ('youtube_api_monitoring', self._create_youtube_api_dashboard),
            ('infrastructure_health', self._create_infrastructure_dashboard),
            ('user_experience_monitoring', self._create_ux_dashboard)
        ]
        
        for name, create_func in dashboards:
            try:
                dashboard = create_func()
                self._deploy_dashboard(dashboard)
                results['created_dashboards'].append(name)
                logger.info(f"Successfully created dashboard: {name}")
            except Exception as e:
                error_msg = f"Failed to create dashboard {name}: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        return results

    def _create_business_dashboard(self) -> Dict[str, Any]:
        """
        Create comprehensive business metrics dashboard
        """
        return {
            "dashboard": {
                "id": None,
                "title": "YTEmpire - Business Metrics",
                "tags": ["ytempire", "business", "kpi"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Revenue Overview",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "sum(revenue_total_dollars)",
                                "legendFormat": "Total Revenue",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "currencyUSD",
                                "color": {"mode": "palette-classic"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": 0},
                                        {"color": "yellow", "value": 1000},
                                        {"color": "green", "value": 5000}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 2,
                        "title": "Videos Generated Today",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
                        "targets": [
                            {
                                "expr": "increase(videos_generated_total[24h])",
                                "legendFormat": "Videos Today",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": 0},
                                        {"color": "yellow", "value": 10},
                                        {"color": "green", "value": 20}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 3,
                        "title": "Cost per Video",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0},
                        "targets": [
                            {
                                "expr": "avg(cost_per_video_dollars)",
                                "legendFormat": "Avg Cost",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "currencyUSD",
                                "color": {"mode": "palette-classic"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": 0},
                                        {"color": "yellow", "value": 2.5},
                                        {"color": "red", "value": 3.0}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 4,
                        "title": "Active Channels",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
                        "targets": [
                            {
                                "expr": "count(channels_active)",
                                "legendFormat": "Active Channels",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 5,
                        "title": "Revenue Trend (7 days)",
                        "type": "timeseries",
                        "gridPos": {"h": 10, "w": 12, "x": 0, "y": 8},
                        "targets": [
                            {
                                "expr": "sum(rate(revenue_total_dollars[1h]))",
                                "legendFormat": "Revenue Rate",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 6,
                        "title": "Video Generation Rate",
                        "type": "timeseries",
                        "gridPos": {"h": 10, "w": 12, "x": 12, "y": 8},
                        "targets": [
                            {
                                "expr": "rate(videos_generated_total[5m])",
                                "legendFormat": "Videos per minute",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 7,
                        "title": "Cost Breakdown by Service",
                        "type": "piechart",
                        "gridPos": {"h": 10, "w": 12, "x": 0, "y": 18},
                        "targets": [
                            {
                                "expr": "sum by (service) (service_cost_dollars)",
                                "legendFormat": "{{service}}",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 8,
                        "title": "Channel Performance",
                        "type": "table",
                        "gridPos": {"h": 10, "w": 12, "x": 12, "y": 18},
                        "targets": [
                            {
                                "expr": "sum by (channel_id) (channel_revenue_dollars)",
                                "legendFormat": "{{channel_id}}",
                                "refId": "A"
                            }
                        ]
                    }
                ],
                "time": {"from": "now-24h", "to": "now"},
                "refresh": "1m"
            }
        }

    def _create_operational_dashboard(self) -> Dict[str, Any]:
        """
        Create operational overview dashboard
        """
        return {
            "dashboard": {
                "id": None,
                "title": "YTEmpire - Operational Overview",
                "tags": ["ytempire", "operations", "infrastructure"],
                "panels": [
                    {
                        "id": 1,
                        "title": "System Health",
                        "type": "stat",
                        "gridPos": {"h": 6, "w": 24, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "up{job=~'ytempire-.*'}",
                                "legendFormat": "{{job}}",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "mappings": [
                                    {"options": {"0": {"text": "DOWN", "color": "red"}}},
                                    {"options": {"1": {"text": "UP", "color": "green"}}}
                                ]
                            }
                        }
                    },
                    {
                        "id": 2,
                        "title": "API Response Times",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 6},
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)",
                                "legendFormat": "95th percentile",
                                "refId": "A"
                            },
                            {
                                "expr": "histogram_quantile(0.50, http_request_duration_seconds_bucket)",
                                "legendFormat": "50th percentile",
                                "refId": "B"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "Request Rate",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 6},
                        "targets": [
                            {
                                "expr": "sum(rate(http_requests_total[5m]))",
                                "legendFormat": "Total RPS",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 4,
                        "title": "Error Rate",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 14},
                        "targets": [
                            {
                                "expr": "sum(rate(http_requests_total{status=~'5..'}[5m])) / sum(rate(http_requests_total[5m]))",
                                "legendFormat": "Error Rate",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 5,
                        "title": "Database Performance",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 14},
                        "targets": [
                            {
                                "expr": "avg(pg_stat_database_tup_returned)",
                                "legendFormat": "Tuples Returned",
                                "refId": "A"
                            }
                        ]
                    }
                ]
            }
        }

    def _create_ai_ml_dashboard(self) -> Dict[str, Any]:
        """
        Create AI/ML pipeline monitoring dashboard
        """
        return {
            "dashboard": {
                "id": None,
                "title": "YTEmpire - AI/ML Pipeline",
                "tags": ["ytempire", "ai", "ml", "pipeline"],
                "panels": [
                    {
                        "id": 1,
                        "title": "Model Inference Latency",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, ml_inference_duration_seconds_bucket)",
                                "legendFormat": "{{model_name}} - p95",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Model Accuracy",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "targets": [
                            {
                                "expr": "avg(model_accuracy_score)",
                                "legendFormat": "Accuracy",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "API Usage by Service",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                        "targets": [
                            {
                                "expr": "sum by (api_service) (rate(api_calls_total[5m]))",
                                "legendFormat": "{{api_service}}",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 4,
                        "title": "Content Quality Scores",
                        "type": "histogram",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                        "targets": [
                            {
                                "expr": "content_quality_score",
                                "legendFormat": "Quality Score",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 5,
                        "title": "Video Processing Pipeline",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
                        "targets": [
                            {
                                "expr": "sum by (stage) (rate(pipeline_stage_duration_seconds[5m]))",
                                "legendFormat": "{{stage}}",
                                "refId": "A"
                            }
                        ]
                    }
                ]
            }
        }

    def _create_cost_dashboard(self) -> Dict[str, Any]:
        """
        Create cost optimization dashboard
        """
        return {
            "dashboard": {
                "id": None,
                "title": "YTEmpire - Cost Optimization",
                "tags": ["ytempire", "cost", "optimization", "financial"],
                "panels": [
                    {
                        "id": 1,
                        "title": "Daily Cost Trend",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "sum(increase(total_cost_dollars[24h]))",
                                "legendFormat": "Daily Cost",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Cost per Video Target vs Actual",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "targets": [
                            {
                                "expr": "avg(cost_per_video_dollars)",
                                "legendFormat": "Actual Cost",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": 0},
                                        {"color": "yellow", "value": 2.5},
                                        {"color": "red", "value": 3.0}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 3,
                        "title": "Cost Breakdown by Service",
                        "type": "piechart",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                        "targets": [
                            {
                                "expr": "sum by (service) (service_cost_dollars)",
                                "legendFormat": "{{service}}",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 4,
                        "title": "Cost Optimization Opportunities",
                        "type": "table",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                        "targets": [
                            {
                                "expr": "cost_optimization_potential_dollars",
                                "refId": "A"
                            }
                        ]
                    }
                ]
            }
        }

    def _create_security_dashboard(self) -> Dict[str, Any]:
        """
        Create security monitoring dashboard
        """
        return {
            "dashboard": {
                "id": None,
                "title": "YTEmpire - Security Monitoring",
                "tags": ["ytempire", "security", "monitoring"],
                "panels": [
                    {
                        "id": 1,
                        "title": "Failed Login Attempts",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "sum(rate(auth_failures_total[5m]))",
                                "legendFormat": "Failed Logins/min",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "API Rate Limiting",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "targets": [
                            {
                                "expr": "sum(rate(rate_limit_exceeded_total[5m]))",
                                "legendFormat": "Rate Limited Requests",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "SSL Certificate Expiry",
                        "type": "stat",
                        "gridPos": {"h": 6, "w": 12, "x": 0, "y": 8},
                        "targets": [
                            {
                                "expr": "ssl_certificate_days_until_expiry",
                                "legendFormat": "Days Until Expiry",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 4,
                        "title": "Vulnerability Scan Results",
                        "type": "stat",
                        "gridPos": {"h": 6, "w": 12, "x": 12, "y": 8},
                        "targets": [
                            {
                                "expr": "security_vulnerabilities_count",
                                "legendFormat": "{{severity}}",
                                "refId": "A"
                            }
                        ]
                    }
                ]
            }
        }

    def _create_performance_dashboard(self) -> Dict[str, Any]:
        """
        Create performance analytics dashboard
        """
        return {
            "dashboard": {
                "id": None,
                "title": "YTEmpire - Performance Analytics",
                "tags": ["ytempire", "performance", "analytics"],
                "panels": [
                    {
                        "id": 1,
                        "title": "Resource Utilization",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "100 - (avg(node_cpu_seconds_total{mode='idle'}) * 100)",
                                "legendFormat": "CPU Usage %",
                                "refId": "A"
                            },
                            {
                                "expr": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100",
                                "legendFormat": "Memory Usage %",
                                "refId": "B"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Disk I/O",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "targets": [
                            {
                                "expr": "rate(node_disk_read_bytes_total[5m])",
                                "legendFormat": "Read",
                                "refId": "A"
                            },
                            {
                                "expr": "rate(node_disk_written_bytes_total[5m])",
                                "legendFormat": "Write",
                                "refId": "B"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "Network Traffic",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                        "targets": [
                            {
                                "expr": "rate(node_network_receive_bytes_total[5m])",
                                "legendFormat": "Received",
                                "refId": "A"
                            },
                            {
                                "expr": "rate(node_network_transmit_bytes_total[5m])",
                                "legendFormat": "Transmitted",
                                "refId": "B"
                            }
                        ]
                    },
                    {
                        "id": 4,
                        "title": "GPU Utilization",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                        "targets": [
                            {
                                "expr": "DCGM_FI_DEV_GPU_UTIL",
                                "legendFormat": "GPU {{gpu}}",
                                "refId": "A"
                            }
                        ]
                    }
                ]
            }
        }

    def _create_video_pipeline_dashboard(self) -> Dict[str, Any]:
        """
        Create video processing pipeline dashboard
        """
        return {
            "dashboard": {
                "id": None,
                "title": "YTEmpire - Video Pipeline",
                "tags": ["ytempire", "video", "pipeline", "processing"],
                "panels": [
                    {
                        "id": 1,
                        "title": "Pipeline Stage Durations",
                        "type": "timeseries",
                        "gridPos": {"h": 10, "w": 24, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "avg by (stage) (pipeline_stage_duration_seconds)",
                                "legendFormat": "{{stage}}",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Queue Depth",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 10},
                        "targets": [
                            {
                                "expr": "celery_queue_length",
                                "legendFormat": "{{queue}}",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "Processing Success Rate",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 10},
                        "targets": [
                            {
                                "expr": "sum(rate(pipeline_success_total[5m])) / sum(rate(pipeline_attempts_total[5m])) * 100",
                                "legendFormat": "Success Rate %",
                                "refId": "A"
                            }
                        ]
                    }
                ]
            }
        }

    def _create_youtube_api_dashboard(self) -> Dict[str, Any]:
        """
        Create YouTube API monitoring dashboard
        """
        return {
            "dashboard": {
                "id": None,
                "title": "YTEmpire - YouTube API Monitoring",
                "tags": ["ytempire", "youtube", "api", "quota"],
                "panels": [
                    {
                        "id": 1,
                        "title": "API Quota Usage",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "youtube_quota_used / youtube_quota_limit * 100",
                                "legendFormat": "Quota Usage %",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "API Calls by Type",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "targets": [
                            {
                                "expr": "sum by (api_method) (rate(youtube_api_calls_total[5m]))",
                                "legendFormat": "{{api_method}}",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "Account Health Score",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                        "targets": [
                            {
                                "expr": "avg(youtube_account_health_score)",
                                "legendFormat": "Health Score",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 4,
                        "title": "Upload Success Rate",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                        "targets": [
                            {
                                "expr": "sum(rate(youtube_upload_success_total[5m])) / sum(rate(youtube_upload_attempts_total[5m])) * 100",
                                "legendFormat": "Success Rate %",
                                "refId": "A"
                            }
                        ]
                    }
                ]
            }
        }

    def _create_infrastructure_dashboard(self) -> Dict[str, Any]:
        """
        Create infrastructure health dashboard
        """
        return {
            "dashboard": {
                "id": None,
                "title": "YTEmpire - Infrastructure Health",
                "tags": ["ytempire", "infrastructure", "health"],
                "panels": [
                    {
                        "id": 1,
                        "title": "Container Status",
                        "type": "stat",
                        "gridPos": {"h": 6, "w": 24, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "up{job=~'ytempire-.*'}",
                                "legendFormat": "{{job}}",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Storage Usage",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 6},
                        "targets": [
                            {
                                "expr": "(node_filesystem_size_bytes - node_filesystem_avail_bytes) / node_filesystem_size_bytes * 100",
                                "legendFormat": "{{mountpoint}}",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "Load Average",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 6},
                        "targets": [
                            {
                                "expr": "node_load1",
                                "legendFormat": "1m",
                                "refId": "A"
                            },
                            {
                                "expr": "node_load5",
                                "legendFormat": "5m",
                                "refId": "B"
                            },
                            {
                                "expr": "node_load15",
                                "legendFormat": "15m",
                                "refId": "C"
                            }
                        ]
                    }
                ]
            }
        }

    def _create_ux_dashboard(self) -> Dict[str, Any]:
        """
        Create user experience monitoring dashboard
        """
        return {
            "dashboard": {
                "id": None,
                "title": "YTEmpire - User Experience",
                "tags": ["ytempire", "ux", "frontend", "performance"],
                "panels": [
                    {
                        "id": 1,
                        "title": "Page Load Times",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, frontend_page_load_duration_seconds_bucket)",
                                "legendFormat": "95th percentile",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "JavaScript Errors",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "targets": [
                            {
                                "expr": "rate(frontend_js_errors_total[5m])",
                                "legendFormat": "Errors/min",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "User Sessions",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                        "targets": [
                            {
                                "expr": "sum(active_user_sessions)",
                                "legendFormat": "Active Sessions",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 4,
                        "title": "Feature Usage",
                        "type": "table",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                        "targets": [
                            {
                                "expr": "sum by (feature) (feature_usage_total)",
                                "refId": "A"
                            }
                        ]
                    }
                ]
            }
        }

    def _deploy_dashboard(self, dashboard: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy dashboard to Grafana
        """
        try:
            result = self.grafana.dashboard.update_dashboard(dashboard)
            logger.info(f"Dashboard deployed: {dashboard['dashboard']['title']}")
            return result
        except Exception as e:
            logger.error(f"Failed to deploy dashboard: {e}")
            raise

    def setup_dashboard_folders(self) -> Dict[str, Any]:
        """
        Set up organized dashboard folders
        """
        logger.info("Setting up dashboard folders")
        
        folders = [
            {"title": "YTEmpire Business", "uid": "ytempire-business"},
            {"title": "YTEmpire Operations", "uid": "ytempire-ops"},
            {"title": "YTEmpire Infrastructure", "uid": "ytempire-infra"},
            {"title": "YTEmpire Security", "uid": "ytempire-security"}
        ]
        
        created_folders = []
        for folder in folders:
            try:
                result = self.grafana.folder.create_folder(
                    title=folder["title"],
                    uid=folder["uid"]
                )
                created_folders.append(result)
                logger.info(f"Created folder: {folder['title']}")
            except Exception as e:
                logger.warning(f"Folder may already exist: {folder['title']} - {e}")
        
        return {"created_folders": created_folders}

    def export_dashboards(self, export_path: str = "C:/Users/Hp/projects/ytempire-mvp/logs/platform_ops/dashboards") -> Dict[str, Any]:
        """
        Export all dashboards for backup/version control
        """
        logger.info("Exporting dashboards")
        
        os.makedirs(export_path, exist_ok=True)
        
        try:
            dashboards = self.grafana.search.search_dashboards()
            exported_dashboards = []
            
            for dashboard in dashboards:
                if 'ytempire' in dashboard.get('title', '').lower():
                    dashboard_detail = self.grafana.dashboard.get_dashboard(dashboard['uid'])
                    
                    filename = f"{dashboard['title'].replace(' ', '_').lower()}.json"
                    filepath = os.path.join(export_path, filename)
                    
                    with open(filepath, 'w') as f:
                        json.dump(dashboard_detail, f, indent=2)
                    
                    exported_dashboards.append(filepath)
                    logger.info(f"Exported dashboard: {dashboard['title']}")
            
            return {
                "exported_dashboards": exported_dashboards,
                "export_path": export_path,
                "count": len(exported_dashboards)
            }
            
        except Exception as e:
            logger.error(f"Dashboard export failed: {e}")
            raise

def main():
    """Main execution function"""
    try:
        # Initialize dashboard manager
        dashboard_manager = AdvancedDashboardManager()
        
        # Set up folders
        logger.info("Setting up dashboard organization")
        dashboard_manager.setup_dashboard_folders()
        
        # Create all dashboards
        logger.info("Creating advanced monitoring dashboards")
        results = dashboard_manager.create_all_dashboards()
        
        # Export dashboards for backup
        logger.info("Exporting dashboards for backup")
        export_results = dashboard_manager.export_dashboards()
        
        # Generate summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "dashboard_creation": results,
            "dashboard_export": export_results,
            "status": "success" if not results["errors"] else "partial_success"
        }
        
        # Save summary
        summary_path = "C:/Users/Hp/projects/ytempire-mvp/logs/platform_ops/dashboard_setup_summary.json"
        os.makedirs('C:/Users/Hp/projects/ytempire-mvp/logs/platform_ops', exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Dashboard setup complete. Summary saved to: {summary_path}")
        
        if results["errors"]:
            logger.warning(f"Some dashboards failed to create: {results['errors']}")
        else:
            logger.info("All dashboards created successfully!")
        
    except Exception as e:
        logger.error(f"Advanced dashboard setup failed: {e}")
        raise

if __name__ == "__main__":
    main()