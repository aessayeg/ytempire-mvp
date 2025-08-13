#!/usr/bin/env python3
"""
Enhanced Cost Optimization Review and Resource Utilization Analysis
Targets 20% cost reduction while maintaining performance
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import psutil
import docker
import boto3
import aiohttp
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.services.cost_optimizer import CostOptimizer, ServiceType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    """Container/service resource metrics"""
    service_name: str
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    network_in_mb: float
    network_out_mb: float
    disk_io_mb: float
    uptime_hours: float
    request_count: int = 0
    error_rate: float = 0.0

@dataclass
class ServiceCostProfile:
    """Cost profile for a service"""
    service: str
    current_monthly_cost: float
    optimal_monthly_cost: float
    savings_potential: float
    utilization_percent: float
    recommended_scaling: str
    auto_scaling_enabled: bool
    
@dataclass
class OptimizationOpportunity:
    """Identified optimization opportunity"""
    category: str
    service: str
    current_state: str
    recommended_state: str
    estimated_savings_percent: float
    estimated_monthly_savings: float
    implementation_effort: str  # low, medium, high
    priority: int  # 1-5, 1 being highest
    risk_level: str  # low, medium, high

class ResourceType(Enum):
    """Resource types for analysis"""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    API = "api"
    DATABASE = "database"
    CACHE = "cache"

class CostOptimizationAnalyzer:
    """Comprehensive cost optimization and resource analysis"""
    
    def __init__(self):
        self.docker_client = None
        try:
            self.docker_client = docker.from_env()
        except:
            logger.warning("Docker client not available")
            
        self.cost_optimizer = CostOptimizer()
        self.metrics_history = []
        self.optimization_opportunities = []
        
    async def perform_complete_analysis(self) -> Dict[str, Any]:
        """Perform complete cost and resource analysis"""
        logger.info("Starting comprehensive cost optimization analysis")
        
        analysis_result = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "resource_utilization": {},
            "cost_analysis": {},
            "optimization_opportunities": [],
            "auto_scaling_recommendations": {},
            "implementation_plan": {}
        }
        
        try:
            # 1. Analyze current resource utilization
            logger.info("Analyzing resource utilization...")
            analysis_result["resource_utilization"] = await self.analyze_resource_utilization()
            
            # 2. Analyze current costs
            logger.info("Analyzing current costs...")
            analysis_result["cost_analysis"] = await self.analyze_current_costs()
            
            # 3. Identify optimization opportunities
            logger.info("Identifying optimization opportunities...")
            analysis_result["optimization_opportunities"] = await self.identify_optimization_opportunities()
            
            # 4. Generate auto-scaling recommendations
            logger.info("Generating auto-scaling recommendations...")
            analysis_result["auto_scaling_recommendations"] = await self.generate_autoscaling_recommendations()
            
            # 5. Create implementation plan
            logger.info("Creating implementation plan...")
            analysis_result["implementation_plan"] = await self.create_implementation_plan()
            
            # 6. Calculate summary metrics
            analysis_result["summary"] = self.calculate_summary_metrics(analysis_result)
            
            # 7. Generate visualizations
            await self.generate_cost_visualizations(analysis_result)
            
            # 8. Generate detailed report
            await self.generate_optimization_report(analysis_result)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            analysis_result["error"] = str(e)
            
        return analysis_result
    
    async def analyze_resource_utilization(self) -> Dict[str, Any]:
        """Analyze current resource utilization across all services"""
        utilization_data = {
            "system_resources": {},
            "container_resources": {},
            "service_metrics": {},
            "efficiency_score": 0
        }
        
        # System-level resources
        utilization_data["system_resources"] = {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "cores": psutil.cpu_count(),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "used_gb": psutil.virtual_memory().used / (1024**3),
                "percent": psutil.virtual_memory().percent,
                "available_gb": psutil.virtual_memory().available / (1024**3)
            },
            "disk": {
                "total_gb": psutil.disk_usage('/').total / (1024**3),
                "used_gb": psutil.disk_usage('/').used / (1024**3),
                "percent": psutil.disk_usage('/').percent
            },
            "network": {
                "bytes_sent_gb": psutil.net_io_counters().bytes_sent / (1024**3),
                "bytes_recv_gb": psutil.net_io_counters().bytes_recv / (1024**3)
            }
        }
        
        # Container-level resources (if Docker available)
        if self.docker_client:
            containers = self.docker_client.containers.list()
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU percentage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                  stats['precpu_stats']['system_cpu_usage']
                    cpu_percent = (cpu_delta / system_delta) * 100 if system_delta > 0 else 0
                    
                    # Calculate memory usage
                    memory_usage = stats['memory_stats'].get('usage', 0) / (1024**2)  # MB
                    memory_limit = stats['memory_stats'].get('limit', 1) / (1024**2)  # MB
                    memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
                    
                    utilization_data["container_resources"][container.name] = {
                        "cpu_percent": round(cpu_percent, 2),
                        "memory_mb": round(memory_usage, 2),
                        "memory_percent": round(memory_percent, 2),
                        "status": container.status
                    }
                except Exception as e:
                    logger.warning(f"Failed to get stats for container {container.name}: {e}")
        
        # Service-specific metrics
        services = [
            "backend", "celery_worker", "redis", "postgres",
            "frontend", "nginx", "flower", "grafana", "prometheus"
        ]
        
        for service in services:
            utilization_data["service_metrics"][service] = await self.get_service_metrics(service)
        
        # Calculate efficiency score
        utilization_data["efficiency_score"] = self.calculate_efficiency_score(utilization_data)
        
        return utilization_data
    
    async def analyze_current_costs(self) -> Dict[str, Any]:
        """Analyze current costs and spending patterns"""
        cost_analysis = {
            "current_month": {},
            "last_month": {},
            "trends": {},
            "by_service": {},
            "by_resource_type": {},
            "cost_per_video": 0,
            "projected_monthly": 0
        }
        
        # Get cost report from optimizer
        cost_report = await self.cost_optimizer.get_cost_report(days=30)
        
        cost_analysis["current_month"] = {
            "total": cost_report.get("total_cost", 0),
            "videos_generated": cost_report.get("total_videos", 0),
            "cost_per_video": cost_report.get("cost_per_video", 0),
            "within_target": cost_report.get("within_target", False)
        }
        
        # Analyze by service
        if "costs_by_service" in cost_report:
            for service, data in cost_report["costs_by_service"].items():
                cost_analysis["by_service"][service] = {
                    "total": data.get("total", 0),
                    "average": data.get("average", 0),
                    "count": data.get("count", 0),
                    "percentage_of_total": (data.get("total", 0) / cost_report.get("total_cost", 1)) * 100
                }
        
        # Analyze by resource type
        cost_analysis["by_resource_type"] = {
            ResourceType.API.value: sum(
                data.get("total", 0) 
                for service, data in cost_analysis["by_service"].items()
                if "api" in service.lower() or "generation" in service.lower()
            ),
            ResourceType.COMPUTE.value: self.estimate_compute_costs(),
            ResourceType.STORAGE.value: self.estimate_storage_costs(),
            ResourceType.NETWORK.value: self.estimate_network_costs(),
            ResourceType.DATABASE.value: self.estimate_database_costs(),
            ResourceType.CACHE.value: self.estimate_cache_costs()
        }
        
        # Calculate trends
        cost_analysis["trends"] = await self.analyze_cost_trends()
        
        # Project monthly costs
        days_in_month = 30
        current_day = datetime.now().day
        if current_day > 0:
            daily_rate = cost_analysis["current_month"]["total"] / current_day
            cost_analysis["projected_monthly"] = daily_rate * days_in_month
        
        return cost_analysis
    
    async def identify_optimization_opportunities(self) -> List[OptimizationOpportunity]:
        """Identify specific optimization opportunities"""
        opportunities = []
        
        # Get current utilization and costs
        utilization = await self.analyze_resource_utilization()
        costs = await self.analyze_current_costs()
        
        # 1. Under-utilized resources
        for service, metrics in utilization.get("service_metrics", {}).items():
            if metrics and metrics.get("cpu_percent", 100) < 30:
                opportunities.append(OptimizationOpportunity(
                    category="Resource Right-sizing",
                    service=service,
                    current_state=f"CPU utilization: {metrics.get('cpu_percent', 0)}%",
                    recommended_state="Reduce allocated CPU by 50%",
                    estimated_savings_percent=15,
                    estimated_monthly_savings=costs["current_month"]["total"] * 0.15 / 10,
                    implementation_effort="low",
                    priority=2,
                    risk_level="low"
                ))
        
        # 2. Expensive API calls without caching
        if costs["by_service"].get("script_generation", {}).get("total", 0) > 20:
            opportunities.append(OptimizationOpportunity(
                category="Caching Strategy",
                service="script_generation",
                current_state="No caching or short TTL",
                recommended_state="Implement 4-hour cache TTL",
                estimated_savings_percent=40,
                estimated_monthly_savings=costs["by_service"]["script_generation"]["total"] * 0.4,
                implementation_effort="low",
                priority=1,
                risk_level="low"
            ))
        
        # 3. Model tier optimization
        if costs["current_month"]["cost_per_video"] > 2.0:
            opportunities.append(OptimizationOpportunity(
                category="Model Selection",
                service="ai_services",
                current_state="Using premium models (GPT-4)",
                recommended_state="Use GPT-3.5 with fallback to GPT-4",
                estimated_savings_percent=60,
                estimated_monthly_savings=costs["current_month"]["total"] * 0.3,
                implementation_effort="medium",
                priority=1,
                risk_level="medium"
            ))
        
        # 4. Batch processing opportunities
        opportunities.append(OptimizationOpportunity(
            category="Batch Processing",
            service="video_generation",
            current_state="Processing videos individually",
            recommended_state="Batch process 5-10 videos together",
            estimated_savings_percent=25,
            estimated_monthly_savings=costs["current_month"]["total"] * 0.1,
            implementation_effort="medium",
            priority=3,
            risk_level="low"
        ))
        
        # 5. Storage optimization
        storage_gb_used = utilization["system_resources"]["disk"]["used_gb"]
        if storage_gb_used > 100:
            opportunities.append(OptimizationOpportunity(
                category="Storage Optimization",
                service="storage",
                current_state=f"Using {storage_gb_used:.1f} GB",
                recommended_state="Implement S3 archival for old videos",
                estimated_savings_percent=30,
                estimated_monthly_savings=self.estimate_storage_costs() * 0.3,
                implementation_effort="medium",
                priority=4,
                risk_level="low"
            ))
        
        # 6. Database optimization
        opportunities.append(OptimizationOpportunity(
            category="Database Optimization",
            service="postgres",
            current_state="No connection pooling",
            recommended_state="Implement PgBouncer connection pooling",
            estimated_savings_percent=20,
            estimated_monthly_savings=self.estimate_database_costs() * 0.2,
            implementation_effort="medium",
            priority=3,
            risk_level="medium"
        ))
        
        # 7. Auto-scaling implementation
        if not self.is_autoscaling_enabled():
            opportunities.append(OptimizationOpportunity(
                category="Auto-scaling",
                service="all_services",
                current_state="Fixed resource allocation",
                recommended_state="Implement horizontal auto-scaling",
                estimated_savings_percent=35,
                estimated_monthly_savings=costs["current_month"]["total"] * 0.2,
                implementation_effort="high",
                priority=2,
                risk_level="medium"
            ))
        
        # 8. Spot/Preemptible instances
        opportunities.append(OptimizationOpportunity(
            category="Compute Optimization",
            service="compute",
            current_state="Using on-demand instances",
            recommended_state="Use spot instances for batch workloads",
            estimated_savings_percent=70,
            estimated_monthly_savings=self.estimate_compute_costs() * 0.5,
            implementation_effort="high",
            priority=3,
            risk_level="high"
        ))
        
        # Sort by priority
        opportunities.sort(key=lambda x: (x.priority, -x.estimated_monthly_savings))
        
        return opportunities
    
    async def generate_autoscaling_recommendations(self) -> Dict[str, Any]:
        """Generate auto-scaling policy recommendations"""
        recommendations = {
            "policies": [],
            "estimated_savings": 0,
            "implementation_priority": []
        }
        
        # Define scaling policies for each service
        scaling_policies = [
            {
                "service": "backend",
                "metric": "cpu",
                "scale_up_threshold": 70,
                "scale_down_threshold": 30,
                "min_replicas": 1,
                "max_replicas": 5,
                "cooldown_seconds": 300
            },
            {
                "service": "celery_worker",
                "metric": "queue_depth",
                "scale_up_threshold": 10,
                "scale_down_threshold": 2,
                "min_replicas": 1,
                "max_replicas": 10,
                "cooldown_seconds": 180
            },
            {
                "service": "video_processor",
                "metric": "gpu_utilization",
                "scale_up_threshold": 80,
                "scale_down_threshold": 20,
                "min_replicas": 0,
                "max_replicas": 3,
                "cooldown_seconds": 600
            }
        ]
        
        for policy in scaling_policies:
            # Analyze current metrics for the service
            current_metrics = await self.get_service_metrics(policy["service"])
            
            recommendation = {
                "service": policy["service"],
                "policy": policy,
                "current_utilization": current_metrics.get(policy["metric"], 0) if current_metrics else 0,
                "recommended_action": self.determine_scaling_action(
                    current_metrics.get(policy["metric"], 0) if current_metrics else 0,
                    policy
                ),
                "estimated_monthly_savings": self.estimate_scaling_savings(policy["service"])
            }
            
            recommendations["policies"].append(recommendation)
            recommendations["estimated_savings"] += recommendation["estimated_monthly_savings"]
        
        # Prioritize implementation
        recommendations["implementation_priority"] = [
            {"service": "celery_worker", "reason": "High impact on video processing efficiency"},
            {"service": "backend", "reason": "Handles user traffic spikes"},
            {"service": "video_processor", "reason": "Most expensive resource (GPU)"}
        ]
        
        return recommendations
    
    async def create_implementation_plan(self) -> Dict[str, Any]:
        """Create detailed implementation plan for cost optimization"""
        opportunities = await self.identify_optimization_opportunities()
        
        plan = {
            "phases": [],
            "timeline_weeks": 0,
            "total_estimated_savings": 0,
            "risk_assessment": {}
        }
        
        # Phase 1: Quick wins (1 week)
        phase1_opportunities = [o for o in opportunities if o.implementation_effort == "low"]
        phase1_savings = sum(o.estimated_monthly_savings for o in phase1_opportunities)
        
        plan["phases"].append({
            "phase": 1,
            "name": "Quick Wins",
            "duration_weeks": 1,
            "opportunities": [
                {
                    "category": o.category,
                    "service": o.service,
                    "action": o.recommended_state,
                    "savings": o.estimated_monthly_savings
                }
                for o in phase1_opportunities
            ],
            "estimated_savings": phase1_savings,
            "tasks": [
                "Enable aggressive caching for AI services",
                "Implement model fallback strategy",
                "Adjust resource allocations for under-utilized services",
                "Configure cache TTL settings"
            ]
        })
        
        # Phase 2: Medium complexity (2 weeks)
        phase2_opportunities = [o for o in opportunities if o.implementation_effort == "medium"]
        phase2_savings = sum(o.estimated_monthly_savings for o in phase2_opportunities)
        
        plan["phases"].append({
            "phase": 2,
            "name": "Infrastructure Optimization",
            "duration_weeks": 2,
            "opportunities": [
                {
                    "category": o.category,
                    "service": o.service,
                    "action": o.recommended_state,
                    "savings": o.estimated_monthly_savings
                }
                for o in phase2_opportunities
            ],
            "estimated_savings": phase2_savings,
            "tasks": [
                "Implement batch processing for video generation",
                "Set up connection pooling for database",
                "Configure S3 for media archival",
                "Implement basic auto-scaling policies"
            ]
        })
        
        # Phase 3: Complex implementations (3 weeks)
        phase3_opportunities = [o for o in opportunities if o.implementation_effort == "high"]
        phase3_savings = sum(o.estimated_monthly_savings for o in phase3_opportunities)
        
        plan["phases"].append({
            "phase": 3,
            "name": "Advanced Optimization",
            "duration_weeks": 3,
            "opportunities": [
                {
                    "category": o.category,
                    "service": o.service,
                    "action": o.recommended_state,
                    "savings": o.estimated_monthly_savings
                }
                for o in phase3_opportunities
            ],
            "estimated_savings": phase3_savings,
            "tasks": [
                "Implement comprehensive auto-scaling",
                "Set up spot instance management",
                "Deploy local ML models for simple tasks",
                "Implement predictive scaling"
            ]
        })
        
        # Calculate totals
        plan["timeline_weeks"] = sum(p["duration_weeks"] for p in plan["phases"])
        plan["total_estimated_savings"] = sum(p["estimated_savings"] for p in plan["phases"])
        
        # Risk assessment
        plan["risk_assessment"] = {
            "low_risk_savings": sum(
                o.estimated_monthly_savings 
                for o in opportunities 
                if o.risk_level == "low"
            ),
            "medium_risk_savings": sum(
                o.estimated_monthly_savings 
                for o in opportunities 
                if o.risk_level == "medium"
            ),
            "high_risk_savings": sum(
                o.estimated_monthly_savings 
                for o in opportunities 
                if o.risk_level == "high"
            ),
            "recommendation": "Start with low-risk optimizations to validate approach"
        }
        
        return plan
    
    def calculate_summary_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary metrics for the analysis"""
        current_monthly_cost = analysis["cost_analysis"]["current_month"]["total"]
        total_savings = sum(
            o.estimated_monthly_savings 
            for o in analysis["optimization_opportunities"]
        )
        
        return {
            "current_monthly_cost": round(current_monthly_cost, 2),
            "optimized_monthly_cost": round(current_monthly_cost - total_savings, 2),
            "total_monthly_savings": round(total_savings, 2),
            "savings_percentage": round((total_savings / current_monthly_cost * 100) if current_monthly_cost > 0 else 0, 1),
            "target_reduction_achieved": total_savings / current_monthly_cost >= 0.2 if current_monthly_cost > 0 else False,
            "implementation_timeline_weeks": analysis["implementation_plan"]["timeline_weeks"],
            "quick_win_savings": round(
                sum(
                    o.estimated_monthly_savings 
                    for o in analysis["optimization_opportunities"]
                    if o.implementation_effort == "low"
                ), 2
            ),
            "roi_months": round(
                1 / (total_savings / current_monthly_cost) if total_savings > 0 else 0, 1
            )
        }
    
    async def get_service_metrics(self, service: str) -> Dict[str, Any]:
        """Get metrics for a specific service"""
        metrics = {
            "cpu_percent": 0,
            "memory_percent": 0,
            "request_rate": 0,
            "error_rate": 0,
            "response_time_ms": 0
        }
        
        # Simulate metrics (in production, would query Prometheus/monitoring system)
        import random
        
        if service == "backend":
            metrics = {
                "cpu_percent": random.uniform(20, 60),
                "memory_percent": random.uniform(30, 70),
                "request_rate": random.uniform(10, 100),
                "error_rate": random.uniform(0, 2),
                "response_time_ms": random.uniform(50, 200)
            }
        elif service == "celery_worker":
            metrics = {
                "cpu_percent": random.uniform(40, 80),
                "memory_percent": random.uniform(50, 90),
                "queue_depth": random.randint(0, 20),
                "tasks_per_minute": random.uniform(5, 50)
            }
        elif service == "postgres":
            metrics = {
                "cpu_percent": random.uniform(10, 40),
                "memory_percent": random.uniform(60, 85),
                "connections": random.randint(5, 50),
                "queries_per_second": random.uniform(10, 100)
            }
        
        return metrics
    
    def calculate_efficiency_score(self, utilization_data: Dict[str, Any]) -> float:
        """Calculate overall efficiency score (0-100)"""
        scores = []
        
        # CPU efficiency
        cpu_percent = utilization_data["system_resources"]["cpu"]["percent"]
        if 40 <= cpu_percent <= 70:
            scores.append(100)
        elif 30 <= cpu_percent < 40 or 70 < cpu_percent <= 80:
            scores.append(80)
        elif 20 <= cpu_percent < 30 or 80 < cpu_percent <= 90:
            scores.append(60)
        else:
            scores.append(40)
        
        # Memory efficiency
        memory_percent = utilization_data["system_resources"]["memory"]["percent"]
        if 50 <= memory_percent <= 75:
            scores.append(100)
        elif 40 <= memory_percent < 50 or 75 < memory_percent <= 85:
            scores.append(80)
        else:
            scores.append(60)
        
        # Container efficiency
        if utilization_data["container_resources"]:
            container_scores = []
            for container, stats in utilization_data["container_resources"].items():
                if 30 <= stats["cpu_percent"] <= 70:
                    container_scores.append(100)
                else:
                    container_scores.append(50)
            
            if container_scores:
                scores.append(sum(container_scores) / len(container_scores))
        
        return sum(scores) / len(scores) if scores else 0
    
    async def analyze_cost_trends(self) -> Dict[str, Any]:
        """Analyze cost trends over time"""
        # Simulate trend data (in production, would query historical data)
        return {
            "daily_trend": "increasing",
            "weekly_trend": "stable",
            "monthly_trend": "decreasing",
            "forecast_next_month": "stable",
            "anomalies_detected": False
        }
    
    def estimate_compute_costs(self) -> float:
        """Estimate compute costs"""
        # Rough estimation based on instance types
        # In production, would use actual cloud provider pricing
        return 150.0  # $150/month for compute
    
    def estimate_storage_costs(self) -> float:
        """Estimate storage costs"""
        # $0.023 per GB for S3 standard
        storage_gb = psutil.disk_usage('/').used / (1024**3)
        return storage_gb * 0.023
    
    def estimate_network_costs(self) -> float:
        """Estimate network/bandwidth costs"""
        # $0.09 per GB for data transfer
        network_gb = (psutil.net_io_counters().bytes_sent + 
                     psutil.net_io_counters().bytes_recv) / (1024**3)
        return network_gb * 0.09
    
    def estimate_database_costs(self) -> float:
        """Estimate database costs"""
        # Fixed cost for PostgreSQL instance
        return 50.0  # $50/month
    
    def estimate_cache_costs(self) -> float:
        """Estimate Redis cache costs"""
        # Fixed cost for Redis instance
        return 20.0  # $20/month
    
    def is_autoscaling_enabled(self) -> bool:
        """Check if auto-scaling is currently enabled"""
        # Check for auto-scaling configuration
        return os.path.exists("/etc/autoscaling/config.yaml")
    
    def determine_scaling_action(self, current_utilization: float, policy: Dict[str, Any]) -> str:
        """Determine scaling action based on current utilization"""
        if current_utilization > policy["scale_up_threshold"]:
            return "scale_up"
        elif current_utilization < policy["scale_down_threshold"]:
            return "scale_down"
        else:
            return "maintain"
    
    def estimate_scaling_savings(self, service: str) -> float:
        """Estimate savings from auto-scaling a service"""
        base_costs = {
            "backend": 30,
            "celery_worker": 50,
            "video_processor": 100
        }
        
        # Assume 30% savings from auto-scaling
        return base_costs.get(service, 20) * 0.3
    
    async def generate_cost_visualizations(self, analysis: Dict[str, Any]):
        """Generate cost visualization charts"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style("whitegrid")
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Cost by Service
            if analysis["cost_analysis"]["by_service"]:
                services = list(analysis["cost_analysis"]["by_service"].keys())
                costs = [data["total"] for data in analysis["cost_analysis"]["by_service"].values()]
                
                axes[0, 0].bar(services, costs, color='steelblue')
                axes[0, 0].set_title("Monthly Cost by Service")
                axes[0, 0].set_xlabel("Service")
                axes[0, 0].set_ylabel("Cost ($)")
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Resource Utilization
            if analysis["resource_utilization"]["system_resources"]:
                resources = ["CPU", "Memory", "Disk"]
                utilization = [
                    analysis["resource_utilization"]["system_resources"]["cpu"]["percent"],
                    analysis["resource_utilization"]["system_resources"]["memory"]["percent"],
                    analysis["resource_utilization"]["system_resources"]["disk"]["percent"]
                ]
                
                axes[0, 1].bar(resources, utilization, color=['green' if u < 70 else 'orange' if u < 90 else 'red' for u in utilization])
                axes[0, 1].set_title("Resource Utilization (%)")
                axes[0, 1].set_xlabel("Resource")
                axes[0, 1].set_ylabel("Utilization (%)")
                axes[0, 1].axhline(y=70, color='orange', linestyle='--', label='Warning')
                axes[0, 1].axhline(y=90, color='red', linestyle='--', label='Critical')
                axes[0, 1].legend()
            
            # 3. Optimization Opportunities
            if analysis["optimization_opportunities"]:
                categories = {}
                for opp in analysis["optimization_opportunities"][:8]:  # Top 8
                    if opp.category not in categories:
                        categories[opp.category] = 0
                    categories[opp.category] += opp.estimated_monthly_savings
                
                axes[1, 0].pie(
                    categories.values(),
                    labels=categories.keys(),
                    autopct='%1.1f%%',
                    startangle=90
                )
                axes[1, 0].set_title("Savings Potential by Category")
            
            # 4. Implementation Timeline
            if analysis["implementation_plan"]["phases"]:
                phases = [f"Phase {p['phase']}" for p in analysis["implementation_plan"]["phases"]]
                savings = [p["estimated_savings"] for p in analysis["implementation_plan"]["phases"]]
                durations = [p["duration_weeks"] for p in analysis["implementation_plan"]["phases"]]
                
                x = range(len(phases))
                width = 0.35
                
                axes[1, 1].bar([i - width/2 for i in x], savings, width, label='Savings ($)', color='green')
                axes[1, 1].bar([i + width/2 for i in x], durations, width, label='Duration (weeks)', color='blue')
                axes[1, 1].set_xlabel("Implementation Phase")
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(phases)
                axes[1, 1].legend()
                axes[1, 1].set_title("Implementation Plan")
            
            plt.tight_layout()
            
            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cost_optimization_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {filename}")
            
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping visualizations")
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
    
    async def generate_optimization_report(self, analysis: Dict[str, Any]):
        """Generate detailed optimization report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"cost_optimization_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# YTEmpire Cost Optimization Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            summary = analysis["summary"]
            f.write(f"- **Current Monthly Cost**: ${summary['current_monthly_cost']:,.2f}\n")
            f.write(f"- **Optimized Monthly Cost**: ${summary['optimized_monthly_cost']:,.2f}\n")
            f.write(f"- **Total Monthly Savings**: ${summary['total_monthly_savings']:,.2f} ({summary['savings_percentage']:.1f}%)\n")
            f.write(f"- **Target 20% Reduction**: {'✅ Achieved' if summary['target_reduction_achieved'] else '❌ Not Yet'}\n")
            f.write(f"- **Implementation Timeline**: {summary['implementation_timeline_weeks']} weeks\n")
            f.write(f"- **Quick Win Savings**: ${summary['quick_win_savings']:,.2f}\n")
            f.write(f"- **ROI Period**: {summary['roi_months']:.1f} months\n\n")
            
            # Resource Utilization
            f.write("## Resource Utilization\n\n")
            f.write("### System Resources\n")
            sys_res = analysis["resource_utilization"]["system_resources"]
            f.write(f"- CPU: {sys_res['cpu']['percent']:.1f}% ({sys_res['cpu']['cores']} cores)\n")
            f.write(f"- Memory: {sys_res['memory']['percent']:.1f}% ({sys_res['memory']['used_gb']:.1f}/{sys_res['memory']['total_gb']:.1f} GB)\n")
            f.write(f"- Disk: {sys_res['disk']['percent']:.1f}% ({sys_res['disk']['used_gb']:.1f}/{sys_res['disk']['total_gb']:.1f} GB)\n")
            f.write(f"- Efficiency Score: {analysis['resource_utilization']['efficiency_score']:.1f}/100\n\n")
            
            # Cost Analysis
            f.write("## Cost Analysis\n\n")
            f.write("### Current Month\n")
            current = analysis["cost_analysis"]["current_month"]
            f.write(f"- Total Cost: ${current['total']:,.2f}\n")
            f.write(f"- Videos Generated: {current['videos_generated']}\n")
            f.write(f"- Cost per Video: ${current['cost_per_video']:.2f}\n")
            f.write(f"- Within $3 Target: {'✅ Yes' if current['within_target'] else '❌ No'}\n\n")
            
            f.write("### Cost by Service\n")
            for service, data in analysis["cost_analysis"]["by_service"].items():
                f.write(f"- **{service}**: ${data['total']:.2f} ({data['percentage_of_total']:.1f}%)\n")
            f.write("\n")
            
            # Optimization Opportunities
            f.write("## Top Optimization Opportunities\n\n")
            for i, opp in enumerate(analysis["optimization_opportunities"][:10], 1):
                f.write(f"### {i}. {opp.category} - {opp.service}\n")
                f.write(f"- **Current State**: {opp.current_state}\n")
                f.write(f"- **Recommended**: {opp.recommended_state}\n")
                f.write(f"- **Estimated Savings**: ${opp.estimated_monthly_savings:.2f}/month ({opp.estimated_savings_percent:.0f}%)\n")
                f.write(f"- **Implementation Effort**: {opp.implementation_effort}\n")
                f.write(f"- **Priority**: {opp.priority}/5\n")
                f.write(f"- **Risk Level**: {opp.risk_level}\n\n")
            
            # Auto-scaling Recommendations
            f.write("## Auto-scaling Recommendations\n\n")
            auto_scaling = analysis["auto_scaling_recommendations"]
            f.write(f"Total Estimated Savings from Auto-scaling: ${auto_scaling['estimated_savings']:.2f}/month\n\n")
            
            for policy in auto_scaling["policies"]:
                f.write(f"### {policy['service']}\n")
                f.write(f"- Current Utilization: {policy['current_utilization']:.1f}%\n")
                f.write(f"- Recommended Action: {policy['recommended_action']}\n")
                f.write(f"- Scaling Policy:\n")
                f.write(f"  - Scale Up: >{policy['policy']['scale_up_threshold']}%\n")
                f.write(f"  - Scale Down: <{policy['policy']['scale_down_threshold']}%\n")
                f.write(f"  - Replicas: {policy['policy']['min_replicas']}-{policy['policy']['max_replicas']}\n\n")
            
            # Implementation Plan
            f.write("## Implementation Plan\n\n")
            for phase in analysis["implementation_plan"]["phases"]:
                f.write(f"### Phase {phase['phase']}: {phase['name']} ({phase['duration_weeks']} weeks)\n")
                f.write(f"**Estimated Savings**: ${phase['estimated_savings']:.2f}/month\n\n")
                f.write("**Tasks**:\n")
                for task in phase["tasks"]:
                    f.write(f"- {task}\n")
                f.write("\n")
            
            # Risk Assessment
            f.write("## Risk Assessment\n\n")
            risk = analysis["implementation_plan"]["risk_assessment"]
            f.write(f"- Low Risk Savings: ${risk['low_risk_savings']:.2f}\n")
            f.write(f"- Medium Risk Savings: ${risk['medium_risk_savings']:.2f}\n")
            f.write(f"- High Risk Savings: ${risk['high_risk_savings']:.2f}\n")
            f.write(f"- Recommendation: {risk['recommendation']}\n\n")
            
            # Next Steps
            f.write("## Next Steps\n\n")
            f.write("1. Review and approve optimization plan\n")
            f.write("2. Start with Phase 1 quick wins (1 week implementation)\n")
            f.write("3. Monitor metrics after each change\n")
            f.write("4. Adjust strategies based on actual results\n")
            f.write("5. Report progress weekly\n")
        
        logger.info(f"Report saved to {report_file}")
        
        # Also save as JSON for programmatic access
        json_file = f"cost_optimization_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"JSON report saved to {json_file}")

async def main():
    """Main execution function"""
    logger.info("Starting YTEmpire Cost Optimization Analysis")
    
    analyzer = CostOptimizationAnalyzer()
    results = await analyzer.perform_complete_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("COST OPTIMIZATION ANALYSIS COMPLETE")
    print("="*60)
    
    summary = results["summary"]
    print(f"Current Monthly Cost: ${summary['current_monthly_cost']:,.2f}")
    print(f"Potential Savings: ${summary['total_monthly_savings']:,.2f} ({summary['savings_percentage']:.1f}%)")
    print(f"Target 20% Reduction: {'✅ ACHIEVED' if summary['target_reduction_achieved'] else '❌ Not Yet'}")
    print(f"Implementation Time: {summary['implementation_timeline_weeks']} weeks")
    print(f"Quick Wins Available: ${summary['quick_win_savings']:,.2f}")
    
    print("\nTop 3 Recommendations:")
    for i, opp in enumerate(results["optimization_opportunities"][:3], 1):
        print(f"{i}. {opp.category}: Save ${opp.estimated_monthly_savings:.2f}/month")
    
    print("\nReports generated:")
    print("- cost_optimization_report_*.md (Detailed report)")
    print("- cost_optimization_report_*.json (Data export)")
    print("- cost_optimization_analysis_*.png (Visualizations)")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())