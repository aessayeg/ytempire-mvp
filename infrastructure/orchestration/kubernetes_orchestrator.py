#!/usr/bin/env python3
"""
Enhanced Container Orchestration System for YTEmpire
Implements comprehensive health checks, HPA, and resource quotas
"""

import os
import sys
import json
import yaml
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import docker
import subprocess
import aiohttp
import psutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

class ResourceType(Enum):
    """Kubernetes resource types"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    GPU = "gpu"

@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    type: str  # http, tcp, exec, grpc
    endpoint: Optional[str] = None
    port: Optional[int] = None
    command: Optional[List[str]] = None
    initial_delay_seconds: int = 30
    period_seconds: int = 10
    timeout_seconds: int = 5
    success_threshold: int = 1
    failure_threshold: int = 3
    
@dataclass
class ResourceQuota:
    """Resource quota specification"""
    namespace: str
    name: str
    hard_limits: Dict[str, str]
    scope_selector: Optional[Dict[str, str]] = None
    
@dataclass
class HPAConfig:
    """Horizontal Pod Autoscaler configuration"""
    name: str
    namespace: str
    target_deployment: str
    min_replicas: int
    max_replicas: int
    target_cpu_utilization: Optional[int] = None
    target_memory_utilization: Optional[int] = None
    custom_metrics: Optional[List[Dict[str, Any]]] = None
    behavior: Optional[Dict[str, Any]] = None

@dataclass
class ServiceHealth:
    """Service health status"""
    service: str
    status: HealthStatus
    ready_replicas: int
    total_replicas: int
    health_checks: List[Dict[str, Any]]
    last_check: datetime
    issues: List[str] = field(default_factory=list)

class KubernetesOrchestrator:
    """Enhanced Kubernetes orchestration management"""
    
    def __init__(self):
        self.config_dir = Path("infrastructure/kubernetes")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.namespace = "ytempire"
        
        # Check if running in Kubernetes
        self.in_kubernetes = os.path.exists("/var/run/secrets/kubernetes.io")
        
        # Docker client for local development
        self.docker_client = None
        if not self.in_kubernetes:
            try:
                self.docker_client = docker.from_env()
            except:
                logger.warning("Docker client not available")
    
    async def deploy_comprehensive_health_checks(self) -> Dict[str, Any]:
        """Deploy comprehensive health checks for all services"""
        logger.info("Deploying comprehensive health checks")
        
        health_checks = {
            "backend": [
                HealthCheck(
                    name="liveness",
                    type="http",
                    endpoint="/health/live",
                    port=8000,
                    initial_delay_seconds=30,
                    period_seconds=10
                ),
                HealthCheck(
                    name="readiness",
                    type="http",
                    endpoint="/health/ready",
                    port=8000,
                    initial_delay_seconds=10,
                    period_seconds=5
                ),
                HealthCheck(
                    name="startup",
                    type="http",
                    endpoint="/health/startup",
                    port=8000,
                    initial_delay_seconds=0,
                    period_seconds=10,
                    failure_threshold=30
                )
            ],
            "celery-worker": [
                HealthCheck(
                    name="liveness",
                    type="exec",
                    command=["celery", "-A", "app.core.celery_app", "inspect", "ping"],
                    period_seconds=30
                ),
                HealthCheck(
                    name="readiness",
                    type="exec",
                    command=["python", "-c", "import celery; print('ready')"],
                    period_seconds=10
                )
            ],
            "postgres": [
                HealthCheck(
                    name="liveness",
                    type="exec",
                    command=["pg_isready", "-U", "postgres"],
                    period_seconds=10
                ),
                HealthCheck(
                    name="readiness",
                    type="tcp",
                    port=5432,
                    period_seconds=5
                )
            ],
            "redis": [
                HealthCheck(
                    name="liveness",
                    type="exec",
                    command=["redis-cli", "ping"],
                    period_seconds=10
                ),
                HealthCheck(
                    name="readiness",
                    type="tcp",
                    port=6379,
                    period_seconds=5
                )
            ],
            "nginx": [
                HealthCheck(
                    name="liveness",
                    type="http",
                    endpoint="/nginx-health",
                    port=80,
                    period_seconds=10
                ),
                HealthCheck(
                    name="readiness",
                    type="tcp",
                    port=80,
                    period_seconds=5
                )
            ]
        }
        
        deployment_results = {}
        
        for service, checks in health_checks.items():
            # Generate Kubernetes deployment with health checks
            deployment = self._generate_deployment_with_health_checks(service, checks)
            
            # Save deployment configuration
            deployment_file = self.config_dir / f"{service}-deployment.yaml"
            with open(deployment_file, 'w') as f:
                yaml.dump(deployment, f, default_flow_style=False)
            
            # Apply deployment (if in Kubernetes)
            if self.in_kubernetes:
                result = await self._apply_kubernetes_resource(deployment_file)
                deployment_results[service] = result
            else:
                # For local development, configure Docker health checks
                deployment_results[service] = await self._configure_docker_health_checks(
                    service, checks
                )
        
        return {
            "status": "success",
            "health_checks_deployed": len(health_checks),
            "services": list(health_checks.keys()),
            "deployment_results": deployment_results
        }
    
    async def setup_horizontal_pod_autoscaling(self) -> Dict[str, Any]:
        """Setup Horizontal Pod Autoscaling for all services"""
        logger.info("Setting up Horizontal Pod Autoscaling")
        
        hpa_configs = [
            HPAConfig(
                name="backend-hpa",
                namespace=self.namespace,
                target_deployment="backend",
                min_replicas=2,
                max_replicas=10,
                target_cpu_utilization=70,
                target_memory_utilization=80,
                behavior={
                    "scaleUp": {
                        "stabilizationWindowSeconds": 60,
                        "selectPolicy": "Max",
                        "policies": [
                            {"type": "Percent", "value": 100, "periodSeconds": 60},
                            {"type": "Pods", "value": 2, "periodSeconds": 60}
                        ]
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "selectPolicy": "Min",
                        "policies": [
                            {"type": "Percent", "value": 50, "periodSeconds": 180},
                            {"type": "Pods", "value": 1, "periodSeconds": 180}
                        ]
                    }
                }
            ),
            HPAConfig(
                name="celery-worker-hpa",
                namespace=self.namespace,
                target_deployment="celery-worker",
                min_replicas=1,
                max_replicas=20,
                target_cpu_utilization=60,
                custom_metrics=[
                    {
                        "type": "External",
                        "external": {
                            "metric": {
                                "name": "celery_queue_length",
                                "selector": {"matchLabels": {"queue": "video_processing"}}
                            },
                            "target": {
                                "type": "Value",
                                "value": "10"
                            }
                        }
                    }
                ]
            ),
            HPAConfig(
                name="video-processor-hpa",
                namespace=self.namespace,
                target_deployment="video-processor",
                min_replicas=0,
                max_replicas=5,
                custom_metrics=[
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "nvidia.com/gpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ]
            ),
            HPAConfig(
                name="frontend-hpa",
                namespace=self.namespace,
                target_deployment="frontend",
                min_replicas=1,
                max_replicas=5,
                target_cpu_utilization=75,
                target_memory_utilization=75
            )
        ]
        
        hpa_results = {}
        
        for hpa in hpa_configs:
            # Generate HPA manifest
            hpa_manifest = self._generate_hpa_manifest(hpa)
            
            # Save HPA configuration
            hpa_file = self.config_dir / f"{hpa.name}.yaml"
            with open(hpa_file, 'w') as f:
                yaml.dump(hpa_manifest, f, default_flow_style=False)
            
            # Apply HPA (if in Kubernetes)
            if self.in_kubernetes:
                result = await self._apply_kubernetes_resource(hpa_file)
                hpa_results[hpa.name] = result
            else:
                # For local development, simulate HPA
                hpa_results[hpa.name] = {
                    "status": "simulated",
                    "config": asdict(hpa)
                }
        
        # Setup metrics server if not present
        metrics_server_result = await self._ensure_metrics_server()
        
        return {
            "status": "success",
            "hpa_configured": len(hpa_configs),
            "hpa_results": hpa_results,
            "metrics_server": metrics_server_result
        }
    
    async def configure_resource_quotas(self) -> Dict[str, Any]:
        """Configure resource quotas for namespaces"""
        logger.info("Configuring resource quotas")
        
        quotas = [
            ResourceQuota(
                namespace=self.namespace,
                name="compute-quota",
                hard_limits={
                    "requests.cpu": "100",
                    "requests.memory": "200Gi",
                    "limits.cpu": "200",
                    "limits.memory": "400Gi",
                    "persistentvolumeclaims": "10",
                    "requests.storage": "1Ti"
                }
            ),
            ResourceQuota(
                namespace=self.namespace,
                name="object-quota",
                hard_limits={
                    "pods": "100",
                    "services": "20",
                    "configmaps": "50",
                    "secrets": "50",
                    "services.loadbalancers": "2",
                    "services.nodeports": "5"
                }
            ),
            ResourceQuota(
                namespace=f"{self.namespace}-dev",
                name="dev-quota",
                hard_limits={
                    "requests.cpu": "20",
                    "requests.memory": "40Gi",
                    "limits.cpu": "40",
                    "limits.memory": "80Gi",
                    "pods": "20"
                }
            ),
            ResourceQuota(
                namespace=f"{self.namespace}-staging",
                name="staging-quota",
                hard_limits={
                    "requests.cpu": "50",
                    "requests.memory": "100Gi",
                    "limits.cpu": "100",
                    "limits.memory": "200Gi",
                    "pods": "50"
                }
            )
        ]
        
        # GPU quota if available
        if await self._check_gpu_availability():
            quotas.append(
                ResourceQuota(
                    namespace=self.namespace,
                    name="gpu-quota",
                    hard_limits={
                        "requests.nvidia.com/gpu": "4",
                        "limits.nvidia.com/gpu": "4"
                    }
                )
            )
        
        quota_results = {}
        
        for quota in quotas:
            # Generate quota manifest
            quota_manifest = self._generate_quota_manifest(quota)
            
            # Save quota configuration
            quota_file = self.config_dir / f"quota-{quota.name}.yaml"
            with open(quota_file, 'w') as f:
                yaml.dump(quota_manifest, f, default_flow_style=False)
            
            # Apply quota (if in Kubernetes)
            if self.in_kubernetes:
                # Create namespace if it doesn't exist
                await self._ensure_namespace(quota.namespace)
                
                result = await self._apply_kubernetes_resource(quota_file)
                quota_results[quota.name] = result
            else:
                quota_results[quota.name] = {
                    "status": "configured",
                    "namespace": quota.namespace,
                    "limits": quota.hard_limits
                }
        
        # Configure limit ranges
        limit_ranges = await self._configure_limit_ranges()
        
        return {
            "status": "success",
            "quotas_configured": len(quotas),
            "quota_results": quota_results,
            "limit_ranges": limit_ranges
        }
    
    async def setup_pod_disruption_budgets(self) -> Dict[str, Any]:
        """Setup Pod Disruption Budgets for high availability"""
        logger.info("Setting up Pod Disruption Budgets")
        
        pdbs = [
            {
                "name": "backend-pdb",
                "namespace": self.namespace,
                "selector": {"matchLabels": {"app": "backend"}},
                "minAvailable": 1
            },
            {
                "name": "postgres-pdb",
                "namespace": self.namespace,
                "selector": {"matchLabels": {"app": "postgres"}},
                "maxUnavailable": 0  # No disruption for database
            },
            {
                "name": "redis-pdb",
                "namespace": self.namespace,
                "selector": {"matchLabels": {"app": "redis"}},
                "minAvailable": 1
            },
            {
                "name": "celery-worker-pdb",
                "namespace": self.namespace,
                "selector": {"matchLabels": {"app": "celery-worker"}},
                "minAvailable": "30%"
            }
        ]
        
        pdb_results = {}
        
        for pdb in pdbs:
            # Generate PDB manifest
            pdb_manifest = {
                "apiVersion": "policy/v1",
                "kind": "PodDisruptionBudget",
                "metadata": {
                    "name": pdb["name"],
                    "namespace": pdb["namespace"]
                },
                "spec": {
                    "selector": pdb["selector"]
                }
            }
            
            if "minAvailable" in pdb:
                pdb_manifest["spec"]["minAvailable"] = pdb["minAvailable"]
            if "maxUnavailable" in pdb:
                pdb_manifest["spec"]["maxUnavailable"] = pdb["maxUnavailable"]
            
            # Save PDB configuration
            pdb_file = self.config_dir / f"{pdb['name']}.yaml"
            with open(pdb_file, 'w') as f:
                yaml.dump(pdb_manifest, f, default_flow_style=False)
            
            if self.in_kubernetes:
                result = await self._apply_kubernetes_resource(pdb_file)
                pdb_results[pdb["name"]] = result
            else:
                pdb_results[pdb["name"]] = {"status": "configured"}
        
        return {
            "status": "success",
            "pdbs_configured": len(pdbs),
            "pdb_results": pdb_results
        }
    
    async def implement_network_policies(self) -> Dict[str, Any]:
        """Implement network policies for security"""
        logger.info("Implementing network policies")
        
        network_policies = [
            {
                "name": "backend-network-policy",
                "namespace": self.namespace,
                "podSelector": {"matchLabels": {"app": "backend"}},
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {"podSelector": {"matchLabels": {"app": "nginx"}}},
                            {"podSelector": {"matchLabels": {"app": "frontend"}}}
                        ],
                        "ports": [{"protocol": "TCP", "port": 8000}]
                    }
                ],
                "egress": [
                    {
                        "to": [
                            {"podSelector": {"matchLabels": {"app": "postgres"}}},
                            {"podSelector": {"matchLabels": {"app": "redis"}}}
                        ]
                    },
                    {
                        "ports": [{"protocol": "TCP", "port": 443}],
                        "to": [{"namespaceSelector": {}}]  # Allow HTTPS to external
                    }
                ]
            },
            {
                "name": "database-network-policy",
                "namespace": self.namespace,
                "podSelector": {"matchLabels": {"app": "postgres"}},
                "policyTypes": ["Ingress"],
                "ingress": [
                    {
                        "from": [
                            {"podSelector": {"matchLabels": {"app": "backend"}}},
                            {"podSelector": {"matchLabels": {"app": "celery-worker"}}}
                        ],
                        "ports": [{"protocol": "TCP", "port": 5432}]
                    }
                ]
            },
            {
                "name": "deny-all-default",
                "namespace": self.namespace,
                "podSelector": {},
                "policyTypes": ["Ingress", "Egress"]
            }
        ]
        
        policy_results = {}
        
        for policy in network_policies:
            # Generate NetworkPolicy manifest
            policy_manifest = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {
                    "name": policy["name"],
                    "namespace": policy["namespace"]
                },
                "spec": policy
            }
            del policy_manifest["spec"]["name"]
            del policy_manifest["spec"]["namespace"]
            
            # Save policy configuration
            policy_file = self.config_dir / f"netpol-{policy['name']}.yaml"
            with open(policy_file, 'w') as f:
                yaml.dump(policy_manifest, f, default_flow_style=False)
            
            if self.in_kubernetes:
                result = await self._apply_kubernetes_resource(policy_file)
                policy_results[policy["name"]] = result
            else:
                policy_results[policy["name"]] = {"status": "configured"}
        
        return {
            "status": "success",
            "policies_configured": len(network_policies),
            "policy_results": policy_results
        }
    
    async def monitor_service_health(self) -> Dict[str, ServiceHealth]:
        """Monitor health of all services"""
        logger.info("Monitoring service health")
        
        services = ["backend", "celery-worker", "postgres", "redis", "nginx", "frontend"]
        health_status = {}
        
        for service in services:
            if self.in_kubernetes:
                health = await self._check_kubernetes_service_health(service)
            else:
                health = await self._check_docker_service_health(service)
            
            health_status[service] = health
        
        # Generate health report
        healthy_services = sum(1 for h in health_status.values() if h.status == HealthStatus.HEALTHY)
        degraded_services = sum(1 for h in health_status.values() if h.status == HealthStatus.DEGRADED)
        unhealthy_services = sum(1 for h in health_status.values() if h.status == HealthStatus.UNHEALTHY)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_services": len(services),
                "healthy": healthy_services,
                "degraded": degraded_services,
                "unhealthy": unhealthy_services
            },
            "services": {
                service: {
                    "status": health.status.value,
                    "ready_replicas": health.ready_replicas,
                    "total_replicas": health.total_replicas,
                    "issues": health.issues
                }
                for service, health in health_status.items()
            }
        }
    
    async def perform_rolling_update(
        self,
        deployment: str,
        new_image: str,
        strategy: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Perform rolling update of a deployment"""
        logger.info(f"Performing rolling update for {deployment}")
        
        if strategy is None:
            strategy = {
                "type": "RollingUpdate",
                "rollingUpdate": {
                    "maxSurge": "25%",
                    "maxUnavailable": "25%"
                }
            }
        
        if self.in_kubernetes:
            # Update Kubernetes deployment
            cmd = [
                "kubectl", "set", "image",
                f"deployment/{deployment}",
                f"{deployment}={new_image}",
                "-n", self.namespace
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Monitor rollout status
                rollout_status = await self._monitor_rollout(deployment)
                
                return {
                    "status": "success",
                    "deployment": deployment,
                    "new_image": new_image,
                    "rollout_status": rollout_status
                }
            else:
                return {
                    "status": "failed",
                    "error": result.stderr
                }
        else:
            # For Docker, simulate rolling update
            return await self._docker_rolling_update(deployment, new_image)
    
    # Helper methods
    def _generate_deployment_with_health_checks(
        self,
        service: str,
        health_checks: List[HealthCheck]
    ) -> Dict[str, Any]:
        """Generate Kubernetes deployment with health checks"""
        
        container_spec = {
            "name": service,
            "image": f"ytempire/{service}:latest",
            "ports": [{"containerPort": self._get_service_port(service)}]
        }
        
        # Add health check probes
        for check in health_checks:
            probe = self._health_check_to_probe(check)
            
            if check.name == "liveness":
                container_spec["livenessProbe"] = probe
            elif check.name == "readiness":
                container_spec["readinessProbe"] = probe
            elif check.name == "startup":
                container_spec["startupProbe"] = probe
        
        # Add resource limits
        container_spec["resources"] = {
            "requests": self._get_resource_requests(service),
            "limits": self._get_resource_limits(service)
        }
        
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": service,
                "namespace": self.namespace,
                "labels": {"app": service}
            },
            "spec": {
                "replicas": self._get_initial_replicas(service),
                "selector": {"matchLabels": {"app": service}},
                "template": {
                    "metadata": {"labels": {"app": service}},
                    "spec": {
                        "containers": [container_spec]
                    }
                }
            }
        }
        
        return deployment
    
    def _health_check_to_probe(self, check: HealthCheck) -> Dict[str, Any]:
        """Convert HealthCheck to Kubernetes probe"""
        probe = {
            "initialDelaySeconds": check.initial_delay_seconds,
            "periodSeconds": check.period_seconds,
            "timeoutSeconds": check.timeout_seconds,
            "successThreshold": check.success_threshold,
            "failureThreshold": check.failure_threshold
        }
        
        if check.type == "http":
            probe["httpGet"] = {
                "path": check.endpoint,
                "port": check.port
            }
        elif check.type == "tcp":
            probe["tcpSocket"] = {
                "port": check.port
            }
        elif check.type == "exec":
            probe["exec"] = {
                "command": check.command
            }
        elif check.type == "grpc":
            probe["grpc"] = {
                "port": check.port
            }
        
        return probe
    
    def _generate_hpa_manifest(self, hpa: HPAConfig) -> Dict[str, Any]:
        """Generate HPA manifest"""
        manifest = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": hpa.name,
                "namespace": hpa.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": hpa.target_deployment
                },
                "minReplicas": hpa.min_replicas,
                "maxReplicas": hpa.max_replicas,
                "metrics": []
            }
        }
        
        # Add CPU metric
        if hpa.target_cpu_utilization:
            manifest["spec"]["metrics"].append({
                "type": "Resource",
                "resource": {
                    "name": "cpu",
                    "target": {
                        "type": "Utilization",
                        "averageUtilization": hpa.target_cpu_utilization
                    }
                }
            })
        
        # Add memory metric
        if hpa.target_memory_utilization:
            manifest["spec"]["metrics"].append({
                "type": "Resource",
                "resource": {
                    "name": "memory",
                    "target": {
                        "type": "Utilization",
                        "averageUtilization": hpa.target_memory_utilization
                    }
                }
            })
        
        # Add custom metrics
        if hpa.custom_metrics:
            manifest["spec"]["metrics"].extend(hpa.custom_metrics)
        
        # Add behavior if specified
        if hpa.behavior:
            manifest["spec"]["behavior"] = hpa.behavior
        
        return manifest
    
    def _generate_quota_manifest(self, quota: ResourceQuota) -> Dict[str, Any]:
        """Generate ResourceQuota manifest"""
        manifest = {
            "apiVersion": "v1",
            "kind": "ResourceQuota",
            "metadata": {
                "name": quota.name,
                "namespace": quota.namespace
            },
            "spec": {
                "hard": quota.hard_limits
            }
        }
        
        if quota.scope_selector:
            manifest["spec"]["scopeSelector"] = quota.scope_selector
        
        return manifest
    
    async def _configure_limit_ranges(self) -> Dict[str, Any]:
        """Configure limit ranges for resource control"""
        limit_ranges = [
            {
                "name": "default-limits",
                "namespace": self.namespace,
                "limits": [
                    {
                        "type": "Pod",
                        "max": {"cpu": "4", "memory": "8Gi"},
                        "min": {"cpu": "100m", "memory": "128Mi"}
                    },
                    {
                        "type": "Container",
                        "default": {"cpu": "500m", "memory": "512Mi"},
                        "defaultRequest": {"cpu": "200m", "memory": "256Mi"},
                        "max": {"cpu": "2", "memory": "4Gi"},
                        "min": {"cpu": "100m", "memory": "128Mi"}
                    },
                    {
                        "type": "PersistentVolumeClaim",
                        "min": {"storage": "1Gi"},
                        "max": {"storage": "100Gi"}
                    }
                ]
            }
        ]
        
        results = {}
        
        for lr in limit_ranges:
            manifest = {
                "apiVersion": "v1",
                "kind": "LimitRange",
                "metadata": {
                    "name": lr["name"],
                    "namespace": lr["namespace"]
                },
                "spec": {
                    "limits": lr["limits"]
                }
            }
            
            lr_file = self.config_dir / f"limitrange-{lr['name']}.yaml"
            with open(lr_file, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False)
            
            if self.in_kubernetes:
                result = await self._apply_kubernetes_resource(lr_file)
                results[lr["name"]] = result
            else:
                results[lr["name"]] = {"status": "configured"}
        
        return results
    
    async def _apply_kubernetes_resource(self, resource_file: Path) -> Dict[str, Any]:
        """Apply Kubernetes resource"""
        try:
            cmd = ["kubectl", "apply", "-f", str(resource_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {"status": "applied", "output": result.stdout}
            else:
                return {"status": "failed", "error": result.stderr}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _ensure_namespace(self, namespace: str):
        """Ensure namespace exists"""
        cmd = ["kubectl", "create", "namespace", namespace, "--dry-run=client", "-o", "yaml"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            cmd = ["kubectl", "apply", "-f", "-"]
            subprocess.run(cmd, input=result.stdout, text=True)
    
    async def _ensure_metrics_server(self) -> Dict[str, Any]:
        """Ensure metrics server is installed"""
        # Check if metrics server is running
        cmd = ["kubectl", "get", "deployment", "metrics-server", "-n", "kube-system"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # Install metrics server
            metrics_server_url = "https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml"
            cmd = ["kubectl", "apply", "-f", metrics_server_url]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {"status": "installed"}
            else:
                return {"status": "failed", "error": result.stderr}
        else:
            return {"status": "already_installed"}
    
    async def _check_gpu_availability(self) -> bool:
        """Check if GPU resources are available"""
        if self.in_kubernetes:
            cmd = ["kubectl", "get", "nodes", "-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                nodes = json.loads(result.stdout)
                for node in nodes.get("items", []):
                    capacity = node.get("status", {}).get("capacity", {})
                    if "nvidia.com/gpu" in capacity:
                        return True
        else:
            # Check local GPU
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True)
                return result.returncode == 0
            except:
                pass
        
        return False
    
    async def _configure_docker_health_checks(
        self,
        service: str,
        checks: List[HealthCheck]
    ) -> Dict[str, Any]:
        """Configure Docker health checks for local development"""
        if not self.docker_client:
            return {"status": "docker_not_available"}
        
        try:
            containers = self.docker_client.containers.list(
                filters={"label": f"com.docker.compose.service={service}"}
            )
            
            for container in containers:
                # Docker supports only one health check
                primary_check = next((c for c in checks if c.name == "liveness"), checks[0])
                
                if primary_check.type == "http":
                    healthcheck = {
                        "test": ["CMD", "curl", "-f", f"http://localhost:{primary_check.port}{primary_check.endpoint}"],
                        "interval": primary_check.period_seconds * 1000000000,  # nanoseconds
                        "timeout": primary_check.timeout_seconds * 1000000000,
                        "retries": primary_check.failure_threshold,
                        "start_period": primary_check.initial_delay_seconds * 1000000000
                    }
                elif primary_check.type == "exec":
                    healthcheck = {
                        "test": ["CMD"] + primary_check.command,
                        "interval": primary_check.period_seconds * 1000000000,
                        "timeout": primary_check.timeout_seconds * 1000000000,
                        "retries": primary_check.failure_threshold,
                        "start_period": primary_check.initial_delay_seconds * 1000000000
                    }
                else:
                    continue
                
                # Update container with health check
                container.update(healthcheck=healthcheck)
            
            return {"status": "configured", "containers": len(containers)}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_kubernetes_service_health(self, service: str) -> ServiceHealth:
        """Check health of Kubernetes service"""
        try:
            cmd = ["kubectl", "get", "deployment", service, "-n", self.namespace, "-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                deployment = json.loads(result.stdout)
                status = deployment.get("status", {})
                
                ready_replicas = status.get("readyReplicas", 0)
                total_replicas = status.get("replicas", 0)
                
                if ready_replicas == total_replicas and ready_replicas > 0:
                    health_status = HealthStatus.HEALTHY
                elif ready_replicas > 0:
                    health_status = HealthStatus.DEGRADED
                else:
                    health_status = HealthStatus.UNHEALTHY
                
                issues = []
                conditions = status.get("conditions", [])
                for condition in conditions:
                    if condition.get("status") != "True":
                        issues.append(f"{condition.get('type')}: {condition.get('message')}")
                
                return ServiceHealth(
                    service=service,
                    status=health_status,
                    ready_replicas=ready_replicas,
                    total_replicas=total_replicas,
                    health_checks=[],
                    last_check=datetime.now(),
                    issues=issues
                )
            else:
                return ServiceHealth(
                    service=service,
                    status=HealthStatus.UNKNOWN,
                    ready_replicas=0,
                    total_replicas=0,
                    health_checks=[],
                    last_check=datetime.now(),
                    issues=["Failed to get deployment status"]
                )
                
        except Exception as e:
            return ServiceHealth(
                service=service,
                status=HealthStatus.UNKNOWN,
                ready_replicas=0,
                total_replicas=0,
                health_checks=[],
                last_check=datetime.now(),
                issues=[str(e)]
            )
    
    async def _check_docker_service_health(self, service: str) -> ServiceHealth:
        """Check health of Docker service"""
        if not self.docker_client:
            return ServiceHealth(
                service=service,
                status=HealthStatus.UNKNOWN,
                ready_replicas=0,
                total_replicas=0,
                health_checks=[],
                last_check=datetime.now(),
                issues=["Docker not available"]
            )
        
        try:
            containers = self.docker_client.containers.list(
                filters={"label": f"com.docker.compose.service={service}"},
                all=True
            )
            
            running_containers = [c for c in containers if c.status == "running"]
            healthy_containers = []
            
            for container in running_containers:
                health = container.attrs.get("State", {}).get("Health", {})
                if health.get("Status") == "healthy":
                    healthy_containers.append(container)
            
            if len(healthy_containers) == len(containers) and len(containers) > 0:
                health_status = HealthStatus.HEALTHY
            elif len(healthy_containers) > 0:
                health_status = HealthStatus.DEGRADED
            elif len(running_containers) > 0:
                health_status = HealthStatus.DEGRADED
            else:
                health_status = HealthStatus.UNHEALTHY
            
            issues = []
            for container in containers:
                if container.status != "running":
                    issues.append(f"Container {container.short_id} is {container.status}")
                elif container not in healthy_containers:
                    health = container.attrs.get("State", {}).get("Health", {})
                    if health:
                        issues.append(f"Container {container.short_id} health: {health.get('Status')}")
            
            return ServiceHealth(
                service=service,
                status=health_status,
                ready_replicas=len(healthy_containers),
                total_replicas=len(containers),
                health_checks=[],
                last_check=datetime.now(),
                issues=issues
            )
            
        except Exception as e:
            return ServiceHealth(
                service=service,
                status=HealthStatus.UNKNOWN,
                ready_replicas=0,
                total_replicas=0,
                health_checks=[],
                last_check=datetime.now(),
                issues=[str(e)]
            )
    
    async def _monitor_rollout(self, deployment: str) -> Dict[str, Any]:
        """Monitor rollout status"""
        max_wait = 300  # 5 minutes
        check_interval = 10
        elapsed = 0
        
        while elapsed < max_wait:
            cmd = ["kubectl", "rollout", "status", f"deployment/{deployment}", "-n", self.namespace]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    "status": "completed",
                    "message": result.stdout
                }
            
            await asyncio.sleep(check_interval)
            elapsed += check_interval
        
        return {
            "status": "timeout",
            "message": f"Rollout did not complete within {max_wait} seconds"
        }
    
    async def _docker_rolling_update(self, service: str, new_image: str) -> Dict[str, Any]:
        """Simulate rolling update for Docker"""
        if not self.docker_client:
            return {"status": "docker_not_available"}
        
        try:
            # Pull new image
            self.docker_client.images.pull(new_image)
            
            # Get existing containers
            containers = self.docker_client.containers.list(
                filters={"label": f"com.docker.compose.service={service}"}
            )
            
            updated_containers = []
            
            for container in containers:
                # Create new container with new image
                config = container.attrs["Config"]
                config["Image"] = new_image
                
                # Stop old container
                container.stop()
                
                # Start new container
                new_container = self.docker_client.containers.run(
                    image=new_image,
                    detach=True,
                    **config
                )
                
                updated_containers.append(new_container.short_id)
                
                # Wait for health check
                await asyncio.sleep(5)
            
            return {
                "status": "success",
                "updated_containers": updated_containers
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _get_service_port(self, service: str) -> int:
        """Get default port for service"""
        ports = {
            "backend": 8000,
            "celery-worker": 5555,
            "postgres": 5432,
            "redis": 6379,
            "nginx": 80,
            "frontend": 3000
        }
        return ports.get(service, 8080)
    
    def _get_initial_replicas(self, service: str) -> int:
        """Get initial replica count for service"""
        replicas = {
            "backend": 2,
            "celery-worker": 2,
            "postgres": 1,
            "redis": 1,
            "nginx": 2,
            "frontend": 1
        }
        return replicas.get(service, 1)
    
    def _get_resource_requests(self, service: str) -> Dict[str, str]:
        """Get resource requests for service"""
        requests = {
            "backend": {"cpu": "200m", "memory": "256Mi"},
            "celery-worker": {"cpu": "500m", "memory": "512Mi"},
            "postgres": {"cpu": "500m", "memory": "1Gi"},
            "redis": {"cpu": "100m", "memory": "128Mi"},
            "nginx": {"cpu": "100m", "memory": "128Mi"},
            "frontend": {"cpu": "100m", "memory": "256Mi"}
        }
        return requests.get(service, {"cpu": "100m", "memory": "128Mi"})
    
    def _get_resource_limits(self, service: str) -> Dict[str, str]:
        """Get resource limits for service"""
        limits = {
            "backend": {"cpu": "1", "memory": "1Gi"},
            "celery-worker": {"cpu": "2", "memory": "2Gi"},
            "postgres": {"cpu": "2", "memory": "4Gi"},
            "redis": {"cpu": "500m", "memory": "512Mi"},
            "nginx": {"cpu": "500m", "memory": "512Mi"},
            "frontend": {"cpu": "500m", "memory": "1Gi"}
        }
        return limits.get(service, {"cpu": "500m", "memory": "512Mi"})

async def main():
    """Main execution function"""
    logger.info("Starting Kubernetes Orchestration Enhancement")
    
    orchestrator = KubernetesOrchestrator()
    
    # Deploy comprehensive health checks
    logger.info("Deploying health checks...")
    health_check_results = await orchestrator.deploy_comprehensive_health_checks()
    logger.info(f"Health checks deployed: {health_check_results['health_checks_deployed']} services")
    
    # Setup HPA
    logger.info("Setting up Horizontal Pod Autoscaling...")
    hpa_results = await orchestrator.setup_horizontal_pod_autoscaling()
    logger.info(f"HPA configured: {hpa_results['hpa_configured']} autoscalers")
    
    # Configure resource quotas
    logger.info("Configuring resource quotas...")
    quota_results = await orchestrator.configure_resource_quotas()
    logger.info(f"Quotas configured: {quota_results['quotas_configured']}")
    
    # Setup Pod Disruption Budgets
    logger.info("Setting up Pod Disruption Budgets...")
    pdb_results = await orchestrator.setup_pod_disruption_budgets()
    logger.info(f"PDBs configured: {pdb_results['pdbs_configured']}")
    
    # Implement network policies
    logger.info("Implementing network policies...")
    network_results = await orchestrator.implement_network_policies()
    logger.info(f"Network policies configured: {network_results['policies_configured']}")
    
    # Monitor service health
    logger.info("Monitoring service health...")
    health_status = await orchestrator.monitor_service_health()
    
    print("\n" + "="*60)
    print("CONTAINER ORCHESTRATION ENHANCEMENT COMPLETE")
    print("="*60)
    print(f"Health Checks: {health_check_results['health_checks_deployed']} services")
    print(f"HPA Configured: {hpa_results['hpa_configured']} autoscalers")
    print(f"Resource Quotas: {quota_results['quotas_configured']}")
    print(f"Pod Disruption Budgets: {pdb_results['pdbs_configured']}")
    print(f"Network Policies: {network_results['policies_configured']}")
    print("\nService Health Status:")
    print(f"  Healthy: {health_status['summary']['healthy']}")
    print(f"  Degraded: {health_status['summary']['degraded']}")
    print(f"  Unhealthy: {health_status['summary']['unhealthy']}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())