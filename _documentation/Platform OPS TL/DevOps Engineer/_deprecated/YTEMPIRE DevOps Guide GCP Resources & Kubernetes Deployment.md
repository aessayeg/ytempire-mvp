# YTEMPIRE DevOps Guide: GCP Resources & Kubernetes Deployment
**Version 1.0 | January 2025**  
**Owner: DevOps Engineering Team**  
**Approved By: Platform Operations Lead**

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [GCP Resource Specifications](#gcp-resource-specifications)
3. [Kubernetes Deployment Architecture](#kubernetes-deployment-architecture)
4. [Deployment Manifests](#deployment-manifests)
5. [Service Configurations](#service-configurations)
6. [Storage and Persistence](#storage-and-persistence)
7. [Monitoring and Observability](#monitoring-and-observability)

---

## Executive Summary

This document provides comprehensive technical specifications for YTEMPIRE's cloud infrastructure on Google Cloud Platform (GCP) and Kubernetes deployment configurations. As a DevOps Engineer, you'll use this guide to provision, deploy, and manage our production infrastructure supporting 100M+ daily operations.

### Key Infrastructure Goals
- **Scalability**: Support 100-500 YouTube channels initially, scaling to 5000+
- **Reliability**: Achieve 99.99% uptime SLA
- **Performance**: Sub-200ms API response times
- **Cost Efficiency**: Optimize for <$3 per video processing cost
- **Security**: Zero-trust architecture with comprehensive monitoring

---

## GCP Resource Specifications

### 1. Project Structure

```yaml
# GCP Project Organization
ytempire-organization/
├── ytempire-production/      # Production environment
│   ├── gke-clusters/
│   ├── cloud-sql/
│   ├── cloud-storage/
│   └── networking/
├── ytempire-staging/         # Staging environment
│   └── (mirrors production)
├── ytempire-development/     # Development environment
└── ytempire-shared/          # Shared resources
    ├── container-registry/
    ├── secrets-manager/
    └── monitoring/
```

### 2. Compute Resources

#### GKE Cluster Specifications

```yaml
# production-gke-cluster.yaml
apiVersion: container.googleapis.com/v1
kind: Cluster
metadata:
  name: ytempire-prod-cluster
spec:
  # Cluster Configuration
  location: us-central1
  locations:
    - us-central1-a
    - us-central1-b
    - us-central1-c
  
  # Master Configuration
  masterAuthorizedNetworksConfig:
    enabled: true
    cidrBlocks:
      - displayName: "Office Network"
        cidrBlock: "203.0.113.0/24"
      - displayName: "VPN Network"
        cidrBlock: "198.51.100.0/24"
  
  # Network Configuration
  networkConfig:
    enableIntraNodeVisibility: true
    defaultSnatStatus:
      disabled: false
  
  # Security Configuration
  binaryAuthorization:
    enabled: true
  workloadIdentityConfig:
    workloadPool: "ytempire-production.svc.id.goog"
  
  # Add-ons
  addonsConfig:
    httpLoadBalancing:
      disabled: false
    horizontalPodAutoscaling:
      disabled: false
    networkPolicyConfig:
      disabled: false
    cloudRunConfig:
      disabled: false
```

#### Node Pool Configurations

```yaml
# Node Pool 1: General Workloads
nodePool:
  name: general-workload-pool
  config:
    machineType: n2-standard-8  # 8 vCPUs, 32GB RAM
    diskSizeGb: 100
    diskType: pd-ssd
    preemptible: false
    
    # Node configuration
    metadata:
      disable-legacy-endpoints: "true"
    oauthScopes:
      - "https://www.googleapis.com/auth/cloud-platform"
    
    # Security
    shieldedInstanceConfig:
      enableSecureBoot: true
      enableIntegrityMonitoring: true
    
  # Auto-scaling
  autoscaling:
    enabled: true
    minNodeCount: 3
    maxNodeCount: 20
  
  # Management
  management:
    autoUpgrade: true
    autoRepair: true
    upgradeOptions:
      autoUpgradeStartTime: "2025-01-01T03:00:00Z"
      description: "Scheduled weekly upgrade"

# Node Pool 2: GPU Workloads (Video Processing)
nodePool:
  name: gpu-processing-pool
  config:
    machineType: n1-standard-8
    accelerators:
      - acceleratorCount: 1
        acceleratorType: nvidia-tesla-t4
        gpuPartitionSize: ""
    
    diskSizeGb: 200
    diskType: pd-ssd
    preemptible: true  # Cost optimization
    
    # GPU-specific configuration
    guestAccelerator:
      - type: "nvidia-tesla-t4"
        count: 1
    
  autoscaling:
    enabled: true
    minNodeCount: 0  # Scale to zero when not needed
    maxNodeCount: 10
  
  # Taints for GPU nodes
  taints:
    - key: nvidia.com/gpu
      value: "true"
      effect: NoSchedule

# Node Pool 3: Memory-Intensive Workloads
nodePool:
  name: memory-intensive-pool
  config:
    machineType: n2-highmem-4  # 4 vCPUs, 32GB RAM
    diskSizeGb: 100
    diskType: pd-standard
    
  autoscaling:
    enabled: true
    minNodeCount: 2
    maxNodeCount: 8
```

### 3. Database Resources

#### Cloud SQL Configuration

```yaml
# Cloud SQL PostgreSQL Instance
resource "google_sql_database_instance" "main" {
  name             = "ytempire-prod-db-primary"
  database_version = "POSTGRES_15"
  region           = "us-central1"
  
  settings {
    tier = "db-custom-16-65536"  # 16 vCPUs, 64GB RAM
    
    # High Availability
    availability_type = "REGIONAL"
    
    # Disk Configuration
    disk_size         = 500  # GB
    disk_type         = "PD_SSD"
    disk_autoresize   = true
    disk_autoresize_limit = 2000  # GB
    
    # Backup Configuration
    backup_configuration {
      enabled                        = true
      start_time                     = "02:00"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 30
        retention_unit   = "COUNT"
      }
    }
    
    # Performance Insights
    insights_config {
      query_insights_enabled  = true
      query_string_length    = 1024
      record_application_tags = true
      record_client_address  = true
    }
    
    # Database Flags
    database_flags {
      name  = "max_connections"
      value = "1000"
    }
    database_flags {
      name  = "shared_buffers"
      value = "16GB"
    }
    database_flags {
      name  = "effective_cache_size"
      value = "48GB"
    }
    database_flags {
      name  = "work_mem"
      value = "32MB"
    }
    database_flags {
      name  = "maintenance_work_mem"
      value = "2GB"
    }
    database_flags {
      name  = "wal_buffers"
      value = "16MB"
    }
    database_flags {
      name  = "checkpoint_completion_target"
      value = "0.9"
    }
  }
  
  # Read Replicas
  replica_configuration {
    failover_target = false
  }
}

# Read Replica Configuration
resource "google_sql_database_instance" "read_replica" {
  count                = 2
  name                 = "ytempire-prod-db-replica-${count.index + 1}"
  master_instance_name = google_sql_database_instance.main.name
  region               = "us-central1"
  
  replica_configuration {
    failover_target = count.index == 0 ? true : false
  }
  
  settings {
    tier = "db-custom-8-32768"  # 8 vCPUs, 32GB RAM
    
    availability_type = "ZONAL"
    disk_size        = 500
    disk_type        = "PD_SSD"
    
    database_flags {
      name  = "max_connections"
      value = "500"
    }
  }
}
```

### 4. Storage Resources

#### Cloud Storage Buckets

```yaml
# Storage Bucket Configuration
resource "google_storage_bucket" "video_storage" {
  name          = "ytempire-prod-videos"
  location      = "US"
  storage_class = "STANDARD"
  
  # Lifecycle Management
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }
  
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "ARCHIVE"
    }
  }
  
  # Versioning
  versioning {
    enabled = true
  }
  
  # CORS Configuration
  cors {
    origin          = ["https://ytempire.com"]
    method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
  
  # Encryption
  encryption {
    default_kms_key_name = "projects/ytempire-production/locations/us/keyRings/ytempire-keyring/cryptoKeys/storage-key"
  }
}

# Backup Storage Bucket
resource "google_storage_bucket" "backups" {
  name          = "ytempire-prod-backups"
  location      = "US"
  storage_class = "NEARLINE"
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
  
  versioning {
    enabled = true
  }
}
```

### 5. Networking Resources

#### VPC Configuration

```yaml
# VPC Network
resource "google_compute_network" "vpc" {
  name                    = "ytempire-prod-vpc"
  auto_create_subnetworks = false
  routing_mode           = "REGIONAL"
}

# Subnets
resource "google_compute_subnetwork" "gke_subnet" {
  name          = "gke-subnet"
  ip_cidr_range = "10.0.0.0/20"
  region        = "us-central1"
  network       = google_compute_network.vpc.id
  
  secondary_ip_range {
    range_name    = "gke-pods"
    ip_cidr_range = "10.4.0.0/14"
  }
  
  secondary_ip_range {
    range_name    = "gke-services"
    ip_cidr_range = "10.8.0.0/20"
  }
  
  private_ip_google_access = true
}

# Cloud NAT for outbound connectivity
resource "google_compute_router_nat" "nat" {
  name                               = "ytempire-prod-nat"
  router                             = google_compute_router.router.name
  region                             = "us-central1"
  nat_ip_allocate_option            = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  
  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}
```

---

## Kubernetes Deployment Architecture

### 1. Namespace Organization

```yaml
# namespaces.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ytempire-core
  labels:
    name: ytempire-core
    environment: production
---
apiVersion: v1
kind: Namespace
metadata:
  name: ytempire-processing
  labels:
    name: ytempire-processing
    environment: production
---
apiVersion: v1
kind: Namespace
metadata:
  name: ytempire-monitoring
  labels:
    name: ytempire-monitoring
    environment: production
---
apiVersion: v1
kind: Namespace
metadata:
  name: ytempire-security
  labels:
    name: ytempire-security
    environment: production
```

### 2. Resource Quotas

```yaml
# resource-quotas.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ytempire-core-quota
  namespace: ytempire-core
spec:
  hard:
    requests.cpu: "100"
    requests.memory: "200Gi"
    limits.cpu: "200"
    limits.memory: "400Gi"
    persistentvolumeclaims: "10"
    services.loadbalancers: "5"
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ytempire-processing-quota
  namespace: ytempire-processing
spec:
  hard:
    requests.cpu: "500"
    requests.memory: "1Ti"
    limits.cpu: "1000"
    limits.memory: "2Ti"
    requests.nvidia.com/gpu: "10"
    persistentvolumeclaims: "50"
```

---

## Deployment Manifests

### 1. API Service Deployment

```yaml
# api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ytempire-api
  namespace: ytempire-core
  labels:
    app: ytempire-api
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: ytempire-api
  template:
    metadata:
      labels:
        app: ytempire-api
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: ytempire-api
      
      # Anti-affinity for high availability
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - ytempire-api
            topologyKey: kubernetes.io/hostname
      
      # Init containers
      initContainers:
      - name: wait-for-db
        image: busybox:1.35
        command: ['sh', '-c', 'until nc -z ytempire-postgres 5432; do echo waiting for db; sleep 2; done']
      
      containers:
      - name: api
        image: gcr.io/ytempire-production/ytempire-api:v1.0.0
        
        # Resource requests and limits
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        
        # Environment variables
        env:
        - name: APP_ENV
          value: "production"
        - name: LOG_LEVEL
          value: "info"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ytempire-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: ytempire-secrets
              key: redis-url
        
        # Environment from ConfigMap
        envFrom:
        - configMapRef:
            name: ytempire-api-config
        
        # Ports
        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /health/live
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /health/ready
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        
        # Startup probe for slow starting containers
        startupProbe:
          httpGet:
            path: /health/ready
            port: http
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 30
        
        # Volume mounts
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: secrets
          mountPath: /app/secrets
          readOnly: true
        - name: tmp
          mountPath: /tmp
        
        # Security context
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
      
      # Volumes
      volumes:
      - name: config
        configMap:
          name: ytempire-api-config
      - name: secrets
        secret:
          secretName: ytempire-api-secrets
      - name: tmp
        emptyDir: {}
      
      # Pod disruption budget
      podDisruptionBudget:
        minAvailable: 2
```

### 2. Video Processing Worker Deployment

```yaml
# video-processor-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: video-processor
  namespace: ytempire-processing
spec:
  replicas: 5
  selector:
    matchLabels:
      app: video-processor
  template:
    metadata:
      labels:
        app: video-processor
    spec:
      serviceAccountName: video-processor
      
      # Node selector for GPU nodes
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
      
      # Tolerations for GPU nodes
      tolerations:
      - key: nvidia.com/gpu
        operator: Equal
        value: "true"
        effect: NoSchedule
      
      containers:
      - name: processor
        image: gcr.io/ytempire-production/video-processor:v1.0.0
        
        resources:
          requests:
            cpu: "2000m"
            memory: "8Gi"
            nvidia.com/gpu: 1
          limits:
            cpu: "4000m"
            memory: "16Gi"
            nvidia.com/gpu: 1
        
        env:
        - name: WORKER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: QUEUE_URL
          valueFrom:
            secretKeyRef:
              name: ytempire-secrets
              key: queue-url
        - name: STORAGE_BUCKET
          value: "ytempire-prod-videos"
        - name: PROCESSING_TIMEOUT
          value: "900"  # 15 minutes
        
        # Volume mounts for video processing
        volumeMounts:
        - name: workspace
          mountPath: /workspace
        - name: cache
          mountPath: /cache
        - name: models
          mountPath: /models
          readOnly: true
        
        # Lifecycle hooks
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 30"]
      
      volumes:
      - name: workspace
        emptyDir:
          sizeLimit: 100Gi
      - name: cache
        emptyDir:
          sizeLimit: 50Gi
      - name: models
        persistentVolumeClaim:
          claimName: ai-models-pvc
```

### 3. Frontend Deployment

```yaml
# frontend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ytempire-frontend
  namespace: ytempire-core
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: ytempire-frontend
  template:
    metadata:
      labels:
        app: ytempire-frontend
    spec:
      containers:
      - name: frontend
        image: gcr.io/ytempire-production/ytempire-frontend:v1.0.0
        
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "500m"
            memory: "512Mi"
        
        ports:
        - name: http
          containerPort: 3000
        
        env:
        - name: API_URL
          value: "https://api.ytempire.com"
        - name: CDN_URL
          value: "https://cdn.ytempire.com"
        
        livenessProbe:
          httpGet:
            path: /
            port: http
          periodSeconds: 10
        
        readinessProbe:
          httpGet:
            path: /
            port: http
          periodSeconds: 5
        
        # Security context
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
```

---

## Service Configurations

### 1. Service Definitions

```yaml
# api-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ytempire-api
  namespace: ytempire-core
  labels:
    app: ytempire-api
  annotations:
    cloud.google.com/neg: '{"ingress": true}'
    cloud.google.com/backend-config: '{"default": "ytempire-api-backendconfig"}'
spec:
  type: ClusterIP
  selector:
    app: ytempire-api
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  sessionAffinity: None
---
# Backend Configuration for Load Balancer
apiVersion: cloud.google.com/v1
kind: BackendConfig
metadata:
  name: ytempire-api-backendconfig
  namespace: ytempire-core
spec:
  healthCheck:
    checkIntervalSec: 10
    timeoutSec: 5
    healthyThreshold: 2
    unhealthyThreshold: 3
    type: HTTP
    requestPath: /health/ready
    port: 8080
  cdn:
    enabled: true
    cachePolicy:
      includeHost: true
      includeProtocol: true
      includeQueryString: false
    negativeCaching: true
    negativeCachingPolicy:
    - code: 404
      ttl: 300
    - code: 410
      ttl: 600
  iap:
    enabled: false
  timeoutSec: 30
  connectionDraining:
    drainingTimeoutSec: 60
```

### 2. Ingress Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ytempire-ingress
  namespace: ytempire-core
  annotations:
    kubernetes.io/ingress.global-static-ip-name: "ytempire-prod-ip"
    kubernetes.io/ingress.class: "gce"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    kubernetes.io/ingress.allow-http: "false"
spec:
  tls:
  - hosts:
    - api.ytempire.com
    - app.ytempire.com
    secretName: ytempire-tls
  rules:
  - host: api.ytempire.com
    http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: ytempire-api
            port:
              number: 80
  - host: app.ytempire.com
    http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: ytempire-frontend
            port:
              number: 80
```

---

## Storage and Persistence

### 1. Persistent Volume Claims

```yaml
# storage-claims.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ai-models-pvc
  namespace: ytempire-processing
spec:
  accessModes:
  - ReadOnlyMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: standard-rwo
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: processing-workspace-pvc
  namespace: ytempire-processing
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 500Gi
  storageClassName: fast-ssd
```

### 2. StatefulSet for Redis

```yaml
# redis-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
  namespace: ytempire-core
spec:
  serviceName: redis-cluster
  replicas: 3
  selector:
    matchLabels:
      app: redis-cluster
  template:
    metadata:
      labels:
        app: redis-cluster
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command:
        - redis-server
        - /config/redis.conf
        
        resources:
          requests:
            cpu: "500m"
            memory: "2Gi"
          limits:
            cpu: "1000m"
            memory: "4Gi"
        
        ports:
        - containerPort: 6379
        
        volumeMounts:
        - name: data
          mountPath: /data
        - name: config
          mountPath: /config
        
        livenessProbe:
          tcpSocket:
            port: 6379
          periodSeconds: 10
        
        readinessProbe:
          exec:
            command:
            - redis-cli
            - ping
          periodSeconds: 5
      
      volumes:
      - name: config
        configMap:
          name: redis-config
  
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 10Gi
```

---

## Monitoring and Observability

### 1. Prometheus Configuration

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: ytempire-monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    alerting:
      alertmanagers:
      - static_configs:
        - targets:
          - alertmanager:9093
    
    rule_files:
      - /etc/prometheus/rules/*.yml
    
    scrape_configs:
    - job_name: 'kubernetes-apiservers'
      kubernetes_sd_configs:
      - role: endpoints
      scheme: https
      tls_config:
        ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
      bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
      relabel_configs:
      - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
        action: keep
        regex: default;kubernetes;https
    
    - job_name: 'kubernetes-nodes'
      kubernetes_sd_configs:
      - role: node
      scheme: https
      tls_config:
        ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
      bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
      relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
    
    - job_name: 'kubernetes-pods'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name
```

### 2. Grafana Dashboards

```yaml
# grafana-dashboard-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards
  namespace: ytempire-monitoring
data:
  ytempire-overview.json: |
    {
      "dashboard": {
        "title": "YTEMPIRE Platform Overview",
        "panels": [
          {
            "title": "API Request Rate",
            "targets": [
              {
                "expr": "sum(rate(http_requests_total[5m])) by (service)"
              }
            ]
          },
          {
            "title": "Video Processing Rate",
            "targets": [
              {
                "expr": "sum(rate(videos_processed_total[5m]))"
              }
            ]
          },
          {
            "title": "Error Rate",
            "targets": [
              {
                "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))"
              }
            ]
          },
          {
            "title": "Resource Utilization",
            "targets": [
              {
                "expr": "avg(container_cpu_usage_seconds_total) by (pod)"
              },
              {
                "expr": "avg(container_memory_usage_bytes) by (pod)"
              }
            ]
          }
        ]
      }
    }
```

### 3. Alerting Rules

```yaml
# alerting-rules.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-alerts
  namespace: ytempire-monitoring
data:
  alerts.yml: |
    groups:
    - name: ytempire-critical
      interval: 30s
      rules:
      - alert: APIHighErrorRate
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
            /
            sum(rate(http_requests_total[5m])) by (service)
          ) > 0.05
        for: 5m
        labels:
          severity: critical
          team: platform-ops
        annotations:
          summary: "High error rate on {{ $labels.service }}"
          description: "Error rate is {{ $value | humanizePercentage }} for {{ $labels.service }}"
      
      - alert: PodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
        for: 5m
        labels:
          severity: critical
          team: platform-ops
        annotations:
          summary: "Pod {{ $labels.namespace }}/{{ $labels.pod }} is crash looping"
          description: "Pod {{ $labels.namespace }}/{{ $labels.pod }} has restarted {{ $value }} times in the last 15 minutes"
      
      - alert: HighMemoryUsage
        expr: |
          (
            container_memory_usage_bytes
            /
            container_spec_memory_limit_bytes
          ) > 0.9
        for: 10m
        labels:
          severity: warning
          team: platform-ops
        annotations:
          summary: "High memory usage for {{ $labels.pod }}"
          description: "Memory usage is {{ $value | humanizePercentage }} for {{ $labels.pod }}"
      
      - alert: VideoProcessingBacklog
        expr: video_processing_queue_depth > 100
        for: 15m
        labels:
          severity: warning
          team: platform-ops
        annotations:
          summary: "Video processing backlog growing"
          description: "Queue depth is {{ $value }} videos"
```

---

## Best Practices and Guidelines

### 1. Deployment Checklist

```markdown
## Pre-Deployment Checklist

### Code Review
- [ ] Code reviewed and approved
- [ ] Security scan passed
- [ ] Unit tests passing (>90% coverage)
- [ ] Integration tests passing
- [ ] Performance tests completed

### Container Security
- [ ] Base image scanned for vulnerabilities
- [ ] No secrets in image layers
- [ ] Non-root user configured
- [ ] Read-only root filesystem
- [ ] Security policies applied

### Kubernetes Configuration
- [ ] Resource limits set appropriately
- [ ] Health checks configured
- [ ] Service accounts with minimal permissions
- [ ] Network policies defined
- [ ] Pod disruption budgets set

### Monitoring
- [ ] Metrics endpoints exposed
- [ ] Dashboards configured
- [ ] Alerts configured
- [ ] Log aggregation working
- [ ] Tracing enabled
```

### 2. Troubleshooting Guide

```bash
# Common Kubernetes debugging commands

# Check pod status
kubectl get pods -n ytempire-core -o wide

# Describe pod for events
kubectl describe pod <pod-name> -n ytempire-core

# Check logs
kubectl logs <pod-name> -n ytempire-core --tail=100 -f

# Execute commands in pod
kubectl exec -it <pod-name> -n ytempire-core -- /bin/sh

# Check resource usage
kubectl top pods -n ytempire-core
kubectl top nodes

# Check service endpoints
kubectl get endpoints -n ytempire-core

# Debug networking
kubectl run debug --image=nicolaka/netshoot:latest -it --rm

# Check events
kubectl get events -n ytempire-core --sort-by='.lastTimestamp'
```

---

## Document Information

**Version**: 1.0  
**Last Updated**: January 2025  
**Next Review**: February 2025  
**Owner**: DevOps Engineering Team  
**Approved By**: Platform Operations Lead

### Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Jan 2025 | DevOps Team | Initial document creation |

---

**Note**: This document contains production configurations. Handle with appropriate security measures and ensure all secrets are properly managed through Google Secret Manager or Kubernetes secrets.