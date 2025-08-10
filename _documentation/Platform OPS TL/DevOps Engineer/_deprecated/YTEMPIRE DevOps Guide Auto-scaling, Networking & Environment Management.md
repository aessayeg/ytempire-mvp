# YTEMPIRE DevOps Guide: Auto-scaling, Networking & Environment Management
**Version 1.0 | January 2025**  
**Owner: DevOps Engineering Team**  
**Approved By: Platform Operations Lead**

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Auto-scaling Policies](#auto-scaling-policies)
3. [Network Architecture](#network-architecture)
4. [Environment Configuration Management](#environment-configuration-management)
5. [Load Balancing Strategies](#load-balancing-strategies)
6. [Service Mesh Configuration](#service-mesh-configuration)
7. [Traffic Management](#traffic-management)
8. [Monitoring and Alerting](#monitoring-and-alerting)

---

## Executive Summary

This document provides comprehensive guidance on implementing auto-scaling policies, network architecture, and environment configuration management for YTEMPIRE's platform. These configurations ensure optimal resource utilization, high availability, and seamless traffic management across our distributed infrastructure.

### Key Architecture Principles
- **Elasticity**: Scale resources based on demand automatically
- **Resilience**: Handle failures gracefully with no user impact
- **Security**: Zero-trust networking with encryption everywhere
- **Efficiency**: Optimize resource usage and minimize costs
- **Observability**: Complete visibility into system behavior

---

## Auto-scaling Policies

### 1. Horizontal Pod Autoscaler (HPA) Configuration

```yaml
# hpa/api-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ytempire-api-hpa
  namespace: ytempire-core
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ytempire-api
  minReplicas: 3
  maxReplicas: 50
  
  # Scaling metrics
  metrics:
  # CPU utilization
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  
  # Memory utilization
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  
  # Custom metrics - Request rate
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  
  # Custom metrics - Response time
  - type: Pods
    pods:
      metric:
        name: http_request_duration_p95
      target:
        type: AverageValue
        averageValue: "200m"  # 200ms
  
  # External metrics - Queue depth
  - type: External
    external:
      metric:
        name: redis_queue_depth
        selector:
          matchLabels:
            queue: video-processing
      target:
        type: Value
        value: "100"
  
  # Scaling behavior
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 minutes
      policies:
      - type: Percent
        value: 10  # Scale down by 10%
        periodSeconds: 60
      - type: Pods
        value: 2   # Scale down max 2 pods
        periodSeconds: 60
      selectPolicy: Min  # Use the most conservative policy
    
    scaleUp:
      stabilizationWindowSeconds: 60  # 1 minute
      policies:
      - type: Percent
        value: 100  # Double the pods
        periodSeconds: 60
      - type: Pods
        value: 10   # Add max 10 pods
        periodSeconds: 60
      selectPolicy: Max  # Use the most aggressive policy

---
# Video Processor HPA with GPU awareness
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: video-processor-hpa
  namespace: ytempire-processing
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: video-processor
  minReplicas: 0  # Scale to zero when no work
  maxReplicas: 20
  
  metrics:
  # GPU utilization
  - type: Pods
    pods:
      metric:
        name: gpu_utilization_percentage
      target:
        type: AverageValue
        averageValue: "80"
  
  # Queue-based scaling
  - type: External
    external:
      metric:
        name: video_processing_queue_depth
      target:
        type: Value
        value: "5"  # 5 videos per pod
  
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 600  # 10 minutes
      policies:
      - type: Pods
        value: 1
        periodSeconds: 300  # Scale down slowly
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Pods
        value: 5  # Scale up aggressively
        periodSeconds: 30
```

### 2. Vertical Pod Autoscaler (VPA) Configuration

```yaml
# vpa/api-vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: ytempire-api-vpa
  namespace: ytempire-core
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ytempire-api
  
  # Update policy
  updatePolicy:
    updateMode: "Auto"  # Auto, Recreate, or Off
    
  # Resource policy
  resourcePolicy:
    containerPolicies:
    - containerName: api
      minAllowed:
        cpu: 200m
        memory: 512Mi
      maxAllowed:
        cpu: 4
        memory: 8Gi
      controlledResources:
      - cpu
      - memory
      
    - containerName: sidecar-proxy
      mode: "Off"  # Don't autoscale sidecars

---
# VPA for batch processing
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: batch-processor-vpa
  namespace: ytempire-processing
spec:
  targetRef:
    apiVersion: batch/v1
    kind: Job
    name: batch-processor
  updatePolicy:
    updateMode: "Initial"  # Only set initial resources
  resourcePolicy:
    containerPolicies:
    - containerName: processor
      minAllowed:
        cpu: 1
        memory: 2Gi
      maxAllowed:
        cpu: 8
        memory: 32Gi
```

### 3. Cluster Autoscaler Configuration

```yaml
# cluster-autoscaler/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      serviceAccountName: cluster-autoscaler
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.27.0
        name: cluster-autoscaler
        resources:
          limits:
            cpu: 100m
            memory: 300Mi
          requests:
            cpu: 100m
            memory: 300Mi
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=gce
        - --expander=least-waste
        - --node-group-auto-discovery=mig:name=ytempire-.*
        - --max-nodes-total=100
        - --cores-total=50:500
        - --memory-total=100:2000
        - --scale-down-enabled=true
        - --scale-down-delay-after-add=10m
        - --scale-down-unneeded-time=10m
        - --scale-down-utilization-threshold=0.5
        - --max-node-provision-time=15m
        - --balance-similar-node-groups=true
        - --skip-nodes-with-system-pods=true
        env:
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: /etc/gcp/service-account.json
        volumeMounts:
        - name: ssl-certs
          mountPath: /etc/ssl/certs/ca-certificates.crt
          readOnly: true
        - name: gcp-service-account
          mountPath: /etc/gcp
          readOnly: true
      volumes:
      - name: ssl-certs
        hostPath:
          path: /etc/ssl/certs/ca-certificates.crt
      - name: gcp-service-account
        secret:
          secretName: cluster-autoscaler-gcp-sa
```

### 4. Custom Metrics for Scaling

```yaml
# custom-metrics/prometheus-adapter.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-adapter-config
  namespace: monitoring
data:
  config.yaml: |
    rules:
    # HTTP request rate metric
    - seriesQuery: 'http_requests_total{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^(.*)_total$"
        as: "${1}_per_second"
      metricsQuery: 'sum(rate(<<.Series>>{<<.LabelMatchers>>}[1m])) by (<<.GroupBy>>)'
    
    # Response time percentiles
    - seriesQuery: 'http_request_duration_seconds{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^(.*)_seconds$"
        as: "${1}_p95"
      metricsQuery: 'histogram_quantile(0.95, sum(rate(<<.Series>>_bucket{<<.LabelMatchers>>}[1m])) by (<<.GroupBy>>, le))'
    
    # GPU utilization
    - seriesQuery: 'nvidia_gpu_utilization{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        as: "gpu_utilization_percentage"
      metricsQuery: 'avg(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
    
    # Queue depth (external metric)
    externalRules:
    - seriesQuery: 'redis_queue_length{queue!=""}'
      name:
        matches: "^redis_queue_length$"
        as: "video_processing_queue_depth"
      metricsQuery: 'sum(<<.Series>>{queue="video-processing"})'
```

---

## Network Architecture

### 1. VPC Network Design

```hcl
# terraform/network.tf
# Production VPC
resource "google_compute_network" "production_vpc" {
  name                    = "ytempire-production-vpc"
  auto_create_subnetworks = false
  routing_mode           = "REGIONAL"
  mtu                    = 1460

  lifecycle {
    prevent_destroy = true
  }
}

# Primary subnet for GKE
resource "google_compute_subnetwork" "gke_subnet" {
  name          = "gke-subnet"
  ip_cidr_range = "10.0.0.0/20"  # 4,094 IPs
  region        = "us-central1"
  network       = google_compute_network.production_vpc.id

  # Secondary ranges for GKE
  secondary_ip_range {
    range_name    = "gke-pods"
    ip_cidr_range = "10.4.0.0/14"  # 262,142 IPs
  }

  secondary_ip_range {
    range_name    = "gke-services"
    ip_cidr_range = "10.8.0.0/20"  # 4,094 IPs
  }

  private_ip_google_access = true

  log_config {
    aggregation_interval = "INTERVAL_10_MIN"
    flow_sampling       = 0.5
    metadata           = "INCLUDE_ALL_METADATA"
  }
}

# Database subnet
resource "google_compute_subnetwork" "database_subnet" {
  name          = "database-subnet"
  ip_cidr_range = "10.1.0.0/24"  # 254 IPs
  region        = "us-central1"
  network       = google_compute_network.production_vpc.id

  private_ip_google_access = true
}

# Management subnet
resource "google_compute_subnetwork" "management_subnet" {
  name          = "management-subnet"
  ip_cidr_range = "10.2.0.0/24"  # 254 IPs
  region        = "us-central1"
  network       = google_compute_network.production_vpc.id

  private_ip_google_access = true
}

# Cloud NAT for outbound traffic
resource "google_compute_router" "router" {
  name    = "ytempire-router"
  region  = "us-central1"
  network = google_compute_network.production_vpc.id

  bgp {
    asn = 64514
  }
}

resource "google_compute_router_nat" "nat" {
  name                               = "ytempire-nat"
  router                             = google_compute_router.router.name
  region                             = google_compute_router.router.region
  nat_ip_allocate_option            = "MANUAL_ONLY"
  nat_ips                           = google_compute_address.nat[*].self_link
  source_subnetwork_ip_ranges_to_nat = "LIST_OF_SUBNETWORKS"

  subnetwork {
    name                    = google_compute_subnetwork.gke_subnet.id
    source_ip_ranges_to_nat = ["ALL_IP_RANGES"]
  }

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Static IPs for NAT
resource "google_compute_address" "nat" {
  count  = 2
  name   = "ytempire-nat-ip-${count.index}"
  region = "us-central1"
}
```

### 2. Firewall Rules

```hcl
# terraform/firewall.tf
# Allow internal communication
resource "google_compute_firewall" "allow_internal" {
  name    = "ytempire-allow-internal"
  network = google_compute_network.production_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = [
    "10.0.0.0/20",  # GKE subnet
    "10.1.0.0/24",  # Database subnet
    "10.2.0.0/24",  # Management subnet
    "10.4.0.0/14",  # GKE pods
    "10.8.0.0/20"   # GKE services
  ]
}

# Allow health checks
resource "google_compute_firewall" "allow_health_checks" {
  name    = "ytempire-allow-health-checks"
  network = google_compute_network.production_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8080", "10256"]
  }

  source_ranges = [
    "35.191.0.0/16",  # Google health check IPs
    "130.211.0.0/22"
  ]

  target_tags = ["gke-node", "load-balancer"]
}

# Deny all ingress by default
resource "google_compute_firewall" "deny_all_ingress" {
  name     = "ytempire-deny-all-ingress"
  network  = google_compute_network.production_vpc.name
  priority = 65534

  deny {
    protocol = "all"
  }

  source_ranges = ["0.0.0.0/0"]
}
```

### 3. Network Policies

```yaml
# network-policies/default-deny.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: ytempire-core
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress

---
# Allow API ingress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-api-ingress
  namespace: ytempire-core
spec:
  podSelector:
    matchLabels:
      app: ytempire-api
  policyTypes:
  - Ingress
  ingress:
  - from:
    # From ingress controller
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    # From frontend pods
    - podSelector:
        matchLabels:
          app: ytempire-frontend
    # From monitoring
    - namespaceSelector:
        matchLabels:
          name: ytempire-monitoring
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 9090  # Metrics

---
# Allow database access
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-database-access
  namespace: ytempire-core
spec:
  podSelector:
    matchLabels:
      app: ytempire-api
  policyTypes:
  - Egress
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
  # Allow DNS
  - to:
    - namespaceSelector: {}
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
```

### 4. Service Mesh Network Configuration

```yaml
# istio/virtual-service.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ytempire-api
  namespace: ytempire-core
spec:
  hosts:
  - api.ytempire.com
  gateways:
  - ytempire-gateway
  http:
  - match:
    - uri:
        prefix: "/v1/"
    route:
    - destination:
        host: ytempire-api
        port:
          number: 80
      weight: 100
    timeout: 30s
    retries:
      attempts: 3
      perTryTimeout: 10s
      retryOn: 5xx,reset,connect-failure,refused-stream
    fault:
      delay:
        percentage:
          value: 0.1
        fixedDelay: 5s
      abort:
        percentage:
          value: 0.1
        httpStatus: 500

---
# Circuit breaker configuration
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: ytempire-api
  namespace: ytempire-core
spec:
  host: ytempire-api
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
        maxRequestsPerConnection: 2
        h2UpgradePolicy: UPGRADE
    loadBalancer:
      simple: LEAST_REQUEST
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
      minHealthPercent: 50
```

---

## Environment Configuration Management

### 1. ConfigMap Management

```yaml
# config/base/api-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ytempire-api-config
  namespace: ytempire-core
data:
  # Application configuration
  app.yaml: |
    server:
      port: 8080
      timeout: 30s
      graceful_shutdown: 30s
    
    database:
      max_connections: 100
      connection_timeout: 5s
      idle_timeout: 300s
      max_lifetime: 3600s
    
    redis:
      max_connections: 50
      dial_timeout: 5s
      read_timeout: 3s
      write_timeout: 3s
    
    features:
      rate_limiting: true
      request_logging: true
      metrics: true
      tracing: true
    
    limits:
      max_upload_size: 100MB
      max_request_size: 10MB
      rate_limit_rps: 100
  
  # Logging configuration
  logging.yaml: |
    level: info
    format: json
    outputs:
      - stdout
      - file
    file:
      path: /var/log/ytempire/api.log
      max_size: 100MB
      max_backups: 5
      max_age: 30
```

### 2. Secret Management

```yaml
# External Secrets Operator configuration
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: gcpsm-secret-store
  namespace: ytempire-core
spec:
  provider:
    gcpsm:
      projectID: ytempire-production
      auth:
        secretRef:
          secretAccessKey:
            name: gcpsm-secret
            key: secret-access-credentials

---
# External secret definition
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: ytempire-api-secrets
  namespace: ytempire-core
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: gcpsm-secret-store
    kind: SecretStore
  target:
    name: ytempire-api-secrets
    creationPolicy: Owner
  data:
  - secretKey: database-url
    remoteRef:
      key: ytempire-database-url
  - secretKey: jwt-secret
    remoteRef:
      key: ytempire-jwt-secret
  - secretKey: openai-api-key
    remoteRef:
      key: ytempire-openai-key
  - secretKey: youtube-credentials
    remoteRef:
      key: ytempire-youtube-creds
```

### 3. Environment-Specific Configurations

```yaml
# kustomization/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - deployment.yaml
  - service.yaml
  - configmap.yaml
  - hpa.yaml

commonLabels:
  app.kubernetes.io/name: ytempire
  app.kubernetes.io/component: api

---
# kustomization/overlays/staging/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

bases:
  - ../../base

patchesStrategicMerge:
  - deployment-patch.yaml
  - configmap-patch.yaml

configMapGenerator:
  - name: ytempire-api-config
    behavior: merge
    literals:
      - LOG_LEVEL=debug
      - ENVIRONMENT=staging

replicas:
  - name: ytempire-api
    count: 2

---
# kustomization/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

bases:
  - ../../base

patchesStrategicMerge:
  - deployment-patch.yaml
  - configmap-patch.yaml

configMapGenerator:
  - name: ytempire-api-config
    behavior: merge
    literals:
      - LOG_LEVEL=info
      - ENVIRONMENT=production

replicas:
  - name: ytempire-api
    count: 5

resources:
  - pdb.yaml
  - network-policy.yaml
```

### 4. Dynamic Configuration with Helm

```yaml
# helm/ytempire/values.yaml
global:
  environment: production
  domain: ytempire.com
  imageRegistry: gcr.io/ytempire-production

api:
  enabled: true
  replicaCount: 5
  image:
    repository: ytempire-api
    tag: v1.0.0
    pullPolicy: IfNotPresent
  
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2000m
      memory: 4Gi
  
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 50
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80
  
  config:
    logLevel: info
    database:
      maxConnections: 100
      ssl: true
    redis:
      maxConnections: 50
      cluster: true
  
  secrets:
    external: true
    provider: google-secret-manager

processor:
  enabled: true
  replicaCount: 10
  gpu:
    enabled: true
    type: nvidia-tesla-t4
    count: 1
  
  resources:
    requests:
      cpu: 2000m
      memory: 8Gi
      nvidia.com/gpu: 1
    limits:
      cpu: 4000m
      memory: 16Gi
      nvidia.com/gpu: 1

monitoring:
  prometheus:
    enabled: true
    retention: 30d
    storage: 100Gi
  
  grafana:
    enabled: true
    adminPassword: changeme
    dashboards:
      - ytempire-overview
      - api-performance
      - gpu-utilization
```

---

## Load Balancing Strategies

### 1. Global Load Balancer Configuration

```yaml
# load-balancer/global-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ytempire-global-ingress
  namespace: ytempire-core
  annotations:
    kubernetes.io/ingress.global-static-ip-name: "ytempire-global-ip"
    kubernetes.io/ingress.class: "gce"
    kubernetes.io/ingress.allow-http: "false"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    # Backend configuration
    cloud.google.com/backend-config: '{"default": "ytempire-backend-config"}'
    # Frontend configuration
    cloud.google.com/frontend-config: "ytempire-frontend-config"
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

---
# Backend configuration
apiVersion: cloud.google.com/v1
kind: BackendConfig
metadata:
  name: ytempire-backend-config
  namespace: ytempire-core
spec:
  # Health check configuration
  healthCheck:
    checkIntervalSec: 10
    timeoutSec: 5
    healthyThreshold: 2
    unhealthyThreshold: 3
    type: HTTP
    requestPath: /health
    port: 8080
  
  # Session affinity
  sessionAffinity:
    affinityType: "CLIENT_IP"
    affinityCookieTtlSec: 3600
  
  # Connection draining
  connectionDraining:
    drainingTimeoutSec: 60
  
  # Timeout
  timeoutSec: 30
  
  # CDN configuration
  cdn:
    enabled: true
    cachePolicy:
      includeHost: true
      includeProtocol: true
      includeQueryString: false
      queryStringWhitelist: ["version", "filter"]
    negativeCaching: true
    negativeCachingPolicy:
    - code: 404
      ttl: 300
    - code: 410
      ttl: 600
  
  # Custom request headers
  customRequestHeaders:
    headers:
    - X-Client-Region:{client_region}
    - X-Client-City:{client_city}
    - X-Client-Country:{client_country}

---
# Frontend configuration
apiVersion: networking.gke.io/v1beta1
kind: FrontendConfig
metadata:
  name: ytempire-frontend-config
  namespace: ytempire-core
spec:
  # SSL policy
  sslPolicy: ytempire-ssl-policy
  
  # Redirect HTTP to HTTPS
  redirectToHttps:
    enabled: true
    responseCodeName: MOVED_PERMANENTLY_DEFAULT
```

### 2. Regional Load Balancing

```yaml
# load-balancer/regional-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ytempire-api-regional
  namespace: ytempire-core
  annotations:
    cloud.google.com/load-balancer-type: "Internal"
    cloud.google.com/backend-config: '{"default": "ytempire-ilb-config"}'
spec:
  type: LoadBalancer
  loadBalancerIP: 10.0.0.100  # Reserved internal IP
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800  # 3 hours
  selector:
    app: ytempire-api
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  - name: grpc
    port: 50051
    targetPort: 50051
    protocol: TCP
```

---

## Service Mesh Configuration

### 1. Istio Service Mesh Setup

```yaml
# istio/control-plane.yaml
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: ytempire-istio
spec:
  profile: production
  
  components:
    pilot:
      k8s:
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
        hpaSpec:
          minReplicas: 2
          maxReplicas: 5
    
    ingressGateways:
    - name: istio-ingressgateway
      enabled: true
      k8s:
        resources:
          requests:
            cpu: 1000m
            memory: 1Gi
        hpaSpec:
          minReplicas: 3
          maxReplicas: 10
        service:
          type: LoadBalancer
          loadBalancerIP: ""  # Auto-assign
    
    egressGateways:
    - name: istio-egressgateway
      enabled: true
      k8s:
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
  
  meshConfig:
    accessLogFile: /dev/stdout
    defaultConfig:
      proxyStatsMatcher:
        inclusionRegexps:
        - ".*outlier_detection.*"
        - ".*circuit_breakers.*"
        - ".*upstream_rq_retry.*"
        - ".*upstream_rq_pending.*"
    
    extensionProviders:
    - name: prometheus
      prometheus:
        service: prometheus.ytempire-monitoring.svc.cluster.local
        port: 9090
    
    - name: zipkin
      zipkin:
        service: zipkin.ytempire-monitoring.svc.cluster.local
        port: 9411
  
  values:
    global:
      proxy:
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 2000m
            memory: 1Gi
```

### 2. Service Mesh Traffic Policies

```yaml
# istio/traffic-management.yaml
# Retry policy
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ytempire-api-retry
  namespace: ytempire-core
spec:
  hosts:
  - ytempire-api
  http:
  - retries:
      attempts: 3
      perTryTimeout: 10s
      retryOn: 5xx,reset,connect-failure,refused-stream
      retryRemoteLocalities: true

---
# Load balancing policy
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: ytempire-api-lb
  namespace: ytempire-core
spec:
  host: ytempire-api
  trafficPolicy:
    loadBalancer:
      consistentHash:
        httpCookie:
          name: "session"
          ttl: 3600s

---
# Circuit breaker
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: ytempire-api-cb
  namespace: ytempire-core
spec:
  host: ytempire-api
  trafficPolicy:
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
      minHealthPercent: 30
      splitExternalLocalOriginErrors: true
```

---

## Traffic Management

### 1. Blue-Green Deployment

```yaml
# traffic/blue-green.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ytempire-api-bg
  namespace: ytempire-core
spec:
  hosts:
  - ytempire-api
  http:
  - match:
    - headers:
        version:
          exact: blue
    route:
    - destination:
        host: ytempire-api
        subset: blue
  - route:
    - destination:
        host: ytempire-api
        subset: green

---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: ytempire-api-bg-dest
  namespace: ytempire-core
spec:
  host: ytempire-api
  subsets:
  - name: blue
    labels:
      version: blue
  - name: green
    labels:
      version: green
```

### 2. Canary Deployment

```yaml
# traffic/canary.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ytempire-api-canary
  namespace: ytempire-core
spec:
  hosts:
  - ytempire-api
  http:
  - match:
    - headers:
        user-group:
          exact: beta
    route:
    - destination:
        host: ytempire-api
        subset: canary
  - route:
    - destination:
        host: ytempire-api
        subset: stable
      weight: 95
    - destination:
        host: ytempire-api
        subset: canary
      weight: 5
```

### 3. Traffic Mirroring

```yaml
# traffic/mirroring.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ytempire-api-mirror
  namespace: ytempire-core
spec:
  hosts:
  - ytempire-api
  http:
  - route:
    - destination:
        host: ytempire-api
        subset: production
    mirror:
      host: ytempire-api
      subset: staging
    mirrorPercentage:
      value: 10.0
```

---

## Monitoring and Alerting

### 1. Auto-scaling Metrics

```yaml
# monitoring/scaling-alerts.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: scaling-alerts
  namespace: ytempire-monitoring
spec:
  groups:
  - name: autoscaling
    interval: 30s
    rules:
    # HPA reaching limits
    - alert: HPAMaxedOut
      expr: |
        kube_horizontalpodautoscaler_status_current_replicas
        ==
        kube_horizontalpodautoscaler_spec_max_replicas
      for: 15m
      labels:
        severity: warning
        team: platform-ops
      annotations:
        summary: "HPA {{ $labels.horizontalpodautoscaler }} has reached max replicas"
        description: "HPA {{ $labels.horizontalpodautoscaler }} in {{ $labels.namespace }} has been at max replicas for 15 minutes"
    
    # Rapid scaling events
    - alert: RapidScaling
      expr: |
        abs(delta(kube_horizontalpodautoscaler_status_current_replicas[5m])) > 10
      for: 5m
      labels:
        severity: warning
        team: platform-ops
      annotations:
        summary: "Rapid scaling detected for {{ $labels.horizontalpodautoscaler }}"
        description: "{{ $labels.horizontalpodautoscaler }} scaled by {{ $value }} replicas in 5 minutes"
    
    # Node pressure
    - alert: NodeMemoryPressure
      expr: |
        kube_node_status_condition{condition="MemoryPressure",status="true"} > 0
      for: 5m
      labels:
        severity: critical
        team: platform-ops
      annotations:
        summary: "Node {{ $labels.node }} under memory pressure"
        description: "Node {{ $labels.node }} has been under memory pressure for 5 minutes"
```

### 2. Network Monitoring

```yaml
# monitoring/network-dashboard.json
{
  "dashboard": {
    "title": "YTEMPIRE Network Overview",
    "panels": [
      {
        "title": "Ingress Traffic Rate",
        "targets": [
          {
            "expr": "sum(rate(istio_request_total{destination_service_namespace=\"ytempire-core\"}[5m])) by (destination_service_name)"
          }
        ]
      },
      {
        "title": "P95 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket[5m])) by (destination_service_name, le))"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "sum(rate(istio_request_total{response_code=~\"5..\"}[5m])) by (destination_service_name) / sum(rate(istio_request_total[5m])) by (destination_service_name)"
          }
        ]
      },
      {
        "title": "Circuit Breaker Status",
        "targets": [
          {
            "expr": "sum(envoy_cluster_circuit_breakers_open) by (cluster_name)"
          }
        ]
      }
    ]
  }
}
```

---

## Best Practices and Guidelines

### 1. Auto-scaling Best Practices

```markdown
## Auto-scaling Configuration Guidelines

### HPA Configuration
- Set conservative scale-down policies to avoid flapping
- Use multiple metrics (CPU, memory, custom) for better decisions
- Always set resource requests/limits for predictable scaling
- Monitor scaling events and adjust thresholds based on patterns
- Use behavior policies to control scaling velocity

### VPA Configuration
- Use "Initial" mode for batch jobs
- Use "Auto" mode carefully in production
- Set reasonable min/max bounds
- Monitor recommendation changes over time
- Consider using VPA in recommendation mode first

### Cluster Autoscaler
- Use node pools with different machine types
- Enable scale-down after careful testing
- Set appropriate scale-down delay (10-15 minutes)
- Use pod disruption budgets to protect workloads
- Monitor node utilization and optimize instance types

### Cost Optimization
- Use preemptible/spot nodes for fault-tolerant workloads
- Implement pod priority classes
- Use horizontal scaling before vertical scaling
- Regular review of resource utilization
- Implement resource quotas per namespace
```

### 2. Network Security Best Practices

```yaml
# network-security-checklist.yaml
network_security_best_practices:
  zero_trust_principles:
    - Deny all traffic by default
    - Explicitly allow required connections
    - Use mutual TLS everywhere possible
    - Implement service-to-service authentication
    - Regular security policy audits
  
  firewall_rules:
    - Minimize source IP ranges
    - Use service accounts for GCP resources
    - Implement egress filtering
    - Regular review of rules
    - Use tags effectively
  
  network_policies:
    - Start with deny-all policy
    - Allow traffic incrementally
    - Use label selectors effectively
    - Test policies in staging first
    - Document all policies
  
  service_mesh:
    - Enable mTLS mesh-wide
    - Use authorization policies
    - Implement rate limiting
    - Enable access logging
    - Regular certificate rotation
```

### 3. Environment Management Best Practices

```markdown
## Environment Configuration Best Practices

### ConfigMap Management
- Never store secrets in ConfigMaps
- Use immutable ConfigMaps when possible
- Version ConfigMaps with suffixes
- Implement configuration validation
- Use Kustomize or Helm for templating

### Secret Management
- Use external secret operators
- Enable secret rotation
- Implement least privilege access
- Audit secret access
- Never commit secrets to git

### Multi-Environment Strategy
- Use GitOps for deployments
- Maintain environment parity
- Implement proper RBAC
- Use namespace isolation
- Regular environment sync validation
```

---

## Troubleshooting Guide

### 1. Auto-scaling Issues

```bash
#!/bin/bash
# troubleshoot-autoscaling.sh

echo "=== HPA Status Check ==="
kubectl get hpa -A
kubectl describe hpa -A | grep -A 5 "Conditions:"

echo -e "\n=== Metrics Server Check ==="
kubectl get deployment metrics-server -n kube-system
kubectl top nodes
kubectl top pods -A

echo -e "\n=== Recent Scaling Events ==="
kubectl get events -A | grep -i "scale"

echo -e "\n=== Pod Resource Usage ==="
kubectl get pods -A -o custom-columns=\
"NAMESPACE:.metadata.namespace,\
NAME:.metadata.name,\
CPU_REQ:.spec.containers[*].resources.requests.cpu,\
CPU_LIM:.spec.containers[*].resources.limits.cpu,\
MEM_REQ:.spec.containers[*].resources.requests.memory,\
MEM_LIM:.spec.containers[*].resources.limits.memory"

echo -e "\n=== VPA Recommendations ==="
kubectl get vpa -A -o custom-columns=\
"NAMESPACE:.metadata.namespace,\
NAME:.metadata.name,\
MODE:.spec.updatePolicy.updateMode,\
CPU:.status.recommendation.containerRecommendations[0].target.cpu,\
MEMORY:.status.recommendation.containerRecommendations[0].target.memory"
```

### 2. Network Connectivity Issues

```bash
#!/bin/bash
# troubleshoot-network.sh

echo "=== Network Policy Check ==="
kubectl get networkpolicies -A

echo -e "\n=== Service Endpoints ==="
kubectl get endpoints -A

echo -e "\n=== Ingress Status ==="
kubectl get ingress -A
kubectl describe ingress -A | grep -A 5 "Address:"

echo -e "\n=== DNS Resolution Test ==="
kubectl run -it --rm debug --image=nicolaka/netshoot --restart=Never -- \
  nslookup ytempire-api.ytempire-core.svc.cluster.local

echo -e "\n=== Service Connectivity Test ==="
kubectl run -it --rm debug --image=nicolaka/netshoot --restart=Never -- \
  curl -v http://ytempire-api.ytempire-core.svc.cluster.local/health

echo -e "\n=== Istio Configuration Check ==="
istioctl analyze -A
istioctl proxy-status
```

### 3. Load Balancer Issues

```bash
#!/bin/bash
# troubleshoot-lb.sh

echo "=== Load Balancer Status ==="
kubectl get svc -A -o wide | grep LoadBalancer

echo -e "\n=== Backend Health ==="
gcloud compute backend-services list
gcloud compute backend-services get-health ytempire-backend-service --global

echo -e "\n=== SSL Certificate Status ==="
kubectl get certificates -A
kubectl describe certificates -A | grep -A 3 "Status:"

echo -e "\n=== Ingress Controller Logs ==="
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx --tail=50
```

---

## Performance Optimization

### 1. Network Performance Tuning

```yaml
# performance/network-optimization.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: network-tuning
  namespace: kube-system
data:
  tune.sh: |
    #!/bin/bash
    # TCP optimization
    sysctl -w net.core.rmem_max=134217728
    sysctl -w net.core.wmem_max=134217728
    sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
    sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
    sysctl -w net.ipv4.tcp_congestion_control=bbr
    sysctl -w net.core.default_qdisc=fq
    
    # Connection tracking
    sysctl -w net.netfilter.nf_conntrack_max=1048576
    sysctl -w net.nf_conntrack_max=1048576
    
    # Kubernetes specific
    sysctl -w net.ipv4.ip_forward=1
    sysctl -w net.bridge.bridge-nf-call-iptables=1
```

### 2. Resource Optimization

```python
#!/usr/bin/env python3
# optimize-resources.py

import subprocess
import json
import yaml

def get_resource_recommendations():
    """Get VPA recommendations for all deployments"""
    cmd = "kubectl get vpa -A -o json"
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    vpas = json.loads(result.stdout)
    
    recommendations = {}
    for vpa in vpas.get('items', []):
        namespace = vpa['metadata']['namespace']
        name = vpa['metadata']['name']
        
        if 'recommendation' in vpa.get('status', {}):
            containers = vpa['status']['recommendation']['containerRecommendations']
            recommendations[f"{namespace}/{name}"] = containers
    
    return recommendations

def generate_optimization_report():
    """Generate resource optimization report"""
    recommendations = get_resource_recommendations()
    
    print("=== Resource Optimization Report ===\n")
    
    for deployment, containers in recommendations.items():
        print(f"Deployment: {deployment}")
        for container in containers:
            print(f"  Container: {container['containerName']}")
            print(f"    Current CPU: {container.get('lowerBound', {}).get('cpu', 'N/A')}")
            print(f"    Target CPU: {container.get('target', {}).get('cpu', 'N/A')}")
            print(f"    Current Memory: {container.get('lowerBound', {}).get('memory', 'N/A')}")
            print(f"    Target Memory: {container.get('target', {}).get('memory', 'N/A')}")
            print()

if __name__ == "__main__":
    generate_optimization_report()
```

---

## Disaster Recovery

### 1. Network Failover

```yaml
# dr/multi-region-failover.yaml
apiVersion: networking.istio.io/v1beta1
kind: ServiceEntry
metadata:
  name: ytempire-api-external
  namespace: ytempire-core
spec:
  hosts:
  - ytempire-api-external
  ports:
  - number: 443
    name: https
    protocol: HTTPS
  location: MESH_EXTERNAL
  resolution: DNS
  endpoints:
  # Primary region
  - address: api-us-central1.ytempire.com
    priority: 0
    weight: 100
  # Failover region
  - address: api-us-east1.ytempire.com
    priority: 1
    weight: 100

---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: ytempire-api-external-dr
  namespace: ytempire-core
spec:
  host: ytempire-api-external
  trafficPolicy:
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http2MaxRequests: 100
        maxRequestsPerConnection: 2
```

### 2. Environment Backup and Restore

```bash
#!/bin/bash
# backup-environment.sh

BACKUP_DIR="/backup/$(date +%Y%m%d-%H%M%S)"
mkdir -p $BACKUP_DIR

echo "Backing up Kubernetes resources..."

# Backup all resources
for resource in $(kubectl api-resources --verbs=list -o name | grep -v "events.events.k8s.io" | grep -v "events" | sort | uniq); do
  echo "Backing up $resource"
  kubectl get $resource --all-namespaces -o yaml > $BACKUP_DIR/${resource}.yaml 2>/dev/null
done

# Backup Helm releases
echo "Backing up Helm releases..."
helm list -A -o json > $BACKUP_DIR/helm-releases.json

# Backup persistent volumes
echo "Backing up PV data..."
kubectl get pv -o yaml > $BACKUP_DIR/persistent-volumes.yaml

# Create restore script
cat > $BACKUP_DIR/restore.sh << 'EOF'
#!/bin/bash
echo "Restoring Kubernetes resources..."
kubectl apply -f persistent-volumes.yaml
kubectl apply -f namespaces.yaml
kubectl apply -f .
EOF

chmod +x $BACKUP_DIR/restore.sh
echo "Backup completed: $BACKUP_DIR"
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

**Note**: This document contains critical infrastructure configurations. Always test changes in staging before applying to production. Ensure proper change management procedures are followed.