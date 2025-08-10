# YTEmpire Monitoring Stack
**Owner: Platform Ops Lead**

## Overview

This monitoring stack provides comprehensive observability for the YTEmpire platform using Prometheus for metrics collection and Grafana for visualization.

## Components

### Prometheus (Port: 9090)
- **Purpose**: Metrics collection and alerting
- **URL**: http://localhost:9090
- **Configuration**: `infrastructure/prometheus/prometheus.yml`
- **Alert Rules**: `infrastructure/prometheus/alerts.yml`

### Grafana (Port: 3001)
- **Purpose**: Metrics visualization and dashboards
- **URL**: http://localhost:3001
- **Credentials**: admin/ytempire_grafana
- **Dashboards**: `infrastructure/grafana/dashboards/`

## Deployment

### Quick Start
```bash
# Deploy monitoring stack
./infrastructure/scripts/deploy-monitoring.sh

# Or manually with Docker Compose
docker-compose up -d prometheus grafana
```

### Manual Configuration
1. Start services: `docker-compose up -d prometheus grafana`
2. Wait 30 seconds for services to initialize
3. Access Grafana at http://localhost:3001
4. Login with admin/ytempire_grafana
5. Dashboards should be automatically loaded

## Metrics Collected

### HTTP Metrics
- `http_requests_total` - Total HTTP requests by method, endpoint, status
- `http_request_duration_seconds` - Request duration histogram
- `http_requests_in_progress` - Current requests in progress

### Authentication Metrics
- `ytempire_auth_attempts_total` - Authentication attempts (success/failed)
- `ytempire_jwt_tokens_issued_total` - JWT tokens issued by type
- `ytempire_rate_limit_hits_total` - Rate limit violations by IP

### Application Metrics
- `ytempire_active_users_total` - Current active users
- `ytempire_videos_generated_total` - Videos generated per user/channel
- `ytempire_videos_uploaded_total` - Videos uploaded to YouTube
- `ytempire_api_costs_total` - API costs by service and user

### Infrastructure Metrics
- `ytempire_database_connections` - Current database connections
- `ytempire_youtube_quota_usage` - YouTube API quota consumption
- `ytempire_celery_tasks_total` - Celery task execution metrics

## Health Endpoints

### Backend Health Checks
- `/health` - Comprehensive health check with dependency status
- `/health/ready` - Kubernetes readiness probe
- `/health/live` - Kubernetes liveness probe
- `/metrics` - Prometheus metrics endpoint

### Service Health Monitoring
All services expose health endpoints that are monitored:
- **Backend**: http://localhost:8000/health
- **PostgreSQL**: Monitored via connection checks
- **Redis**: Monitored via ping checks
- **Celery**: Monitored via Flower at http://localhost:5555

## Alerting Rules

### Critical Alerts
1. **ServiceDown**: Any monitored service is unreachable (1min threshold)
2. **DatabaseDown**: PostgreSQL unavailable (1min threshold)
3. **RedisDown**: Redis unavailable (1min threshold)

### Warning Alerts  
1. **HighErrorRate**: HTTP error rate > 10% (5min threshold)
2. **HighResponseTime**: 95th percentile > 2s (5min threshold)
3. **HighMemoryUsage**: Memory usage > 1GB (10min threshold)
4. **NoActiveWorkers**: No Celery workers active (5min threshold)
5. **HighTaskQueueLength**: >100 pending Celery tasks (10min threshold)

## Dashboards

### YTEmpire Backend Dashboard
- **File**: `ytempire-backend.json`
- **Panels**:
  - HTTP Request Rate (requests/second)
  - Response Time (95th/50th percentiles)
  - Service Health Status
  - Error Rate Tracking

## Troubleshooting

### Common Issues

1. **Grafana Shows "No Data"**
   ```bash
   # Check if Prometheus is scraping
   curl http://localhost:9090/api/v1/targets
   
   # Check backend metrics endpoint
   curl http://localhost:8000/metrics
   ```

2. **Prometheus Can't Scrape Backend**
   - Ensure backend service is running on port 8000
   - Check network connectivity between containers
   - Verify prometheus.yml configuration

3. **High Memory Usage Alerts**
   - Check application for memory leaks
   - Monitor database connection pooling
   - Review Celery task memory usage

4. **Rate Limiting Issues**
   - Monitor client IP patterns
   - Adjust rate limits in middleware
   - Check for DDoS patterns

### Log Locations
- **Prometheus**: Docker logs via `docker logs ytempire-prometheus`
- **Grafana**: Docker logs via `docker logs ytempire-grafana`
- **Backend**: Application logs + metrics in stdout

## Performance Tuning

### Prometheus Retention
```yaml
# prometheus.yml - adjust retention policy
global:
  scrape_interval: 15s     # Default scraping interval
  evaluation_interval: 15s # Rule evaluation interval
```

### Grafana Performance
- Enable caching for dashboards
- Use appropriate time ranges
- Optimize dashboard queries
- Set reasonable refresh intervals

## Security Considerations

1. **Authentication**: Grafana uses basic auth (change default password)
2. **Network**: Monitor services only accessible internally
3. **Metrics**: Avoid exposing sensitive data in metrics labels
4. **Alerts**: Configure secure alert channels (email/Slack)

## Maintenance

### Regular Tasks
1. **Weekly**: Review alert thresholds and tune as needed
2. **Monthly**: Check disk usage for metrics storage
3. **Quarterly**: Update dashboard queries and add new metrics
4. **Yearly**: Review retention policies and archive old data

### Backup Strategy
- Grafana dashboards are stored in git (infrastructure/grafana/)
- Prometheus data is ephemeral (stored in Docker volume)
- Alert rules are stored in git (infrastructure/prometheus/)

## Adding New Metrics

1. **Define Metric**: Add to `backend/app/core/metrics.py`
2. **Collect Data**: Use `metrics.record_*()` methods in application code
3. **Create Dashboard**: Add panels to Grafana dashboard
4. **Set Alerts**: Add alert rules in `prometheus/alerts.yml`

## Integration with CI/CD

The monitoring stack integrates with the development workflow:
- Health checks used in deployment validation
- Metrics help identify performance regressions
- Alerts notify of issues in staging/production
- Dashboards provide visibility during deployments