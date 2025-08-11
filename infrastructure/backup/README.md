# YTEmpire Disaster Recovery System

Comprehensive backup and disaster recovery solution for the YTEmpire MVP platform.

## Overview

This disaster recovery system provides:
- **Automated Backups**: Full and incremental backups of PostgreSQL, Redis, and file systems
- **High Availability**: PostgreSQL streaming replication and Redis Sentinel
- **Monitoring**: Comprehensive health monitoring and alerting
- **Recovery**: Point-in-time recovery capabilities
- **Storage**: Local and S3-compatible cloud storage options

## Architecture

### Components
- **Primary Database**: PostgreSQL 15 with streaming replication
- **Replica Database**: Hot standby for read scaling and backup
- **Cache Layer**: Redis with persistence and Sentinel for HA
- **Backup Manager**: Automated backup orchestration
- **Health Monitor**: System health monitoring and alerting
- **Storage**: MinIO (S3-compatible) and AWS S3 support
- **Monitoring Stack**: Prometheus, Grafana, AlertManager, Loki

### Recovery Objectives
- **RTO (Recovery Time Objective)**: 4 hours
- **RPO (Recovery Point Objective)**: 1 hour
- **Availability Target**: 99.9% uptime

## Quick Start

### 1. Initial Setup
```bash
# Run the disaster recovery setup script
cd scripts/disaster-recovery
./dr-setup.sh

# Review and update environment variables
vim .env.disaster-recovery
```

### 2. Start Services
```bash
# Start the disaster recovery stack
docker-compose -f docker-compose.disaster-recovery.yml up -d

# Verify services are running
docker-compose -f docker-compose.disaster-recovery.yml ps
```

### 3. Monitor Health
```bash
# Check overall system health
docker-compose -f docker-compose.disaster-recovery.yml exec health_monitor \
  python /app/monitoring/health_check.py

# Access monitoring dashboards
# Grafana: http://localhost:3001
# Prometheus: http://localhost:9090
```

## Backup Operations

### Manual Backup
```bash
# Create full backup
docker-compose -f docker-compose.disaster-recovery.yml exec backup_manager \
  python /app/backup/backup_manager.py backup --type full

# Create incremental backup  
docker-compose -f docker-compose.disaster-recovery.yml exec backup_manager \
  python /app/backup/backup_manager.py backup --type incremental
```

### Automated Backups
Backups run automatically on schedule:
- **Full Backups**: Every 24 hours (configurable)
- **Incremental Backups**: Every 4 hours (configurable)
- **Retention**: 7 days local, 90 days in S3

### Backup Types

#### Full Backup
- Complete database dump
- Redis snapshot
- File system archive
- Configuration backup

#### Incremental Backup
- PostgreSQL WAL files
- Redis AOF changes
- Changed files only

## Restore Operations

### Full System Restore
```bash
# Stop services
docker-compose -f docker-compose.disaster-recovery.yml stop

# Restore from backup
./scripts/restore-backup.sh full_20240101_120000

# Verify restoration
docker-compose -f docker-compose.disaster-recovery.yml exec postgres_primary \
  psql -U postgres -d ytempire -c "SELECT count(*) FROM users;"
```

### Point-in-Time Recovery
```bash
# Restore to specific timestamp
docker-compose -f docker-compose.disaster-recovery.yml exec backup_manager \
  python /app/backup/backup_manager.py restore \
  --backup-id full_20240101_120000 \
  --target-time "2024-01-01 15:30:00"
```

### Component-Specific Restore

#### Database Only
```bash
# Restore database from backup file
docker-compose -f docker-compose.disaster-recovery.yml exec postgres_primary \
  pg_restore -U postgres -d ytempire /var/backups/ytempire/backup_file.sql
```

#### Redis Only
```bash
# Restore Redis from RDB file
docker-compose -f docker-compose.disaster-recovery.yml exec redis_primary \
  redis-cli -p 6379 --rdb /data/backup.rdb
```

## Monitoring and Alerting

### Health Checks
The system performs continuous health monitoring:
- Database connectivity and performance
- Redis availability and memory usage
- API endpoints responsiveness
- Storage space and accessibility
- System resources (CPU, memory, disk)

### Monitoring Dashboards
- **Grafana**: Comprehensive dashboards at http://localhost:3001
- **Prometheus**: Metrics collection at http://localhost:9090
- **AlertManager**: Alert management at http://localhost:9093

### Alerting Rules
Alerts are triggered for:
- Service unavailability
- High resource usage (>85% memory, >90% disk)
- Backup failures
- Replication lag (>60 seconds)
- API response time degradation

## Configuration

### Environment Variables
Key configuration options in `.env.disaster-recovery`:

```bash
# Database
DATABASE_PASSWORD=secure_password
REPLICATION_PASSWORD=replication_password

# Redis
REDIS_PASSWORD=redis_password

# AWS S3 (optional)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
BACKUP_S3_BUCKET=your-backup-bucket

# Backup Settings
BACKUP_ENCRYPTION_KEY=encryption_key

# Monitoring
GRAFANA_PASSWORD=grafana_password
ALERT_EMAIL=admin@yourdomain.com
```

### Backup Configuration
Customize backup behavior in `backup_manager.py`:

```python
@dataclass
class BackupConfig:
    # Retention settings
    local_retention_days: int = 7
    s3_retention_days: int = 90
    
    # Schedule settings
    full_backup_interval_hours: int = 24
    incremental_backup_interval_hours: int = 4
```

## Disaster Scenarios

### Database Server Failure
1. **Automatic**: Replica promotes to primary via Sentinel
2. **Manual**: Promote replica using trigger file
3. **Recovery**: Restore primary from backup when available

### Complete Infrastructure Loss
1. **Deploy**: Set up new infrastructure
2. **Restore**: Download latest backup from S3
3. **Recover**: Restore all components from backup
4. **Verify**: Run health checks and validation

### Data Corruption
1. **Identify**: Use monitoring to detect corruption
2. **Isolate**: Stop writes to affected services
3. **Restore**: Point-in-time recovery to before corruption
4. **Validate**: Verify data integrity post-restore

## Security

### Encryption
- **At Rest**: Database tablespaces and backup files encrypted
- **In Transit**: TLS for all network communications
- **Backup**: AES-256 encryption for backup files

### Access Control
- **Database**: Role-based access with limited permissions
- **Backups**: Separate service account with minimal privileges
- **Monitoring**: Authentication required for dashboard access

### Network Security
- **Isolation**: Services run in isolated Docker network
- **Firewall**: Only necessary ports exposed
- **VPN**: Production access through secure VPN

## Troubleshooting

### Common Issues

#### Backup Failures
```bash
# Check backup logs
docker-compose -f docker-compose.disaster-recovery.yml logs backup_manager

# Test database connection
docker-compose -f docker-compose.disaster-recovery.yml exec backup_manager \
  python -c "import psycopg2; psycopg2.connect(host='postgres_primary', user='postgres')"
```

#### Replication Lag
```bash
# Check replication status
docker-compose -f docker-compose.disaster-recovery.yml exec postgres_primary \
  psql -U postgres -c "SELECT * FROM pg_stat_replication;"

# Check replica status
docker-compose -f docker-compose.disaster-recovery.yml exec postgres_replica \
  psql -U postgres -c "SELECT pg_is_in_recovery(), pg_last_wal_receive_lsn();"
```

#### Disk Space Issues
```bash
# Check disk usage
df -h

# Clean old backups
docker-compose -f docker-compose.disaster-recovery.yml exec backup_manager \
  find /var/backups/ytempire -name "*.gz" -mtime +7 -delete
```

### Log Locations
- **Application Logs**: `/var/log/ytempire/`
- **Container Logs**: `docker-compose logs <service_name>`
- **Backup Logs**: `/var/backups/ytempire/logs/`
- **System Logs**: `/var/log/messages` or journalctl

## Testing

### Backup Testing
```bash
# Test backup creation
docker-compose -f docker-compose.disaster-recovery.yml exec backup_manager \
  python /app/backup/backup_manager.py backup --type full --test

# Validate backup integrity
docker-compose -f docker-compose.disaster-recovery.yml exec backup_manager \
  python /app/backup/backup_manager.py validate --backup-id <backup_id>
```

### Recovery Testing
```bash
# Test restore in isolated environment
docker-compose -f docker-compose.test.yml up -d
./scripts/test-restore.sh <backup_id>
```

### Monitoring Testing
```bash
# Test health checks
docker-compose -f docker-compose.disaster-recovery.yml exec health_monitor \
  python /app/monitoring/health_check.py --component database

# Test alerting
curl -X POST http://localhost:9093/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '[{"labels":{"alertname":"TestAlert","severity":"critical"}}]'
```

## Performance Optimization

### Database Tuning
- **Connection Pooling**: Use pgBouncer for connection management
- **Query Optimization**: Monitor slow queries and add indexes
- **Memory Settings**: Tune shared_buffers and effective_cache_size

### Backup Optimization
- **Compression**: Use pg_dump compression for smaller backups
- **Parallel Processing**: Enable parallel backup processes
- **Network**: Use dedicated backup network for large transfers

### Monitoring Optimization
- **Metrics Retention**: Configure appropriate retention periods
- **Sampling**: Use recording rules for expensive queries
- **Storage**: Use appropriate storage for metrics data

## Compliance

### Data Protection
- **GDPR**: Implement data anonymization for backups
- **Retention**: Automatic cleanup based on retention policies
- **Audit**: Comprehensive logging of all backup operations

### Industry Standards
- **ISO 27001**: Information security management
- **SOC 2**: Security and availability controls
- **PCI DSS**: Payment card data protection (if applicable)

## Support and Maintenance

### Regular Maintenance
- **Weekly**: Review backup success rates and storage usage
- **Monthly**: Test restore procedures and update documentation
- **Quarterly**: Review and update disaster recovery procedures

### Support Contacts
- **Primary**: DevOps Team (devops@ytempire.com)
- **Secondary**: Platform Engineering (platform@ytempire.com)
- **Emergency**: 24/7 On-call rotation

## Documentation
- **Runbooks**: Detailed operational procedures
- **Architecture**: System design and component interactions
- **Procedures**: Step-by-step recovery procedures
- **Training**: Disaster recovery training materials