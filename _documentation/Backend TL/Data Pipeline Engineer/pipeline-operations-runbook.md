# Data Pipeline Operations Runbook & Troubleshooting Guide

**Document Version**: 3.0 (Consolidated)  
**Date**: January 2025  
**Audience**: Data Pipeline Engineer, On-Call Support  
**Critical**: Keep this document updated with every incident

---

## Table of Contents
1. [System Overview & Key Metrics](#1-system-overview--key-metrics)
2. [Daily Operations Procedures](#2-daily-operations-procedures)
3. [Common Issues & Solutions](#3-common-issues--solutions)
4. [Emergency Procedures](#4-emergency-procedures)
5. [Performance Tuning Guide](#5-performance-tuning-guide)
6. [Disaster Recovery Procedures](#6-disaster-recovery-procedures)

---

## 1. System Overview & Key Metrics

### Critical System Components

```
Component         | Location              | Port  | Critical?
------------------|----------------------|-------|----------
PostgreSQL        | localhost            | 5432  | YES
Redis             | localhost            | 6379  | YES
Pipeline API      | localhost            | 8000  | YES
Prometheus        | localhost            | 9090  | NO
Grafana          | localhost            | 3000  | NO
N8N              | localhost            | 5678  | YES
```

### Key Performance Indicators

| Metric | Normal Range | Warning | Critical | Action |
|--------|--------------|---------|----------|--------|
| Queue Depth | 0-50 | 50-80 | >80 | Scale workers |
| Processing Time | <8 min | 8-10 min | >10 min | Investigate bottleneck |
| Cost per Video | <$2.50 | $2.50-2.80 | >$3.00 | HALT processing |
| GPU Utilization | 60-85% | 85-95% | >95% | Queue throttling |
| Error Rate | <5% | 5-10% | >10% | Debug immediately |
| API Response | <500ms | 500ms-1s | >1s | Check database |

### Critical Thresholds

```python
CRITICAL_THRESHOLDS = {
    "cost_per_video": 3.00,          # Hard stop
    "processing_time_seconds": 600,   # 10 minutes
    "queue_depth": 100,              # Max capacity
    "gpu_memory_mb": 30720,          # 30GB of 32GB
    "error_rate_percent": 10,        # Investigate
    "disk_usage_percent": 80         # Clean up
}
```

---

## 2. Daily Operations Procedures

### Morning Checklist (9:00 AM)

```bash
#!/bin/bash
# morning_check.sh

echo "=== YTEMPIRE Pipeline Morning Check ==="

# 1. Check system health
curl -s http://localhost:8000/health/detailed | jq '.'

# 2. Check queue status
psql -U ytempire -c "SELECT status, COUNT(*) FROM video_queue 
                     WHERE created_at > NOW() - INTERVAL '24 hours' 
                     GROUP BY status;"

# 3. Check yesterday's metrics
curl -s http://localhost:8000/metrics/daily | jq '.'

# 4. Check disk usage
df -h | grep -E "nvme|ssd|hdd"

# 5. Check GPU status
nvidia-smi

# 6. Check for stuck jobs
psql -U ytempire -c "SELECT id, processing_started_at 
                     FROM video_queue 
                     WHERE status = 'processing' 
                     AND processing_started_at < NOW() - INTERVAL '30 minutes';"

echo "=== Check Complete ==="
```

### Hourly Monitoring Tasks

```python
# hourly_monitor.py
async def hourly_checks():
    """Run every hour via cron"""
    
    # 1. Cost verification
    daily_cost = await get_daily_cost()
    if daily_cost > 150:  # $150 daily budget
        send_alert("Daily cost exceeding budget", daily_cost)
    
    # 2. Queue health
    queue_metrics = await get_queue_metrics()
    if queue_metrics['depth'] > 80:
        send_warning("Queue backing up", queue_metrics)
    
    # 3. Success rate
    success_rate = await calculate_success_rate()
    if success_rate < 90:
        send_alert("Success rate dropping", success_rate)
    
    # 4. Resource check
    resources = await check_resources()
    if resources['gpu_memory_free'] < 4096:  # 4GB minimum
        await trigger_gpu_cleanup()
```

### End of Day Procedures (6:00 PM)

```bash
#!/bin/bash
# evening_wrap.sh

# 1. Generate daily report
python3 generate_daily_report.py

# 2. Backup critical data
pg_dump ytempire > /backups/ytempire_$(date +%Y%m%d).sql
redis-cli BGSAVE

# 3. Clean up old files
find /tmp -name "*.mp4" -mtime +1 -delete
find /logs -name "*.log" -mtime +7 -delete

# 4. Optimize database
psql -U ytempire -c "VACUUM ANALYZE video_queue;"
psql -U ytempire -c "VACUUM ANALYZE pipeline_costs;"

# 5. Send daily summary
python3 send_daily_summary.py
```

---

## 3. Common Issues & Solutions

### Issue: Queue Not Processing

**Symptoms:**
- Queue depth increasing
- No videos completing
- Workers appear idle

**Diagnosis:**
```bash
# Check worker status
redis-cli SMEMBERS queue:processing

# Check for database locks
psql -U ytempire -c "SELECT * FROM pg_locks WHERE granted = false;"

# Check Redis connectivity
redis-cli ping
```

**Solution:**
```python
# restart_queue.py
async def restart_queue_processing():
    # 1. Clear stuck processing jobs
    processing = await redis_client.smembers("queue:processing")
    for video_id in processing:
        # Verify if actually processing
        status = await check_video_status(video_id)
        if status != 'processing':
            await redis_client.srem("queue:processing", video_id)
    
    # 2. Reset stuck database jobs
    await db.execute("""
        UPDATE video_queue 
        SET status = 'queued', processing_started_at = NULL
        WHERE status = 'processing' 
        AND processing_started_at < NOW() - INTERVAL '30 minutes'
    """)
    
    # 3. Restart workers
    os.system("supervisorctl restart pipeline-workers")
```

### Issue: High Cost Per Video

**Symptoms:**
- Cost exceeds $2.50 per video
- Cost alerts triggering
- Budget concerns

**Diagnosis:**
```sql
-- Identify expensive operations
SELECT service, AVG(amount) as avg_cost, COUNT(*) as count
FROM pipeline_costs
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY service
ORDER BY avg_cost DESC;
```

**Solution:**
```python
# optimize_costs.py
async def emergency_cost_optimization():
    # 1. Switch to cheaper models
    config.openai_model = "gpt-3.5-turbo"  # Instead of gpt-4
    
    # 2. Enable aggressive caching
    config.cache_ttl = 3600  # 1 hour cache
    
    # 3. Batch operations
    config.batch_size = 10  # Process in batches
    
    # 4. Reduce quality settings
    config.video_quality = "720p"  # Instead of 1080p
    config.audio_bitrate = "128k"  # Instead of 192k
    
    # 5. Implement rate limiting
    config.max_concurrent = 3  # Reduce from 7
```

### Issue: GPU Out of Memory

**Symptoms:**
- CUDA OOM errors
- GPU utilization at 100%
- Videos failing to render

**Diagnosis:**
```bash
# Check GPU memory
nvidia-smi

# Check allocations
curl http://localhost:8000/resources/gpu/status

# Find memory leaks
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

**Solution:**
```python
# gpu_recovery.py
async def recover_gpu_memory():
    # 1. Clear CUDA cache
    import torch
    torch.cuda.empty_cache()
    
    # 2. Kill stuck processes
    os.system("fuser -k /dev/nvidia0")
    
    # 3. Reset allocations
    await resource_scheduler.clear_all_gpu_allocations()
    
    # 4. Reduce concurrent GPU jobs
    resource_scheduler.gpu_semaphore = asyncio.Semaphore(2)
    
    # 5. Force garbage collection
    import gc
    gc.collect()
```

### Issue: Slow Processing Times

**Symptoms:**
- Videos taking >10 minutes
- Pipeline stages timing out
- Queue backing up

**Diagnosis:**
```python
# identify_bottleneck.py
async def find_bottleneck():
    # Get stage timings
    stage_times = await db.fetch("""
        SELECT stage, 
               AVG(duration) as avg_duration,
               MAX(duration) as max_duration
        FROM pipeline_stage_times
        WHERE timestamp > NOW() - INTERVAL '1 hour'
        GROUP BY stage
        ORDER BY avg_duration DESC
    """)
    
    print("Slowest stages:", stage_times[:3])
```

**Solution:**
```python
# optimize_pipeline.py
async def speed_optimization():
    # 1. Enable parallel processing
    config.parallel_stages = ["media_collection", "audio_synthesis"]
    
    # 2. Reduce timeouts
    config.stage_timeouts = {
        "script_generation": 30,  # From 60
        "audio_synthesis": 60,    # From 120
        "video_rendering": 180    # From 300
    }
    
    # 3. Skip non-essential stages
    config.skip_quality_check = True  # Temporary
    
    # 4. Use faster services
    config.tts_service = "google"  # Faster than elevenlabs
```

### Issue: Database Connection Exhaustion

**Symptoms:**
- "Too many connections" errors
- Slow queries
- API timeouts

**Solution:**
```python
# fix_db_connections.py
async def fix_database_connections():
    # 1. Kill idle connections
    await db.execute("""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE state = 'idle' 
        AND state_change < NOW() - INTERVAL '10 minutes'
    """)
    
    # 2. Adjust pool settings
    db_pool = await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=5,
        max_size=15,  # Reduced from 20
        max_inactive_connection_lifetime=300
    )
    
    # 3. Enable connection pooling
    # pgbouncer configuration
```

---

## 4. Emergency Procedures

### EMERGENCY: Cost Overrun (>$3/video)

**IMMEDIATE ACTIONS:**
```bash
# 1. HALT ALL PROCESSING
curl -X POST http://localhost:8000/pipeline/emergency/halt

# 2. Check current costs
psql -U ytempire -c "SELECT video_id, SUM(amount) as total 
                     FROM pipeline_costs 
                     WHERE timestamp > NOW() - INTERVAL '1 hour' 
                     GROUP BY video_id 
                     HAVING SUM(amount) > 3.0;"

# 3. Notify team
./send_critical_alert.sh "COST OVERRUN DETECTED"
```

**Recovery Steps:**
1. Identify expensive operations
2. Disable premium features
3. Switch to economy mode
4. Clear queue of complex videos
5. Resume with cost monitoring

### EMERGENCY: Complete Pipeline Failure

**IMMEDIATE ACTIONS:**
```bash
#!/bin/bash
# emergency_restart.sh

echo "EMERGENCY PIPELINE RESTART INITIATED"

# 1. Stop all services
supervisorctl stop all

# 2. Clear Redis
redis-cli FLUSHDB

# 3. Reset database state
psql -U ytempire -c "UPDATE video_queue 
                     SET status = 'queued' 
                     WHERE status = 'processing';"

# 4. Restart services in order
systemctl restart postgresql
systemctl restart redis
supervisorctl start pipeline-api
supervisorctl start pipeline-workers

# 5. Verify health
sleep 10
curl http://localhost:8000/health

echo "RESTART COMPLETE"
```

### EMERGENCY: Data Corruption

**Detection:**
```sql
-- Check for data inconsistencies
SELECT COUNT(*) FROM video_queue 
WHERE status = 'completed' 
AND completed_at IS NULL;

SELECT COUNT(*) FROM pipeline_costs 
WHERE amount < 0 OR amount > 10;
```

**Recovery:**
```bash
#!/bin/bash
# data_recovery.sh

# 1. Stop processing
curl -X POST http://localhost:8000/pipeline/pause

# 2. Backup current state
pg_dump ytempire > /emergency/backup_$(date +%s).sql

# 3. Run integrity checks
python3 data_integrity_check.py

# 4. Fix inconsistencies
psql -U ytempire -f fix_data_inconsistencies.sql

# 5. Restore from last known good if needed
# psql -U ytempire < /backups/last_known_good.sql

# 6. Resume processing
curl -X POST http://localhost:8000/pipeline/resume
```

---

## 5. Performance Tuning Guide

### Database Optimization

```sql
-- Key indexes (verify these exist)
CREATE INDEX CONCURRENTLY idx_queue_processing 
ON video_queue(status, priority DESC, created_at ASC) 
WHERE status IN ('queued', 'processing');

CREATE INDEX CONCURRENTLY idx_costs_recent 
ON pipeline_costs(timestamp DESC) 
WHERE timestamp > NOW() - INTERVAL '7 days';

-- Maintenance queries (run weekly)
VACUUM ANALYZE video_queue;
REINDEX CONCURRENTLY idx_queue_processing;

-- Connection pool tuning
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
```

### Redis Optimization

```bash
# redis.conf optimizations
maxmemory 8gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### Python Performance

```python
# performance_config.py
import asyncio
import uvloop

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Connection pool settings
DB_POOL_MIN = 5
DB_POOL_MAX = 20
REDIS_POOL_SIZE = 10

# Batch processing settings
BATCH_SIZE = 10
PARALLEL_WORKERS = 4

# Caching settings
CACHE_TTL = 3600  # 1 hour
CACHE_MAX_SIZE = 1000
```

### GPU Optimization

```python
# gpu_optimization.py
import os

# CUDA settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# FFmpeg GPU settings
FFMPEG_GPU_PARAMS = [
    '-hwaccel', 'cuda',
    '-hwaccel_output_format', 'cuda',
    '-c:v', 'h264_nvenc',
    '-preset', 'p7',  # Fastest preset
    '-rc', 'vbr',
    '-cq', '23',     # Quality level
    '-b:v', '4M',
    '-maxrate', '6M'
]
```

---

## 6. Disaster Recovery Procedures

### Backup Strategy

```bash
#!/bin/bash
# backup_strategy.sh

# Hourly backups (keep 24)
0 * * * * pg_dump ytempire | gzip > /backups/hourly/ytempire_$(date +\%Y\%m\%d_\%H).sql.gz
0 * * * * find /backups/hourly -mtime +1 -delete

# Daily backups (keep 7)
0 2 * * * pg_dump ytempire | gzip > /backups/daily/ytempire_$(date +\%Y\%m\%d).sql.gz
0 2 * * * find /backups/daily -mtime +7 -delete

# Weekly backups (keep 4)
0 3 * * 0 pg_dump ytempire | gzip > /backups/weekly/ytempire_$(date +\%Y\%m\%d).sql.gz
0 3 * * 0 find /backups/weekly -mtime +28 -delete

# Redis backups
*/30 * * * * redis-cli BGSAVE
```

### Recovery Time Objectives

| Scenario | RTO | RPO | Procedure |
|----------|-----|-----|-----------|
| Service Crash | 5 min | 0 | Automatic restart |
| Database Corruption | 1 hour | 1 hour | Restore from backup |
| Server Failure | 4 hours | 1 hour | Failover to backup |
| Complete Loss | 24 hours | 24 hours | Full rebuild |

### Full System Recovery

```bash
#!/bin/bash
# full_recovery.sh

echo "Starting full system recovery..."

# 1. Install dependencies
apt-get update
apt-get install -y postgresql redis python3.11

# 2. Restore database
psql -U postgres -c "CREATE DATABASE ytempire;"
gunzip -c /backups/latest.sql.gz | psql -U postgres ytempire

# 3. Restore Redis
redis-cli --rdb /backups/dump.rdb

# 4. Restore application
git clone https://github.com/ytempire/pipeline.git
cd pipeline
pip install -r requirements.txt

# 5. Restore configuration
cp /backups/config/.env .env
cp /backups/config/settings.yaml config/

# 6. Start services
systemctl start postgresql redis
python3 main.py

echo "Recovery complete!"
```

### Post-Recovery Validation

```python
# validate_recovery.py
async def validate_recovery():
    checks = {
        "database": check_database_connection(),
        "redis": check_redis_connection(),
        "api": check_api_health(),
        "queue": check_queue_processing(),
        "gpu": check_gpu_availability()
    }
    
    results = await asyncio.gather(*checks.values())
    
    for name, result in zip(checks.keys(), results):
        print(f"{name}: {'✓' if result else '✗'}")
    
    if all(results):
        print("System fully recovered!")
        return True
    else:
        print("Recovery incomplete - manual intervention required")
        return False
```

---

## Critical Contact Information

### Escalation Path

1. **Level 1 (0-15 min):** On-call Data Pipeline Engineer
2. **Level 2 (15-30 min):** Backend Team Lead
3. **Level 3 (30-60 min):** CTO/Technical Director
4. **Level 4 (60+ min):** CEO/Founder

### Emergency Contacts

| Role | Name | Contact | When to Call |
|------|------|---------|--------------|
| On-Call Engineer | Rotation | Check PagerDuty | First contact |
| Backend Lead | TBD | Slack/Phone | Architecture issues |
| Platform Ops | TBD | Slack/Phone | Infrastructure |
| CTO | TBD | Phone | Critical decisions |

### Vendor Support

| Service | Support | Priority | Account |
|---------|---------|----------|---------|
| OpenAI | support@openai.com | Enterprise | #12345 |
| Google Cloud | 1-800-XXX | Gold | ytempire-prod |
| Pexels API | api@pexels.com | Pro | api_key_xxx |

---

## Appendix: Quick Commands

```bash
# Most used commands
alias queuedepth='redis-cli zcard queue:priority'
alias processing='redis-cli smembers queue:processing'
alias todaycount='psql -U ytempire -c "SELECT COUNT(*) FROM video_queue WHERE created_at > CURRENT_DATE;"'
alias todaycost='psql -U ytempire -c "SELECT SUM(amount) FROM pipeline_costs WHERE timestamp > CURRENT_DATE;"'
alias gpucheck='nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu --format=csv'
alias pipelinelogs='tail -f /var/log/pipeline/pipeline.log'
alias restartpipeline='supervisorctl restart pipeline-workers'
alias emergency='curl -X POST http://localhost:8000/pipeline/emergency/halt'
```

---

**Remember:** This runbook is a living document. Update it after every incident, optimization, or system change. Your future self (at 3 AM during an outage) will thank you!