# YTEMPIRE Analytics Engineer Documentation
## 5. OPERATIONAL PLAYBOOKS

**Document Version**: 1.0  
**Date**: January 2025  
**Status**: CONSOLIDATED - PRODUCTION READY  
**Purpose**: Complete operational procedures and playbooks

---

## 5.1 Daily Operations

### Morning Checklist

```bash
#!/bin/bash
# morning_checklist.sh
# Daily morning checklist for Analytics Engineer
# Run time: 8:00 AM daily

echo "========================================="
echo "YTEMPIRE Analytics Morning Checklist"
echo "Date: $(date)"
echo "========================================="

# 1. Check data freshness
echo -e "\n[1/8] Checking data freshness..."
psql -U analytics_user -d ytempire -c "
SELECT 
    'Video Performance' as dataset,
    MAX(timestamp) as last_update,
    EXTRACT(MINUTE FROM (NOW() - MAX(timestamp))) as minutes_ago,
    CASE 
        WHEN EXTRACT(MINUTE FROM (NOW() - MAX(timestamp))) > 60 THEN 'STALE'
        WHEN EXTRACT(MINUTE FROM (NOW() - MAX(timestamp))) > 15 THEN 'DELAYED'
        ELSE 'FRESH'
    END as status
FROM analytics.fact_video_performance
UNION ALL
SELECT 
    'Revenue Data',
    MAX(created_at),
    EXTRACT(MINUTE FROM (NOW() - MAX(created_at))),
    CASE 
        WHEN EXTRACT(MINUTE FROM (NOW() - MAX(created_at))) > 360 THEN 'STALE'
        WHEN EXTRACT(MINUTE FROM (NOW() - MAX(created_at))) > 60 THEN 'DELAYED'
        ELSE 'FRESH'
    END
FROM analytics.fact_revenue;"

# 2. Check overnight job status
echo -e "\n[2/8] Checking overnight jobs..."
psql -U analytics_user -d ytempire -c "
SELECT 
    job_name,
    status,
    execution_time_seconds,
    error_message
FROM analytics.job_execution_history
WHERE start_time >= CURRENT_DATE - INTERVAL '1 day'
    AND start_time < CURRENT_DATE + INTERVAL '6 hours'
ORDER BY start_time DESC;"

# 3. Check dashboard performance
echo -e "\n[3/8] Checking dashboard performance..."
psql -U analytics_user -d ytempire -c "
SELECT 
    dashboard_name,
    AVG(execution_time_ms) as avg_time_ms,
    MAX(execution_time_ms) as max_time_ms,
    COUNT(*) as query_count
FROM analytics.query_performance_log
WHERE timestamp >= CURRENT_DATE
GROUP BY dashboard_name
HAVING AVG(execution_time_ms) > 2000
ORDER BY avg_time_ms DESC;"

# 4. Check API quota usage
echo -e "\n[4/8] Checking YouTube API quota..."
psql -U analytics_user -d ytempire -c "
SELECT * FROM analytics.check_api_quota();"

# 5. Check cost tracking
echo -e "\n[5/8] Checking yesterday's costs..."
psql -U analytics_user -d ytempire -c "
SELECT 
    cost_category,
    COUNT(DISTINCT video_id) as videos,
    SUM(total_cost_cents) / 100.0 as total_cost,
    AVG(total_cost_cents) / 100.0 as avg_cost_per_video
FROM analytics.fact_costs
WHERE DATE(cost_timestamp) = CURRENT_DATE - 1
GROUP BY cost_category
UNION ALL
SELECT 
    'TOTAL',
    COUNT(DISTINCT video_id),
    SUM(total_cost_cents) / 100.0,
    SUM(total_cost_cents) / NULLIF(COUNT(DISTINCT video_id), 0) / 100.0
FROM analytics.fact_costs
WHERE DATE(cost_timestamp) = CURRENT_DATE - 1;"

# 6. Check data quality
echo -e "\n[6/8] Running data quality checks..."
psql -U analytics_user -d ytempire -c "
SELECT analytics.run_all_quality_checks();
SELECT 
    check_type,
    COUNT(*) FILTER (WHERE status = 'passed') as passed,
    COUNT(*) FILTER (WHERE status = 'warning') as warnings,
    COUNT(*) FILTER (WHERE status = 'failed') as failed
FROM analytics.data_quality_checks
WHERE check_timestamp >= CURRENT_DATE
GROUP BY check_type;"

# 7. Check system resources
echo -e "\n[7/8] Checking system resources..."
df -h | grep -E "Filesystem|nvme"
free -h
psql -U analytics_user -d ytempire -c "
SELECT 
    pg_database_size(current_database()) / (1024*1024*1024.0) as db_size_gb,
    (SELECT COUNT(*) FROM pg_stat_activity) as active_connections;"

# 8. Generate morning summary
echo -e "\n[8/8] Generating morning summary..."
psql -U analytics_user -d ytempire -t -c "
WITH summary AS (
    SELECT 
        (SELECT COUNT(*) FROM analytics.dim_channel WHERE is_active = true) as active_channels,
        (SELECT COUNT(*) FROM analytics.dim_video WHERE published_at >= CURRENT_DATE) as videos_today,
        (SELECT SUM(actual_revenue_cents)/100 FROM analytics.fact_revenue WHERE date_key = (SELECT date_key FROM analytics.dim_date WHERE full_date = CURRENT_DATE - 1)) as revenue_yesterday,
        (SELECT COUNT(DISTINCT user_id) FROM users.accounts WHERE is_active = true) as active_users
)
SELECT 
    'Morning Summary:' || E'\n' ||
    '- Active Channels: ' || active_channels || E'\n' ||
    '- Videos Today: ' || videos_today || E'\n' ||
    '- Revenue Yesterday: $' || COALESCE(revenue_yesterday, 0) || E'\n' ||
    '- Active Users: ' || active_users
FROM summary;"

echo -e "\n========================================="
echo "Morning checklist complete!"
echo "========================================="
```

### Monitoring Routine

```python
# monitoring/hourly_monitoring.py
"""
Hourly monitoring routine for Analytics Engineer
"""

import psycopg2
import redis
import requests
from datetime import datetime, timedelta
import json

class HourlyMonitor:
    def __init__(self):
        self.db = psycopg2.connect(
            host="localhost",
            database="ytempire",
            user="analytics_user",
            password="password"
        )
        self.redis = redis.Redis(host='localhost', port=6379)
        self.slack_webhook = "https://hooks.slack.com/services/YOUR_WEBHOOK"
        
    def run_hourly_checks(self):
        """Main hourly monitoring routine"""
        print(f"Running hourly checks at {datetime.now()}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # 1. Check real-time metrics
        results['checks']['realtime_metrics'] = self.check_realtime_metrics()
        
        # 2. Check cost thresholds
        results['checks']['cost_status'] = self.check_cost_thresholds()
        
        # 3. Check user progress
        results['checks']['user_progress'] = self.check_user_progress()
        
        # 4. Check trending videos
        results['checks']['trending'] = self.check_trending_content()
        
        # 5. Check cache performance
        results['checks']['cache'] = self.check_cache_performance()
        
        # Send alerts if needed
        self.process_alerts(results)
        
        # Log results
        self.log_monitoring_results(results)
        
        return results
    
    def check_realtime_metrics(self):
        """Check real-time performance metrics"""
        cursor = self.db.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT video_id) as videos_last_hour,
                SUM(views) as views_last_hour,
                AVG(engagement_rate) as avg_engagement,
                SUM(actual_revenue_cents) / 100.0 as revenue_last_hour
            FROM analytics.fact_video_performance
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
        """)
        
        result = cursor.fetchone()
        
        return {
            'videos_published': result[0],
            'views': result[1],
            'engagement_rate': float(result[2]) if result[2] else 0,
            'revenue': float(result[3]) if result[3] else 0,
            'status': 'OK' if result[0] > 0 else 'NO_ACTIVITY'
        }
    
    def check_cost_thresholds(self):
        """Check if costs are within thresholds"""
        cursor = self.db.cursor()
        
        cursor.execute("""
            WITH hourly_costs AS (
                SELECT 
                    DATE_TRUNC('hour', cost_timestamp) as hour,
                    SUM(total_cost_cents) / 100.0 as hourly_cost,
                    COUNT(DISTINCT video_id) as video_count
                FROM analytics.fact_costs
                WHERE cost_timestamp >= NOW() - INTERVAL '1 hour'
                GROUP BY DATE_TRUNC('hour', cost_timestamp)
            )
            SELECT 
                hourly_cost,
                video_count,
                CASE 
                    WHEN video_count > 0 THEN hourly_cost / video_count
                    ELSE 0
                END as cost_per_video
            FROM hourly_costs
        """)
        
        result = cursor.fetchone()
        
        if result:
            hourly_cost, video_count, cost_per_video = result
            
            status = 'OK'
            if cost_per_video > 3.0:
                status = 'OVER_BUDGET'
            elif cost_per_video > 2.5:
                status = 'WARNING'
            
            return {
                'hourly_cost': float(hourly_cost) if hourly_cost else 0,
                'videos_processed': video_count,
                'cost_per_video': float(cost_per_video) if cost_per_video else 0,
                'status': status
            }
        
        return {'status': 'NO_DATA'}
    
    def check_user_progress(self):
        """Check user progress toward revenue goals"""
        cursor = self.db.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_users,
                COUNT(*) FILTER (WHERE current_month_revenue >= 10000) as at_goal,
                COUNT(*) FILTER (WHERE current_month_revenue >= 5000) as halfway,
                AVG(goal_progress_pct) as avg_progress
            FROM analytics.v_user_revenue_tracking
            WHERE lifetime_revenue > 0
        """)
        
        result = cursor.fetchone()
        
        return {
            'total_users': result[0],
            'users_at_10k': result[1],
            'users_at_5k': result[2],
            'avg_progress_pct': float(result[3]) if result[3] else 0
        }
    
    def check_trending_content(self):
        """Identify trending videos"""
        cursor = self.db.cursor()
        
        cursor.execute("""
            SELECT 
                v.video_title,
                v.channel_id,
                vp.views,
                vp.engagement_rate,
                analytics.calculate_trending_score(
                    vp.views,
                    LAG(vp.views) OVER (PARTITION BY v.video_key ORDER BY vp.timestamp),
                    vp.engagement_rate,
                    EXTRACT(HOUR FROM (NOW() - v.published_at))
                ) as trending_score
            FROM analytics.fact_video_performance vp
            JOIN analytics.dim_video v ON vp.video_key = v.video_key
            WHERE vp.timestamp >= NOW() - INTERVAL '24 hours'
            ORDER BY trending_score DESC
            LIMIT 5
        """)
        
        trending = []
        for row in cursor.fetchall():
            trending.append({
                'title': row[0],
                'channel': row[1],
                'views': row[2],
                'engagement': float(row[3]) if row[3] else 0,
                'score': float(row[4]) if row[4] else 0
            })
        
        return trending
    
    def check_cache_performance(self):
        """Check Redis cache performance"""
        info = self.redis.info('stats')
        
        hit_rate = 0
        if info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0) > 0:
            hit_rate = (info.get('keyspace_hits', 0) / 
                       (info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0))) * 100
        
        return {
            'hit_rate': round(hit_rate, 2),
            'total_keys': self.redis.dbsize(),
            'memory_used_mb': round(info.get('used_memory', 0) / (1024*1024), 2),
            'status': 'OK' if hit_rate > 80 else 'LOW_HIT_RATE'
        }
    
    def process_alerts(self, results):
        """Process monitoring results and send alerts"""
        alerts = []
        
        # Check cost alerts
        if results['checks']['cost_status'].get('status') == 'OVER_BUDGET':
            alerts.append({
                'level': 'CRITICAL',
                'message': f"Cost per video is ${results['checks']['cost_status']['cost_per_video']:.2f} (threshold: $3.00)"
            })
        
        # Check cache performance
        if results['checks']['cache'].get('status') == 'LOW_HIT_RATE':
            alerts.append({
                'level': 'WARNING',
                'message': f"Cache hit rate is {results['checks']['cache']['hit_rate']}% (threshold: 80%)"
            })
        
        # Send alerts to Slack
        for alert in alerts:
            self.send_slack_alert(alert)
    
    def send_slack_alert(self, alert):
        """Send alert to Slack"""
        color = 'danger' if alert['level'] == 'CRITICAL' else 'warning'
        
        payload = {
            'attachments': [{
                'color': color,
                'title': f"Analytics Alert - {alert['level']}",
                'text': alert['message'],
                'footer': 'YTEMPIRE Analytics Monitor',
                'ts': int(datetime.now().timestamp())
            }]
        }
        
        requests.post(self.slack_webhook, json=payload)
    
    def log_monitoring_results(self, results):
        """Log monitoring results to database"""
        cursor = self.db.cursor()
        
        cursor.execute("""
            INSERT INTO analytics.monitoring_log (timestamp, results)
            VALUES (%s, %s)
        """, (datetime.now(), json.dumps(results)))
        
        self.db.commit()

if __name__ == "__main__":
    monitor = HourlyMonitor()
    monitor.run_hourly_checks()
```

### End-of-Day Tasks

```bash
#!/bin/bash
# end_of_day_tasks.sh
# Daily end-of-day tasks for Analytics Engineer
# Run time: 6:00 PM daily

echo "========================================="
echo "YTEMPIRE Analytics End-of-Day Tasks"
echo "Date: $(date)"
echo "========================================="

# 1. Generate daily summary report
echo -e "\n[1/6] Generating daily summary report..."
python3 /opt/ytempire/analytics/scripts/generate_daily_report.py

# 2. Backup critical data
echo -e "\n[2/6] Backing up critical data..."
pg_dump -U analytics_user -d ytempire -t analytics.fact_video_performance -t analytics.fact_revenue -f /backup/daily/analytics_$(date +%Y%m%d).sql

# 3. Archive old logs
echo -e "\n[3/6] Archiving old logs..."
find /opt/ytempire/logs -name "*.log" -mtime +7 -exec gzip {} \;
find /opt/ytempire/logs -name "*.gz" -mtime +30 -delete

# 4. Update materialized views
echo -e "\n[4/6] Refreshing materialized views..."
psql -U analytics_user -d ytempire -c "SELECT analytics.refresh_materialized_views();"

# 5. Generate user reports
echo -e "\n[5/6] Generating user revenue reports..."
psql -U analytics_user -d ytempire -c "
INSERT INTO analytics.user_daily_reports (user_id, report_date, metrics)
SELECT 
    user_id,
    CURRENT_DATE,
    jsonb_build_object(
        'revenue_today', revenue_30d / 30,
        'videos_published', total_videos,
        'goal_progress', goal_progress_pct,
        'days_to_10k', days_to_10k_goal
    )
FROM analytics.v_user_revenue_tracking;"

# 6. Send daily summary email
echo -e "\n[6/6] Sending daily summary email..."
python3 /opt/ytempire/analytics/scripts/send_daily_email.py

echo -e "\n========================================="
echo "End-of-day tasks complete!"
echo "========================================="
```

---

## 5.2 Maintenance Procedures

### Weekly Maintenance

```bash
#!/bin/bash
# weekly_maintenance.sh
# Weekly maintenance tasks for Analytics Engineer
# Run time: Sunday 2:00 AM

echo "========================================="
echo "YTEMPIRE Analytics Weekly Maintenance"
echo "Date: $(date)"
echo "========================================="

# 1. Full VACUUM on large tables
echo -e "\n[1/7] Running VACUUM FULL on large tables..."
psql -U analytics_user -d ytempire -c "VACUUM FULL ANALYZE analytics.fact_video_performance;"
psql -U analytics_user -d ytempire -c "VACUUM FULL ANALYZE analytics.fact_revenue;"
psql -U analytics_user -d ytempire -c "VACUUM FULL ANALYZE analytics.fact_costs;"

# 2. Rebuild indexes
echo -e "\n[2/7] Rebuilding indexes..."
psql -U analytics_user -d ytempire -c "REINDEX TABLE analytics.fact_video_performance;"
psql -U analytics_user -d ytempire -c "REINDEX TABLE analytics.fact_revenue;"
psql -U analytics_user -d ytempire -c "REINDEX TABLE analytics.dim_video;"
psql -U analytics_user -d ytempire -c "REINDEX TABLE analytics.dim_channel;"

# 3. Update table statistics
echo -e "\n[3/7] Updating table statistics..."
psql -U analytics_user -d ytempire -c "ANALYZE;"

# 4. Clean old data
echo -e "\n[4/7] Cleaning old data..."
psql -U analytics_user -d ytempire -c "
-- Clean old monitoring logs
DELETE FROM analytics.query_performance_log 
WHERE timestamp < CURRENT_DATE - INTERVAL '30 days';

-- Clean old data quality checks
DELETE FROM analytics.data_quality_checks 
WHERE check_timestamp < CURRENT_DATE - INTERVAL '90 days';

-- Archive old API logs
INSERT INTO analytics.youtube_api_usage_archive
SELECT * FROM analytics.youtube_api_usage
WHERE timestamp < CURRENT_DATE - INTERVAL '30 days';

DELETE FROM analytics.youtube_api_usage
WHERE timestamp < CURRENT_DATE - INTERVAL '30 days';"

# 5. Generate weekly performance report
echo -e "\n[5/7] Generating weekly performance report..."
python3 /opt/ytempire/analytics/scripts/generate_weekly_report.py

# 6. Check and optimize slow queries
echo -e "\n[6/7] Analyzing slow queries..."
psql -U analytics_user -d ytempire -c "
SELECT * FROM analytics.get_slow_queries();"

# 7. Backup configuration
echo -e "\n[7/7] Backing up configuration..."
tar -czf /backup/weekly/config_$(date +%Y%m%d).tar.gz /opt/ytempire/config/

echo -e "\n========================================="
echo "Weekly maintenance complete!"
echo "========================================="
```

### Monthly Tasks

```python
# maintenance/monthly_tasks.py
"""
Monthly maintenance tasks for Analytics Engineer
"""

import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class MonthlyMaintenance:
    def __init__(self):
        self.db = psycopg2.connect(
            host="localhost",
            database="ytempire",
            user="analytics_user",
            password="password"
        )
        
    def run_monthly_tasks(self):
        """Execute all monthly maintenance tasks"""
        print(f"Starting monthly maintenance - {datetime.now()}")
        
        # 1. Generate monthly analytics report
        self.generate_monthly_report()
        
        # 2. Analyze growth trends
        self.analyze_growth_trends()
        
        # 3. Optimize database
        self.optimize_database()
        
        # 4. Update cost baselines
        self.update_cost_baselines()
        
        # 5. Archive old data
        self.archive_old_data()
        
        # 6. Review and update alerts
        self.review_alert_thresholds()
        
        print("Monthly maintenance complete!")
    
    def generate_monthly_report(self):
        """Generate comprehensive monthly report"""
        cursor = self.db.cursor()
        
        # Get monthly summary
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT u.user_id) as total_users,
                COUNT(DISTINCT c.channel_key) as total_channels,
                COUNT(DISTINCT v.video_key) as total_videos,
                SUM(r.total_revenue_cents) / 100.0 as total_revenue,
                SUM(fc.total_cost_cents) / 100.0 as total_costs,
                (SUM(r.total_revenue_cents) - SUM(fc.total_cost_cents)) / 100.0 as total_profit
            FROM analytics.dim_user u
            LEFT JOIN analytics.dim_channel c ON u.user_id = c.user_id
            LEFT JOIN analytics.dim_video v ON c.channel_id = v.channel_id
            LEFT JOIN analytics.fact_revenue r ON v.video_key = r.video_key
            LEFT JOIN analytics.fact_costs fc ON v.video_id = fc.video_id
            WHERE DATE_TRUNC('month', v.published_at) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
        """)
        
        summary = cursor.fetchone()
        
        # Create report
        report = f"""
        YTEMPIRE Monthly Analytics Report
        Month: {(datetime.now() - timedelta(days=30)).strftime('%B %Y')}
        
        === EXECUTIVE SUMMARY ===
        Total Users: {summary[0]}
        Total Channels: {summary[1]}
        Total Videos: {summary[2]}
        Total Revenue: ${summary[3]:,.2f}
        Total Costs: ${summary[4]:,.2f}
        Total Profit: ${summary[5]:,.2f}
        ROI: {((summary[5] / summary[4]) * 100):.1f}%
        
        === USER PERFORMANCE ===
        """
        
        # Get user performance
        cursor.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE current_month_revenue >= 10000) as users_at_10k,
                COUNT(*) FILTER (WHERE current_month_revenue >= 5000) as users_at_5k,
                COUNT(*) FILTER (WHERE current_month_revenue >= 1000) as users_at_1k,
                AVG(current_month_revenue) as avg_revenue
            FROM analytics.v_user_revenue_tracking
        """)
        
        user_perf = cursor.fetchone()
        
        report += f"""
        Users at $10K+: {user_perf[0]}
        Users at $5K+: {user_perf[1]}
        Users at $1K+: {user_perf[2]}
        Average Revenue: ${user_perf[3]:,.2f}
        """
        
        # Save report
        with open(f'/reports/monthly_report_{datetime.now().strftime("%Y%m")}.txt', 'w') as f:
            f.write(report)
        
        return report
    
    def optimize_database(self):
        """Run database optimization"""
        cursor = self.db.cursor()
        
        # Update pg_stat_statements
        cursor.execute("SELECT pg_stat_statements_reset();")
        
        # Rebuild statistics
        cursor.execute("ANALYZE;")
        
        # Check for bloated tables
        cursor.execute("""
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
                n_dead_tup,
                n_live_tup,
                ROUND(n_dead_tup::numeric / NULLIF(n_live_tup + n_dead_tup, 0) * 100, 2) AS dead_tuple_pct
            FROM pg_stat_user_tables
            WHERE schemaname = 'analytics'
                AND n_dead_tup > 10000
            ORDER BY n_dead_tup DESC
        """)
        
        bloated_tables = cursor.fetchall()
        
        # Vacuum bloated tables
        for table in bloated_tables:
            if table[5] > 20:  # More than 20% dead tuples
                cursor.execute(f"VACUUM FULL {table[0]}.{table[1]};")
        
        self.db.commit()

### Performance Tuning

```sql
-- ============================================
-- PERFORMANCE TUNING PROCEDURES
-- ============================================

-- Function to identify and fix performance issues
CREATE OR REPLACE FUNCTION analytics.tune_performance()
RETURNS TABLE(
    issue_type TEXT,
    object_name TEXT,
    recommendation TEXT,
    impact TEXT
) AS $$
BEGIN
    -- Check for missing indexes
    RETURN QUERY
    SELECT 
        'MISSING_INDEX'::TEXT,
        schemaname || '.' || tablename::TEXT,
        'CREATE INDEX ON ' || schemaname || '.' || tablename || '(' || attname || ')'::TEXT,
        'HIGH'::TEXT
    FROM pg_stats
    WHERE schemaname = 'analytics'
        AND n_distinct > 100
        AND correlation < 0.1
        AND NOT EXISTS (
            SELECT 1 FROM pg_indexes
            WHERE schemaname = pg_stats.schemaname
                AND tablename = pg_stats.tablename
                AND indexdef LIKE '%' || attname || '%'
        );
    
    -- Check for unused indexes
    RETURN QUERY
    SELECT 
        'UNUSED_INDEX'::TEXT,
        indexrelname::TEXT,
        'DROP INDEX ' || schemaname || '.' || indexrelname::TEXT,
        'LOW'::TEXT
    FROM pg_stat_user_indexes
    WHERE schemaname = 'analytics'
        AND idx_scan = 0
        AND indexrelname NOT LIKE '%_pkey';
    
    -- Check for table bloat
    RETURN QUERY
    SELECT 
        'TABLE_BLOAT'::TEXT,
        schemaname || '.' || tablename::TEXT,
        'VACUUM FULL ' || schemaname || '.' || tablename::TEXT,
        CASE 
            WHEN dead_tuple_pct > 30 THEN 'HIGH'
            WHEN dead_tuple_pct > 20 THEN 'MEDIUM'
            ELSE 'LOW'
        END::TEXT
    FROM (
        SELECT 
            schemaname,
            tablename,
            ROUND(n_dead_tup::numeric / NULLIF(n_live_tup + n_dead_tup, 0) * 100, 2) AS dead_tuple_pct
        FROM pg_stat_user_tables
        WHERE schemaname = 'analytics'
    ) t
    WHERE dead_tuple_pct > 10;
END;
$$ LANGUAGE plpgsql;
```

---

## 5.3 Incident Response

### Alert Handling

```python
# incident/alert_handler.py
"""
Alert handling and incident response
"""

import psycopg2
import redis
import requests
from datetime import datetime
import json

class AlertHandler:
    def __init__(self):
        self.db = psycopg2.connect(
            host="localhost",
            database="ytempire",
            user="analytics_user"
        )
        self.redis = redis.Redis(host='localhost', port=6379)
        self.alert_thresholds = self.load_alert_thresholds()
        
    def load_alert_thresholds(self):
        """Load alert threshold configuration"""
        return {
            'data_freshness': {
                'critical': 120,  # minutes
                'warning': 60
            },
            'cost_per_video': {
                'critical': 3.50,  # dollars
                'warning': 3.00
            },
            'api_quota': {
                'critical': 9000,  # units
                'warning': 7500
            },
            'dashboard_performance': {
                'critical': 5000,  # milliseconds
                'warning': 2000
            },
            'cache_hit_rate': {
                'critical': 60,  # percentage
                'warning': 80
            }
        }
    
    def check_all_alerts(self):
        """Check all alert conditions"""
        alerts = []
        
        # Check data freshness
        freshness_alert = self.check_data_freshness()
        if freshness_alert:
            alerts.append(freshness_alert)
        
        # Check costs
        cost_alert = self.check_costs()
        if cost_alert:
            alerts.append(cost_alert)
        
        # Check API quota
        quota_alert = self.check_api_quota()
        if quota_alert:
            alerts.append(quota_alert)
        
        # Check performance
        perf_alert = self.check_performance()
        if perf_alert:
            alerts.append(perf_alert)
        
        # Process alerts
        for alert in alerts:
            self.process_alert(alert)
        
        return alerts
    
    def check_data_freshness(self):
        """Check if data is stale"""
        cursor = self.db.cursor()
        
        cursor.execute("""
            SELECT 
                'fact_video_performance' as table_name,
                EXTRACT(MINUTE FROM (NOW() - MAX(timestamp))) as minutes_stale
            FROM analytics.fact_video_performance
        """)
        
        result = cursor.fetchone()
        
        if result:
            minutes_stale = result[1]
            
            if minutes_stale > self.alert_thresholds['data_freshness']['critical']:
                return {
                    'level': 'CRITICAL',
                    'type': 'DATA_FRESHNESS',
                    'message': f'Data is {minutes_stale} minutes old',
                    'action': 'Check ETL pipeline immediately'
                }
            elif minutes_stale > self.alert_thresholds['data_freshness']['warning']:
                return {
                    'level': 'WARNING',
                    'type': 'DATA_FRESHNESS',
                    'message': f'Data is {minutes_stale} minutes old',
                    'action': 'Monitor ETL pipeline'
                }
        
        return None
    
    def process_alert(self, alert):
        """Process and route alert"""
        # Log alert
        self.log_alert(alert)
        
        # Send notifications based on level
        if alert['level'] == 'CRITICAL':
            self.send_pagerduty(alert)
            self.send_slack(alert, channel='#alerts-critical')
            self.send_email(alert, recipients=['analytics-oncall@ytempire.com'])
        elif alert['level'] == 'WARNING':
            self.send_slack(alert, channel='#alerts-warning')
        
        # Execute automatic remediation if applicable
        self.attempt_auto_remediation(alert)
    
    def attempt_auto_remediation(self, alert):
        """Attempt automatic remediation for certain alerts"""
        if alert['type'] == 'CACHE_HIT_RATE':
            # Clear and warm cache
            self.redis.flushdb()
            self.warm_cache()
            
        elif alert['type'] == 'DASHBOARD_PERFORMANCE':
            # Refresh materialized views
            cursor = self.db.cursor()
            cursor.execute("SELECT analytics.refresh_materialized_views();")
            self.db.commit()
    
    def send_pagerduty(self, alert):
        """Send alert to PagerDuty"""
        # PagerDuty integration
        pass
    
    def send_slack(self, alert, channel):
        """Send alert to Slack"""
        # Slack integration
        pass
    
    def send_email(self, alert, recipients):
        """Send alert email"""
        # Email integration
        pass
    
    def log_alert(self, alert):
        """Log alert to database"""
        cursor = self.db.cursor()
        
        cursor.execute("""
            INSERT INTO analytics.alerts_log (
                alert_level, alert_type, message, action, timestamp
            ) VALUES (%s, %s, %s, %s, %s)
        """, (
            alert['level'],
            alert['type'],
            alert['message'],
            alert.get('action', ''),
            datetime.now()
        ))
        
        self.db.commit()
```

### Troubleshooting Guide

```markdown
# YTEMPIRE Analytics Troubleshooting Guide

## Common Issues & Solutions

### Issue: Dashboard Loading Slowly
**Symptoms**: Dashboard takes >10 seconds to load
**Root Causes**:
1. Unoptimized queries
2. Cache miss
3. High concurrent usage
4. Stale materialized views

**Resolution Steps**:
1. Check query execution plan:
   ```sql
   EXPLAIN ANALYZE <your_query>;
   ```
2. Verify materialized views are current:
   ```sql
   SELECT * FROM analytics.refresh_materialized_views();
   ```
3. Review cache hit rates:
   ```bash
   redis-cli INFO stats
   ```
4. Consider query redesign or add indexes

### Issue: Data Discrepancy
**Symptoms**: Numbers don't match between reports
**Root Causes**:
1. Different calculation methods
2. Timing differences
3. Data quality issues

**Resolution Steps**:
1. Verify data freshness:
   ```sql
   SELECT MAX(timestamp) FROM analytics.fact_video_performance;
   ```
2. Check business logic definitions
3. Run reconciliation queries:
   ```sql
   SELECT * FROM analytics.check_data_accuracy();
   ```
4. Review data lineage

### Issue: High Costs Per Video
**Symptoms**: Cost per video exceeds $3.00
**Root Causes**:
1. Inefficient AI model usage
2. Excessive API calls
3. Redundant processing

**Resolution Steps**:
1. Review cost breakdown:
   ```sql
   SELECT * FROM analytics.v_cost_analysis 
   WHERE total_cost > 3.00 
   ORDER BY total_cost DESC;
   ```
2. Check for duplicate processing
3. Optimize API call batching
4. Review AI model parameters

### Issue: API Quota Exceeded
**Symptoms**: YouTube API errors, quota limit reached
**Root Causes**:
1. Inefficient API usage
2. Missing cache layer
3. Duplicate requests

**Resolution Steps**:
1. Check current quota usage:
   ```sql
   SELECT * FROM analytics.check_api_quota();
   ```
2. Review API call patterns
3. Implement request batching
4. Increase cache TTL for stable data
```

### Escalation Procedures

```yaml
# escalation_procedures.yml
escalation_matrix:
  data_issues:
    level_1:
      - Issue: Data delay < 1 hour
      - Owner: Analytics Engineer
      - Action: Check ETL logs, restart if needed
      - SLA: 30 minutes
      
    level_2:
      - Issue: Data delay > 1 hour
      - Owner: Data Engineer
      - Action: Debug pipeline, check source systems
      - SLA: 1 hour
      
    level_3:
      - Issue: Complete data outage
      - Owner: Data Team Lead
      - Action: Coordinate recovery, communicate impact
      - SLA: 2 hours
  
  performance_issues:
    level_1:
      - Issue: Dashboard slow (2-5 seconds)
      - Owner: Analytics Engineer
      - Action: Optimize queries, refresh cache
      - SLA: 1 hour
      
    level_2:
      - Issue: Dashboard timeout (>5 seconds)
      - Owner: Backend Team
      - Action: Scale resources, optimize database
      - SLA: 2 hours
      
    level_3:
      - Issue: System-wide performance degradation
      - Owner: Platform Ops Lead
      - Action: Emergency scaling, incident command
      - SLA: 30 minutes
  
  cost_overruns:
    level_1:
      - Issue: Cost per video $3.00-$3.50
      - Owner: Analytics Engineer
      - Action: Alert stakeholders, identify cause
      - SLA: 2 hours
      
    level_2:
      - Issue: Cost per video > $3.50
      - Owner: Data Team Lead
      - Action: Implement cost controls, optimize
      - SLA: 1 hour
      
    level_3:
      - Issue: Daily costs > 2x budget
      - Owner: VP of AI
      - Action: Emergency shutdown of non-critical processes
      - SLA: 15 minutes

contact_list:
  analytics_engineer:
    - Name: [Your Name]
    - Phone: [Phone]
    - Email: analytics-eng@ytempire.com
    - Slack: @analytics-eng
  
  data_engineer:
    - Name: [Name]
    - Phone: [Phone]
    - Email: data-eng@ytempire.com
    - Slack: @data-eng
  
  data_team_lead:
    - Name: [Name]
    - Phone: [Phone]
    - Email: data-lead@ytempire.com
    - Slack: @data-lead
  
  vp_of_ai:
    - Name: [Name]
    - Phone: [Phone]
    - Email: vp-ai@ytempire.com
    - Slack: @vp-ai

on_call_schedule:
  primary:
    - Monday-Friday: Analytics Engineer
    - Nights/Weekends: Rotating
  
  backup:
    - Always: Data Engineer
  
  escalation:
    - Always: Data Team Lead
```