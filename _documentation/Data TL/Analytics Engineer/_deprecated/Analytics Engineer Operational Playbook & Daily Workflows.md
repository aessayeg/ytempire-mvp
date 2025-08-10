# Analytics Engineer Operational Playbook & Daily Workflows

**Document Version**: 1.0  
**Date**: January 2025  
**Purpose**: Day-to-day operational procedures and workflows  
**Scope**: Daily tasks, troubleshooting, maintenance, and incident response

---

## 1. Daily Operations Schedule

### Morning Routine (9:00 AM - 10:00 AM)

```bash
#!/bin/bash
# morning_checks.sh - Run every morning at 9:00 AM

echo "=== YTEMPIRE Analytics Morning Checks ==="
echo "Date: $(date)"

# 1. Check data freshness
psql -U ytempire_app -d ytempire -c "SELECT * FROM analytics.check_data_freshness();"

# 2. Check overnight job status
psql -U ytempire_app -d ytempire -c "
    SELECT job_name, status, last_run, error_message 
    FROM analytics.overnight_jobs_log 
    WHERE last_run >= CURRENT_DATE - INTERVAL '1 day'
    ORDER BY last_run DESC;"

# 3. Check system resources
echo "Disk Usage:"
df -h | grep -E 'Filesystem|ytempire'

echo "Database Size:"
psql -U ytempire_app -d ytempire -c "
    SELECT pg_database_size('ytempire')/1024/1024/1024 as size_gb;"

# 4. Check for alerts
psql -U ytempire_app -d ytempire -c "
    SELECT * FROM analytics.active_alerts 
    WHERE acknowledged = false 
    ORDER BY severity, created_at;"

# 5. Generate morning report
python3 /opt/ytempire/scripts/generate_morning_report.py
```

### Hourly Tasks (Automated)

```sql
-- Hourly refresh procedure
CREATE OR REPLACE FUNCTION analytics.hourly_refresh()
RETURNS void AS $$
BEGIN
    -- Refresh recent performance view
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.mv_recent_performance;
    
    -- Update cache for dashboards
    PERFORM analytics.warm_dashboard_cache();
    
    -- Check and alert on cost thresholds
    PERFORM analytics.check_cost_alerts();
    
    -- Log performance metrics
    INSERT INTO analytics.hourly_metrics_log (hour, metrics)
    SELECT 
        DATE_TRUNC('hour', NOW()),
        jsonb_build_object(
            'active_channels', COUNT(DISTINCT channel_id),
            'videos_processed', COUNT(DISTINCT video_id),
            'total_views', SUM(views),
            'avg_cost_per_video', AVG(cost_cents) / 100.0
        )
    FROM analytics.fact_video_performance
    WHERE timestamp >= NOW() - INTERVAL '1 hour';
    
    RAISE NOTICE 'Hourly refresh completed at %', NOW();
END;
$$ LANGUAGE plpgsql;

-- Schedule with pg_cron
SELECT cron.schedule('hourly_refresh', '0 * * * *', 'SELECT analytics.hourly_refresh()');
```

### End of Day Procedures (5:00 PM - 6:00 PM)

```python
# end_of_day_report.py
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import json

class EndOfDayReport:
    def __init__(self):
        self.conn = psycopg2.connect(
            host="localhost",
            database="ytempire",
            user="ytempire_app",
            password="ytempire2025!"
        )
        
    def generate_report(self):
        """Generate comprehensive end of day report"""
        
        report_data = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'metrics': {}
        }
        
        # 1. Daily Performance Summary
        daily_perf = pd.read_sql("""
            SELECT 
                COUNT(DISTINCT channel_id) as active_channels,
                COUNT(DISTINCT video_id) as videos_published,
                SUM(views) as total_views,
                AVG(engagement_rate) as avg_engagement,
                SUM(estimated_revenue_cents) / 100.0 as total_revenue,
                SUM(cost_cents) / 100.0 as total_cost,
                (SUM(estimated_revenue_cents) - SUM(cost_cents)) / 100.0 as profit
            FROM analytics.fact_video_performance
            WHERE DATE(timestamp) = CURRENT_DATE
        """, self.conn)
        
        report_data['metrics']['daily_performance'] = daily_perf.to_dict('records')[0]
        
        # 2. Top Performing Content
        top_videos = pd.read_sql("""
            SELECT 
                v.title,
                v.channel_id,
                vp.views,
                vp.engagement_rate,
                vp.estimated_revenue_cents / 100.0 as revenue
            FROM analytics.dim_video v
            JOIN analytics.fact_video_performance vp ON v.video_id = vp.video_id
            WHERE DATE(vp.timestamp) = CURRENT_DATE
            ORDER BY vp.views DESC
            LIMIT 10
        """, self.conn)
        
        report_data['metrics']['top_videos'] = top_videos.to_dict('records')
        
        # 3. Cost Analysis
        cost_breakdown = pd.read_sql("""
            SELECT 
                cost_type,
                COUNT(*) as transactions,
                SUM(cost_cents) / 100.0 as total_cost,
                AVG(cost_cents) / 100.0 as avg_cost
            FROM analytics.cost_tracking
            WHERE DATE(timestamp) = CURRENT_DATE
            GROUP BY cost_type
            ORDER BY total_cost DESC
        """, self.conn)
        
        report_data['metrics']['cost_breakdown'] = cost_breakdown.to_dict('records')
        
        # 4. Alert Summary
        alerts = pd.read_sql("""
            SELECT 
                severity,
                COUNT(*) as count,
                array_agg(alert_type) as types
            FROM analytics.alerts_log
            WHERE DATE(created_at) = CURRENT_DATE
            GROUP BY severity
        """, self.conn)
        
        report_data['metrics']['alerts'] = alerts.to_dict('records')
        
        return report_data
    
    def send_report(self, report_data):
        """Send report via email"""
        
        html_content = self.format_html_report(report_data)
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"YTEMPIRE Daily Report - {report_data['date']}"
        msg['From'] = "analytics@ytempire.com"
        msg['To'] = "team@ytempire.com"
        
        msg.attach(MIMEText(html_content, 'html'))
        
        # Send email (configure SMTP settings)
        # smtp = smtplib.SMTP('localhost')
        # smtp.send_message(msg)
        # smtp.quit()
        
        # Save report to disk
        with open(f"/opt/ytempire/reports/daily_{report_data['date']}.json", 'w') as f:
            json.dump(report_data, f, indent=2)
            
    def format_html_report(self, data):
        """Format report as HTML"""
        
        perf = data['metrics']['daily_performance']
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .metric {{ font-size: 24px; font-weight: bold; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>YTEMPIRE Daily Report - {data['date']}</h1>
            
            <h2>Key Metrics</h2>
            <table>
                <tr>
                    <td>Active Channels</td>
                    <td class="metric">{perf['active_channels']}</td>
                </tr>
                <tr>
                    <td>Videos Published</td>
                    <td class="metric">{perf['videos_published']}</td>
                </tr>
                <tr>
                    <td>Total Views</td>
                    <td class="metric">{perf['total_views']:,}</td>
                </tr>
                <tr>
                    <td>Revenue</td>
                    <td class="metric positive">${perf['total_revenue']:.2f}</td>
                </tr>
                <tr>
                    <td>Cost</td>
                    <td class="metric negative">${perf['total_cost']:.2f}</td>
                </tr>
                <tr>
                    <td>Profit</td>
                    <td class="metric {'positive' if perf['profit'] > 0 else 'negative'}">
                        ${perf['profit']:.2f}
                    </td>
                </tr>
            </table>
            
            <h2>Top Videos Today</h2>
            <table>
                <tr>
                    <th>Title</th>
                    <th>Views</th>
                    <th>Revenue</th>
                </tr>
                {''.join([f"<tr><td>{v['title'][:50]}...</td><td>{v['views']}</td><td>${v['revenue']:.2f}</td></tr>" for v in data['metrics']['top_videos']])}
            </table>
        </body>
        </html>
        """
        
        return html

if __name__ == "__main__":
    reporter = EndOfDayReport()
    report = reporter.generate_report()
    reporter.send_report(report)
    print(f"End of day report generated for {report['date']}")
```

---

## 2. Common Troubleshooting Procedures

### 2.1 Dashboard Performance Issues

```sql
-- Diagnose slow dashboard queries
CREATE OR REPLACE FUNCTION analytics.diagnose_dashboard_performance(
    p_dashboard_name VARCHAR DEFAULT NULL
) RETURNS TABLE(
    query_id TEXT,
    execution_time_ms NUMERIC,
    query_text TEXT,
    recommendations TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH slow_queries AS (
        SELECT 
            queryid::TEXT,
            mean_exec_time,
            query,
            calls
        FROM pg_stat_statements
        WHERE query LIKE '%' || COALESCE(p_dashboard_name, 'analytics') || '%'
          AND mean_exec_time > 500
        ORDER BY mean_exec_time DESC
        LIMIT 10
    )
    SELECT 
        sq.queryid as query_id,
        ROUND(sq.mean_exec_time, 2) as execution_time_ms,
        sq.query as query_text,
        CASE 
            WHEN sq.query LIKE '%fact_video_performance%' AND sq.mean_exec_time > 2000 
                THEN 'Consider using materialized view mv_recent_performance'
            WHEN sq.query LIKE '%GROUP BY%' AND sq.mean_exec_time > 1500
                THEN 'Add index on GROUP BY columns'
            WHEN sq.query LIKE '%JOIN%' AND sq.mean_exec_time > 1000
                THEN 'Check join conditions and consider denormalization'
            WHEN sq.calls > 1000
                THEN 'High frequency query - implement caching'
            ELSE 'Review query execution plan with EXPLAIN ANALYZE'
        END as recommendations
    FROM slow_queries sq;
END;
$$ LANGUAGE plpgsql;

-- Quick fix for common issues
CREATE OR REPLACE FUNCTION analytics.quick_fix_performance()
RETURNS TEXT AS $$
DECLARE
    v_result TEXT := '';
BEGIN
    -- 1. Update statistics
    ANALYZE analytics.fact_video_performance;
    ANALYZE analytics.dim_video;
    ANALYZE analytics.dim_channel;
    v_result := v_result || 'Statistics updated. ';
    
    -- 2. Clear bloat
    VACUUM analytics.fact_video_performance;
    v_result := v_result || 'Vacuum completed. ';
    
    -- 3. Refresh materialized views
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.mv_recent_performance;
    v_result := v_result || 'Materialized views refreshed. ';
    
    -- 4. Reset cache
    PERFORM analytics.clear_dashboard_cache();
    PERFORM analytics.warm_dashboard_cache();
    v_result := v_result || 'Cache reset. ';
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql;
```

### 2.2 Data Quality Issues

```python
# data_quality_fixer.py
class DataQualityFixer:
    """Automated data quality issue resolution"""
    
    def __init__(self):
        self.conn = psycopg2.connect(
            host="localhost",
            database="ytempire",
            user="ytempire_app",
            password="ytempire2025!"
        )
        self.issues_found = []
        self.fixes_applied = []
    
    def diagnose_and_fix(self):
        """Main diagnostic and fix routine"""
        
        print("Starting data quality diagnosis...")
        
        # 1. Check for missing data
        self.fix_missing_data()
        
        # 2. Check for duplicate entries
        self.fix_duplicates()
        
        # 3. Check for anomalous values
        self.fix_anomalies()
        
        # 4. Check for referential integrity
        self.fix_referential_integrity()
        
        # 5. Generate report
        self.generate_fix_report()
    
    def fix_missing_data(self):
        """Identify and fix missing data gaps"""
        
        cursor = self.conn.cursor()
        
        # Find gaps in time series data
        cursor.execute("""
            WITH time_gaps AS (
                SELECT 
                    channel_id,
                    timestamp,
                    LAG(timestamp) OVER (PARTITION BY channel_id ORDER BY timestamp) as prev_timestamp,
                    timestamp - LAG(timestamp) OVER (PARTITION BY channel_id ORDER BY timestamp) as gap
                FROM analytics.fact_video_performance
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
            )
            SELECT 
                channel_id,
                prev_timestamp as gap_start,
                timestamp as gap_end,
                EXTRACT(EPOCH FROM gap) / 3600 as hours_missing
            FROM time_gaps
            WHERE gap > INTERVAL '2 hours'
        """)
        
        gaps = cursor.fetchall()
        
        for gap in gaps:
            channel_id, gap_start, gap_end, hours_missing = gap
            
            if hours_missing < 6:
                # Interpolate missing data
                print(f"Interpolating {hours_missing} hours of data for channel {channel_id}")
                
                cursor.execute("""
                    INSERT INTO analytics.fact_video_performance 
                        (video_id, channel_id, timestamp, views, engagement_rate, estimated_revenue_cents, cost_cents)
                    SELECT 
                        video_id,
                        channel_id,
                        timestamp + INTERVAL '1 hour' * generate_series(1, %s),
                        views + (views * 0.1 * generate_series(1, %s)),  -- Estimated growth
                        engagement_rate,
                        estimated_revenue_cents,
                        cost_cents
                    FROM analytics.fact_video_performance
                    WHERE channel_id = %s 
                      AND timestamp = %s
                    ON CONFLICT DO NOTHING
                """, (int(hours_missing), int(hours_missing), channel_id, gap_start))
                
                self.fixes_applied.append(f"Interpolated {hours_missing} hours for channel {channel_id}")
            else:
                self.issues_found.append(f"Large gap ({hours_missing} hours) for channel {channel_id} - manual review needed")
        
        self.conn.commit()
    
    def fix_duplicates(self):
        """Remove duplicate entries"""
        
        cursor = self.conn.cursor()
        
        # Find duplicates
        cursor.execute("""
            WITH duplicates AS (
                SELECT 
                    video_id,
                    timestamp,
                    COUNT(*) as count
                FROM analytics.fact_video_performance
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                GROUP BY video_id, timestamp
                HAVING COUNT(*) > 1
            )
            DELETE FROM analytics.fact_video_performance
            WHERE (video_id, timestamp) IN (
                SELECT video_id, timestamp FROM duplicates
            )
            AND ctid NOT IN (
                SELECT MIN(ctid)
                FROM analytics.fact_video_performance
                WHERE (video_id, timestamp) IN (
                    SELECT video_id, timestamp FROM duplicates
                )
                GROUP BY video_id, timestamp
            )
            RETURNING video_id
        """)
        
        deleted = cursor.fetchall()
        if deleted:
            self.fixes_applied.append(f"Removed {len(deleted)} duplicate entries")
        
        self.conn.commit()
    
    def fix_anomalies(self):
        """Fix anomalous values"""
        
        cursor = self.conn.cursor()
        
        # Fix negative values
        cursor.execute("""
            UPDATE analytics.fact_video_performance
            SET views = 0
            WHERE views < 0
            RETURNING video_id
        """)
        
        fixed_views = cursor.fetchall()
        if fixed_views:
            self.fixes_applied.append(f"Fixed {len(fixed_views)} negative view counts")
        
        # Fix impossible engagement rates
        cursor.execute("""
            UPDATE analytics.fact_video_performance
            SET engagement_rate = 1.0
            WHERE engagement_rate > 1.0
            RETURNING video_id
        """)
        
        fixed_engagement = cursor.fetchall()
        if fixed_engagement:
            self.fixes_applied.append(f"Fixed {len(fixed_engagement)} impossible engagement rates")
        
        # Fix excessive costs
        cursor.execute("""
            UPDATE analytics.cost_tracking
            SET cost_cents = 50  -- Set to $0.50 threshold
            WHERE cost_cents > 100  -- More than $1.00 per video
              AND cost_type = 'ai_generation'
            RETURNING video_id
        """)
        
        fixed_costs = cursor.fetchall()
        if fixed_costs:
            self.fixes_applied.append(f"Capped {len(fixed_costs)} excessive cost entries")
        
        self.conn.commit()
    
    def fix_referential_integrity(self):
        """Fix broken references"""
        
        cursor = self.conn.cursor()
        
        # Find orphaned video records
        cursor.execute("""
            DELETE FROM analytics.fact_video_performance
            WHERE video_id NOT IN (
                SELECT video_id FROM analytics.dim_video
            )
            RETURNING video_id
        """)
        
        orphaned = cursor.fetchall()
        if orphaned:
            self.fixes_applied.append(f"Removed {len(orphaned)} orphaned video records")
        
        self.conn.commit()
    
    def generate_fix_report(self):
        """Generate report of issues and fixes"""
        
        print("\n=== Data Quality Fix Report ===")
        print(f"Timestamp: {datetime.now()}")
        
        if self.issues_found:
            print("\nIssues Found (Require Manual Review):")
            for issue in self.issues_found:
                print(f"  - {issue}")
        
        if self.fixes_applied:
            print("\nFixes Applied Automatically:")
            for fix in self.fixes_applied:
                print(f"  ✓ {fix}")
        
        if not self.issues_found and not self.fixes_applied:
            print("\n✓ No data quality issues found!")
        
        # Log to database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO analytics.data_quality_log (timestamp, issues_found, fixes_applied)
            VALUES (%s, %s, %s)
        """, (datetime.now(), json.dumps(self.issues_found), json.dumps(self.fixes_applied)))
        
        self.conn.commit()
        cursor.close()

if __name__ == "__main__":
    fixer = DataQualityFixer()
    fixer.diagnose_and_fix()
```

---

## 3. Emergency Response Procedures

### 3.1 Cost Overrun Alert Response

```python
# cost_emergency_response.py
class CostEmergencyHandler:
    """Handle cost emergencies when thresholds are breached"""
    
    COST_THRESHOLDS = {
        'warning': 0.40,    # $0.40 per video
        'critical': 0.45,   # $0.45 per video
        'emergency': 0.50   # $0.50 per video - STOP
    }
    
    def __init__(self):
        self.conn = psycopg2.connect(
            host="localhost",
            database="ytempire",
            user="ytempire_app",
            password="ytempire2025!"
        )
        
    def check_and_respond(self):
        """Check costs and trigger appropriate response"""
        
        cursor = self.conn.cursor()
        
        # Get current average cost
        cursor.execute("""
            SELECT 
                AVG(cost_cents) / 100.0 as avg_cost,
                MAX(cost_cents) / 100.0 as max_cost,
                COUNT(*) as video_count
            FROM analytics.cost_tracking
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
        """)
        
        avg_cost, max_cost, video_count = cursor.fetchone()
        
        if avg_cost >= self.COST_THRESHOLDS['emergency']:
            self.emergency_shutdown(avg_cost)
        elif avg_cost >= self.COST_THRESHOLDS['critical']:
            self.critical_response(avg_cost)
        elif avg_cost >= self.COST_THRESHOLDS['warning']:
            self.warning_response(avg_cost)
        else:
            print(f"✓ Costs normal: ${avg_cost:.2f} per video")
    
    def emergency_shutdown(self, current_cost):
        """Emergency: Stop all expensive operations"""
        
        print(f"🚨 EMERGENCY: Cost at ${current_cost:.2f} - SHUTTING DOWN EXPENSIVE OPERATIONS")
        
        cursor = self.conn.cursor()
        
        # 1. Log emergency
        cursor.execute("""
            INSERT INTO analytics.emergency_log (event_type, severity, details, timestamp)
            VALUES ('COST_OVERRUN', 'EMERGENCY', %s, NOW())
        """, (json.dumps({'avg_cost': current_cost, 'threshold': self.COST_THRESHOLDS['emergency']}),))
        
        # 2. Send alerts
        self.send_emergency_alert(f"EMERGENCY: Costs at ${current_cost:.2f} per video!")
        
        # 3. Disable expensive features
        cursor.execute("""
            UPDATE analytics.system_config 
            SET value = 'false' 
            WHERE key IN ('enable_gpt4', 'enable_premium_voice', 'enable_4k_video')
        """)
        
        # 4. Switch to cache-only mode
        cursor.execute("""
            UPDATE analytics.system_config 
            SET value = 'cache_only' 
            WHERE key = 'data_source_mode'
        """)
        
        self.conn.commit()
        
        print("Emergency actions taken:")
        print("  - GPT-4 disabled")
        print("  - Premium features disabled")
        print("  - Switched to cache-only mode")
        print("  - Alerts sent to all stakeholders")
    
    def critical_response(self, current_cost):
        """Critical: Reduce costs immediately"""
        
        print(f"⚠️ CRITICAL: Cost at ${current_cost:.2f} - Implementing cost reduction")
        
        cursor = self.conn.cursor()
        
        # 1. Log critical event
        cursor.execute("""
            INSERT INTO analytics.alerts_log (severity, alert_type, message, created_at)
            VALUES ('CRITICAL', 'COST_ALERT', %s, NOW())
        """, (f"Cost at ${current_cost:.2f} - above critical threshold",))
        
        # 2. Reduce batch sizes
        cursor.execute("""
            UPDATE analytics.system_config 
            SET value = '5' 
            WHERE key = 'batch_size'
        """)
        
        # 3. Increase cache TTL
        cursor.execute("""
            UPDATE analytics.system_config 
            SET value = '7200' 
            WHERE key = 'cache_ttl_seconds'
        """)
        
        self.conn.commit()
        
        print("Critical actions taken:")
        print("  - Batch size reduced")
        print("  - Cache TTL increased")
        print("  - Team alerted")
    
    def warning_response(self, current_cost):
        """Warning: Monitor closely"""
        
        print(f"⚠️ WARNING: Cost at ${current_cost:.2f} - monitoring closely")
        
        cursor = self.conn.cursor()
        
        # Log warning
        cursor.execute("""
            INSERT INTO analytics.alerts_log (severity, alert_type, message, created_at)
            VALUES ('WARNING', 'COST_ALERT', %s, NOW())
        """, (f"Cost at ${current_cost:.2f} - approaching threshold",))
        
        self.conn.commit()
    
    def send_emergency_alert(self, message):
        """Send emergency alerts via multiple channels"""
        
        # Slack webhook
        webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
        slack_data = {
            'text': f"🚨 {message}",
            'channel': '#emergency-alerts',
            'username': 'Cost Monitor',
            'icon_emoji': ':warning:'
        }
        
        # Email alert
        # SMS alert via Twilio
        # PagerDuty incident
        
        print(f"Emergency alert sent: {message}")
```

### 3.2 Data Pipeline Failure Recovery

```sql
-- Pipeline failure detection and recovery
CREATE OR REPLACE FUNCTION analytics.handle_pipeline_failure(
    p_pipeline_name VARCHAR,
    p_error_message TEXT
) RETURNS TEXT AS $
DECLARE
    v_recovery_action TEXT;
    v_last_successful_run TIMESTAMPTZ;
BEGIN
    -- Log the failure
    INSERT INTO analytics.pipeline_failures (
        pipeline_name,
        error_message,
        failure_time
    ) VALUES (
        p_pipeline_name,
        p_error_message,
        NOW()
    );
    
    -- Determine recovery action based on pipeline
    CASE p_pipeline_name
        WHEN 'youtube_data_sync' THEN
            -- Get last successful run
            SELECT MAX(completed_at) INTO v_last_successful_run
            FROM analytics.pipeline_runs
            WHERE pipeline_name = p_pipeline_name
              AND status = 'SUCCESS';
            
            -- Attempt recovery
            IF NOW() - v_last_successful_run < INTERVAL '6 hours' THEN
                -- Recent data exists, can wait
                v_recovery_action := 'Scheduled retry in 30 minutes';
                PERFORM pg_notify('pipeline_retry', json_build_object(
                    'pipeline', p_pipeline_name,
                    'retry_in', 30
                )::text);
            ELSE
                -- Data getting stale, immediate action
                v_recovery_action := 'Immediate retry with fallback to cache';
                PERFORM analytics.retry_with_cache(p_pipeline_name);
            END IF;
            
        WHEN 'cost_aggregation' THEN
            -- Critical for cost monitoring
            v_recovery_action := 'Immediate retry with manual calculation fallback';
            PERFORM analytics.manual_cost_calculation();
            
        WHEN 'revenue_attribution' THEN
            -- Revenue tracking is critical
            v_recovery_action := 'Switch to estimated revenue model';
            PERFORM analytics.enable_revenue_estimation();
            
        ELSE
            v_recovery_action := 'Standard retry with exponential backoff';
    END CASE;
    
    -- Send alert
    PERFORM analytics.send_pipeline_alert(p_pipeline_name, p_error_message, v_recovery_action);
    
    RETURN v_recovery_action;
END;
$ LANGUAGE plpgsql;

-- Automated recovery procedures
CREATE OR REPLACE FUNCTION analytics.auto_recover_pipeline()
RETURNS void AS $
DECLARE
    v_failed_pipeline RECORD;
BEGIN
    -- Find failed pipelines
    FOR v_failed_pipeline IN 
        SELECT 
            pipeline_name,
            failure_count,
            last_failure,
            last_success
        FROM analytics.pipeline_status
        WHERE status = 'FAILED'
          AND last_failure > NOW() - INTERVAL '1 hour'
    LOOP
        -- Attempt recovery based on failure count
        IF v_failed_pipeline.failure_count < 3 THEN
            -- Simple retry
            PERFORM analytics.retry_pipeline(v_failed_pipeline.pipeline_name);
            
        ELSIF v_failed_pipeline.failure_count < 5 THEN
            -- Retry with reduced load
            PERFORM analytics.retry_pipeline_reduced(v_failed_pipeline.pipeline_name);
            
        ELSE
            -- Manual intervention required
            PERFORM analytics.escalate_to_oncall(v_failed_pipeline.pipeline_name);
        END IF;
    END LOOP;
END;
$ LANGUAGE plpgsql;
```

---

## 4. Performance Optimization Runbook

### 4.1 Query Optimization Workflow

```python
# query_optimizer.py
class QueryOptimizer:
    """Automated query optimization toolkit"""
    
    def __init__(self):
        self.conn = psycopg2.connect(
            host="localhost",
            database="ytempire",
            user="ytempire_app",
            password="ytempire2025!"
        )
    
    def analyze_and_optimize(self, query_text):
        """Analyze query and suggest optimizations"""
        
        cursor = self.conn.cursor()
        
        # 1. Get execution plan
        cursor.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query_text}")
        plan = cursor.fetchone()[0][0]
        
        # 2. Analyze plan
        issues = self.analyze_plan(plan)
        
        # 3. Generate recommendations
        recommendations = self.generate_recommendations(issues)
        
        # 4. Apply auto-fixes where possible
        auto_fixes = self.apply_auto_fixes(issues)
        
        return {
            'original_time': plan['Execution Time'],
            'issues': issues,
            'recommendations': recommendations,
            'auto_fixes_applied': auto_fixes
        }
    
    def analyze_plan(self, plan):
        """Analyze execution plan for issues"""
        
        issues = []
        
        # Check for sequential scans on large tables
        if 'Seq Scan' in str(plan):
            issues.append({
                'type': 'sequential_scan',
                'severity': 'high',
                'details': 'Sequential scan detected - missing index'
            })
        
        # Check for nested loops with high row counts
        if 'Nested Loop' in str(plan) and plan.get('Plan Rows', 0) > 1000:
            issues.append({
                'type': 'nested_loop',
                'severity': 'medium',
                'details': 'Nested loop with high row count'
            })
        
        # Check for sort operations
        if 'Sort' in str(plan) and plan.get('Sort Space Used', 0) > 1000:
            issues.append({
                'type': 'expensive_sort',
                'severity': 'medium',
                'details': 'Large sort operation in memory'
            })
        
        return issues
    
    def generate_recommendations(self, issues):
        """Generate optimization recommendations"""
        
        recommendations = []
        
        for issue in issues:
            if issue['type'] == 'sequential_scan':
                recommendations.append({
                    'action': 'CREATE INDEX',
                    'priority': 'HIGH',
                    'sql': self.suggest_index(issue)
                })
            
            elif issue['type'] == 'nested_loop':
                recommendations.append({
                    'action': 'Use Hash Join',
                    'priority': 'MEDIUM',
                    'sql': 'SET enable_nestloop = off;'
                })
            
            elif issue['type'] == 'expensive_sort':
                recommendations.append({
                    'action': 'Increase work_mem',
                    'priority': 'LOW',
                    'sql': 'SET work_mem = "256MB";'
                })
        
        return recommendations
    
    def suggest_index(self, issue):
        """Suggest appropriate index"""
        
        # This would be more sophisticated in production
        return """
        -- Example index suggestion
        CREATE INDEX CONCURRENTLY idx_performance_channel_time 
        ON analytics.fact_video_performance(channel_id, timestamp DESC)
        WHERE timestamp >= NOW() - INTERVAL '30 days';
        """
    
    def apply_auto_fixes(self, issues):
        """Apply automatic fixes where safe"""
        
        fixes_applied = []
        cursor = self.conn.cursor()
        
        for issue in issues:
            if issue['severity'] == 'low':
                # Safe to auto-fix low severity issues
                if issue['type'] == 'missing_statistics':
                    cursor.execute("ANALYZE;")
                    fixes_applied.append("Updated table statistics")
        
        self.conn.commit()
        return fixes_applied
```

### 4.2 Cache Management

```sql
-- Redis cache management functions
CREATE OR REPLACE FUNCTION analytics.manage_dashboard_cache()
RETURNS TABLE(
    cache_key TEXT,
    size_bytes BIGINT,
    ttl_seconds INTEGER,
    hit_rate NUMERIC,
    action TEXT
) AS $
DECLARE
    v_redis_conn TEXT;
BEGIN
    -- This would interface with Redis in production
    -- Showing structure for documentation
    
    RETURN QUERY
    SELECT 
        'dashboard:revenue:daily' as cache_key,
        1024::BIGINT as size_bytes,
        3600::INTEGER as ttl_seconds,
        0.85::NUMERIC as hit_rate,
        'KEEP'::TEXT as action
    UNION ALL
    SELECT 
        'dashboard:channels:performance',
        2048::BIGINT,
        1800::INTEGER,
        0.45::NUMERIC,
        'REFRESH'::TEXT  -- Low hit rate, should refresh
    UNION ALL
    SELECT 
        'dashboard:costs:hourly',
        512::BIGINT,
        7200::INTEGER,
        0.92::NUMERIC,
        'KEEP'::TEXT;
END;
$ LANGUAGE plpgsql;

-- Warm cache for critical dashboards
CREATE OR REPLACE FUNCTION analytics.warm_dashboard_cache()
RETURNS void AS $
BEGIN
    -- Pre-compute and cache critical metrics
    
    -- 1. Revenue summary
    PERFORM analytics.cache_set(
        'dashboard:revenue:summary',
        (SELECT jsonb_build_object(
            'daily', SUM(CASE WHEN revenue_date = CURRENT_DATE THEN total_revenue_cents END) / 100.0,
            'weekly', SUM(CASE WHEN revenue_date >= CURRENT_DATE - 7 THEN total_revenue_cents END) / 100.0,
            'monthly', SUM(total_revenue_cents) / 100.0
        )
        FROM analytics.revenue_attribution
        WHERE revenue_date >= CURRENT_DATE - INTERVAL '30 days'),
        3600  -- 1 hour TTL
    );
    
    -- 2. Channel performance
    PERFORM analytics.cache_set(
        'dashboard:channels:top10',
        (SELECT jsonb_agg(row_to_json(t))
        FROM (
            SELECT channel_id, channel_name, total_revenue, profit_rank
            FROM analytics.channel_performance_view
            ORDER BY profit_rank
            LIMIT 10
        ) t),
        1800  -- 30 minutes TTL
    );
    
    -- 3. Cost metrics
    PERFORM analytics.cache_set(
        'dashboard:costs:current',
        (SELECT jsonb_build_object(
            'current_avg', AVG(cost_cents) / 100.0,
            'trend', CASE 
                WHEN AVG(cost_cents) > LAG(AVG(cost_cents)) OVER (ORDER BY DATE_TRUNC('hour', timestamp)) 
                THEN 'increasing' 
                ELSE 'decreasing' 
            END
        )
        FROM analytics.cost_tracking
        WHERE timestamp >= NOW() - INTERVAL '1 hour'),
        300  -- 5 minutes TTL for critical cost data
    );
    
    RAISE NOTICE 'Dashboard cache warmed successfully';
END;
$ LANGUAGE plpgsql;
```

---

## 5. Monitoring & Alerting Setup

### 5.1 Alert Rules Configuration

```yaml
# alerts.yml - Grafana alert rules configuration
groups:
  - name: analytics_alerts
    interval: 1m
    rules:
      - alert: HighCostPerVideo
        expr: avg(cost_per_video_dollars) > 0.40
        for: 5m
        labels:
          severity: warning
          team: analytics
        annotations:
          summary: "Cost per video above $0.40"
          description: "Current average cost: {{ $value }}"
          
      - alert: CriticalCostPerVideo
        expr: avg(cost_per_video_dollars) > 0.45
        for: 2m
        labels:
          severity: critical
          team: analytics
        annotations:
          summary: "CRITICAL: Cost per video above $0.45"
          description: "Immediate action required. Cost: {{ $value }}"
          
      - alert: DataStaleness
        expr: (time() - max(data_last_updated)) > 21600  # 6 hours
        for: 10m
        labels:
          severity: warning
          team: analytics
        annotations:
          summary: "Data is stale (>6 hours old)"
          description: "Last update: {{ $value }} seconds ago"
          
      - alert: DashboardSlowResponse
        expr: histogram_quantile(0.95, dashboard_response_time) > 2
        for: 5m
        labels:
          severity: warning
          team: analytics
        annotations:
          summary: "Dashboard response time >2 seconds"
          description: "P95 response time: {{ $value }}s"
          
      - alert: RevenueDataMismatch
        expr: abs(youtube_revenue - calculated_revenue) / youtube_revenue > 0.05
        for: 15m
        labels:
          severity: critical
          team: analytics
        annotations:
          summary: "Revenue data mismatch >5%"
          description: "Difference: {{ $value }}%"
```

### 5.2 Monitoring Dashboard Queries

```sql
-- Create monitoring views for Grafana
CREATE OR REPLACE VIEW analytics.monitoring_metrics AS
SELECT 
    NOW() as timestamp,
    -- Performance metrics
    (SELECT AVG(execution_time_ms) FROM analytics.query_performance_log 
     WHERE timestamp >= NOW() - INTERVAL '5 minutes') as avg_query_time_ms,
    
    -- Data freshness
    (SELECT EXTRACT(EPOCH FROM (NOW() - MAX(timestamp))) / 3600 
     FROM analytics.fact_video_performance) as hours_since_last_update,
    
    -- Cost metrics
    (SELECT AVG(cost_cents) / 100.0 FROM analytics.cost_tracking 
     WHERE timestamp >= NOW() - INTERVAL '1 hour') as current_avg_cost,
    
    -- System health
    (SELECT COUNT(*) FROM analytics.active_alerts 
     WHERE severity IN ('critical', 'emergency')) as critical_alerts,
    
    -- Pipeline status
    (SELECT COUNT(*) FROM analytics.pipeline_status 
     WHERE status = 'FAILED' AND last_failure > NOW() - INTERVAL '1 hour') as failed_pipelines,
    
    -- Cache performance
    (SELECT AVG(hit_rate) FROM analytics.cache_metrics 
     WHERE timestamp >= NOW() - INTERVAL '5 minutes') as cache_hit_rate;

-- Grant select to Grafana user
GRANT SELECT ON analytics.monitoring_metrics TO grafana_user;
```

---

## 6. Weekly Maintenance Tasks

### 6.1 Weekly Optimization Script

```bash
#!/bin/bash
# weekly_maintenance.sh - Run every Sunday at 2 AM

echo "=== YTEMPIRE Weekly Maintenance - $(date) ==="

# 1. Full VACUUM on large tables
echo "Running VACUUM FULL on large tables..."
psql -U ytempire_app -d ytempire -c "VACUUM FULL ANALYZE analytics.fact_video_performance;"
psql -U ytempire_app -d ytempire -c "VACUUM FULL ANALYZE analytics.cost_tracking;"

# 2. Rebuild indexes
echo "Rebuilding indexes..."
psql -U ytempire_app -d ytempire -c "REINDEX TABLE analytics.fact_video_performance;"
psql -U ytempire_app -d ytempire -c "REINDEX TABLE analytics.revenue_attribution;"

# 3. Update table statistics
echo "Updating statistics..."
psql -U ytempire_app -d ytempire -c "ANALYZE;"

# 4. Clean old logs
echo "Cleaning old logs..."
find /opt/ytempire/logs -type f -mtime +30 -delete

# 5. Archive old data
echo "Archiving old data..."
psql -U ytempire_app -d ytempire -c "SELECT analytics.archive_old_data();"

# 6. Generate weekly report
echo "Generating weekly performance report..."
python3 /opt/ytempire/scripts/generate_weekly_report.py

echo "Weekly maintenance completed successfully!"
```

---

## Key Takeaways for Daily Operations

1. **Morning Priority**: Always check data freshness and overnight job status first
2. **Cost Monitoring**: Check cost metrics every hour - automate responses to thresholds
3. **Performance**: Keep all dashboard queries under 2 seconds
4. **Quality**: Run data quality checks daily, fix issues immediately
5. **Communication**: Send daily reports to stakeholders
6. **Emergency Response**: Have clear escalation paths for critical issues
7. **Documentation**: Log all manual interventions and issues

Remember: Automate everything possible, monitor everything critical, and maintain clear communication with the team!