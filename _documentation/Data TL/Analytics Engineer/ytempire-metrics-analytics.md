# YTEMPIRE Analytics Engineer Documentation
## 4. METRICS & ANALYTICS

**Document Version**: 1.0  
**Date**: January 2025  
**Status**: CONSOLIDATED - PRODUCTION READY  
**Purpose**: Complete metrics definitions and analytics frameworks

---

## 4.1 Business Metrics

### KPI Definitions

```python
# analytics/metrics_definitions.py
"""
Central metrics repository for YTEMPIRE analytics
"""

METRICS_CATALOG = {
    # Revenue Metrics
    'revenue': {
        'total_revenue': {
            'formula': 'SUM(ad_revenue + sponsorship_revenue + affiliate_revenue)',
            'description': 'Total revenue across all sources',
            'granularity': ['daily', 'channel', 'video'],
            'tier': 'executive',
            'target': None,
            'unit': 'currency'
        },
        'rpm': {
            'formula': '(total_revenue / total_views) * 1000',
            'description': 'Revenue per thousand views',
            'benchmark': 5.0,  # $5 RPM target
            'tier': 'operational',
            'unit': 'currency'
        },
        'ltv_per_subscriber': {
            'formula': 'total_revenue / unique_subscribers',
            'description': 'Lifetime value per subscriber',
            'calculation_window': '365d',
            'tier': 'strategic',
            'unit': 'currency'
        },
        'revenue_per_video': {
            'formula': 'total_revenue / video_count',
            'description': 'Average revenue generated per video',
            'target': 2.0,  # $2 per video
            'tier': 'operational',
            'unit': 'currency'
        }
    },
    
    # Growth Metrics
    'growth': {
        'channel_growth_rate': {
            'formula': '(current_subscribers - previous_subscribers) / previous_subscribers',
            'description': 'Month-over-month subscriber growth',
            'target': 0.15,  # 15% monthly growth
            'tier': 'executive',
            'unit': 'percentage'
        },
        'viral_growth_factor': {
            'formula': 'new_subscribers_from_shares / total_new_subscribers',
            'description': 'Percentage of growth from viral sharing',
            'tier': 'strategic',
            'unit': 'percentage'
        },
        'view_velocity': {
            'formula': 'views_first_48_hours / total_views',
            'description': 'Percentage of views in first 48 hours',
            'benchmark': 0.7,  # 70% of views in first 48h
            'tier': 'operational',
            'unit': 'percentage'
        }
    },
    
    # Efficiency Metrics
    'efficiency': {
        'content_roi': {
            'formula': '(revenue - production_cost) / production_cost',
            'description': 'Return on investment per video',
            'target': 10.0,  # 10x ROI
            'tier': 'executive',
            'unit': 'ratio'
        },
        'automation_rate': {
            'formula': 'automated_videos / total_videos',
            'description': 'Percentage of fully automated content',
            'target': 0.95,  # 95% automation
            'tier': 'operational',
            'unit': 'percentage'
        },
        'cost_per_view': {
            'formula': 'total_cost / total_views',
            'description': 'Cost per view generated',
            'target': 0.001,  # $0.001 per view
            'tier': 'operational',
            'unit': 'currency'
        }
    },
    
    # Engagement Metrics
    'engagement': {
        'engagement_rate': {
            'formula': '(likes + comments + shares) / views',
            'description': 'Overall engagement rate',
            'benchmark': 0.05,  # 5% engagement
            'viral_threshold': 0.10,  # 10% for viral
            'tier': 'operational',
            'unit': 'percentage'
        },
        'comment_sentiment': {
            'formula': 'positive_comments / total_comments',
            'description': 'Percentage of positive comments',
            'target': 0.80,  # 80% positive
            'tier': 'strategic',
            'unit': 'percentage'
        },
        'retention_rate': {
            'formula': 'average_view_duration / video_length',
            'description': 'Average percentage of video watched',
            'target': 0.50,  # 50% retention
            'tier': 'operational',
            'unit': 'percentage'
        }
    }
}
```

### Revenue Metrics

```sql
-- ============================================
-- REVENUE METRICS CALCULATIONS
-- ============================================

-- Daily Revenue Summary
CREATE OR REPLACE VIEW analytics.v_daily_revenue AS
SELECT 
    d.full_date,
    COUNT(DISTINCT u.user_id) AS active_users,
    COUNT(DISTINCT c.channel_key) AS active_channels,
    COUNT(DISTINCT v.video_key) AS videos_published,
    
    -- Revenue breakdown
    SUM(r.ad_revenue_cents) / 100.0 AS ad_revenue,
    SUM(r.sponsorship_revenue_cents) / 100.0 AS sponsorship_revenue,
    SUM(r.affiliate_revenue_cents) / 100.0 AS affiliate_revenue,
    SUM(r.total_revenue_cents) / 100.0 AS total_revenue,
    
    -- Cost breakdown
    SUM(vp.cost_cents) / 100.0 AS total_cost,
    
    -- Profit
    (SUM(r.total_revenue_cents) - SUM(vp.cost_cents)) / 100.0 AS profit,
    
    -- Key metrics
    CASE 
        WHEN SUM(vp.views) > 0 
        THEN (SUM(r.total_revenue_cents) / SUM(vp.views)::FLOAT * 1000) / 100.0
        ELSE 0 
    END AS rpm,
    
    CASE 
        WHEN COUNT(DISTINCT v.video_key) > 0 
        THEN SUM(r.total_revenue_cents) / COUNT(DISTINCT v.video_key) / 100.0
        ELSE 0 
    END AS revenue_per_video,
    
    CASE 
        WHEN SUM(vp.cost_cents) > 0 
        THEN (SUM(r.total_revenue_cents) - SUM(vp.cost_cents))::FLOAT / SUM(vp.cost_cents)
        ELSE 0 
    END AS roi

FROM analytics.dim_date d
LEFT JOIN analytics.fact_revenue r ON d.date_key = r.date_key
LEFT JOIN analytics.fact_video_performance vp ON r.video_key = vp.video_key AND r.date_key = vp.date_key
LEFT JOIN analytics.dim_video v ON vp.video_key = v.video_key
LEFT JOIN analytics.dim_channel c ON v.channel_id = c.channel_id
LEFT JOIN analytics.dim_user u ON c.user_id = u.user_id
WHERE d.full_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY d.full_date
ORDER BY d.full_date DESC;

-- User Revenue Tracking
CREATE OR REPLACE VIEW analytics.v_user_revenue_tracking AS
SELECT 
    u.user_id,
    u.email,
    u.subscription_tier,
    COUNT(DISTINCT c.channel_key) AS channel_count,
    COUNT(DISTINCT v.video_key) AS total_videos,
    
    -- Current month revenue
    SUM(CASE 
        WHEN d.month = EXTRACT(MONTH FROM CURRENT_DATE) 
        AND d.year = EXTRACT(YEAR FROM CURRENT_DATE)
        THEN r.total_revenue_cents 
        ELSE 0 
    END) / 100.0 AS current_month_revenue,
    
    -- Progress to $10K goal
    (SUM(CASE 
        WHEN d.month = EXTRACT(MONTH FROM CURRENT_DATE) 
        AND d.year = EXTRACT(YEAR FROM CURRENT_DATE)
        THEN r.total_revenue_cents 
        ELSE 0 
    END) / 100.0) / 10000.0 * 100 AS goal_progress_pct,
    
    -- Last 30 days revenue
    SUM(CASE 
        WHEN d.full_date >= CURRENT_DATE - INTERVAL '30 days'
        THEN r.total_revenue_cents 
        ELSE 0 
    END) / 100.0 AS revenue_30d,
    
    -- Lifetime revenue
    SUM(r.total_revenue_cents) / 100.0 AS lifetime_revenue,
    
    -- Average daily revenue
    SUM(r.total_revenue_cents) / NULLIF(COUNT(DISTINCT d.full_date), 0) / 100.0 AS avg_daily_revenue,
    
    -- Days to $10K at current rate
    CASE 
        WHEN SUM(r.total_revenue_cents) > 0 
        THEN CEIL(
            (1000000 - SUM(CASE 
                WHEN d.month = EXTRACT(MONTH FROM CURRENT_DATE) 
                AND d.year = EXTRACT(YEAR FROM CURRENT_DATE)
                THEN r.total_revenue_cents 
                ELSE 0 
            END)) / 
            (SUM(CASE 
                WHEN d.full_date >= CURRENT_DATE - INTERVAL '7 days'
                THEN r.total_revenue_cents 
                ELSE 0 
            END) / 7.0)
        )
        ELSE NULL 
    END AS days_to_10k_goal

FROM analytics.dim_user u
LEFT JOIN analytics.dim_channel c ON u.user_id = c.user_id
LEFT JOIN analytics.dim_video v ON c.channel_id = v.channel_id
LEFT JOIN analytics.fact_revenue r ON v.video_key = r.video_key
LEFT JOIN analytics.dim_date d ON r.date_key = d.date_key
WHERE u.is_active = true
GROUP BY u.user_id, u.email, u.subscription_tier
ORDER BY current_month_revenue DESC;
```

### Cost Tracking

```sql
-- ============================================
-- COST TRACKING AND ANALYSIS
-- ============================================

-- Cost per video analysis
CREATE OR REPLACE VIEW analytics.v_cost_analysis AS
SELECT 
    v.video_key,
    v.video_title,
    v.content_type,
    c.channel_name,
    
    -- Cost breakdown
    SUM(CASE WHEN fc.cost_category = 'ai' THEN fc.total_cost_cents ELSE 0 END) / 100.0 AS ai_cost,
    SUM(CASE WHEN fc.cost_category = 'api' THEN fc.total_cost_cents ELSE 0 END) / 100.0 AS api_cost,
    SUM(CASE WHEN fc.cost_category = 'storage' THEN fc.total_cost_cents ELSE 0 END) / 100.0 AS storage_cost,
    SUM(CASE WHEN fc.cost_category = 'compute' THEN fc.total_cost_cents ELSE 0 END) / 100.0 AS compute_cost,
    SUM(fc.total_cost_cents) / 100.0 AS total_cost,
    
    -- Revenue
    SUM(r.total_revenue_cents) / 100.0 AS revenue,
    
    -- Profit
    (SUM(r.total_revenue_cents) - SUM(fc.total_cost_cents)) / 100.0 AS profit,
    
    -- ROI
    CASE 
        WHEN SUM(fc.total_cost_cents) > 0 
        THEN ((SUM(r.total_revenue_cents) - SUM(fc.total_cost_cents))::FLOAT / SUM(fc.total_cost_cents)) * 100
        ELSE 0 
    END AS roi_percentage,
    
    -- Performance metrics
    MAX(vp.views) AS views,
    MAX(vp.engagement_rate) AS engagement_rate,
    
    -- Cost efficiency
    CASE 
        WHEN MAX(vp.views) > 0 
        THEN SUM(fc.total_cost_cents) / MAX(vp.views)::FLOAT 
        ELSE 0 
    END AS cost_per_view_cents,
    
    -- Status
    CASE 
        WHEN SUM(fc.total_cost_cents) > 300 THEN 'over_budget'  -- $3.00 threshold
        WHEN SUM(r.total_revenue_cents) > SUM(fc.total_cost_cents) THEN 'profitable'
        ELSE 'unprofitable'
    END AS status

FROM analytics.dim_video v
LEFT JOIN analytics.dim_channel c ON v.channel_id = c.channel_id
LEFT JOIN analytics.fact_costs fc ON v.video_id = fc.video_id
LEFT JOIN analytics.fact_revenue r ON v.video_key = r.video_key
LEFT JOIN analytics.fact_video_performance vp ON v.video_key = vp.video_key
GROUP BY v.video_key, v.video_title, v.content_type, c.channel_name
ORDER BY total_cost DESC;

-- Daily cost summary
CREATE OR REPLACE VIEW analytics.v_daily_cost_summary AS
SELECT 
    d.full_date,
    
    -- Cost by category
    SUM(CASE WHEN cost_category = 'ai' THEN total_cost_cents ELSE 0 END) / 100.0 AS ai_costs,
    SUM(CASE WHEN cost_category = 'api' THEN total_cost_cents ELSE 0 END) / 100.0 AS api_costs,
    SUM(CASE WHEN cost_category = 'storage' THEN total_cost_cents ELSE 0 END) / 100.0 AS storage_costs,
    SUM(CASE WHEN cost_category = 'compute' THEN total_cost_cents ELSE 0 END) / 100.0 AS compute_costs,
    SUM(total_cost_cents) / 100.0 AS total_costs,
    
    -- Video count
    COUNT(DISTINCT video_id) AS videos_produced,
    
    -- Average cost per video
    SUM(total_cost_cents) / NULLIF(COUNT(DISTINCT video_id), 0) / 100.0 AS avg_cost_per_video,
    
    -- Cost trend
    LAG(SUM(total_cost_cents), 1) OVER (ORDER BY d.full_date) / 100.0 AS previous_day_cost,
    (SUM(total_cost_cents) - LAG(SUM(total_cost_cents), 1) OVER (ORDER BY d.full_date)) / 100.0 AS cost_change

FROM analytics.dim_date d
LEFT JOIN analytics.fact_costs fc ON d.date_key = fc.date_key
WHERE d.full_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY d.full_date
ORDER BY d.full_date DESC;
```

---

## 4.2 Operational Metrics

### Performance Indicators

```sql
-- ============================================
-- OPERATIONAL PERFORMANCE INDICATORS
-- ============================================

-- System health metrics
CREATE OR REPLACE VIEW analytics.v_system_health AS
WITH api_metrics AS (
    SELECT 
        COUNT(*) AS total_api_calls,
        SUM(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 ELSE 0 END) AS successful_calls,
        SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END) AS server_errors,
        AVG(response_time_ms) AS avg_response_time,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) AS p95_response_time,
        MAX(response_time_ms) AS max_response_time
    FROM analytics.api_logs
    WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '5 minutes'
),
database_health AS (
    SELECT 
        datname,
        numbackends AS active_connections,
        xact_commit AS transactions_committed,
        xact_rollback AS transactions_rolled_back,
        blks_read AS blocks_read,
        blks_hit AS blocks_hit,
        (blks_hit::FLOAT / NULLIF(blks_hit + blks_read, 0)) * 100 AS cache_hit_ratio
    FROM pg_stat_database
    WHERE datname = current_database()
),
job_health AS (
    SELECT 
        COUNT(*) AS total_jobs,
        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successful_jobs,
        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_jobs,
        AVG(execution_time_seconds) AS avg_execution_time
    FROM analytics.job_execution_history
    WHERE start_time >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
)
SELECT 
    -- API Health
    api.total_api_calls,
    ROUND((api.successful_calls::FLOAT / NULLIF(api.total_api_calls, 0) * 100)::numeric, 2) AS api_success_rate,
    api.avg_response_time,
    api.p95_response_time,
    
    -- Database Health
    db.active_connections,
    db.cache_hit_ratio,
    
    -- Job Health
    job.total_jobs,
    ROUND((job.successful_jobs::FLOAT / NULLIF(job.total_jobs, 0) * 100)::numeric, 2) AS job_success_rate,
    
    -- Overall Status
    CASE 
        WHEN api.server_errors > 10 OR db.cache_hit_ratio < 90 OR job.failed_jobs > 5 THEN 'DEGRADED'
        WHEN api.server_errors > 0 OR job.failed_jobs > 0 THEN 'WARNING'
        ELSE 'HEALTHY'
    END AS overall_status
    
FROM api_metrics api
CROSS JOIN database_health db
CROSS JOIN job_health job;

-- Dashboard performance metrics
CREATE OR REPLACE VIEW analytics.v_dashboard_performance AS
SELECT 
    dashboard_name,
    query_name,
    COUNT(*) AS execution_count,
    AVG(execution_time_ms) AS avg_time_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) AS p95_time_ms,
    MAX(execution_time_ms) AS max_time_ms,
    MIN(timestamp) AS first_execution,
    MAX(timestamp) AS last_execution,
    
    -- Performance classification
    CASE 
        WHEN AVG(execution_time_ms) < 500 THEN 'EXCELLENT'
        WHEN AVG(execution_time_ms) < 2000 THEN 'GOOD'
        WHEN AVG(execution_time_ms) < 5000 THEN 'ACCEPTABLE'
        ELSE 'NEEDS_OPTIMIZATION'
    END AS performance_rating
    
FROM analytics.query_performance_log
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY dashboard_name, query_name
ORDER BY avg_time_ms DESC;
```

### System Health

```python
# monitoring/system_health.py
"""
System health monitoring and alerting
"""

import psutil
import psycopg2
from datetime import datetime
import redis

class SystemHealthMonitor:
    def __init__(self):
        self.db = psycopg2.connect(
            host="localhost",
            database="ytempire",
            user="analytics_user"
        )
        self.redis = redis.Redis(host='localhost', port=6379)
        
    def collect_metrics(self):
        """Collect system health metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            
            # CPU metrics
            'cpu': {
                'usage_percent': psutil.cpu_percent(interval=1),
                'core_count': psutil.cpu_count(),
                'load_average': psutil.getloadavg()
            },
            
            # Memory metrics
            'memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'used_gb': psutil.virtual_memory().used / (1024**3),
                'percent': psutil.virtual_memory().percent,
                'available_gb': psutil.virtual_memory().available / (1024**3)
            },
            
            # Disk metrics
            'disk': {
                'total_gb': psutil.disk_usage('/').total / (1024**3),
                'used_gb': psutil.disk_usage('/').used / (1024**3),
                'percent': psutil.disk_usage('/').percent,
                'free_gb': psutil.disk_usage('/').free / (1024**3)
            },
            
            # Database metrics
            'database': self.get_database_metrics(),
            
            # Redis metrics
            'redis': self.get_redis_metrics()
        }
        
        return metrics
    
    def get_database_metrics(self):
        """Get PostgreSQL metrics"""
        cursor = self.db.cursor()
        
        # Connection count
        cursor.execute("SELECT COUNT(*) FROM pg_stat_activity")
        connection_count = cursor.fetchone()[0]
        
        # Database size
        cursor.execute("SELECT pg_database_size(current_database()) / (1024*1024*1024.0)")
        db_size_gb = cursor.fetchone()[0]
        
        # Cache hit ratio
        cursor.execute("""
            SELECT 
                sum(blks_hit)::float / nullif(sum(blks_hit + blks_read), 0) * 100
            FROM pg_stat_database
            WHERE datname = current_database()
        """)
        cache_hit_ratio = cursor.fetchone()[0] or 0
        
        return {
            'connections': connection_count,
            'size_gb': round(db_size_gb, 2),
            'cache_hit_ratio': round(cache_hit_ratio, 2)
        }
    
    def get_redis_metrics(self):
        """Get Redis metrics"""
        info = self.redis.info()
        
        return {
            'used_memory_mb': info['used_memory'] / (1024*1024),
            'connected_clients': info['connected_clients'],
            'total_commands_processed': info['total_commands_processed'],
            'keyspace_hits': info.get('keyspace_hits', 0),
            'keyspace_misses': info.get('keyspace_misses', 0),
            'hit_ratio': round(
                info.get('keyspace_hits', 0) / 
                max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 1), 1) * 100, 
                2
            )
        }
    
    def check_thresholds(self, metrics):
        """Check if metrics exceed thresholds"""
        alerts = []
        
        # CPU threshold
        if metrics['cpu']['usage_percent'] > 80:
            alerts.append({
                'level': 'WARNING',
                'component': 'CPU',
                'message': f"CPU usage at {metrics['cpu']['usage_percent']}%"
            })
        
        # Memory threshold
        if metrics['memory']['percent'] > 85:
            alerts.append({
                'level': 'WARNING',
                'component': 'Memory',
                'message': f"Memory usage at {metrics['memory']['percent']}%"
            })
        
        # Disk threshold
        if metrics['disk']['percent'] > 80:
            alerts.append({
                'level': 'CRITICAL' if metrics['disk']['percent'] > 90 else 'WARNING',
                'component': 'Disk',
                'message': f"Disk usage at {metrics['disk']['percent']}%"
            })
        
        # Database connections
        if metrics['database']['connections'] > 90:
            alerts.append({
                'level': 'WARNING',
                'component': 'Database',
                'message': f"High connection count: {metrics['database']['connections']}"
            })
        
        return alerts
```

### API Usage

```sql
-- ============================================
-- API USAGE TRACKING AND MONITORING
-- ============================================

-- YouTube API usage tracking
CREATE TABLE IF NOT EXISTS analytics.youtube_api_usage (
    id BIGSERIAL PRIMARY KEY,
    api_method VARCHAR(100),
    endpoint VARCHAR(255),
    quota_cost INTEGER,
    user_id UUID,
    channel_id VARCHAR(50),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    response_time_ms INTEGER,
    status_code INTEGER,
    error_message TEXT
);

-- API usage summary view
CREATE OR REPLACE VIEW analytics.v_api_usage_summary AS
WITH daily_usage AS (
    SELECT 
        DATE(timestamp) AS date,
        api_method,
        COUNT(*) AS call_count,
        SUM(quota_cost) AS total_quota_used,
        AVG(response_time_ms) AS avg_response_time,
        SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) AS error_count
    FROM analytics.youtube_api_usage
    WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY DATE(timestamp), api_method
),
quota_tracking AS (
    SELECT 
        date,
        SUM(total_quota_used) AS daily_quota_used,
        10000 - SUM(total_quota_used) AS quota_remaining,
        (SUM(total_quota_used)::FLOAT / 10000) * 100 AS quota_usage_pct
    FROM daily_usage
    GROUP BY date
)
SELECT 
    d.date,
    d.api_method,
    d.call_count,
    d.total_quota_used,
    d.avg_response_time,
    d.error_count,
    q.daily_quota_used,
    q.quota_remaining,
    q.quota_usage_pct,
    
    -- Alert level
    CASE 
        WHEN q.quota_usage_pct > 90 THEN 'CRITICAL'
        WHEN q.quota_usage_pct > 75 THEN 'WARNING'
        ELSE 'OK'
    END AS quota_alert_level
    
FROM daily_usage d
JOIN quota_tracking q ON d.date = q.date
ORDER BY d.date DESC, d.total_quota_used DESC;

-- Function to check API quota
CREATE OR REPLACE FUNCTION analytics.check_api_quota()
RETURNS TABLE(
    quota_used INTEGER,
    quota_remaining INTEGER,
    usage_percentage NUMERIC,
    estimated_hours_remaining NUMERIC,
    alert_level TEXT
) AS $$
DECLARE
    v_current_usage INTEGER;
    v_hourly_rate NUMERIC;
BEGIN
    -- Get today's usage
    SELECT COALESCE(SUM(quota_cost), 0)
    INTO v_current_usage
    FROM analytics.youtube_api_usage
    WHERE DATE(timestamp) = CURRENT_DATE;
    
    -- Calculate hourly rate
    SELECT COALESCE(AVG(hourly_usage), 0)
    INTO v_hourly_rate
    FROM (
        SELECT 
            DATE_TRUNC('hour', timestamp) AS hour,
            SUM(quota_cost) AS hourly_usage
        FROM analytics.youtube_api_usage
        WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
        GROUP BY DATE_TRUNC('hour', timestamp)
    ) t;
    
    RETURN QUERY
    SELECT 
        v_current_usage AS quota_used,
        10000 - v_current_usage AS quota_remaining,
        ROUND((v_current_usage::NUMERIC / 10000) * 100, 2) AS usage_percentage,
        CASE 
            WHEN v_hourly_rate > 0 
            THEN ROUND((10000 - v_current_usage)::NUMERIC / v_hourly_rate, 2)
            ELSE 999
        END AS estimated_hours_remaining,
        CASE 
            WHEN v_current_usage > 9000 THEN 'CRITICAL'
            WHEN v_current_usage > 7500 THEN 'WARNING'
            ELSE 'OK'
        END AS alert_level;
END;
$$ LANGUAGE plpgsql;
```

---

## 4.3 Advanced Analytics

### Predictive Models

```sql
-- ============================================
-- PREDICTIVE ANALYTICS MODELS
-- ============================================

-- Video performance prediction
CREATE OR REPLACE FUNCTION analytics.predict_video_performance(
    p_channel_id VARCHAR,
    p_content_type VARCHAR,
    p_publish_hour INTEGER
)
RETURNS TABLE(
    predicted_views BIGINT,
    predicted_engagement_rate NUMERIC,
    predicted_revenue NUMERIC,
    confidence_score NUMERIC
) AS $$
DECLARE
    v_channel_avg_views NUMERIC;
    v_channel_avg_engagement NUMERIC;
    v_content_multiplier NUMERIC;
    v_hour_multiplier NUMERIC;
BEGIN
    -- Get channel historical averages
    SELECT 
        AVG(views),
        AVG(engagement_rate)
    INTO v_channel_avg_views, v_channel_avg_engagement
    FROM analytics.fact_video_performance vp
    JOIN analytics.dim_video v ON vp.video_key = v.video_key
    WHERE v.channel_id = p_channel_id
        AND vp.timestamp >= CURRENT_DATE - INTERVAL '30 days';
    
    -- Get content type multiplier
    SELECT 
        AVG(vp.views) / NULLIF(AVG(all_vp.views), 0)
    INTO v_content_multiplier
    FROM analytics.fact_video_performance vp
    JOIN analytics.dim_video v ON vp.video_key = v.video_key
    CROSS JOIN (
        SELECT AVG(views) AS views
        FROM analytics.fact_video_performance
        WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
    ) all_vp
    WHERE v.content_type = p_content_type
        AND vp.timestamp >= CURRENT_DATE - INTERVAL '30 days';
    
    -- Get hour multiplier
    SELECT 
        AVG(views) / NULLIF(
            (SELECT AVG(views) FROM analytics.fact_video_performance 
             WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'), 0)
    INTO v_hour_multiplier
    FROM analytics.fact_video_performance vp
    JOIN analytics.dim_video v ON vp.video_key = v.video_key
    WHERE EXTRACT(HOUR FROM v.published_at) = p_publish_hour
        AND vp.timestamp >= CURRENT_DATE - INTERVAL '30 days';
    
    RETURN QUERY
    SELECT 
        (v_channel_avg_views * COALESCE(v_content_multiplier, 1) * COALESCE(v_hour_multiplier, 1))::BIGINT AS predicted_views,
        v_channel_avg_engagement AS predicted_engagement_rate,
        (v_channel_avg_views * COALESCE(v_content_multiplier, 1) * COALESCE(v_hour_multiplier, 1) * 0.005)::NUMERIC AS predicted_revenue,
        LEAST(
            GREATEST(
                (1 - ABS(COALESCE(v_content_multiplier, 1) - 1)) * 
                (1 - ABS(COALESCE(v_hour_multiplier, 1) - 1)),
                0.3
            ),
            0.9
        )::NUMERIC AS confidence_score;
END;
$$ LANGUAGE plpgsql;

-- Trend detection and forecasting
CREATE OR REPLACE VIEW analytics.v_trend_forecast AS
WITH historical_data AS (
    SELECT 
        d.full_date,
        COUNT(DISTINCT v.video_key) AS videos_published,
        SUM(vp.views) AS total_views,
        SUM(r.total_revenue_cents) / 100.0 AS total_revenue
    FROM analytics.dim_date d
    LEFT JOIN analytics.fact_video_performance vp ON d.date_key = vp.date_key
    LEFT JOIN analytics.dim_video v ON vp.video_key = v.video_key
    LEFT JOIN analytics.fact_revenue r ON vp.video_key = r.video_key
    WHERE d.full_date >= CURRENT_DATE - INTERVAL '90 days'
        AND d.full_date < CURRENT_DATE
    GROUP BY d.full_date
),
trend_analysis AS (
    SELECT 
        full_date,
        videos_published,
        total_views,
        total_revenue,
        
        -- Moving averages
        AVG(videos_published) OVER (ORDER BY full_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7_videos,
        AVG(total_views) OVER (ORDER BY full_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7_views,
        AVG(total_revenue) OVER (ORDER BY full_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7_revenue,
        
        -- Trend (slope of linear regression over 7 days)
        REGR_SLOPE(videos_published, EXTRACT(EPOCH FROM full_date)) 
            OVER (ORDER BY full_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS videos_trend,
        REGR_SLOPE(total_views, EXTRACT(EPOCH FROM full_date)) 
            OVER (ORDER BY full_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS views_trend,
        REGR_SLOPE(total_revenue, EXTRACT(EPOCH FROM full_date)) 
            OVER (ORDER BY full_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS revenue_trend
            
    FROM historical_data
)
SELECT 
    full_date,
    videos_published,
    total_views,
    total_revenue,
    ma7_videos,
    ma7_views,
    ma7_revenue,
    
    -- Forecast next 7 days (simple linear projection)
    ma7_videos + (videos_trend * 86400 * 7) AS forecast_videos_7d,
    ma7_views + (views_trend * 86400 * 7) AS forecast_views_7d,
    ma7_revenue + (revenue_trend * 86400 * 7) AS forecast_revenue_7d,
    
    -- Trend classification
    CASE 
        WHEN revenue_trend > 0 AND views_trend > 0 THEN 'GROWTH'
        WHEN revenue_trend > 0 AND views_trend <= 0 THEN 'MONETIZATION_IMPROVING'
        WHEN revenue_trend <= 0 AND views_trend > 0 THEN 'MONETIZATION_DECLINING'
        ELSE 'DECLINING'
    END AS trend_status
    
FROM trend_analysis
WHERE full_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY full_date DESC;
```

### A/B Testing Framework

```sql
-- ============================================
-- A/B TESTING FRAMEWORK
-- ============================================

-- A/B Test Configuration
CREATE TABLE analytics.ab_tests (
    test_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_name VARCHAR(255) NOT NULL,
    test_type VARCHAR(50) NOT NULL, -- 'split', 'multivariate', 'bandit'
    status VARCHAR(50) DEFAULT 'draft', -- 'draft', 'running', 'completed', 'stopped'
    
    -- Test details
    hypothesis TEXT,
    primary_metric VARCHAR(100) NOT NULL,
    secondary_metrics JSONB,
    
    -- Configuration
    traffic_allocation DECIMAL(5,2) DEFAULT 100.0, -- Percentage of traffic
    min_sample_size INTEGER,
    max_duration_days INTEGER DEFAULT 30,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    
    -- Results
    winner_variant VARCHAR(50),
    decision_notes TEXT,
    
    -- Metadata
    created_by VARCHAR(100),
    tags TEXT[]
);

-- Test Variants
CREATE TABLE analytics.ab_test_variants (
    variant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id UUID REFERENCES analytics.ab_tests(test_id),
    variant_name VARCHAR(50) NOT NULL,
    variant_type VARCHAR(20) DEFAULT 'treatment', -- 'control', 'treatment'
    
    -- Configuration
    traffic_split DECIMAL(5,2), -- Percentage for this variant
    configuration JSONB, -- Variant-specific config
    
    -- Tracking
    is_active BOOLEAN DEFAULT true,
    
    UNIQUE(test_id, variant_name)
);

-- Test Results
CREATE TABLE analytics.ab_test_results (
    result_id BIGSERIAL PRIMARY KEY,
    test_id UUID REFERENCES analytics.ab_tests(test_id),
    variant_id UUID REFERENCES analytics.ab_test_variants(variant_id),
    
    -- Metrics
    sample_size INTEGER,
    conversions INTEGER,
    conversion_rate DECIMAL(5,4),
    avg_metric_value DECIMAL(12,4),
    
    -- Statistical measures
    confidence_level DECIMAL(5,4),
    p_value DECIMAL(10,9),
    is_significant BOOLEAN,
    
    -- Timestamp
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Function to analyze A/B test results
CREATE OR REPLACE FUNCTION analytics.analyze_ab_test(p_test_id UUID)
RETURNS TABLE(
    variant_name VARCHAR,
    sample_size INTEGER,
    conversion_rate NUMERIC,
    confidence_level NUMERIC,
    is_winner BOOLEAN,
    lift_percentage NUMERIC
) AS $
DECLARE
    v_control_rate NUMERIC;
BEGIN
    -- Get control conversion rate
    SELECT conversion_rate 
    INTO v_control_rate
    FROM analytics.ab_test_results r
    JOIN analytics.ab_test_variants v ON r.variant_id = v.variant_id
    WHERE r.test_id = p_test_id 
        AND v.variant_type = 'control'
    ORDER BY r.calculated_at DESC
    LIMIT 1;
    
    RETURN QUERY
    SELECT 
        v.variant_name,
        r.sample_size,
        r.conversion_rate,
        r.confidence_level,
        r.is_significant AND r.conversion_rate > v_control_rate AS is_winner,
        CASE 
            WHEN v_control_rate > 0 
            THEN ((r.conversion_rate - v_control_rate) / v_control_rate * 100)
            ELSE 0 
        END AS lift_percentage
    FROM analytics.ab_test_results r
    JOIN analytics.ab_test_variants v ON r.variant_id = v.variant_id
    WHERE r.test_id = p_test_id
        AND r.calculated_at = (
            SELECT MAX(calculated_at) 
            FROM analytics.ab_test_results 
            WHERE test_id = p_test_id
        )
    ORDER BY r.conversion_rate DESC;
END;
$ LANGUAGE plpgsql;