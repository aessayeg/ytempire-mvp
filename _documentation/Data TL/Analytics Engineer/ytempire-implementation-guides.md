# YTEMPIRE Analytics Engineer Documentation
## 3. IMPLEMENTATION GUIDES

**Document Version**: 1.0  
**Date**: January 2025  
**Status**: CONSOLIDATED - PRODUCTION READY  
**Purpose**: Complete implementation guides for SQL, dashboards, and data quality

---

## 3.1 SQL Development

### Query Patterns Library

#### Executive Dashboard Queries

```sql
-- ============================================
-- CEO DASHBOARD: HIGH-LEVEL COMPANY METRICS
-- Refresh: Every 5 minutes
-- Performance target: < 2 seconds
-- ============================================

WITH current_period_metrics AS (
    SELECT 
        -- Channel metrics
        COUNT(DISTINCT c.channel_key) AS active_channels,
        SUM(CASE WHEN dc.created_date >= CURRENT_DATE - INTERVAL '30 days' THEN 1 ELSE 0 END) AS new_channels_30d,
        
        -- Video metrics
        COUNT(DISTINCT v.video_key) AS total_videos,
        SUM(CASE WHEN v.published_at >= CURRENT_DATE THEN 1 ELSE 0 END) AS videos_today,
        SUM(CASE WHEN v.published_at >= CURRENT_DATE - INTERVAL '7 days' THEN 1 ELSE 0 END) AS videos_week,
        
        -- View metrics
        SUM(vp.views) AS total_views,
        SUM(CASE WHEN d.full_date = CURRENT_DATE THEN vp.views ELSE 0 END) AS views_today,
        
        -- Revenue metrics
        SUM(r.total_revenue_cents) / 100.0 AS total_revenue,
        SUM(CASE WHEN d.full_date = CURRENT_DATE THEN r.total_revenue_cents ELSE 0 END) / 100.0 AS revenue_today,
        
        -- Cost metrics
        SUM(vp.cost_cents) / 100.0 AS total_costs,
        
        -- Profit
        (SUM(r.total_revenue_cents) - SUM(vp.cost_cents)) / 100.0 AS total_profit
        
    FROM analytics.dim_channel dc
    LEFT JOIN analytics.fact_video_performance vp ON dc.channel_key = vp.channel_key
    LEFT JOIN analytics.dim_video v ON vp.video_key = v.video_key
    LEFT JOIN analytics.fact_revenue r ON vp.video_key = r.video_key
    LEFT JOIN analytics.dim_date d ON vp.date_key = d.date_key
    WHERE dc.is_active = true
        AND vp.timestamp >= CURRENT_DATE - INTERVAL '30 days'
),
growth_metrics AS (
    SELECT 
        -- Calculate week-over-week growth
        curr.total_views AS current_views,
        prev.total_views AS previous_views,
        CASE 
            WHEN prev.total_views > 0 
            THEN ((curr.total_views - prev.total_views)::FLOAT / prev.total_views * 100)
            ELSE 0 
        END AS view_growth_pct,
        
        curr.total_revenue AS current_revenue,
        prev.total_revenue AS previous_revenue,
        CASE 
            WHEN prev.total_revenue > 0 
            THEN ((curr.total_revenue - prev.total_revenue) / prev.total_revenue * 100)
            ELSE 0 
        END AS revenue_growth_pct
        
    FROM (
        SELECT 
            SUM(views) AS total_views,
            SUM(actual_revenue_cents) / 100.0 AS total_revenue
        FROM analytics.fact_video_performance
        WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
    ) curr,
    (
        SELECT 
            SUM(views) AS total_views,
            SUM(actual_revenue_cents) / 100.0 AS total_revenue
        FROM analytics.fact_video_performance
        WHERE timestamp >= CURRENT_DATE - INTERVAL '14 days'
            AND timestamp < CURRENT_DATE - INTERVAL '7 days'
    ) prev
)
SELECT 
    cpm.*,
    gm.view_growth_pct,
    gm.revenue_growth_pct,
    -- Calculate key ratios
    CASE 
        WHEN cpm.total_views > 0 
        THEN (cpm.total_revenue / cpm.total_views * 1000) 
        ELSE 0 
    END AS rpm,  -- Revenue per mille
    CASE 
        WHEN cpm.total_videos > 0 
        THEN cpm.total_costs / cpm.total_videos 
        ELSE 0 
    END AS cost_per_video,
    CASE 
        WHEN cpm.total_costs > 0 
        THEN (cpm.total_revenue / cpm.total_costs) 
        ELSE 0 
    END AS roi
FROM current_period_metrics cpm
CROSS JOIN growth_metrics gm;
```

#### Channel Performance Analysis

```sql
-- ============================================
-- CHANNEL PERFORMANCE MATRIX
-- Purpose: Analyze all channels performance
-- Refresh: Every 15 minutes
-- ============================================

WITH channel_performance AS (
    SELECT 
        c.channel_key,
        c.channel_name,
        c.niche,
        u.email AS user_email,
        COUNT(DISTINCT v.video_key) AS video_count,
        SUM(vp.views) AS total_views,
        AVG(vp.engagement_rate) AS avg_engagement,
        SUM(vp.actual_revenue_cents) / 100.0 AS total_revenue,
        SUM(vp.cost_cents) / 100.0 AS total_cost,
        (SUM(vp.actual_revenue_cents) - SUM(vp.cost_cents)) / 100.0 AS profit,
        AVG(vp.average_view_duration_seconds) AS avg_watch_time,
        
        -- Growth metrics
        SUM(CASE WHEN vp.timestamp >= CURRENT_DATE - INTERVAL '7 days' THEN vp.views ELSE 0 END) AS views_7d,
        SUM(CASE WHEN vp.timestamp >= CURRENT_DATE - INTERVAL '30 days' THEN vp.views ELSE 0 END) AS views_30d,
        
        -- Calculate viral rate
        SUM(CASE WHEN vp.engagement_rate > 0.10 THEN 1 ELSE 0 END)::FLOAT / 
            NULLIF(COUNT(DISTINCT v.video_key), 0) AS viral_rate
            
    FROM analytics.dim_channel c
    LEFT JOIN analytics.dim_user u ON c.user_id = u.user_id
    LEFT JOIN analytics.dim_video v ON c.channel_id = v.channel_id
    LEFT JOIN analytics.fact_video_performance vp ON v.video_key = vp.video_key
    WHERE vp.timestamp >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY c.channel_key, c.channel_name, c.niche, u.email
),
channel_ranking AS (
    SELECT 
        *,
        RANK() OVER (ORDER BY total_revenue DESC) AS revenue_rank,
        RANK() OVER (ORDER BY total_views DESC) AS views_rank,
        RANK() OVER (ORDER BY profit DESC) AS profit_rank,
        RANK() OVER (ORDER BY avg_engagement DESC) AS engagement_rank,
        
        -- Performance tier
        CASE 
            WHEN profit > 1000 AND avg_engagement > 0.05 THEN 'star_performer'
            WHEN profit > 500 OR avg_engagement > 0.03 THEN 'high_performer'
            WHEN profit > 0 THEN 'profitable'
            WHEN total_views > 10000 THEN 'high_reach'
            ELSE 'needs_optimization'
        END AS performance_tier
        
    FROM channel_performance
)
SELECT * FROM channel_ranking
ORDER BY profit_rank;
```

#### Video Performance Trends

```sql
-- ============================================
-- VIDEO PERFORMANCE TREND ANALYSIS
-- Purpose: Track video performance over time
-- ============================================

WITH video_metrics AS (
    SELECT 
        DATE_TRUNC('day', vp.timestamp) AS day,
        v.video_key,
        v.video_title,
        v.content_type,
        MAX(vp.views) AS daily_views,
        MAX(vp.engagement_rate) AS daily_engagement,
        MAX(vp.actual_revenue_cents) / 100.0 AS daily_revenue,
        
        -- Time since publish
        EXTRACT(DAY FROM (DATE_TRUNC('day', vp.timestamp) - v.published_at)) AS days_since_publish
        
    FROM analytics.fact_video_performance vp
    JOIN analytics.dim_video v ON vp.video_key = v.video_key
    WHERE vp.timestamp >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY DATE_TRUNC('day', vp.timestamp), v.video_key, v.video_title, v.content_type, v.published_at
),
video_growth AS (
    SELECT 
        video_key,
        video_title,
        content_type,
        day,
        daily_views,
        LAG(daily_views) OVER (PARTITION BY video_key ORDER BY day) AS prev_views,
        daily_views - LAG(daily_views) OVER (PARTITION BY video_key ORDER BY day) AS view_growth,
        daily_revenue,
        SUM(daily_revenue) OVER (PARTITION BY video_key ORDER BY day) AS cumulative_revenue,
        
        -- Lifecycle stage
        CASE
            WHEN days_since_publish <= 1 THEN 'launch'
            WHEN days_since_publish <= 7 THEN 'early'
            WHEN days_since_publish <= 30 THEN 'growth'
            ELSE 'mature'
        END AS lifecycle_stage
        
    FROM video_metrics
)
SELECT 
    *,
    CASE 
        WHEN prev_views > 0 
        THEN (view_growth::FLOAT / prev_views * 100) 
        ELSE 0 
    END AS growth_rate_pct
FROM video_growth
WHERE day = CURRENT_DATE - INTERVAL '1 day'
ORDER BY daily_views DESC
LIMIT 100;
```

### Performance Optimization

#### Index Strategy

```sql
-- ============================================
-- PERFORMANCE OPTIMIZATION INDEXES
-- ============================================

-- Drop existing indexes if needed for rebuild
-- DROP INDEX IF EXISTS idx_name;

-- Core performance indexes
CREATE INDEX CONCURRENTLY idx_video_perf_composite 
ON analytics.fact_video_performance(channel_key, timestamp DESC, views)
WHERE timestamp >= NOW() - INTERVAL '30 days';

CREATE INDEX CONCURRENTLY idx_revenue_date_channel 
ON analytics.fact_revenue(date_key DESC, channel_key, total_revenue_cents);

CREATE INDEX CONCURRENTLY idx_channel_user_active
ON analytics.dim_channel(user_id, is_active)
WHERE is_active = true;

-- Specialized indexes for common queries
CREATE INDEX CONCURRENTLY idx_video_content_type
ON analytics.dim_video(content_type, published_at DESC);

CREATE INDEX CONCURRENTLY idx_costs_category_time
ON analytics.fact_costs(cost_category, cost_timestamp DESC);

-- Partial indexes for recent data
CREATE INDEX CONCURRENTLY idx_recent_performance
ON analytics.fact_video_performance(timestamp DESC)
WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days';

-- JSON indexes for JSONB columns
CREATE INDEX idx_channel_target_audience 
ON analytics.dim_channel USING GIN(target_audience);

CREATE INDEX idx_user_dashboard_prefs 
ON analytics.user_dashboard_config USING GIN(dashboard_preferences);

-- Function to analyze index usage
CREATE OR REPLACE FUNCTION analytics.analyze_index_usage()
RETURNS TABLE(
    schemaname TEXT,
    tablename TEXT,
    indexname TEXT,
    index_size TEXT,
    idx_scan BIGINT,
    idx_tup_read BIGINT,
    idx_tup_fetch BIGINT,
    is_unique BOOLEAN,
    is_primary BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.schemaname::TEXT,
        s.tablename::TEXT,
        s.indexrelname::TEXT,
        pg_size_pretty(pg_relation_size(s.indexrelid))::TEXT,
        s.idx_scan,
        s.idx_tup_read,
        s.idx_tup_fetch,
        i.indisunique,
        i.indisprimary
    FROM pg_stat_user_indexes s
    JOIN pg_index i ON s.indexrelid = i.indexrelid
    WHERE s.schemaname = 'analytics'
    ORDER BY s.idx_scan;
END;
$$ LANGUAGE plpgsql;
```

#### Query Performance Monitoring

```sql
-- ============================================
-- QUERY PERFORMANCE MONITORING
-- ============================================

-- Enable query performance tracking
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Performance monitoring table
CREATE TABLE IF NOT EXISTS analytics.query_performance_log (
    id BIGSERIAL PRIMARY KEY,
    query_name VARCHAR(100),
    execution_time_ms INTEGER,
    rows_returned INTEGER,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    success BOOLEAN DEFAULT true,
    error_message TEXT
);

-- Function to log query performance
CREATE OR REPLACE FUNCTION analytics.log_query_performance(
    p_query_name VARCHAR,
    p_start_time TIMESTAMPTZ,
    p_rows INTEGER DEFAULT 0,
    p_success BOOLEAN DEFAULT true,
    p_error TEXT DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    INSERT INTO analytics.query_performance_log 
        (query_name, execution_time_ms, rows_returned, success, error_message)
    VALUES 
        (p_query_name, 
         EXTRACT(MILLISECONDS FROM (NOW() - p_start_time))::INTEGER,
         p_rows,
         p_success,
         p_error);
END;
$$ LANGUAGE plpgsql;

-- Function to identify slow queries
CREATE OR REPLACE FUNCTION analytics.get_slow_queries()
RETURNS TABLE(
    query TEXT,
    calls BIGINT,
    mean_time DOUBLE PRECISION,
    total_time DOUBLE PRECISION,
    min_time DOUBLE PRECISION,
    max_time DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        query::TEXT,
        calls,
        mean_exec_time as mean_time,
        total_exec_time as total_time,
        min_exec_time as min_time,
        max_exec_time as max_time
    FROM pg_stat_statements
    WHERE query LIKE '%analytics%'
      AND mean_exec_time > 1000  -- Queries taking more than 1 second
    ORDER BY mean_exec_time DESC
    LIMIT 20;
END;
$$ LANGUAGE plpgsql;

-- Materialized view refresh optimization
CREATE OR REPLACE FUNCTION analytics.refresh_materialized_views()
RETURNS void AS $$
BEGIN
    -- Refresh in dependency order
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.video_performance_hourly;
    
    -- Log refresh completion
    INSERT INTO analytics.query_performance_log(query_name, execution_time_ms)
    VALUES ('refresh_materialized_views', 0);
END;
$$ LANGUAGE plpgsql;

-- Schedule with pg_cron (if available) or N8N
-- SELECT cron.schedule('refresh_analytics_views', '0 * * * *', 'SELECT analytics.refresh_materialized_views()');
```

### Best Practices

#### SQL Style Guide

```sql
-- ============================================
-- SQL BEST PRACTICES AND STYLE GUIDE
-- ============================================

/*
1. QUERY STRUCTURE
   - Use CTEs for complex queries
   - One CTE per logical unit of work
   - Name CTEs descriptively
   - Comment complex logic
*/

-- GOOD: Clear CTE structure
WITH user_metrics AS (
    -- Get user-level aggregations
    SELECT 
        user_key,
        COUNT(DISTINCT channel_key) AS channel_count,
        SUM(revenue_cents) AS total_revenue_cents
    FROM analytics.fact_revenue
    WHERE date_key >= 20250101
    GROUP BY user_key
),
user_rankings AS (
    -- Rank users by revenue
    SELECT 
        *,
        RANK() OVER (ORDER BY total_revenue_cents DESC) AS revenue_rank
    FROM user_metrics
)
SELECT * FROM user_rankings WHERE revenue_rank <= 10;

/*
2. PERFORMANCE PATTERNS
   - Use appropriate data types in comparisons
   - Leverage window functions over self-joins
   - Use FILTER clause for conditional aggregation
   - Avoid SELECT * in production
*/

-- GOOD: Efficient conditional aggregation
SELECT 
    channel_key,
    COUNT(*) AS total_videos,
    COUNT(*) FILTER (WHERE engagement_rate > 0.05) AS high_engagement_videos,
    AVG(views) AS avg_views,
    AVG(views) FILTER (WHERE content_type = 'shorts') AS avg_shorts_views
FROM analytics.fact_video_performance
GROUP BY channel_key;

-- BAD: Inefficient CASE in aggregation
SELECT 
    channel_key,
    COUNT(CASE WHEN engagement_rate > 0.05 THEN 1 END) AS high_engagement_videos
FROM analytics.fact_video_performance
GROUP BY channel_key;

/*
3. JOIN OPTIMIZATION
   - Always use explicit JOIN syntax
   - Join on indexed columns
   - Filter early in subqueries
   - Consider join order for performance
*/

-- GOOD: Optimized join with early filtering
SELECT 
    c.channel_name,
    SUM(vp.views) AS total_views
FROM analytics.dim_channel c
INNER JOIN (
    SELECT channel_key, views
    FROM analytics.fact_video_performance
    WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'  -- Filter early
) vp ON c.channel_key = vp.channel_key
WHERE c.is_active = true
GROUP BY c.channel_name;

/*
4. NULL HANDLING
   - Use COALESCE for default values
   - Use NULLIF to avoid division by zero
   - Be explicit about NULL handling in JOINs
*/

-- GOOD: Safe division and NULL handling
SELECT 
    channel_key,
    COALESCE(revenue_cents, 0) AS revenue,
    COALESCE(revenue_cents / NULLIF(views, 0), 0) AS revenue_per_view
FROM analytics.fact_video_performance;

/*
5. DATE/TIME HANDLING
   - Use date functions for date math
   - Store timestamps in UTC
   - Use appropriate time zones for display
*/

-- GOOD: Proper date handling
SELECT 
    DATE_TRUNC('week', timestamp AT TIME ZONE 'UTC') AS week,
    COUNT(*) AS video_count
FROM analytics.fact_video_performance
WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE_TRUNC('week', timestamp AT TIME ZONE 'UTC');
```

---

## 3.2 Dashboard Development

### Grafana Setup

#### Initial Configuration

```yaml
# grafana/datasources/postgresql.yml
apiVersion: 1

datasources:
  - name: PostgreSQL-Analytics
    type: postgres
    access: proxy
    url: localhost:5432
    database: ytempire
    user: analytics_reader
    jsonData:
      sslmode: 'require'
      maxOpenConns: 10
      maxIdleConns: 2
      connMaxLifetime: 14400
      postgresVersion: 1500  # PostgreSQL 15
      timescaledb: true
    secureJsonData:
      password: ${POSTGRES_PASSWORD}
    editable: false
    
  - name: Redis-Cache
    type: redis-datasource
    access: proxy
    url: redis://localhost:6379
    jsonData:
      client: standalone
    editable: false
```

#### Dashboard JSON Template

```json
{
  "dashboard": {
    "title": "YTEMPIRE Executive Dashboard",
    "uid": "ytempire-executive",
    "timezone": "browser",
    "refresh": "5m",
    "panels": [
      {
        "id": 1,
        "type": "stat",
        "title": "Active Channels",
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0},
        "targets": [
          {
            "datasource": "PostgreSQL-Analytics",
            "rawSql": "SELECT COUNT(*) FROM analytics.dim_channel WHERE is_active = true",
            "format": "time_series"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 50},
                {"color": "green", "value": 100}
              ]
            },
            "unit": "short"
          }
        }
      },
      {
        "id": 2,
        "type": "graph",
        "title": "Revenue Trend",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
        "targets": [
          {
            "datasource": "PostgreSQL-Analytics",
            "rawSql": "SELECT timestamp, SUM(actual_revenue_cents)/100 as revenue FROM analytics.fact_video_performance WHERE timestamp >= NOW() - INTERVAL '30 days' GROUP BY timestamp ORDER BY timestamp",
            "format": "time_series"
          }
        ]
      }
    ]
  }
}
```

### Dashboard Templates

#### Executive Dashboard Configuration

```yaml
# dashboards/executive_dashboard.yml
dashboard:
  name: "YTEMPIRE Executive Dashboard"
  refresh_rate: "5 minutes"
  
  sections:
    - name: "Empire Overview"
      layout: "grid"
      widgets:
        - type: "scorecard"
          title: "Active Channels"
          metric: "COUNT(DISTINCT channel_id)"
          comparison: "previous_period"
          
        - type: "scorecard"
          title: "Daily Videos"
          metric: "COUNT(DISTINCT video_id)"
          target: 300
          
        - type: "scorecard"
          title: "Total Revenue"
          metric: "SUM(revenue)"
          format: "currency"
          
        - type: "scorecard"
          title: "Automation Rate"
          metric: "automation_percentage"
          format: "percentage"
          target: 99.9
    
    - name: "Performance Trends"
      widgets:
        - type: "time_series"
          title: "Revenue Growth"
          metrics: ["daily_revenue", "7d_ma_revenue", "30d_ma_revenue"]
          granularity: "daily"
          
        - type: "combo_chart"
          title: "Views vs Engagement"
          metrics: 
            bars: ["total_views"]
            line: ["engagement_rate"]
          
    - name: "Channel Performance Matrix"
      widgets:
        - type: "heatmap"
          title: "Channel Performance by Hour"
          dimensions: ["channel_name", "publish_hour"]
          metric: "avg_views"
          
        - type: "scatter"
          title: "Revenue vs Effort"
          x_axis: "production_time_minutes"
          y_axis: "total_revenue"
          size: "views"
          color: "performance_tier"
```

#### User Dashboard Template

```yaml
# dashboards/user_dashboard.yml
dashboard:
  name: "User Revenue Tracker"
  refresh_rate: "5 minutes"
  user_specific: true
  
  sections:
    - name: "Revenue Progress"
      widgets:
        - type: "gauge"
          title: "Monthly Revenue Progress"
          metric: "current_month_revenue"
          target: 10000
          thresholds:
            - value: 0
              color: "red"
            - value: 5000
              color: "yellow"
            - value: 10000
              color: "green"
        
        - type: "line_chart"
          title: "Daily Revenue Trend"
          metric: "daily_revenue"
          period: "30_days"
          show_target_line: true
          target: 333  # Daily target for $10k/month
        
    - name: "Channel Analytics"
      widgets:
        - type: "table"
          title: "Channel Performance"
          columns:
            - field: "channel_name"
              title: "Channel"
            - field: "videos_today"
              title: "Videos Today"
            - field: "views_today"
              title: "Views"
            - field: "revenue_today"
              title: "Revenue"
              format: "currency"
            - field: "engagement_rate"
              title: "Engagement"
              format: "percentage"
        
    - name: "Cost Analysis"
      widgets:
        - type: "pie_chart"
          title: "Cost Breakdown"
          dimensions: ["cost_category"]
          metric: "total_cost"
        
        - type: "bar_chart"
          title: "ROI by Channel"
          dimension: "channel_name"
          metrics: ["revenue", "cost", "profit"]
```

### Visualization Standards

```javascript
// grafana/visualization_standards.js

const YTEMPIRE_THEME = {
  colors: {
    primary: '#7C3AED',    // Purple
    success: '#10B981',    // Green
    warning: '#F59E0B',    // Amber
    danger: '#EF4444',     // Red
    info: '#3B82F6',       // Blue
    
    // Chart colors
    series: [
      '#7C3AED', '#10B981', '#F59E0B', '#3B82F6', '#EF4444',
      '#8B5CF6', '#34D399', '#FCD34D', '#60A5FA', '#F87171'
    ]
  },
  
  thresholds: {
    engagement_rate: [
      { value: 0, color: 'danger' },
      { value: 0.02, color: 'warning' },
      { value: 0.05, color: 'success' },
      { value: 0.10, color: 'primary' }  // Viral
    ],
    
    cost_per_video: [
      { value: 0, color: 'success' },
      { value: 2.0, color: 'warning' },
      { value: 3.0, color: 'danger' }
    ],
    
    revenue_per_video: [
      { value: 0, color: 'danger' },
      { value: 1.0, color: 'warning' },
      { value: 2.0, color: 'success' }
    ]
  },
  
  formatting: {
    currency: {
      prefix: ',
      decimals: 2
    },
    
    percentage: {
      suffix: '%',
      decimals: 1,
      factor: 100
    },
    
    largeNumbers: {
      notation: 'compact',
      maximumFractionDigits: 1
    }
  }
};

// Panel configuration standards
const PANEL_STANDARDS = {
  stat: {
    orientation: 'auto',
    textMode: 'value_and_name',
    colorMode: 'value',
    graphMode: 'area',
    justifyMode: 'center'
  },
  
  timeseries: {
    legend: {
      displayMode: 'list',
      placement: 'bottom'
    },
    tooltip: {
      mode: 'multi'
    },
    lineWidth: 2,
    fillOpacity: 10,
    pointSize: 5,
    showPoints: 'never',
    spanNulls: true
  },
  
  table: {
    showHeader: true,
    frameIndex: 0,
    sortBy: {
      displayName: 'Revenue',
      desc: true
    }
  }
};
```

---

## 3.3 Data Quality

### Validation Frameworks

```sql
-- ============================================
-- DATA QUALITY VALIDATION FRAMEWORK
-- ============================================

-- Create data quality tracking table
CREATE TABLE IF NOT EXISTS analytics.data_quality_checks (
    check_id BIGSERIAL PRIMARY KEY,
    check_name VARCHAR(100) NOT NULL,
    check_type VARCHAR(50), -- 'completeness', 'accuracy', 'consistency', 'timeliness'
    table_name VARCHAR(100),
    status VARCHAR(20), -- 'passed', 'failed', 'warning'
    details JSONB,
    check_timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Completeness checks
CREATE OR REPLACE FUNCTION analytics.check_data_completeness()
RETURNS TABLE(
    check_name TEXT,
    status TEXT,
    missing_percentage NUMERIC,
    details TEXT
) AS $
BEGIN
    -- Check for missing channel data
    RETURN QUERY
    SELECT 
        'channel_completeness'::TEXT,
        CASE 
            WHEN COUNT(*) FILTER (WHERE channel_name IS NULL) > 0 
            THEN 'failed'::TEXT 
            ELSE 'passed'::TEXT 
        END,
        ROUND(COUNT(*) FILTER (WHERE channel_name IS NULL)::NUMERIC / COUNT(*) * 100, 2),
        FORMAT('Missing channel names: %s', COUNT(*) FILTER (WHERE channel_name IS NULL))::TEXT
    FROM analytics.dim_channel;
    
    -- Check for missing video metrics
    RETURN QUERY
    SELECT 
        'video_metrics_completeness'::TEXT,
        CASE 
            WHEN missing_pct > 1 THEN 'failed'::TEXT
            WHEN missing_pct > 0 THEN 'warning'::TEXT
            ELSE 'passed'::TEXT
        END,
        missing_pct,
        FORMAT('Videos without metrics in last 24h: %s', missing_count)::TEXT
    FROM (
        SELECT 
            COUNT(*) FILTER (WHERE vp.video_key IS NULL) AS missing_count,
            ROUND(COUNT(*) FILTER (WHERE vp.video_key IS NULL)::NUMERIC / COUNT(*) * 100, 2) AS missing_pct
        FROM analytics.dim_video v
        LEFT JOIN analytics.fact_video_performance vp 
            ON v.video_key = vp.video_key 
            AND vp.timestamp >= CURRENT_DATE - INTERVAL '1 day'
        WHERE v.published_at >= CURRENT_DATE - INTERVAL '1 day'
    ) t;
END;
$ LANGUAGE plpgsql;

-- Accuracy checks
CREATE OR REPLACE FUNCTION analytics.check_data_accuracy()
RETURNS TABLE(
    check_name TEXT,
    status TEXT,
    error_rate NUMERIC,
    details TEXT
) AS $
BEGIN
    -- Check for invalid engagement rates
    RETURN QUERY
    SELECT 
        'engagement_rate_validity'::TEXT,
        CASE 
            WHEN COUNT(*) > 0 THEN 'failed'::TEXT 
            ELSE 'passed'::TEXT 
        END,
        COUNT(*)::NUMERIC,
        FORMAT('Invalid engagement rates (>1 or <0): %s', COUNT(*))::TEXT
    FROM analytics.fact_video_performance
    WHERE engagement_rate > 1 OR engagement_rate < 0;
    
    -- Check for cost anomalies
    RETURN QUERY
    SELECT 
        'cost_anomalies'::TEXT,
        CASE 
            WHEN COUNT(*) > 0 THEN 'warning'::TEXT 
            ELSE 'passed'::TEXT 
        END,
        COUNT(*)::NUMERIC,
        FORMAT('Videos with cost > $5: %s', COUNT(*))::TEXT
    FROM analytics.fact_video_performance
    WHERE cost_cents > 500;  -- $5.00 threshold
    
    -- Revenue reconciliation
    RETURN QUERY
    WITH revenue_comparison AS (
        SELECT 
            d.full_date,
            COALESCE(f.revenue, 0) AS fact_revenue,
            COALESCE(y.revenue, 0) AS youtube_revenue,
            ABS(COALESCE(f.revenue, 0) - COALESCE(y.revenue, 0)) AS difference
        FROM analytics.dim_date d
        LEFT JOIN (
            SELECT date_key, SUM(total_revenue_cents)/100.0 AS revenue
            FROM analytics.fact_revenue
            GROUP BY date_key
        ) f ON d.date_key = f.date_key
        LEFT JOIN (
            SELECT date, SUM(revenue_usd) AS revenue
            FROM analytics.youtube_metrics
            GROUP BY date
        ) y ON d.full_date = y.date
        WHERE d.full_date = CURRENT_DATE - 1
    )
    SELECT 
        'revenue_reconciliation'::TEXT,
        CASE 
            WHEN MAX(difference) > 10 THEN 'failed'::TEXT
            WHEN MAX(difference) > 1 THEN 'warning'::TEXT
            ELSE 'passed'::TEXT
        END,
        ROUND(MAX(difference)::NUMERIC, 2),
        FORMAT('Max revenue discrepancy: $%s', MAX(difference))::TEXT
    FROM revenue_comparison;
END;
$ LANGUAGE plpgsql;

-- Consistency checks
CREATE OR REPLACE FUNCTION analytics.check_data_consistency()
RETURNS TABLE(
    check_name TEXT,
    status TEXT,
    inconsistency_count NUMERIC,
    details TEXT
) AS $
BEGIN
    -- Check for duplicate videos
    RETURN QUERY
    SELECT 
        'duplicate_videos'::TEXT,
        CASE 
            WHEN COUNT(*) > 0 THEN 'failed'::TEXT 
            ELSE 'passed'::TEXT 
        END,
        COUNT(*)::NUMERIC,
        FORMAT('Duplicate video IDs: %s', COUNT(*))::TEXT
    FROM (
        SELECT video_id, COUNT(*) as cnt
        FROM analytics.dim_video
        GROUP BY video_id
        HAVING COUNT(*) > 1
    ) t;
    
    -- Check channel-user relationship consistency
    RETURN QUERY
    SELECT 
        'channel_user_consistency'::TEXT,
        CASE 
            WHEN COUNT(*) > 0 THEN 'failed'::TEXT 
            ELSE 'passed'::TEXT 
        END,
        COUNT(*)::NUMERIC,
        FORMAT('Channels exceeding 5 per user: %s users', COUNT(*))::TEXT
    FROM (
        SELECT user_id, COUNT(*) as channel_count
        FROM analytics.dim_channel
        WHERE is_active = true
        GROUP BY user_id
        HAVING COUNT(*) > 5
    ) t;
END;
$ LANGUAGE plpgsql;

-- Timeliness checks
CREATE OR REPLACE FUNCTION analytics.check_data_timeliness()
RETURNS TABLE(
    check_name TEXT,
    status TEXT,
    delay_minutes NUMERIC,
    details TEXT
) AS $
BEGIN
    -- Check data freshness
    RETURN QUERY
    WITH freshness AS (
        SELECT 
            'video_metrics'::TEXT as data_type,
            MAX(timestamp) as last_update,
            EXTRACT(MINUTE FROM (NOW() - MAX(timestamp))) as minutes_delay
        FROM analytics.fact_video_performance
    )
    SELECT 
        'data_freshness_' || data_type,
        CASE 
            WHEN minutes_delay > 60 THEN 'failed'::TEXT
            WHEN minutes_delay > 15 THEN 'warning'::TEXT
            ELSE 'passed'::TEXT
        END,
        minutes_delay,
        FORMAT('Last update: %s minutes ago', minutes_delay)::TEXT
    FROM freshness;
END;
$ LANGUAGE plpgsql;

-- Master data quality check function
CREATE OR REPLACE FUNCTION analytics.run_all_quality_checks()
RETURNS VOID AS $
DECLARE
    v_record RECORD;
BEGIN
    -- Run completeness checks
    FOR v_record IN SELECT * FROM analytics.check_data_completeness() LOOP
        INSERT INTO analytics.data_quality_checks (check_name, check_type, status, details)
        VALUES (v_record.check_name, 'completeness', v_record.status, 
                jsonb_build_object('percentage', v_record.missing_percentage, 'details', v_record.details));
    END LOOP;
    
    -- Run accuracy checks
    FOR v_record IN SELECT * FROM analytics.check_data_accuracy() LOOP
        INSERT INTO analytics.data_quality_checks (check_name, check_type, status, details)
        VALUES (v_record.check_name, 'accuracy', v_record.status,
                jsonb_build_object('error_rate', v_record.error_rate, 'details', v_record.details));
    END LOOP;
    
    -- Run consistency checks
    FOR v_record IN SELECT * FROM analytics.check_data_consistency() LOOP
        INSERT INTO analytics.data_quality_checks (check_name, check_type, status, details)
        VALUES (v_record.check_name, 'consistency', v_record.status,
                jsonb_build_object('count', v_record.inconsistency_count, 'details', v_record.details));
    END LOOP;
    
    -- Run timeliness checks
    FOR v_record IN SELECT * FROM analytics.check_data_timeliness() LOOP
        INSERT INTO analytics.data_quality_checks (check_name, check_type, status, details)
        VALUES (v_record.check_name, 'timeliness', v_record.status,
                jsonb_build_object('delay_minutes', v_record.delay_minutes, 'details', v_record.details));
    END LOOP;
END;
$ LANGUAGE plpgsql;
```

### Testing Strategies

```yaml
# tests/schema.yml
version: 2

models:
  - name: fact_video_performance
    description: "Video performance metrics fact table"
    
    columns:
      - name: video_key
        description: "Unique video identifier"
        tests:
          - not_null
          - relationships:
              to: ref('dim_video')
              field: video_key
              
      - name: timestamp
        description: "Metric timestamp"
        tests:
          - not_null
          
      - name: engagement_rate
        description: "Engagement rate (0-1)"
        tests:
          - not_null
          - accepted_values:
              values: [0, 1]
              quote: false
          
      - name: cost_cents
        description: "Production cost in cents"
        tests:
          - not_null
    
    tests:
      - unique:
          column_name: "video_key || '-' || timestamp"
```

### Monitoring Setup

```python
# monitoring/data_quality_monitor.py
"""
Real-time data quality monitoring
"""

import psycopg2
from datetime import datetime, timedelta
import json
import requests

class DataQualityMonitor:
    def __init__(self):
        self.db_connection = psycopg2.connect(
            host="localhost",
            database="ytempire",
            user="analytics_user",
            password="password"
        )
        self.slack_webhook = "https://hooks.slack.com/services/..."
        
    def check_critical_metrics(self):
        """Check critical data quality metrics"""
        cursor = self.db_connection.cursor()
        
        # Check data freshness
        cursor.execute("""
            SELECT 
                EXTRACT(MINUTE FROM (NOW() - MAX(timestamp))) as minutes_delay
            FROM analytics.fact_video_performance
        """)
        
        delay = cursor.fetchone()[0]
        
        if delay > 60:
            self.send_alert(
                level='CRITICAL',
                message=f'Data is {delay} minutes old. Immediate action required!'
            )
        elif delay > 15:
            self.send_alert(
                level='WARNING',
                message=f'Data is {delay} minutes old. Please investigate.'
            )
    
    def check_anomalies(self):
        """Detect anomalies in metrics"""
        cursor = self.db_connection.cursor()
        
        # Check for sudden drops in video production
        cursor.execute("""
            WITH daily_videos AS (
                SELECT 
                    DATE(published_at) as date,
                    COUNT(*) as video_count
                FROM analytics.dim_video
                WHERE published_at >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY DATE(published_at)
            )
            SELECT 
                date,
                video_count,
                AVG(video_count) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING) as avg_count
            FROM daily_videos
            WHERE date = CURRENT_DATE
        """)
        
        result = cursor.fetchone()
        if result:
            date, count, avg = result
            if count < avg * 0.5:  # 50% drop
                self.send_alert(
                    level='WARNING',
                    message=f'Video production dropped to {count} from average of {avg:.0f}'
                )
    
    def send_alert(self, level, message):
        """Send alert to Slack"""
        color = {
            'CRITICAL': 'danger',
            'WARNING': 'warning',
            'INFO': 'good'
        }.get(level, 'info')
        
        payload = {
            'attachments': [{
                'color': color,
                'title': f'Data Quality Alert - {level}',
                'text': message,
                'footer': 'YTEMPIRE Analytics',
                'ts': int(datetime.now().timestamp())
            }]
        }
        
        requests.post(self.slack_webhook, json=payload)
    
    def generate_quality_report(self):
        """Generate daily data quality report"""
        cursor = self.db_connection.cursor()
        
        cursor.execute("""
            SELECT 
                check_type,
                COUNT(*) FILTER (WHERE status = 'passed') as passed,
                COUNT(*) FILTER (WHERE status = 'warning') as warnings,
                COUNT(*) FILTER (WHERE status = 'failed') as failed
            FROM analytics.data_quality_checks
            WHERE check_timestamp >= CURRENT_DATE
            GROUP BY check_type
        """)
        
        results = cursor.fetchall()
        
        report = {
            'date': datetime.now().isoformat(),
            'summary': {
                check_type: {
                    'passed': passed,
                    'warnings': warnings,
                    'failed': failed
                }
                for check_type, passed, warnings, failed in results
            }
        }
        
        return report

# Schedule monitoring jobs
if __name__ == "__main__":
    monitor = DataQualityMonitor()
    
    # Run critical checks every 5 minutes
    monitor.check_critical_metrics()
    
    # Run anomaly detection every hour
    monitor.check_anomalies()
    
    # Generate daily report at midnight
    if datetime.now().hour == 0:
        report = monitor.generate_quality_report()
        print(json.dumps(report, indent=2))
```