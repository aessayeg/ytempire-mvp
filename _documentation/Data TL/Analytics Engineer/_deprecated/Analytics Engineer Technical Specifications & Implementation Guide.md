# Analytics Engineer Technical Specifications & Implementation Guide

**Document Version**: 1.0  
**Date**: January 2025  
**Purpose**: Technical implementation details for Analytics Engineering  
**Scope**: Database schemas, query patterns, and dashboard specifications

---

## 1. Database Schema & Data Models

### 1.1 Core Analytics Schema

```sql
-- Create analytics schema
CREATE SCHEMA IF NOT EXISTS analytics;

-- Set search path
SET search_path TO analytics, public;

-- Enable TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Main fact table for video performance
CREATE TABLE analytics.fact_video_performance (
    id BIGSERIAL PRIMARY KEY,
    video_id VARCHAR(50) NOT NULL,
    channel_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Metrics
    views BIGINT DEFAULT 0,
    likes INTEGER DEFAULT 0,
    dislikes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    shares INTEGER DEFAULT 0,
    watch_time_minutes DECIMAL(12,2) DEFAULT 0,
    average_view_duration_seconds DECIMAL(10,2) DEFAULT 0,
    
    -- Calculated metrics
    engagement_rate DECIMAL(5,4) GENERATED ALWAYS AS (
        CASE 
            WHEN views > 0 THEN (likes + comments + shares)::DECIMAL / views
            ELSE 0
        END
    ) STORED,
    
    -- Financial
    estimated_revenue_cents INTEGER DEFAULT 0,
    actual_revenue_cents INTEGER DEFAULT 0,
    cost_cents INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('analytics.fact_video_performance', 'timestamp');

-- Create indexes for common query patterns
CREATE INDEX idx_video_performance_channel_time ON analytics.fact_video_performance(channel_id, timestamp DESC);
CREATE INDEX idx_video_performance_video_time ON analytics.fact_video_performance(video_id, timestamp DESC);

-- Channel dimension table
CREATE TABLE analytics.dim_channel (
    channel_id VARCHAR(50) PRIMARY KEY,
    channel_name VARCHAR(255) NOT NULL,
    channel_handle VARCHAR(100),
    niche VARCHAR(100),
    subscriber_count INTEGER DEFAULT 0,
    total_video_count INTEGER DEFAULT 0,
    monetization_enabled BOOLEAN DEFAULT false,
    created_date DATE,
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB DEFAULT '{}',
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Video dimension table
CREATE TABLE analytics.dim_video (
    video_id VARCHAR(50) PRIMARY KEY,
    channel_id VARCHAR(50) REFERENCES analytics.dim_channel(channel_id),
    title TEXT NOT NULL,
    description TEXT,
    duration_seconds INTEGER,
    publish_date TIMESTAMPTZ,
    video_type VARCHAR(50), -- 'educational', 'entertainment', 'news'
    thumbnail_url TEXT,
    tags TEXT[],
    category VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Cost tracking table
CREATE TABLE analytics.cost_tracking (
    id BIGSERIAL PRIMARY KEY,
    video_id VARCHAR(50),
    channel_id VARCHAR(50),
    cost_type VARCHAR(50) NOT NULL, -- 'ai_generation', 'voice_synthesis', 'api_calls'
    cost_cents INTEGER NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    details JSONB DEFAULT '{}'
);

-- Create aggregation tables for performance
CREATE MATERIALIZED VIEW analytics.hourly_channel_metrics AS
SELECT 
    time_bucket('1 hour', timestamp) AS hour,
    channel_id,
    COUNT(DISTINCT video_id) as videos_count,
    SUM(views) as total_views,
    AVG(engagement_rate) as avg_engagement_rate,
    SUM(estimated_revenue_cents) / 100.0 as estimated_revenue,
    SUM(cost_cents) / 100.0 as total_cost
FROM analytics.fact_video_performance
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY hour, channel_id
WITH NO DATA;

-- Create continuous aggregate refresh policy
SELECT add_continuous_aggregate_policy('analytics.hourly_channel_metrics',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes');
```

### 1.2 Revenue Analytics Schema

```sql
-- Revenue tracking with attribution
CREATE TABLE analytics.revenue_attribution (
    id BIGSERIAL PRIMARY KEY,
    video_id VARCHAR(50) NOT NULL,
    channel_id VARCHAR(50) NOT NULL,
    revenue_date DATE NOT NULL,
    
    -- Revenue breakdown
    ad_revenue_cents INTEGER DEFAULT 0,
    sponsorship_revenue_cents INTEGER DEFAULT 0,
    affiliate_revenue_cents INTEGER DEFAULT 0,
    other_revenue_cents INTEGER DEFAULT 0,
    
    -- Cost breakdown
    ai_cost_cents INTEGER DEFAULT 0,
    api_cost_cents INTEGER DEFAULT 0,
    infrastructure_cost_cents INTEGER DEFAULT 0,
    
    -- Calculated fields
    total_revenue_cents INTEGER GENERATED ALWAYS AS (
        ad_revenue_cents + sponsorship_revenue_cents + affiliate_revenue_cents + other_revenue_cents
    ) STORED,
    
    total_cost_cents INTEGER GENERATED ALWAYS AS (
        ai_cost_cents + api_cost_cents + infrastructure_cost_cents
    ) STORED,
    
    profit_cents INTEGER GENERATED ALWAYS AS (
        (ad_revenue_cents + sponsorship_revenue_cents + affiliate_revenue_cents + other_revenue_cents) -
        (ai_cost_cents + api_cost_cents + infrastructure_cost_cents)
    ) STORED,
    
    roi_percentage DECIMAL(10,2) GENERATED ALWAYS AS (
        CASE 
            WHEN (ai_cost_cents + api_cost_cents + infrastructure_cost_cents) > 0 
            THEN ((ad_revenue_cents + sponsorship_revenue_cents + affiliate_revenue_cents + other_revenue_cents - 
                   ai_cost_cents - api_cost_cents - infrastructure_cost_cents)::DECIMAL / 
                  (ai_cost_cents + api_cost_cents + infrastructure_cost_cents)::DECIMAL) * 100
            ELSE 0
        END
    ) STORED,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(video_id, revenue_date)
);

-- Daily revenue summary
CREATE MATERIALIZED VIEW analytics.daily_revenue_summary AS
SELECT 
    revenue_date,
    COUNT(DISTINCT channel_id) as active_channels,
    COUNT(DISTINCT video_id) as monetized_videos,
    SUM(total_revenue_cents) / 100.0 as total_revenue,
    SUM(total_cost_cents) / 100.0 as total_cost,
    SUM(profit_cents) / 100.0 as total_profit,
    AVG(roi_percentage) as avg_roi,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY roi_percentage) as median_roi
FROM analytics.revenue_attribution
WHERE revenue_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY revenue_date;

-- Refresh daily at 2 AM
CREATE INDEX idx_daily_revenue_date ON analytics.daily_revenue_summary(revenue_date DESC);
```

---

## 2. Key SQL Queries & Patterns

### 2.1 Performance Analytics Queries

```sql
-- Query 1: Channel Performance Dashboard
WITH channel_performance AS (
    SELECT 
        c.channel_id,
        c.channel_name,
        c.niche,
        COUNT(DISTINCT v.video_id) as video_count,
        SUM(vp.views) as total_views,
        AVG(vp.engagement_rate) as avg_engagement,
        SUM(vp.estimated_revenue_cents) / 100.0 as total_revenue,
        SUM(vp.cost_cents) / 100.0 as total_cost,
        (SUM(vp.estimated_revenue_cents) - SUM(vp.cost_cents)) / 100.0 as profit,
        AVG(vp.watch_time_minutes) as avg_watch_time
    FROM analytics.dim_channel c
    LEFT JOIN analytics.dim_video v ON c.channel_id = v.channel_id
    LEFT JOIN analytics.fact_video_performance vp ON v.video_id = vp.video_id
    WHERE vp.timestamp >= NOW() - INTERVAL '7 days'
    GROUP BY c.channel_id, c.channel_name, c.niche
),
channel_ranking AS (
    SELECT 
        *,
        RANK() OVER (ORDER BY total_revenue DESC) as revenue_rank,
        RANK() OVER (ORDER BY total_views DESC) as views_rank,
        RANK() OVER (ORDER BY profit DESC) as profit_rank
    FROM channel_performance
)
SELECT * FROM channel_ranking
ORDER BY profit_rank;

-- Query 2: Video Performance Trends
WITH video_metrics AS (
    SELECT 
        DATE_TRUNC('day', timestamp) as day,
        video_id,
        MAX(views) as daily_views,
        MAX(engagement_rate) as daily_engagement,
        MAX(estimated_revenue_cents) / 100.0 as daily_revenue
    FROM analytics.fact_video_performance
    WHERE timestamp >= NOW() - INTERVAL '30 days'
    GROUP BY day, video_id
),
video_growth AS (
    SELECT 
        video_id,
        day,
        daily_views,
        LAG(daily_views) OVER (PARTITION BY video_id ORDER BY day) as prev_views,
        daily_views - LAG(daily_views) OVER (PARTITION BY video_id ORDER BY day) as view_growth,
        daily_revenue,
        SUM(daily_revenue) OVER (PARTITION BY video_id ORDER BY day) as cumulative_revenue
    FROM video_metrics
)
SELECT 
    v.video_id,
    d.title,
    d.channel_id,
    MAX(v.daily_views) as peak_daily_views,
    AVG(v.view_growth) as avg_daily_growth,
    MAX(v.cumulative_revenue) as total_revenue,
    COUNT(CASE WHEN v.view_growth > 1000 THEN 1 END) as viral_days
FROM video_growth v
JOIN analytics.dim_video d ON v.video_id = d.video_id
GROUP BY v.video_id, d.title, d.channel_id
HAVING MAX(v.daily_views) > 1000
ORDER BY total_revenue DESC
LIMIT 100;

-- Query 3: Cost Analysis
WITH cost_breakdown AS (
    SELECT 
        DATE_TRUNC('day', timestamp) as day,
        cost_type,
        SUM(cost_cents) / 100.0 as total_cost,
        COUNT(*) as transaction_count,
        AVG(cost_cents) / 100.0 as avg_cost_per_transaction
    FROM analytics.cost_tracking
    WHERE timestamp >= NOW() - INTERVAL '30 days'
    GROUP BY day, cost_type
),
daily_totals AS (
    SELECT 
        day,
        SUM(total_cost) as daily_total_cost,
        SUM(transaction_count) as daily_transactions
    FROM cost_breakdown
    GROUP BY day
)
SELECT 
    cb.day,
    cb.cost_type,
    cb.total_cost,
    cb.transaction_count,
    cb.avg_cost_per_transaction,
    ROUND((cb.total_cost / dt.daily_total_cost) * 100, 2) as percentage_of_daily_cost
FROM cost_breakdown cb
JOIN daily_totals dt ON cb.day = dt.day
ORDER BY cb.day DESC, cb.total_cost DESC;
```

### 2.2 Revenue Optimization Queries

```sql
-- Query 4: Revenue Per Thousand Views (RPM) Analysis
WITH rpm_calc AS (
    SELECT 
        c.channel_id,
        c.channel_name,
        c.niche,
        DATE_TRUNC('week', vp.timestamp) as week,
        SUM(vp.views) as total_views,
        SUM(vp.estimated_revenue_cents) as total_revenue_cents,
        CASE 
            WHEN SUM(vp.views) > 0 
            THEN (SUM(vp.estimated_revenue_cents) / SUM(vp.views)::DECIMAL) * 1000
            ELSE 0
        END as rpm
    FROM analytics.dim_channel c
    JOIN analytics.dim_video v ON c.channel_id = v.channel_id
    JOIN analytics.fact_video_performance vp ON v.video_id = vp.video_id
    WHERE vp.timestamp >= NOW() - INTERVAL '12 weeks'
    GROUP BY c.channel_id, c.channel_name, c.niche, week
),
rpm_trends AS (
    SELECT 
        channel_id,
        channel_name,
        niche,
        week,
        rpm,
        AVG(rpm) OVER (PARTITION BY channel_id ORDER BY week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) as moving_avg_rpm,
        STDDEV(rpm) OVER (PARTITION BY channel_id) as rpm_volatility
    FROM rpm_calc
)
SELECT 
    channel_id,
    channel_name,
    niche,
    AVG(rpm) as avg_rpm,
    MAX(rpm) as max_rpm,
    MIN(rpm) as min_rpm,
    AVG(rpm_volatility) as volatility,
    CASE 
        WHEN AVG(rpm) > 5 THEN 'High Performer'
        WHEN AVG(rpm) > 2 THEN 'Average Performer'
        ELSE 'Needs Optimization'
    END as performance_tier
FROM rpm_trends
GROUP BY channel_id, channel_name, niche
ORDER BY avg_rpm DESC;

-- Query 5: ROI by Video Type
WITH video_roi AS (
    SELECT 
        v.video_type,
        v.category,
        COUNT(DISTINCT v.video_id) as video_count,
        AVG(r.roi_percentage) as avg_roi,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY r.roi_percentage) as median_roi,
        SUM(r.total_revenue_cents) / 100.0 as total_revenue,
        SUM(r.total_cost_cents) / 100.0 as total_cost,
        SUM(r.profit_cents) / 100.0 as total_profit
    FROM analytics.dim_video v
    JOIN analytics.revenue_attribution r ON v.video_id = r.video_id
    WHERE r.revenue_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY v.video_type, v.category
)
SELECT 
    video_type,
    category,
    video_count,
    ROUND(avg_roi, 2) as avg_roi_percent,
    ROUND(median_roi, 2) as median_roi_percent,
    total_revenue,
    total_cost,
    total_profit,
    ROUND(total_profit / NULLIF(video_count, 0), 2) as profit_per_video
FROM video_roi
WHERE video_count >= 5  -- Minimum sample size
ORDER BY median_roi DESC;
```

---

## 3. Grafana Dashboard Specifications

### 3.1 Executive Dashboard Configuration

```json
{
  "dashboard": {
    "title": "YTEMPIRE Executive Dashboard",
    "tags": ["executive", "revenue", "performance"],
    "timezone": "browser",
    "refresh": "5m",
    "time": {
      "from": "now-7d",
      "to": "now"
    },
    "panels": [
      {
        "id": 1,
        "type": "stat",
        "title": "Total Revenue (7 Days)",
        "targets": [
          {
            "datasource": "PostgreSQL",
            "rawSql": "SELECT SUM(total_revenue_cents) / 100.0 as value FROM analytics.revenue_attribution WHERE revenue_date >= CURRENT_DATE - INTERVAL '7 days'",
            "format": "time_series"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "decimals": 2,
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 5000},
                {"color": "green", "value": 10000}
              ]
            }
          }
        },
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "type": "stat",
        "title": "Average Cost Per Video",
        "targets": [
          {
            "datasource": "PostgreSQL",
            "rawSql": "SELECT AVG(total_cost_cents) / 100.0 as value FROM analytics.revenue_attribution WHERE revenue_date >= CURRENT_DATE - INTERVAL '7 days'",
            "format": "time_series"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "decimals": 2,
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 0.40},
                {"color": "orange", "value": 0.45},
                {"color": "red", "value": 0.50}
              ]
            }
          }
        },
        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 0}
      },
      {
        "id": 3,
        "type": "gauge",
        "title": "Average ROI",
        "targets": [
          {
            "datasource": "PostgreSQL",
            "rawSql": "SELECT AVG(roi_percentage) as value FROM analytics.revenue_attribution WHERE revenue_date >= CURRENT_DATE - INTERVAL '7 days'",
            "format": "time_series"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 500,
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 100},
                {"color": "green", "value": 200}
              ]
            }
          }
        },
        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0}
      },
      {
        "id": 4,
        "type": "stat",
        "title": "Active Channels",
        "targets": [
          {
            "datasource": "PostgreSQL",
            "rawSql": "SELECT COUNT(DISTINCT channel_id) as value FROM analytics.fact_video_performance WHERE timestamp >= NOW() - INTERVAL '24 hours'",
            "format": "time_series"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "short",
            "decimals": 0
          }
        },
        "gridPos": {"h": 4, "w": 6, "x": 18, "y": 0}
      },
      {
        "id": 5,
        "type": "timeseries",
        "title": "Revenue vs Cost Trend",
        "targets": [
          {
            "datasource": "PostgreSQL",
            "rawSql": "SELECT revenue_date as time, SUM(total_revenue_cents) / 100.0 as revenue, SUM(total_cost_cents) / 100.0 as cost FROM analytics.revenue_attribution WHERE revenue_date >= CURRENT_DATE - INTERVAL '30 days' GROUP BY revenue_date ORDER BY revenue_date",
            "format": "time_series"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "custom": {
              "lineWidth": 2,
              "fillOpacity": 10
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4}
      }
    ]
  }
}
```

### 3.2 Channel Performance Dashboard

```yaml
dashboard:
  title: "Channel Performance Matrix"
  refresh: 10m
  
  variables:
    - name: channel_filter
      type: query
      query: "SELECT channel_name FROM analytics.dim_channel ORDER BY channel_name"
      multi: true
      includeAll: true
    
    - name: time_range
      type: interval
      options: ["1h", "6h", "12h", "24h", "7d", "30d"]
      default: "24h"
  
  panels:
    - channelTable:
        type: table
        title: "Channel Performance Overview"
        query: |
          SELECT 
            channel_name,
            video_count,
            total_views,
            ROUND(avg_engagement * 100, 2) as engagement_rate,
            total_revenue,
            total_cost,
            profit,
            revenue_rank
          FROM channel_performance_view
          WHERE ($channel_filter = 'All' OR channel_name IN ($channel_filter))
          ORDER BY profit DESC
        
    - viewsHeatmap:
        type: heatmap
        title: "Views Distribution by Hour"
        query: |
          SELECT 
            EXTRACT(hour FROM timestamp) as hour,
            EXTRACT(dow FROM timestamp) as day_of_week,
            SUM(views) as views
          FROM analytics.fact_video_performance
          WHERE timestamp >= NOW() - INTERVAL '$time_range'
          GROUP BY hour, day_of_week
          
    - topVideos:
        type: bar
        title: "Top 10 Videos by Revenue"
        query: |
          SELECT 
            v.title,
            SUM(r.total_revenue_cents) / 100.0 as revenue
          FROM analytics.dim_video v
          JOIN analytics.revenue_attribution r ON v.video_id = r.video_id
          WHERE r.revenue_date >= CURRENT_DATE - INTERVAL '7 days'
          GROUP BY v.title
          ORDER BY revenue DESC
          LIMIT 10
```

---

## 4. Query Optimization Techniques

### 4.1 Materialized Views Strategy

```sql
-- Create materialized views for expensive queries
-- Refresh strategy: Daily at 3 AM for historical, hourly for recent

-- Historical performance (refreshed daily)
CREATE MATERIALIZED VIEW analytics.mv_historical_performance AS
SELECT 
    DATE_TRUNC('day', timestamp) as day,
    channel_id,
    COUNT(DISTINCT video_id) as videos,
    SUM(views) as views,
    AVG(engagement_rate) as avg_engagement,
    SUM(estimated_revenue_cents) / 100.0 as revenue
FROM analytics.fact_video_performance
WHERE timestamp < DATE_TRUNC('day', NOW())
GROUP BY day, channel_id;

CREATE UNIQUE INDEX ON analytics.mv_historical_performance(day, channel_id);

-- Recent performance (refreshed every hour)
CREATE MATERIALIZED VIEW analytics.mv_recent_performance AS
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    channel_id,
    video_id,
    MAX(views) as views,
    MAX(engagement_rate) as engagement_rate,
    MAX(estimated_revenue_cents) as revenue_cents
FROM analytics.fact_video_performance
WHERE timestamp >= NOW() - INTERVAL '48 hours'
GROUP BY hour, channel_id, video_id;

CREATE INDEX ON analytics.mv_recent_performance(hour, channel_id);

-- Refresh policies
CREATE OR REPLACE FUNCTION analytics.refresh_materialized_views()
RETURNS void AS $$
BEGIN
    -- Refresh historical daily at 3 AM
    IF EXTRACT(hour FROM NOW()) = 3 AND EXTRACT(minute FROM NOW()) < 5 THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.mv_historical_performance;
    END IF;
    
    -- Refresh recent every hour
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.mv_recent_performance;
END;
$$ LANGUAGE plpgsql;

-- Schedule with pg_cron
SELECT cron.schedule('refresh_analytics_views', '0 * * * *', 'SELECT analytics.refresh_materialized_views()');
```

### 4.2 Query Performance Optimization

```sql
-- Enable query performance tracking
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

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

-- Optimize common query patterns with indexes
CREATE INDEX CONCURRENTLY idx_video_perf_composite 
ON analytics.fact_video_performance(channel_id, timestamp DESC, views)
WHERE timestamp >= NOW() - INTERVAL '30 days';

CREATE INDEX CONCURRENTLY idx_revenue_date_channel 
ON analytics.revenue_attribution(revenue_date DESC, channel_id, total_revenue_cents);

-- Partition large tables by time
CREATE TABLE analytics.fact_video_performance_2025_01 
PARTITION OF analytics.fact_video_performance
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE analytics.fact_video_performance_2025_02 
PARTITION OF analytics.fact_video_performance
FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
```

---

## 5. Data Quality Monitoring

### 5.1 Quality Check Queries

```sql
-- Data freshness check
CREATE OR REPLACE FUNCTION analytics.check_data_freshness()
RETURNS TABLE(
    table_name TEXT,
    last_update TIMESTAMPTZ,
    hours_since_update NUMERIC,
    status TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH freshness AS (
        SELECT 
            'fact_video_performance' as table_name,
            MAX(timestamp) as last_update
        FROM analytics.fact_video_performance
        UNION ALL
        SELECT 
            'revenue_attribution' as table_name,
            MAX(created_at) as last_update
        FROM analytics.revenue_attribution
    )
    SELECT 
        f.table_name,
        f.last_update,
        EXTRACT(EPOCH FROM (NOW() - f.last_update)) / 3600 as hours_since_update,
        CASE 
            WHEN EXTRACT(EPOCH FROM (NOW() - f.last_update)) / 3600 > 6 THEN 'STALE'
            WHEN EXTRACT(EPOCH FROM (NOW() - f.last_update)) / 3600 > 2 THEN 'WARNING'
            ELSE 'FRESH'
        END as status
    FROM freshness f;
END;
$$ LANGUAGE plpgsql;

-- Revenue reconciliation check
CREATE OR REPLACE FUNCTION analytics.check_revenue_reconciliation()
RETURNS TABLE(
    check_date DATE,
    youtube_revenue DECIMAL,
    calculated_revenue DECIMAL,
    difference DECIMAL,
    difference_percentage DECIMAL,
    status TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH reconciliation AS (
        SELECT 
            revenue_date,
            SUM(ad_revenue_cents + sponsorship_revenue_cents + affiliate_revenue_cents) / 100.0 as internal_revenue,
            -- This would be compared against YouTube Analytics API data
            SUM(estimated_revenue_cents) / 100.0 as youtube_revenue
        FROM analytics.revenue_attribution r
        JOIN analytics.fact_video_performance v ON r.video_id = v.video_id
        WHERE revenue_date >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY revenue_date
    )
    SELECT 
        revenue_date as check_date,
        youtube_revenue,
        internal_revenue as calculated_revenue,
        youtube_revenue - internal_revenue as difference,
        CASE 
            WHEN youtube_revenue > 0 
            THEN ((youtube_revenue - internal_revenue) / youtube_revenue) * 100
            ELSE 0
        END as difference_percentage,
        CASE 
            WHEN ABS(youtube_revenue - internal_revenue) / NULLIF(youtube_revenue, 0) > 0.05 THEN 'MISMATCH'
            WHEN ABS(youtube_revenue - internal_revenue) / NULLIF(youtube_revenue, 0) > 0.02 THEN 'WARNING'
            ELSE 'OK'
        END as status
    FROM reconciliation
    ORDER BY revenue_date DESC;
END;
$$ LANGUAGE plpgsql;

-- Missing data detection
CREATE OR REPLACE FUNCTION analytics.check_missing_data()
RETURNS TABLE(
    channel_id VARCHAR,
    missing_hours INTEGER,
    last_data_point TIMESTAMPTZ,
    status TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH expected_hours AS (
        SELECT 
            c.channel_id,
            generate_series(
                NOW() - INTERVAL '24 hours',
                NOW(),
                INTERVAL '1 hour'
            ) as expected_hour
        FROM analytics.dim_channel c
        WHERE c.status = 'active'
    ),
    actual_hours AS (
        SELECT 
            channel_id,
            DATE_TRUNC('hour', timestamp) as actual_hour
        FROM analytics.fact_video_performance
        WHERE timestamp >= NOW() - INTERVAL '24 hours'
        GROUP BY channel_id, DATE_TRUNC('hour', timestamp)
    ),
    missing AS (
        SELECT 
            e.channel_id,
            COUNT(*) as missing_hours,
            MAX(a.actual_hour) as last_data
        FROM expected_hours e
        LEFT JOIN actual_hours a 
            ON e.channel_id = a.channel_id 
            AND e.expected_hour = a.actual_hour
        WHERE a.actual_hour IS NULL
        GROUP BY e.channel_id
    )
    SELECT 
        m.channel_id,
        m.missing_hours,
        m.last_data as last_data_point,
        CASE 
            WHEN m.missing_hours > 12 THEN 'CRITICAL'
            WHEN m.missing_hours > 6 THEN 'WARNING'
            ELSE 'OK'
        END as status
    FROM missing m
    WHERE m.missing_hours > 0
    ORDER BY m.missing_hours DESC;
END;
$$ LANGUAGE plpgsql;
```

---

## 6. Performance Monitoring & Alerting

### 6.1 Query Performance Monitoring

```sql
-- Create performance monitoring table
CREATE TABLE analytics.query_performance_log (
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
         EXTRACT(MILLISECONDS FROM (NOW() - p_start_time)),
         p_rows,
         p_success,
         p_error);
END;
$$ LANGUAGE plpgsql;

-- Alert on slow queries
CREATE OR REPLACE FUNCTION analytics.check_slow_queries()
RETURNS TABLE(
    alert_level TEXT,
    query_name VARCHAR,
    avg_execution_time_ms NUMERIC,
    max_execution_time_ms NUMERIC,
    failure_rate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH query_stats AS (
        SELECT 
            query_name,
            AVG(execution_time_ms) as avg_time,
            MAX(execution_time_ms) as max_time,
            COUNT(CASE WHEN NOT success THEN 1 END)::NUMERIC / COUNT(*) as fail_rate
        FROM analytics.query_performance_log
        WHERE timestamp >= NOW() - INTERVAL '1 hour'
        GROUP BY query_name
    )
    SELECT 
        CASE 
            WHEN avg_time > 2000 OR fail_rate > 0.1 THEN 'CRITICAL'
            WHEN avg_time > 1000 OR fail_rate > 0.05 THEN 'WARNING'
            ELSE 'OK'
        END as alert_level,
        query_name,
        ROUND(avg_time, 2) as avg_execution_time_ms,
        max_time as max_execution_time_ms,
        ROUND(fail_rate * 100, 2) as failure_rate
    FROM query_stats
    WHERE avg_time > 500 OR fail_rate > 0
    ORDER BY avg_time DESC;
END;
$$ LANGUAGE plpgsql;
```

---

## 7. Implementation Checklist

### Week 1: Foundation
- [ ] Set up analytics schema
- [ ] Create dimension and fact tables
- [ ] Implement cost tracking tables
- [ ] Build first materialized views
- [ ] Deploy basic Grafana dashboards

### Week 2: Optimization
- [ ] Add all necessary indexes
- [ ] Implement query performance logging
- [ ] Create data quality checks
- [ ] Set up automated refresh policies
- [ ] Optimize slow queries

### Week 3: Advanced Analytics
- [ ] Build revenue attribution models
- [ ] Create predictive metrics
- [ ] Implement A/B testing framework
- [ ] Deploy advanced dashboards
- [ ] Set up alerting rules

### Week 4: Production Ready
- [ ] Complete performance testing
- [ ] Document all metrics
- [ ] Train team on dashboards
- [ ] Set up monitoring alerts
- [ ] Create runbooks for issues

---

## Next Steps

1. Review and customize these specifications for your environment
2. Start with the schema creation scripts
3. Implement the core queries
4. Deploy Grafana dashboards
5. Set up monitoring and alerting
6. Begin optimization based on actual usage patterns

Remember: Start simple, measure everything, optimize based on data!