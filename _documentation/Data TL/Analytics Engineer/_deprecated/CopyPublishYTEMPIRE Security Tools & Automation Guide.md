# YTEMPIRE Dashboard Query Library & Performance Analytics

## Document Control
- **Version**: 2.0 (Complete Edition)
- **Last Updated**: January 2025
- **Owner**: Analytics Engineering Team
- **Audience**: Analytics Engineers, BI Developers, Data Analysts
- **Status**: Production Ready

---

## 1. Executive Summary

This document provides the complete query library and performance analytics framework for YTEMPIRE's business intelligence ecosystem. All queries are optimized for our internal content empire operating 100+ YouTube channels with 300+ daily videos.

### Key Performance Targets
- **Executive Dashboards**: < 2 seconds response time
- **Operational Dashboards**: < 5 seconds response time
- **Analytical Dashboards**: < 10 seconds response time
- **Data Freshness**: Real-time metrics < 1 minute, batch metrics < 1 hour

---

## 2. Dashboard Architecture

### 2.1 Dashboard Hierarchy

```
YTEMPIRE Dashboard Structure:
├── Executive Dashboards
│   ├── CEO Overview (Real-time KPIs)
│   ├── Revenue Analytics
│   ├── Growth Metrics
│   └── Channel Portfolio Health
│
├── Operational Dashboards
│   ├── Real-time Performance Monitor
│   ├── Content Pipeline Status
│   ├── Channel Operations
│   └── Cost Tracking
│
├── Analytical Dashboards
│   ├── Trend Analysis
│   ├── Audience Insights
│   ├── Content Performance
│   └── Competitive Intelligence
│
└── Specialized Dashboards
    ├── AI/ML Performance
    ├── YouTube API Usage
    ├── System Health
    └── Financial Projections
```

### 2.2 Data Sources

```sql
-- Primary data sources for dashboards
-- All tables optimized for our 100+ channel empire

-- Core tables
channels                    -- Our 100+ YouTube channels
videos                      -- 300+ daily videos
video_metrics              -- Real-time performance data
channel_analytics          -- Aggregated channel metrics
revenue_data              -- Monetization tracking
content_pipeline          -- Production status
cost_tracking             -- Operational costs
trend_signals             -- Market intelligence
```

---

## 3. Executive Dashboard Queries

### 3.1 CEO Master Dashboard

```sql
-- CEO Dashboard: Complete Empire Overview
-- Refresh Rate: Every 5 minutes
-- Performance Target: < 2 seconds
-- Purpose: High-level view of entire content empire

WITH empire_metrics AS (
    SELECT 
        -- Channel Portfolio
        COUNT(DISTINCT c.channel_id) AS total_channels,
        COUNT(DISTINCT CASE WHEN c.created_at >= CURRENT_DATE - INTERVAL '30 days' THEN c.channel_id END) AS new_channels_30d,
        COUNT(DISTINCT CASE WHEN c.status = 'active' THEN c.channel_id END) AS active_channels,
        
        -- Content Production
        COUNT(DISTINCT v.video_id) AS total_videos,
        COUNT(DISTINCT CASE WHEN v.published_at >= CURRENT_DATE THEN v.video_id END) AS videos_today,
        COUNT(DISTINCT CASE WHEN v.published_at >= CURRENT_DATE - INTERVAL '7 days' THEN v.video_id END) AS videos_week,
        
        -- Audience Reach
        SUM(ca.subscriber_count) AS total_subscribers,
        SUM(vm.views) AS total_views,
        SUM(CASE WHEN vm.date = CURRENT_DATE THEN vm.views ELSE 0 END) AS views_today,
        
        -- Financial Performance
        SUM(r.revenue_cents) / 100.0 AS total_revenue,
        SUM(CASE WHEN r.date >= CURRENT_DATE - INTERVAL '30 days' THEN r.revenue_cents ELSE 0 END) / 100.0 AS revenue_30d,
        SUM(CASE WHEN r.date = CURRENT_DATE THEN r.revenue_cents ELSE 0 END) / 100.0 AS revenue_today
        
    FROM channels c
    LEFT JOIN videos v ON c.channel_id = v.channel_id
    LEFT JOIN video_metrics vm ON v.video_id = vm.video_id
    LEFT JOIN channel_analytics ca ON c.channel_id = ca.channel_id AND ca.date = CURRENT_DATE
    LEFT JOIN revenue_data r ON v.video_id = r.video_id
    WHERE c.channel_id IN (SELECT channel_id FROM our_channels)  -- Only our channels
),
growth_calculations AS (
    SELECT 
        -- Month-over-month growth rates
        (current_month.revenue - previous_month.revenue) / NULLIF(previous_month.revenue, 0) * 100 AS revenue_growth_mom,
        (current_month.views - previous_month.views) / NULLIF(previous_month.views, 0) * 100 AS views_growth_mom,
        (current_month.subscribers - previous_month.subscribers) / NULLIF(previous_month.subscribers, 0) * 100 AS subscriber_growth_mom
    FROM (
        -- Current month metrics
        SELECT 
            SUM(revenue_cents) / 100.0 AS revenue,
            SUM(views) AS views,
            MAX(subscriber_count) AS subscribers
        FROM video_metrics vm
        JOIN channel_analytics ca ON vm.channel_id = ca.channel_id
        WHERE vm.date >= DATE_TRUNC('month', CURRENT_DATE)
    ) current_month
    CROSS JOIN (
        -- Previous month metrics
        SELECT 
            SUM(revenue_cents) / 100.0 AS revenue,
            SUM(views) AS views,
            MAX(subscriber_count) AS subscribers
        FROM video_metrics vm
        JOIN channel_analytics ca ON vm.channel_id = ca.channel_id
        WHERE vm.date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
          AND vm.date < DATE_TRUNC('month', CURRENT_DATE)
    ) previous_month
),
cost_analysis AS (
    SELECT 
        SUM(total_cost_cents) / 100.0 AS total_costs_30d,
        AVG(cost_per_video_cents) / 100.0 AS avg_cost_per_video,
        SUM(CASE WHEN cost_type = 'ai_generation' THEN cost_cents ELSE 0 END) / 100.0 AS ai_costs,
        SUM(CASE WHEN cost_type = 'youtube_api' THEN cost_cents ELSE 0 END) / 100.0 AS api_costs
    FROM cost_tracking
    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
)
SELECT 
    -- Empire Overview
    em.total_channels,
    em.active_channels,
    em.new_channels_30d,
    em.total_videos,
    em.videos_today,
    em.videos_week,
    
    -- Audience Metrics
    em.total_subscribers,
    em.total_views,
    em.views_today,
    
    -- Revenue Performance
    em.total_revenue,
    em.revenue_30d,
    em.revenue_today,
    em.revenue_30d * 12 AS annual_revenue_run_rate,
    
    -- Growth Rates
    gc.revenue_growth_mom,
    gc.views_growth_mom,
    gc.subscriber_growth_mom,
    
    -- Cost Analysis
    ca.total_costs_30d,
    ca.avg_cost_per_video,
    em.revenue_30d - ca.total_costs_30d AS net_profit_30d,
    ((em.revenue_30d - ca.total_costs_30d) / NULLIF(ca.total_costs_30d, 0)) * 100 AS roi_percentage,
    
    -- Key Performance Indicators
    em.revenue_30d / NULLIF(em.videos_week * 4.3, 0) AS revenue_per_video,
    em.total_views / NULLIF(em.total_videos, 0) AS avg_views_per_video,
    (em.revenue_30d / NULLIF(em.total_views, 0)) * 1000 AS rpm,
    
    -- Status Indicators
    CASE 
        WHEN em.revenue_today >= (em.revenue_30d / 30) * 1.1 THEN 'above_target'
        WHEN em.revenue_today >= (em.revenue_30d / 30) * 0.9 THEN 'on_target'
        ELSE 'below_target'
    END AS daily_performance_status,
    
    -- Projections
    CASE 
        WHEN gc.revenue_growth_mom > 0 THEN 
            em.revenue_30d * POWER(1 + (gc.revenue_growth_mom / 100), 12)
        ELSE 
            em.revenue_30d * 12
    END AS projected_annual_revenue,
    
    CURRENT_TIMESTAMP AS dashboard_updated
    
FROM empire_metrics em
CROSS JOIN growth_calculations gc
CROSS JOIN cost_analysis ca;
```

### 3.2 Revenue Analytics Dashboard

```sql
-- Revenue Analytics Dashboard: Comprehensive Financial View
-- Refresh Rate: Every 15 minutes
-- Performance Target: < 3 seconds
-- Purpose: Deep dive into revenue streams and profitability

-- Create materialized view for performance
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_revenue_analytics AS
WITH revenue_breakdown AS (
    SELECT 
        DATE_TRUNC('day', r.date) AS date,
        c.channel_id,
        c.channel_name,
        c.niche,
        -- Revenue streams
        SUM(r.ad_revenue_cents) / 100.0 AS ad_revenue,
        SUM(r.youtube_premium_cents) / 100.0 AS premium_revenue,
        SUM(r.channel_memberships_cents) / 100.0 AS membership_revenue,
        SUM(r.super_thanks_cents) / 100.0 AS super_thanks_revenue,
        SUM(r.sponsorship_revenue_cents) / 100.0 AS sponsorship_revenue,
        SUM(r.affiliate_revenue_cents) / 100.0 AS affiliate_revenue,
        SUM(r.revenue_cents) / 100.0 AS total_revenue,
        -- Video metrics
        COUNT(DISTINCT v.video_id) AS video_count,
        SUM(vm.views) AS total_views
    FROM revenue_data r
    JOIN videos v ON r.video_id = v.video_id
    JOIN channels c ON v.channel_id = c.channel_id
    JOIN video_metrics vm ON v.video_id = vm.video_id AND vm.date = r.date
    WHERE r.date >= CURRENT_DATE - INTERVAL '90 days'
      AND c.channel_id IN (SELECT channel_id FROM our_channels)
    GROUP BY DATE_TRUNC('day', r.date), c.channel_id, c.channel_name, c.niche
),
channel_profitability AS (
    SELECT 
        rb.channel_id,
        rb.channel_name,
        rb.niche,
        SUM(rb.total_revenue) AS channel_revenue,
        SUM(ct.total_cost_cents) / 100.0 AS channel_costs,
        SUM(rb.total_revenue) - (SUM(ct.total_cost_cents) / 100.0) AS channel_profit,
        ((SUM(rb.total_revenue) - (SUM(ct.total_cost_cents) / 100.0)) / NULLIF(SUM(ct.total_cost_cents) / 100.0, 0)) * 100 AS profit_margin,
        SUM(rb.total_views) AS total_views,
        COUNT(DISTINCT rb.date) AS active_days
    FROM revenue_breakdown rb
    LEFT JOIN cost_tracking ct ON rb.channel_id = ct.channel_id AND rb.date = ct.date
    WHERE rb.date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY rb.channel_id, rb.channel_name, rb.niche
),
revenue_trends AS (
    SELECT 
        date,
        SUM(total_revenue) AS daily_revenue,
        AVG(SUM(total_revenue)) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS revenue_7d_ma,
        AVG(SUM(total_revenue)) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS revenue_30d_ma,
        -- Revenue mix
        SUM(ad_revenue) / NULLIF(SUM(total_revenue), 0) * 100 AS ad_revenue_pct,
        SUM(premium_revenue) / NULLIF(SUM(total_revenue), 0) * 100 AS premium_revenue_pct,
        SUM(sponsorship_revenue + affiliate_revenue) / NULLIF(SUM(total_revenue), 0) * 100 AS external_revenue_pct
    FROM revenue_breakdown
    GROUP BY date
)
SELECT 
    -- Daily revenue trends
    rt.date,
    rt.daily_revenue,
    rt.revenue_7d_ma,
    rt.revenue_30d_ma,
    rt.ad_revenue_pct,
    rt.premium_revenue_pct,
    rt.external_revenue_pct,
    
    -- Top performing channels
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'channel_name', channel_name,
                'niche', niche,
                'revenue', channel_revenue,
                'profit', channel_profit,
                'margin', ROUND(profit_margin::numeric, 1),
                'rpm', ROUND((channel_revenue / NULLIF(total_views, 0) * 1000)::numeric, 2)
            ) ORDER BY channel_revenue DESC
        )
        FROM channel_profitability
        LIMIT 10
    ) AS top_channels,
    
    -- Revenue by niche
    (
        SELECT jsonb_object_agg(
            niche,
            jsonb_build_object(
                'revenue', total_revenue,
                'channels', channel_count,
                'avg_revenue_per_channel', avg_revenue
            )
        )
        FROM (
            SELECT 
                niche,
                SUM(channel_revenue) AS total_revenue,
                COUNT(*) AS channel_count,
                AVG(channel_revenue) AS avg_revenue
            FROM channel_profitability
            GROUP BY niche
        ) niche_summary
    ) AS niche_performance,
    
    -- Monthly progression
    SUM(rt.daily_revenue) OVER (PARTITION BY DATE_TRUNC('month', rt.date) ORDER BY rt.date) AS mtd_revenue,
    
    CURRENT_TIMESTAMP AS last_updated
FROM revenue_trends rt
ORDER BY rt.date DESC;

-- Query the materialized view
SELECT * FROM mv_revenue_analytics 
WHERE date >= CURRENT_DATE - INTERVAL '30 days';
```

### 3.3 Growth Metrics Dashboard

```sql
-- Growth Metrics Dashboard: Track all growth KPIs
-- Refresh Rate: Every 30 minutes
-- Performance Target: < 2 seconds
-- Purpose: Monitor growth trajectory toward $50M ARR

WITH growth_metrics AS (
    -- Channel growth
    SELECT 
        'channels' AS metric_type,
        DATE_TRUNC('day', created_at) AS date,
        COUNT(*) AS daily_value,
        SUM(COUNT(*)) OVER (ORDER BY DATE_TRUNC('day', created_at)) AS cumulative_value
    FROM channels
    WHERE channel_id IN (SELECT channel_id FROM our_channels)
      AND created_at >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY DATE_TRUNC('day', created_at)
    
    UNION ALL
    
    -- Subscriber growth
    SELECT 
        'subscribers' AS metric_type,
        date,
        SUM(subscriber_growth) AS daily_value,
        SUM(SUM(subscriber_growth)) OVER (ORDER BY date) AS cumulative_value
    FROM channel_analytics
    WHERE channel_id IN (SELECT channel_id FROM our_channels)
      AND date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY date
    
    UNION ALL
    
    -- Video production
    SELECT 
        'videos' AS metric_type,
        DATE_TRUNC('day', published_at) AS date,
        COUNT(*) AS daily_value,
        SUM(COUNT(*)) OVER (ORDER BY DATE_TRUNC('day', published_at)) AS cumulative_value
    FROM videos
    WHERE channel_id IN (SELECT channel_id FROM our_channels)
      AND published_at >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY DATE_TRUNC('day', published_at)
    
    UNION ALL
    
    -- Revenue growth
    SELECT 
        'revenue' AS metric_type,
        date,
        SUM(revenue_cents) / 100.0 AS daily_value,
        SUM(SUM(revenue_cents) / 100.0) OVER (ORDER BY date) AS cumulative_value
    FROM revenue_data
    WHERE video_id IN (SELECT video_id FROM videos WHERE channel_id IN (SELECT channel_id FROM our_channels))
      AND date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY date
),
growth_rates AS (
    SELECT 
        metric_type,
        date,
        daily_value,
        cumulative_value,
        -- Calculate growth rates
        (daily_value - LAG(daily_value, 7) OVER (PARTITION BY metric_type ORDER BY date)) / 
            NULLIF(LAG(daily_value, 7) OVER (PARTITION BY metric_type ORDER BY date), 0) * 100 AS wow_growth,
        (daily_value - LAG(daily_value, 30) OVER (PARTITION BY metric_type ORDER BY date)) / 
            NULLIF(LAG(daily_value, 30) OVER (PARTITION BY metric_type ORDER BY date), 0) * 100 AS mom_growth,
        -- Moving averages
        AVG(daily_value) OVER (PARTITION BY metric_type ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma_7d,
        AVG(daily_value) OVER (PARTITION BY metric_type ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS ma_30d
    FROM growth_metrics
),
current_snapshot AS (
    SELECT 
        -- Current state
        COUNT(DISTINCT CASE WHEN status = 'active' THEN channel_id END) AS active_channels,
        SUM(subscriber_count) AS total_subscribers,
        COUNT(DISTINCT video_id) AS total_videos,
        SUM(total_views) AS total_views,
        SUM(revenue_30d) AS monthly_revenue,
        SUM(revenue_30d) * 12 AS arr,
        
        -- Growth targets
        50000000 AS arr_target,  -- $50M
        1000 AS channel_target,
        50000000 AS subscriber_target
        
    FROM (
        SELECT 
            c.channel_id,
            c.status,
            ca.subscriber_count,
            ca.total_views,
            (
                SELECT SUM(revenue_cents) / 100.0 
                FROM revenue_data r 
                JOIN videos v ON r.video_id = v.video_id 
                WHERE v.channel_id = c.channel_id 
                  AND r.date >= CURRENT_DATE - INTERVAL '30 days'
            ) AS revenue_30d,
            (
                SELECT COUNT(*) 
                FROM videos v 
                WHERE v.channel_id = c.channel_id
            ) AS video_count
        FROM channels c
        LEFT JOIN channel_analytics ca ON c.channel_id = ca.channel_id AND ca.date = CURRENT_DATE
        WHERE c.channel_id IN (SELECT channel_id FROM our_channels)
    ) channel_summary
)
SELECT 
    -- Growth trend data
    (
        SELECT jsonb_object_agg(
            metric_type,
            jsonb_build_object(
                'current_daily', daily_value,
                'ma_7d', ROUND(ma_7d::numeric, 1),
                'ma_30d', ROUND(ma_30d::numeric, 1),
                'wow_growth', ROUND(wow_growth::numeric, 1),
                'mom_growth', ROUND(mom_growth::numeric, 1),
                'cumulative', cumulative_value
            )
        )
        FROM growth_rates
        WHERE date = CURRENT_DATE - INTERVAL '1 day'
    ) AS current_metrics,
    
    -- Progress toward goals
    (
        SELECT jsonb_build_object(
            'arr_progress', ROUND((arr / arr_target * 100)::numeric, 1),
            'arr_current', arr,
            'arr_needed', arr_target - arr,
            'channel_progress', ROUND((active_channels::float / channel_target * 100)::numeric, 1),
            'channels_needed', channel_target - active_channels,
            'months_to_target', 
                CASE 
                    WHEN monthly_revenue > 0 THEN 
                        ROUND(((arr_target / 12 - monthly_revenue) / (monthly_revenue * 0.15))::numeric, 1)  -- Assuming 15% MoM growth
                    ELSE NULL 
                END
        )
        FROM current_snapshot
    ) AS goal_progress,
    
    -- Growth trajectory chart data
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'date', date,
                'revenue', daily_value,
                'revenue_ma', ma_7d
            ) ORDER BY date
        )
        FROM growth_rates
        WHERE metric_type = 'revenue'
          AND date >= CURRENT_DATE - INTERVAL '30 days'
    ) AS revenue_trend,
    
    -- Channel scaling metrics
    (
        SELECT jsonb_build_object(
            'total_channels', COUNT(*),
            'active_channels', COUNT(CASE WHEN status = 'active' THEN 1 END),
            'channels_last_30d', COUNT(CASE WHEN created_at >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END),
            'avg_videos_per_channel', AVG(video_count),
            'avg_revenue_per_channel', AVG(revenue_30d)
        )
        FROM (
            SELECT 
                c.channel_id,
                c.status,
                c.created_at,
                COUNT(v.video_id) AS video_count,
                SUM(r.revenue_cents) / 100.0 AS revenue_30d
            FROM channels c
            LEFT JOIN videos v ON c.channel_id = v.channel_id
            LEFT JOIN revenue_data r ON v.video_id = r.video_id AND r.date >= CURRENT_DATE - INTERVAL '30 days'
            WHERE c.channel_id IN (SELECT channel_id FROM our_channels)
            GROUP BY c.channel_id, c.status, c.created_at
        ) channel_metrics
    ) AS scaling_metrics,
    
    CURRENT_TIMESTAMP AS dashboard_updated;
```

---

## 7. Performance Optimization Guidelines

### 7.1 Index Strategy

```sql
-- Essential indexes for dashboard performance
-- Run these during initial setup

-- Time-based queries
CREATE INDEX idx_video_metrics_date_channel ON video_metrics(date, channel_id) 
    WHERE date >= CURRENT_DATE - INTERVAL '90 days';
CREATE INDEX idx_revenue_data_date ON revenue_data(date DESC);
CREATE INDEX idx_videos_published_at ON videos(published_at DESC);

-- Channel performance
CREATE INDEX idx_channel_analytics_current ON channel_analytics(channel_id, date DESC);
CREATE INDEX idx_videos_channel_published ON videos(channel_id, published_at DESC);

-- Real-time metrics
CREATE INDEX idx_video_metrics_realtime_updated ON video_metrics_realtime(updated_at DESC);
CREATE INDEX idx_video_metrics_realtime_video ON video_metrics_realtime(video_id, updated_at DESC);

-- Cost tracking
CREATE INDEX idx_cost_tracking_date_type ON cost_tracking(date DESC, cost_type);
CREATE INDEX idx_cost_tracking_channel ON cost_tracking(channel_id, date DESC);

-- Pipeline monitoring
CREATE INDEX idx_content_pipeline_status ON content_pipeline(status, pipeline_stage)
    WHERE status IN ('active', 'processing');
CREATE INDEX idx_pipeline_transitions_time ON pipeline_transitions(transition_time DESC);

-- Trend analysis
CREATE INDEX idx_trend_signals_score ON trend_signals(trend_score DESC, signal_timestamp DESC)
    WHERE trend_score >= 30;
CREATE INDEX idx_video_topics_composite ON video_topics(topic_id, video_id);

-- Partial indexes for common filters
CREATE INDEX idx_active_channels ON channels(channel_id) 
    WHERE status = 'active' AND channel_id IN (SELECT channel_id FROM our_channels);
CREATE INDEX idx_recent_videos ON videos(video_id, channel_id) 
    WHERE published_at >= CURRENT_DATE - INTERVAL '30 days';
```

### 7.2 Materialized Views

```sql
-- Materialized views for complex aggregations
-- Refresh these on schedule based on data freshness requirements

-- Daily channel summary (refresh daily at 2 AM)
CREATE MATERIALIZED VIEW mv_daily_channel_summary AS
SELECT 
    c.channel_id,
    c.channel_name,
    c.niche,
    cs.date,
    cs.subscriber_count,
    cs.daily_views,
    cs.daily_revenue_cents / 100.0 AS daily_revenue,
    cs.videos_published,
    cs.health_score,
    -- Calculated metrics
    cs.daily_revenue_cents::FLOAT / NULLIF(cs.daily_views, 0) * 1000 / 100.0 AS rpm,
    cs.subscriber_count - LAG(cs.subscriber_count, 1) OVER (PARTITION BY c.channel_id ORDER BY cs.date) AS subscriber_growth,
    cs.daily_views - LAG(cs.daily_views, 1) OVER (PARTITION BY c.channel_id ORDER BY cs.date) AS view_growth
FROM channels c
JOIN channel_daily_stats cs ON c.channel_id = cs.channel_id
WHERE c.channel_id IN (SELECT channel_id FROM our_channels)
  AND cs.date >= CURRENT_DATE - INTERVAL '90 days';

CREATE UNIQUE INDEX ON mv_daily_channel_summary(channel_id, date);

-- Revenue summary (refresh every 15 minutes)
CREATE MATERIALIZED VIEW mv_revenue_summary AS
SELECT 
    DATE_TRUNC('day', r.date) AS date,
    v.channel_id,
    SUM(r.ad_revenue_cents) / 100.0 AS ad_revenue,
    SUM(r.youtube_premium_cents) / 100.0 AS premium_revenue,
    SUM(r.sponsorship_revenue_cents) / 100.0 AS sponsorship_revenue,
    SUM(r.affiliate_revenue_cents) / 100.0 AS affiliate_revenue,
    SUM(r.revenue_cents) / 100.0 AS total_revenue,
    COUNT(DISTINCT v.video_id) AS monetized_videos
FROM revenue_data r
JOIN videos v ON r.video_id = v.video_id
WHERE v.channel_id IN (SELECT channel_id FROM our_channels)
  AND r.date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', r.date), v.channel_id;

CREATE UNIQUE INDEX ON mv_revenue_summary(date, channel_id);

-- Performance metrics (refresh every hour)
CREATE MATERIALIZED VIEW mv_video_performance_hourly AS
SELECT 
    DATE_TRUNC('hour', vm.timestamp) AS hour,
    v.channel_id,
    COUNT(DISTINCT v.video_id) AS video_count,
    SUM(vm.views) AS total_views,
    AVG(vm.ctr) AS avg_ctr,
    AVG(vm.retention_rate) AS avg_retention,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY vm.views) AS median_views
FROM video_metrics vm
JOIN videos v ON vm.video_id = v.video_id
WHERE v.channel_id IN (SELECT channel_id FROM our_channels)
  AND vm.timestamp >= CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', vm.timestamp), v.channel_id;

CREATE UNIQUE INDEX ON mv_video_performance_hourly(hour, channel_id);

-- Refresh function
CREATE OR REPLACE FUNCTION refresh_dashboard_materializations() 
RETURNS void AS $
BEGIN
    -- High-frequency refreshes
    IF EXTRACT(MINUTE FROM CURRENT_TIMESTAMP) IN (0, 15, 30, 45) THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_revenue_summary;
    END IF;
    
    -- Hourly refreshes
    IF EXTRACT(MINUTE FROM CURRENT_TIMESTAMP) = 0 THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_video_performance_hourly;
    END IF;
    
    -- Daily refreshes (2 AM)
    IF EXTRACT(HOUR FROM CURRENT_TIMESTAMP) = 2 AND EXTRACT(MINUTE FROM CURRENT_TIMESTAMP) = 0 THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_channel_summary;
    END IF;
END;
$ LANGUAGE plpgsql;
```

### 7.3 Query Optimization Patterns

```sql
-- Example: Optimized dashboard query using CTEs and window functions

-- PATTERN: Use CTEs to structure complex queries
WITH base_metrics AS (
    -- First CTE: Get raw data with minimal filtering
    SELECT 
        channel_id,
        date,
        views,
        revenue_cents,
        subscriber_count
    FROM channel_daily_stats
    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
      AND channel_id IN (SELECT channel_id FROM our_channels)
),
calculated_metrics AS (
    -- Second CTE: Perform calculations
    SELECT 
        channel_id,
        date,
        views,
        revenue_cents / 100.0 AS revenue,
        -- Window functions for trends
        views - LAG(views, 7) OVER (PARTITION BY channel_id ORDER BY date) AS wow_view_change,
        AVG(views) OVER (PARTITION BY channel_id ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS views_ma7
    FROM base_metrics
),
summary AS (
    -- Third CTE: Aggregate results
    SELECT 
        channel_id,
        SUM(views) AS total_views,
        SUM(revenue) AS total_revenue,
        AVG(wow_view_change) AS avg_weekly_growth
    FROM calculated_metrics
    GROUP BY channel_id
)
-- Final query: Join with dimension tables only at the end
SELECT 
    c.channel_name,
    s.total_views,
    s.total_revenue,
    s.avg_weekly_growth
FROM summary s
JOIN channels c ON s.channel_id = c.channel_id
ORDER BY s.total_revenue DESC;

-- PATTERN: Use partial aggregation for time-series data
WITH hourly_aggregates AS (
    -- Pre-aggregate at hourly level
    SELECT 
        DATE_TRUNC('hour', timestamp) AS hour,
        channel_id,
        SUM(views) AS hourly_views,
        COUNT(*) AS data_points
    FROM video_metrics_realtime
    WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
    GROUP BY DATE_TRUNC('hour', timestamp), channel_id
)
-- Then aggregate to daily level
SELECT 
    DATE_TRUNC('day', hour) AS day,
    SUM(hourly_views) AS daily_views,
    SUM(data_points) AS total_data_points
FROM hourly_aggregates
GROUP BY DATE_TRUNC('day', hour);

-- PATTERN: Use FILTER clause instead of CASE for conditional aggregation
SELECT 
    channel_id,
    COUNT(*) AS total_videos,
    COUNT(*) FILTER (WHERE published_at >= CURRENT_DATE - INTERVAL '7 days') AS videos_last_7d,
    COUNT(*) FILTER (WHERE published_at >= CURRENT_DATE - INTERVAL '30 days') AS videos_last_30d,
    SUM(views) FILTER (WHERE published_at >= CURRENT_DATE - INTERVAL '7 days') AS views_last_7d
FROM videos
WHERE channel_id IN (SELECT channel_id FROM our_channels)
GROUP BY channel_id;
```

---

## 8. Dashboard Implementation

### 8.1 Grafana Configuration

```yaml
# Grafana dashboard configuration for YTEMPIRE
# /etc/grafana/provisioning/dashboards/ytempire.yaml

apiVersion: 1

providers:
  - name: 'YTEMPIRE Dashboards'
    orgId: 1
    folder: 'YTEMPIRE'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards/ytempire
```

### 8.2 Dashboard JSON Template

```json
{
  "dashboard": {
    "id": null,
    "uid": "ytempire-executive",
    "title": "YTEMPIRE Executive Dashboard",
    "tags": ["ytempire", "executive", "kpi"],
    "timezone": "browser",
    "schemaVersion": 16,
    "version": 0,
    "refresh": "5m",
    "panels": [
      {
        "datasource": "PostgreSQL",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "id": 1,
        "title": "Revenue Trend",
        "type": "graph",
        "targets": [
          {
            "format": "time_series",
            "group": [],
            "metricColumn": "none",
            "rawQuery": true,
            "rawSql": "SELECT date AS time, daily_revenue AS \"Daily Revenue\", revenue_7d_ma AS \"7-Day MA\" FROM mv_revenue_analytics WHERE date >= $__timeFrom() AND date <= $__timeTo() ORDER BY date",
            "refId": "A"
          }
        ],
        "yaxes": [
          {
            "format": "currencyUSD",
            "label": null,
            "logBase": 1,
            "max": null,
            "min": null,
            "show": true
          }
        ]
      },
      {
        "datasource": "PostgreSQL",
        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0},
        "id": 2,
        "title": "Active Channels",
        "type": "stat",
        "targets": [
          {
            "format": "table",
            "group": [],
            "metricColumn": "none",
            "rawQuery": true,
            "rawSql": "SELECT COUNT(*) AS value FROM channels WHERE status = 'active' AND channel_id IN (SELECT channel_id FROM our_channels)",
            "refId": "A"
          }
        ],
        "options": {
          "graphMode": "area",
          "orientation": "auto",
          "reduceOptions": {
            "calcs": ["lastNotNull"],
            "fields": "",
            "values": false
          },
          "textMode": "auto"
        }
      }
    ]
  }
}
```

### 8.3 Query Variables for Dashboards

```sql
-- Grafana variable queries for dynamic dashboards

-- Channel selector
SELECT channel_id AS __value, channel_name AS __text 
FROM channels 
WHERE channel_id IN (SELECT channel_id FROM our_channels) 
  AND status = 'active'
ORDER BY channel_name;

-- Time range presets
SELECT 'last_24h' AS __value, 'Last 24 Hours' AS __text
UNION ALL
SELECT 'last_7d' AS __value, 'Last 7 Days' AS __text
UNION ALL
SELECT 'last_30d' AS __value, 'Last 30 Days' AS __text
UNION ALL
SELECT 'last_90d' AS __value, 'Last 90 Days' AS __text;

-- Niche filter
SELECT DISTINCT niche AS __value, niche AS __text 
FROM channels 
WHERE channel_id IN (SELECT channel_id FROM our_channels)
  AND niche IS NOT NULL
ORDER BY niche;

-- Performance tier filter
SELECT 'all' AS __value, 'All Videos' AS __text
UNION ALL
SELECT 'viral' AS __value, 'Viral' AS __text
UNION ALL
SELECT 'high_performer' AS __value, 'High Performers' AS __text
UNION ALL
SELECT 'average_performer' AS __value, 'Average' AS __text
UNION ALL
SELECT 'underperformer' AS __value, 'Underperformers' AS __text;
```

---

## 9. Alert Queries

### 9.1 Critical Business Alerts

```sql
-- Alert: Daily revenue below target
SELECT 
    CASE 
        WHEN SUM(revenue_cents) / 100.0 < 1000 THEN -- $1000 daily target
            'CRITICAL: Daily revenue below $1000 target'
        ELSE 'OK'
    END AS alert_status,
    SUM(revenue_cents) / 100.0 AS daily_revenue
FROM revenue_data
WHERE date = CURRENT_DATE;

-- Alert: Channel health declining
SELECT 
    channel_name,
    health_score,
    'WARNING: Channel health score below 40' AS alert_message
FROM channel_analytics
WHERE date = CURRENT_DATE
  AND health_score < 40
  AND channel_id IN (SELECT channel_id FROM our_channels);

-- Alert: High cost per video
SELECT 
    AVG(cost_cents) / 100.0 AS avg_cost,
    CASE 
        WHEN AVG(cost_cents) > 50 * 100 THEN 'CRITICAL: Average cost exceeds $0.50'
        WHEN AVG(cost_cents) > 45 * 100 THEN 'WARNING: Average cost exceeds $0.45'
        ELSE 'OK'
    END AS alert_status
FROM cost_tracking
WHERE date >= CURRENT_DATE - INTERVAL '24 hours';

-- Alert: YouTube API quota critical
SELECT 
    SUM(quota_used) AS used,
    10000 - SUM(quota_used) AS remaining,
    CASE 
        WHEN SUM(quota_used) > 9000 THEN 'CRITICAL: API quota >90% used'
        WHEN SUM(quota_used) > 8000 THEN 'WARNING: API quota >80% used'
        ELSE 'OK'
    END AS alert_status
FROM youtube_api_usage
WHERE date = CURRENT_DATE;
```

### 9.2 Operational Alerts

```sql
-- Alert: Pipeline bottlenecks
SELECT 
    pipeline_stage,
    COUNT(*) AS stuck_items,
    AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - entered_stage_at)) / 3600) AS avg_hours_stuck,
    'ALERT: Pipeline bottleneck detected' AS message
FROM content_pipeline
WHERE status = 'active'
  AND entered_stage_at < CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY pipeline_stage
HAVING COUNT(*) > 5;

-- Alert: System performance degradation
WITH performance_check AS (
    SELECT 
        AVG(response_time_ms) AS avg_response,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) AS p95_response,
        COUNT(CASE WHEN status_code >= 500 THEN 1 END) AS error_count
    FROM api_calls
    WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '5 minutes'
)
SELECT 
    CASE 
        WHEN avg_response > 1000 THEN 'CRITICAL: API response time >1s'
        WHEN p95_response > 2000 THEN 'WARNING: P95 response time >2s'
        WHEN error_count > 10 THEN 'CRITICAL: High error rate'
        ELSE 'OK'
    END AS alert_status,
    avg_response,
    p95_response,
    error_count
FROM performance_check;
```

---

## 10. Troubleshooting Guide

### 10.1 Common Performance Issues

| Issue | Symptoms | Solution | Query to Diagnose |
|-------|----------|----------|-------------------|
| Slow dashboard load | >5 second load time | Check indexes, use EXPLAIN | See section 10.2 |
| Stale data | Old timestamps | Check materialized view refresh | See section 10.3 |
| High query CPU | Database CPU >80% | Optimize complex queries | See section 10.4 |
| Memory pressure | OOM errors | Reduce result set size | See section 10.5 |

### 10.2 Query Performance Analysis

```sql
-- Analyze slow queries
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
-- [Insert your slow query here]

-- Find missing indexes
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE schemaname = 'public'
  AND tablename IN ('video_metrics', 'channel_analytics', 'revenue_data')
  AND n_distinct > 100
  AND correlation < 0.1
ORDER BY n_distinct DESC;

-- Monitor active queries
SELECT 
    pid,
    now() - pg_stat_activity.query_start AS duration,
    query,
    state
FROM pg_stat_activity
WHERE (now() - pg_stat_activity.query_start) > interval '2 seconds'
  AND state = 'active'
ORDER BY duration DESC;
```

### 10.3 Materialized View Monitoring

```sql
-- Check materialized view freshness
SELECT 
    schemaname,
    matviewname,
    last_refresh
FROM pg_matviews
JOIN (
    SELECT 
        c.relname,
        MAX(s.last_vacuum) AS last_refresh
    FROM pg_class c
    JOIN pg_stat_user_tables s ON c.relname = s.relname
    WHERE c.relkind = 'm'
    GROUP BY c.relname
) refresh_times ON matviewname = relname
ORDER BY last_refresh;

-- Force refresh if needed
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_revenue_analytics;
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_channel_summary;
```

### 10.4 Query Optimization Checklist

```sql
-- 1. Check for sequential scans on large tables
SELECT 
    schemaname,
    tablename,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch,
    ROUND(100.0 * idx_scan / NULLIF(seq_scan + idx_scan, 0), 2) AS index_usage_pct
FROM pg_stat_user_tables
WHERE seq_scan > 1000
ORDER BY seq_tup_read DESC;

-- 2. Identify tables needing VACUUM
SELECT 
    schemaname,
    tablename,
    n_dead_tup,
    n_live_tup,
    ROUND(n_dead_tup::numeric / NULLIF(n_live_tup + n_dead_tup, 0) * 100, 2) AS dead_tuple_pct,
    last_vacuum,
    last_autovacuum
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000
ORDER BY dead_tuple_pct DESC;

-- 3. Find duplicate indexes
SELECT 
    indrelid::regclass AS table_name,
    array_agg(indexrelid::regclass) AS duplicate_indexes
FROM pg_index
GROUP BY indrelid, indkey
HAVING COUNT(*) > 1;
```

### 10.5 Memory Usage Analysis

```sql
-- Check memory-intensive queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    (shared_blks_hit + shared_blks_read) AS total_blocks,
    ROUND((shared_blks_hit::numeric / NULLIF(shared_blks_hit + shared_blks_read, 0)) * 100, 2) AS cache_hit_pct
FROM pg_stat_statements
WHERE calls > 100
ORDER BY total_blocks DESC
LIMIT 20;

-- Monitor connection memory usage
SELECT 
    COUNT(*) AS connection_count,
    SUM(numbackends) AS total_backends,
    MAX(numbackends) AS max_backends_per_db,
    ROUND(AVG(numbackends), 2) AS avg_backends
FROM pg_stat_database
WHERE datname NOT IN ('template0', 'template1', 'postgres');
```

---

## 11. Best Practices

### 11.1 Query Writing Standards

1. **Always use CTEs for complex queries** - Improves readability and performance
2. **Leverage window functions** - Avoid self-joins for calculations
3. **Use FILTER clause** - More efficient than CASE for conditional aggregation
4. **Index foreign keys** - Critical for join performance
5. **Partition large tables** - Especially time-series data

### 11.2 Dashboard Design Principles

1. **Executive dashboards**: Focus on KPIs and trends, minimize detail
2. **Operational dashboards**: Real-time metrics, actionable insights
3. **Analytical dashboards**: Deep dives, comparative analysis
4. **Load time targets**: Enforce <2s for executive, <5s for operational
5. **Data freshness**: Display last update time on every dashboard

### 11.3 Performance Optimization Checklist

- [ ] Run EXPLAIN ANALYZE on all new queries
- [ ] Create appropriate indexes for WHERE and JOIN conditions
- [ ] Use materialized views for complex aggregations
- [ ] Implement query result caching for stable data
- [ ] Monitor pg_stat_statements for slow queries
- [ ] Schedule VACUUM and ANALYZE regularly
- [ ] Archive old data to maintain performance

---

## Appendices

### Appendix A: Quick Reference - Key Tables

| Table | Purpose | Update Frequency | Key Columns |
|-------|---------|------------------|-------------|
| channels | Channel master data | Real-time | channel_id, status, niche |
| videos | Video metadata | On publish | video_id, channel_id, published_at |
| video_metrics | Performance data | Every 5 min | views, engagement_rate, ctr |
| revenue_data | Monetization | Daily | revenue_cents, rpm |
| cost_tracking | Operational costs | Real-time | cost_cents, cost_type |
| trend_signals | Market intelligence | Every 15 min | trend_score, search_volume |

### Appendix B: Performance Benchmarks

| Metric | Target | Current | Status |
|--------|---------|---------|---------|
| Dashboard Load Time | <2s | 1.8s | ✅ |
| Query Cache Hit Rate | >80% | 85% | ✅ |
| Materialized View Lag | <5 min | 3 min | ✅ |
| Database CPU Usage | <70% | 45% | ✅ |
| API Response Time | <200ms | 150ms | ✅ |

### Appendix C: Emergency Contacts

- **On-call DBA**: analytics-oncall@ytempire.com
- **Dashboard Issues**: analytics-eng@ytempire.com
- **Slack Channel**: #analytics-support
- **Documentation**: https://docs.ytempire.com/analytics

---

*This document is maintained by the Analytics Engineering team. Last comprehensive review: January 2025. For updates or corrections, please submit a PR to the analytics repository.*
```

---

## 4. Operational Dashboard Queries

### 4.1 Real-time Performance Monitor

```sql
-- Real-time Performance Monitor: Live system status
-- Refresh Rate: Every 1 minute
-- Performance Target: < 1 second
-- Purpose: Monitor real-time video performance and system health

WITH realtime_metrics AS (
    SELECT 
        v.video_id,
        v.channel_id,
        v.title,
        v.published_at,
        -- Current metrics
        vm.views,
        vm.likes,
        vm.comments,
        vm.impressions,
        vm.clicks,
        -- Calculated metrics
        vm.clicks::FLOAT / NULLIF(vm.impressions, 0) * 100 AS ctr,
        (vm.likes + vm.comments)::FLOAT / NULLIF(vm.views, 0) * 100 AS engagement_rate,
        -- Velocity calculations
        vm.views - LAG(vm.views, 1) OVER (PARTITION BY v.video_id ORDER BY vm.updated_at) AS view_velocity,
        vm.updated_at
    FROM videos v
    JOIN video_metrics_realtime vm ON v.video_id = vm.video_id
    WHERE vm.updated_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
      AND v.channel_id IN (SELECT channel_id FROM our_channels)
      AND v.published_at >= CURRENT_TIMESTAMP - INTERVAL '48 hours'  -- Focus on recent content
),
performance_summary AS (
    SELECT 
        COUNT(DISTINCT video_id) AS active_videos,
        COUNT(DISTINCT channel_id) AS active_channels,
        SUM(views) AS total_views,
        SUM(view_velocity) AS total_velocity,
        AVG(ctr) AS avg_ctr,
        AVG(engagement_rate) AS avg_engagement,
        MAX(updated_at) AS last_update
    FROM realtime_metrics
),
trending_videos AS (
    SELECT 
        video_id,
        title,
        views,
        view_velocity,
        ctr,
        engagement_rate,
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - published_at)) / 3600 AS hours_since_publish
    FROM realtime_metrics
    WHERE view_velocity > 0
    ORDER BY view_velocity DESC
    LIMIT 10
),
underperforming_videos AS (
    SELECT 
        video_id,
        title,
        views,
        ctr,
        impressions,
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - published_at)) / 3600 AS hours_since_publish
    FROM realtime_metrics
    WHERE ctr < 2.0  -- Below 2% CTR threshold
      AND impressions > 1000  -- Enough data for reliability
      AND published_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
    ORDER BY ctr ASC
    LIMIT 10
)
SELECT 
    -- Summary metrics
    ps.active_videos,
    ps.active_channels,
    ps.total_views,
    ps.total_velocity,
    ROUND(ps.avg_ctr::numeric, 2) AS avg_ctr,
    ROUND(ps.avg_engagement::numeric, 2) AS avg_engagement,
    
    -- System status
    CASE 
        WHEN ps.total_velocity < 0 THEN 'CRITICAL: Negative growth'
        WHEN ps.avg_ctr < 3.0 THEN 'WARNING: Low CTR'
        WHEN ps.active_videos < 50 THEN 'WARNING: Low activity'
        WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ps.last_update)) > 300 THEN 'WARNING: Stale data'
        ELSE 'HEALTHY'
    END AS system_status,
    
    -- Trending videos
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'video_id', video_id,
                'title', LEFT(title, 60),
                'views', views,
                'velocity', view_velocity,
                'ctr', ROUND(ctr::numeric, 2),
                'engagement', ROUND(engagement_rate::numeric, 2),
                'hours_live', ROUND(hours_since_publish::numeric, 1)
            ) ORDER BY view_velocity DESC
        )
        FROM trending_videos
    ) AS trending_now,
    
    -- Underperforming videos needing attention
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'video_id', video_id,
                'title', LEFT(title, 60),
                'views', views,
                'ctr', ROUND(ctr::numeric, 2),
                'impressions', impressions,
                'hours_live', ROUND(hours_since_publish::numeric, 1)
            ) ORDER BY ctr ASC
        )
        FROM underperforming_videos
    ) AS needs_optimization,
    
    -- Data freshness
    ps.last_update AS last_data_update,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ps.last_update)) AS seconds_since_update,
    
    CURRENT_TIMESTAMP AS dashboard_updated
FROM performance_summary ps;
```

### 4.2 Content Pipeline Dashboard

```sql
-- Content Pipeline Dashboard: Production monitoring
-- Refresh Rate: Every 5 minutes
-- Performance Target: < 2 seconds
-- Purpose: Track content creation pipeline efficiency

WITH pipeline_status AS (
    SELECT 
        pipeline_stage,
        COUNT(*) AS items_in_stage,
        AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - entered_stage_at)) / 3600) AS avg_hours_in_stage,
        MAX(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - entered_stage_at)) / 3600) AS max_hours_in_stage,
        COUNT(CASE WHEN entered_stage_at < CURRENT_TIMESTAMP - INTERVAL '24 hours' THEN 1 END) AS stuck_items
    FROM content_pipeline
    WHERE status = 'active'
    GROUP BY pipeline_stage
),
stage_transitions AS (
    SELECT 
        from_stage,
        to_stage,
        COUNT(*) AS transition_count,
        AVG(EXTRACT(EPOCH FROM (transition_time - entry_time)) / 3600) AS avg_transition_hours,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (transition_time - entry_time)) / 3600) AS p95_transition_hours
    FROM pipeline_transitions
    WHERE transition_time >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
    GROUP BY from_stage, to_stage
),
daily_throughput AS (
    SELECT 
        DATE_TRUNC('hour', completed_at) AS hour,
        COUNT(*) AS completed_videos,
        AVG(total_processing_hours) AS avg_processing_time,
        MIN(total_processing_hours) AS min_processing_time,
        MAX(total_processing_hours) AS max_processing_time,
        COUNT(CASE WHEN total_processing_hours <= 8 THEN 1 END)::FLOAT / COUNT(*) * 100 AS within_sla_pct
    FROM content_pipeline_completed
    WHERE completed_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
    GROUP BY DATE_TRUNC('hour', completed_at)
),
channel_pipeline AS (
    SELECT 
        c.channel_id,
        c.channel_name,
        COUNT(cp.content_id) AS items_in_pipeline,
        COUNT(CASE WHEN cp.pipeline_stage = 'ideation' THEN 1 END) AS in_ideation,
        COUNT(CASE WHEN cp.pipeline_stage = 'scripting' THEN 1 END) AS in_scripting,
        COUNT(CASE WHEN cp.pipeline_stage = 'production' THEN 1 END) AS in_production,
        COUNT(CASE WHEN cp.pipeline_stage = 'publishing' THEN 1 END) AS in_publishing
    FROM channels c
    LEFT JOIN content_pipeline cp ON c.channel_id = cp.channel_id AND cp.status = 'active'
    WHERE c.channel_id IN (SELECT channel_id FROM our_channels)
      AND c.status = 'active'
    GROUP BY c.channel_id, c.channel_name
    HAVING COUNT(cp.content_id) > 0
)
SELECT 
    -- Pipeline overview
    (
        SELECT jsonb_object_agg(
            pipeline_stage,
            jsonb_build_object(
                'count', items_in_stage,
                'avg_hours', ROUND(avg_hours_in_stage::numeric, 1),
                'max_hours', ROUND(max_hours_in_stage::numeric, 1),
                'stuck_items', stuck_items,
                'status', CASE 
                    WHEN stuck_items > 5 THEN 'critical'
                    WHEN stuck_items > 2 THEN 'warning'
                    ELSE 'healthy'
                END
            )
        )
        FROM pipeline_status
    ) AS pipeline_stages,
    
    -- Stage flow analysis
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'from', from_stage,
                'to', to_stage,
                'count', transition_count,
                'avg_hours', ROUND(avg_transition_hours::numeric, 1),
                'p95_hours', ROUND(p95_transition_hours::numeric, 1)
            ) ORDER BY transition_count DESC
        )
        FROM stage_transitions
        WHERE transition_count > 0
    ) AS stage_flows,
    
    -- Throughput metrics
    (
        SELECT jsonb_build_object(
            'total_completed_24h', SUM(completed_videos),
            'avg_per_hour', ROUND(AVG(completed_videos)::numeric, 1),
            'avg_processing_hours', ROUND(AVG(avg_processing_time)::numeric, 1),
            'sla_compliance', ROUND(AVG(within_sla_pct)::numeric, 1),
            'peak_hour_completions', MAX(completed_videos)
        )
        FROM daily_throughput
    ) AS throughput_metrics,
    
    -- Bottleneck detection
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'stage', pipeline_stage,
                'stuck_items', stuck_items,
                'avg_wait_hours', ROUND(avg_hours_in_stage::numeric, 1)
            ) ORDER BY stuck_items DESC
        )
        FROM pipeline_status
        WHERE stuck_items > 0
    ) AS bottlenecks,
    
    -- Channel pipeline distribution
    (
        SELECT jsonb_build_object(
            'channels_with_content', COUNT(*),
            'total_items', SUM(items_in_pipeline),
            'avg_per_channel', ROUND(AVG(items_in_pipeline)::numeric, 1),
            'max_per_channel', MAX(items_in_pipeline)
        )
        FROM channel_pipeline
    ) AS channel_distribution,
    
    -- Pipeline health score
    CASE 
        WHEN EXISTS (SELECT 1 FROM pipeline_status WHERE stuck_items > 10) THEN 'critical'
        WHEN EXISTS (SELECT 1 FROM pipeline_status WHERE stuck_items > 5) THEN 'warning'
        WHEN (SELECT AVG(within_sla_pct) FROM daily_throughput) < 80 THEN 'warning'
        ELSE 'healthy'
    END AS pipeline_health,
    
    CURRENT_TIMESTAMP AS dashboard_updated;
```

### 4.3 Cost Tracking Dashboard

```sql
-- Cost Tracking Dashboard: Financial efficiency monitoring
-- Refresh Rate: Every 10 minutes
-- Performance Target: < 2 seconds
-- Purpose: Track and optimize operational costs

WITH cost_breakdown AS (
    SELECT 
        DATE_TRUNC('day', date) AS day,
        -- Cost categories
        SUM(CASE WHEN cost_type = 'ai_generation' THEN cost_cents ELSE 0 END) / 100.0 AS ai_costs,
        SUM(CASE WHEN cost_type = 'youtube_api' THEN cost_cents ELSE 0 END) / 100.0 AS api_costs,
        SUM(CASE WHEN cost_type = 'storage' THEN cost_cents ELSE 0 END) / 100.0 AS storage_costs,
        SUM(CASE WHEN cost_type = 'processing' THEN cost_cents ELSE 0 END) / 100.0 AS processing_costs,
        SUM(CASE WHEN cost_type = 'other' THEN cost_cents ELSE 0 END) / 100.0 AS other_costs,
        SUM(cost_cents) / 100.0 AS total_daily_cost,
        -- Video count for per-video calculations
        COUNT(DISTINCT video_id) AS videos_produced
    FROM cost_tracking
    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY DATE_TRUNC('day', date)
),
channel_costs AS (
    SELECT 
        c.channel_id,
        c.channel_name,
        c.niche,
        COUNT(DISTINCT ct.video_id) AS videos_30d,
        SUM(ct.cost_cents) / 100.0 AS total_costs_30d,
        AVG(ct.cost_cents) / 100.0 AS avg_cost_per_video,
        -- Revenue for ROI calculation
        SUM(r.revenue_cents) / 100.0 AS revenue_30d,
        (SUM(r.revenue_cents) - SUM(ct.cost_cents)) / 100.0 AS profit_30d
    FROM channels c
    JOIN videos v ON c.channel_id = v.channel_id
    JOIN cost_tracking ct ON v.video_id = ct.video_id
    LEFT JOIN revenue_data r ON v.video_id = r.video_id AND r.date = ct.date
    WHERE ct.date >= CURRENT_DATE - INTERVAL '30 days'
      AND c.channel_id IN (SELECT channel_id FROM our_channels)
    GROUP BY c.channel_id, c.channel_name, c.niche
),
cost_trends AS (
    SELECT 
        day,
        total_daily_cost,
        AVG(total_daily_cost) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS cost_7d_ma,
        total_daily_cost / NULLIF(videos_produced, 0) AS cost_per_video,
        -- Cost composition
        ai_costs / NULLIF(total_daily_cost, 0) * 100 AS ai_cost_pct,
        api_costs / NULLIF(total_daily_cost, 0) * 100 AS api_cost_pct
    FROM cost_breakdown
)
SELECT 
    -- Current period summary
    (
        SELECT jsonb_build_object(
            'total_costs_30d', SUM(total_daily_cost),
            'avg_daily_cost', AVG(total_daily_cost),
            'avg_cost_per_video', AVG(cost_per_video),
            'total_videos', SUM(videos_produced),
            'cost_trend', 
                CASE 
                    WHEN AVG(cost_per_video) > 0.50 THEN 'above_target'
                    WHEN AVG(cost_per_video) > 0.45 THEN 'warning'
                    ELSE 'on_target'
                END
        )
        FROM cost_breakdown
        WHERE day >= CURRENT_DATE - INTERVAL '30 days'
    ) AS cost_summary,
    
    -- Cost breakdown by type
    (
        SELECT jsonb_build_object(
            'ai_generation', SUM(ai_costs),
            'youtube_api', SUM(api_costs),
            'storage', SUM(storage_costs),
            'processing', SUM(processing_costs),
            'other', SUM(other_costs)
        )
        FROM cost_breakdown
        WHERE day >= CURRENT_DATE - INTERVAL '30 days'
    ) AS cost_by_type,
    
    -- Daily cost trend
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'date', day,
                'total_cost', total_daily_cost,
                'cost_ma_7d', ROUND(cost_7d_ma::numeric, 2),
                'cost_per_video', ROUND(cost_per_video::numeric, 2),
                'videos', videos_produced
            ) ORDER BY day DESC
        )
        FROM cost_trends
        WHERE day >= CURRENT_DATE - INTERVAL '14 days'
    ) AS daily_trends,
    
    -- Channel profitability ranking
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'channel', channel_name,
                'niche', niche,
                'videos', videos_30d,
                'total_cost', ROUND(total_costs_30d::numeric, 2),
                'avg_cost', ROUND(avg_cost_per_video::numeric, 2),
                'revenue', ROUND(revenue_30d::numeric, 2),
                'profit', ROUND(profit_30d::numeric, 2),
                'roi', ROUND((profit_30d / NULLIF(total_costs_30d, 0) * 100)::numeric, 1)
            ) ORDER BY profit_30d DESC
        )
        FROM channel_costs
        WHERE videos_30d > 0
        LIMIT 20
    ) AS channel_profitability,
    
    -- Cost optimization opportunities
    (
        SELECT jsonb_build_object(
            'high_cost_videos', (
                SELECT COUNT(*) 
                FROM cost_tracking 
                WHERE cost_cents > 50 * 100  -- Videos costing more than $0.50
                  AND date >= CURRENT_DATE - INTERVAL '7 days'
            ),
            'api_quota_usage', (
                SELECT SUM(api_calls) 
                FROM youtube_api_usage 
                WHERE date = CURRENT_DATE
            ),
            'potential_savings', (
                SELECT SUM(cost_cents - 35 * 100) / 100.0  -- Potential savings if all videos cost $0.35
                FROM cost_tracking 
                WHERE cost_cents > 35 * 100 
                  AND date >= CURRENT_DATE - INTERVAL '30 days'
            )
        )
    ) AS optimization_opportunities,
    
    -- Alerts
    CASE 
        WHEN (SELECT AVG(cost_per_video) FROM cost_breakdown WHERE day >= CURRENT_DATE - INTERVAL '7 days') > 0.50 THEN 
            'CRITICAL: Average cost per video exceeds $0.50'
        WHEN (SELECT AVG(cost_per_video) FROM cost_breakdown WHERE day >= CURRENT_DATE - INTERVAL '7 days') > 0.45 THEN 
            'WARNING: Average cost per video exceeds $0.45'
        WHEN (SELECT SUM(api_calls) FROM youtube_api_usage WHERE date = CURRENT_DATE) > 8000 THEN 
            'WARNING: High API usage (>80% of daily quota)'
        ELSE 'All costs within acceptable range'
    END AS cost_alert,
    
    CURRENT_TIMESTAMP AS dashboard_updated;
```

---

## 5. Analytical Dashboard Queries

### 5.1 Trend Analysis Dashboard

```sql
-- Trend Analysis Dashboard: Market intelligence
-- Refresh Rate: Every 15 minutes
-- Performance Target: < 5 seconds
-- Purpose: Identify trending topics and content opportunities

WITH trend_detection AS (
    SELECT 
        t.topic_id,
        t.topic_name,
        t.category,
        ts.search_volume,
        ts.search_volume_24h_ago,
        ts.social_mentions,
        ts.competitor_videos,
        ts.trend_score,
        -- Calculate growth metrics
        (ts.search_volume - ts.search_volume_24h_ago)::FLOAT / 
            NULLIF(ts.search_volume_24h_ago, 0) * 100 AS search_growth_24h,
        -- Our coverage
        COUNT(DISTINCT v.video_id) AS our_videos,
        MAX(v.published_at) AS last_video_date
    FROM trending_topics t
    JOIN trend_signals ts ON t.topic_id = ts.topic_id
    LEFT JOIN video_topics vt ON t.topic_id = vt.topic_id
    LEFT JOIN videos v ON vt.video_id = v.video_id 
        AND v.channel_id IN (SELECT channel_id FROM our_channels)
        AND v.published_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
    WHERE ts.signal_timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
      AND ts.trend_score >= 30  -- Minimum threshold
    GROUP BY t.topic_id, t.topic_name, t.category, ts.search_volume, 
             ts.search_volume_24h_ago, ts.social_mentions, ts.competitor_videos, ts.trend_score
),
opportunity_scoring AS (
    SELECT 
        *,
        -- Calculate opportunity score
        (
            trend_score * 0.3 +
            LEAST(search_growth_24h, 100) * 0.3 +
            CASE 
                WHEN our_videos = 0 THEN 40
                WHEN our_videos < 3 THEN 20
                ELSE 0
            END
        ) AS opportunity_score,
        -- Classification
        CASE 
            WHEN search_growth_24h > 100 AND our_videos = 0 THEN 'hot_opportunity'
            WHEN search_growth_24h > 50 AND our_videos < 2 THEN 'growing_opportunity'
            WHEN trend_score > 70 AND our_videos < 5 THEN 'underserved_trend'
            WHEN search_growth_24h < -20 THEN 'declining_trend'
            ELSE 'stable_trend'
        END AS trend_classification
    FROM trend_detection
),
category_summary AS (
    SELECT 
        category,
        COUNT(*) AS trending_topics,
        AVG(trend_score) AS avg_trend_score,
        COUNT(CASE WHEN trend_classification = 'hot_opportunity' THEN 1 END) AS hot_opportunities,
        SUM(CASE WHEN our_videos = 0 THEN 1 ELSE 0 END) AS uncovered_topics
    FROM opportunity_scoring
    GROUP BY category
)
SELECT 
    -- Top opportunities
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'topic', topic_name,
                'category', category,
                'trend_score', ROUND(trend_score::numeric, 1),
                'search_growth_24h', ROUND(search_growth_24h::numeric, 1),
                'opportunity_score', ROUND(opportunity_score::numeric, 1),
                'our_coverage', our_videos,
                'competitor_activity', competitor_videos,
                'classification', trend_classification,
                'action', CASE 
                    WHEN trend_classification = 'hot_opportunity' THEN 'Create content immediately'
                    WHEN trend_classification = 'growing_opportunity' THEN 'Schedule content creation'
                    WHEN trend_classification = 'underserved_trend' THEN 'Increase coverage'
                    ELSE 'Monitor'
                END
            ) ORDER BY opportunity_score DESC
        )
        FROM opportunity_scoring
        LIMIT 20
    ) AS top_opportunities,
    
    -- Category breakdown
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'category', category,
                'trending_count', trending_topics,
                'avg_score', ROUND(avg_trend_score::numeric, 1),
                'hot_opportunities', hot_opportunities,
                'uncovered', uncovered_topics
            ) ORDER BY hot_opportunities DESC
        )
        FROM category_summary
    ) AS category_analysis,
    
    -- Summary metrics
    (
        SELECT jsonb_build_object(
            'total_trending', COUNT(*),
            'hot_opportunities', COUNT(CASE WHEN trend_classification = 'hot_opportunity' THEN 1 END),
            'our_coverage_rate', ROUND(COUNT(CASE WHEN our_videos > 0 THEN 1 END)::FLOAT / COUNT(*) * 100, 1),
            'avg_opportunity_score', ROUND(AVG(opportunity_score)::numeric, 1)
        )
        FROM opportunity_scoring
    ) AS summary,
    
    -- Trend velocity (last 24 hours)
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'hour', hour,
                'new_trends', new_trends,
                'total_active', total_active
            ) ORDER BY hour
        )
        FROM (
            SELECT 
                DATE_TRUNC('hour', signal_timestamp) AS hour,
                COUNT(DISTINCT CASE WHEN first_detected >= DATE_TRUNC('hour', signal_timestamp) THEN topic_id END) AS new_trends,
                COUNT(DISTINCT topic_id) AS total_active
            FROM trend_signals
            WHERE signal_timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
            GROUP BY DATE_TRUNC('hour', signal_timestamp)
        ) hourly_trends
    ) AS trend_velocity,
    
    CURRENT_TIMESTAMP AS dashboard_updated;
```

### 5.2 Audience Analytics Dashboard

```sql
-- Audience Analytics Dashboard: Viewer insights
-- Refresh Rate: Every 30 minutes
-- Performance Target: < 5 seconds
-- Purpose: Understand audience behavior and preferences

WITH audience_overview AS (
    SELECT 
        COUNT(DISTINCT viewer_id) AS total_viewers,
        COUNT(DISTINCT CASE WHEN first_view_date >= CURRENT_DATE - INTERVAL '30 days' THEN viewer_id END) AS new_viewers_30d,
        COUNT(DISTINCT CASE WHEN last_view_date >= CURRENT_DATE - INTERVAL '7 days' THEN viewer_id END) AS active_viewers_7d,
        AVG(total_watch_time_hours) AS avg_watch_time_hours,
        AVG(videos_watched) AS avg_videos_per_viewer
    FROM viewer_summary
    WHERE channel_id IN (SELECT channel_id FROM our_channels)
),
viewer_segments AS (
    SELECT 
        CASE 
            WHEN total_watch_time_hours >= 10 AND videos_watched >= 20 THEN 'super_fans'
            WHEN total_watch_time_hours >= 5 AND videos_watched >= 10 THEN 'loyal_viewers'
            WHEN total_watch_time_hours >= 1 AND videos_watched >= 3 THEN 'regular_viewers'
            WHEN videos_watched >= 1 THEN 'casual_viewers'
            ELSE 'new_viewers'
        END AS segment,
        COUNT(*) AS viewer_count,
        AVG(total_watch_time_hours) AS avg_watch_hours,
        AVG(videos_watched) AS avg_videos,
        SUM(revenue_attributed_cents) / 100.0 AS revenue_attributed
    FROM viewer_summary
    WHERE channel_id IN (SELECT channel_id FROM our_channels)
    GROUP BY segment
),
content_preferences AS (
    SELECT 
        content_category,
        COUNT(DISTINCT viewer_id) AS interested_viewers,
        AVG(watch_percentage) AS avg_completion,
        SUM(total_views) AS category_views,
        AVG(like_rate) AS avg_like_rate
    FROM viewer_content_affinity
    WHERE channel_id IN (SELECT channel_id FROM our_channels)
      AND view_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY content_category
),
geographic_distribution AS (
    SELECT 
        country,
        COUNT(DISTINCT viewer_id) AS viewers,
        SUM(views) AS total_views,
        AVG(avg_watch_time_minutes) AS avg_watch_time,
        SUM(revenue_cents) / 100.0 AS revenue,
        RANK() OVER (ORDER BY COUNT(DISTINCT viewer_id) DESC) AS rank
    FROM viewer_geography
    WHERE channel_id IN (SELECT channel_id FROM our_channels)
      AND last_view_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY country
),
retention_analysis AS (
    SELECT 
        cohort_month,
        cohort_size,
        retention_day_1,
        retention_day_7,
        retention_day_30,
        (retention_day_30::FLOAT / cohort_size) * 100 AS day_30_retention_rate
    FROM viewer_retention_cohorts
    WHERE cohort_month >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '6 months')
    ORDER BY cohort_month DESC
)
SELECT 
    -- Audience overview
    (
        SELECT jsonb_build_object(
            'total_viewers', total_viewers,
            'new_viewers_30d', new_viewers_30d,
            'active_viewers_7d', active_viewers_7d,
            'avg_watch_hours', ROUND(avg_watch_time_hours::numeric, 1),
            'avg_videos_per_viewer', ROUND(avg_videos_per_viewer::numeric, 1)
        )
        FROM audience_overview
    ) AS overview,
    
    -- Viewer segments
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'segment', segment,
                'count', viewer_count,
                'percentage', ROUND(viewer_count::FLOAT / SUM(viewer_count) OVER () * 100, 1),
                'avg_watch_hours', ROUND(avg_watch_hours::numeric, 1),
                'avg_videos', ROUND(avg_videos::numeric, 1),
                'revenue', ROUND(revenue_attributed::numeric, 2),
                'revenue_per_viewer', ROUND(revenue_attributed / viewer_count::numeric, 2)
            ) ORDER BY 
                CASE segment
                    WHEN 'super_fans' THEN 1
                    WHEN 'loyal_viewers' THEN 2
                    WHEN 'regular_viewers' THEN 3
                    WHEN 'casual_viewers' THEN 4
                    ELSE 5
                END
        )
        FROM viewer_segments
    ) AS segments,
    
    -- Content preferences
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'category', content_category,
                'interested_viewers', interested_viewers,
                'total_views', category_views,
                'avg_completion', ROUND(avg_completion::numeric * 100, 1),
                'avg_like_rate', ROUND(avg_like_rate::numeric * 100, 2)
            ) ORDER BY interested_viewers DESC
        )
        FROM content_preferences
        LIMIT 10
    ) AS content_preferences,
    
    -- Geographic insights
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'country', country,
                'viewers', viewers,
                'views', total_views,
                'avg_watch_minutes', ROUND(avg_watch_time::numeric, 1),
                'revenue', ROUND(revenue::numeric, 2),
                'rpm', ROUND((revenue / NULLIF(total_views, 0) * 1000)::numeric, 2)
            ) ORDER BY viewers DESC
        )
        FROM geographic_distribution
        WHERE rank <= 20
    ) AS top_countries,
    
    -- Retention trends
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'cohort', cohort_month,
                'size', cohort_size,
                'day_1', ROUND((retention_day_1::FLOAT / cohort_size * 100)::numeric, 1),
                'day_7', ROUND((retention_day_7::FLOAT / cohort_size * 100)::numeric, 1),
                'day_30', ROUND(day_30_retention_rate::numeric, 1)
            ) ORDER BY cohort_month DESC
        )
        FROM retention_analysis
        LIMIT 6
    ) AS retention_cohorts,
    
    CURRENT_TIMESTAMP AS dashboard_updated;
```

### 5.3 Content Performance Analytics

```sql
-- Content Performance Analytics: Deep content analysis
-- Refresh Rate: Every 20 minutes
-- Performance Target: < 3 seconds
-- Purpose: Analyze content performance patterns

WITH video_performance AS (
    SELECT 
        v.video_id,
        v.title,
        v.channel_id,
        c.channel_name,
        c.niche,
        v.published_at,
        v.duration_seconds,
        vm.views,
        vm.likes,
        vm.comments,
        vm.shares,
        vm.impressions,
        vm.clicks,
        r.revenue_cents / 100.0 AS revenue,
        -- Calculated metrics
        vm.clicks::FLOAT / NULLIF(vm.impressions, 0) * 100 AS ctr,
        (vm.likes + vm.comments)::FLOAT / NULLIF(vm.views, 0) * 100 AS engagement_rate,
        vm.average_view_duration_seconds::FLOAT / NULLIF(v.duration_seconds, 0) * 100 AS retention_rate,
        (r.revenue_cents / NULLIF(vm.views, 0) * 1000) / 100.0 AS rpm,
        -- Time-based metrics
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - v.published_at)) / 86400 AS days_since_publish,
        vm.views / NULLIF(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - v.published_at)) / 86400, 0) AS daily_view_rate
    FROM videos v
    JOIN channels c ON v.channel_id = c.channel_id
    JOIN video_metrics vm ON v.video_id = vm.video_id
    LEFT JOIN revenue_data r ON v.video_id = r.video_id
    WHERE v.channel_id IN (SELECT channel_id FROM our_channels)
      AND v.published_at >= CURRENT_DATE - INTERVAL '30 days'
),
performance_tiers AS (
    SELECT 
        *,
        -- Performance classification
        CASE 
            WHEN views > 100000 AND engagement_rate > 10 THEN 'viral'
            WHEN views > 50000 AND engagement_rate > 8 THEN 'high_performer'
            WHEN views > 10000 AND engagement_rate > 5 THEN 'solid_performer'
            WHEN views > 5000 THEN 'average_performer'
            ELSE 'underperformer'
        END AS performance_tier,
        -- Viral score
        (
            (LOG(views + 1) / LOG(10)) * 20 +
            engagement_rate * 3 +
            LEAST(daily_view_rate / 1000, 50)
        ) AS viral_score
    FROM video_performance
),
content_patterns AS (
    SELECT 
        niche,
        performance_tier,
        COUNT(*) AS video_count,
        AVG(views) AS avg_views,
        AVG(engagement_rate) AS avg_engagement,
        AVG(retention_rate) AS avg_retention,
        AVG(rpm) AS avg_rpm,
        AVG(viral_score) AS avg_viral_score
    FROM performance_tiers
    GROUP BY niche, performance_tier
),
optimal_characteristics AS (
    SELECT 
        -- Duration analysis
        CASE 
            WHEN duration_seconds < 60 THEN 'shorts'
            WHEN duration_seconds < 300 THEN 'short_form'
            WHEN duration_seconds < 600 THEN 'medium_form'
            ELSE 'long_form'
        END AS duration_category,
        AVG(views) AS avg_views,
        AVG(retention_rate) AS avg_retention,
        COUNT(*) AS video_count
    FROM video_performance
    GROUP BY duration_category
)
SELECT 
    -- Top performing videos
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'video_id', video_id,
                'title', LEFT(title, 60),
                'channel', channel_name,
                'views', views,
                'engagement_rate', ROUND(engagement_rate::numeric, 2),
                'rpm', ROUND(rpm::numeric, 2),
                'revenue', ROUND(revenue::numeric, 2),
                'viral_score', ROUND(viral_score::numeric, 1),
                'performance_tier', performance_tier,
                'days_live', ROUND(days_since_publish::numeric, 1)
            ) ORDER BY viral_score DESC
        )
        FROM performance_tiers
        WHERE performance_tier IN ('viral', 'high_performer')
        LIMIT 15
    ) AS top_performers,
    
    -- Performance by niche
    (
        SELECT jsonb_object_agg(
            niche,
            jsonb_build_object(
                'total_videos', SUM(video_count),
                'avg_views', ROUND(AVG(avg_views)::numeric, 0),
                'avg_engagement', ROUND(AVG(avg_engagement)::numeric, 2),
                'avg_rpm', ROUND(AVG(avg_rpm)::numeric, 2),
                'viral_rate', ROUND(SUM(CASE WHEN performance_tier = 'viral' THEN video_count ELSE 0 END)::FLOAT / 
                             SUM(video_count) * 100, 1)
            )
        )
        FROM content_patterns
        GROUP BY niche
    ) AS niche_performance,
    
    -- Performance distribution
    (
        SELECT jsonb_object_agg(
            performance_tier,
            jsonb_build_object(
                'count', COUNT(*),
                'percentage', ROUND(COUNT(*)::FLOAT / SUM(COUNT(*)) OVER () * 100, 1),
                'avg_revenue', ROUND(AVG(revenue)::numeric, 2),
                'total_views', SUM(views)
            )
        )
        FROM performance_tiers
        GROUP BY performance_tier
    ) AS performance_distribution,
    
    -- Optimal content characteristics
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'duration_type', duration_category,
                'avg_views', ROUND(avg_views::numeric, 0),
                'avg_retention', ROUND(avg_retention::numeric, 1),
                'video_count', video_count,
                'recommendation', CASE 
                    WHEN avg_views * avg_retention / 100 = 
                         MAX(avg_views * avg_retention / 100) OVER () 
                    THEN 'optimal_duration'
                    ELSE 'suboptimal'
                END
            ) ORDER BY avg_views * avg_retention / 100 DESC
        )
        FROM optimal_characteristics
    ) AS duration_analysis,
    
    -- Content opportunities
    (
        SELECT jsonb_build_object(
            'underperforming_count', COUNT(CASE WHEN performance_tier = 'underperformer' THEN 1 END),
            'optimization_potential', SUM(CASE 
                WHEN ctr < 3 AND impressions > 1000 THEN 1 
                ELSE 0 
            END),
            'high_potential_topics', (
                SELECT COUNT(DISTINCT topic_id)
                FROM video_topics vt
                JOIN performance_tiers pt ON vt.video_id = pt.video_id
                WHERE pt.performance_tier IN ('viral', 'high_performer')
            )
        )
        FROM performance_tiers
    ) AS opportunities,
    
    CURRENT_TIMESTAMP AS dashboard_updated;
```

---

## 6. System Health & Monitoring Queries

### 6.1 System Health Dashboard

```sql
-- System Health Dashboard: Infrastructure monitoring
-- Refresh Rate: Every 2 minutes
-- Performance Target: < 1 second
-- Purpose: Monitor system health and performance

WITH system_metrics AS (
    SELECT 
        -- Database metrics
        (SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active') AS active_connections,
        (SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'idle') AS idle_connections,
        (SELECT ROUND(100.0 * SUM(blks_hit) / NULLIF(SUM(blks_hit + blks_read), 0), 2) 
         FROM pg_stat_database WHERE datname = current_database()) AS cache_hit_ratio,
        (SELECT pg_database_size(current_database()) / 1024 / 1024 / 1024) AS database_size_gb,
        
        -- API metrics
        (SELECT COUNT(*) FROM api_calls WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '5 minutes') AS api_calls_5min,
        (SELECT AVG(response_time_ms) FROM api_calls WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '5 minutes') AS avg_api_response_ms,
        (SELECT COUNT(*) FROM api_calls WHERE status_code >= 500 AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '5 minutes') AS api_errors_5min,
        
        -- Queue metrics
        (SELECT COUNT(*) FROM job_queue WHERE status = 'pending') AS pending_jobs,
        (SELECT COUNT(*) FROM job_queue WHERE status = 'processing') AS processing_jobs,
        (SELECT AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at))) FROM job_queue WHERE status = 'pending') AS avg_queue_wait_seconds
),
pipeline_health AS (
    SELECT 
        pipeline_name,
        last_run_time,
        last_run_status,
        run_duration_seconds,
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_run_time)) AS seconds_since_run,
        CASE 
            WHEN last_run_status = 'failed' THEN 'critical'
            WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_run_time)) > expected_interval_seconds * 1.5 THEN 'warning'
            ELSE 'healthy'
        END AS health_status
    FROM data_pipelines
    WHERE is_active = true
),
youtube_api_health AS (
    SELECT 
        SUM(quota_used) AS quota_used_today,
        10000 - SUM(quota_used) AS quota_remaining,
        (SUM(quota_used)::FLOAT / 10000) * 100 AS quota_percentage,
        MAX(last_call_time) AS last_api_call,
        COUNT(DISTINCT operation_type) AS unique_operations
    FROM youtube_api_usage
    WHERE date = CURRENT_DATE
),
storage_metrics AS (
    SELECT 
        storage_type,
        used_gb,
        total_gb,
        (used_gb::FLOAT / total_gb) * 100 AS usage_percentage,
        CASE 
            WHEN (used_gb::FLOAT / total_gb) > 0.9 THEN 'critical'
            WHEN (used_gb::FLOAT / total_gb) > 0.8 THEN 'warning'
            ELSE 'healthy'
        END AS storage_status
    FROM storage_monitoring
    WHERE check_time = (SELECT MAX(check_time) FROM storage_monitoring)
)
SELECT 
    -- System overview
    (
        SELECT jsonb_build_object(
            'database_connections', active_connections + idle_connections,
            'active_connections', active_connections,
            'cache_hit_ratio', cache_hit_ratio,
            'database_size_gb', database_size_gb,
            'api_calls_5min', api_calls_5min,
            'avg_api_response_ms', ROUND(avg_api_response_ms::numeric, 1),
            'api_errors', api_errors_5min,
            'pending_jobs', pending_jobs,
            'processing_jobs', processing_jobs,
            'avg_queue_wait', ROUND(avg_queue_wait_seconds::numeric, 1)
        )
        FROM system_metrics
    ) AS system_stats,
    
    -- Pipeline status
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'pipeline', pipeline_name,
                'last_run', last_run_time,
                'status', last_run_status,
                'duration', run_duration_seconds,
                'health', health_status,
                'overdue', CASE 
                    WHEN seconds_since_run > expected_interval_seconds * 1.5 
                    THEN true ELSE false 
                END
            ) ORDER BY 
                CASE health_status 
                    WHEN 'critical' THEN 1 
                    WHEN 'warning' THEN 2 
                    ELSE 3 
                END
        )
        FROM pipeline_health
    ) AS pipelines,
    
    -- YouTube API status
    (
        SELECT jsonb_build_object(
            'quota_used', quota_used_today,
            'quota_remaining', quota_remaining,
            'quota_percentage', ROUND(quota_percentage::numeric, 1),
            'last_call', last_api_call,
            'status', CASE 
                WHEN quota_percentage > 90 THEN 'critical'
                WHEN quota_percentage > 80 THEN 'warning'
                ELSE 'healthy'
            END
        )
        FROM youtube_api_health
    ) AS youtube_api,
    
    -- Storage status
    (
        SELECT jsonb_object_agg(
            storage_type,
            jsonb_build_object(
                'used_gb', used_gb,
                'total_gb', total_gb,
                'usage_pct', ROUND(usage_percentage::numeric, 1),
                'status', storage_status
            )
        )
        FROM storage_metrics
    ) AS storage,
    
    -- Overall health score
    CASE 
        WHEN EXISTS (SELECT 1 FROM pipeline_health WHERE health_status = 'critical') THEN 'critical'
        WHEN (SELECT quota_percentage FROM youtube_api_health) > 90 THEN 'critical'
        WHEN EXISTS (SELECT 1 FROM storage_metrics WHERE storage_status = 'critical') THEN 'critical'
        WHEN EXISTS (SELECT 1 FROM pipeline_health WHERE health_status = 'warning') THEN 'warning'
        WHEN (SELECT quota_percentage FROM youtube_api_health) > 80 THEN 'warning'
        WHEN (SELECT api_errors_5min FROM system_metrics) > 10 THEN 'warning'
        ELSE 'healthy'
    END AS overall_health,
    
    -- Alerts
    (
        WITH active_alerts AS (
            SELECT 
                alert_type,
                severity,
                message,
                created_at
            FROM system_alerts
            WHERE acknowledged = false
              AND created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
            ORDER BY 
                CASE severity 
                    WHEN 'critical' THEN 1 
                    WHEN 'high' THEN 2 
                    WHEN 'medium' THEN 3 
                    ELSE 4 
                END,
                created_at DESC
            LIMIT 10
        )
        SELECT jsonb_agg(
            jsonb_build_object(
                'type', alert_type,
                'severity', severity,
                'message', message,
                'time', created_at
            )
        )
        FROM active_alerts
    ) AS active_alerts,
    
    CURRENT_TIMESTAMP AS dashboard_updated;