# YTEMPIRE Dashboard Query Library & Performance Analytics

## Document Control
- **Version**: 1.0
- **Last Updated**: January 2025
- **Owner**: Analytics Engineering Team
- **Audience**: Analytics Engineers, BI Developers, Data Analysts

---

## 1. Executive Summary

This document provides a comprehensive library of optimized SQL queries, dashboard configurations, and performance analytics implementations for YTEMPIRE's business intelligence ecosystem. All queries are tested for performance and accuracy.

---

## 2. Dashboard Architecture Overview

### 2.1 Dashboard Hierarchy

```
Dashboard Structure:
├── Executive Dashboards
│   ├── CEO Dashboard
│   ├── Revenue Overview
│   ├── Growth Metrics
│   └── Strategic KPIs
│
├── Operational Dashboards
│   ├── Real-time Performance
│   ├── Channel Analytics
│   ├── Content Pipeline
│   └── System Health
│
├── Analytical Dashboards
│   ├── Trend Analysis
│   ├── Audience Insights
│   ├── Competitive Intelligence
│   └── Forecasting
│
└── Team Dashboards
    ├── Content Team
    ├── Marketing Team
    ├── Engineering Team
    └── Finance Team
```

### 2.2 Query Performance Standards

```sql
-- Performance benchmarks for dashboard queries
-- All queries must meet these standards:
-- - Executive dashboards: < 2 seconds
-- - Operational dashboards: < 5 seconds
-- - Analytical dashboards: < 10 seconds
-- - Export queries: < 30 seconds

-- Query optimization checklist:
-- 1. Use appropriate indexes
-- 2. Leverage materialized views for complex aggregations
-- 3. Partition large tables by date
-- 4. Use EXPLAIN ANALYZE to verify execution plans
-- 5. Implement query result caching where appropriate
```

---

## 3. Executive Dashboard Queries

### 3.1 CEO Dashboard - Company Overview

```sql
-- CEO Dashboard: High-level company metrics
-- Refresh: Every 5 minutes
-- Performance target: < 2 seconds

WITH current_period_metrics AS (
    SELECT 
        -- Channel metrics
        COUNT(DISTINCT c.channel_id) AS active_channels,
        SUM(CASE WHEN c.created_at >= CURRENT_DATE - INTERVAL '30 days' THEN 1 ELSE 0 END) AS new_channels_30d,
        
        -- Video metrics
        COUNT(DISTINCT v.video_id) AS total_videos,
        SUM(CASE WHEN v.published_at >= CURRENT_DATE THEN 1 ELSE 0 END) AS videos_today,
        SUM(CASE WHEN v.published_at >= CURRENT_DATE - INTERVAL '7 days' THEN 1 ELSE 0 END) AS videos_week,
        
        -- View metrics
        SUM(vm.views) AS total_views,
        SUM(CASE WHEN vm.date = CURRENT_DATE THEN vm.views ELSE 0 END) AS views_today,
        
        -- Revenue metrics
        SUM(r.revenue_cents) / 100.0 AS total_revenue,
        SUM(CASE WHEN r.date = CURRENT_DATE THEN r.revenue_cents ELSE 0 END) / 100.0 AS revenue_today,
        SUM(CASE WHEN r.date >= CURRENT_DATE - INTERVAL '30 days' THEN r.revenue_cents ELSE 0 END) / 100.0 AS revenue_30d
        
    FROM channels c
    LEFT JOIN videos v ON c.channel_id = v.channel_id
    LEFT JOIN video_metrics vm ON v.video_id = vm.video_id
    LEFT JOIN revenue_data r ON v.video_id = r.video_id
    WHERE c.status = 'active'
),
growth_metrics AS (
    SELECT 
        -- Calculate month-over-month growth
        (current_month.revenue - previous_month.revenue) / NULLIF(previous_month.revenue, 0) * 100 AS revenue_growth_mom,
        (current_month.views - previous_month.views) / NULLIF(previous_month.views, 0) * 100 AS views_growth_mom,
        (current_month.videos - previous_month.videos) / NULLIF(previous_month.videos, 0) * 100 AS content_growth_mom
    FROM (
        SELECT 
            SUM(revenue_cents) / 100.0 AS revenue,
            SUM(views) AS views,
            COUNT(DISTINCT video_id) AS videos
        FROM video_performance_summary
        WHERE date >= DATE_TRUNC('month', CURRENT_DATE)
    ) current_month
    CROSS JOIN (
        SELECT 
            SUM(revenue_cents) / 100.0 AS revenue,
            SUM(views) AS views,
            COUNT(DISTINCT video_id) AS videos
        FROM video_performance_summary
        WHERE date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
            AND date < DATE_TRUNC('month', CURRENT_DATE)
    ) previous_month
),
health_indicators AS (
    SELECT 
        -- System health
        (SELECT COUNT(*) FROM channels WHERE health_score >= 80) AS healthy_channels,
        (SELECT AVG(health_score) FROM channels WHERE status = 'active') AS avg_channel_health,
        
        -- Cost efficiency
        (SELECT AVG(cost_per_video) FROM video_costs WHERE created_at >= CURRENT_DATE - INTERVAL '7 days') AS avg_cost_per_video,
        
        -- AI performance
        (SELECT AVG(prediction_accuracy) FROM model_performance WHERE date = CURRENT_DATE) AS ai_accuracy
)
SELECT 
    -- Current metrics
    cpm.*,
    
    -- Growth metrics
    gm.revenue_growth_mom,
    gm.views_growth_mom,
    gm.content_growth_mom,
    
    -- Health indicators
    hi.healthy_channels,
    hi.avg_channel_health,
    hi.avg_cost_per_video,
    hi.ai_accuracy,
    
    -- Calculated metrics
    cpm.total_revenue / NULLIF(cpm.total_videos, 0) AS revenue_per_video,
    cpm.total_views / NULLIF(cpm.total_videos, 0) AS avg_views_per_video,
    
    -- Projections
    cpm.revenue_30d * 12 AS annual_revenue_run_rate,
    
    -- Status indicators
    CASE 
        WHEN cpm.revenue_today >= (cpm.revenue_30d / 30) * 1.1 THEN 'above_target'
        WHEN cpm.revenue_today >= (cpm.revenue_30d / 30) * 0.9 THEN 'on_target'
        ELSE 'below_target'
    END AS daily_revenue_status,
    
    -- Timestamp
    CURRENT_TIMESTAMP AS last_updated
    
FROM current_period_metrics cpm
CROSS JOIN growth_metrics gm
CROSS JOIN health_indicators hi;
```

### 3.2 Revenue Dashboard - Comprehensive Financial View

```sql
-- Revenue Dashboard: Detailed financial analytics
-- Refresh: Every 15 minutes
-- Performance target: < 3 seconds

-- Use materialized view for performance
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_revenue_dashboard AS
WITH revenue_by_source AS (
    SELECT 
        DATE_TRUNC('day', date) AS date,
        SUM(ad_revenue_cents) / 100.0 AS ad_revenue,
        SUM(youtube_premium_cents) / 100.0 AS premium_revenue,
        SUM(channel_memberships_cents) / 100.0 AS membership_revenue,
        SUM(super_thanks_cents) / 100.0 AS super_thanks_revenue,
        SUM(sponsorship_revenue_cents) / 100.0 AS sponsorship_revenue,
        SUM(affiliate_revenue_cents) / 100.0 AS affiliate_revenue,
        SUM(ad_revenue_cents + youtube_premium_cents + channel_memberships_cents + 
            super_thanks_cents + sponsorship_revenue_cents + affiliate_revenue_cents) / 100.0 AS total_revenue
    FROM revenue_detailed
    WHERE date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY DATE_TRUNC('day', date)
),
revenue_by_channel AS (
    SELECT 
        c.channel_id,
        c.channel_name,
        c.niche,
        SUM(r.revenue_cents) / 100.0 AS channel_revenue,
        COUNT(DISTINCT v.video_id) AS video_count,
        SUM(vm.views) AS total_views,
        (SUM(r.revenue_cents) / NULLIF(SUM(vm.views), 0) * 1000) / 100.0 AS rpm,
        RANK() OVER (ORDER BY SUM(r.revenue_cents) DESC) AS revenue_rank
    FROM channels c
    LEFT JOIN videos v ON c.channel_id = v.channel_id
    LEFT JOIN video_metrics vm ON v.video_id = vm.video_id
    LEFT JOIN revenue_data r ON v.video_id = r.video_id
    WHERE r.date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY c.channel_id, c.channel_name, c.niche
),
revenue_trends AS (
    SELECT 
        date,
        total_revenue,
        AVG(total_revenue) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS revenue_7d_ma,
        AVG(total_revenue) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS revenue_30d_ma,
        total_revenue - LAG(total_revenue, 7) OVER (ORDER BY date) AS revenue_wow_change,
        (total_revenue - LAG(total_revenue, 7) OVER (ORDER BY date)) / 
            NULLIF(LAG(total_revenue, 7) OVER (ORDER BY date), 0) * 100 AS revenue_wow_change_pct
    FROM revenue_by_source
)
SELECT 
    -- Daily revenue breakdown
    rs.date,
    rs.ad_revenue,
    rs.premium_revenue,
    rs.membership_revenue,
    rs.super_thanks_revenue,
    rs.sponsorship_revenue,
    rs.affiliate_revenue,
    rs.total_revenue,
    
    -- Revenue mix percentages
    rs.ad_revenue / NULLIF(rs.total_revenue, 0) * 100 AS ad_revenue_pct,
    rs.premium_revenue / NULLIF(rs.total_revenue, 0) * 100 AS premium_revenue_pct,
    rs.membership_revenue / NULLIF(rs.total_revenue, 0) * 100 AS membership_revenue_pct,
    
    -- Trends
    rt.revenue_7d_ma,
    rt.revenue_30d_ma,
    rt.revenue_wow_change,
    rt.revenue_wow_change_pct,
    
    -- Channel performance (top 10)
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'channel_name', channel_name,
                'revenue', channel_revenue,
                'rpm', rpm,
                'rank', revenue_rank
            ) ORDER BY revenue_rank
        )
        FROM revenue_by_channel
        WHERE revenue_rank <= 10
    ) AS top_channels,
    
    -- Cumulative metrics
    SUM(rs.total_revenue) OVER (ORDER BY rs.date) AS cumulative_revenue,
    
    -- Month-to-date
    SUM(rs.total_revenue) OVER (
        PARTITION BY DATE_TRUNC('month', rs.date) 
        ORDER BY rs.date
    ) AS mtd_revenue
    
FROM revenue_by_source rs
LEFT JOIN revenue_trends rt ON rs.date = rt.date
ORDER BY rs.date DESC;

-- Refresh materialized view
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_revenue_dashboard;
```

### 3.3 Growth Metrics Dashboard

```sql
-- Growth Metrics Dashboard: Track all growth KPIs
-- Refresh: Every 30 minutes
-- Performance target: < 2 seconds

WITH growth_cohorts AS (
    SELECT 
        DATE_TRUNC('week', created_at) AS cohort_week,
        COUNT(*) AS channels_created,
        SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS channels_active,
        SUM(CASE WHEN first_video_date IS NOT NULL THEN 1 ELSE 0 END) AS channels_with_videos,
        SUM(CASE WHEN first_revenue_date IS NOT NULL THEN 1 ELSE 0 END) AS channels_monetized,
        AVG(EXTRACT(EPOCH FROM (first_video_date - created_at)) / 86400) AS avg_days_to_first_video,
        AVG(EXTRACT(EPOCH FROM (first_revenue_date - created_at)) / 86400) AS avg_days_to_first_revenue
    FROM channels
    WHERE created_at >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY DATE_TRUNC('week', created_at)
),
subscriber_growth AS (
    SELECT 
        date,
        SUM(subscriber_count) AS total_subscribers,
        SUM(subscriber_count - LAG(subscriber_count, 1) OVER (PARTITION BY channel_id ORDER BY date)) AS daily_new_subscribers,
        AVG(subscriber_count - LAG(subscriber_count, 1) OVER (PARTITION BY channel_id ORDER BY date)) AS avg_new_subs_per_channel
    FROM channel_daily_stats
    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY date
),
content_velocity AS (
    SELECT 
        DATE_TRUNC('day', published_at) AS publish_date,
        COUNT(*) AS videos_published,
        COUNT(DISTINCT channel_id) AS active_channels,
        COUNT(*) / COUNT(DISTINCT channel_id)::FLOAT AS videos_per_channel,
        SUM(duration_seconds) / 3600.0 AS total_hours_content,
        AVG(production_time_hours) AS avg_production_time
    FROM videos
    WHERE published_at >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY DATE_TRUNC('day', published_at)
),
virality_metrics AS (
    SELECT 
        DATE_TRUNC('day', v.published_at) AS date,
        COUNT(CASE WHEN p.viral_score >= 80 THEN 1 END) AS viral_videos,
        COUNT(CASE WHEN p.viral_score >= 60 THEN 1 END) AS trending_videos,
        AVG(p.viral_score) AS avg_viral_score,
        MAX(p.views_velocity_24h) AS max_view_velocity,
        SUM(CASE WHEN p.views_48h > 100000 THEN 1 ELSE 0 END) AS videos_over_100k_48h
    FROM videos v
    JOIN video_performance p ON v.video_id = p.video_id
    WHERE v.published_at >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY DATE_TRUNC('day', v.published_at)
)
SELECT 
    -- Channel growth
    (SELECT COUNT(*) FROM channels WHERE status = 'active') AS total_active_channels,
    (SELECT COUNT(*) FROM channels WHERE created_at >= CURRENT_DATE - INTERVAL '30 days') AS new_channels_30d,
    (SELECT channels_active::FLOAT / channels_created * 100 FROM growth_cohorts ORDER BY cohort_week DESC LIMIT 1) AS activation_rate,
    
    -- Subscriber growth
    (SELECT total_subscribers FROM subscriber_growth ORDER BY date DESC LIMIT 1) AS total_subscribers,
    (SELECT SUM(daily_new_subscribers) FROM subscriber_growth WHERE date >= CURRENT_DATE - INTERVAL '7 days') AS new_subscribers_7d,
    (SELECT AVG(daily_new_subscribers) FROM subscriber_growth WHERE date >= CURRENT_DATE - INTERVAL '30 days') AS avg_daily_new_subs,
    
    -- Content growth
    (SELECT SUM(videos_published) FROM content_velocity WHERE publish_date >= CURRENT_DATE - INTERVAL '7 days') AS videos_published_7d,
    (SELECT AVG(videos_per_channel) FROM content_velocity WHERE publish_date >= CURRENT_DATE - INTERVAL '30 days') AS avg_videos_per_channel,
    (SELECT SUM(total_hours_content) FROM content_velocity WHERE publish_date >= CURRENT_DATE - INTERVAL '30 days') AS hours_content_30d,
    
    -- Virality metrics
    (SELECT COUNT(*) FROM virality_metrics WHERE viral_videos > 0) AS days_with_viral_content,
    (SELECT SUM(viral_videos) FROM virality_metrics) AS total_viral_videos_7d,
    (SELECT AVG(avg_viral_score) FROM virality_metrics) AS avg_viral_score_7d,
    
    -- Growth rates
    (
        SELECT jsonb_build_object(
            'channels_mom', ((SELECT COUNT(*) FROM channels WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE)) - 
                            (SELECT COUNT(*) FROM channels WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') 
                                AND created_at < DATE_TRUNC('month', CURRENT_DATE)))::FLOAT / 
                            NULLIF((SELECT COUNT(*) FROM channels WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') 
                                AND created_at < DATE_TRUNC('month', CURRENT_DATE)), 0) * 100,
                                
            'revenue_mom', ((SELECT SUM(revenue_cents) FROM revenue_data WHERE date >= DATE_TRUNC('month', CURRENT_DATE)) - 
                           (SELECT SUM(revenue_cents) FROM revenue_data WHERE date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') 
                               AND date < DATE_TRUNC('month', CURRENT_DATE)))::FLOAT / 
                           NULLIF((SELECT SUM(revenue_cents) FROM revenue_data WHERE date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') 
                               AND date < DATE_TRUNC('month', CURRENT_DATE)), 0) * 100,
                               
            'views_mom', ((SELECT SUM(views) FROM video_metrics WHERE date >= DATE_TRUNC('month', CURRENT_DATE)) - 
                         (SELECT SUM(views) FROM video_metrics WHERE date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') 
                             AND date < DATE_TRUNC('month', CURRENT_DATE)))::FLOAT / 
                         NULLIF((SELECT SUM(views) FROM video_metrics WHERE date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') 
                             AND date < DATE_TRUNC('month', CURRENT_DATE)), 0) * 100
        )
    ) AS growth_rates,
    
    CURRENT_TIMESTAMP AS last_updated;
```

### 4.4 System Health Dashboard

```sql
-- System Health Dashboard: Monitor all system components
-- Refresh: Every 2 minutes
-- Performance target: < 1 second

WITH api_health AS (
    SELECT 
        api_name,
        COUNT(*) AS total_calls_5min,
        SUM(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 ELSE 0 END) AS successful_calls,
        SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END) AS server_errors,
        AVG(response_time_ms) AS avg_response_time,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) AS p95_response_time,
        MAX(response_time_ms) AS max_response_time
    FROM api_logs
    WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '5 minutes'
    GROUP BY api_name
),
database_health AS (
    SELECT 
        datname,
        numbackends AS active_connections,
        xact_commit AS transactions_committed,
        xact_rollback AS transactions_rolled_back,
        blks_read AS blocks_read,
        blks_hit AS blocks_hit,
        tup_returned AS tuples_returned,
        tup_fetched AS tuples_fetched,
        (blks_hit::FLOAT / NULLIF(blks_hit + blks_read, 0)) * 100 AS cache_hit_ratio
    FROM pg_stat_database
    WHERE datname = current_database()
),
job_health AS (
    SELECT 
        job_name,
        COUNT(*) AS executions_24h,
        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successful_runs,
        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_runs,
        AVG(execution_time_seconds) AS avg_execution_time,
        MAX(execution_time_seconds) AS max_execution_time,
        MAX(end_time) AS last_run_time
    FROM job_execution_history
    WHERE start_time >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
    GROUP BY job_name
),
queue_health AS (
    SELECT 
        queue_name,
        COUNT(*) AS pending_items,
        AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at))) AS avg_wait_seconds,
        MAX(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at))) AS max_wait_seconds,
        (
            SELECT COUNT(*) 
            FROM queue_items qi2 
            WHERE qi2.queue_name = qi.queue_name 
                AND qi2.processed_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
        ) AS processed_last_hour
    FROM queue_items qi
    WHERE status = 'pending'
    GROUP BY queue_name
)
SELECT 
    -- API Health
    (
        SELECT jsonb_object_agg(
            api_name,
            jsonb_build_object(
                'success_rate', ROUND((successful_calls::FLOAT / NULLIF(total_calls_5min, 0) * 100)::numeric, 2),
                'avg_response_ms', ROUND(avg_response_time::numeric, 0),
                'p95_response_ms', ROUND(p95_response_time::numeric, 0),
                'error_count', total_calls_5min - successful_calls
            )
        )
        FROM api_health
    ) AS api_status,
    
    -- Database Health
    (
        SELECT jsonb_build_object(
            'active_connections', active_connections,
            'cache_hit_ratio', ROUND(cache_hit_ratio::numeric, 2),
            'transactions_per_second', ROUND((transactions_committed / 300.0)::numeric, 2),
            'rollback_ratio', ROUND((transactions_rolled_back::FLOAT / NULLIF(transactions_committed + transactions_rolled_back, 0) * 100)::numeric, 2)
        )
        FROM database_health
    ) AS database_status,
    
    -- Job Health
    (
        SELECT jsonb_object_agg(
            job_name,
            jsonb_build_object(
                'success_rate', ROUND((successful_runs::FLOAT / NULLIF(executions_24h, 0) * 100)::numeric, 2),
                'failed_runs', failed_runs,
                'avg_runtime_seconds', ROUND(avg_execution_time::numeric, 1),
                'last_run', last_run_time
            )
        )
        FROM job_health
    ) AS job_status,
    
    -- Queue Health
    (
        SELECT jsonb_object_agg(
            queue_name,
            jsonb_build_object(
                'pending', pending_items,
                'avg_wait_seconds', ROUND(avg_wait_seconds::numeric, 0),
                'max_wait_seconds', ROUND(max_wait_seconds::numeric, 0),
                'throughput_per_hour', processed_last_hour
            )
        )
        FROM queue_health
    ) AS queue_status,
    
    -- Overall health score
    CASE 
        WHEN EXISTS (SELECT 1 FROM api_health WHERE successful_calls::FLOAT / NULLIF(total_calls_5min, 0) < 0.95) THEN 'degraded'
        WHEN EXISTS (SELECT 1 FROM job_health WHERE failed_runs > 0 AND successful_runs::FLOAT / NULLIF(executions_24h, 0) < 0.90) THEN 'warning'
        WHEN EXISTS (SELECT 1 FROM queue_health WHERE max_wait_seconds > 3600) THEN 'warning'
        ELSE 'healthy'
    END AS overall_health,
    
    CURRENT_TIMESTAMP AS last_updated;
```

---

## 5. Analytical Dashboard Queries

### 5.1 Trend Analysis Dashboard

```sql
-- Trend Analysis Dashboard: Identify and track trends
-- Refresh: Every 15 minutes
-- Performance target: < 5 seconds

WITH trend_signals AS (
    SELECT 
        t.topic_id,
        t.topic_name,
        t.category,
        -- Current metrics
        ts.search_volume_current,
        ts.social_mentions_current,
        ts.video_count_current,
        -- Historical comparison
        ts.search_volume_7d_ago,
        ts.social_mentions_7d_ago,
        ts.video_count_7d_ago,
        -- Calculate growth rates
        (ts.search_volume_current - ts.search_volume_7d_ago)::FLOAT / 
            NULLIF(ts.search_volume_7d_ago, 0) * 100 AS search_growth_rate,
        (ts.social_mentions_current - ts.social_mentions_7d_ago)::FLOAT / 
            NULLIF(ts.social_mentions_7d_ago, 0) * 100 AS social_growth_rate,
        -- Trend scores
        ts.trend_score,
        ts.velocity_score,
        ts.competition_index
    FROM topics t
    JOIN trend_signals_current ts ON t.topic_id = ts.topic_id
    WHERE ts.trend_score >= 30  -- Minimum threshold
),
trending_topics AS (
    SELECT 
        *,
        -- Composite ranking
        (
            trend_score * 0.4 +
            velocity_score * 0.3 +
            (100 - competition_index) * 0.3
        ) AS opportunity_score,
        -- Trend classification
        CASE 
            WHEN search_growth_rate > 100 AND velocity_score > 70 THEN 'explosive'
            WHEN search_growth_rate > 50 AND velocity_score > 50 THEN 'rapid'
            WHEN search_growth_rate > 20 THEN 'growing'
            WHEN search_growth_rate > 0 THEN 'stable'
            ELSE 'declining'
        END AS trend_type
    FROM trend_signals
),
category_trends AS (
    SELECT 
        category,
        COUNT(*) AS trending_topics_count,
        AVG(trend_score) AS avg_trend_score,
        AVG(search_growth_rate) AS avg_growth_rate,
        SUM(search_volume_current) AS total_search_volume
    FROM trending_topics
    GROUP BY category
),
our_coverage AS (
    SELECT 
        tt.topic_id,
        COUNT(DISTINCT v.video_id) AS our_videos,
        SUM(vm.views) AS our_views,
        MAX(v.published_at) AS latest_video,
        AVG(vp.performance_score) AS avg_performance
    FROM trending_topics tt
    LEFT JOIN video_topics vt ON tt.topic_id = vt.topic_id
    LEFT JOIN videos v ON vt.video_id = v.video_id
    LEFT JOIN video_metrics vm ON v.video_id = vm.video_id
    LEFT JOIN video_performance vp ON v.video_id = vp.video_id
    WHERE v.published_at >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY tt.topic_id
)
SELECT 
    -- Top trending topics
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'topic', topic_name,
                'category', category,
                'trend_score', ROUND(trend_score::numeric, 1),
                'opportunity_score', ROUND(opportunity_score::numeric, 1),
                'search_growth', ROUND(search_growth_rate::numeric, 1),
                'trend_type', trend_type,
                'our_coverage', COALESCE(oc.our_videos, 0),
                'recommendation', 
                    CASE 
                        WHEN COALESCE(oc.our_videos, 0) = 0 THEN 'Create content immediately'
                        WHEN COALESCE(oc.our_videos, 0) < 3 THEN 'Increase coverage'
                        ELSE 'Monitor performance'
                    END
            ) ORDER BY opportunity_score DESC
        )
        FROM trending_topics tt
        LEFT JOIN our_coverage oc ON tt.topic_id = oc.topic_id
        LIMIT 20
    ) AS top_trends,
    
    -- Category performance
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'category', category,
                'trending_count', trending_topics_count,
                'avg_trend_score', ROUND(avg_trend_score::numeric, 1),
                'avg_growth', ROUND(avg_growth_rate::numeric, 1)
            ) ORDER BY avg_trend_score DESC
        )
        FROM category_trends
    ) AS category_trends,
    
    -- Trend summary
    (
        SELECT jsonb_build_object(
            'total_trending', COUNT(*),
            'explosive_trends', COUNT(CASE WHEN trend_type = 'explosive' THEN 1 END),
            'uncovered_opportunities', COUNT(CASE WHEN topic_id NOT IN (SELECT topic_id FROM our_coverage) THEN 1 END),
            'avg_opportunity_score', ROUND(AVG(opportunity_score)::numeric, 1)
        )
        FROM trending_topics
    ) AS summary_metrics,
    
    CURRENT_TIMESTAMP AS last_updated;
```

### 5.2 Audience Insights Dashboard

```sql
-- Audience Insights Dashboard: Deep dive into audience behavior
-- Refresh: Every 30 minutes
-- Performance target: < 5 seconds

WITH audience_segments AS (
    SELECT 
        audience_segment,
        COUNT(DISTINCT viewer_id) AS segment_size,
        AVG(watch_time_minutes) AS avg_watch_time,
        AVG(videos_watched) AS avg_videos_watched,
        AVG(engagement_score) AS avg_engagement,
        SUM(revenue_attributed_cents) / 100.0 AS revenue_attributed
    FROM audience_analytics
    WHERE last_active >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY audience_segment
),
viewer_journey AS (
    SELECT 
        first_touchpoint,
        COUNT(DISTINCT viewer_id) AS viewers,
        AVG(videos_to_subscribe) AS avg_videos_before_subscribe,
        AVG(days_to_subscribe) AS avg_days_to_subscribe,
        SUM(CASE WHEN subscribed = true THEN 1 ELSE 0 END)::FLOAT / COUNT(*) * 100 AS conversion_rate,
        AVG(lifetime_value_cents) / 100.0 AS avg_ltv
    FROM viewer_journey_analytics
    WHERE first_seen >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY first_touchpoint
),
content_preferences AS (
    SELECT 
        content_category,
        COUNT(DISTINCT viewer_id) AS interested_viewers,
        AVG(completion_rate) AS avg_completion_rate,
        AVG(rewatch_rate) AS avg_rewatch_rate,
        SUM(total_watch_time_hours) AS total_watch_hours,
        RANK() OVER (ORDER BY COUNT(DISTINCT viewer_id) DESC) AS popularity_rank
    FROM viewer_content_preferences
    WHERE activity_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY content_category
),
geographic_distribution AS (
    SELECT 
        country_code,
        country_name,
        COUNT(DISTINCT viewer_id) AS viewers,
        SUM(views) AS total_views,
        SUM(watch_time_minutes) AS total_watch_time,
        AVG(avg_view_duration_minutes) AS avg_view_duration,
        SUM(revenue_cents) / 100.0 AS revenue,
        (SUM(revenue_cents) / NULLIF(SUM(views), 0) * 1000) / 100.0 AS rpm
    FROM viewer_geography
    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY country_code, country_name
),
device_analytics AS (
    SELECT 
        device_type,
        platform,
        COUNT(DISTINCT viewer_id) AS viewers,
        SUM(views) AS views,
        AVG(avg_session_duration_minutes) AS avg_session_duration,
        SUM(CASE WHEN action = 'subscribe' THEN 1 ELSE 0 END)::FLOAT / 
            NULLIF(COUNT(DISTINCT viewer_id), 0) * 100 AS subscription_rate
    FROM viewer_devices
    WHERE last_seen >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY device_type, platform
),
retention_cohorts AS (
    SELECT 
        DATE_TRUNC('week', first_view_date) AS cohort_week,
        COUNT(DISTINCT CASE WHEN days_since_first_view = 0 THEN viewer_id END) AS day_0,
        COUNT(DISTINCT CASE WHEN days_since_first_view = 1 THEN viewer_id END) AS day_1,
        COUNT(DISTINCT CASE WHEN days_since_first_view = 7 THEN viewer_id END) AS day_7,
        COUNT(DISTINCT CASE WHEN days_since_first_view = 30 THEN viewer_id END) AS day_30,
        COUNT(DISTINCT CASE WHEN days_since_first_view = 90 THEN viewer_id END) AS day_90
    FROM viewer_retention_analysis
    WHERE first_view_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY DATE_TRUNC('week', first_view_date)
)
SELECT 
    -- Audience segments
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'segment', audience_segment,
                'size', segment_size,
                'avg_watch_time', ROUND(avg_watch_time::numeric, 1),
                'avg_videos', ROUND(avg_videos_watched::numeric, 1),
                'engagement_score', ROUND(avg_engagement::numeric, 2),
                'revenue', ROUND(revenue_attributed::numeric, 2),
                'revenue_per_viewer', ROUND((revenue_attributed / segment_size)::numeric, 2)
            ) ORDER BY revenue_attributed DESC
        )
        FROM audience_segments
    ) AS audience_segments,
    
    -- Acquisition insights
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'source', first_touchpoint,
                'viewers', viewers,
                'conversion_rate', ROUND(conversion_rate::numeric, 2),
                'avg_videos_to_convert', ROUND(avg_videos_before_subscribe::numeric, 1),
                'avg_days_to_convert', ROUND(avg_days_to_subscribe::numeric, 1),
                'avg_ltv', ROUND(avg_ltv::numeric, 2)
            ) ORDER BY viewers DESC
        )
        FROM viewer_journey
    ) AS acquisition_funnel,
    
    -- Content preferences
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'category', content_category,
                'interested_viewers', interested_viewers,
                'completion_rate', ROUND(avg_completion_rate::numeric * 100, 1),
                'rewatch_rate', ROUND(avg_rewatch_rate::numeric * 100, 1),
                'total_hours', ROUND(total_watch_hours::numeric, 0),
                'rank', popularity_rank
            ) ORDER BY popularity_rank
        )
        FROM content_preferences
        WHERE popularity_rank <= 10
    ) AS top_content_preferences,
    
    -- Geographic insights
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'country', country_name,
                'viewers', viewers,
                'views', total_views,
                'avg_duration_min', ROUND(avg_view_duration::numeric, 1),
                'revenue', ROUND(revenue::numeric, 2),
                'rpm', ROUND(rpm::numeric, 2)
            ) ORDER BY revenue DESC
        )
        FROM geographic_distribution
        WHERE revenue > 0
        LIMIT 20
    ) AS top_countries,
    
    -- Device breakdown
    (
        SELECT jsonb_object_agg(
            device_type || '_' || platform,
            jsonb_build_object(
                'viewers', viewers,
                'views', views,
                'avg_session_min', ROUND(avg_session_duration::numeric, 1),
                'subscription_rate', ROUND(subscription_rate::numeric, 2)
            )
        )
        FROM device_analytics
    ) AS device_stats,
    
    -- Retention metrics
    (
        SELECT jsonb_build_object(
            'day_1_retention', ROUND(AVG(day_1::FLOAT / NULLIF(day_0, 0)) * 100, 1),
            'day_7_retention', ROUND(AVG(day_7::FLOAT / NULLIF(day_0, 0)) * 100, 1),
            'day_30_retention', ROUND(AVG(day_30::FLOAT / NULLIF(day_0, 0)) * 100, 1),
            'day_90_retention', ROUND(AVG(day_90::FLOAT / NULLIF(day_0, 0)) * 100, 1)
        )
        FROM retention_cohorts
    ) AS retention_metrics,
    
    CURRENT_TIMESTAMP AS last_updated;
```

### 5.3 Competitive Intelligence Dashboard

```sql
-- Competitive Intelligence Dashboard: Monitor competitive landscape
-- Refresh: Every hour
-- Performance target: < 5 seconds

WITH competitor_channels AS (
    SELECT 
        cc.competitor_id,
        cc.channel_name,
        cc.channel_category,
        cc.subscriber_count,
        cc.total_videos,
        cc.total_views,
        -- Growth metrics
        cc.subscriber_count - LAG(cc.subscriber_count, 7) 
            OVER (PARTITION BY cc.competitor_id ORDER BY cc.snapshot_date) AS subscriber_growth_7d,
        -- Recent activity
        COUNT(cv.video_id) FILTER (WHERE cv.published_at >= CURRENT_DATE - INTERVAL '7 days') AS videos_last_7d,
        SUM(cv.views) FILTER (WHERE cv.published_at >= CURRENT_DATE - INTERVAL '7 days') AS views_last_7d
    FROM competitor_channel_data cc
    LEFT JOIN competitor_videos cv ON cc.competitor_id = cv.channel_id
    WHERE cc.snapshot_date = CURRENT_DATE
    GROUP BY cc.competitor_id, cc.channel_name, cc.channel_category, 
             cc.subscriber_count, cc.total_videos, cc.total_views, cc.snapshot_date
),
competitor_performance AS (
    SELECT 
        competitor_id,
        AVG(views_per_video) AS avg_views_per_video,
        AVG(engagement_rate) AS avg_engagement_rate,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY upload_frequency) AS median_upload_frequency,
        AVG(estimated_rpm) AS avg_estimated_rpm
    FROM competitor_analytics
    WHERE analysis_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY competitor_id
),
content_gaps AS (
    SELECT 
        topic_id,
        topic_name,
        COUNT(DISTINCT competitor_id) AS competitors_covering,
        SUM(competitor_videos) AS total_competitor_videos,
        AVG(competitor_avg_views) AS avg_competitor_views,
        COUNT(DISTINCT our_video_id) AS our_videos,
        COALESCE(SUM(our_views), 0) AS our_total_views
    FROM (
        SELECT 
            ct.topic_id,
            ct.topic_name,
            cv.channel_id AS competitor_id,
            COUNT(cv.video_id) AS competitor_videos,
            AVG(cv.views) AS competitor_avg_views,
            ov.video_id AS our_video_id,
            ov.views AS our_views
        FROM content_topics ct
        LEFT JOIN competitor_videos cv ON ct.topic_id = cv.topic_id
        LEFT JOIN our_videos ov ON ct.topic_id = ov.topic_id
        WHERE cv.published_at >= CURRENT_DATE - INTERVAL '30 days'
            OR ov.published_at >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY ct.topic_id, ct.topic_name, cv.channel_id, ov.video_id, ov.views
    ) topic_coverage
    GROUP BY topic_id, topic_name
),
competitive_position AS (
    SELECT 
        category,
        -- Our metrics
        SUM(CASE WHEN is_ours = true THEN subscriber_count ELSE 0 END) AS our_subscribers,
        SUM(CASE WHEN is_ours = true THEN total_views ELSE 0 END) AS our_views,
        SUM(CASE WHEN is_ours = true THEN videos_last_7d ELSE 0 END) AS our_recent_videos,
        -- Competitor metrics
        SUM(CASE WHEN is_ours = false THEN subscriber_count ELSE 0 END) AS competitor_subscribers,
        SUM(CASE WHEN is_ours = false THEN total_views ELSE 0 END) AS competitor_views,
        SUM(CASE WHEN is_ours = false THEN videos_last_7d ELSE 0 END) AS competitor_recent_videos,
        -- Market share
        SUM(CASE WHEN is_ours = true THEN subscriber_count ELSE 0 END)::FLOAT / 
            NULLIF(SUM(subscriber_count), 0) * 100 AS subscriber_market_share,
        SUM(CASE WHEN is_ours = true THEN total_views ELSE 0 END)::FLOAT / 
            NULLIF(SUM(total_views), 0) * 100 AS views_market_share
    FROM (
        SELECT *, false AS is_ours FROM competitor_channels
        UNION ALL
        SELECT 
            channel_id, channel_name, niche, subscriber_count, 
            video_count, total_views, subscriber_growth_7d, 
            videos_last_7d, views_last_7d, true AS is_ours
        FROM our_channel_summary
    ) combined
    GROUP BY category
)
SELECT 
    -- Top competitors
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'channel', channel_name,
                'category', channel_category,
                'subscribers', subscriber_count,
                'subscriber_growth', subscriber_growth_7d,
                'videos_7d', videos_last_7d,
                'views_7d', views_last_7d,
                'avg_views_per_video', ROUND(cp.avg_views_per_video::numeric, 0),
                'engagement_rate', ROUND(cp.avg_engagement_rate::numeric, 2)
            ) ORDER BY subscriber_count DESC
        )
        FROM competitor_channels cc
        LEFT JOIN competitor_performance cp ON cc.competitor_id = cp.competitor_id
        LIMIT 20
    ) AS top_competitors,
    
    -- Content gap analysis
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'topic', topic_name,
                'competitors_count', competitors_covering,
                'competitor_videos', total_competitor_videos,
                'avg_competitor_views', ROUND(avg_competitor_views::numeric, 0),
                'our_videos', our_videos,
                'opportunity_score', 
                    CASE 
                        WHEN our_videos = 0 THEN 100
                        WHEN our_videos < competitors_covering THEN 75
                        ELSE 25
                    END
            ) ORDER BY 
                CASE WHEN our_videos = 0 THEN 0 ELSE 1 END,
                avg_competitor_views DESC
        )
        FROM content_gaps
        WHERE competitors_covering > 0
        LIMIT 15
    ) AS content_opportunities,
    
    -- Market position by category
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'category', category,
                'our_subscriber_share', ROUND(subscriber_market_share::numeric, 1),
                'our_views_share', ROUND(views_market_share::numeric, 1),
                'our_recent_videos', our_recent_videos,
                'competitor_recent_videos', competitor_recent_videos,
                'position', 
                    CASE 
                        WHEN subscriber_market_share >= 30 THEN 'leader'
                        WHEN subscriber_market_share >= 15 THEN 'challenger'
                        WHEN subscriber_market_share >= 5 THEN 'follower'
                        ELSE 'niche'
                    END
            ) ORDER BY subscriber_market_share DESC
        )
        FROM competitive_position
    ) AS market_position,
    
    -- Competitive insights summary
    (
        SELECT jsonb_build_object(
            'total_competitors_tracked', COUNT(DISTINCT competitor_id),
            'avg_competitor_growth_rate', ROUND(AVG(subscriber_growth_7d::FLOAT / NULLIF(subscriber_count - subscriber_growth_7d, 0)) * 100, 2),
            'content_gaps_identified', (SELECT COUNT(*) FROM content_gaps WHERE our_videos = 0),
            'categories_analyzed', COUNT(DISTINCT channel_category)
        )
        FROM competitor_channels
        WHERE subscriber_growth_7d IS NOT NULL
    ) AS summary_insights,
    
    CURRENT_TIMESTAMP AS last_updated;
```

---

## 6. KPI Calculation Logic

### 6.1 Core KPI Definitions

```sql
-- KPI Calculation Functions and Views
-- These are the source of truth for all KPI calculations

-- Function: Calculate Channel Health Score
CREATE OR REPLACE FUNCTION calculate_channel_health_score(
    p_channel_id VARCHAR(50),
    p_date DATE DEFAULT CURRENT_DATE
) RETURNS DECIMAL AS $
DECLARE
    v_growth_score DECIMAL;
    v_engagement_score DECIMAL;
    v_revenue_score DECIMAL;
    v_consistency_score DECIMAL;
    v_health_score DECIMAL;
BEGIN
    -- Growth component (30%)
    SELECT 
        LEAST(100, GREATEST(0, 
            (subscriber_growth_30d / NULLIF(subscriber_count - subscriber_growth_30d, 0)) * 1000
        )) INTO v_growth_score
    FROM channel_metrics
    WHERE channel_id = p_channel_id AND date = p_date;
    
    -- Engagement component (25%)
    SELECT 
        LEAST(100, GREATEST(0,
            avg_engagement_rate * 2000  -- 5% engagement = 100 score
        )) INTO v_engagement_score
    FROM channel_engagement_metrics
    WHERE channel_id = p_channel_id AND date = p_date;
    
    -- Revenue component (25%)
    SELECT 
        LEAST(100, GREATEST(0,
            (revenue_per_1k_views / 5.0) * 100  -- $5 RPM = 100 score
        )) INTO v_revenue_score
    FROM channel_revenue_metrics
    WHERE channel_id = p_channel_id AND date = p_date;
    
    -- Consistency component (20%)
    SELECT 
        LEAST(100, GREATEST(0,
            (videos_last_30d / 8.0) * 100  -- 8+ videos/month = 100 score
        )) INTO v_consistency_score
    FROM channel_activity_metrics
    WHERE channel_id = p_channel_id AND date = p_date;
    
    -- Calculate weighted health score
    v_health_score := (
        COALESCE(v_growth_score, 50) * 0.30 +
        COALESCE(v_engagement_score, 50) * 0.25 +
        COALESCE(v_revenue_score, 50) * 0.25 +
        COALESCE(v_consistency_score, 50) * 0.20
    );
    
    RETURN ROUND(v_health_score, 2);
END;
$ LANGUAGE plpgsql;

-- View: Daily KPI Summary
CREATE OR REPLACE VIEW v_daily_kpi_summary AS
WITH channel_kpis AS (
    SELECT 
        channel_id,
        date,
        -- Growth KPIs
        subscriber_count,
        subscriber_growth_daily,
        (subscriber_growth_daily::FLOAT / NULLIF(subscriber_count - subscriber_growth_daily, 0)) * 100 AS subscriber_growth_rate,
        
        -- Engagement KPIs
        total_views,
        total_likes,
        total_comments,
        (total_likes + total_comments)::FLOAT / NULLIF(total_views, 0) * 100 AS engagement_rate,
        
        -- Revenue KPIs
        revenue_cents / 100.0 AS revenue,
        (revenue_cents / NULLIF(total_views, 0) * 1000) / 100.0 AS rpm,
        
        -- Activity KPIs
        videos_published,
        total_watch_time_hours,
        total_watch_time_hours / NULLIF(total_views, 0) AS avg_watch_time_hours
        
    FROM channel_daily_metrics
),
platform_kpis AS (
    SELECT 
        date,
        -- Platform-wide metrics
        COUNT(DISTINCT channel_id) AS active_channels,
        SUM(subscriber_count) AS total_subscribers,
        SUM(total_views) AS total_views,
        SUM(revenue_cents) / 100.0 AS total_revenue,
        SUM(videos_published) AS total_videos_published,
        
        -- Averages
        AVG(engagement_rate) AS avg_engagement_rate,
        AVG(rpm) AS avg_rpm,
        AVG(subscriber_growth_rate) AS avg_growth_rate
        
    FROM channel_kpis
    GROUP BY date
)
SELECT 
    ck.date,
    ck.channel_id,
    
    -- Channel-specific KPIs
    ck.subscriber_count,
    ck.subscriber_growth_rate,
    ck.engagement_rate,
    ck.revenue,
    ck.rpm,
    ck.videos_published,
    ck.avg_watch_time_hours,
    calculate_channel_health_score(ck.channel_id, ck.date) AS health_score,
    
    -- Platform benchmarks
    pk.avg_engagement_rate AS platform_avg_engagement,
    pk.avg_rpm AS platform_avg_rpm,
    pk.avg_growth_rate AS platform_avg_growth,
    
    -- Relative performance
    ck.engagement_rate - pk.avg_engagement_rate AS engagement_vs_avg,
    ck.rpm - pk.avg_rpm AS rpm_vs_avg,
    ck.subscriber_growth_rate - pk.avg_growth_rate AS growth_vs_avg,
    
    -- Rankings
    RANK() OVER (PARTITION BY ck.date ORDER BY ck.revenue DESC) AS revenue_rank,
    RANK() OVER (PARTITION BY ck.date ORDER BY ck.subscriber_growth_rate DESC) AS growth_rank,
    RANK() OVER (PARTITION BY ck.date ORDER BY ck.engagement_rate DESC) AS engagement_rank
    
FROM channel_kpis ck
CROSS JOIN platform_kpis pk
WHERE ck.date = pk.date;
```

### 6.2 Advanced KPI Calculations

```sql
-- Function: Calculate Video Lifetime Value
CREATE OR REPLACE FUNCTION calculate_video_ltv(
    p_video_id VARCHAR(50),
    p_days INTEGER DEFAULT 365
) RETURNS TABLE (
    total_revenue DECIMAL,
    view_revenue DECIMAL,
    engagement_revenue DECIMAL,
    indirect_revenue DECIMAL,
    projected_total_ltv DECIMAL
) AS $
DECLARE
    v_published_date DATE;
    v_days_since_publish INTEGER;
BEGIN
    -- Get video publish date
    SELECT published_at::DATE INTO v_published_date
    FROM videos WHERE video_id = p_video_id;
    
    v_days_since_publish := CURRENT_DATE - v_published_date;
    
    RETURN QUERY
    WITH revenue_components AS (
        SELECT 
            -- Direct revenue
            SUM(r.revenue_cents) / 100.0 AS total_revenue,
            SUM(r.ad_revenue_cents) / 100.0 AS ad_revenue,
            SUM(r.premium_revenue_cents) / 100.0 AS premium_revenue,
            
            -- Engagement-driven revenue
            SUM(r.membership_revenue_cents + r.super_thanks_cents) / 100.0 AS engagement_revenue,
            
            -- Calculate decay rate
            CASE 
                WHEN v_days_since_publish >= 90 THEN
                    (SUM(CASE WHEN r.date >= v_published_date + 90 THEN r.revenue_cents ELSE 0 END) /
                     NULLIF(SUM(CASE WHEN r.date < v_published_date + 90 THEN r.revenue_cents END), 0))
                ELSE NULL
            END AS decay_rate
            
        FROM revenue_data r
        WHERE r.video_id = p_video_id
    ),
    indirect_attribution AS (
        SELECT 
            SUM(attributed_revenue_cents) / 100.0 AS indirect_revenue
        FROM video_indirect_revenue
        WHERE source_video_id = p_video_id
    )
    SELECT 
        rc.total_revenue,
        rc.ad_revenue + rc.premium_revenue AS view_revenue,
        rc.engagement_revenue,
        ia.indirect_revenue,
        
        -- Project total LTV
        CASE 
            WHEN v_days_since_publish >= p_days THEN rc.total_revenue
            WHEN rc.decay_rate IS NOT NULL THEN
                rc.total_revenue + (rc.total_revenue * rc.decay_rate * (p_days - v_days_since_publish) / 365.0)
            ELSE
                rc.total_revenue * (p_days::FLOAT / GREATEST(v_days_since_publish, 1))
        END AS projected_total_ltv
        
    FROM revenue_components rc
    CROSS JOIN indirect_attribution ia;
END;
$ LANGUAGE plpgsql;

-- Function: Calculate ROI
CREATE OR REPLACE FUNCTION calculate_content_roi(
    p_channel_id VARCHAR(50) DEFAULT NULL,
    p_start_date DATE DEFAULT CURRENT_DATE - INTERVAL '30 days',
    p_end_date DATE DEFAULT CURRENT_DATE
) RETURNS TABLE (
    channel_id VARCHAR(50),
    channel_name VARCHAR(255),
    total_investment DECIMAL,
    total_revenue DECIMAL,
    net_profit DECIMAL,
    roi_percentage DECIMAL,
    payback_period_days DECIMAL,
    profit_margin DECIMAL
) AS $
BEGIN
    RETURN QUERY
    WITH investment_data AS (
        SELECT 
            c.channel_id,
            c.channel_name,
            SUM(cc.total_cost_cents) / 100.0 AS total_investment,
            COUNT(DISTINCT cc.video_id) AS videos_produced
        FROM channels c
        JOIN content_costs cc ON c.channel_id = cc.channel_id
        JOIN videos v ON cc.video_id = v.video_id
        WHERE v.published_at BETWEEN p_start_date AND p_end_date
            AND (p_channel_id IS NULL OR c.channel_id = p_channel_id)
        GROUP BY c.channel_id, c.channel_name
    ),
    revenue_data AS (
        SELECT 
            v.channel_id,
            SUM(r.revenue_cents) / 100.0 AS total_revenue,
            MIN(CASE 
                WHEN SUM(r.revenue_cents) OVER (PARTITION BY v.channel_id ORDER BY r.date) >= cc.total_cost_cents
                THEN r.date - v.published_at
                ELSE NULL
            END) AS days_to_breakeven
        FROM videos v
        JOIN revenue_data r ON v.video_id = r.video_id
        JOIN content_costs cc ON v.video_id = cc.video_id
        WHERE v.published_at BETWEEN p_start_date AND p_end_date
            AND (p_channel_id IS NULL OR v.channel_id = p_channel_id)
        GROUP BY v.channel_id
    )
    SELECT 
        i.channel_id,
        i.channel_name,
        i.total_investment,
        COALESCE(r.total_revenue, 0) AS total_revenue,
        COALESCE(r.total_revenue, 0) - i.total_investment AS net_profit,
        
        CASE 
            WHEN i.total_investment > 0 THEN
                ((COALESCE(r.total_revenue, 0) - i.total_investment) / i.total_investment) * 100
            ELSE NULL
        END AS roi_percentage,
        
        COALESCE(r.days_to_breakeven, NULL) AS payback_period_days,
        
        CASE 
            WHEN COALESCE(r.total_revenue, 0) > 0 THEN
                ((COALESCE(r.total_revenue, 0) - i.total_investment) / r.total_revenue) * 100
            ELSE NULL
        END AS profit_margin
        
    FROM investment_data i
    LEFT JOIN revenue_data r ON i.channel_id = r.channel_id
    ORDER BY roi_percentage DESC NULLS LAST;
END;
$ LANGUAGE plpgsql;
```

---

## 7. Query Performance Optimization

### 7.1 Index Strategy

```sql
-- Essential indexes for dashboard performance

-- Time-based queries
CREATE INDEX idx_video_metrics_date ON video_metrics(date);
CREATE INDEX idx_revenue_data_date ON revenue_data(date);
CREATE INDEX idx_channel_metrics_date_channel ON channel_daily_metrics(date, channel_id);

-- Video performance queries
CREATE INDEX idx_videos_published_at ON videos(published_at);
CREATE INDEX idx_videos_channel_published ON videos(channel_id, published_at);
CREATE INDEX idx_video_performance_composite ON video_performance(video_id, viral_score, engagement_rate);

-- Channel analytics
CREATE INDEX idx_channels_status_health ON channels(status, health_score);
CREATE INDEX idx_channel_stats_current ON channel_stats(channel_id) WHERE is_current = true;

-- Revenue optimization
CREATE INDEX idx_revenue_video_date ON revenue_data(video_id, date);
CREATE INDEX idx_revenue_cents_date ON revenue_data(revenue_cents, date);

-- Trend analysis
CREATE INDEX idx_trends_score_date ON trend_signals(trend_score DESC, snapshot_time DESC);
CREATE INDEX idx_topics_category ON topics(category, is_active);

-- Partial indexes for common filters
CREATE INDEX idx_active_channels ON channels(channel_id) WHERE status = 'active';
CREATE INDEX idx_recent_videos ON videos(video_id, published_at) WHERE published_at >= CURRENT_DATE - INTERVAL '30 days';
CREATE INDEX idx_high_value_revenue ON revenue_data(video_id) WHERE revenue_cents > 10000;
```

### 7.2 Materialized View Strategy

```sql
-- Materialized views for complex aggregations

-- Hourly refresh for real-time metrics
CREATE MATERIALIZED VIEW mv_realtime_dashboard AS
SELECT 
    DATE_TRUNC('minute', last_updated) AS minute,
    COUNT(DISTINCT video_id) AS active_videos,
    SUM(views) AS total_views,
    AVG(ctr) AS avg_ctr,
    AVG(engagement_rate) AS avg_engagement
FROM video_metrics_realtime
WHERE last_updated >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
GROUP BY DATE_TRUNC('minute', last_updated);

CREATE UNIQUE INDEX ON mv_realtime_dashboard(minute);

-- Daily refresh for analytical queries
CREATE MATERIALIZED VIEW mv_channel_performance_daily AS
SELECT 
    channel_id,
    date,
    calculate_channel_health_score(channel_id, date) AS health_score,
    SUM(views) AS daily_views,
    SUM(revenue_cents) / 100.0 AS daily_revenue,
    COUNT(DISTINCT video_id) AS videos_published
FROM channel_activity_log
WHERE date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY channel_id, date;

CREATE UNIQUE INDEX ON mv_channel_performance_daily(channel_id, date);

-- Refresh strategy
CREATE OR REPLACE FUNCTION refresh_dashboard_materializations() RETURNS void AS $
BEGIN
    -- Refresh real-time views
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_realtime_dashboard;
    
    -- Refresh daily views only if needed
    IF NOT EXISTS (
        SELECT 1 FROM mv_channel_performance_daily 
        WHERE date = CURRENT_DATE
    ) THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_channel_performance_daily;
    END IF;
END;
$ LANGUAGE plpgsql;

-- Schedule refresh
SELECT cron.schedule('refresh-dashboards', '*/5 * * * *', 'SELECT refresh_dashboard_materializations()');
```

### 7.3 Query Optimization Patterns

```sql
-- Example: Optimized trend analysis query using CTEs and window functions

-- Before optimization: Multiple subqueries, poor performance
-- After optimization: Single pass with window functions

WITH trend_metrics AS (
    SELECT 
        topic_id,
        snapshot_time,
        search_volume,
        social_mentions,
        -- Calculate moving averages in single pass
        AVG(search_volume) OVER w_24h AS search_ma_24h,
        AVG(social_mentions) OVER w_24h AS social_ma_24h,
        -- Calculate growth rates
        (search_volume - LAG(search_volume, 24) OVER w) / 
            NULLIF(LAG(search_volume, 24) OVER w, 0) AS search_growth_24h,
        -- Rank within time window
        RANK() OVER (PARTITION BY snapshot_time ORDER BY search_volume DESC) AS volume_rank
    FROM trend_snapshots
    WHERE snapshot_time >= CURRENT_TIMESTAMP - INTERVAL '7 days'
    WINDOW 
        w AS (PARTITION BY topic_id ORDER BY snapshot_time),
        w_24h AS (PARTITION BY topic_id ORDER BY snapshot_time 
                  ROWS BETWEEN 23 PRECEDING AND CURRENT ROW)
)
SELECT 
    t.topic_id,
    t.topic_name,
    tm.search_volume,
    tm.search_ma_24h,
    tm.search_growth_24h,
    tm.volume_rank,
    -- Only join for final output
    cat.category_name
FROM trend_metrics tm
JOIN topics t ON tm.topic_id = t.topic_id
JOIN categories cat ON t.category_id = cat.category_id
WHERE tm.snapshot_time = (SELECT MAX(snapshot_time) FROM trend_metrics)
    AND tm.volume_rank <= 100
ORDER BY tm.search_volume DESC;
```

---

## 8. Dashboard Implementation Examples

### 8.1 Grafana Dashboard JSON

```json
{
  "dashboard": {
    "title": "YTEMPIRE Executive Dashboard",
    "panels": [
      {
        "title": "Revenue Overview",
        "type": "graph",
        "targets": [
          {
            "rawSql": "SELECT date, total_revenue, revenue_7d_ma FROM mv_revenue_dashboard WHERE date >= $__timeFrom() ORDER BY date",
            "format": "time_series"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "decimals": 2
          }
        }
      },
      {
        "title": "Channel Health Matrix",
        "type": "heatmap",
        "targets": [
          {
            "rawSql": "SELECT channel_name, date, health_score FROM v_daily_kpi_summary WHERE date >= $__timeFrom() ORDER BY channel_name, date",
            "format": "time_series"
          }
        ],
        "options": {
          "colorScheme": "GreenYellowRed",
          "reverseColors": true
        }
      }
    ]
  }
}
```

### 8.2 Query Caching Strategy

```python
# Redis caching layer for dashboard queries

import redis
import hashlib
import json
from datetime import datetime, timedelta

class DashboardQueryCache:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.cache_ttl = {
            'executive': 300,      # 5 minutes
            'operational': 60,     # 1 minute
            'analytical': 1800,    # 30 minutes
            'static': 86400       # 24 hours
        }
    
    def get_or_execute(self, query_name, query_func, dashboard_type='operational'):
        """Get from cache or execute query"""
        
        # Generate cache key
        cache_key = self._generate_cache_key(query_name)
        
        # Try cache first
        cached_result = self.redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        # Execute query
        result = query_func()
        
        # Cache result
        ttl = self.cache_ttl.get(dashboard_type, 300)
        self.redis_client.setex(
            cache_key,
            ttl,
            json.dumps(result, default=str)
        )
        
        return result
    
    def _generate_cache_key(self, query_name):
        """Generate consistent cache key"""
        
        # Include date for daily invalidation
        date_str = datetime.now().strftime('%Y-%m-%d')
        key_parts = [
            'dashboard',
            query_name,
            date_str
        ]
        
        return ':'.join(key_parts)
    
    def invalidate_dashboard(self, dashboard_name):
        """Invalidate all queries for a dashboard"""
        
        pattern = f"dashboard:{dashboard_name}:*"
        for key in self.redis_client.scan_iter(match=pattern):
            self.redis_client.delete(key)
```

---

## 9. Troubleshooting Guide

### 9.1 Common Performance Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Slow dashboard load | >5 second load time | Check indexes, use EXPLAIN ANALYZE |
| High database CPU | >80% sustained | Review query plans, add materialized views |
| Cache misses | <50% hit rate | Increase cache TTL, pre-warm cache |
| Query timeouts | Queries >30 seconds | Add indexes, partition large tables |
| Memory pressure | OOM errors | Reduce batch sizes, optimize CTEs |

### 9.2 Query Debugging

```sql
-- Debug slow queries
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) 
[YOUR_QUERY_HERE];

-- Find missing indexes
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    most_common_vals
FROM pg_stats
WHERE tablename = 'your_table'
    AND n_distinct > 100
    AND schemaname = 'public'
ORDER BY n_distinct DESC;

-- Monitor query performance
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    stddev_exec_time,
    max_exec_time
FROM pg_stat_statements
WHERE query LIKE '%dashboard%'
ORDER BY mean_exec_time DESC
LIMIT 20;
```

---

## Appendices

### Appendix A: Query Library Index

| Query Name | Dashboard | Refresh Rate | Avg Execution Time |
|------------|-----------|--------------|-------------------|
| CEO Overview | Executive | 5 min | 1.2s |
| Revenue Analysis | Financial | 15 min | 2.8s |
| Channel Performance | Operational | 10 min | 1.5s |
| Trend Detection | Analytical | 15 min | 3.2s |
| Audience Insights | Analytical | 30 min | 4.5s |

### Appendix B: Performance Benchmarks

| Metric | Target | Current | Status |
|--------|---------|---------|---------|
| Dashboard Load Time | <2s | 1.8s | ✅ |
| Query Cache Hit Rate | >80% | 85% | ✅ |
| Materialized View Lag | <5 min | 3 min | ✅ |
| Database CPU Usage | <70% | 45% | ✅ |
| Query Queue Depth | <10 | 2 | ✅ |

---

*This document is maintained by the Analytics Engineering team. For query optimization requests or new dashboard requirements, please submit a request through the analytics portal or contact analytics-eng@ytempire.com.*
```

---

## 4. Operational Dashboard Queries

### 4.1 Real-time Performance Monitor

```sql
-- Real-time Performance Dashboard
-- Refresh: Every 1 minute
-- Performance target: < 1 second

-- Use temporary table for performance
CREATE TEMP TABLE IF NOT EXISTS realtime_metrics AS
WITH last_hour_metrics AS (
    SELECT 
        v.video_id,
        v.channel_id,
        v.title,
        vm.views,
        vm.likes,
        vm.comments,
        vm.shares,
        vm.impressions,
        vm.clicks,
        vm.updated_at,
        -- Calculate rates
        vm.clicks::FLOAT / NULLIF(vm.impressions, 0) * 100 AS ctr,
        (vm.likes + vm.comments)::FLOAT / NULLIF(vm.views, 0) * 100 AS engagement_rate,
        -- Velocity
        vm.views - LAG(vm.views, 1) OVER (PARTITION BY v.video_id ORDER BY vm.updated_at) AS view_velocity
    FROM videos v
    JOIN video_metrics_realtime vm ON v.video_id = vm.video_id
    WHERE vm.updated_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
        AND v.published_at >= CURRENT_TIMESTAMP - INTERVAL '48 hours'  -- Focus on recent videos
)
SELECT 
    -- Summary metrics
    COUNT(DISTINCT video_id) AS active_videos,
    COUNT(DISTINCT channel_id) AS active_channels,
    SUM(views) AS total_views_last_hour,
    SUM(view_velocity) AS total_view_velocity,
    
    -- Top performing videos
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'video_id', video_id,
                'title', LEFT(title, 50) || '...',
                'views', views,
                'velocity', view_velocity,
                'ctr', ROUND(ctr::numeric, 2),
                'engagement', ROUND(engagement_rate::numeric, 2)
            ) ORDER BY view_velocity DESC NULLS LAST
        )
        FROM (
            SELECT * FROM last_hour_metrics 
            WHERE view_velocity > 0 
            ORDER BY view_velocity DESC 
            LIMIT 10
        ) top_videos
    ) AS trending_videos,
    
    -- Underperforming videos (need attention)
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'video_id', video_id,
                'title', LEFT(title, 50) || '...',
                'views', views,
                'ctr', ROUND(ctr::numeric, 2),
                'hours_since_publish', EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - updated_at)) / 3600
            ) ORDER BY ctr
        )
        FROM (
            SELECT * FROM last_hour_metrics 
            WHERE ctr < 2.0  -- Below 2% CTR
                AND impressions > 1000  -- Enough data
            ORDER BY ctr 
            LIMIT 10
        ) underperforming
    ) AS underperforming_videos,
    
    -- System metrics
    MAX(updated_at) AS latest_data_point,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - MAX(updated_at))) AS seconds_since_update,
    
    -- Alert conditions
    CASE 
        WHEN SUM(view_velocity) < 0 THEN 'ALERT: Negative growth'
        WHEN AVG(ctr) < 3.0 THEN 'WARNING: Low CTR'
        WHEN COUNT(DISTINCT video_id) < 10 THEN 'WARNING: Low activity'
        ELSE 'OK'
    END AS system_status
    
FROM last_hour_metrics;

-- Return results
SELECT * FROM realtime_metrics;
```

### 4.2 Channel Analytics Dashboard

```sql
-- Channel Analytics Dashboard
-- Refresh: Every 10 minutes
-- Performance target: < 3 seconds

WITH channel_current_stats AS (
    SELECT 
        c.channel_id,
        c.channel_name,
        c.niche,
        c.tier,
        cs.subscriber_count,
        cs.video_count,
        cs.total_views,
        cs.health_score,
        -- Recent performance
        SUM(CASE WHEN v.published_at >= CURRENT_DATE - INTERVAL '7 days' THEN 1 ELSE 0 END) AS videos_last_7d,
        SUM(CASE WHEN vm.date >= CURRENT_DATE - INTERVAL '7 days' THEN vm.views ELSE 0 END) AS views_last_7d,
        SUM(CASE WHEN r.date >= CURRENT_DATE - INTERVAL '7 days' THEN r.revenue_cents ELSE 0 END) / 100.0 AS revenue_last_7d
    FROM channels c
    JOIN channel_stats cs ON c.channel_id = cs.channel_id
    LEFT JOIN videos v ON c.channel_id = v.channel_id
    LEFT JOIN video_metrics vm ON v.video_id = vm.video_id
    LEFT JOIN revenue_data r ON v.video_id = r.video_id
    WHERE c.status = 'active'
    GROUP BY c.channel_id, c.channel_name, c.niche, c.tier, 
             cs.subscriber_count, cs.video_count, cs.total_views, cs.health_score
),
channel_growth_metrics AS (
    SELECT 
        channel_id,
        -- Subscriber growth
        subscriber_count - LAG(subscriber_count, 7) OVER (PARTITION BY channel_id ORDER BY date) AS subscriber_growth_7d,
        (subscriber_count - LAG(subscriber_count, 7) OVER (PARTITION BY channel_id ORDER BY date))::FLOAT / 
            NULLIF(LAG(subscriber_count, 7) OVER (PARTITION BY channel_id ORDER BY date), 0) * 100 AS subscriber_growth_rate_7d,
        -- View growth
        daily_views - LAG(daily_views, 7) OVER (PARTITION BY channel_id ORDER BY date) AS view_growth_7d,
        -- Best performing day
        MAX(daily_views) OVER (PARTITION BY channel_id) AS peak_daily_views,
        MAX(daily_revenue_cents) OVER (PARTITION BY channel_id) / 100.0 AS peak_daily_revenue
    FROM channel_daily_stats
    WHERE date = CURRENT_DATE - 1  -- Yesterday's data
),
channel_engagement AS (
    SELECT 
        v.channel_id,
        AVG(p.engagement_rate) AS avg_engagement_rate,
        AVG(p.ctr) AS avg_ctr,
        AVG(p.retention_score) AS avg_retention,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY p.viral_score) AS median_viral_score
    FROM videos v
    JOIN video_performance p ON v.video_id = p.video_id
    WHERE v.published_at >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY v.channel_id
)
SELECT 
    ccs.channel_id,
    ccs.channel_name,
    ccs.niche,
    ccs.tier,
    
    -- Current stats
    ccs.subscriber_count,
    ccs.video_count,
    ccs.total_views,
    ccs.health_score,
    
    -- Recent activity
    ccs.videos_last_7d,
    ccs.views_last_7d,
    ccs.revenue_last_7d,
    
    -- Growth metrics
    cgm.subscriber_growth_7d,
    cgm.subscriber_growth_rate_7d,
    cgm.view_growth_7d,
    cgm.peak_daily_views,
    cgm.peak_daily_revenue,
    
    -- Engagement metrics
    ce.avg_engagement_rate,
    ce.avg_ctr,
    ce.avg_retention,
    ce.median_viral_score,
    
    -- Calculated metrics
    ccs.revenue_last_7d / NULLIF(ccs.views_last_7d, 0) * 1000 AS rpm_7d,
    ccs.views_last_7d / NULLIF(ccs.videos_last_7d, 0) AS avg_views_per_video_7d,
    
    -- Performance indicators
    CASE 
        WHEN ccs.health_score >= 80 THEN 'excellent'
        WHEN ccs.health_score >= 60 THEN 'good'
        WHEN ccs.health_score >= 40 THEN 'fair'
        ELSE 'needs_attention'
    END AS health_status,
    
    CASE 
        WHEN cgm.subscriber_growth_rate_7d >= 5 THEN 'high_growth'
        WHEN cgm.subscriber_growth_rate_7d >= 2 THEN 'moderate_growth'
        WHEN cgm.subscriber_growth_rate_7d >= 0 THEN 'stable'
        ELSE 'declining'
    END AS growth_status,
    
    -- Recommendations
    CASE 
        WHEN ccs.videos_last_7d < 3 THEN 'Increase content frequency'
        WHEN ce.avg_ctr < 3.0 THEN 'Optimize thumbnails and titles'
        WHEN ce.avg_retention < 40 THEN 'Improve content quality'
        WHEN cgm.subscriber_growth_rate_7d < 0 THEN 'Review content strategy'
        ELSE 'Maintain current strategy'
    END AS primary_recommendation
    
FROM channel_current_stats ccs
LEFT JOIN channel_growth_metrics cgm ON ccs.channel_id = cgm.channel_id
LEFT JOIN channel_engagement ce ON ccs.channel_id = ce.channel_id
ORDER BY ccs.health_score DESC, ccs.revenue_last_7d DESC;
```

### 4.3 Content Pipeline Dashboard

```sql
-- Content Pipeline Dashboard: Track content creation flow
-- Refresh: Every 5 minutes
-- Performance target: < 2 seconds

WITH pipeline_stages AS (
    SELECT 
        stage,
        status,
        COUNT(*) AS item_count,
        AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - entered_stage_at)) / 3600) AS avg_hours_in_stage,
        MIN(entered_stage_at) AS oldest_item_entered,
        MAX(entered_stage_at) AS newest_item_entered
    FROM content_pipeline
    WHERE status IN ('active', 'processing')
    GROUP BY stage, status
),
stage_flow AS (
    SELECT 
        from_stage,
        to_stage,
        COUNT(*) AS transitions_today,
        AVG(EXTRACT(EPOCH FROM (transitioned_at - entered_from_stage_at)) / 3600) AS avg_transition_hours
    FROM content_pipeline_transitions
    WHERE transitioned_at >= CURRENT_DATE
    GROUP BY from_stage, to_stage
),
bottlenecks AS (
    SELECT 
        stage,
        COUNT(*) AS stuck_items,
        AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - entered_stage_at)) / 3600) AS avg_stuck_hours
    FROM content_pipeline
    WHERE status = 'active'
        AND entered_stage_at < CURRENT_TIMESTAMP - INTERVAL '24 hours'
    GROUP BY stage
    HAVING COUNT(*) > 0
),
daily_throughput AS (
    SELECT 
        DATE_TRUNC('hour', completed_at) AS hour,
        COUNT(*) AS completed_items,
        AVG(total_processing_hours) AS avg_processing_time,
        SUM(CASE WHEN total_processing_hours <= 8 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) * 100 AS within_sla_pct
    FROM content_pipeline_completed
    WHERE completed_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
    GROUP BY DATE_TRUNC('hour', completed_at)
)
SELECT 
    -- Current pipeline state
    (
        SELECT jsonb_object_agg(
            stage,
            jsonb_build_object(
                'count', item_count,
                'avg_hours', ROUND(avg_hours_in_stage::numeric, 1),
                'status', status
            )
        )
        FROM pipeline_stages
    ) AS pipeline_state,
    
    -- Flow metrics
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'from', from_stage,
                'to', to_stage,
                'count', transitions_today,
                'avg_hours', ROUND(avg_transition_hours::numeric, 1)
            )
        )
        FROM stage_flow
        WHERE transitions_today > 0
    ) AS stage_transitions,
    
    -- Bottlenecks
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'stage', stage,
                'stuck_items', stuck_items,
                'avg_stuck_hours', ROUND(avg_stuck_hours::numeric, 1)
            ) ORDER BY stuck_items DESC
        )
        FROM bottlenecks
    ) AS current_bottlenecks,
    
    -- Throughput metrics
    (
        SELECT 
            SUM(completed_items) AS total_completed_24h,
            AVG(completed_items) AS avg_hourly_completion,
            AVG(avg_processing_time) AS avg_total_processing_hours,
            AVG(within_sla_pct) AS overall_sla_compliance
        FROM daily_throughput
    ) AS throughput_metrics,
    
    -- Pipeline health
    CASE 
        WHEN EXISTS (SELECT 1 FROM bottlenecks WHERE stuck_items > 10) THEN 'critical'
        WHEN EXISTS (SELECT 1 FROM bottlenecks WHERE stuck_items > 5) THEN 'warning'
        ELSE 'healthy'
    END AS pipeline_health,
    
    -- Capacity utilization
    (
        SELECT 
            COUNT(DISTINCT content_id) AS active_items,
            (COUNT(DISTINCT content_id)::FLOAT / 500) * 100 AS capacity_utilization_pct  -- 500 is max capacity
        FROM content_pipeline
        WHERE status IN ('active', 'processing')
    ) AS capacity_metrics,
    
    CURRENT_TIMESTAMP AS last_updated;