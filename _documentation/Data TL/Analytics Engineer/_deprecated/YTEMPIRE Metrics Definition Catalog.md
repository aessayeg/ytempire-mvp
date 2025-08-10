# YTEMPIRE Metrics Definition Catalog

## Document Control
- **Version**: 1.0
- **Last Updated**: January 2025
- **Owner**: Analytics Engineering Team
- **Audience**: Analytics Engineers, Data Scientists, Product Managers

---

## 1. Executive Summary

This catalog provides comprehensive definitions for all metrics used across YTEMPIRE's analytics ecosystem. Each metric includes calculation logic, data sources, update frequency, and business context to ensure consistent implementation across all dashboards and reports.

---

## 2. Metric Categories

### 2.1 Channel Performance Metrics

#### 2.1.1 Channel Health Score (CHS)
```sql
-- Composite metric measuring overall channel performance
WITH channel_metrics AS (
    SELECT 
        channel_id,
        -- Growth component (40% weight)
        (subscriber_growth_rate_30d / 0.10) * 0.4 AS growth_score,
        
        -- Engagement component (30% weight)
        (avg_engagement_rate / 0.05) * 0.3 AS engagement_score,
        
        -- Revenue component (30% weight)
        (revenue_per_1k_views / 5.0) * 0.3 AS revenue_score
    FROM channel_analytics
    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
)
SELECT 
    channel_id,
    LEAST(100, (growth_score + engagement_score + revenue_score) * 100) AS channel_health_score
FROM channel_metrics;
```

**Business Context**: Measures overall channel vitality on 0-100 scale
**Update Frequency**: Daily at 02:00 UTC
**Data Sources**: `channel_analytics`, `revenue_data`
**Thresholds**: 
- Excellent: 80-100
- Good: 60-79
- Needs Attention: 40-59
- Critical: <40

#### 2.1.2 Subscriber Growth Rate
```sql
-- Month-over-month subscriber growth percentage
SELECT 
    channel_id,
    date,
    subscribers,
    LAG(subscribers, 30) OVER (PARTITION BY channel_id ORDER BY date) AS subscribers_30d_ago,
    CASE 
        WHEN LAG(subscribers, 30) OVER (PARTITION BY channel_id ORDER BY date) > 0
        THEN ((subscribers - LAG(subscribers, 30) OVER (PARTITION BY channel_id ORDER BY date))::FLOAT 
              / LAG(subscribers, 30) OVER (PARTITION BY channel_id ORDER BY date)) * 100
        ELSE 0
    END AS subscriber_growth_rate_30d
FROM channel_daily_stats;
```

**Business Context**: Key growth indicator for channel momentum
**Update Frequency**: Daily
**Data Sources**: YouTube Analytics API
**Industry Benchmark**: 3-5% monthly growth is healthy

#### 2.1.3 Channel Velocity Index (CVI)
```sql
-- Measures content production speed relative to performance
WITH channel_velocity AS (
    SELECT 
        c.channel_id,
        COUNT(DISTINCT v.video_id) AS videos_last_7d,
        AVG(v.views_first_24h) AS avg_24h_views,
        AVG(v.views_first_7d) AS avg_7d_views
    FROM channels c
    LEFT JOIN videos v ON c.channel_id = v.channel_id
    WHERE v.published_at >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY c.channel_id
)
SELECT 
    channel_id,
    videos_last_7d * LOG(avg_24h_views + 1) * 0.7 + 
    videos_last_7d * LOG(avg_7d_views + 1) * 0.3 AS channel_velocity_index
FROM channel_velocity;
```

**Business Context**: Balances quantity with quality of content production
**Update Frequency**: Hourly
**Data Sources**: `videos`, `video_performance`

### 2.2 Video Performance Metrics

#### 2.2.1 Video Success Score (VSS)
```sql
-- Comprehensive video performance metric
WITH video_metrics AS (
    SELECT 
        video_id,
        views,
        watch_time_minutes,
        likes,
        comments,
        shares,
        -- CTR component
        CASE 
            WHEN impressions > 0 THEN (clicks::FLOAT / impressions) 
            ELSE 0 
        END AS ctr,
        -- Retention component
        CASE 
            WHEN views > 0 THEN (watch_time_minutes * 60) / (duration_seconds * views)
            ELSE 0
        END AS avg_view_percentage
    FROM video_analytics
    WHERE date = CURRENT_DATE - 1
)
SELECT 
    video_id,
    (
        -- Views impact (30%)
        LEAST(LOG(views + 1) / LOG(1000000), 1) * 30 +
        
        -- Engagement impact (25%)
        LEAST(((likes + comments * 2 + shares * 3)::FLOAT / NULLIF(views, 0)) * 1000, 1) * 25 +
        
        -- CTR impact (25%)
        LEAST(ctr / 0.10, 1) * 25 +
        
        -- Retention impact (20%)
        LEAST(avg_view_percentage / 0.50, 1) * 20
    ) AS video_success_score
FROM video_metrics;
```

**Business Context**: Holistic measure of video performance (0-100)
**Update Frequency**: Daily
**Data Sources**: YouTube Analytics API
**Components**:
- Views (logarithmic scale): 30%
- Engagement rate: 25%
- Click-through rate: 25%
- Average view duration: 20%

#### 2.2.2 Viral Velocity Score (VVS)
```sql
-- Measures how quickly a video is gaining traction
WITH hourly_views AS (
    SELECT 
        video_id,
        published_at,
        hour_since_publish,
        views_cumulative,
        views_hourly,
        -- Calculate acceleration
        views_hourly - LAG(views_hourly, 1) OVER (
            PARTITION BY video_id ORDER BY hour_since_publish
        ) AS view_acceleration
    FROM video_hourly_stats
    WHERE hour_since_publish <= 48
)
SELECT 
    video_id,
    -- Velocity calculation focusing on first 48 hours
    SUM(
        CASE 
            WHEN hour_since_publish <= 24 THEN views_hourly * 2  -- Double weight for first 24h
            ELSE views_hourly 
        END
    ) / 1000 AS views_velocity,
    
    -- Acceleration bonus
    SUM(GREATEST(view_acceleration, 0)) / 100 AS acceleration_bonus,
    
    -- Combined score
    LOG(SUM(views_hourly) + 1) * 10 + 
    GREATEST(AVG(view_acceleration), 0) * 5 AS viral_velocity_score
    
FROM hourly_views
GROUP BY video_id;
```

**Business Context**: Early indicator of viral potential
**Update Frequency**: Every 15 minutes for videos <48 hours old
**Data Sources**: Real-time YouTube API polling
**Alert Threshold**: VVS > 75 triggers viral content protocol

#### 2.2.3 Revenue Per Mille (RPM)
```sql
-- Revenue per 1000 views
SELECT 
    video_id,
    date,
    (revenue_cents / GREATEST(views, 1)) * 1000 / 100 AS rpm_dollars,
    
    -- Breakdown by revenue source
    (ad_revenue_cents / GREATEST(views, 1)) * 1000 / 100 AS ad_rpm,
    (youtube_premium_cents / GREATEST(views, 1)) * 1000 / 100 AS premium_rpm,
    (super_thanks_cents / GREATEST(views, 1)) * 1000 / 100 AS super_thanks_rpm
    
FROM video_revenue_daily
WHERE views > 100;  -- Minimum threshold for meaningful RPM
```

**Business Context**: Core monetization efficiency metric
**Update Frequency**: Daily with 2-day lag (YouTube reporting delay)
**Data Sources**: YouTube Analytics API, AdSense API
**Industry Benchmark**: $3-5 RPM is average, >$8 is excellent

### 2.3 Content Optimization Metrics

#### 2.3.1 Thumbnail Click-Through Rate (CTR)
```sql
-- Measures thumbnail effectiveness
WITH thumbnail_performance AS (
    SELECT 
        v.video_id,
        v.thumbnail_version,
        t.thumbnail_url,
        t.thumbnail_type,
        SUM(i.impressions) AS total_impressions,
        SUM(i.clicks) AS total_clicks,
        -- Calculate CTR by thumbnail version
        CASE 
            WHEN SUM(i.impressions) > 100 
            THEN (SUM(i.clicks)::FLOAT / SUM(i.impressions)) * 100
            ELSE NULL 
        END AS ctr_percentage
    FROM videos v
    JOIN thumbnails t ON v.video_id = t.video_id
    JOIN impression_data i ON v.video_id = i.video_id
    WHERE i.date >= v.published_at
    GROUP BY v.video_id, v.thumbnail_version, t.thumbnail_url, t.thumbnail_type
)
SELECT 
    video_id,
    thumbnail_version,
    ctr_percentage,
    -- Performance compared to channel average
    ctr_percentage - AVG(ctr_percentage) OVER (
        PARTITION BY thumbnail_type
    ) AS ctr_vs_average,
    -- Rank within thumbnail type
    RANK() OVER (
        PARTITION BY thumbnail_type 
        ORDER BY ctr_percentage DESC
    ) AS ctr_rank
FROM thumbnail_performance
WHERE total_impressions >= 1000;  -- Statistical significance threshold
```

**Business Context**: Primary metric for thumbnail optimization
**Update Frequency**: Every 6 hours
**Data Sources**: YouTube Analytics API (Traffic Sources report)
**A/B Test Threshold**: Minimum 1000 impressions per variant

#### 2.3.2 Title Optimization Score (TOS)
```sql
-- Analyzes title effectiveness across multiple dimensions
WITH title_analysis AS (
    SELECT 
        v.video_id,
        v.title,
        v.title_version,
        -- Length score (optimal: 50-60 characters)
        CASE 
            WHEN LENGTH(v.title) BETWEEN 50 AND 60 THEN 100
            WHEN LENGTH(v.title) BETWEEN 40 AND 70 THEN 80
            ELSE 60
        END AS length_score,
        
        -- Keyword score
        (
            SELECT COUNT(DISTINCT k.keyword)
            FROM trending_keywords k
            WHERE v.title ILIKE '%' || k.keyword || '%'
        ) * 10 AS keyword_score,
        
        -- Emotional trigger score
        CASE 
            WHEN v.title ~* '(amazing|incredible|shocking|unbelievable)' THEN 20
            WHEN v.title ~* '(how to|tutorial|guide|tips)' THEN 15
            ELSE 10
        END AS emotion_score,
        
        -- Performance metrics
        p.ctr AS actual_ctr,
        p.avg_view_duration AS actual_retention
        
    FROM videos v
    JOIN video_performance p ON v.video_id = p.video_id
)
SELECT 
    video_id,
    title,
    -- Composite title score
    (length_score * 0.2 + 
     LEAST(keyword_score, 100) * 0.3 + 
     emotion_score * 0.2 + 
     (actual_ctr / 0.10) * 100 * 0.3) AS title_optimization_score,
     
    -- Individual components for optimization
    length_score,
    keyword_score,
    emotion_score,
    actual_ctr * 100 AS ctr_percentage
    
FROM title_analysis;
```

**Business Context**: Guides title optimization decisions
**Update Frequency**: Daily
**Data Sources**: `videos`, `video_performance`, `trending_keywords`

### 2.4 Audience Engagement Metrics

#### 2.4.1 Audience Retention Score (ARS)
```sql
-- Measures ability to keep viewers watching
WITH retention_curve AS (
    SELECT 
        video_id,
        retention_point,  -- Percentage through video (0-100)
        audience_percentage,  -- Percentage still watching
        -- Calculate area under retention curve
        (retention_point - LAG(retention_point, 1) OVER (
            PARTITION BY video_id ORDER BY retention_point
        )) * audience_percentage AS area_segment
    FROM video_retention_data
)
SELECT 
    video_id,
    -- Average retention (simple)
    AVG(audience_percentage) AS avg_retention_percentage,
    
    -- Area under curve (more sophisticated)
    SUM(area_segment) / 100 AS retention_area_score,
    
    -- Key retention points
    MAX(CASE WHEN retention_point = 30 THEN audience_percentage END) AS retention_at_30_percent,
    MAX(CASE WHEN retention_point = 50 THEN audience_percentage END) AS retention_at_50_percent,
    MAX(CASE WHEN retention_point = 70 THEN audience_percentage END) AS retention_at_70_percent,
    
    -- Drop-off rate in first 30 seconds
    100 - MAX(CASE WHEN retention_point <= 10 THEN audience_percentage END) AS early_drop_off_rate
    
FROM retention_curve
GROUP BY video_id;
```

**Business Context**: Critical for YouTube algorithm favorability
**Update Frequency**: Daily
**Data Sources**: YouTube Analytics API (Audience Retention report)
**Target**: >50% average retention for videos <10 minutes

#### 2.4.2 Engagement Velocity Rate (EVR)
```sql
-- Measures how quickly engagement accumulates
WITH engagement_timeline AS (
    SELECT 
        video_id,
        hours_since_publish,
        SUM(likes) OVER (PARTITION BY video_id ORDER BY hours_since_publish) AS cumulative_likes,
        SUM(comments) OVER (PARTITION BY video_id ORDER BY hours_since_publish) AS cumulative_comments,
        SUM(shares) OVER (PARTITION BY video_id ORDER BY hours_since_publish) AS cumulative_shares,
        views
    FROM video_hourly_engagement
    WHERE hours_since_publish <= 72
)
SELECT 
    video_id,
    -- Engagement per 1000 views at different time points
    (cumulative_likes::FLOAT / GREATEST(views, 1)) * 1000 AS likes_per_1k_views_72h,
    (cumulative_comments::FLOAT / GREATEST(views, 1)) * 1000 AS comments_per_1k_views_72h,
    
    -- Velocity score (weighted by recency)
    SUM(
        CASE 
            WHEN hours_since_publish <= 24 THEN 
                ((cumulative_likes + cumulative_comments * 2 + cumulative_shares * 3)::FLOAT / GREATEST(views, 1)) * 3
            WHEN hours_since_publish <= 48 THEN 
                ((cumulative_likes + cumulative_comments * 2 + cumulative_shares * 3)::FLOAT / GREATEST(views, 1)) * 2
            ELSE 
                ((cumulative_likes + cumulative_comments * 2 + cumulative_shares * 3)::FLOAT / GREATEST(views, 1))
        END
    ) * 1000 AS engagement_velocity_rate
    
FROM engagement_timeline
GROUP BY video_id;
```

**Business Context**: Predicts long-term engagement potential
**Update Frequency**: Hourly for videos <72 hours old
**Data Sources**: Real-time engagement tracking

### 2.5 Financial Metrics

#### 2.5.1 Customer Acquisition Cost (CAC) by Channel
```sql
-- Calculates cost to acquire monetizable channel audience
WITH channel_costs AS (
    SELECT 
        channel_id,
        date_trunc('month', date) AS month,
        SUM(production_cost_cents) / 100 AS production_costs,
        SUM(promotion_cost_cents) / 100 AS promotion_costs,
        SUM(overhead_allocation_cents) / 100 AS overhead_costs
    FROM channel_expenses
    GROUP BY channel_id, date_trunc('month', date)
),
channel_growth AS (
    SELECT 
        channel_id,
        date_trunc('month', date) AS month,
        MAX(subscribers) - MIN(subscribers) AS new_subscribers,
        MAX(members) - MIN(members) AS new_members
    FROM channel_daily_stats
    GROUP BY channel_id, date_trunc('month', date)
)
SELECT 
    c.channel_id,
    c.month,
    (c.production_costs + c.promotion_costs + c.overhead_costs) AS total_costs,
    g.new_subscribers,
    g.new_members,
    -- CAC per subscriber
    CASE 
        WHEN g.new_subscribers > 0 
        THEN (c.production_costs + c.promotion_costs + c.overhead_costs) / g.new_subscribers
        ELSE NULL 
    END AS cac_per_subscriber,
    -- CAC per paying member
    CASE 
        WHEN g.new_members > 0 
        THEN (c.production_costs + c.promotion_costs + c.overhead_costs) / g.new_members
        ELSE NULL 
    END AS cac_per_member
FROM channel_costs c
JOIN channel_growth g ON c.channel_id = g.channel_id AND c.month = g.month;
```

**Business Context**: Efficiency metric for channel investment
**Update Frequency**: Monthly
**Data Sources**: `channel_expenses`, `channel_daily_stats`
**Target**: CAC < 3-month subscriber LTV

#### 2.5.2 Return on Content Investment (ROCI)
```sql
-- Measures profitability of content creation
WITH video_investment AS (
    SELECT 
        v.video_id,
        v.channel_id,
        v.published_at,
        -- Total investment
        c.script_cost_cents / 100 AS script_cost,
        c.voice_cost_cents / 100 AS voice_cost,
        c.video_production_cents / 100 AS production_cost,
        c.thumbnail_cost_cents / 100 AS thumbnail_cost,
        (c.script_cost_cents + c.voice_cost_cents + 
         c.video_production_cents + c.thumbnail_cost_cents) / 100 AS total_cost
    FROM videos v
    JOIN content_costs c ON v.video_id = c.video_id
),
video_returns AS (
    SELECT 
        video_id,
        SUM(revenue_cents) / 100 AS total_revenue,
        SUM(CASE WHEN date <= published_at + INTERVAL '30 days' 
            THEN revenue_cents ELSE 0 END) / 100 AS revenue_30d,
        SUM(CASE WHEN date <= published_at + INTERVAL '90 days' 
            THEN revenue_cents ELSE 0 END) / 100 AS revenue_90d
    FROM video_revenue_daily
    GROUP BY video_id
)
SELECT 
    i.video_id,
    i.channel_id,
    i.total_cost,
    r.total_revenue,
    r.revenue_30d,
    r.revenue_90d,
    -- ROCI calculations
    (r.total_revenue - i.total_cost) AS net_profit,
    CASE 
        WHEN i.total_cost > 0 
        THEN ((r.total_revenue - i.total_cost) / i.total_cost) * 100
        ELSE 0 
    END AS roci_percentage,
    -- Time to break even
    CASE 
        WHEN r.revenue_30d >= i.total_cost THEN '< 30 days'
        WHEN r.revenue_90d >= i.total_cost THEN '30-90 days'
        WHEN r.total_revenue >= i.total_cost THEN '> 90 days'
        ELSE 'Not profitable'
    END AS break_even_period
FROM video_investment i
LEFT JOIN video_returns r ON i.video_id = r.video_id;
```

**Business Context**: Core profitability metric
**Update Frequency**: Daily
**Data Sources**: `content_costs`, `video_revenue_daily`
**Target**: ROCI > 200% within 90 days

### 2.6 Operational Metrics

#### 2.6.1 Content Production Velocity (CPV)
```sql
-- Measures content pipeline efficiency
WITH production_pipeline AS (
    SELECT 
        date_trunc('day', created_at) AS date,
        COUNT(*) FILTER (WHERE status = 'ideation') AS ideas_created,
        COUNT(*) FILTER (WHERE status = 'script_complete') AS scripts_completed,
        COUNT(*) FILTER (WHERE status = 'video_rendered') AS videos_rendered,
        COUNT(*) FILTER (WHERE status = 'published') AS videos_published,
        AVG(EXTRACT(EPOCH FROM (published_at - created_at))/3600) 
            FILTER (WHERE status = 'published') AS avg_production_hours
    FROM content_pipeline
    GROUP BY date_trunc('day', created_at)
)
SELECT 
    date,
    ideas_created,
    scripts_completed,
    videos_rendered,
    videos_published,
    avg_production_hours,
    -- Conversion rates
    CASE 
        WHEN ideas_created > 0 
        THEN (videos_published::FLOAT / ideas_created) * 100 
        ELSE 0 
    END AS idea_to_publish_rate,
    -- Daily velocity score
    videos_published * (24.0 / GREATEST(avg_production_hours, 1)) AS production_velocity_score
FROM production_pipeline;
```

**Business Context**: Operational efficiency indicator
**Update Frequency**: Real-time
**Data Sources**: `content_pipeline` internal tracking
**Target**: 15 videos/day per channel minimum

#### 2.6.2 AI Model Performance Index (MPI)
```sql
-- Tracks AI component effectiveness
WITH model_metrics AS (
    SELECT 
        model_name,
        model_version,
        date_trunc('hour', inference_time) AS hour,
        COUNT(*) AS total_inferences,
        AVG(inference_latency_ms) AS avg_latency,
        AVG(quality_score) AS avg_quality,
        SUM(CASE WHEN error_flag = true THEN 1 ELSE 0 END) AS error_count,
        AVG(cost_per_inference_cents) / 100 AS avg_cost_dollars
    FROM ml_model_inferences
    WHERE inference_time >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
    GROUP BY model_name, model_version, date_trunc('hour', inference_time)
)
SELECT 
    model_name,
    model_version,
    -- Performance index calculation
    (
        -- Quality component (40%)
        (avg_quality / 100) * 40 +
        
        -- Speed component (30%) - inverse of latency
        LEAST((1000.0 / GREATEST(avg_latency, 1)), 1) * 30 +
        
        -- Reliability component (20%)
        ((total_inferences - error_count)::FLOAT / total_inferences) * 20 +
        
        -- Cost efficiency (10%) - inverse of cost
        LEAST((1.0 / GREATEST(avg_cost_dollars, 0.01)), 1) * 10
    ) AS model_performance_index,
    
    -- Individual metrics
    avg_latency AS avg_latency_ms,
    avg_quality AS avg_quality_score,
    (error_count::FLOAT / total_inferences) * 100 AS error_rate_percentage,
    avg_cost_dollars
    
FROM model_metrics
ORDER BY model_performance_index DESC;
```

**Business Context**: Guides AI optimization decisions
**Update Frequency**: Hourly
**Data Sources**: ML platform telemetry
**Alert Threshold**: MPI < 70 triggers investigation

---

## 3. Metric Calculation Guidelines

### 3.1 Data Quality Requirements
- All metrics must handle NULL values gracefully
- Use GREATEST(value, 1) to prevent division by zero
- Apply statistical significance thresholds
- Document edge cases in metric definitions

### 3.2 Performance Optimization
```sql
-- Example: Pre-aggregated materialized view for common metrics
CREATE MATERIALIZED VIEW mv_daily_channel_metrics AS
SELECT 
    channel_id,
    date,
    -- Pre-calculate expensive metrics
    SUM(views) AS total_views,
    AVG(ctr) AS avg_ctr,
    SUM(revenue_cents) / 100 AS total_revenue,
    COUNT(DISTINCT video_id) AS videos_published,
    -- Pre-calculate growth rates
    LAG(SUM(views), 7) OVER (PARTITION BY channel_id ORDER BY date) AS views_7d_ago,
    LAG(SUM(revenue_cents), 30) OVER (PARTITION BY channel_id ORDER BY date) AS revenue_30d_ago
FROM video_daily_stats
GROUP BY channel_id, date;

-- Refresh strategy
CREATE INDEX idx_mv_channel_date ON mv_daily_channel_metrics(channel_id, date);
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_channel_metrics;
```

### 3.3 Metric Versioning
```sql
-- Track metric definition changes
CREATE TABLE metric_definitions (
    metric_name VARCHAR(100),
    version INTEGER,
    definition_sql TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    change_reason TEXT,
    is_active BOOLEAN DEFAULT true,
    PRIMARY KEY (metric_name, version)
);

-- Example version tracking
INSERT INTO metric_definitions (metric_name, version, definition_sql, created_by, change_reason)
VALUES (
    'channel_health_score',
    2,
    '-- New weighted calculation...',
    'analytics_team',
    'Added revenue component to health score calculation'
);
```

---

## 4. Metric Implementation Standards

### 4.1 Naming Conventions
- Use snake_case for all metric names
- Suffix with unit of measure (_rate, _percentage, _score, _index)
- Prefix with domain (channel_, video_, audience_)
- Be descriptive but concise

### 4.2 Documentation Requirements
Each metric must include:
1. Business definition and context
2. SQL calculation logic
3. Data sources and dependencies
4. Update frequency and latency
5. Historical tracking requirements
6. Alert thresholds if applicable

### 4.3 Testing Framework
```python
# Example metric validation test
def test_channel_health_score():
    """Validate channel health score calculation"""
    
    # Test case 1: Perfect channel
    test_data = {
        'subscriber_growth_rate_30d': 0.10,  # 10%
        'avg_engagement_rate': 0.05,  # 5%
        'revenue_per_1k_views': 5.0  # $5 RPM
    }
    expected_score = 100.0
    
    actual_score = calculate_channel_health_score(test_data)
    assert abs(actual_score - expected_score) < 0.01
    
    # Test case 2: Edge cases
    test_edge_cases = {
        'negative_growth': {'subscriber_growth_rate_30d': -0.05, ...},
        'zero_engagement': {'avg_engagement_rate': 0, ...},
        'no_revenue': {'revenue_per_1k_views': 0, ...}
    }
    
    for case_name, test_data in test_edge_cases.items():
        score = calculate_channel_health_score(test_data)
        assert 0 <= score <= 100, f"Score out of bounds for {case_name}"
```

---

## 5. Advanced Composite Metrics

### 5.1 Content-Market Fit Score (CMFS)
```sql
-- Measures how well content matches market demand
WITH content_market_analysis AS (
    SELECT 
        v.video_id,
        v.channel_id,
        v.primary_topic,
        -- Market demand indicators
        t.search_volume AS topic_search_volume,
        t.competition_index AS topic_competition,
        t.trend_momentum AS topic_momentum,
        -- Content performance
        p.views_velocity_24h,
        p.ctr AS click_through_rate,
        p.retention_score
    FROM videos v
    JOIN trending_topics t ON v.primary_topic = t.topic_id
    JOIN video_performance p ON v.video_id = p.video_id
)
SELECT 
    video_id,
    channel_id,
    -- Demand-Supply fit (40%)
    (topic_search_volume / GREATEST(topic_competition, 1)) * 0.4 AS demand_supply_score,
    
    -- Performance validation (40%)
    ((views_velocity_24h / 1000) * click_through_rate * retention_score) * 0.4 AS performance_score,
    
    -- Timing bonus (20%)
    (topic_momentum / 100) * 0.2 AS timing_score,
    
    -- Combined CMFS
    (
        (topic_search_volume / GREATEST(topic_competition, 1)) * 0.4 +
        ((views_velocity_24h / 1000) * click_through_rate * retention_score) * 0.4 +
        (topic_momentum / 100) * 0.2
    ) * 100 AS content_market_fit_score
    
FROM content_market_analysis;
```

### 5.2 Channel Diversification Index (CDI)
```sql
-- Measures revenue stream and content diversification
WITH channel_diversity AS (
    SELECT 
        channel_id,
        -- Revenue diversification (Herfindahl index)
        1 - (
            POWER(ad_revenue_share, 2) + 
            POWER(membership_revenue_share, 2) + 
            POWER(merchandise_revenue_share, 2) +
            POWER(sponsorship_revenue_share, 2)
        ) AS revenue_diversity_score,
        
        -- Content topic diversification
        1 - SUM(POWER(topic_share, 2)) AS content_diversity_score,
        
        -- Audience geo diversification  
        1 - SUM(POWER(country_view_share, 2)) AS geo_diversity_score
        
    FROM channel_diversification_stats
)
SELECT 
    channel_id,
    -- Weighted diversification index
    (revenue_diversity_score * 0.4 + 
     content_diversity_score * 0.3 + 
     geo_diversity_score * 0.3) * 100 AS channel_diversification_index,
     
    -- Risk assessment
    CASE 
        WHEN revenue_diversity_score < 0.3 THEN 'High Risk - Single Revenue Source'
        WHEN content_diversity_score < 0.3 THEN 'High Risk - Single Content Type'
        WHEN geo_diversity_score < 0.3 THEN 'High Risk - Single Market'
        ELSE 'Diversified'
    END AS risk_assessment
    
FROM channel_diversity;
```

---

## 6. Real-time Metrics Implementation

### 6.1 Streaming Metrics Architecture
```python
# Kafka streaming metric processor
from kafka import KafkaConsumer
from prometheus_client import Gauge, Counter, Histogram
import json

class RealTimeMetricsProcessor:
    def __init__(self):
        self.consumer = KafkaConsumer(
            'video-events',
            'engagement-events',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        # Prometheus metrics
        self.views_counter = Counter('ytempire_views_total', 'Total views')
        self.engagement_gauge = Gauge('ytempire_engagement_rate', 'Current engagement rate')
        self.latency_histogram = Histogram('ytempire_processing_latency', 'Processing latency')
        
    def process_metrics(self):
        for message in self.consumer:
            event = message.value
            
            if event['type'] == 'view':
                self.views_counter.inc()
                self.update_velocity_metrics(event)
                
            elif event['type'] == 'engagement':
                self.update_engagement_metrics(event)
                
            # Update derived metrics
            self.calculate_real_time_scores()
```

### 6.2 Metric Alerting Configuration
```yaml
# Prometheus alerting rules
groups:
  - name: ytempire_metrics
    interval: 30s
    rules:
      - alert: LowChannelHealthScore
        expr: channel_health_score < 40
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Channel {{ $labels.channel_id }} health score below threshold"
          
      - alert: HighProductionCost
        expr: avg_cost_per_video > 0.50
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Production costs exceeding target"
          
      - alert: DecliningEngagement
        expr: rate(engagement_rate[1h]) < -0.1
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Engagement declining rapidly"
```

---

## 7. Metric Usage Guidelines

### 7.1 Dashboard Integration
```sql
-- Example: Executive dashboard query
WITH current_metrics AS (
    SELECT 
        COUNT(DISTINCT channel_id) AS active_channels,
        SUM(videos_published_today) AS total_videos_today,
        AVG(channel_health_score) AS avg_health_score,
        SUM(revenue_today) AS total_revenue_today
    FROM mv_daily_channel_metrics
    WHERE date = CURRENT_DATE
),
trends AS (
    SELECT 
        (SUM(CASE WHEN date = CURRENT_DATE THEN revenue_cents END) - 
         SUM(CASE WHEN date = CURRENT_DATE - 7 THEN revenue_cents END)) / 
         NULLIF(SUM(CASE WHEN date = CURRENT_DATE - 7 THEN revenue_cents END), 0) * 100 
         AS revenue_wow_growth
    FROM video_revenue_daily
    WHERE date IN (CURRENT_DATE, CURRENT_DATE - 7)
)
SELECT 
    c.*,
    t.revenue_wow_growth,
    -- Traffic light status
    CASE 
        WHEN c.avg_health_score >= 80 THEN 'green'
        WHEN c.avg_health_score >= 60 THEN 'yellow'
        ELSE 'red'
    END AS overall_status
FROM current_metrics c
CROSS JOIN trends t;
```

### 7.2 Automated Reporting
```python
# Daily metric report generator
def generate_daily_metrics_report():
    """Generate automated daily metrics summary"""
    
    metrics = {
        'date': datetime.now().date(),
        'summary': {},
        'alerts': [],
        'recommendations': []
    }
    
    # Fetch key metrics
    metrics['summary'] = fetch_executive_metrics()
    
    # Check for anomalies
    anomalies = detect_metric_anomalies()
    if anomalies:
        metrics['alerts'].extend(format_anomaly_alerts(anomalies))
    
    # Generate recommendations
    if metrics['summary']['avg_ctr'] < 0.04:
        metrics['recommendations'].append({
            'type': 'thumbnail_optimization',
            'priority': 'high',
            'action': 'Review thumbnail designs for underperforming videos'
        })
    
    return generate_report_html(metrics)
```

---

## 8. Metric Maintenance Procedures

### 8.1 Monthly Metric Review Checklist
- [ ] Validate all metric calculations against source data
- [ ] Review metric usage in dashboards and reports
- [ ] Update thresholds based on performance trends
- [ ] Document any formula adjustments
- [ ] Backfill historical data if calculations changed
- [ ] Update metric documentation
- [ ] Communicate changes to stakeholders

### 8.2 Metric Deprecation Process
1. Identify metrics with <5% usage in last 30 days
2. Assess impact on downstream dependencies
3. Provide 30-day deprecation notice
4. Migrate dependent systems to alternative metrics
5. Archive metric definition and historical data
6. Remove from active calculations

---

## 9. Appendices

### Appendix A: Metric Quick Reference
| Metric | Category | Update Frequency | Primary Use Case |
|--------|----------|------------------|------------------|
| Channel Health Score | Performance | Daily | Executive Dashboard |
| Video Success Score | Performance | Daily | Content Optimization |
| Viral Velocity Score | Performance | Real-time | Trend Detection |
| Revenue Per Mille | Financial | Daily | Monetization Analysis |
| Content Production Velocity | Operational | Hourly | Capacity Planning |
| AI Model Performance Index | Operational | Hourly | System Optimization |

### Appendix B: Data Source Dependencies
| Data Source | Metrics Affected | Update Frequency | SLA |
|-------------|------------------|------------------|-----|
| YouTube Analytics API | All video/channel metrics | 2-hour delay | 99.9% |
| Internal Pipeline | Operational metrics | Real-time | 99.95% |
| Cost Tracking System | Financial metrics | Daily | 99.9% |
| ML Platform | AI performance metrics | Real-time | 99.9% |

### Appendix C: Metric Calculation Change Log
| Date | Metric | Version | Change Description | Author |
|------|--------|---------|-------------------|---------|
| 2025-01-15 | Channel Health Score | 2.0 | Added revenue component | Analytics Team |
| 2025-01-10 | Viral Velocity Score | 1.1 | Adjusted time weighting | Data Science |
| 2025-01-05 | ROCI | 1.2 | Include overhead allocation | Finance Team |

---

*This document is maintained by the Analytics Engineering team. For questions or suggestions, please contact analytics-eng@ytempire.com or submit a PR to the metrics-catalog repository.*