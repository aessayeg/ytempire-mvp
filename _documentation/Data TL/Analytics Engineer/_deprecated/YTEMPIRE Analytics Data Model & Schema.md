# YTEMPIRE Analytics Data Model & Schema
**Version**: 1.0  
**Date**: January 2025  
**Author**: Data Team Lead  
**For**: Analytics Engineer  
**Status**: READY FOR IMPLEMENTATION

---

## Executive Summary

This document provides the complete data model, schema definitions, and analytics requirements for the YTEMPIRE MVP. The schema supports tracking 100+ channels, 300+ daily videos, and comprehensive cost/performance analytics with real-time updates and historical trending.

---

## Core Database Schema

### 1. Operational Tables (PostgreSQL + TimescaleDB)

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb";

-- Main operational schema
CREATE SCHEMA IF NOT EXISTS ytempire;

-- ============================================
-- CHANNEL MANAGEMENT
-- ============================================

CREATE TABLE ytempire.channels (
    channel_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    youtube_channel_id VARCHAR(50) UNIQUE NOT NULL,
    channel_name VARCHAR(255) NOT NULL,
    channel_handle VARCHAR(100) UNIQUE,
    niche VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'suspended', 'archived')),
    tier VARCHAR(20) DEFAULT 'standard' CHECK (tier IN ('test', 'standard', 'premium', 'flagship')),
    
    -- YouTube OAuth credentials (encrypted)
    oauth_refresh_token TEXT,
    oauth_token_expires_at TIMESTAMP,
    
    -- Channel metrics snapshot
    current_subscribers INTEGER DEFAULT 0,
    total_videos INTEGER DEFAULT 0,
    total_views BIGINT DEFAULT 0,
    
    -- Financial
    monthly_revenue_target DECIMAL(10,2),
    current_month_revenue DECIMAL(10,2) DEFAULT 0,
    lifetime_revenue DECIMAL(12,2) DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_video_published_at TIMESTAMP,
    
    -- Indexes
    INDEX idx_channels_status (status),
    INDEX idx_channels_niche (niche),
    INDEX idx_channels_tier (tier)
);

-- ============================================
-- VIDEO TRACKING
-- ============================================

CREATE TABLE ytempire.videos (
    video_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    channel_id UUID NOT NULL REFERENCES ytempire.channels(channel_id),
    youtube_video_id VARCHAR(20) UNIQUE,
    
    -- Content details
    title VARCHAR(500) NOT NULL,
    description TEXT,
    tags TEXT[],
    category VARCHAR(100),
    content_type VARCHAR(50) CHECK (content_type IN ('educational', 'entertainment', 'news', 'tutorial', 'review')),
    
    -- Generation details
    generation_method VARCHAR(50) NOT NULL,
    script_id UUID,
    thumbnail_id UUID,
    
    -- Video specs
    duration_seconds INTEGER,
    resolution VARCHAR(20) DEFAULT '1920x1080',
    file_size_mb DECIMAL(10,2),
    
    -- Status tracking
    status VARCHAR(30) DEFAULT 'queued' CHECK (status IN ('queued', 'generating', 'processing', 'uploading', 'published', 'failed', 'deleted')),
    
    -- Timestamps
    queued_at TIMESTAMP DEFAULT NOW(),
    generation_started_at TIMESTAMP,
    generation_completed_at TIMESTAMP,
    uploaded_at TIMESTAMP,
    published_at TIMESTAMP,
    
    -- Performance metrics (updated regularly)
    views INTEGER DEFAULT 0,
    likes INTEGER DEFAULT 0,
    dislikes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    shares INTEGER DEFAULT 0,
    watch_time_hours DECIMAL(12,2) DEFAULT 0,
    
    -- Calculated metrics
    ctr DECIMAL(5,4),
    retention_rate DECIMAL(5,4),
    engagement_rate DECIMAL(5,4),
    
    -- Revenue
    estimated_revenue DECIMAL(10,2) DEFAULT 0,
    actual_revenue DECIMAL(10,2),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_videos_channel (channel_id),
    INDEX idx_videos_status (status),
    INDEX idx_videos_published (published_at DESC),
    INDEX idx_videos_performance (views DESC, engagement_rate DESC)
);

-- ============================================
-- VIDEO GENERATION PIPELINE
-- ============================================

CREATE TABLE ytempire.video_generation_jobs (
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id UUID REFERENCES ytempire.videos(video_id),
    channel_id UUID NOT NULL REFERENCES ytempire.channels(channel_id),
    
    -- Job configuration
    priority INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    topic VARCHAR(500) NOT NULL,
    style VARCHAR(100),
    target_duration_seconds INTEGER,
    
    -- Pipeline stages tracking
    current_stage VARCHAR(50) DEFAULT 'queued',
    stages_completed JSONB DEFAULT '[]'::jsonb,
    stage_timings JSONB DEFAULT '{}'::jsonb,
    
    -- Progress
    progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage BETWEEN 0 AND 100),
    
    -- Status
    status VARCHAR(30) DEFAULT 'pending',
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    failed_at TIMESTAMP,
    
    -- Resource usage
    cpu_seconds_used DECIMAL(10,2),
    gpu_seconds_used DECIMAL(10,2),
    memory_mb_peak INTEGER,
    
    -- Indexes
    INDEX idx_generation_status (status, priority DESC),
    INDEX idx_generation_channel (channel_id),
    INDEX idx_generation_created (created_at DESC)
);

-- ============================================
-- COST TRACKING (CRITICAL FOR <$3/VIDEO)
-- ============================================

CREATE TABLE ytempire.video_costs (
    cost_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id UUID NOT NULL REFERENCES ytempire.videos(video_id),
    
    -- API Costs (in cents to avoid float precision issues)
    openai_cost_cents INTEGER DEFAULT 0,
    elevenlabs_cost_cents INTEGER DEFAULT 0,
    stability_ai_cost_cents INTEGER DEFAULT 0,
    pexels_cost_cents INTEGER DEFAULT 0,
    other_api_cost_cents INTEGER DEFAULT 0,
    
    -- Compute Costs
    cpu_compute_cost_cents INTEGER DEFAULT 0,
    gpu_compute_cost_cents INTEGER DEFAULT 0,
    storage_cost_cents INTEGER DEFAULT 0,
    bandwidth_cost_cents INTEGER DEFAULT 0,
    
    -- Totals
    total_api_cost_cents INTEGER GENERATED ALWAYS AS (
        openai_cost_cents + elevenlabs_cost_cents + stability_ai_cost_cents + 
        pexels_cost_cents + other_api_cost_cents
    ) STORED,
    
    total_compute_cost_cents INTEGER GENERATED ALWAYS AS (
        cpu_compute_cost_cents + gpu_compute_cost_cents + 
        storage_cost_cents + bandwidth_cost_cents
    ) STORED,
    
    total_cost_cents INTEGER GENERATED ALWAYS AS (
        openai_cost_cents + elevenlabs_cost_cents + stability_ai_cost_cents + 
        pexels_cost_cents + other_api_cost_cents + cpu_compute_cost_cents + 
        gpu_compute_cost_cents + storage_cost_cents + bandwidth_cost_cents
    ) STORED,
    
    -- Cost in dollars for easy reading
    total_cost_dollars DECIMAL(10,2) GENERATED ALWAYS AS (total_cost_cents / 100.0) STORED,
    
    -- Tracking
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT max_cost_per_video CHECK (total_cost_cents <= 300),  -- $3.00 max
    
    -- Indexes
    INDEX idx_costs_video (video_id),
    INDEX idx_costs_total (total_cost_dollars DESC)
);

-- ============================================
-- TIME SERIES METRICS (TimescaleDB)
-- ============================================

CREATE TABLE ytempire.video_metrics_timeseries (
    time TIMESTAMP NOT NULL,
    video_id UUID NOT NULL,
    channel_id UUID NOT NULL,
    
    -- Metrics snapshot
    views INTEGER,
    views_delta INTEGER,  -- Change since last measurement
    likes INTEGER,
    comments INTEGER,
    watch_time_hours DECIMAL(10,2),
    
    -- Calculated metrics
    ctr DECIMAL(5,4),
    retention_rate DECIMAL(5,4),
    engagement_rate DECIMAL(5,4),
    
    -- Revenue
    estimated_revenue_cents INTEGER,
    
    PRIMARY KEY (video_id, time)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('ytempire.video_metrics_timeseries', 'time');

-- Create continuous aggregate for hourly rollups
CREATE MATERIALIZED VIEW ytempire.video_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS hour,
    video_id,
    channel_id,
    
    -- Aggregations
    MAX(views) as views,
    SUM(views_delta) as views_gained,
    MAX(likes) as likes,
    MAX(comments) as comments,
    AVG(ctr) as avg_ctr,
    AVG(retention_rate) as avg_retention,
    AVG(engagement_rate) as avg_engagement,
    MAX(estimated_revenue_cents) as revenue_cents
FROM ytempire.video_metrics_timeseries
GROUP BY hour, video_id, channel_id
WITH NO DATA;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('ytempire.video_metrics_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes'
);

-- ============================================
-- YOUTUBE API TRACKING
-- ============================================

CREATE TABLE ytempire.youtube_api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_method VARCHAR(100) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    
    -- Quota tracking
    quota_cost INTEGER NOT NULL,
    account_used VARCHAR(100),  -- Which of our 15 accounts
    
    -- Request/Response
    request_params JSONB,
    response_code INTEGER,
    response_time_ms INTEGER,
    error_message TEXT,
    
    -- Tracking
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_api_usage_time (created_at DESC),
    INDEX idx_api_usage_account (account_used),
    INDEX idx_api_usage_method (api_method)
);

-- Daily quota tracking view
CREATE VIEW ytempire.youtube_quota_daily AS
SELECT 
    DATE(created_at) as date,
    account_used,
    SUM(quota_cost) as quota_used,
    10000 - SUM(quota_cost) as quota_remaining,
    COUNT(*) as api_calls,
    AVG(response_time_ms) as avg_response_time_ms
FROM ytempire.youtube_api_usage
WHERE created_at >= CURRENT_DATE
GROUP BY DATE(created_at), account_used;
```

### 2. Analytics Views and Materialized Views

```sql
-- ============================================
-- CHANNEL PERFORMANCE ANALYTICS
-- ============================================

CREATE MATERIALIZED VIEW ytempire.channel_performance_daily AS
SELECT 
    c.channel_id,
    c.channel_name,
    c.niche,
    c.tier,
    DATE(v.published_at) as date,
    
    -- Video metrics
    COUNT(DISTINCT v.video_id) as videos_published,
    AVG(v.duration_seconds) as avg_video_duration,
    
    -- Performance metrics
    SUM(v.views) as total_views,
    AVG(v.views) as avg_views_per_video,
    SUM(v.likes) as total_likes,
    SUM(v.comments) as total_comments,
    AVG(v.ctr) as avg_ctr,
    AVG(v.retention_rate) as avg_retention,
    AVG(v.engagement_rate) as avg_engagement,
    
    -- Financial metrics
    SUM(v.estimated_revenue) as daily_revenue,
    AVG(vc.total_cost_dollars) as avg_cost_per_video,
    SUM(v.estimated_revenue) - SUM(vc.total_cost_dollars) as daily_profit,
    
    -- ROI calculation
    CASE 
        WHEN SUM(vc.total_cost_dollars) > 0 
        THEN (SUM(v.estimated_revenue) - SUM(vc.total_cost_dollars)) / SUM(vc.total_cost_dollars) * 100
        ELSE 0 
    END as roi_percentage
    
FROM ytempire.channels c
JOIN ytempire.videos v ON c.channel_id = v.channel_id
LEFT JOIN ytempire.video_costs vc ON v.video_id = vc.video_id
WHERE v.published_at IS NOT NULL
GROUP BY c.channel_id, c.channel_name, c.niche, c.tier, DATE(v.published_at);

CREATE INDEX idx_channel_perf_daily ON ytempire.channel_performance_daily(channel_id, date DESC);

-- ============================================
-- VIDEO GENERATION EFFICIENCY
-- ============================================

CREATE MATERIALIZED VIEW ytempire.generation_pipeline_metrics AS
SELECT 
    DATE(g.created_at) as date,
    
    -- Volume metrics
    COUNT(*) as total_jobs,
    COUNT(CASE WHEN g.status = 'completed' THEN 1 END) as successful_jobs,
    COUNT(CASE WHEN g.status = 'failed' THEN 1 END) as failed_jobs,
    
    -- Success rate
    COUNT(CASE WHEN g.status = 'completed' THEN 1 END)::FLOAT / 
        NULLIF(COUNT(*), 0) * 100 as success_rate,
    
    -- Timing metrics (in minutes)
    AVG(EXTRACT(EPOCH FROM (g.completed_at - g.started_at)) / 60) as avg_generation_time_minutes,
    MIN(EXTRACT(EPOCH FROM (g.completed_at - g.started_at)) / 60) as min_generation_time_minutes,
    MAX(EXTRACT(EPOCH FROM (g.completed_at - g.started_at)) / 60) as max_generation_time_minutes,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (g.completed_at - g.started_at)) / 60) as median_generation_time_minutes,
    
    -- Resource usage
    AVG(g.cpu_seconds_used) as avg_cpu_seconds,
    AVG(g.gpu_seconds_used) as avg_gpu_seconds,
    AVG(g.memory_mb_peak) as avg_memory_mb,
    
    -- Cost analysis
    AVG(vc.total_cost_dollars) as avg_cost_per_video,
    SUM(vc.total_cost_dollars) as total_daily_cost
    
FROM ytempire.video_generation_jobs g
LEFT JOIN ytempire.videos v ON g.video_id = v.video_id
LEFT JOIN ytempire.video_costs vc ON v.video_id = vc.video_id
GROUP BY DATE(g.created_at);

-- ============================================
-- REAL-TIME DASHBOARD VIEWS
-- ============================================

CREATE VIEW ytempire.dashboard_overview AS
SELECT 
    -- System status
    (SELECT COUNT(*) FROM ytempire.channels WHERE status = 'active') as active_channels,
    (SELECT COUNT(*) FROM ytempire.videos WHERE DATE(published_at) = CURRENT_DATE) as videos_today,
    (SELECT COUNT(*) FROM ytempire.video_generation_jobs WHERE status = 'processing') as videos_processing,
    (SELECT COUNT(*) FROM ytempire.video_generation_jobs WHERE status = 'queued') as videos_queued,
    
    -- Performance today
    (SELECT SUM(views) FROM ytempire.videos WHERE DATE(published_at) = CURRENT_DATE) as views_today,
    (SELECT AVG(engagement_rate) FROM ytempire.videos WHERE DATE(published_at) = CURRENT_DATE) as avg_engagement_today,
    
    -- Financial today
    (SELECT SUM(estimated_revenue) FROM ytempire.videos WHERE DATE(published_at) = CURRENT_DATE) as revenue_today,
    (SELECT AVG(total_cost_dollars) FROM ytempire.video_costs vc 
     JOIN ytempire.videos v ON vc.video_id = v.video_id 
     WHERE DATE(v.published_at) = CURRENT_DATE) as avg_cost_today,
    
    -- System health
    (SELECT quota_used FROM ytempire.youtube_quota_daily WHERE date = CURRENT_DATE LIMIT 1) as youtube_quota_used,
    (SELECT COUNT(*) FROM ytempire.video_generation_jobs 
     WHERE status = 'failed' AND DATE(created_at) = CURRENT_DATE) as failures_today;

-- ============================================
-- COST OPTIMIZATION ANALYTICS
-- ============================================

CREATE VIEW ytempire.cost_breakdown_analysis AS
SELECT 
    DATE(v.published_at) as date,
    v.content_type,
    v.generation_method,
    
    -- Average costs by component
    AVG(vc.openai_cost_cents / 100.0) as avg_openai_cost,
    AVG(vc.elevenlabs_cost_cents / 100.0) as avg_elevenlabs_cost,
    AVG(vc.stability_ai_cost_cents / 100.0) as avg_stability_cost,
    AVG((vc.cpu_compute_cost_cents + vc.gpu_compute_cost_cents) / 100.0) as avg_compute_cost,
    AVG(vc.total_cost_dollars) as avg_total_cost,
    
    -- Cost distribution
    COUNT(CASE WHEN vc.total_cost_dollars < 1.0 THEN 1 END) as videos_under_1_dollar,
    COUNT(CASE WHEN vc.total_cost_dollars BETWEEN 1.0 AND 2.0 THEN 1 END) as videos_1_to_2_dollars,
    COUNT(CASE WHEN vc.total_cost_dollars BETWEEN 2.0 AND 3.0 THEN 1 END) as videos_2_to_3_dollars,
    COUNT(CASE WHEN vc.total_cost_dollars > 3.0 THEN 1 END) as videos_over_3_dollars,
    
    -- Performance vs cost
    AVG(v.views) as avg_views,
    AVG(v.estimated_revenue) as avg_revenue,
    AVG(v.estimated_revenue / NULLIF(vc.total_cost_dollars, 0)) as avg_roi_ratio
    
FROM ytempire.videos v
JOIN ytempire.video_costs vc ON v.video_id = vc.video_id
WHERE v.published_at IS NOT NULL
GROUP BY DATE(v.published_at), v.content_type, v.generation_method
ORDER BY date DESC;
```

---

## Data Collection Requirements

### 1. Real-Time Data Collection Points

```python
class DataCollectionPoints:
    """
    Critical data collection points for Analytics Engineer
    """
    
    # Event-based collection (immediate)
    REAL_TIME_EVENTS = [
        'video_generation_started',
        'video_generation_stage_complete',
        'video_generation_failed',
        'video_upload_started',
        'video_published',
        'cost_threshold_exceeded',
        'api_quota_warning'
    ]
    
    # Polling-based collection
    POLLING_INTERVALS = {
        'youtube_metrics': 300,        # 5 minutes
        'channel_statistics': 3600,    # 1 hour
        'cost_calculations': 600,      # 10 minutes
        'system_health': 60,           # 1 minute
        'quota_usage': 300             # 5 minutes
    }
    
    # Batch collection (daily)
    DAILY_BATCH_JOBS = [
        'youtube_analytics_export',
        'revenue_reconciliation',
        'channel_performance_summary',
        'cost_optimization_report',
        'trend_analysis'
    ]
```

### 2. Metrics Calculation Formulas

```sql
-- Key metrics definitions for consistency

-- Engagement Rate
-- (likes + comments * 2 + shares * 3) / views * 100

-- Click-Through Rate (CTR)
-- impressions / clicks * 100

-- Retention Rate
-- average_view_duration / video_duration * 100

-- ROI Percentage
-- (revenue - costs) / costs * 100

-- Viral Coefficient
-- shares / initial_1000_views

-- Cost Per View (CPV)
-- total_cost / total_views

-- Revenue Per Mille (RPM)
-- (revenue / views) * 1000

-- Channel Health Score (custom composite metric)
-- (engagement_rate * 0.3 + retention_rate * 0.3 + ctr * 0.2 + roi * 0.2) * 100
```

---

## Dashboard Requirements

### 1. Executive Dashboard (Real-Time)

```yaml
Executive Dashboard Components:
  
  KPI Cards (Top Row):
    - Active Channels: Current count
    - Today's Revenue: $X,XXX
    - Videos Published: Count today
    - Average Cost/Video: $X.XX
    - System Health: Green/Yellow/Red
    
  Charts (Main Area):
    - Revenue Trend: Line chart (30 days)
    - Channel Performance: Bar chart (top 10)
    - Cost Breakdown: Pie chart
    - Video Pipeline: Funnel visualization
    - Engagement Heatmap: By hour/day
    
  Tables (Bottom):
    - Top Videos Today: By views
    - Failing Channels: Need attention
    - Cost Alerts: Videos >$2.50
```

### 2. Operational Dashboard (Real-Time)

```yaml
Operational Dashboard Components:
  
  Pipeline Monitor:
    - Queue Depth: Visual gauge
    - Processing Rate: Videos/hour
    - Success Rate: Percentage
    - Average Time: Minutes
    
  Resource Monitor:
    - CPU Usage: Real-time graph
    - GPU Usage: Real-time graph
    - Memory: Usage bar
    - Storage: Available space
    
  API Monitor:
    - YouTube Quota: Usage bar (X/10000)
    - OpenAI Usage: Tokens/hour
    - ElevenLabs: Characters/hour
    - Error Rates: By API
    
  Alert Feed:
    - Real-time alert stream
    - Sortable by severity
    - Acknowledgment capability
```

### 3. Channel Performance Dashboard

```yaml
Channel Dashboard Components:
  
  Channel Selector: Dropdown with search
  
  Performance Metrics:
    - Subscriber Growth: Line chart
    - View Velocity: Trend line
    - Revenue Tracking: vs Target
    - Best Videos: Top 5 list
    
  Optimization Suggestions:
    - AI-generated insights
    - A/B test results
    - Content recommendations
    - Upload timing optimization
```

---

## Data Pipeline Architecture

### 1. ETL Pipeline Structure

```python
class YTEmpireETLPipeline:
    """
    Main ETL pipeline for Analytics Engineer implementation
    """
    
    def __init__(self):
        self.source_systems = {
            'youtube_api': YouTubeAPIConnector(),
            'internal_db': PostgreSQLConnector(),
            'cost_calculator': CostCalculationService(),
            'ai_services': AIServiceMonitor()
        }
        
        self.transformation_layer = {
            'cleaners': DataCleaners(),
            'enrichers': DataEnrichers(),
            'aggregators': DataAggregators(),
            'validators': DataValidators()
        }
        
        self.target_systems = {
            'operational_db': PostgreSQLWriter(),
            'timeseries_db': TimescaleDBWriter(),
            'cache_layer': RedisWriter(),
            'event_stream': KafkaProducer()
        }
    
    def run_pipeline(self):
        """
        Main pipeline execution
        """
        # Extract
        raw_data = self.extract_from_sources()
        
        # Transform
        cleaned_data = self.transformation_layer['cleaners'].clean(raw_data)
        enriched_data = self.transformation_layer['enrichers'].enrich(cleaned_data)
        validated_data = self.transformation_layer['validators'].validate(enriched_data)
        
        # Load
        self.load_to_targets(validated_data)
        
        # Update materialized views
        self.refresh_materialized_views()
        
        return PipelineResult(success=True, records_processed=len(validated_data))
```

### 2. Real-Time Stream Processing

```python
class RealTimeStreamProcessor:
    """
    Kafka-based real-time processing for Analytics Engineer
    """
    
    def __init__(self):
        self.kafka_consumer = KafkaConsumer(
            'video-events',
            'cost-events',
            'api-events',
            bootstrap_servers='localhost:9092',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
    def process_events(self):
        """
        Real-time event processing
        """
        for message in self.kafka_consumer:
            event_type = message.value['event_type']
            
            if event_type == 'video_published':
                self.update_video_metrics(message.value)
                self.trigger_dashboard_refresh(message.value['channel_id'])
                
            elif event_type == 'cost_calculated':
                self.update_cost_tracking(message.value)
                if message.value['total_cost'] > 2.50:
                    self.send_cost_alert(message.value)
                    
            elif event_type == 'api_quota_update':
                self.update_quota_tracking(message.value)
                if message.value['remaining'] < 1000:
                    self.send_quota_warning(message.value)
```

---

## Implementation Priorities

### Week 1-2: Foundation
1. Set up PostgreSQL with TimescaleDB
2. Create all base tables and indexes
3. Implement cost tracking tables (CRITICAL)
4. Set up basic data ingestion

### Week 3-4: Analytics Layer
1. Create materialized views
2. Implement real-time views
3. Set up continuous aggregates
4. Build ETL pipelines

### Week 5-6: Dashboards
1. Executive dashboard
2. Operational dashboard
3. Channel performance dashboard
4. Cost optimization dashboard

### Week 7-8: Optimization
1. Query optimization
2. Index tuning
3. Caching layer
4. Performance testing

---

## Critical Success Factors

1. **Cost Tracking Accuracy**: Must track to the penny
2. **Real-Time Updates**: <1 minute data freshness
3. **Dashboard Performance**: <2 second load times
4. **Data Quality**: 99% accuracy on metrics
5. **Scalability**: Support 300+ videos/day

---

## Next Steps for Analytics Engineer

1. **Review this schema** and identify any gaps
2. **Set up development environment** with PostgreSQL + TimescaleDB
3. **Create tables** in order of dependencies
4. **Build first ETL pipeline** for cost tracking
5. **Create first dashboard** (Executive Overview)

This completes the data model and analytics requirements. You have everything needed to begin implementation immediately.