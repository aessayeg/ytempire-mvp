# YTEMPIRE Attribution Modeling Documentation

## Document Control
- **Version**: 1.0
- **Last Updated**: January 2025
- **Owner**: Analytics Engineering Team
- **Audience**: Analytics Engineers, Data Scientists, Product Managers

---

## 1. Executive Summary

This document outlines YTEMPIRE's attribution modeling framework for tracking and measuring the impact of AI decisions, content strategies, and marketing efforts on business outcomes. Our multi-touch attribution system enables precise ROI calculation and optimization of our content empire.

---

## 2. Attribution Framework Overview

### 2.1 Attribution Objectives

1. **Decision Attribution**: Track which AI decisions led to video success
2. **Revenue Attribution**: Assign revenue credit to contributing factors
3. **Cost Attribution**: Allocate costs accurately across content pieces
4. **Channel Attribution**: Understand cross-channel viewer journeys
5. **Model Attribution**: Measure AI model contribution to outcomes

### 2.2 Attribution Hierarchy

```
Attribution Levels:
├── Strategic Level
│   ├── Content Strategy Attribution
│   ├── Channel Mix Attribution
│   └── Timing Attribution
│
├── Tactical Level
│   ├── Topic Selection Attribution
│   ├── Format Choice Attribution
│   ├── Thumbnail Design Attribution
│   └── Title Optimization Attribution
│
├── Operational Level
│   ├── AI Model Attribution
│   ├── Publishing Time Attribution
│   ├── Promotion Channel Attribution
│   └── Cross-promotion Attribution
│
└── Technical Level
    ├── Algorithm Version Attribution
    ├── Feature Attribution
    ├── Parameter Attribution
    └── Infrastructure Attribution
```

---

## 3. Core Attribution Models

### 3.1 First-Touch Attribution Model

```sql
-- Assigns 100% credit to the first touchpoint
WITH first_touch_attribution AS (
    SELECT 
        viewer_id,
        channel_id,
        video_id,
        attribution_id,
        touchpoint_type,
        touchpoint_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY viewer_id, channel_id 
            ORDER BY touchpoint_timestamp
        ) AS touch_order
    FROM viewer_touchpoints
    WHERE touchpoint_timestamp >= CURRENT_DATE - INTERVAL '30 days'
)
SELECT 
    channel_id,
    touchpoint_type AS first_touch_source,
    COUNT(DISTINCT viewer_id) AS attributed_viewers,
    SUM(subsequent_revenue) AS attributed_revenue,
    AVG(subsequent_ltv) AS avg_attributed_ltv
FROM first_touch_attribution
WHERE touch_order = 1
GROUP BY channel_id, touchpoint_type;
```

**Use Cases**:
- New viewer acquisition analysis
- Channel discovery optimization
- External traffic source evaluation

### 3.2 Last-Touch Attribution Model

```sql
-- Assigns 100% credit to the last touchpoint before conversion
WITH last_touch_attribution AS (
    SELECT 
        t.viewer_id,
        t.channel_id,
        t.video_id,
        t.attribution_id,
        t.touchpoint_type,
        t.touchpoint_timestamp,
        c.conversion_timestamp,
        c.conversion_value,
        ROW_NUMBER() OVER (
            PARTITION BY t.viewer_id, c.conversion_id 
            ORDER BY t.touchpoint_timestamp DESC
        ) AS reverse_touch_order
    FROM viewer_touchpoints t
    JOIN conversions c ON t.viewer_id = c.viewer_id
    WHERE t.touchpoint_timestamp <= c.conversion_timestamp
        AND t.touchpoint_timestamp >= c.conversion_timestamp - INTERVAL '7 days'
)
SELECT 
    video_id,
    touchpoint_type AS last_touch_source,
    COUNT(DISTINCT viewer_id) AS converting_viewers,
    SUM(conversion_value) AS total_conversion_value,
    AVG(conversion_value) AS avg_conversion_value
FROM last_touch_attribution
WHERE reverse_touch_order = 1
GROUP BY video_id, touchpoint_type;
```

**Use Cases**:
- Conversion optimization
- Direct response measurement
- Short-term campaign effectiveness

### 3.3 Linear Attribution Model

```sql
-- Distributes credit equally across all touchpoints
WITH linear_attribution AS (
    SELECT 
        t.viewer_id,
        t.channel_id,
        t.video_id,
        t.attribution_id,
        t.touchpoint_type,
        c.conversion_value,
        COUNT(*) OVER (
            PARTITION BY t.viewer_id, c.conversion_id
        ) AS total_touchpoints,
        1.0 / COUNT(*) OVER (
            PARTITION BY t.viewer_id, c.conversion_id
        ) AS attribution_weight
    FROM viewer_touchpoints t
    JOIN conversions c ON t.viewer_id = c.viewer_id
    WHERE t.touchpoint_timestamp <= c.conversion_timestamp
        AND t.touchpoint_timestamp >= c.conversion_timestamp - INTERVAL '30 days'
)
SELECT 
    video_id,
    touchpoint_type,
    SUM(attribution_weight) AS total_attribution_score,
    SUM(conversion_value * attribution_weight) AS attributed_revenue,
    COUNT(DISTINCT viewer_id) AS unique_viewers_touched
FROM linear_attribution
GROUP BY video_id, touchpoint_type
ORDER BY attributed_revenue DESC;
```

**Use Cases**:
- Holistic journey analysis
- Multi-touch campaign evaluation
- Content series attribution

### 3.4 Time-Decay Attribution Model

```sql
-- Gives more credit to touchpoints closer to conversion
WITH time_decay_attribution AS (
    SELECT 
        t.viewer_id,
        t.video_id,
        t.attribution_id,
        t.touchpoint_type,
        t.touchpoint_timestamp,
        c.conversion_timestamp,
        c.conversion_value,
        -- Calculate days before conversion
        EXTRACT(EPOCH FROM (c.conversion_timestamp - t.touchpoint_timestamp)) / 86400 AS days_before_conversion,
        -- Apply exponential decay with 7-day half-life
        POWER(0.5, EXTRACT(EPOCH FROM (c.conversion_timestamp - t.touchpoint_timestamp)) / (86400 * 7)) AS decay_weight
    FROM viewer_touchpoints t
    JOIN conversions c ON t.viewer_id = c.viewer_id
    WHERE t.touchpoint_timestamp <= c.conversion_timestamp
        AND t.touchpoint_timestamp >= c.conversion_timestamp - INTERVAL '30 days'
),
normalized_attribution AS (
    SELECT 
        *,
        decay_weight / SUM(decay_weight) OVER (
            PARTITION BY viewer_id, conversion_timestamp
        ) AS normalized_weight
    FROM time_decay_attribution
)
SELECT 
    video_id,
    touchpoint_type,
    AVG(days_before_conversion) AS avg_days_before_conversion,
    SUM(normalized_weight) AS total_attribution_score,
    SUM(conversion_value * normalized_weight) AS attributed_revenue
FROM normalized_attribution
GROUP BY video_id, touchpoint_type;
```

**Use Cases**:
- Recent interaction emphasis
- Purchase journey analysis
- Momentum-based optimization

### 3.5 Custom AI Decision Attribution Model

```sql
-- YTEMPIRE's proprietary attribution model for AI decisions
WITH ai_decision_attribution AS (
    SELECT 
        d.decision_id,
        d.attribution_id,
        d.decision_type,
        d.model_version,
        d.confidence_score,
        d.timestamp AS decision_timestamp,
        v.video_id,
        v.published_at,
        -- Outcomes
        p.views,
        p.revenue_cents,
        p.engagement_rate,
        p.video_success_score
    FROM ai_decisions d
    JOIN videos v ON d.attribution_id = v.attribution_id
    JOIN video_performance p ON v.video_id = p.video_id
),
decision_impact AS (
    SELECT 
        decision_id,
        attribution_id,
        decision_type,
        model_version,
        -- Calculate impact scores
        views * confidence_score AS confidence_weighted_views,
        revenue_cents * confidence_score AS confidence_weighted_revenue,
        
        -- Compare to baseline (videos without this decision type)
        views / NULLIF(AVG(views) OVER (
            PARTITION BY decision_type
        ), 0) AS relative_view_performance,
        
        revenue_cents / NULLIF(AVG(revenue_cents) OVER (
            PARTITION BY decision_type
        ), 0) AS relative_revenue_performance,
        
        -- Multi-factor score
        (
            0.3 * (views / NULLIF(AVG(views) OVER (), 0)) +
            0.4 * (revenue_cents / NULLIF(AVG(revenue_cents) OVER (), 0)) +
            0.3 * (engagement_rate / NULLIF(AVG(engagement_rate) OVER (), 0))
        ) * confidence_score AS decision_impact_score
        
    FROM ai_decision_attribution
)
SELECT 
    decision_type,
    model_version,
    COUNT(DISTINCT attribution_id) AS total_decisions,
    AVG(confidence_score) AS avg_confidence,
    
    -- Performance metrics
    AVG(relative_view_performance) AS avg_view_multiplier,
    AVG(relative_revenue_performance) AS avg_revenue_multiplier,
    AVG(decision_impact_score) AS avg_impact_score,
    
    -- Financial impact
    SUM(confidence_weighted_revenue) / 100 AS total_attributed_revenue,
    
    -- Percentile analysis
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY decision_impact_score) AS median_impact,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY decision_impact_score) AS p90_impact
    
FROM decision_impact
GROUP BY decision_type, model_version
ORDER BY avg_impact_score DESC;
```

**Use Cases**:
- AI model performance evaluation
- Decision optimization
- ROI calculation for AI investments

---

## 4. Multi-Channel Attribution

### 4.1 Cross-Channel Journey Mapping

```sql
-- Track viewer journeys across multiple channels
WITH cross_channel_journeys AS (
    SELECT 
        viewer_id,
        session_id,
        channel_id,
        video_id,
        platform,
        touchpoint_timestamp,
        touchpoint_type,
        -- Create journey string
        STRING_AGG(
            channel_id || ':' || platform || ':' || touchpoint_type,
            ' → '
            ORDER BY touchpoint_timestamp
        ) AS journey_path,
        -- Journey metrics
        COUNT(*) AS touchpoint_count,
        COUNT(DISTINCT channel_id) AS channels_touched,
        COUNT(DISTINCT platform) AS platforms_used
    FROM viewer_touchpoints
    WHERE touchpoint_timestamp >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY viewer_id, session_id
),
journey_patterns AS (
    SELECT 
        journey_path,
        COUNT(*) AS journey_frequency,
        AVG(touchpoint_count) AS avg_touchpoints,
        AVG(channels_touched) AS avg_channels,
        -- Outcome metrics
        AVG(total_watch_time_minutes) AS avg_watch_time,
        AVG(total_revenue_cents) / 100 AS avg_revenue,
        SUM(CASE WHEN converted = true THEN 1 ELSE 0 END) AS conversions
    FROM cross_channel_journeys j
    LEFT JOIN viewer_outcomes o ON j.viewer_id = o.viewer_id
    GROUP BY journey_path
    HAVING COUNT(*) >= 10  -- Minimum sample size
)
SELECT 
    journey_path,
    journey_frequency,
    avg_touchpoints,
    avg_channels,
    avg_watch_time,
    avg_revenue,
    conversions,
    conversions::FLOAT / journey_frequency AS conversion_rate,
    avg_revenue * conversions AS total_attributed_revenue
FROM journey_patterns
ORDER BY total_attributed_revenue DESC
LIMIT 100;
```

### 4.2 Channel Interaction Effects

```sql
-- Measure synergies between channels
WITH channel_combinations AS (
    SELECT 
        viewer_id,
        ARRAY_AGG(DISTINCT channel_id ORDER BY channel_id) AS channel_set,
        COUNT(DISTINCT video_id) AS videos_watched,
        SUM(watch_time_minutes) AS total_watch_time,
        SUM(revenue_cents) / 100 AS total_revenue
    FROM viewer_channel_activity
    WHERE activity_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY viewer_id
),
channel_synergies AS (
    SELECT 
        channel_set,
        CARDINALITY(channel_set) AS channel_count,
        COUNT(DISTINCT viewer_id) AS viewer_count,
        AVG(videos_watched) AS avg_videos_per_viewer,
        AVG(total_watch_time) AS avg_watch_time_per_viewer,
        AVG(total_revenue) AS avg_revenue_per_viewer,
        -- Calculate lift vs single channel
        AVG(total_revenue) / NULLIF(
            AVG(AVG(total_revenue)) OVER (PARTITION BY CARDINALITY(channel_set) = 1),
            0
        ) AS revenue_lift_multiplier
    FROM channel_combinations
    GROUP BY channel_set
)
SELECT 
    channel_set,
    channel_count,
    viewer_count,
    avg_videos_per_viewer,
    avg_watch_time_per_viewer,
    avg_revenue_per_viewer,
    revenue_lift_multiplier,
    (revenue_lift_multiplier - 1) * 100 AS synergy_percentage,
    avg_revenue_per_viewer * viewer_count AS total_channel_group_revenue
FROM channel_synergies
WHERE channel_count > 1
    AND viewer_count >= 100  -- Statistical significance
ORDER BY revenue_lift_multiplier DESC;
```

---

## 5. Revenue Attribution Framework

### 5.1 Direct Revenue Attribution

```sql
-- Attribute revenue directly to content and decisions
CREATE TABLE revenue_attribution (
    attribution_id UUID PRIMARY KEY,
    video_id VARCHAR(50),
    channel_id VARCHAR(50),
    revenue_date DATE,
    
    -- Revenue breakdown
    ad_revenue_cents INTEGER,
    premium_revenue_cents INTEGER,
    membership_revenue_cents INTEGER,
    super_thanks_cents INTEGER,
    
    -- Attribution factors
    thumbnail_attribution_id UUID,
    title_attribution_id UUID,
    topic_attribution_id UUID,
    timing_attribution_id UUID,
    
    -- Attribution weights (sum to 1.0)
    content_weight DECIMAL(3,2),
    thumbnail_weight DECIMAL(3,2),
    title_weight DECIMAL(3,2),
    timing_weight DECIMAL(3,2),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(20)
);

-- Calculate attributed revenue by factor
WITH factor_attribution AS (
    SELECT 
        attribution_id,
        video_id,
        (ad_revenue_cents + premium_revenue_cents + membership_revenue_cents + super_thanks_cents) / 100 AS total_revenue,
        
        -- Attribute to each factor
        thumbnail_attribution_id,
        total_revenue * thumbnail_weight AS thumbnail_attributed_revenue,
        
        title_attribution_id,
        total_revenue * title_weight AS title_attributed_revenue,
        
        topic_attribution_id,
        total_revenue * content_weight AS content_attributed_revenue,
        
        timing_attribution_id,
        total_revenue * timing_weight AS timing_attributed_revenue
        
    FROM revenue_attribution
    WHERE revenue_date >= CURRENT_DATE - INTERVAL '30 days'
)
SELECT 
    'thumbnail' AS factor_type,
    thumbnail_attribution_id AS attribution_id,
    COUNT(DISTINCT video_id) AS videos_impacted,
    SUM(thumbnail_attributed_revenue) AS total_attributed_revenue,
    AVG(thumbnail_attributed_revenue) AS avg_revenue_per_video
FROM factor_attribution
GROUP BY thumbnail_attribution_id

UNION ALL

SELECT 
    'title' AS factor_type,
    title_attribution_id AS attribution_id,
    COUNT(DISTINCT video_id) AS videos_impacted,
    SUM(title_attributed_revenue) AS total_attributed_revenue,
    AVG(title_attributed_revenue) AS avg_revenue_per_video
FROM factor_attribution
GROUP BY title_attribution_id

-- Continue for other factors...
ORDER BY total_attributed_revenue DESC;
```

### 5.2 Indirect Revenue Attribution

```sql
-- Track indirect revenue effects (halo effect, channel growth)
WITH indirect_effects AS (
    SELECT 
        v1.video_id AS source_video_id,
        v1.channel_id,
        v2.video_id AS influenced_video_id,
        
        -- Measure influence based on viewer overlap and timing
        COUNT(DISTINCT vw1.viewer_id) AS shared_viewers,
        
        -- Revenue from influenced videos
        SUM(r2.revenue_cents) / 100 AS influenced_revenue,
        
        -- Calculate influence score
        COUNT(DISTINCT vw1.viewer_id)::FLOAT / 
            NULLIF(COUNT(DISTINCT vw2.viewer_id), 0) AS viewer_overlap_ratio,
            
        -- Time decay factor
        EXP(-EXTRACT(EPOCH FROM (v2.published_at - v1.published_at)) / (86400 * 7)) AS time_decay_factor
        
    FROM videos v1
    JOIN video_views vw1 ON v1.video_id = vw1.video_id
    JOIN video_views vw2 ON vw1.viewer_id = vw2.viewer_id
    JOIN videos v2 ON vw2.video_id = v2.video_id
    JOIN video_revenue r2 ON v2.video_id = r2.video_id
    
    WHERE v1.channel_id = v2.channel_id
        AND v2.published_at > v1.published_at
        AND v2.published_at <= v1.published_at + INTERVAL '30 days'
        AND vw2.view_timestamp > vw1.view_timestamp
        
    GROUP BY v1.video_id, v1.channel_id, v2.video_id, v2.published_at, v1.published_at
)
SELECT 
    source_video_id,
    channel_id,
    COUNT(DISTINCT influenced_video_id) AS videos_influenced,
    SUM(shared_viewers) AS total_viewer_connections,
    SUM(influenced_revenue * viewer_overlap_ratio * time_decay_factor) AS attributed_indirect_revenue,
    
    -- ROI of creating influential content
    SUM(influenced_revenue * viewer_overlap_ratio * time_decay_factor) / 
        NULLIF((SELECT production_cost FROM video_costs WHERE video_id = source_video_id), 0) AS indirect_roi
        
FROM indirect_effects
GROUP BY source_video_id, channel_id
HAVING COUNT(DISTINCT influenced_video_id) >= 3  -- Minimum influence threshold
ORDER BY attributed_indirect_revenue DESC;
```

---

## 6. Attribution Windows and Lookback Periods

### 6.1 Standard Attribution Windows

```sql
-- Define standard attribution windows for different conversion types
CREATE TABLE attribution_windows (
    conversion_type VARCHAR(50) PRIMARY KEY,
    click_lookback_days INTEGER,
    view_lookback_days INTEGER,
    engagement_lookback_days INTEGER,
    
    -- Window weights (for hybrid models)
    immediate_weight DECIMAL(3,2),  -- 0-1 day
    short_term_weight DECIMAL(3,2), -- 2-7 days  
    medium_term_weight DECIMAL(3,2), -- 8-30 days
    long_term_weight DECIMAL(3,2),  -- 31-90 days
    
    CHECK (immediate_weight + short_term_weight + medium_term_weight + long_term_weight = 1.0)
);

-- Standard window configurations
INSERT INTO attribution_windows VALUES
('video_view', 1, 7, 3, 0.5, 0.3, 0.15, 0.05),
('channel_subscribe', 7, 30, 14, 0.3, 0.4, 0.2, 0.1),
('membership_join', 30, 90, 30, 0.2, 0.3, 0.3, 0.2),
('merchandise_purchase', 14, 60, 30, 0.4, 0.3, 0.2, 0.1),
('sponsored_conversion', 1, 14, 7, 0.6, 0.3, 0.1, 0.0);
```

### 6.2 Dynamic Attribution Window Adjustment

```sql
-- Adjust attribution windows based on content type and viewer behavior
WITH viewer_behavior_analysis AS (
    SELECT 
        channel_id,
        content_type,
        AVG(EXTRACT(EPOCH FROM (conversion_timestamp - first_touch_timestamp)) / 86400) AS avg_days_to_convert,
        STDDEV(EXTRACT(EPOCH FROM (conversion_timestamp - first_touch_timestamp)) / 86400) AS stddev_days_to_convert,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (conversion_timestamp - first_touch_timestamp)) / 86400) AS p25_days,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (conversion_timestamp - first_touch_timestamp)) / 86400) AS p75_days,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (conversion_timestamp - first_touch_timestamp)) / 86400) AS p95_days
    FROM viewer_conversions
    WHERE conversion_timestamp >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY channel_id, content_type
)
SELECT 
    channel_id,
    content_type,
    avg_days_to_convert,
    -- Recommend optimal attribution window
    CASE 
        WHEN avg_days_to_convert <= 1 THEN '1-day window'
        WHEN avg_days_to_convert <= 7 THEN '7-day window'
        WHEN avg_days_to_convert <= 30 THEN '30-day window'
        ELSE '90-day window'
    END AS recommended_window,
    
    -- Statistical confidence
    CASE 
        WHEN stddev_days_to_convert / NULLIF(avg_days_to_convert, 0) < 0.5 THEN 'High confidence'
        WHEN stddev_days_to_convert / NULLIF(avg_days_to_convert, 0) < 1.0 THEN 'Medium confidence'
        ELSE 'Low confidence - high variability'
    END AS confidence_level,
    
    -- Window boundaries
    GREATEST(0, avg_days_to_convert - stddev_days_to_convert) AS lower_bound_days,
    avg_days_to_convert + stddev_days_to_convert AS upper_bound_days,
    p95_days AS conservative_window_days
    
FROM viewer_behavior_analysis
ORDER BY channel_id, content_type;
```

---

## 7. Attribution Data Pipeline

### 7.1 Real-time Attribution Processing

```python
from kafka import KafkaConsumer, KafkaProducer
import json
import uuid
from datetime import datetime

class RealTimeAttributionProcessor:
    """Process attribution events in real-time"""
    
    def __init__(self):
        self.consumer = KafkaConsumer(
            'user-events',
            'ai-decisions',
            'video-performance',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.attribution_cache = {}
        
    def process_event(self, event):
        """Process incoming event for attribution"""
        
        event_type = event.get('type')
        
        if event_type == 'ai_decision':
            attribution_id = self.create_attribution_record(event)
            
        elif event_type == 'video_view':
            self.update_attribution_chain(event)
            
        elif event_type == 'conversion':
            self.calculate_attribution_weights(event)
            
    def create_attribution_record(self, decision_event):
        """Create new attribution record for AI decision"""
        
        attribution_id = str(uuid.uuid4())
        
        attribution_record = {
            'attribution_id': attribution_id,
            'decision_type': decision_event['decision_type'],
            'model_version': decision_event['model_version'],
            'confidence_score': decision_event['confidence'],
            'timestamp': datetime.now().isoformat(),
            'parameters': decision_event['parameters'],
            'expected_outcome': decision_event['expected_outcome']
        }
        
        # Store in cache and database
        self.attribution_cache[attribution_id] = attribution_record
        self.producer.send('attribution-records', attribution_record)
        
        return attribution_id
    
    def update_attribution_chain(self, view_event):
        """Update attribution chain with new touchpoint"""
        
        touchpoint = {
            'viewer_id': view_event['viewer_id'],
            'video_id': view_event['video_id'],
            'attribution_ids': view_event.get('attribution_ids', []),
            'timestamp': view_event['timestamp'],
            'source': view_event['source'],
            'session_id': view_event['session_id']
        }
        
        self.producer.send('attribution-touchpoints', touchpoint)
    
    def calculate_attribution_weights(self, conversion_event):
        """Calculate attribution weights for conversion"""
        
        # Retrieve touchpoint chain
        touchpoints = self.get_touchpoint_chain(
            conversion_event['viewer_id'],
            conversion_event['session_id']
        )
        
        # Apply attribution model
        weights = self.apply_attribution_model(
            touchpoints,
            conversion_event['conversion_type']
        )
        
        # Store attribution results
        attribution_result = {
            'conversion_id': conversion_event['conversion_id'],
            'attribution_weights': weights,
            'total_value': conversion_event['value'],
            'attributed_values': self.distribute_value(
                conversion_event['value'],
                weights
            )
        }
        
        self.producer.send('attribution-results', attribution_result)
```

### 7.2 Batch Attribution Processing

```sql
-- Daily batch job for comprehensive attribution analysis
CREATE OR REPLACE PROCEDURE calculate_daily_attribution()
LANGUAGE plpgsql
AS $
BEGIN
    -- Step 1: Create attribution snapshot
    INSERT INTO attribution_daily_snapshot
    SELECT 
        CURRENT_DATE AS snapshot_date,
        attribution_id,
        decision_type,
        model_version,
        COUNT(DISTINCT video_id) AS videos_impacted,
        COUNT(DISTINCT viewer_id) AS viewers_reached,
        SUM(views) AS total_views,
        SUM(revenue_cents) / 100 AS total_revenue,
        AVG(engagement_rate) AS avg_engagement_rate
    FROM attribution_performance
    WHERE date = CURRENT_DATE - 1
    GROUP BY attribution_id, decision_type, model_version;
    
    -- Step 2: Calculate incremental lift
    WITH baseline_performance AS (
        SELECT 
            decision_type,
            AVG(revenue_cents) AS baseline_revenue,
            AVG(views) AS baseline_views,
            AVG(engagement_rate) AS baseline_engagement
        FROM video_performance
        WHERE attribution_id IS NULL
            AND published_at >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY decision_type
    )
    UPDATE attribution_daily_snapshot s
    SET 
        revenue_lift = (s.total_revenue - b.baseline_revenue) / NULLIF(b.baseline_revenue, 0),
        view_lift = (s.total_views - b.baseline_views) / NULLIF(b.baseline_views, 0),
        engagement_lift = (s.avg_engagement_rate - b.baseline_engagement) / NULLIF(b.baseline_engagement, 0)
    FROM baseline_performance b
    WHERE s.decision_type = b.decision_type
        AND s.snapshot_date = CURRENT_DATE;
    
    -- Step 3: Update attribution scores
    CALL update_attribution_scores(CURRENT_DATE);
    
    -- Step 4: Generate attribution reports
    CALL generate_attribution_reports(CURRENT_DATE);
    
    COMMIT;
END;
$;
```

---

## 8. Attribution Validation and Testing

### 8.1 Attribution Model Validation

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

class AttributionValidator:
    """Validate attribution model accuracy"""
    
    def __init__(self, model_type='time_decay'):
        self.model_type = model_type
        self.validation_results = {}
        
    def holdout_validation(self, data, holdout_period_days=30):
        """Validate using holdout method"""
        
        # Split data
        cutoff_date = data['date'].max() - pd.Timedelta(days=holdout_period_days)
        train_data = data[data['date'] <= cutoff_date]
        test_data = data[data['date'] > cutoff_date]
        
        # Calculate attribution on training data
        train_attribution = self.calculate_attribution(train_data)
        
        # Predict test period revenue
        predicted_revenue = self.predict_revenue(train_attribution, test_data)
        
        # Compare with actual
        actual_revenue = test_data.groupby('attribution_id')['revenue'].sum()
        
        # Calculate metrics
        mae = mean_absolute_error(actual_revenue, predicted_revenue)
        r2 = r2_score(actual_revenue, predicted_revenue)
        mape = np.mean(np.abs((actual_revenue - predicted_revenue) / actual_revenue)) * 100
        
        self.validation_results['holdout'] = {
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'period': holdout_period_days
        }
        
        return self.validation_results
    
    def incrementality_test(self, test_group, control_group):
        """Validate using incrementality testing"""
        
        # Calculate lift
        test_revenue = test_group['revenue'].sum()
        control_revenue = control_group['revenue'].sum()
        
        lift = (test_revenue - control_revenue) / control_revenue
        
        # Statistical significance
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(
            test_group['revenue'],
            control_group['revenue']
        )
        
        self.validation_results['incrementality'] = {
            'lift': lift,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'test_size': len(test_group),
            'control_size': len(control_group)
        }
        
        return self.validation_results
```

### 8.2 Attribution Consistency Checks

```sql
-- Ensure attribution consistency and data quality
WITH attribution_consistency AS (
    SELECT 
        attribution_id,
        decision_type,
        COUNT(DISTINCT video_id) AS video_count,
        COUNT(DISTINCT channel_id) AS channel_count,
        SUM(attribution_weight) AS total_weight,
        MIN(created_at) AS first_seen,
        MAX(created_at) AS last_seen
    FROM attribution_records
    WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY attribution_id, decision_type
),
consistency_issues AS (
    SELECT 
        attribution_id,
        decision_type,
        CASE 
            WHEN total_weight < 0.99 OR total_weight > 1.01 
                THEN 'Weight sum not equal to 1'
            WHEN video_count = 0 
                THEN 'No videos associated'
            WHEN channel_count > 1 
                THEN 'Multiple channels (unexpected)'
            WHEN EXTRACT(EPOCH FROM (last_seen - first_seen)) > 86400 * 30 
                THEN 'Attribution span too long'
            ELSE 'OK'
        END AS issue_type
    FROM attribution_consistency
)
SELECT 
    issue_type,
    COUNT(*) AS issue_count,
    COUNT(DISTINCT attribution_id) AS affected_attributions,
    ARRAY_AGG(DISTINCT decision_type) AS affected_decision_types
FROM consistency_issues
WHERE issue_type != 'OK'
GROUP BY issue_type;
```

---

## 9. Attribution Reporting

### 9.1 Executive Attribution Dashboard

```sql
-- Executive summary of attribution performance
CREATE VIEW executive_attribution_summary AS
WITH current_period AS (
    SELECT 
        decision_type,
        model_version,
        SUM(attributed_revenue) AS revenue,
        SUM(attributed_views) AS views,
        COUNT(DISTINCT attribution_id) AS decisions,
        AVG(confidence_score) AS avg_confidence
    FROM attribution_performance
    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY decision_type, model_version
),
previous_period AS (
    SELECT 
        decision_type,
        model_version,
        SUM(attributed_revenue) AS revenue,
        SUM(attributed_views) AS views
    FROM attribution_performance
    WHERE date >= CURRENT_DATE - INTERVAL '60 days'
        AND date < CURRENT_DATE - INTERVAL '30 days'
    GROUP BY decision_type, model_version
)
SELECT 
    c.decision_type,
    c.model_version,
    c.revenue AS current_revenue,
    c.views AS current_views,
    c.decisions AS total_decisions,
    c.avg_confidence,
    
    -- Period-over-period growth
    (c.revenue - COALESCE(p.revenue, 0)) / NULLIF(p.revenue, 0) * 100 AS revenue_growth_pct,
    (c.views - COALESCE(p.views, 0)) / NULLIF(p.views, 0) * 100 AS view_growth_pct,
    
    -- Efficiency metrics
    c.revenue / NULLIF(c.decisions, 0) AS revenue_per_decision,
    c.views / NULLIF(c.decisions, 0) AS views_per_decision,
    
    -- Ranking
    RANK() OVER (ORDER BY c.revenue DESC) AS revenue_rank,
    RANK() OVER (ORDER BY c.revenue / NULLIF(c.decisions, 0) DESC) AS efficiency_rank
    
FROM current_period c
LEFT JOIN previous_period p 
    ON c.decision_type = p.decision_type 
    AND c.model_version = p.model_version
ORDER BY c.revenue DESC;
```

### 9.2 Attribution Deep Dive Reports

```python
def generate_attribution_report(start_date, end_date, output_format='html'):
    """Generate comprehensive attribution analysis report"""
    
    report_sections = []
    
    # Section 1: Executive Summary
    exec_summary = generate_executive_summary(start_date, end_date)
    report_sections.append(exec_summary)
    
    # Section 2: Model Performance Comparison
    model_comparison = compare_attribution_models(start_date, end_date)
    report_sections.append(model_comparison)
    
    # Section 3: Decision Impact Analysis
    decision_impact = analyze_decision_impact(start_date, end_date)
    report_sections.append(decision_impact)
    
    # Section 4: Revenue Attribution Breakdown
    revenue_breakdown = calculate_revenue_attribution(start_date, end_date)
    report_sections.append(revenue_breakdown)
    
    # Section 5: Recommendations
    recommendations = generate_recommendations(
        model_comparison, 
        decision_impact,
        revenue_breakdown
    )
    report_sections.append(recommendations)
    
    # Format output
    if output_format == 'html':
        return format_as_html(report_sections)
    elif output_format == 'pdf':
        return format_as_pdf(report_sections)
    else:
        return format_as_json(report_sections)
```

---

## 10. Attribution Implementation Guidelines

### 10.1 Best Practices

1. **Data Collection**
   - Capture all touchpoints, not just clicks
   - Include zero-value touchpoints (impressions)
   - Maintain consistent user/viewer identification
   - Track cross-device and cross-platform journeys

2. **Model Selection**
   - Use multiple models for comparison
   - Select based on business objectives
   - Validate with holdout testing
   - Consider hybrid approaches

3. **Implementation**
   - Start with simple models, evolve complexity
   - Ensure real-time processing capability
   - Build fallback mechanisms
   - Monitor attribution quality metrics

### 10.2 Common Pitfalls and Solutions

| Pitfall | Impact | Solution |
|---------|---------|----------|
| Over-attribution | Inflated ROI calculations | Implement attribution caps and validation |
| Under-attribution | Missed optimization opportunities | Include view-through attribution |
| Selection bias | Skewed results | Use control groups and incrementality testing |
| Data gaps | Incomplete attribution | Implement data quality monitoring |
| Model overfitting | Poor predictive power | Regular validation and model refresh |

### 10.3 Implementation Checklist

- [ ] Define business objectives for attribution
- [ ] Map all customer touchpoints
- [ ] Select appropriate attribution models
- [ ] Implement data collection infrastructure
- [ ] Build attribution calculation pipeline
- [ ] Create validation framework
- [ ] Design reporting dashboards
- [ ] Train stakeholders on interpretation
- [ ] Schedule regular model reviews
- [ ] Document attribution methodology

---

## 11. Advanced Attribution Techniques

### 11.1 Machine Learning Attribution

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import shap

class MLAttributionModel:
    """Machine learning-based attribution model"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def prepare_features(self, data):
        """Prepare features for ML model"""
        
        features = pd.DataFrame({
            'thumbnail_quality_score': data['thumbnail_score'],
            'title_optimization_score': data['title_score'],
            'topic_trending_score': data['trending_score'],
            'publish_hour': data['publish_timestamp'].dt.hour,
            'publish_day_of_week': data['publish_timestamp'].dt.dayofweek,
            'channel_subscriber_count': data['subscriber_count'],
            'channel_health_score': data['health_score'],
            'content_length_minutes': data['duration_seconds'] / 60,
            'ai_confidence_score': data['confidence_score']
        })
        
        return features
    
    def train(self, train_data):
        """Train ML attribution model"""
        
        X = self.prepare_features(train_data)
        y = train_data['revenue']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Calculate feature importance
        self.feature_importance = dict(zip(
            X.columns,
            self.model.feature_importances_
        ))
        
        # SHAP analysis for interpretability
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)
        
        return self
    
    def attribute_revenue(self, data):
        """Attribute revenue using ML model"""
        
        X = self.prepare_features(data)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and feature contributions
        predictions = self.model.predict(X_scaled)
        
        # Calculate attribution weights based on feature importance
        attribution_weights = {}
        for feature, importance in self.feature_importance.items():
            if 'thumbnail' in feature:
                attribution_weights['thumbnail'] = importance
            elif 'title' in feature:
                attribution_weights['title'] = importance
            # Continue for other attribution categories...
            
        return attribution_weights
```

### 11.2 Markov Chain Attribution

```sql
-- Implement Markov chain attribution for multi-touch journeys
WITH journey_transitions AS (
    SELECT 
        viewer_id,
        touchpoint_sequence,
        touchpoint_type,
        LEAD(touchpoint_type) OVER (
            PARTITION BY viewer_id 
            ORDER BY touchpoint_sequence
        ) AS next_touchpoint,
        conversion_flag
    FROM viewer_journey_data
),
transition_probabilities AS (
    SELECT 
        touchpoint_type AS from_state,
        next_touchpoint AS to_state,
        COUNT(*) AS transition_count,
        SUM(COUNT(*)) OVER (PARTITION BY touchpoint_type) AS total_from_state,
        COUNT(*)::FLOAT / SUM(COUNT(*)) OVER (PARTITION BY touchpoint_type) AS transition_probability
    FROM journey_transitions
    WHERE next_touchpoint IS NOT NULL
    GROUP BY touchpoint_type, next_touchpoint
),
removal_effects AS (
    -- Calculate removal effect for each touchpoint
    SELECT 
        touchpoint_type,
        -- Simulate removal and calculate conversion probability change
        1 - (
            SELECT SUM(transition_probability * conversion_rate)
            FROM transition_probabilities tp
            JOIN conversion_rates cr ON tp.to_state = cr.state
            WHERE tp.from_state != touchpoint_type
        ) AS removal_effect
    FROM (SELECT DISTINCT touchpoint_type FROM journey_transitions) t
)
SELECT 
    touchpoint_type,
    removal_effect,
    removal_effect / SUM(removal_effect) OVER () AS markov_attribution_weight,
    RANK() OVER (ORDER BY removal_effect DESC) AS importance_rank
FROM removal_effects
ORDER BY removal_effect DESC;
```

---

## Appendices

### Appendix A: Attribution Glossary

| Term | Definition |
|------|------------|
| **Attribution Window** | Time period during which touchpoints are considered for attribution |
| **Conversion Path** | Sequence of touchpoints leading to a conversion |
| **Incrementality** | Additional value generated beyond baseline |
| **Lookback Period** | Historical timeframe for attribution analysis |
| **Multi-Touch Attribution** | Distributing credit across multiple touchpoints |
| **Removal Effect** | Impact of removing a touchpoint from conversion path |
| **Time Decay** | Attribution weight decreasing over time |
| **Touchpoint** | Any interaction between viewer and content |

### Appendix B: Attribution Model Comparison

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| First-Touch | Awareness campaigns | Simple, clear | Ignores nurturing |
| Last-Touch | Direct response | Easy to implement | Misses journey context |
| Linear | Long sales cycles | Fair distribution | Oversimplifies |
| Time-Decay | Recent influence | Recency weighted | Arbitrary decay rate |
| Data-Driven | Complex journeys | Accurate | Requires large dataset |
| Custom AI | YTEMPIRE specific | Tailored to business | Complex to maintain |

### Appendix C: SQL Functions for Attribution

```sql
-- Reusable SQL functions for attribution calculations

-- Function: Calculate attribution weight
CREATE OR REPLACE FUNCTION calculate_attribution_weight(
    model_type VARCHAR,
    touchpoint_position INT,
    total_touchpoints INT,
    days_before_conversion INT
) RETURNS DECIMAL AS $
BEGIN
    CASE model_type
        WHEN 'first_touch' THEN
            RETURN CASE WHEN touchpoint_position = 1 THEN 1.0 ELSE 0.0 END;
        WHEN 'last_touch' THEN
            RETURN CASE WHEN touchpoint_position = total_touchpoints THEN 1.0 ELSE 0.0 END;
        WHEN 'linear' THEN
            RETURN 1.0 / total_touchpoints;
        WHEN 'time_decay' THEN
            RETURN POWER(0.5, days_before_conversion / 7.0);
        ELSE
            RETURN 1.0 / total_touchpoints; -- Default to linear
    END CASE;
END;
$ LANGUAGE plpgsql;

-- Function: Attribute revenue
CREATE OR REPLACE FUNCTION attribute_revenue(
    total_revenue DECIMAL,
    attribution_weights JSONB
) RETURNS JSONB AS $
DECLARE
    attributed_revenue JSONB = '{}';
    touchpoint TEXT;
    weight DECIMAL;
BEGIN
    FOR touchpoint, weight IN SELECT * FROM jsonb_each_text(attribution_weights)
    LOOP
        attributed_revenue = jsonb_set(
            attributed_revenue,
            ARRAY[touchpoint],
            to_jsonb((total_revenue * weight::DECIMAL)::DECIMAL(10,2))
        );
    END LOOP;
    
    RETURN attributed_revenue;
END;
$ LANGUAGE plpgsql;
```

---

*This document is maintained by the Analytics Engineering team. For questions about attribution modeling, please contact analytics-eng@ytempire.com or consult the #attribution-modeling Slack channel.*