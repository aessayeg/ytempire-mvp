# YTEMPIRE Analytics Metrics & Dashboard Specifications

## Document Control
- **Version**: 1.0
- **Date**: January 2025
- **Author**: Head of Analytics / VP of Data
- **Audience**: Analytics Engineer (Primary), All Teams
- **Status**: FINAL - Ready for Implementation

---

## 1. Core Metrics Definitions

### 1.1 Business Metrics

```sql
-- Monthly Recurring Revenue (MRR)
CREATE VIEW analytics.mrr_calculation AS
WITH monthly_revenue AS (
    SELECT 
        DATE_TRUNC('month', date) as month,
        user_id,
        subscription_tier,
        CASE 
            WHEN subscription_tier = 'beta' THEN 0
            WHEN subscription_tier = 'starter' THEN 297
            WHEN subscription_tier = 'growth' THEN 997
            WHEN subscription_tier = 'enterprise' THEN 2997
        END as base_mrr,
        COALESCE(SUM(overage_revenue), 0) as overage_mrr
    FROM billing_records
    WHERE status = 'active'
    GROUP BY 1, 2, 3
)
SELECT 
    month,
    COUNT(DISTINCT user_id) as active_users,
    SUM(base_mrr) as base_mrr,
    SUM(overage_mrr) as overage_mrr,
    SUM(base_mrr + overage_mrr) as total_mrr,
    AVG(base_mrr + overage_mrr) as arpu,
    -- Month-over-month growth
    LAG(SUM(base_mrr + overage_mrr)) OVER (ORDER BY month) as previous_mrr,
    (SUM(base_mrr + overage_mrr) - LAG(SUM(base_mrr + overage_mrr)) OVER (ORDER BY month)) 
        / NULLIF(LAG(SUM(base_mrr + overage_mrr)) OVER (ORDER BY month), 0) * 100 as mrr_growth_rate
FROM monthly_revenue
GROUP BY month
ORDER BY month DESC;

-- Customer Lifetime Value (LTV)
CREATE VIEW analytics.ltv_calculation AS
WITH user_revenue AS (
    SELECT 
        u.user_id,
        u.signup_date,
        MIN(br.date) as first_payment_date,
        MAX(br.date) as last_payment_date,
        COUNT(DISTINCT DATE_TRUNC('month', br.date)) as active_months,
        SUM(br.amount) as total_revenue,
        u.churn_date
    FROM users u
    LEFT JOIN billing_records br ON u.user_id = br.user_id
    GROUP BY u.user_id, u.signup_date, u.churn_date
),
cohort_analysis AS (
    SELECT 
        DATE_TRUNC('month', signup_date) as cohort_month,
        AVG(CASE WHEN churn_date IS NULL THEN 
            EXTRACT(EPOCH FROM (NOW() - first_payment_date)) / 2592000  -- months
        ELSE 
            EXTRACT(EPOCH FROM (churn_date - first_payment_date)) / 2592000
        END) as avg_lifetime_months,
        AVG(total_revenue) as avg_total_revenue,
        AVG(total_revenue / NULLIF(active_months, 0)) as avg_monthly_revenue
    FROM user_revenue
    WHERE first_payment_date IS NOT NULL
    GROUP BY cohort_month
)
SELECT 
    cohort_month,
    avg_lifetime_months,
    avg_monthly_revenue,
    avg_lifetime_months * avg_monthly_revenue as calculated_ltv,
    COUNT(*) as cohort_size
FROM cohort_analysis
ORDER BY cohort_month DESC;

-- Customer Acquisition Cost (CAC)
CREATE VIEW analytics.cac_calculation AS
WITH marketing_costs AS (
    SELECT 
        DATE_TRUNC('month', date) as month,
        SUM(CASE WHEN category = 'paid_ads' THEN amount ELSE 0 END) as paid_ads_cost,
        SUM(CASE WHEN category = 'content_marketing' THEN amount ELSE 0 END) as content_cost,
        SUM(CASE WHEN category = 'sales' THEN amount ELSE 0 END) as sales_cost,
        SUM(amount) as total_marketing_cost
    FROM expenses
    WHERE department = 'marketing'
    GROUP BY month
),
new_customers AS (
    SELECT 
        DATE_TRUNC('month', signup_date) as month,
        COUNT(*) as new_customers,
        COUNT(CASE WHEN acquisition_channel = 'paid' THEN 1 END) as paid_customers,
        COUNT(CASE WHEN acquisition_channel = 'organic' THEN 1 END) as organic_customers
    FROM users
    WHERE signup_date IS NOT NULL
    GROUP BY month
)
SELECT 
    mc.month,
    mc.total_marketing_cost,
    nc.new_customers,
    mc.total_marketing_cost / NULLIF(nc.new_customers, 0) as overall_cac,
    mc.paid_ads_cost / NULLIF(nc.paid_customers, 0) as paid_cac,
    mc.content_cost / NULLIF(nc.organic_customers, 0) as organic_cac,
    -- CAC Payback Period (in months)
    (mc.total_marketing_cost / NULLIF(nc.new_customers, 0)) / 
        (SELECT AVG(base_mrr + overage_mrr) FROM analytics.mrr_calculation WHERE month = mc.month) as cac_payback_months
FROM marketing_costs mc
JOIN new_customers nc ON mc.month = nc.month
ORDER BY mc.month DESC;
```

### 1.2 Operational Metrics

```sql
-- Video Generation Metrics
CREATE MATERIALIZED VIEW analytics.video_generation_metrics AS
WITH video_stats AS (
    SELECT 
        DATE_TRUNC('day', vg.created_at) as date,
        vg.channel_id,
        c.user_id,
        vg.video_id,
        vg.generation_time_seconds,
        vg.total_cost_cents / 100.0 as generation_cost,
        vg.quality_score,
        vg.status,
        v.views,
        v.revenue_cents / 100.0 as revenue,
        v.engagement_rate
    FROM video_generation_log vg
    JOIN channels c ON vg.channel_id = c.channel_id
    LEFT JOIN videos v ON vg.video_id = v.video_id
    WHERE vg.created_at >= CURRENT_DATE - INTERVAL '90 days'
)
SELECT 
    date,
    COUNT(DISTINCT user_id) as active_users,
    COUNT(DISTINCT channel_id) as active_channels,
    COUNT(video_id) as videos_generated,
    COUNT(CASE WHEN status = 'published' THEN 1 END) as videos_published,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as videos_failed,
    
    -- Performance Metrics
    AVG(generation_time_seconds) as avg_generation_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY generation_time_seconds) as p95_generation_time,
    AVG(quality_score) as avg_quality_score,
    
    -- Cost Metrics
    AVG(generation_cost) as avg_cost_per_video,
    SUM(generation_cost) as total_generation_cost,
    
    -- Success Metrics
    COUNT(CASE WHEN status = 'published' THEN 1 END)::FLOAT / NULLIF(COUNT(video_id), 0) as success_rate,
    COUNT(CASE WHEN generation_cost <= 3.00 THEN 1 END)::FLOAT / NULLIF(COUNT(video_id), 0) as cost_target_achievement,
    
    -- ROI Metrics (for published videos)
    AVG(CASE WHEN status = 'published' THEN revenue - generation_cost END) as avg_profit_per_video,
    SUM(CASE WHEN status = 'published' THEN revenue END) / 
        NULLIF(SUM(CASE WHEN status = 'published' THEN generation_cost END), 0) as overall_roi
FROM video_stats
GROUP BY date
ORDER BY date DESC;

-- Platform Performance Metrics
CREATE VIEW analytics.platform_performance AS
SELECT 
    -- API Performance
    (SELECT AVG(response_time_ms) FROM api_logs WHERE timestamp > NOW() - INTERVAL '1 hour') as avg_api_latency,
    (SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) 
     FROM api_logs WHERE timestamp > NOW() - INTERVAL '1 hour') as p95_api_latency,
    
    -- YouTube Quota Usage
    (SELECT SUM(quota_units) FROM youtube_api_calls WHERE date = CURRENT_DATE) as youtube_quota_used_today,
    10000 - (SELECT SUM(quota_units) FROM youtube_api_calls WHERE date = CURRENT_DATE) as youtube_quota_remaining,
    
    -- System Resources
    (SELECT AVG(cpu_usage) FROM system_metrics WHERE timestamp > NOW() - INTERVAL '1 hour') as avg_cpu_usage,
    (SELECT AVG(memory_usage) FROM system_metrics WHERE timestamp > NOW() - INTERVAL '1 hour') as avg_memory_usage,
    
    -- Error Rates
    (SELECT COUNT(*) FROM error_logs WHERE timestamp > NOW() - INTERVAL '1 hour' AND severity = 'error') as errors_last_hour,
    (SELECT COUNT(*) FROM error_logs WHERE timestamp > NOW() - INTERVAL '1 hour' AND severity = 'critical') as critical_errors_last_hour,
    
    -- Pipeline Status
    (SELECT COUNT(*) FROM n8n_workflows WHERE status = 'running') as active_workflows,
    (SELECT COUNT(*) FROM n8n_workflows WHERE status = 'failed' AND timestamp > NOW() - INTERVAL '1 hour') as failed_workflows_last_hour;
```

### 1.3 YouTube Performance Metrics

```sql
-- Channel Performance Aggregation
CREATE MATERIALIZED VIEW analytics.channel_performance AS
WITH channel_metrics AS (
    SELECT 
        c.channel_id,
        c.channel_name,
        c.user_id,
        c.niche,
        c.created_at as channel_created_at,
        COUNT(DISTINCT v.video_id) as total_videos,
        COUNT(DISTINCT CASE WHEN v.published_at >= CURRENT_DATE - INTERVAL '30 days' THEN v.video_id END) as videos_last_30d,
        
        -- View Metrics
        SUM(v.views) as total_views,
        SUM(CASE WHEN v.published_at >= CURRENT_DATE - INTERVAL '30 days' THEN v.views ELSE 0 END) as views_last_30d,
        AVG(v.views) as avg_views_per_video,
        
        -- Engagement Metrics
        AVG(v.engagement_rate) as avg_engagement_rate,
        SUM(v.likes + v.comments + v.shares) as total_engagements,
        
        -- Revenue Metrics
        SUM(v.revenue_cents) / 100.0 as total_revenue,
        SUM(CASE WHEN v.published_at >= CURRENT_DATE - INTERVAL '30 days' THEN v.revenue_cents ELSE 0 END) / 100.0 as revenue_last_30d,
        AVG(v.revenue_cents) / 100.0 as avg_revenue_per_video,
        
        -- Subscriber Metrics
        MAX(ca.subscriber_count) as current_subscribers,
        MAX(ca.subscriber_count) - MIN(ca.subscriber_count) as subscriber_growth_30d
        
    FROM channels c
    LEFT JOIN videos v ON c.channel_id = v.channel_id
    LEFT JOIN channel_analytics ca ON c.channel_id = ca.channel_id 
        AND ca.date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY c.channel_id, c.channel_name, c.user_id, c.niche, c.created_at
),
channel_trends AS (
    SELECT 
        channel_id,
        -- Growth Rates
        CASE 
            WHEN LAG(views_last_30d) OVER (PARTITION BY channel_id ORDER BY CURRENT_DATE) > 0
            THEN (views_last_30d - LAG(views_last_30d) OVER (PARTITION BY channel_id ORDER BY CURRENT_DATE)) 
                / LAG(views_last_30d) OVER (PARTITION BY channel_id ORDER BY CURRENT_DATE) * 100
            ELSE 0 
        END as view_growth_rate,
        
        -- Performance Tier
        CASE 
            WHEN revenue_last_30d >= 10000 THEN 'elite'
            WHEN revenue_last_30d >= 5000 THEN 'growth'
            WHEN revenue_last_30d >= 2000 THEN 'established'
            WHEN revenue_last_30d >= 500 THEN 'developing'
            ELSE 'struggling'
        END as performance_tier
    FROM channel_metrics
)
SELECT 
    cm.*,
    ct.view_growth_rate,
    ct.performance_tier,
    -- Calculate velocity score (momentum indicator)
    (cm.views_last_30d / NULLIF(cm.total_views, 0)) * 100 as recent_activity_percentage,
    -- Revenue per view (RPM/1000)
    (cm.total_revenue / NULLIF(cm.total_views, 0)) * 1000 as rpm
FROM channel_metrics cm
JOIN channel_trends ct ON cm.channel_id = ct.channel_id
ORDER BY cm.revenue_last_30d DESC;

-- Video Performance Analysis
CREATE VIEW analytics.video_performance_analysis AS
WITH video_metrics AS (
    SELECT 
        v.*,
        c.niche,
        c.user_id,
        -- Calculate hours since publish
        EXTRACT(EPOCH FROM (NOW() - v.published_at)) / 3600 as hours_since_publish,
        
        -- Performance ratios
        v.views / NULLIF(EXTRACT(EPOCH FROM (NOW() - v.published_at)) / 3600, 0) as views_per_hour,
        (v.likes + v.comments) / NULLIF(v.views, 0) * 100 as engagement_rate,
        v.revenue_cents / NULLIF(v.views, 0) as revenue_per_view,
        
        -- Thumbnail performance (A/B test results if available)
        vt.click_through_rate as thumbnail_ctr,
        vt.variant as thumbnail_variant
        
    FROM videos v
    JOIN channels c ON v.channel_id = c.channel_id
    LEFT JOIN video_thumbnails vt ON v.video_id = vt.video_id AND vt.is_winner = true
),
video_benchmarks AS (
    SELECT 
        niche,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY views) as views_p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY views) as views_p50,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY views) as views_p75,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY views) as views_p90,
        AVG(engagement_rate) as avg_engagement_rate,
        AVG(revenue_per_view) as avg_revenue_per_view
    FROM video_metrics
    WHERE hours_since_publish > 168  -- Only videos older than 1 week
    GROUP BY niche
)
SELECT 
    vm.*,
    
    -- Performance vs Benchmarks
    CASE 
        WHEN vm.views >= vb.views_p90 THEN 'top_10_percent'
        WHEN vm.views >= vb.views_p75 THEN 'top_25_percent'
        WHEN vm.views >= vb.views_p50 THEN 'above_average'
        WHEN vm.views >= vb.views_p25 THEN 'below_average'
        ELSE 'bottom_25_percent'
    END as performance_tier,
    
    -- Relative performance scores
    vm.views / NULLIF(vb.views_p50, 0) as views_vs_median,
    vm.engagement_rate / NULLIF(vb.avg_engagement_rate, 0) as engagement_vs_average,
    vm.revenue_per_view / NULLIF(vb.avg_revenue_per_view, 0) as revenue_vs_average
    
FROM video_metrics vm
JOIN video_benchmarks vb ON vm.niche = vb.niche
ORDER BY vm.published_at DESC;
```

---

## 2. Grafana Dashboard Configurations

### 2.1 Executive Dashboard

```yaml
dashboard:
  title: "YTEMPIRE Executive Dashboard"
  refresh: "30s"
  time: 
    from: "now-30d"
    to: "now"
  
  rows:
    - title: "Key Business Metrics"
      height: 300
      panels:
        - title: "MRR"
          type: stat
          targets:
            - query: "SELECT total_mrr FROM analytics.mrr_calculation ORDER BY month DESC LIMIT 1"
          fieldConfig:
            defaults:
              unit: "currencyUSD"
              decimals: 0
              thresholds:
                steps:
                  - value: 0
                    color: "red"
                  - value: 50000
                    color: "yellow"
                  - value: 100000
                    color: "green"
        
        - title: "Active Users"
          type: stat
          targets:
            - query: "SELECT COUNT(DISTINCT user_id) FROM users WHERE status = 'active'"
          
        - title: "Total Channels"
          type: stat
          targets:
            - query: "SELECT COUNT(*) FROM channels WHERE status = 'active'"
        
        - title: "Videos Today"
          type: stat
          targets:
            - query: "SELECT COUNT(*) FROM videos WHERE published_at >= CURRENT_DATE"
    
    - title: "Revenue Trends"
      height: 400
      panels:
        - title: "MRR Growth"
          type: graph
          gridPos: {x: 0, y: 0, w: 12, h: 8}
          targets:
            - query: |
                SELECT 
                  month as time,
                  total_mrr as "Total MRR",
                  base_mrr as "Base MRR",
                  overage_mrr as "Overage MRR"
                FROM analytics.mrr_calculation
                WHERE month >= NOW() - INTERVAL '12 months'
                ORDER BY month
          
        - title: "Revenue by Channel Tier"
          type: piechart
          gridPos: {x: 12, y: 0, w: 12, h: 8}
          targets:
            - query: |
                SELECT 
                  performance_tier,
                  SUM(revenue_last_30d) as revenue
                FROM analytics.channel_performance
                GROUP BY performance_tier
```

### 2.2 Operations Dashboard

```yaml
dashboard:
  title: "YTEMPIRE Operations Dashboard"
  refresh: "10s"
  
  rows:
    - title: "System Health"
      panels:
        - title: "API Response Time"
          type: graph
          targets:
            - query: |
                SELECT 
                  time_bucket('1 minute', timestamp) as time,
                  AVG(response_time_ms) as "Average",
                  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as "P95"
                FROM api_logs
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                GROUP BY time
                ORDER BY time
          fieldConfig:
            defaults:
              unit: "ms"
              custom:
                drawStyle: "line"
                lineInterpolation: "smooth"
        
        - title: "YouTube Quota Usage"
          type: gauge
          targets:
            - query: "SELECT youtube_quota_used_today as value, 10000 as max FROM analytics.platform_performance"
          fieldConfig:
            defaults:
              unit: "short"
              thresholds:
                steps:
                  - value: 0
                    color: "green"
                  - value: 7000
                    color: "yellow"
                  - value: 9000
                    color: "red"
    
    - title: "Video Generation Pipeline"
      panels:
        - title: "Generation Success Rate"
          type: timeseries
          targets:
            - query: |
                SELECT 
                  date as time,
                  success_rate * 100 as "Success Rate %"
                FROM analytics.video_generation_metrics
                WHERE date >= NOW() - INTERVAL '7 days'
                ORDER BY date
        
        - title: "Cost Per Video Trend"
          type: timeseries
          targets:
            - query: |
                SELECT 
                  date as time,
                  avg_cost_per_video as "Average Cost",
                  3.00 as "Target Cost"
                FROM analytics.video_generation_metrics
                WHERE date >= NOW() - INTERVAL '30 days'
                ORDER BY date
```

### 2.3 Channel Performance Dashboard

```yaml
dashboard:
  title: "Channel Performance Analytics"
  
  variables:
    - name: "user_id"
      type: "query"
      query: "SELECT user_id as value, email as text FROM users WHERE status = 'active' ORDER BY email"
      multi: false
      
    - name: "time_range"
      type: "interval"
      options: ["7d", "30d", "90d", "180d", "365d"]
      default: "30d"
  
  rows:
    - title: "Channel Overview"
      panels:
        - title: "Channel Performance Matrix"
          type: "scatter"
          gridPos: {x: 0, y: 0, w: 24, h: 10}
          targets:
            - query: |
                SELECT 
                  channel_name,
                  views_last_30d as x,
                  revenue_last_30d as y,
                  current_subscribers as size,
                  performance_tier as color
                FROM analytics.channel_performance
                WHERE user_id = '$user_id'
          fieldConfig:
            defaults:
              custom:
                scatterPlot:
                  xAxis: "Views (30d)"
                  yAxis: "Revenue (30d)"
                  sizeField: "Subscribers"
                  colorField: "Tier"
    
    - title: "Video Performance"
      panels:
        - title: "Top Performing Videos"
          type: table
          targets:
            - query: |
                SELECT 
                  title,
                  views,
                  engagement_rate,
                  revenue_cents / 100.0 as revenue,
                  performance_tier,
                  published_at
                FROM analytics.video_performance_analysis
                WHERE user_id = '$user_id'
                  AND published_at >= NOW() - INTERVAL '$time_range'
                ORDER BY views DESC
                LIMIT 20
```

---

## 3. Real-time Analytics Implementation

### 3.1 Streaming Metrics Pipeline

```python
class RealTimeMetricsProcessor:
    """
    Process real-time metrics for dashboards
    """
    
    def __init__(self):
        self.redis = Redis()
        self.metrics_buffer = []
        self.flush_interval = 5  # seconds
        
    async def process_event(self, event: dict):
        """
        Process incoming events in real-time
        """
        event_type = event.get('type')
        
        if event_type == 'video_view':
            await self.update_view_metrics(event)
        elif event_type == 'video_generated':
            await self.update_generation_metrics(event)
        elif event_type == 'revenue_update':
            await self.update_revenue_metrics(event)
        elif event_type == 'api_call':
            await self.update_api_metrics(event)
    
    async def update_view_metrics(self, event: dict):
        """
        Update real-time view metrics
        """
        video_id = event['video_id']
        channel_id = event['channel_id']
        
        # Increment counters
        await self.redis.hincrby(f"views:today", channel_id, 1)
        await self.redis.hincrby(f"views:video:{video_id}", "total", 1)
        
        # Update hourly bucket
        hour_bucket = datetime.now().strftime("%Y-%m-%d:%H")
        await self.redis.hincrby(f"views:hourly:{hour_bucket}", channel_id, 1)
        
        # Calculate velocity (views per hour)
        video_age_hours = await self.get_video_age_hours(video_id)
        if video_age_hours > 0:
            velocity = await self.redis.hget(f"views:video:{video_id}", "total") / video_age_hours
            await self.redis.hset(f"velocity:current", video_id, velocity)
    
    async def get_dashboard_metrics(self) -> dict:
        """
        Get current metrics for dashboard
        """
        return {
            'current_hour_views': await self.get_current_hour_views(),
            'today_revenue': await self.get_today_revenue(),
            'active_generations': await self.get_active_generations(),
            'trending_videos': await self.get_trending_videos(),
            'system_health': await self.get_system_health()
        }
```

### 3.2 Alert Rules Configuration

```yaml
alert_rules:
  - name: "High Error Rate"
    query: |
      SELECT COUNT(*) as error_count
      FROM error_logs
      WHERE timestamp > NOW() - INTERVAL '5 minutes'
        AND severity IN ('error', 'critical')
    threshold: 50
    frequency: "5m"
    severity: "warning"
    notification:
      - slack: "#alerts"
      - email: "oncall@ytempire.com"
  
  - name: "YouTube Quota Critical"
    query: |
      SELECT youtube_quota_used_today
      FROM analytics.platform_performance
    threshold: 9000
    frequency: "1m"
    severity: "critical"
    notification:
      - slack: "#critical-alerts"
      - pagerduty: "youtube-quota"
  
  - name: "Revenue Target Miss"
    query: |
      SELECT 
        SUM(revenue_cents) / 100.0 as today_revenue,
        COUNT(DISTINCT user_id) * 333 as daily_target  -- $10k/month = $333/day per user
      FROM videos
      WHERE published_at >= CURRENT_DATE
    condition: "today_revenue < daily_target * 0.8"
    frequency: "1h"
    severity: "warning"
  
  - name: "Video Generation Failures"
    query: |
      SELECT 
        COUNT(CASE WHEN status = 'failed' THEN 1 END)::FLOAT / COUNT(*) as failure_rate
      FROM video_generation_log
      WHERE created_at > NOW() - INTERVAL '1 hour'
    threshold: 0.1  # 10% failure rate
    frequency: "15m"
    severity: "warning"
```

---

## 4. Data Export and Reporting

### 4.1 Automated Reports

```python
class AutomatedReportGenerator:
    """
    Generate automated reports for users
    """
    
    REPORT_TEMPLATES = {
        'daily_summary': {
            'schedule': '09:00',
            'recipients': 'user',
            'sections': [
                'revenue_summary',
                'channel_performance',
                'top_videos',
                'system_health'
            ]
        },
        'weekly_insights': {
            'schedule': 'monday 08:00',
            'recipients': 'user',
            'sections': [
                'weekly_trends',
                'performance_analysis',
                'optimization_opportunities',
                'competitor_insights'
            ]
        },
        'monthly_report': {
            'schedule': 'first_day 10:00',
            'recipients': 'user',
            'sections': [
                'financial_summary',
                'growth_metrics',
                'channel_analysis',
                'recommendations'
            ]
        }
    }
    
    async def generate_daily_summary(self, user_id: str) -> dict:
        """
        Generate daily summary report
        """
        report_data = {
            'user_id': user_id,
            'date': datetime.now().date(),
            'sections': {}
        }
        
        # Revenue Summary
        revenue_query = """
            SELECT 
                COUNT(DISTINCT v.video_id) as videos_published,
                SUM(v.revenue_cents) / 100.0 as total_revenue,
                AVG(v.revenue_cents) / 100.0 as avg_revenue_per_video,
                SUM(v.views) as total_views
            FROM videos v
            JOIN channels c ON v.channel_id = c.channel_id
            WHERE c.user_id = $1
              AND v.published_at >= CURRENT_DATE
        """
        
        report_data['sections']['revenue_summary'] = await self.db.fetchone(revenue_query, user_id)
        
        # Channel Performance
        channel_query = """
            SELECT 
                c.channel_name,
                COUNT(v.video_id) as videos_today,
                SUM(v.views) as views_today,
                SUM(v.revenue_cents) / 100.0 as revenue_today
            FROM channels c
            LEFT JOIN videos v ON c.channel_id = v.channel_id 
                AND v.published_at >= CURRENT_DATE
            WHERE c.user_id = $1
            GROUP BY c.channel_id, c.channel_name
            ORDER BY revenue_today DESC
        """
        
        report_data['sections']['channel_performance'] = await self.db.fetch(channel_query, user_id)
        
        return report_data
```

### 4.2 Export Formats

```python
class DataExporter:
    """
    Export data in various formats
    """
    
    EXPORT_FORMATS = ['csv', 'json', 'excel', 'pdf']
    
    async def export_analytics(self, user_id: str, date_range: dict, format: str) -> bytes:
        """
        Export analytics data
        """
        # Fetch data
        data = await self.fetch_analytics_data(user_id, date_range)
        
        if format == 'csv':
            return self.export_to_csv(data)
        elif format == 'json':
            return self.export_to_json(data)
        elif format == 'excel':
            return self.export_to_excel(data)
        elif format == 'pdf':
            return await self.export_to_pdf(data)
    
    def export_to_csv(self, data: dict) -> bytes:
        """
        Export data to CSV format
        """
        output = io.StringIO()
        
        # Channel data
        if 'channels' in data:
            writer = csv.DictWriter(output, fieldnames=data['channels'][0].keys())
            writer.writeheader()
            writer.writerows(data['channels'])
        
        return output.getvalue().encode('utf-8')
```

---

## 5. Performance Optimization

### 5.1 Query Optimization

```sql
-- Optimized indexes for common queries
CREATE INDEX idx_videos_published_user ON videos(published_at, channel_id) 
WHERE published_at >= CURRENT_DATE - INTERVAL '90 days';

CREATE INDEX idx_video_metrics_performance ON videos(views DESC, revenue_cents DESC) 
WHERE status = 'published';

CREATE INDEX idx_channel_analytics_date ON channel_analytics(channel_id, date DESC);

-- Materialized view refresh strategy
CREATE OR REPLACE FUNCTION refresh_analytics_views() RETURNS void AS $$
BEGIN
    -- Refresh in dependency order
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.video_generation_metrics;
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.channel_performance;
    
    -- Update last refresh timestamp
    UPDATE system_metadata 
    SET value = NOW()::TEXT 
    WHERE key = 'last_analytics_refresh';
END;
$$ LANGUAGE plpgsql;

-- Schedule hourly refresh
SELECT cron.schedule('refresh-analytics', '0 * * * *', 'SELECT refresh_analytics_views()');
```

### 5.2 Caching Strategy

```python
class AnalyticsCacheManager:
    """
    Manage analytics caching for performance
    """
    
    CACHE_CONFIGS = {
        'dashboard_metrics': {
            'ttl': 60,  # 1 minute
            'key_pattern': 'analytics:dashboard:{user_id}:{metric}'
        },
        'channel_performance': {
            'ttl': 300,  # 5 minutes
            'key_pattern': 'analytics:channel:{channel_id}:{timeframe}'
        },
        'revenue_calculations': {
            'ttl': 3600,  # 1 hour
            'key_pattern': 'analytics:revenue:{user_id}:{date}'
        }
    }
    
    async def get_or_compute(self, cache_key: str, compute_func, ttl: int = 300):
        """
        Get from cache or compute if missing
        """
        # Try cache first
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Compute if not in cache
        result = await compute_func()
        
        # Store in cache
        await self.redis.setex(
            cache_key,
            ttl,
            json.dumps(result, default=str)
        )
        
        return result
```

---

## Next Steps for Analytics Engineer

1. **Database Setup**:
   - Create all views and materialized views
   - Set up indexes for optimal performance
   - Configure refresh schedules
   - Implement partitioning for large tables

2. **Grafana Implementation**:
   - Import dashboard configurations
   - Set up data sources
   - Configure variables and filters
   - Create alert rules

3. **Real-time Pipeline**:
   - Implement Redis-based metrics collection
   - Set up WebSocket connections
   - Configure streaming aggregations
   - Build cache warming strategies

4. **Reporting System**:
   - Create report templates
   - Set up email delivery
   - Implement export functionality
   - Configure scheduling

5. **Testing & Validation**:
   - Verify metric calculations
   - Test dashboard performance
   - Validate alert thresholds
   - Ensure data accuracy

This document provides comprehensive analytics specifications for implementing the complete YTEMPIRE analytics system. Focus on building a scalable, performant solution that provides real-time insights while maintaining data accuracy.