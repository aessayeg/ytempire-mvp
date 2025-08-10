# YTEMPIRE Project Brief - Data Team

## Executive Summary

This brief defines the Data Team's responsibilities for YTEMPIRE's MVP - a **B2B SaaS platform** that enables **50 external users** to each manage **5 YouTube channels** (250 total channels across all users). As a 3-person team reporting to the VP of AI, we will build a lean data infrastructure using PostgreSQL, Redis, and N8N on local hardware. Our mission is to provide the data foundation that helps our users achieve $10,000+ monthly revenue within 90 days, while maintaining operational costs under $3 per video.

## Critical Clarification: What YTEMPIRE Is

### YTEMPIRE IS:
✅ **A B2B SaaS platform for external users**
✅ **A tool that helps 50 beta users manage their own YouTube channels**
✅ **A platform where each user operates 5 channels independently**
✅ **A service that automates YouTube management for entrepreneurs**

### YTEMPIRE IS NOT:
❌ **NOT a single entity operating 100+ channels internally**
❌ **NOT YTEMPIRE itself running YouTube channels**
❌ **NOT a content empire we operate ourselves**
❌ **NOT an internal content production system**

**Business Model**: We provide the platform; users operate their own channels using our automation.

## Project Scope & MVP Reality

### MVP Targets (12-Week Development)
- **Users**: 50 external beta users (digital entrepreneurs)
- **Channels**: 250 total (5 per user, not operated by us)
- **Videos**: ~500 daily across all users
- **Revenue Goal**: Each user achieves $10,000/month within 90 days
- **Automation**: 95% of YouTube operations automated
- **User Time**: <1 hour weekly management per user

### Data Volume Reality Check (Revised Estimates)
- **Data Ingestion**: ~5-10GB daily for 250 active channels
  - YouTube Analytics: ~2GB (metrics for 250 channels)
  - Video metadata: ~3GB (500 videos with descriptions, tags, etc.)
  - Thumbnail data: ~2GB
  - User activity logs: ~1GB
  - System metrics: ~1-2GB
- **Events**: ~500K-1M daily (more realistic for active platform)
  - API calls: ~100K
  - User actions: ~50K
  - Video processing events: ~100K
  - Analytics updates: ~250K
  - System events: ~100K
- **Storage Need**: 1-1.5TB for 90 days of data
- **API Calls**: ~3,000 YouTube API units daily (well under 10,000 limit)

## Complete Team Structure (17 People Total)

### Technical Organization
**Technical Leadership:**
- **CTO/Technical Director** - Overall technical strategy
  - **Backend Team Lead**
    - **API Developer Engineer** - Platform APIs
    - **Data Pipeline Engineer** - ETL processes
    - **Integration Specialist** - Third-party integrations
  - **Frontend Team Lead**
    - **React Engineer** - User dashboard
    - **Dashboard Specialist** - Analytics UI
    - **UI/UX Designer** - User experience
  - **Platform Ops Lead**
    - **DevOps Engineer** - Infrastructure
    - **Security Engineer** - Security & compliance
    - **QA Engineer** - Quality assurance

**AI Leadership:**
- **VP of AI** - AI strategy
  - **AI/ML Team Lead**
    - **ML Engineer** - Model development
  - **Data Team Lead** (Reports to VP of AI)
    - **Data Engineer** - Pipeline development
    - **Analytics Engineer** - BI and reporting

### Data Team Composition
**Total Data Team Members**: 3 people
- **Data Team Lead**: Strategy, coordination, architecture decisions
- **Data Engineer**: Pipeline development, data ingestion, storage
- **Analytics Engineer**: 
  - Primary: Dashboard development and maintenance
  - User-facing analytics and reporting
  - SQL query optimization for Grafana
  - Business metrics calculation and validation
  - User revenue tracking and insights
  - Performance monitoring dashboards
  - Cost analysis and ROI calculations

**Note**: The Analytics Engineer is one of the 3 data team members, not additional.

## Technology Stack (MVP Only)

### Confirmed MVP Technologies
```yaml
# ACTUAL MVP STACK - NOTHING ELSE
databases:
  primary: PostgreSQL 15 with TimescaleDB
  cache: Redis 7.2
  
orchestration:
  workflow: N8N
  scheduling: N8N built-in scheduler
  
apis:
  youtube_data: YouTube Data API v3  # For video/channel/playlist information
  youtube_analytics: YouTube Analytics API v2  # For metrics (v2 is current, not v3)
  ai: OpenAI GPT-4
  voice: ElevenLabs
  payment: Stripe
  
  # API Clarification:
  # YouTube has TWO separate APIs:
  # 1. YouTube Data API (currently v3) - for content management
  # 2. YouTube Analytics API (currently v2) - for metrics/statistics
  # Both are correct and current versions as of 2025
  
analytics:
  dashboards: Grafana (shared with Platform Ops)
  queries: PostgreSQL native SQL
  
monitoring:
  metrics: Prometheus + Grafana
  logs: Local file system with rotation
  
storage:
  local: 4TB NVMe SSD
  backup: External drives + cloud backup (S3)
```

### NOT Using in MVP (Future Considerations Only)
❌ **Apache Kafka** - No streaming needed for 250 channels
❌ **Apache Spark** - PostgreSQL handles our scale fine
❌ **Snowflake** - Too expensive and unnecessary
❌ **DBT Cloud** - Using simple SQL scripts
❌ **Tableau** - Grafana is sufficient
❌ **Kubernetes** - Local deployment only
❌ **Multi-region** - Single location for MVP

## Hardware Specifications

### Actual Available Hardware
- **CPU**: AMD Ryzen 9 7950X (16 cores, 32 threads)
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **RAM**: 128GB DDR5
- **Storage**: 4TB NVMe SSD
- **Network**: 1Gbps connection

### Resource Allocation
- PostgreSQL: 32GB RAM
- Redis: 8GB RAM
- N8N: 4GB RAM
- OS & Services: 20GB RAM
- Buffer/Cache: 64GB RAM

## Core Data Responsibilities

### 1. Multi-User Data Architecture

Since YTEMPIRE is a **B2B SaaS platform**, our data architecture must support multiple users:

```sql
-- User account management
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    subscription_tier VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    target_revenue DECIMAL(10,2) DEFAULT 10000.00  -- $10K target
);

-- User's channels (5 per user)
CREATE TABLE channels (
    channel_id VARCHAR(50) PRIMARY KEY,  -- YouTube channel ID
    user_id UUID REFERENCES users(user_id),
    channel_name VARCHAR(255),
    niche VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    CONSTRAINT max_channels_per_user CHECK (
        (SELECT COUNT(*) FROM channels WHERE user_id = channels.user_id) <= 5
    )
);

-- Metrics partitioned by user
CREATE TABLE analytics.youtube_metrics (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    channel_id VARCHAR(50) NOT NULL,
    video_id VARCHAR(50),
    date DATE NOT NULL,
    views BIGINT DEFAULT 0,
    watch_time_minutes DECIMAL(12,2),
    revenue_usd DECIMAL(10,2) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (channel_id) REFERENCES channels(channel_id),
    UNIQUE(channel_id, date)
);

-- User-specific dashboards
CREATE TABLE analytics.user_dashboard_config (
    user_id UUID PRIMARY KEY REFERENCES users(user_id),
    dashboard_preferences JSONB,
    notification_settings JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2. YouTube API Integration (Clarified Versions)

```python
class YouTubeDataCollection:
    """
    Handles both YouTube APIs correctly
    """
    
    def __init__(self):
        # YouTube Data API v3 - for video/channel/playlist information
        # This is the current version for content management
        self.youtube_data = build('youtube', 'v3', credentials=self.creds)
        
        # YouTube Analytics API v2 - for metrics and statistics
        # v2 is the current version for analytics (there is no v3 for Analytics)
        self.youtube_analytics = build('youtubeAnalytics', 'v2', credentials=self.creds)
        
        self.db = psycopg2.connect(
            dbname='ytempire',
            user='data_user',
            host='localhost'
        )
    
    def fetch_user_channel_metrics(self, user_id):
        """
        Fetch metrics for all channels belonging to a user
        """
        # Get user's channels (max 5)
        channels = self.get_user_channels(user_id)
        
        for channel in channels:
            # Use YouTube Analytics API v2 for metrics
            analytics_response = self.youtube_analytics.reports().query(
                ids=f'channel=={channel["channel_id"]}',
                startDate='7daysAgo',
                endDate='today',
                metrics='views,estimatedMinutesWatched,subscribersGained,estimatedRevenue',
                dimensions='day'
            ).execute()
            
            # Use YouTube Data API v3 for video details
            videos_response = self.youtube_data.search().list(
                channelId=channel['channel_id'],
                part='id,snippet',
                order='date',
                maxResults=10
            ).execute()
            
            self.store_metrics(user_id, channel['channel_id'], analytics_response, videos_response)
```

### 3. N8N Workflows for Multi-User Platform

```javascript
// N8N Workflow: Multi-User Data Sync
{
  "name": "Multi-User YouTube Sync",
  "nodes": [
    {
      "name": "Every 15 Minutes",
      "type": "n8n-nodes-base.scheduleTrigger",
      "parameters": {
        "rule": {
          "interval": [{ "field": "minutes", "value": 15 }]
        }
      }
    },
    {
      "name": "Get Active Users",
      "type": "n8n-nodes-base.postgres",
      "parameters": {
        "query": "SELECT user_id, email FROM users WHERE subscription_tier != 'inactive'"
      }
    },
    {
      "name": "Get User Channels",
      "type": "n8n-nodes-base.postgres",
      "parameters": {
        "query": "SELECT channel_id FROM channels WHERE user_id = '{{$json.user_id}}' AND is_active = true"
      }
    },
    {
      "name": "Fetch YouTube Analytics",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "https://youtubeanalytics.googleapis.com/v2/reports",
        "authentication": "oAuth2",
        "method": "GET"
      }
    },
    {
      "name": "Store User Metrics",
      "type": "n8n-nodes-base.postgres",
      "parameters": {
        "operation": "insert",
        "table": "analytics.youtube_metrics",
        "columns": "user_id,channel_id,date,views,revenue_usd"
      }
    }
  ]
}
```

### 4. User-Specific Analytics & Dashboards

```sql
-- User revenue tracking toward $10K goal
CREATE VIEW analytics.user_revenue_progress AS
SELECT 
    u.user_id,
    u.email,
    COUNT(DISTINCT c.channel_id) as active_channels,
    DATE_TRUNC('month', CURRENT_DATE) as current_month,
    COALESCE(SUM(m.revenue_usd), 0) as monthly_revenue,
    10000.00 as target_revenue,
    LEAST(100, (COALESCE(SUM(m.revenue_usd), 0) / 10000.00) * 100) as progress_percentage,
    CASE 
        WHEN COALESCE(SUM(m.revenue_usd), 0) >= 10000 THEN 'TARGET_ACHIEVED'
        WHEN COALESCE(SUM(m.revenue_usd), 0) >= 5000 THEN 'ON_TRACK'
        WHEN COALESCE(SUM(m.revenue_usd), 0) >= 1000 THEN 'NEEDS_OPTIMIZATION'
        ELSE 'JUST_STARTED'
    END as status
FROM users u
LEFT JOIN channels c ON u.user_id = c.user_id AND c.is_active = true
LEFT JOIN analytics.youtube_metrics m ON c.channel_id = m.channel_id 
    AND m.date >= DATE_TRUNC('month', CURRENT_DATE)
GROUP BY u.user_id, u.email;

-- Per-user cost tracking
CREATE VIEW analytics.user_cost_analysis AS
SELECT 
    u.user_id,
    u.email,
    COUNT(DISTINCT v.video_id) as videos_created,
    COUNT(DISTINCT v.video_id) * 3.00 as total_cost,  -- $3 per video
    COALESCE(SUM(m.revenue_usd), 0) as total_revenue,
    COALESCE(SUM(m.revenue_usd), 0) - (COUNT(DISTINCT v.video_id) * 3.00) as profit,
    CASE 
        WHEN COUNT(DISTINCT v.video_id) > 0 
        THEN (COALESCE(SUM(m.revenue_usd), 0) / (COUNT(DISTINCT v.video_id) * 3.00))
        ELSE 0 
    END as roi
FROM users u
LEFT JOIN channels c ON u.user_id = c.user_id
LEFT JOIN analytics.videos v ON c.channel_id = v.channel_id
LEFT JOIN analytics.youtube_metrics m ON v.video_id = m.video_id
WHERE v.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY u.user_id, u.email;
```

### 5. Analytics Engineer Responsibilities (Detailed)

```python
class AnalyticsEngineerTasks:
    """
    Specific responsibilities for the Analytics Engineer role
    """
    
    def build_user_dashboards(self):
        """
        Create and maintain Grafana dashboards for each user
        """
        dashboards = {
            'revenue_tracker': {
                'purpose': 'Track progress toward $10K/month goal',
                'refresh_rate': '5 minutes',
                'panels': [
                    'current_month_revenue',
                    'daily_revenue_trend',
                    'channel_performance_comparison',
                    'revenue_growth_rate',
                    'days_to_10k_projection'
                ]
            },
            'channel_analytics': {
                'purpose': 'Per-channel performance metrics',
                'refresh_rate': '15 minutes',
                'panels': [
                    'views_by_channel',
                    'ctr_trends',
                    'retention_analysis',
                    'best_performing_videos',
                    'upload_schedule_effectiveness'
                ]
            },
            'cost_analysis': {
                'purpose': 'ROI and profitability tracking',
                'refresh_rate': 'hourly',
                'panels': [
                    'cost_per_video',
                    'revenue_per_video',
                    'profit_margins',
                    'roi_by_channel',
                    'break_even_analysis'
                ]
            }
        }
        return dashboards
    
    def optimize_queries(self):
        """
        Ensure all dashboard queries run in <2 seconds
        """
        optimization_tasks = [
            'Create materialized views for complex aggregations',
            'Implement query result caching in Redis',
            'Add appropriate indexes for common filter patterns',
            'Partition large tables by date',
            'Pre-aggregate hourly and daily summaries'
        ]
        return optimization_tasks
    
    def generate_user_reports(self):
        """
        Automated reporting for users
        """
        reports = {
            'daily_summary': {
                'schedule': '9:00 AM user timezone',
                'content': [
                    'Yesterday revenue',
                    'Top performing video',
                    'Channels needing attention',
                    'Progress toward $10K goal'
                ]
            },
            'weekly_insights': {
                'schedule': 'Monday 8:00 AM',
                'content': [
                    'Week-over-week growth',
                    'Best practices from top channels',
                    'Optimization recommendations',
                    'Upcoming opportunities'
                ]
            },
            'monthly_statement': {
                'schedule': 'First of month',
                'content': [
                    'Total revenue',
                    'Total costs',
                    'Net profit',
                    'Channel rankings',
                    'Growth trajectory'
                ]
            }
        }
        return reports
    
    def monitor_data_quality(self):
        """
        Ensure data accuracy for user trust
        """
        quality_checks = [
            'Revenue reconciliation with YouTube Analytics',
            'View count verification',
            'Missing data detection',
            'Anomaly detection in metrics',
            'User data isolation verification'
        ]
        return quality_checks
```

### Analytics Engineer Key SQL Queries

```sql
-- User Success Tracking (Primary Dashboard)
CREATE MATERIALIZED VIEW analytics.user_success_metrics AS
WITH monthly_progress AS (
    SELECT 
        u.user_id,
        u.email,
        DATE_TRUNC('month', m.date) as month,
        SUM(m.revenue_usd) as monthly_revenue,
        COUNT(DISTINCT m.video_id) as videos_published,
        AVG(m.views) as avg_views_per_video,
        10000.00 as target_revenue
    FROM users u
    JOIN channels c ON u.user_id = c.user_id
    JOIN analytics.youtube_metrics m ON c.channel_id = m.channel_id
    WHERE m.date >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '3 months'
    GROUP BY u.user_id, u.email, DATE_TRUNC('month', m.date)
),
growth_metrics AS (
    SELECT 
        user_id,
        month,
        monthly_revenue,
        LAG(monthly_revenue) OVER (PARTITION BY user_id ORDER BY month) as prev_month_revenue,
        CASE 
            WHEN LAG(monthly_revenue) OVER (PARTITION BY user_id ORDER BY month) > 0
            THEN ((monthly_revenue - LAG(monthly_revenue) OVER (PARTITION BY user_id ORDER BY month)) / 
                  LAG(monthly_revenue) OVER (PARTITION BY user_id ORDER BY month)) * 100
            ELSE 0
        END as growth_rate
    FROM monthly_progress
)
SELECT 
    mp.*,
    gm.growth_rate,
    CASE 
        WHEN mp.monthly_revenue >= 10000 THEN 'GOAL_ACHIEVED'
        WHEN mp.monthly_revenue >= 7500 THEN 'ALMOST_THERE'
        WHEN mp.monthly_revenue >= 5000 THEN 'ON_TRACK'
        WHEN mp.monthly_revenue >= 2500 THEN 'GROWING'
        ELSE 'GETTING_STARTED'
    END as success_tier,
    CASE 
        WHEN gm.growth_rate > 0 AND mp.monthly_revenue > 0
        THEN CEIL((10000 - mp.monthly_revenue) / (mp.monthly_revenue * (gm.growth_rate/100)))
        ELSE NULL
    END as months_to_goal
FROM monthly_progress mp
JOIN growth_metrics gm ON mp.user_id = gm.user_id AND mp.month = gm.month
WHERE mp.month = DATE_TRUNC('month', CURRENT_DATE);

-- Refresh every hour
CREATE INDEX idx_user_success_metrics_user ON analytics.user_success_metrics(user_id);
```

## Development Timeline (12 Weeks to MVP)

### Phase 1: Foundation (Weeks 1-3)
- **Week 1**: 
  - PostgreSQL 15 + TimescaleDB setup
  - User and channel schema creation
  - Basic authentication system
- **Week 2**: 
  - N8N platform installation
  - YouTube Data API v3 integration
  - YouTube Analytics API v2 integration
- **Week 3**: 
  - Multi-user data ingestion pipeline
  - User isolation and security

### Phase 2: Core Development (Weeks 4-6)
- **Week 4**: 
  - Complete data pipeline for 5 test users
  - Each test user with 2-3 channels
- **Week 5**: 
  - User-specific dashboards in Grafana
  - Revenue tracking implementation
- **Week 6**: 
  - **Milestone: Internal Alpha (5 users, 25 channels)**

### Phase 3: Scaling (Weeks 7-9)
- **Week 7**: 
  - Scale testing to 20 users (100 channels)
  - Performance optimization
- **Week 8**: 
  - Complete analytics suite
  - Cost tracking per user
- **Week 9**: 
  - Data quality framework
  - Monitoring and alerts

### Phase 4: Production Ready (Weeks 10-12)
- **Week 10**: 
  - **Milestone: Investor Demo (20 users live)**
- **Week 11**: 
  - Scale to 50 users (250 channels)
  - Final testing and optimization
- **Week 12**: 
  - **Milestone: Private Beta Launch**

## Budget Allocation

### Total MVP Budget: $200,000

**Data Team Allocation: $30,000**
- **Infrastructure & Tools**: $10,000
  - N8N licenses: $2,000
  - Grafana enterprise features: $3,000
  - Backup storage (S3): $2,000
  - Development tools: $3,000
- **Operational Costs (3 months)**: $15,000
  - YouTube API quota buffer: $5,000
  - Cloud backup: $3,000
  - Monitoring services: $2,000
  - Contingency: $5,000
- **Documentation & Training**: $5,000

### Cost Per Video Target (Complete Breakdown)
- **Data infrastructure share**: <$0.65
  - Compute resources: $0.30
  - Storage: $0.15
  - API quota allocation: $0.10
  - Monitoring/backup: $0.10
- **AI/ML costs** (managed by AI team): ~$1.50
  - GPT-4 usage: $0.50
  - ElevenLabs voice: $0.30
  - Video processing: $0.40
  - Other AI services: $0.30
- **Platform overhead** (managed by Platform Ops): ~$0.85
  - Infrastructure: $0.35
  - Bandwidth: $0.25
  - Security/compliance: $0.25
- **Total platform cost: <$3.00 per video ✓**

## Success Metrics

### User-Focused Metrics (Primary)
- ✅ 50 users successfully onboarded
- ✅ Each user operating 5 channels
- ✅ Average user reaching $5,000/month by day 45
- ✅ 30% of users achieving $10,000/month by day 90
- ✅ User data dashboard load time <2 seconds

### Platform Metrics (Supporting)
- ✅ 250 total channels ingesting data
- ✅ ~500 videos processed daily
- ✅ 99.5% data pipeline uptime
- ✅ <15 minute data freshness
- ✅ Zero data breaches or user data leaks

### Data Quality Metrics
- ✅ 100% user data isolation (no cross-contamination)
- ✅ >95% data completeness
- ✅ <1% data discrepancy with YouTube Analytics
- ✅ All users receiving daily updates

## Risk Management

### Primary Risks (MVP-Specific)

1. **User Data Isolation**
   - **Risk**: Data leakage between users
   - **Mitigation**: Row-level security, user_id validation
   - **Owner**: Data Engineer

2. **YouTube API Quotas (Shared Across Users)**
   - **Risk**: 50 users × 5 channels = 250 channels exceeding quotas
   - **Mitigation**: Intelligent batching, caching, quota monitoring
   - **Owner**: Data Team Lead

3. **Uneven User Load**
   - **Risk**: Some users generating 100+ videos while others generate few
   - **Mitigation**: User-level quotas, fair scheduling
   - **Owner**: Analytics Engineer

4. **User Revenue Tracking**
   - **Risk**: Incorrect revenue attribution affecting user trust
   - **Mitigation**: Daily reconciliation with YouTube Analytics
   - **Owner**: Analytics Engineer

## Communication & Reporting

### Team Meetings
- **Daily**: 15-minute Data Team standup
- **Weekly**: Cross-team sync with Backend and Frontend teams
- **Weekly**: User metrics review with VP of AI
- **Bi-weekly**: Platform performance review with CTO

### User-Facing Deliverables
- Real-time revenue dashboard per user
- Daily email reports on channel performance
- Weekly optimization recommendations
- Monthly revenue summaries

## Conclusion

The Data Team's mission is clear: build a data infrastructure that supports **50 external users** in achieving their **$10,000/month revenue goals** through our **B2B SaaS platform**. We are NOT building an internal content empire but rather empowering entrepreneurs to build their own.

Every technical decision must support our multi-user architecture:
- User data isolation ✓
- Per-user analytics ✓
- Scalable to 250 channels across 50 users ✓
- Built on PostgreSQL + Redis + N8N ✓
- Deployed on local hardware ✓
- Within $30,000 budget ✓

**Remember**: 
- We're building FOR users, not operating channels ourselves
- Each user is independent with their own 5 channels
- Success means our users achieve $10K/month
- Simplicity and reliability over complexity
- MVP first, enterprise features later

The path is clear: PostgreSQL for data, N8N for orchestration, Grafana for visualization, all running on local hardware to serve 50 paying users. No Kafka, no Snowflake, no complexity - just solid, working data infrastructure that enables user success.