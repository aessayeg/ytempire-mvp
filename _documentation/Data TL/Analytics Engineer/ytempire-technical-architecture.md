# YTEMPIRE Analytics Engineer Documentation
## 2. TECHNICAL ARCHITECTURE

**Document Version**: 1.0  
**Date**: January 2025  
**Status**: CONSOLIDATED - PRODUCTION READY  
**Purpose**: Complete technical architecture for Analytics Engineering

---

## 2.1 Technology Stack

### Core Technologies

#### Confirmed MVP Technologies
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
  dashboards: Grafana 10.2.3 (shared with Platform Ops)
  queries: PostgreSQL native SQL
  notebooks: Jupyter (ad-hoc analysis only)
  sql_ide: DBeaver/DataGrip
  
monitoring:
  metrics: Prometheus + Grafana
  logs: Local file system with rotation
  
storage:
  local: 4TB NVMe SSD
  backup: External drives + cloud backup (S3)
  search: PostgreSQL Full Text Search

programming:
  primary: SQL (PostgreSQL dialect)
  secondary: Python 3.11+ (pandas, psycopg2)
  visualization: Grafana query language
```

#### NOT Using in MVP (Future Considerations Only)
- ❌ **Apache Kafka** - No streaming needed for 250 channels
- ❌ **Apache Spark** - PostgreSQL handles our scale fine
- ❌ **Snowflake** - Too expensive and unnecessary
- ❌ **DBT Cloud** - Using simple SQL scripts and local DBT only
- ❌ **Tableau** - Grafana is sufficient
- ❌ **Kubernetes** - Local deployment only
- ❌ **Multi-region** - Single location for MVP

### Development Environment

```bash
# Your local setup requirements
- MacBook Pro or equivalent
- Docker Desktop for local PostgreSQL/Redis
- Git for version control
- VS Code with SQL extensions
- Access to production read replica

# Setup local environment
git clone https://github.com/ytempire/analytics.git
cd analytics

# Install dependencies
pip install psycopg2-binary==2.9.9
pip install pandas==2.1.4
pip install sqlalchemy==2.0.23
pip install redis==5.0.1
pip install grafana-api==1.0.3

# PostgreSQL client tools
brew install postgresql@15
brew install pgcli  # Better PostgreSQL CLI

# Development tools
pip install jupyter==1.0.0  # For ad-hoc analysis
pip install sqlfluff==2.3.0  # SQL linter
pip install pre-commit==3.3.0  # Code quality
```

### Infrastructure Allocation

#### Hardware Specifications
- **CPU**: AMD Ryzen 9 7950X (16 cores, 32 threads)
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **RAM**: 128GB DDR5
- **Storage**: 4TB NVMe SSD
- **Network**: 1Gbps connection

#### Resource Allocation
- PostgreSQL: 32GB RAM
- Redis: 8GB RAM
- N8N: 4GB RAM
- Grafana: 2GB RAM
- OS & Services: 20GB RAM
- Buffer/Cache: 64GB RAM

---

## 2.2 Data Architecture

### Database Schemas

#### Multi-User Data Architecture

Since YTEMPIRE is a B2B SaaS platform, our data architecture must support multiple users:

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- User account management
CREATE SCHEMA IF NOT EXISTS users;
CREATE TABLE users.accounts (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    subscription_tier VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    target_revenue DECIMAL(10,2) DEFAULT 10000.00,  -- $10K target
    is_active BOOLEAN DEFAULT true
);

-- User's channels (5 per user)
CREATE TABLE users.channels (
    channel_id VARCHAR(50) PRIMARY KEY,  -- YouTube channel ID
    user_id UUID REFERENCES users.accounts(user_id),
    channel_name VARCHAR(255),
    niche VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    CONSTRAINT max_channels_per_user CHECK (
        (SELECT COUNT(*) FROM users.channels WHERE user_id = channels.user_id) <= 5
    )
);

-- Create analytics schema
CREATE SCHEMA IF NOT EXISTS analytics;
SET search_path TO analytics, public;

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
    FOREIGN KEY (channel_id) REFERENCES users.channels(channel_id),
    UNIQUE(channel_id, date)
);

-- User-specific dashboards
CREATE TABLE analytics.user_dashboard_config (
    user_id UUID PRIMARY KEY REFERENCES users.accounts(user_id),
    dashboard_preferences JSONB,
    notification_settings JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Data Models (Fact/Dimension)

#### Dimensional Modeling Architecture

```sql
-- DIMENSIONAL MODELING FOLLOWING KIMBALL METHODOLOGY
-- Three-layer approach: Staging → Intermediate → Presentation

-- ================================================
-- DIMENSION TABLES
-- ================================================

-- Channel Dimension
CREATE TABLE analytics.dim_channel (
    channel_key SERIAL PRIMARY KEY,
    channel_id VARCHAR(50) NOT NULL UNIQUE,
    user_id UUID NOT NULL,
    channel_name VARCHAR(255),
    channel_handle VARCHAR(100),
    niche VARCHAR(100),
    sub_niche VARCHAR(100),
    target_audience JSONB,
    content_strategy JSONB,
    monetization_status VARCHAR(50),
    created_date DATE,
    is_active BOOLEAN DEFAULT true,
    -- SCD Type 2 fields
    valid_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_to TIMESTAMP DEFAULT '9999-12-31',
    is_current BOOLEAN DEFAULT true
);

-- Video Dimension
CREATE TABLE analytics.dim_video (
    video_key SERIAL PRIMARY KEY,
    video_id VARCHAR(50) NOT NULL UNIQUE,
    channel_id VARCHAR(50) NOT NULL,
    video_title VARCHAR(500),
    video_description TEXT,
    video_tags TEXT[],
    duration_seconds INTEGER,
    content_type VARCHAR(50), -- 'shorts', 'standard', 'long_form'
    production_method VARCHAR(50), -- 'ai_generated', 'manual', 'hybrid'
    thumbnail_url TEXT,
    published_at TIMESTAMP,
    is_monetized BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true
);

-- Date Dimension (pre-populated)
CREATE TABLE analytics.dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date DATE NOT NULL UNIQUE,
    year INTEGER,
    quarter INTEGER,
    month INTEGER,
    week INTEGER,
    day_of_month INTEGER,
    day_of_week INTEGER,
    day_name VARCHAR(20),
    month_name VARCHAR(20),
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER
);

-- User Dimension
CREATE TABLE analytics.dim_user (
    user_key SERIAL PRIMARY KEY,
    user_id UUID NOT NULL UNIQUE,
    email VARCHAR(255),
    subscription_tier VARCHAR(50),
    signup_date DATE,
    first_video_date DATE,
    total_channels INTEGER DEFAULT 0,
    total_videos INTEGER DEFAULT 0,
    lifetime_revenue DECIMAL(12,2) DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    churn_date DATE,
    -- SCD Type 2 fields
    valid_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_to TIMESTAMP DEFAULT '9999-12-31',
    is_current BOOLEAN DEFAULT true
);

-- ================================================
-- FACT TABLES
-- ================================================

-- Main fact table for video performance
CREATE TABLE analytics.fact_video_performance (
    id BIGSERIAL PRIMARY KEY,
    video_key INTEGER REFERENCES analytics.dim_video(video_key),
    channel_key INTEGER REFERENCES analytics.dim_channel(channel_key),
    user_key INTEGER REFERENCES analytics.dim_user(user_key),
    date_key INTEGER REFERENCES analytics.dim_date(date_key),
    
    -- Time-specific snapshot
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
    profit_cents INTEGER GENERATED ALWAYS AS (
        actual_revenue_cents - cost_cents
    ) STORED,
    
    -- YouTube specific metrics
    impressions BIGINT DEFAULT 0,
    click_through_rate DECIMAL(5,4) DEFAULT 0,
    average_percentage_viewed DECIMAL(5,2) DEFAULT 0,
    subscriber_gained INTEGER DEFAULT 0,
    subscriber_lost INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Channel daily aggregates fact table
CREATE TABLE analytics.fact_channel_daily (
    id BIGSERIAL PRIMARY KEY,
    channel_key INTEGER REFERENCES analytics.dim_channel(channel_key),
    user_key INTEGER REFERENCES analytics.dim_user(user_key),
    date_key INTEGER REFERENCES analytics.dim_date(date_key),
    
    -- Production metrics
    videos_published INTEGER DEFAULT 0,
    total_duration_seconds INTEGER DEFAULT 0,
    
    -- Engagement metrics
    total_views BIGINT DEFAULT 0,
    total_likes INTEGER DEFAULT 0,
    total_comments INTEGER DEFAULT 0,
    total_shares INTEGER DEFAULT 0,
    avg_engagement_rate DECIMAL(5,4) DEFAULT 0,
    
    -- Financial metrics
    revenue_cents INTEGER DEFAULT 0,
    cost_cents INTEGER DEFAULT 0,
    profit_cents INTEGER DEFAULT 0,
    
    -- Subscriber metrics
    subscriber_count INTEGER DEFAULT 0,
    subscriber_gained INTEGER DEFAULT 0,
    subscriber_lost INTEGER DEFAULT 0,
    
    -- Calculated metrics
    revenue_per_view_cents DECIMAL(10,4) DEFAULT 0,
    cost_per_video_cents DECIMAL(10,2) DEFAULT 0,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Cost tracking fact table
CREATE TABLE analytics.fact_costs (
    id BIGSERIAL PRIMARY KEY,
    user_key INTEGER REFERENCES analytics.dim_user(user_key),
    date_key INTEGER REFERENCES analytics.dim_date(date_key),
    
    cost_category VARCHAR(50), -- 'ai', 'api', 'storage', 'compute'
    cost_subcategory VARCHAR(100),
    
    -- Cost details
    quantity DECIMAL(12,4),
    unit_cost_cents INTEGER,
    total_cost_cents INTEGER,
    
    -- Attribution
    video_id VARCHAR(50),
    channel_id VARCHAR(50),
    
    -- Metadata
    cost_timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Revenue attribution fact table
CREATE TABLE analytics.fact_revenue (
    id BIGSERIAL PRIMARY KEY,
    video_key INTEGER REFERENCES analytics.dim_video(video_key),
    channel_key INTEGER REFERENCES analytics.dim_channel(channel_key),
    user_key INTEGER REFERENCES analytics.dim_user(user_key),
    date_key INTEGER REFERENCES analytics.dim_date(date_key),
    
    -- Revenue sources
    ad_revenue_cents INTEGER DEFAULT 0,
    sponsorship_revenue_cents INTEGER DEFAULT 0,
    affiliate_revenue_cents INTEGER DEFAULT 0,
    membership_revenue_cents INTEGER DEFAULT 0,
    super_chat_revenue_cents INTEGER DEFAULT 0,
    
    -- Total
    total_revenue_cents INTEGER GENERATED ALWAYS AS (
        ad_revenue_cents + sponsorship_revenue_cents + affiliate_revenue_cents + 
        membership_revenue_cents + super_chat_revenue_cents
    ) STORED,
    
    -- Attribution metrics
    attributed_views BIGINT DEFAULT 0,
    conversion_rate DECIMAL(5,4) DEFAULT 0,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### TimescaleDB Configuration

```sql
-- ================================================
-- TIMESCALEDB CONFIGURATION
-- ================================================

-- Convert fact tables to hypertables
SELECT create_hypertable('analytics.fact_video_performance', 'timestamp');
SELECT create_hypertable('analytics.fact_channel_daily', 'created_at');
SELECT create_hypertable('analytics.fact_costs', 'cost_timestamp');
SELECT create_hypertable('analytics.fact_revenue', 'created_at');

-- Create continuous aggregates for real-time rollups
CREATE MATERIALIZED VIEW analytics.video_performance_hourly
WITH (timescaledb.continuous) AS
SELECT
    video_key,
    channel_key,
    user_key,
    time_bucket('1 hour', timestamp) AS hour,
    
    -- Aggregated metrics
    MAX(views) as views,
    MAX(likes) as likes,
    MAX(comments) as comments,
    AVG(engagement_rate) as avg_engagement_rate,
    SUM(actual_revenue_cents) as revenue_cents,
    SUM(cost_cents) as cost_cents
    
FROM analytics.fact_video_performance
GROUP BY video_key, channel_key, user_key, hour
WITH NO DATA;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('analytics.video_performance_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- Compression policy for older data
ALTER TABLE analytics.fact_video_performance SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'channel_key',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('analytics.fact_video_performance', INTERVAL '7 days');

-- Retention policy (keep 90 days of detailed data)
SELECT add_retention_policy('analytics.fact_video_performance', INTERVAL '90 days');

-- Create indexes for common query patterns
CREATE INDEX idx_video_performance_channel_time 
ON analytics.fact_video_performance(channel_key, timestamp DESC);

CREATE INDEX idx_video_performance_user_time 
ON analytics.fact_video_performance(user_key, timestamp DESC);

CREATE INDEX idx_channel_daily_user_date 
ON analytics.fact_channel_daily(user_key, date_key);

CREATE INDEX idx_costs_user_category 
ON analytics.fact_costs(user_key, cost_category, cost_timestamp DESC);

CREATE INDEX idx_revenue_channel_date 
ON analytics.fact_revenue(channel_key, date_key);

-- Optimize chunk size for our workload (1 day chunks)
SELECT set_chunk_time_interval('analytics.fact_video_performance', INTERVAL '1 day');
```

---

## 2.3 Analytics Layer

### DBT Project Structure (Local DBT Only)

```yaml
# dbt_project.yml
name: 'ytempire_analytics'
version: '1.0.0'
profile: 'ytempire'

model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["data"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

target-path: "target"
clean-targets:
  - "target"
  - "dbt_packages"

models:
  ytempire_analytics:
    staging:
      +materialized: view
      +schema: staging
      youtube:
        +tags: ['daily', 'youtube']
      social:
        +tags: ['hourly', 'social']
    
    intermediate:
      +materialized: table
      +schema: intermediate
      content:
        +tags: ['daily']
      performance:
        +tags: ['hourly']
    
    marts:
      +materialized: table
      +schema: analytics
      executive:
        +materialized: incremental
        +unique_key: 'id'
        +on_schema_change: 'sync_all_columns'
      content:
        +post-hook: "GRANT SELECT ON {{ this }} TO reporter"
      finance:
        +tags: ['finance', 'sensitive']
```

#### DBT Model Examples

```sql
-- models/staging/youtube/stg_youtube__videos.sql
{{
  config(
    materialized='view',
    tags=['youtube', 'staging']
  )
}}

WITH source AS (
    SELECT *
    FROM {{ source('youtube_raw', 'videos') }}
    WHERE deleted_at IS NULL
),

renamed AS (
    SELECT
        -- Primary Key
        video_id::VARCHAR AS video_id,
        
        -- Foreign Keys
        channel_id::VARCHAR AS channel_id,
        user_id::UUID AS user_id,
        
        -- Descriptive Fields
        title::VARCHAR(500) AS video_title,
        description::TEXT AS video_description,
        tags::ARRAY AS video_tags,
        
        -- Metrics
        duration_seconds::INTEGER AS duration_seconds,
        
        -- Timestamps
        published_at::TIMESTAMP AS published_at,
        created_at::TIMESTAMP AS created_at,
        updated_at::TIMESTAMP AS updated_at,
        
        -- Calculated Fields
        CASE 
            WHEN duration_seconds < 60 THEN 'shorts'
            WHEN duration_seconds < 600 THEN 'standard'
            ELSE 'long_form'
        END AS content_type,
        
        DATE_TRUNC('day', published_at) AS published_date,
        EXTRACT(HOUR FROM published_at) AS published_hour,
        
        -- Data Quality
        IFF(video_id IS NOT NULL AND channel_id IS NOT NULL, TRUE, FALSE) AS is_valid
        
    FROM source
)

SELECT * FROM renamed
WHERE is_valid = TRUE
```

### Semantic Layer

```python
# analytics/semantic_layer.py
"""
Semantic layer for business-friendly data access
"""

class SemanticLayer:
    """
    Define business-friendly data model
    """
    def __init__(self):
        self.connection = PostgreSQLConnection()
        
    def create_semantic_model(self):
        """
        Define semantic model for self-service analytics
        """
        return {
            'cubes': [{
                'name': 'VideoPerformance',
                'sql_table': 'analytics.fact_video_performance',
                
                'measures': {
                    'total_views': {
                        'sql': 'SUM(views)',
                        'type': 'number',
                        'description': 'Total video views'
                    },
                    'engagement_rate': {
                        'sql': 'AVG(engagement_rate)',
                        'type': 'percentage',
                        'format': 'percent'
                    },
                    'revenue': {
                        'sql': 'SUM(actual_revenue_cents) / 100.0',
                        'type': 'currency',
                        'format': 'currency'
                    },
                    'profit': {
                        'sql': 'SUM(profit_cents) / 100.0',
                        'type': 'currency',
                        'format': 'currency'
                    },
                    'cost_per_video': {
                        'sql': 'AVG(cost_cents) / 100.0',
                        'type': 'currency',
                        'format': 'currency'
                    }
                },
                
                'dimensions': {
                    'channel': {
                        'sql': 'channel_key',
                        'type': 'number',
                        'primaryKey': True
                    },
                    'video': {
                        'sql': 'video_key',
                        'type': 'number',
                        'primaryKey': True
                    },
                    'user': {
                        'sql': 'user_key',
                        'type': 'number'
                    },
                    'publish_date': {
                        'sql': 'timestamp',
                        'type': 'time'
                    },
                    'content_type': {
                        'sql': "(SELECT content_type FROM analytics.dim_video WHERE video_key = fact_video_performance.video_key)",
                        'type': 'string'
                    }
                },
                
                'segments': {
                    'viral_videos': {
                        'sql': 'engagement_rate > 0.10'
                    },
                    'profitable_videos': {
                        'sql': 'profit_cents > 0'
                    },
                    'recent_videos': {
                        'sql': "timestamp >= CURRENT_DATE - INTERVAL '7 days'"
                    },
                    'high_cost_videos': {
                        'sql': 'cost_cents > 300'  # $3.00
                    }
                },
                
                'preAggregations': {
                    'daily_rollup': {
                        'type': 'rollup',
                        'measureReferences': ['total_views', 'revenue', 'profit'],
                        'dimensionReferences': ['channel', 'publish_date'],
                        'timeDimensionReference': 'publish_date',
                        'granularity': 'day'
                    }
                }
            }]
        }
```

### Caching Strategy

```python
# analytics/caching.py
"""
Multi-layer caching for analytics queries
"""

import redis
import json
import hashlib
from datetime import datetime, timedelta

class AnalyticsCache:
    """
    Multi-layer caching for analytics queries
    """
    def __init__(self):
        self.redis = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        
    def get_cache_key(self, query: str, params: dict = None) -> str:
        """Generate consistent cache key"""
        cache_data = f"{query}:{json.dumps(params, sort_keys=True)}"
        return f"analytics:{hashlib.md5(cache_data.encode()).hexdigest()}"
    
    def get_with_cache(self, query: str, query_func, params: dict = None, ttl: int = 300):
        """
        Multi-layer cache retrieval
        TTL in seconds (default 5 minutes)
        """
        cache_key = self.get_cache_key(query, params)
        
        # Check Redis cache
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Execute query
        result = query_func(query, params)
        
        # Cache result
        self.redis.setex(
            cache_key,
            ttl,
            json.dumps(result, default=str)
        )
        
        return result
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        for key in self.redis.scan_iter(match=f"analytics:{pattern}*"):
            self.redis.delete(key)
    
    def get_cache_stats(self):
        """Get cache statistics"""
        info = self.redis.info('stats')
        return {
            'total_keys': self.redis.dbsize(),
            'hits': info.get('keyspace_hits', 0),
            'misses': info.get('keyspace_misses', 0),
            'hit_rate': info.get('keyspace_hits', 0) / 
                       (info.get('keyspace_hits', 0) + info.get('keyspace_misses', 1)) * 100,
            'memory_used': info.get('used_memory_human', '0B')
        }

# Cache configuration for different query types
CACHE_CONFIG = {
    'executive_dashboard': {
        'ttl': 300,  # 5 minutes
        'priority': 'high'
    },
    'operational_metrics': {
        'ttl': 60,   # 1 minute
        'priority': 'high'
    },
    'historical_analysis': {
        'ttl': 3600,  # 1 hour
        'priority': 'medium'
    },
    'user_reports': {
        'ttl': 1800,  # 30 minutes
        'priority': 'low'
    }
}

# Query result caching decorator
def cached_query(cache_type='operational_metrics'):
    """Decorator for caching query results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = AnalyticsCache()
            config = CACHE_CONFIG.get(cache_type, {'ttl': 300})
            
            # Generate cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            return cache.get_with_cache(
                cache_key,
                lambda q, p: func(*args, **kwargs),
                ttl=config['ttl']
            )
        return wrapper
    return decorator
```

#### Redis Cache Warming Strategy

```python
# analytics/cache_warmer.py
"""
Proactive cache warming for critical dashboards
"""

class CacheWarmer:
    def __init__(self):
        self.cache = AnalyticsCache()
        self.db = PostgreSQLConnection()
        
    def warm_executive_dashboard(self):
        """Pre-populate executive dashboard queries"""
        queries = [
            ('active_channels', "SELECT COUNT(*) FROM analytics.dim_channel WHERE is_active = true"),
            ('daily_revenue', "SELECT SUM(revenue_cents)/100 FROM analytics.fact_channel_daily WHERE date_key = (SELECT date_key FROM analytics.dim_date WHERE full_date = CURRENT_DATE)"),
            ('total_videos', "SELECT COUNT(*) FROM analytics.dim_video WHERE published_at >= CURRENT_DATE"),
            ('avg_engagement', "SELECT AVG(engagement_rate) FROM analytics.fact_video_performance WHERE timestamp >= CURRENT_DATE")
        ]
        
        for name, query in queries:
            result = self.db.execute(query)
            self.cache.redis.setex(
                f"analytics:executive:{name}",
                300,  # 5 minutes
                json.dumps(result, default=str)
            )
    
    def warm_user_dashboards(self):
        """Pre-populate user-specific metrics"""
        users = self.db.execute("SELECT user_id FROM users.accounts WHERE is_active = true")
        
        for user in users:
            user_id = user['user_id']
            
            # User revenue
            revenue_query = f"""
                SELECT SUM(revenue_cents)/100 as revenue
                FROM analytics.fact_revenue
                WHERE user_key = (SELECT user_key FROM analytics.dim_user WHERE user_id = '{user_id}')
                AND date_key >= (SELECT date_key FROM analytics.dim_date WHERE full_date = CURRENT_DATE - 30)
            """
            
            result = self.db.execute(revenue_query)
            self.cache.redis.setex(
                f"analytics:user:{user_id}:monthly_revenue",
                1800,  # 30 minutes
                json.dumps(result, default=str)
            )
    
    def schedule_warming(self):
        """Schedule cache warming jobs"""
        # This would be integrated with N8N or cron
        schedule = {
            'executive_dashboard': '*/5 * * * *',  # Every 5 minutes
            'user_dashboards': '*/30 * * * *',     # Every 30 minutes
            'trending_content': '0 * * * *',       # Every hour
        }
        return schedule
```