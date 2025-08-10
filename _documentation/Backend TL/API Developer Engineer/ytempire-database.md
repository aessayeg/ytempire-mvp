# 6. DATABASE DESIGN - YTEMPIRE

**Version 2.0 | January 2025**  
**Document Status: Consolidated & Standardized**  
**Last Updated: January 2025**

---

## 6.1 Schema Overview

### Database Architecture

```yaml
Database System: PostgreSQL 15
Architecture Pattern: Single database, multiple schemas
Connection Pool: 10-100 connections
Backup Strategy: Daily full + hourly incremental
Replication: Primary + Read replica (future)

Schemas:
  public:        # Core application tables
  analytics:     # Analytics and metrics
  audit:         # Audit logs and history
  queue:         # Job queue tables
```

### Design Principles

1. **Normalization**: 3NF for transactional data
2. **UUID Keys**: For distributed systems compatibility
3. **Soft Deletes**: Maintain data history
4. **Audit Trails**: Track all changes
5. **JSON Flexibility**: JSONB for variable data
6. **Time Zones**: All timestamps in UTC
7. **Constraints**: Foreign keys and check constraints
8. **Indexes**: Cover common query patterns

### Entity Relationship Overview

```mermaid
erDiagram
    User ||--o{ Channel : owns
    User ||--o{ APIKey : has
    User ||--o{ Subscription : has
    Channel ||--o{ Video : contains
    Channel ||--o{ ChannelAnalytics : tracks
    Video ||--o{ VideoCost : incurs
    Video ||--o{ VideoAnalytics : generates
    Video ||--|| VideoContent : has
    YouTubeAccount ||--o{ Video : uploads
    User ||--o{ AuditLog : generates
```

---

## 6.2 Table Definitions

### Core Tables

#### Users Table
```sql
CREATE TABLE users (
    -- Primary Key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Authentication
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    
    -- Profile
    full_name VARCHAR(100),
    avatar_url VARCHAR(500),
    timezone VARCHAR(50) DEFAULT 'UTC',
    
    -- Account Status
    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('user', 'admin', 'super_admin')),
    tier VARCHAR(20) DEFAULT 'free' CHECK (tier IN ('free', 'starter', 'growth', 'scale')),
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    email_verified_at TIMESTAMP WITH TIME ZONE,
    
    -- Limits (based on tier)
    channel_limit INTEGER DEFAULT 1,
    daily_video_limit INTEGER DEFAULT 1,
    
    -- Billing
    stripe_customer_id VARCHAR(255) UNIQUE,
    stripe_subscription_id VARCHAR(255),
    trial_ends_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    settings JSONB DEFAULT '{}',
    last_login_at TIMESTAMP WITH TIME ZONE,
    login_count INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Indexes
CREATE INDEX idx_users_email ON users(email) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_username ON users(username) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_stripe_customer ON users(stripe_customer_id) WHERE stripe_customer_id IS NOT NULL;
CREATE INDEX idx_users_tier ON users(tier) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_created ON users(created_at DESC);
```

#### Channels Table
```sql
CREATE TABLE channels (
    -- Primary Key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Relationships
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    youtube_account_id UUID REFERENCES youtube_accounts(id),
    
    -- Channel Info
    name VARCHAR(100) NOT NULL,
    description TEXT,
    niche VARCHAR(50),
    youtube_channel_id VARCHAR(100) UNIQUE,
    youtube_handle VARCHAR(100),
    
    -- Status
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'deleted')),
    is_monetized BOOLEAN DEFAULT false,
    
    -- Configuration
    upload_schedule JSONB DEFAULT '{}', -- {"days": ["mon", "wed", "fri"], "time": "10:00"}
    content_strategy JSONB DEFAULT '{}',
    quality_settings JSONB DEFAULT '{}',
    
    -- Metrics
    subscriber_count INTEGER DEFAULT 0,
    video_count INTEGER DEFAULT 0,
    total_views BIGINT DEFAULT 0,
    total_revenue DECIMAL(10,2) DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE,
    last_synced_at TIMESTAMP WITH TIME ZONE
);

-- Indexes
CREATE INDEX idx_channels_user ON channels(user_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_channels_status ON channels(status);
CREATE INDEX idx_channels_youtube ON channels(youtube_channel_id) WHERE youtube_channel_id IS NOT NULL;
CREATE INDEX idx_channels_created ON channels(created_at DESC);
```

#### Videos Table
```sql
CREATE TABLE videos (
    -- Primary Key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Relationships
    channel_id UUID NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
    youtube_account_id UUID REFERENCES youtube_accounts(id),
    
    -- Video Metadata
    title VARCHAR(100) NOT NULL,
    description TEXT,
    tags TEXT[], -- Array of tags
    topic VARCHAR(200),
    style VARCHAR(50) CHECK (style IN ('educational', 'entertainment', 'review', 'tutorial', 'news')),
    
    -- YouTube Data
    youtube_video_id VARCHAR(50) UNIQUE,
    youtube_url VARCHAR(500),
    thumbnail_url VARCHAR(500),
    
    -- Status Tracking
    status VARCHAR(20) DEFAULT 'pending' CHECK (
        status IN ('pending', 'queued', 'processing', 'completed', 'failed', 'published', 'deleted')
    ),
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Generation Details
    duration_seconds INTEGER,
    file_size_mb DECIMAL(10,2),
    resolution VARCHAR(20) DEFAULT '1080p',
    fps INTEGER DEFAULT 30,
    
    -- Quality Metrics
    quality_score DECIMAL(3,2) CHECK (quality_score >= 0 AND quality_score <= 1),
    ai_confidence DECIMAL(3,2),
    
    -- Cost Tracking
    total_cost DECIMAL(10,4) DEFAULT 0,
    cost_breakdown JSONB DEFAULT '{}', -- {"openai": 0.45, "elevenlabs": 0.22, ...}
    
    -- File Paths
    video_path VARCHAR(500),
    audio_path VARCHAR(500),
    script_path VARCHAR(500),
    thumbnail_path VARCHAR(500),
    
    -- Processing Times
    queued_at TIMESTAMP WITH TIME ZONE,
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    published_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Indexes
CREATE INDEX idx_videos_channel ON videos(channel_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_videos_status ON videos(status);
CREATE INDEX idx_videos_youtube_id ON videos(youtube_video_id) WHERE youtube_video_id IS NOT NULL;
CREATE INDEX idx_videos_created ON videos(created_at DESC);
CREATE INDEX idx_videos_published ON videos(published_at DESC) WHERE published_at IS NOT NULL;
CREATE INDEX idx_videos_cost ON videos(total_cost DESC);
```

### YouTube Integration Tables

#### YouTube Accounts Table
```sql
CREATE TABLE youtube_accounts (
    -- Primary Key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Account Info
    email VARCHAR(255) UNIQUE NOT NULL,
    account_name VARCHAR(100),
    channel_id VARCHAR(100) UNIQUE,
    channel_handle VARCHAR(100),
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    is_reserve BOOLEAN DEFAULT false, -- Part of 5 reserve accounts
    health_score DECIMAL(5,2) DEFAULT 100.00,
    
    -- OAuth Credentials (encrypted)
    credentials JSONB NOT NULL, -- Encrypted OAuth tokens
    token_expires_at TIMESTAMP WITH TIME ZONE,
    refresh_token TEXT,
    
    -- Quota Management (10,000 units/day limit)
    quota_used_today INTEGER DEFAULT 0,
    quota_limit INTEGER DEFAULT 10000,
    uploads_today INTEGER DEFAULT 0,
    upload_limit INTEGER DEFAULT 5,
    last_quota_reset TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Usage Statistics
    total_uploads INTEGER DEFAULT 0,
    total_quota_used BIGINT DEFAULT 0,
    total_errors INTEGER DEFAULT 0,
    consecutive_errors INTEGER DEFAULT 0,
    
    -- Timestamps
    last_upload_at TIMESTAMP WITH TIME ZONE,
    last_error_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_youtube_accounts_active ON youtube_accounts(is_active);
CREATE INDEX idx_youtube_accounts_health ON youtube_accounts(health_score);
CREATE INDEX idx_youtube_accounts_quota ON youtube_accounts(quota_used_today);
```

### Cost Tracking Tables

#### Video Costs Table
```sql
CREATE TABLE video_costs (
    -- Primary Key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Relationships
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    
    -- Cost Details
    service VARCHAR(50) NOT NULL, -- 'openai', 'elevenlabs', 'stability', etc.
    operation VARCHAR(100), -- 'script_generation', 'voice_synthesis', etc.
    amount DECIMAL(10,4) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    
    -- Service Details
    units_consumed DECIMAL(10,2), -- tokens, characters, credits
    unit_type VARCHAR(50), -- 'tokens', 'characters', 'seconds'
    rate_per_unit DECIMAL(10,6),
    
    -- Metadata
    request_id VARCHAR(100),
    response_time_ms INTEGER,
    metadata JSONB DEFAULT '{}',
    
    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_video_costs_video ON video_costs(video_id);
CREATE INDEX idx_video_costs_service ON video_costs(service);
CREATE INDEX idx_video_costs_created ON video_costs(created_at DESC);
CREATE INDEX idx_video_costs_amount ON video_costs(amount DESC);
```

### Analytics Tables

#### Channel Analytics Table
```sql
CREATE TABLE channel_analytics (
    -- Primary Key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Relationships
    channel_id UUID NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
    
    -- Time Period
    date DATE NOT NULL,
    hour INTEGER CHECK (hour >= 0 AND hour <= 23),
    
    -- YouTube Metrics
    views INTEGER DEFAULT 0,
    watch_time_minutes INTEGER DEFAULT 0,
    subscribers_gained INTEGER DEFAULT 0,
    subscribers_lost INTEGER DEFAULT 0,
    likes INTEGER DEFAULT 0,
    dislikes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    shares INTEGER DEFAULT 0,
    
    -- Revenue Metrics
    estimated_revenue DECIMAL(10,2) DEFAULT 0,
    ad_revenue DECIMAL(10,2) DEFAULT 0,
    channel_memberships DECIMAL(10,2) DEFAULT 0,
    super_chat DECIMAL(10,2) DEFAULT 0,
    
    -- Performance Metrics
    ctr DECIMAL(5,2), -- Click-through rate
    average_view_duration_seconds INTEGER,
    average_percentage_viewed DECIMAL(5,2),
    
    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint
    CONSTRAINT unique_channel_analytics UNIQUE (channel_id, date, hour)
);

-- Indexes
CREATE INDEX idx_channel_analytics_channel_date ON channel_analytics(channel_id, date DESC);
CREATE INDEX idx_channel_analytics_date ON channel_analytics(date DESC);
```

---

## 6.3 Relationships & Constraints

### Foreign Key Relationships

```sql
-- User relationships
ALTER TABLE channels 
    ADD CONSTRAINT fk_channels_user 
    FOREIGN KEY (user_id) 
    REFERENCES users(id) 
    ON DELETE CASCADE;

ALTER TABLE api_keys 
    ADD CONSTRAINT fk_api_keys_user 
    FOREIGN KEY (user_id) 
    REFERENCES users(id) 
    ON DELETE CASCADE;

-- Channel relationships
ALTER TABLE videos 
    ADD CONSTRAINT fk_videos_channel 
    FOREIGN KEY (channel_id) 
    REFERENCES channels(id) 
    ON DELETE CASCADE;

ALTER TABLE channel_analytics 
    ADD CONSTRAINT fk_channel_analytics_channel 
    FOREIGN KEY (channel_id) 
    REFERENCES channels(id) 
    ON DELETE CASCADE;

-- Video relationships
ALTER TABLE video_costs 
    ADD CONSTRAINT fk_video_costs_video 
    FOREIGN KEY (video_id) 
    REFERENCES videos(id) 
    ON DELETE CASCADE;

ALTER TABLE video_analytics 
    ADD CONSTRAINT fk_video_analytics_video 
    FOREIGN KEY (video_id) 
    REFERENCES videos(id) 
    ON DELETE CASCADE;

-- YouTube account relationships
ALTER TABLE channels 
    ADD CONSTRAINT fk_channels_youtube_account 
    FOREIGN KEY (youtube_account_id) 
    REFERENCES youtube_accounts(id) 
    ON DELETE SET NULL;

ALTER TABLE videos 
    ADD CONSTRAINT fk_videos_youtube_account 
    FOREIGN KEY (youtube_account_id) 
    REFERENCES youtube_accounts(id) 
    ON DELETE SET NULL;
```

### Check Constraints

```sql
-- User constraints
ALTER TABLE users 
    ADD CONSTRAINT check_channel_limit 
    CHECK (channel_limit >= 0 AND channel_limit <= 100);

ALTER TABLE users 
    ADD CONSTRAINT check_daily_video_limit 
    CHECK (daily_video_limit >= 0 AND daily_video_limit <= 1000);

-- Video constraints
ALTER TABLE videos 
    ADD CONSTRAINT check_quality_score 
    CHECK (quality_score >= 0 AND quality_score <= 1);

ALTER TABLE videos 
    ADD CONSTRAINT check_duration 
    CHECK (duration_seconds > 0 AND duration_seconds <= 3600);

ALTER TABLE videos 
    ADD CONSTRAINT check_cost 
    CHECK (total_cost >= 0);

-- YouTube account constraints
ALTER TABLE youtube_accounts 
    ADD CONSTRAINT check_health_score 
    CHECK (health_score >= 0 AND health_score <= 100);

ALTER TABLE youtube_accounts 
    ADD CONSTRAINT check_quota 
    CHECK (quota_used_today >= 0 AND quota_used_today <= quota_limit);
```

### Unique Constraints

```sql
-- Prevent duplicate channels
ALTER TABLE channels 
    ADD CONSTRAINT unique_user_channel_name 
    UNIQUE (user_id, name) 
    WHERE deleted_at IS NULL;

-- Prevent duplicate API keys
ALTER TABLE api_keys 
    ADD CONSTRAINT unique_api_key 
    UNIQUE (key_hash);

-- Prevent duplicate analytics entries
ALTER TABLE channel_analytics 
    ADD CONSTRAINT unique_channel_date_hour 
    UNIQUE (channel_id, date, hour);

ALTER TABLE video_analytics 
    ADD CONSTRAINT unique_video_date_hour 
    UNIQUE (video_id, date, hour);
```

---

## 6.4 Migration Strategy

### Migration Tools

```yaml
Tool: Alembic
Version Control: Git
Migration Naming: YYYYMMDD_HHMMSS_description.py
Rollback Support: Yes
Test Environment: Required before production
```

### Migration Process

```python
# alembic.ini configuration
[alembic]
script_location = migrations
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = postgresql://user:pass@localhost/ytempire

# Migration command examples
alembic init migrations                    # Initialize migrations
alembic revision -m "add_users_table"     # Create new migration
alembic upgrade head                       # Apply all migrations
alembic downgrade -1                       # Rollback one migration
alembic history                            # View migration history
alembic current                            # Show current version
```

### Migration Template

```python
"""Add video costs table

Revision ID: 20250115_100000
Revises: 20250114_120000
Create Date: 2025-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '20250115_100000'
down_revision = '20250114_120000'
branch_labels = None
depends_on = None

def upgrade():
    # Create video_costs table
    op.create_table('video_costs',
        sa.Column('id', postgresql.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('video_id', postgresql.UUID(), nullable=False),
        sa.Column('service', sa.String(50), nullable=False),
        sa.Column('amount', sa.Numeric(10, 4), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE')
    )
    
    # Create indexes
    op.create_index('idx_video_costs_video', 'video_costs', ['video_id'])
    op.create_index('idx_video_costs_service', 'video_costs', ['service'])

def downgrade():
    # Drop indexes
    op.drop_index('idx_video_costs_service', 'video_costs')
    op.drop_index('idx_video_costs_video', 'video_costs')
    
    # Drop table
    op.drop_table('video_costs')
```

### Migration Best Practices

1. **Always test migrations** in development first
2. **Include both upgrade and downgrade** functions
3. **Create indexes** for foreign keys and commonly queried columns
4. **Use transactions** for data migrations
5. **Backup before** production migrations
6. **Document breaking changes** clearly
7. **Version control** all migration files
8. **Never edit** applied migrations

---

## 6.5 Performance Optimization

### Indexing Strategy

```sql
-- Primary indexes (automatically created)
-- All primary keys have indexes

-- Foreign key indexes
CREATE INDEX idx_fk_channels_user_id ON channels(user_id);
CREATE INDEX idx_fk_videos_channel_id ON videos(channel_id);
CREATE INDEX idx_fk_video_costs_video_id ON video_costs(video_id);

-- Query optimization indexes
CREATE INDEX idx_videos_status_created ON videos(status, created_at DESC);
CREATE INDEX idx_videos_channel_status ON videos(channel_id, status) WHERE deleted_at IS NULL;
CREATE INDEX idx_channels_user_status ON channels(user_id, status) WHERE deleted_at IS NULL;

-- Partial indexes for common queries
CREATE INDEX idx_videos_pending ON videos(created_at) WHERE status = 'pending';
CREATE INDEX idx_videos_published ON videos(published_at DESC) WHERE status = 'published';
CREATE INDEX idx_active_channels ON channels(user_id) WHERE status = 'active' AND deleted_at IS NULL;

-- Composite indexes for analytics
CREATE INDEX idx_analytics_channel_date ON channel_analytics(channel_id, date DESC);
CREATE INDEX idx_analytics_date_channel ON channel_analytics(date DESC, channel_id);

-- Text search indexes
CREATE INDEX idx_videos_title_gin ON videos USING gin(to_tsvector('english', title));
CREATE INDEX idx_videos_tags_gin ON videos USING gin(tags);

-- JSONB indexes
CREATE INDEX idx_users_settings_gin ON users USING gin(settings);
CREATE INDEX idx_videos_metadata_gin ON videos USING gin(metadata);
```

### Query Optimization

```python
# SQLAlchemy query optimization examples

from sqlalchemy import select, func
from sqlalchemy.orm import selectinload, joinedload

class VideoRepository:
    def get_videos_with_costs(self, channel_id: str):
        """Optimized query with eager loading"""
        return (
            db.query(Video)
            .options(selectinload(Video.costs))  # Eager load costs
            .filter(Video.channel_id == channel_id)
            .filter(Video.deleted_at.is_(None))
            .order_by(Video.created_at.desc())
            .limit(20)
            .all()
        )
    
    def get_channel_stats(self, user_id: str):
        """Optimized aggregation query"""
        return (
            db.query(
                Channel.id,
                Channel.name,
                func.count(Video.id).label('video_count'),
                func.sum(Video.total_cost).label('total_cost'),
                func.avg(Video.quality_score).label('avg_quality')
            )
            .outerjoin(Video)
            .filter(Channel.user_id == user_id)
            .filter(Channel.deleted_at.is_(None))
            .group_by(Channel.id, Channel.name)
            .all()
        )
```

### Database Configuration

```sql
-- PostgreSQL performance tuning
ALTER SYSTEM SET shared_buffers = '32GB';           -- 25% of RAM
ALTER SYSTEM SET effective_cache_size = '96GB';     -- 75% of RAM
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET work_mem = '128MB';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET random_page_cost = 1.1;            -- SSD optimization
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET max_wal_size = '4GB';
ALTER SYSTEM SET min_wal_size = '1GB';

-- Connection pooling configuration
-- pgBouncer settings
[databases]
ytempire = host=localhost port=5432 dbname=ytempire

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
reserve_pool_size = 5
reserve_pool_timeout = 3
server_lifetime = 3600
```

### Monitoring Queries

```sql
-- Slow query monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Find slow queries
SELECT 
    query,
    calls,
    total_time / 1000 as total_seconds,
    mean / 1000 as mean_seconds,
    max / 1000 as max_seconds
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
ORDER BY mean DESC
LIMIT 20;

-- Table size monitoring
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Index usage monitoring
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Cache hit ratio
SELECT 
    sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as cache_hit_ratio
FROM pg_statio_user_tables;
```

---

## Document Control

- **Version**: 2.0
- **Last Updated**: January 2025
- **Next Review**: February 2025
- **Owner**: API Development Engineer
- **Approved By**: Backend Team Lead

---

## Navigation

- [← Previous: API Specifications](./5-api-specifications.md)
- [→ Next: Implementation Guides](./7-implementation-guides.md)