# YTEMPIRE Database Schema & Models Document
**Version 1.0 | January 2025**  
**Owner: Backend Team Lead**  
**Approved By: CTO/Technical Director**  
**Status: Ready for Implementation**

---

## Executive Summary

This document defines the complete database schema and data models for YTEMPIRE's MVP implementation. The schema is designed to support automated YouTube channel management at scale while maintaining data integrity, performance, and security.

**Core Design Principles:**
- **Scalability First**: Designed to handle 500+ channels and 100K+ videos
- **Security by Design**: PII encryption, audit trails, secure defaults
- **Performance Optimized**: Strategic indexing, efficient queries
- **Extensibility**: Ready for future features without breaking changes

---

## 1. Database Architecture Overview

### 1.1 Technology Stack

```yaml
database_configuration:
  primary_database:
    engine: PostgreSQL 15
    purpose: Transactional data, ACID compliance
    storage: 300GB NVMe (expandable)
    connections: 200 (pooled)
    
  cache_layer:
    engine: Redis 7
    purpose: Session management, hot data, queues
    memory: 8GB
    persistence: AOF every second
    
  time_series:
    engine: TimescaleDB (PostgreSQL extension)
    purpose: Analytics, metrics, performance data
    retention: 90 days detailed, 2 years aggregated
```

### 1.2 Security Considerations

```yaml
security_implementation:
  encryption:
    at_rest: AES-256-GCM
    in_transit: TLS 1.3
    sensitive_fields: PII, API keys, tokens
    
  access_control:
    method: Row Level Security (RLS)
    authentication: JWT tokens
    authorization: Role-based (RBAC)
    
  audit:
    all_changes: Tracked in audit_logs
    retention: 1 year
    compliance: GDPR, CCPA ready
```

---

## 2. Core Schema Design

### 2.1 Users & Authentication

```sql
-- Users table: Core user accounts
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    password_hash VARCHAR(255) NOT NULL, -- bcrypt
    full_name VARCHAR(255),
    avatar_url TEXT,
    role VARCHAR(50) DEFAULT 'user', -- user, admin, beta_tester
    status VARCHAR(50) DEFAULT 'active', -- active, suspended, deleted
    
    -- Subscription & Billing
    subscription_tier VARCHAR(50) DEFAULT 'free', -- free, starter, pro, enterprise
    subscription_status VARCHAR(50) DEFAULT 'active',
    stripe_customer_id VARCHAR(255),
    trial_ends_at TIMESTAMP,
    
    -- Settings & Preferences
    settings JSONB DEFAULT '{}',
    timezone VARCHAR(50) DEFAULT 'UTC',
    language VARCHAR(10) DEFAULT 'en',
    
    -- Security
    two_factor_enabled BOOLEAN DEFAULT FALSE,
    two_factor_secret VARCHAR(255), -- encrypted
    last_login_at TIMESTAMP,
    last_login_ip INET,
    failed_login_attempts INT DEFAULT 0,
    locked_until TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    deleted_at TIMESTAMP, -- Soft delete
    
    -- Indexes
    CONSTRAINT email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

CREATE INDEX idx_users_email ON users(email) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_subscription ON users(subscription_tier, subscription_status);

-- User sessions: Active sessions management
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) UNIQUE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP NOT NULL,
    revoked_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT valid_session CHECK (expires_at > created_at)
);

CREATE INDEX idx_sessions_user ON user_sessions(user_id) WHERE revoked_at IS NULL;
CREATE INDEX idx_sessions_token ON user_sessions(token_hash) WHERE revoked_at IS NULL;
CREATE INDEX idx_sessions_expiry ON user_sessions(expires_at) WHERE revoked_at IS NULL;

-- API keys: For programmatic access
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    permissions JSONB DEFAULT '[]', -- Array of allowed operations
    rate_limit INT DEFAULT 1000, -- Requests per hour
    last_used_at TIMESTAMP,
    expires_at TIMESTAMP,
    revoked_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_user_key_name UNIQUE(user_id, name)
);

CREATE INDEX idx_api_keys_user ON api_keys(user_id) WHERE revoked_at IS NULL;
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash) WHERE revoked_at IS NULL;
```

### 2.2 YouTube Channels

```sql
-- YouTube channels: Core channel management
CREATE TABLE youtube_channels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- YouTube Integration
    youtube_channel_id VARCHAR(255) UNIQUE NOT NULL,
    youtube_channel_handle VARCHAR(255),
    channel_title VARCHAR(255) NOT NULL,
    channel_description TEXT,
    channel_thumbnail_url TEXT,
    
    -- OAuth Tokens (encrypted)
    access_token TEXT, -- encrypted
    refresh_token TEXT, -- encrypted
    token_expires_at TIMESTAMP,
    
    -- Channel Configuration
    niche VARCHAR(100) NOT NULL,
    target_audience JSONB DEFAULT '{}',
    content_strategy JSONB DEFAULT '{}',
    branding_assets JSONB DEFAULT '{}', -- logos, banners, watermarks
    
    -- Automation Settings
    automation_enabled BOOLEAN DEFAULT TRUE,
    videos_per_day INT DEFAULT 1,
    publishing_schedule JSONB DEFAULT '[]', -- Array of time slots
    auto_publish BOOLEAN DEFAULT TRUE,
    
    -- Monetization
    monetization_enabled BOOLEAN DEFAULT FALSE,
    adsense_connected BOOLEAN DEFAULT FALSE,
    affiliate_settings JSONB DEFAULT '{}',
    sponsorship_settings JSONB DEFAULT '{}',
    
    -- Analytics Snapshot (updated daily)
    subscriber_count INT DEFAULT 0,
    total_views BIGINT DEFAULT 0,
    total_videos INT DEFAULT 0,
    estimated_revenue DECIMAL(10,2) DEFAULT 0,
    
    -- Status & Health
    status VARCHAR(50) DEFAULT 'active', -- active, paused, suspended, deleted
    health_score INT DEFAULT 100, -- 0-100 based on performance
    last_sync_at TIMESTAMP,
    last_error TEXT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    deleted_at TIMESTAMP,
    
    CONSTRAINT valid_videos_per_day CHECK (videos_per_day BETWEEN 0 AND 10)
);

CREATE INDEX idx_channels_user ON youtube_channels(user_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_channels_status ON youtube_channels(status);
CREATE INDEX idx_channels_niche ON youtube_channels(niche);
CREATE INDEX idx_channels_health ON youtube_channels(health_score);

-- Channel categories: Predefined niches
CREATE TABLE channel_categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    parent_category_id UUID REFERENCES channel_categories(id),
    
    -- Performance Metrics
    avg_cpm DECIMAL(10,2), -- Average CPM for this category
    competition_level VARCHAR(50), -- low, medium, high
    growth_potential VARCHAR(50), -- low, medium, high
    
    -- Content Templates
    video_templates JSONB DEFAULT '[]',
    keyword_suggestions JSONB DEFAULT '[]',
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO channel_categories (name, description, avg_cpm, competition_level, growth_potential) VALUES
('Technology', 'Tech reviews, tutorials, news', 8.50, 'high', 'high'),
('Gaming', 'Gaming content, reviews, walkthroughs', 4.20, 'high', 'medium'),
('Education', 'Educational and how-to content', 12.30, 'medium', 'high'),
('Entertainment', 'General entertainment content', 3.80, 'high', 'medium'),
('Finance', 'Financial advice, investing, crypto', 18.50, 'medium', 'high');
```

### 2.3 Videos & Content

```sql
-- Videos: Core video content management
CREATE TABLE videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_id UUID NOT NULL REFERENCES youtube_channels(id) ON DELETE CASCADE,
    
    -- YouTube Integration
    youtube_video_id VARCHAR(255) UNIQUE,
    youtube_url TEXT,
    
    -- Content Details
    title VARCHAR(255) NOT NULL,
    description TEXT,
    tags TEXT[], -- Array of tags
    category VARCHAR(100),
    
    -- Generation Details
    script TEXT NOT NULL,
    voice_model VARCHAR(100) DEFAULT 'en-US-Standard-C',
    background_music VARCHAR(255),
    
    -- File References
    video_file_path TEXT,
    thumbnail_file_path TEXT,
    subtitle_file_path TEXT,
    
    -- Processing Metadata
    generation_status VARCHAR(50) DEFAULT 'pending', 
    -- pending, generating, processing, ready, published, failed
    generation_started_at TIMESTAMP,
    generation_completed_at TIMESTAMP,
    generation_duration_seconds INT,
    generation_error TEXT,
    
    -- Publishing
    scheduled_publish_at TIMESTAMP,
    published_at TIMESTAMP,
    publish_status VARCHAR(50), -- scheduled, published, failed
    
    -- Analytics (updated periodically)
    view_count INT DEFAULT 0,
    like_count INT DEFAULT 0,
    dislike_count INT DEFAULT 0,
    comment_count INT DEFAULT 0,
    average_view_duration INT, -- seconds
    click_through_rate DECIMAL(5,2),
    
    -- Monetization
    monetization_status VARCHAR(50), -- eligible, active, demonetized
    estimated_revenue DECIMAL(10,2) DEFAULT 0,
    affiliate_links JSONB DEFAULT '[]',
    
    -- Cost Tracking
    generation_cost JSONB DEFAULT '{}', -- Breakdown of costs
    total_cost DECIMAL(10,2) DEFAULT 0,
    
    -- Quality Metrics
    quality_score INT, -- 0-100 AI-generated quality score
    copyright_check_status VARCHAR(50),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    deleted_at TIMESTAMP,
    
    CONSTRAINT valid_quality_score CHECK (quality_score BETWEEN 0 AND 100)
);

CREATE INDEX idx_videos_channel ON videos(channel_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_videos_status ON videos(generation_status);
CREATE INDEX idx_videos_scheduled ON videos(scheduled_publish_at) WHERE publish_status = 'scheduled';
CREATE INDEX idx_videos_published ON videos(published_at) WHERE published_at IS NOT NULL;
CREATE INDEX idx_videos_performance ON videos(view_count DESC);

-- Video templates: Reusable content templates
CREATE TABLE video_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    
    -- Template Content
    title_template TEXT, -- With variables like {topic}, {date}
    description_template TEXT,
    tags_template TEXT[],
    
    -- Generation Settings
    script_template TEXT,
    voice_settings JSONB DEFAULT '{}',
    visual_settings JSONB DEFAULT '{}',
    
    -- Performance Metrics
    times_used INT DEFAULT 0,
    avg_performance_score DECIMAL(5,2),
    
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_templates_user ON video_templates(user_id);
CREATE INDEX idx_templates_public ON video_templates(is_public) WHERE is_public = TRUE;
```

### 2.4 Analytics & Metrics

```sql
-- Channel analytics: Time-series data
CREATE TABLE channel_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_id UUID NOT NULL REFERENCES youtube_channels(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    
    -- Growth Metrics
    subscriber_count INT,
    subscriber_change INT,
    view_count INT,
    watch_time_minutes INT,
    
    -- Engagement Metrics
    likes INT,
    comments INT,
    shares INT,
    average_view_duration INT,
    
    -- Revenue Metrics
    estimated_revenue DECIMAL(10,2),
    ad_revenue DECIMAL(10,2),
    channel_memberships_revenue DECIMAL(10,2),
    
    -- Content Metrics
    videos_published INT,
    videos_scheduled INT,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_channel_date UNIQUE(channel_id, date)
);

CREATE INDEX idx_analytics_channel_date ON channel_analytics(channel_id, date DESC);

-- Video analytics: Detailed video performance
CREATE TABLE video_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    
    -- View Metrics
    views INT DEFAULT 0,
    unique_viewers INT DEFAULT 0,
    impressions INT DEFAULT 0,
    click_through_rate DECIMAL(5,2),
    
    -- Engagement
    watch_time_minutes INT DEFAULT 0,
    average_view_percentage DECIMAL(5,2),
    likes INT DEFAULT 0,
    dislikes INT DEFAULT 0,
    comments INT DEFAULT 0,
    shares INT DEFAULT 0,
    
    -- Audience
    subscriber_views INT DEFAULT 0,
    non_subscriber_views INT DEFAULT 0,
    
    -- Revenue
    estimated_revenue DECIMAL(10,2) DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_video_date UNIQUE(video_id, date)
);

CREATE INDEX idx_video_analytics_video_date ON video_analytics(video_id, date DESC);

-- Cost tracking: Platform operation costs
CREATE TABLE cost_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Reference
    entity_type VARCHAR(50) NOT NULL, -- video, channel, user
    entity_id UUID NOT NULL,
    
    -- Cost Details
    service VARCHAR(100) NOT NULL, -- openai, elevenlabs, storage, compute
    operation VARCHAR(100), -- script_generation, voice_synthesis, etc.
    
    -- Amounts
    amount DECIMAL(10,4) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    
    -- Metadata
    details JSONB DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_cost_entity ON cost_tracking(entity_type, entity_id);
CREATE INDEX idx_cost_timestamp ON cost_tracking(timestamp DESC);
CREATE INDEX idx_cost_service ON cost_tracking(service);
```

### 2.5 Automation & Workflows

```sql
-- N8N workflows: Automation pipeline tracking
CREATE TABLE automation_workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    workflow_type VARCHAR(100) NOT NULL, -- video_generation, publishing, analytics
    
    -- Configuration
    n8n_workflow_id VARCHAR(255),
    trigger_type VARCHAR(100), -- schedule, webhook, manual
    schedule_cron VARCHAR(255),
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMP,
    last_run_status VARCHAR(50),
    next_run_at TIMESTAMP,
    
    -- Metadata
    configuration JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Workflow executions: Track automation runs
CREATE TABLE workflow_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES automation_workflows(id) ON DELETE CASCADE,
    
    -- Execution Details
    execution_id VARCHAR(255), -- N8N execution ID
    trigger_data JSONB DEFAULT '{}',
    
    -- Status
    status VARCHAR(50) NOT NULL, -- running, success, failed, timeout
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    duration_seconds INT,
    
    -- Results
    output_data JSONB DEFAULT '{}',
    error_message TEXT,
    
    -- Resources
    items_processed INT DEFAULT 0,
    cost DECIMAL(10,4) DEFAULT 0
);

CREATE INDEX idx_executions_workflow ON workflow_executions(workflow_id);
CREATE INDEX idx_executions_status ON workflow_executions(status);
CREATE INDEX idx_executions_started ON workflow_executions(started_at DESC);

-- Job queue: Async task management
CREATE TABLE job_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Job Details
    job_type VARCHAR(100) NOT NULL,
    priority INT DEFAULT 5, -- 1-10, 1 is highest
    payload JSONB NOT NULL,
    
    -- Scheduling
    scheduled_for TIMESTAMP DEFAULT NOW(),
    
    -- Execution
    status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed
    attempts INT DEFAULT 0,
    max_attempts INT DEFAULT 3,
    
    -- Worker Info
    worker_id VARCHAR(255),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Error Handling
    last_error TEXT,
    next_retry_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT valid_priority CHECK (priority BETWEEN 1 AND 10)
);

CREATE INDEX idx_queue_status_priority ON job_queue(status, priority, scheduled_for);
CREATE INDEX idx_queue_type ON job_queue(job_type);
CREATE INDEX idx_queue_retry ON job_queue(next_retry_at) WHERE status = 'failed' AND attempts < max_attempts;
```

### 2.6 System & Audit Tables

```sql
-- Audit logs: Track all system changes
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Actor
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    ip_address INET,
    user_agent TEXT,
    
    -- Action
    action VARCHAR(100) NOT NULL, -- create, update, delete, login, etc.
    entity_type VARCHAR(100) NOT NULL,
    entity_id UUID,
    
    -- Changes
    old_values JSONB,
    new_values JSONB,
    
    -- Metadata
    timestamp TIMESTAMP DEFAULT NOW(),
    
    -- Security
    risk_score INT DEFAULT 0 -- 0-100, for anomaly detection
);

CREATE INDEX idx_audit_user ON audit_logs(user_id);
CREATE INDEX idx_audit_entity ON audit_logs(entity_type, entity_id);
CREATE INDEX idx_audit_timestamp ON audit_logs(timestamp DESC);
CREATE INDEX idx_audit_action ON audit_logs(action);
CREATE INDEX idx_audit_risk ON audit_logs(risk_score) WHERE risk_score > 50;

-- System notifications: User notifications
CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Notification Details
    type VARCHAR(100) NOT NULL, -- info, success, warning, error
    title VARCHAR(255) NOT NULL,
    message TEXT,
    
    -- Action
    action_url TEXT,
    action_label VARCHAR(100),
    
    -- Status
    is_read BOOLEAN DEFAULT FALSE,
    read_at TIMESTAMP,
    
    -- Delivery
    email_sent BOOLEAN DEFAULT FALSE,
    push_sent BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP
);

CREATE INDEX idx_notifications_user_unread ON notifications(user_id, created_at DESC) 
    WHERE is_read = FALSE AND (expires_at IS NULL OR expires_at > NOW());

-- Feature flags: Control feature rollout
CREATE TABLE feature_flags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    
    -- Targeting
    enabled BOOLEAN DEFAULT FALSE,
    rollout_percentage INT DEFAULT 0, -- 0-100
    
    -- Rules
    user_rules JSONB DEFAULT '[]', -- Specific user IDs
    segment_rules JSONB DEFAULT '[]', -- User segments
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT valid_rollout CHECK (rollout_percentage BETWEEN 0 AND 100)
);

-- System configuration: Key-value config store
CREATE TABLE system_config (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    is_encrypted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

---

## 3. Database Optimization

### 3.1 Indexing Strategy

```sql
-- Performance-critical indexes
CREATE INDEX CONCURRENTLY idx_videos_generation_queue 
    ON videos(generation_status, created_at) 
    WHERE generation_status IN ('pending', 'generating');

CREATE INDEX CONCURRENTLY idx_channels_automation 
    ON youtube_channels(user_id, status) 
    WHERE automation_enabled = TRUE AND deleted_at IS NULL;

CREATE INDEX CONCURRENTLY idx_analytics_recent 
    ON channel_analytics(date DESC, channel_id);

-- Full-text search indexes
CREATE INDEX idx_videos_search ON videos 
    USING gin(to_tsvector('english', title || ' ' || description));

CREATE INDEX idx_channels_search ON youtube_channels 
    USING gin(to_tsvector('english', channel_title || ' ' || channel_description));
```

### 3.2 Partitioning Strategy

```sql
-- Partition large tables by date
CREATE TABLE video_analytics_2025_01 PARTITION OF video_analytics
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE video_analytics_2025_02 PARTITION OF video_analytics
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

-- Automated partition management
CREATE OR REPLACE FUNCTION create_monthly_partition()
RETURNS void AS $$
DECLARE
    start_date date;
    end_date date;
    partition_name text;
BEGIN
    start_date := date_trunc('month', CURRENT_DATE + interval '1 month');
    end_date := start_date + interval '1 month';
    partition_name := 'video_analytics_' || to_char(start_date, 'YYYY_MM');
    
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF video_analytics FOR VALUES FROM (%L) TO (%L)',
        partition_name, start_date, end_date);
END;
$$ LANGUAGE plpgsql;

-- Schedule monthly partition creation
SELECT cron.schedule('create-partitions', '0 0 25 * *', 'SELECT create_monthly_partition()');
```

### 3.3 Query Optimization

```sql
-- Materialized views for expensive queries
CREATE MATERIALIZED VIEW channel_performance_summary AS
SELECT 
    c.id,
    c.user_id,
    c.channel_title,
    c.subscriber_count,
    COUNT(v.id) as total_videos,
    AVG(v.view_count) as avg_views,
    SUM(v.estimated_revenue) as total_revenue,
    MAX(v.published_at) as last_published
FROM youtube_channels c
LEFT JOIN videos v ON c.id = v.channel_id
WHERE c.deleted_at IS NULL
GROUP BY c.id;

CREATE UNIQUE INDEX ON channel_performance_summary(id);

-- Refresh schedule
SELECT cron.schedule('refresh-channel-summary', '0 * * * *', 
    'REFRESH MATERIALIZED VIEW CONCURRENTLY channel_performance_summary');
```

---

## 4. Security Implementation

### 4.1 Row Level Security

```sql
-- Enable RLS on sensitive tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE youtube_channels ENABLE ROW LEVEL SECURITY;
ALTER TABLE videos ENABLE ROW LEVEL SECURITY;

-- User can only see their own data
CREATE POLICY users_isolation ON users
    FOR ALL
    USING (id = current_setting('app.current_user_id')::uuid);

CREATE POLICY channels_isolation ON youtube_channels
    FOR ALL
    USING (user_id = current_setting('app.current_user_id')::uuid);

CREATE POLICY videos_isolation ON videos
    FOR ALL
    USING (channel_id IN (
        SELECT id FROM youtube_channels 
        WHERE user_id = current_setting('app.current_user_id')::uuid
    ));

-- Admin override
CREATE POLICY admin_all_access ON users
    FOR ALL
    USING (current_setting('app.current_user_role') = 'admin');
```

### 4.2 Encryption Functions

```sql
-- Encryption helper functions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Encrypt sensitive data
CREATE OR REPLACE FUNCTION encrypt_sensitive(data text)
RETURNS text AS $$
BEGIN
    RETURN pgp_sym_encrypt(
        data,
        current_setting('app.encryption_key'),
        'compress-algo=1, cipher-algo=aes256'
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Decrypt sensitive data
CREATE OR REPLACE FUNCTION decrypt_sensitive(encrypted_data text)
RETURNS text AS $$
BEGIN
    RETURN pgp_sym_decrypt(
        encrypted_data::bytea,
        current_setting('app.encryption_key')
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Apply to sensitive columns
CREATE TRIGGER encrypt_user_tokens
    BEFORE INSERT OR UPDATE ON youtube_channels
    FOR EACH ROW
    WHEN (NEW.access_token IS NOT NULL)
    EXECUTE FUNCTION encrypt_column('access_token', 'refresh_token');
```

---

## 5. Database Maintenance

### 5.1 Backup Strategy

```bash
#!/bin/bash
# backup.sh - Database backup script

# Configuration
DB_NAME="ytempire"
BACKUP_DIR="/mnt/backups/postgres"
S3_BUCKET="ytempire-backups"
RETENTION_DAYS=30

# Create backup
BACKUP_FILE="$BACKUP_DIR/ytempire_$(date +%Y%m%d_%H%M%S).sql.gz"
pg_dump -h localhost -U ytempire -d $DB_NAME | gzip > $BACKUP_FILE

# Upload to S3
aws s3 cp $BACKUP_FILE s3://$S3_BUCKET/postgres/

# Clean old backups
find $BACKUP_DIR -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete

# Verify backup
pg_restore --list $BACKUP_FILE > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "Backup successful: $BACKUP_FILE"
else
    echo "Backup verification failed!"
    exit 1
fi
```

### 5.2 Performance Monitoring

```sql
-- Monitor slow queries
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- View slow queries
CREATE VIEW slow_queries AS
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    max_exec_time,
    rows
FROM pg_stat_statements
WHERE mean_exec_time > 100
ORDER BY mean_exec_time DESC
LIMIT 20;

-- Table bloat monitoring
CREATE VIEW table_bloat AS
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS bloat_size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

---

## 6. Migration Strategy

### 6.1 Initial Setup

```bash
#!/bin/bash
# setup-database.sh

# Create database and user
psql -U postgres << EOF
CREATE USER ytempire WITH PASSWORD '$DB_PASSWORD';
CREATE DATABASE ytempire OWNER ytempire;
GRANT ALL PRIVILEGES ON DATABASE ytempire TO ytempire;
EOF

# Run migrations
npm run migrate:up

# Seed initial data
npm run seed:initial
```

### 6.2 Migration Files

```javascript
// migrations/001_initial_schema.js
exports.up = async (knex) => {
    // Create all tables in order
    await knex.raw(fs.readFileSync('schema/users.sql', 'utf8'));
    await knex.raw(fs.readFileSync('schema/channels.sql', 'utf8'));
    await knex.raw(fs.readFileSync('schema/videos.sql', 'utf8'));
    await knex.raw(fs.readFileSync('schema/analytics.sql', 'utf8'));
    await knex.raw(fs.readFileSync('schema/automation.sql', 'utf8'));
    await knex.raw(fs.readFileSync('schema/system.sql', 'utf8'));
};

exports.down = async (knex) => {
    // Drop all tables in reverse order
    await knex.raw('DROP SCHEMA public CASCADE');
    await knex.raw('CREATE SCHEMA public');
};
```

---

## Document Control

- **Version**: 1.0
- **Last Updated**: January 2025
- **Owner**: Backend Team Lead
- **Review Cycle**: Weekly during MVP
- **Security Classification**: Confidential

**Approval Chain:**
1. Backend Team Lead ✅
2. Security Engineer (Review Required)
3. CTO/Technical Director (Final Approval)

---

## Security Engineer Action Items

Based on this schema, the Security Engineer should:

1. **Review and validate** all encryption implementations
2. **Implement RLS policies** for all sensitive tables
3. **Set up audit logging** triggers for compliance
4. **Configure backup encryption** at rest
5. **Establish key rotation** procedures
6. **Create security views** for monitoring
7. **Document data classification** for each field
8. **Implement PII detection** and masking functions

This schema provides the foundation for YTEMPIRE's data layer with security considerations built in from the ground up.