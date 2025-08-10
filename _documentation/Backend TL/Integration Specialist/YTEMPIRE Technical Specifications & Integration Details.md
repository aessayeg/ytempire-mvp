# YTEMPIRE Technical Specifications & Integration Details

**Document Version**: 1.0  
**Date**: January 2025  
**Author**: Backend Team Lead  
**Audience**: Integration Specialist  
**Scope**: Complete Technical Specifications for MVP Implementation

---

## Table of Contents
1. [Database Schema Specifications](#1-database-schema-specifications)
2. [N8N Workflow Specifications](#2-n8n-workflow-specifications)
3. [Video Processing Pipeline](#3-video-processing-pipeline)
4. [Security & Authentication](#4-security--authentication)
5. [Content Strategy Algorithm](#5-content-strategy-algorithm)

---

## 1. Database Schema Specifications

### 1.1 Complete PostgreSQL Schema

```sql
-- =============================================
-- YTEMPIRE MVP Database Schema
-- PostgreSQL 14+ with extensions
-- =============================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================
-- USER MANAGEMENT SCHEMA
-- =============================================

CREATE SCHEMA IF NOT EXISTS users;

-- Core users table
CREATE TABLE users.accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    company_name VARCHAR(255),
    phone VARCHAR(50),
    
    -- Subscription information
    stripe_customer_id VARCHAR(255) UNIQUE,
    subscription_tier VARCHAR(50) DEFAULT 'free', -- free, starter, growth, scale
    subscription_status VARCHAR(50) DEFAULT 'inactive', -- active, trialing, past_due, canceled
    subscription_id VARCHAR(255),
    trial_ends_at TIMESTAMP,
    
    -- Account limits based on tier
    channel_limit INTEGER DEFAULT 5,
    daily_video_limit INTEGER DEFAULT 15,
    
    -- Account status
    is_active BOOLEAN DEFAULT TRUE,
    is_beta_user BOOLEAN DEFAULT FALSE,
    onboarding_completed BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login_at TIMESTAMP WITH TIME ZONE,
    
    -- Indexes
    INDEX idx_users_email (email),
    INDEX idx_users_stripe (stripe_customer_id),
    INDEX idx_users_subscription (subscription_status)
);

-- User sessions for authentication
CREATE TABLE users.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.accounts(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) UNIQUE NOT NULL,
    refresh_token_hash VARCHAR(255) UNIQUE,
    
    ip_address INET,
    user_agent TEXT,
    device_id VARCHAR(255),
    
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    refresh_expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_sessions_token (token_hash),
    INDEX idx_sessions_user (user_id),
    INDEX idx_sessions_expires (expires_at)
);

-- User preferences and settings
CREATE TABLE users.preferences (
    user_id UUID PRIMARY KEY REFERENCES users.accounts(id) ON DELETE CASCADE,
    
    -- Notification preferences
    email_notifications JSONB DEFAULT '{
        "video_complete": true,
        "weekly_report": true,
        "payment_issues": true,
        "system_updates": false
    }'::jsonb,
    
    -- Dashboard preferences
    dashboard_layout JSONB DEFAULT '{"view": "grid", "theme": "light"}'::jsonb,
    timezone VARCHAR(50) DEFAULT 'UTC',
    language VARCHAR(10) DEFAULT 'en',
    
    -- Content preferences
    default_video_style VARCHAR(50) DEFAULT 'educational',
    default_video_length INTEGER DEFAULT 480, -- seconds
    auto_publish BOOLEAN DEFAULT FALSE,
    
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================
-- CHANNEL MANAGEMENT SCHEMA
-- =============================================

CREATE SCHEMA IF NOT EXISTS channels;

-- YouTube channels managed by users
CREATE TABLE channels.youtube_channels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.accounts(id) ON DELETE CASCADE,
    
    -- YouTube information
    youtube_channel_id VARCHAR(255) UNIQUE NOT NULL,
    youtube_channel_handle VARCHAR(255),
    channel_title VARCHAR(255) NOT NULL,
    channel_description TEXT,
    
    -- Channel configuration
    niche VARCHAR(100) NOT NULL,
    sub_niche VARCHAR(100),
    target_audience JSONB,
    content_strategy JSONB,
    
    -- Automation settings
    automation_enabled BOOLEAN DEFAULT TRUE,
    auto_publish BOOLEAN DEFAULT FALSE,
    publish_schedule JSONB, -- {"days": ["mon", "wed", "fri"], "times": ["09:00", "15:00"]}
    videos_per_day INTEGER DEFAULT 3,
    
    -- Performance metrics
    subscriber_count INTEGER DEFAULT 0,
    total_views BIGINT DEFAULT 0,
    total_videos INTEGER DEFAULT 0,
    monetization_enabled BOOLEAN DEFAULT FALSE,
    estimated_monthly_revenue DECIMAL(10,2) DEFAULT 0.00,
    
    -- Status
    status VARCHAR(50) DEFAULT 'active', -- active, paused, suspended, deleted
    health_score DECIMAL(3,2) DEFAULT 1.00, -- 0.00 to 1.00
    last_video_at TIMESTAMP WITH TIME ZONE,
    
    -- OAuth credentials (encrypted)
    oauth_credentials JSONB, -- Encrypted OAuth tokens
    oauth_refresh_token TEXT, -- Encrypted
    oauth_expires_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_channels_user (user_id),
    INDEX idx_channels_status (status),
    INDEX idx_channels_youtube (youtube_channel_id)
);

-- Channel performance history
CREATE TABLE channels.performance_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    channel_id UUID NOT NULL REFERENCES channels.youtube_channels(id) ON DELETE CASCADE,
    
    date DATE NOT NULL,
    
    -- Daily metrics
    views INTEGER DEFAULT 0,
    likes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    shares INTEGER DEFAULT 0,
    watch_time_minutes DECIMAL(10,2) DEFAULT 0.00,
    
    -- Growth metrics
    subscribers_gained INTEGER DEFAULT 0,
    subscribers_lost INTEGER DEFAULT 0,
    
    -- Revenue metrics
    estimated_revenue DECIMAL(10,2) DEFAULT 0.00,
    ad_revenue DECIMAL(10,2) DEFAULT 0.00,
    affiliate_revenue DECIMAL(10,2) DEFAULT 0.00,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(channel_id, date),
    INDEX idx_performance_channel_date (channel_id, date DESC)
);

-- =============================================
-- VIDEO MANAGEMENT SCHEMA
-- =============================================

CREATE SCHEMA IF NOT EXISTS videos;

-- Video generation and tracking
CREATE TABLE videos.video_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    channel_id UUID NOT NULL REFERENCES channels.youtube_channels(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users.accounts(id) ON DELETE CASCADE,
    
    -- Video metadata
    title VARCHAR(500) NOT NULL,
    description TEXT,
    tags TEXT[], -- Array of tags
    category_id VARCHAR(50),
    
    -- Content
    script TEXT,
    script_model VARCHAR(50), -- gpt-3.5-turbo, gpt-4, etc.
    voice_provider VARCHAR(50), -- google_tts, elevenlabs
    voice_id VARCHAR(100),
    
    -- File references
    video_file_path VARCHAR(500),
    thumbnail_file_path VARCHAR(500),
    audio_file_path VARCHAR(500),
    
    -- YouTube information
    youtube_video_id VARCHAR(255) UNIQUE,
    youtube_url VARCHAR(500),
    youtube_upload_status VARCHAR(50), -- pending, uploading, processing, published, failed
    
    -- Processing information
    status VARCHAR(50) DEFAULT 'queued', -- queued, processing, completed, failed, published
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    processing_duration_seconds INTEGER,
    
    -- Quality metrics
    quality_score DECIMAL(3,2), -- 0.00 to 1.00
    compliance_score DECIMAL(3,2), -- 0.00 to 1.00
    
    -- Scheduling
    scheduled_publish_at TIMESTAMP WITH TIME ZONE,
    published_at TIMESTAMP WITH TIME ZONE,
    
    -- Error tracking
    error_message TEXT,
    error_count INTEGER DEFAULT 0,
    last_error_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_videos_channel (channel_id),
    INDEX idx_videos_user (user_id),
    INDEX idx_videos_status (status),
    INDEX idx_videos_youtube (youtube_video_id),
    INDEX idx_videos_created (created_at DESC)
);

-- Video performance metrics
CREATE TABLE videos.performance_metrics (
    video_id UUID PRIMARY KEY REFERENCES videos.video_records(id) ON DELETE CASCADE,
    
    -- YouTube Analytics
    views INTEGER DEFAULT 0,
    likes INTEGER DEFAULT 0,
    dislikes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    shares INTEGER DEFAULT 0,
    
    -- Engagement metrics
    average_view_duration_seconds DECIMAL(10,2),
    average_percentage_viewed DECIMAL(5,2),
    click_through_rate DECIMAL(5,2),
    
    -- Revenue metrics
    estimated_revenue DECIMAL(10,2) DEFAULT 0.00,
    ad_impressions INTEGER DEFAULT 0,
    
    -- Time-based metrics
    views_first_hour INTEGER DEFAULT 0,
    views_first_day INTEGER DEFAULT 0,
    views_first_week INTEGER DEFAULT 0,
    
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================
-- COST TRACKING SCHEMA
-- =============================================

CREATE SCHEMA IF NOT EXISTS costs;

-- Detailed cost tracking per video
CREATE TABLE costs.video_costs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id UUID NOT NULL REFERENCES videos.video_records(id) ON DELETE CASCADE,
    
    -- Service costs
    script_generation_cost DECIMAL(10,4) DEFAULT 0.0000,
    voice_synthesis_cost DECIMAL(10,4) DEFAULT 0.0000,
    video_processing_cost DECIMAL(10,4) DEFAULT 0.0000,
    thumbnail_generation_cost DECIMAL(10,4) DEFAULT 0.0000,
    storage_cost DECIMAL(10,4) DEFAULT 0.0000,
    
    -- API costs
    openai_cost DECIMAL(10,4) DEFAULT 0.0000,
    elevenlabs_cost DECIMAL(10,4) DEFAULT 0.0000,
    google_tts_cost DECIMAL(10,4) DEFAULT 0.0000,
    youtube_api_cost DECIMAL(10,4) DEFAULT 0.0000,
    
    -- Totals
    total_cost DECIMAL(10,4) DEFAULT 0.0000,
    
    -- Cost optimization level used
    optimization_level VARCHAR(20) DEFAULT 'standard', -- economy, standard, premium
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_cost_video (video_id),
    INDEX idx_cost_created (created_at DESC)
);

-- Daily cost aggregations
CREATE TABLE costs.daily_costs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    user_id UUID REFERENCES users.accounts(id) ON DELETE CASCADE,
    
    -- Service totals
    total_openai_cost DECIMAL(10,2) DEFAULT 0.00,
    total_tts_cost DECIMAL(10,2) DEFAULT 0.00,
    total_storage_cost DECIMAL(10,2) DEFAULT 0.00,
    total_processing_cost DECIMAL(10,2) DEFAULT 0.00,
    
    -- Counts
    videos_generated INTEGER DEFAULT 0,
    videos_published INTEGER DEFAULT 0,
    
    -- Averages
    average_cost_per_video DECIMAL(10,4) DEFAULT 0.0000,
    
    -- Budget tracking
    daily_budget DECIMAL(10,2) DEFAULT 50.00,
    budget_remaining DECIMAL(10,2),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(date, user_id),
    INDEX idx_daily_costs_date (date DESC),
    INDEX idx_daily_costs_user (user_id)
);

-- =============================================
-- PAYMENT & SUBSCRIPTION SCHEMA
-- =============================================

CREATE SCHEMA IF NOT EXISTS payments;

-- Payment history
CREATE TABLE payments.transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.accounts(id) ON DELETE CASCADE,
    
    -- Stripe information
    stripe_payment_intent_id VARCHAR(255) UNIQUE,
    stripe_invoice_id VARCHAR(255),
    stripe_charge_id VARCHAR(255),
    
    -- Transaction details
    amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    description TEXT,
    
    -- Type and status
    transaction_type VARCHAR(50), -- subscription, one_time, refund
    status VARCHAR(50), -- pending, processing, succeeded, failed, refunded
    
    -- Metadata
    metadata JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_transactions_user (user_id),
    INDEX idx_transactions_stripe (stripe_payment_intent_id),
    INDEX idx_transactions_created (created_at DESC)
);

-- Subscription history
CREATE TABLE payments.subscription_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.accounts(id) ON DELETE CASCADE,
    
    stripe_subscription_id VARCHAR(255),
    
    -- Plan details
    plan_name VARCHAR(50), -- starter, growth, scale
    plan_price DECIMAL(10,2),
    
    -- Period
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    
    -- Status
    status VARCHAR(50), -- active, canceled, expired
    canceled_at TIMESTAMP WITH TIME ZONE,
    cancellation_reason TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_subscription_user (user_id),
    INDEX idx_subscription_stripe (stripe_subscription_id)
);

-- =============================================
-- SYSTEM & MONITORING SCHEMA
-- =============================================

CREATE SCHEMA IF NOT EXISTS system;

-- API request logging
CREATE TABLE system.api_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Request information
    endpoint VARCHAR(255),
    method VARCHAR(10),
    user_id UUID REFERENCES users.accounts(id),
    ip_address INET,
    
    -- Response information
    status_code INTEGER,
    response_time_ms INTEGER,
    
    -- Rate limiting
    rate_limit_remaining INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_api_requests_user (user_id),
    INDEX idx_api_requests_endpoint (endpoint),
    INDEX idx_api_requests_created (created_at DESC)
);

-- System events and audit log
CREATE TABLE system.audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    user_id UUID REFERENCES users.accounts(id),
    event_type VARCHAR(100),
    event_description TEXT,
    
    -- Change tracking
    entity_type VARCHAR(50),
    entity_id UUID,
    old_value JSONB,
    new_value JSONB,
    
    -- Request context
    ip_address INET,
    user_agent TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_audit_user (user_id),
    INDEX idx_audit_type (event_type),
    INDEX idx_audit_entity (entity_type, entity_id),
    INDEX idx_audit_created (created_at DESC)
);

-- =============================================
-- FUNCTIONS AND TRIGGERS
-- =============================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to all relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users.accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_channels_updated_at BEFORE UPDATE ON channels.youtube_channels
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_videos_updated_at BEFORE UPDATE ON videos.video_records
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

---

## 2. N8N Workflow Specifications

### 2.1 Core Video Generation Workflow

```json
{
  "name": "YTEMPIRE_Video_Generation_Pipeline",
  "nodes": [
    {
      "id": "trigger_node",
      "type": "n8n-nodes-base.webhook",
      "position": [250, 300],
      "parameters": {
        "httpMethod": "POST",
        "path": "video-generation",
        "responseMode": "responseNode",
        "options": {
          "responseData": "allEntries",
          "responsePropertyName": "data"
        }
      }
    },
    {
      "id": "validate_request",
      "type": "n8n-nodes-base.function",
      "position": [450, 300],
      "parameters": {
        "functionCode": "// Validate incoming request\nconst required = ['channel_id', 'topic', 'style'];\nconst data = items[0].json;\n\nfor (const field of required) {\n  if (!data[field]) {\n    throw new Error(`Missing required field: ${field}`);\n  }\n}\n\n// Add metadata\ndata.workflow_id = $executionId;\ndata.timestamp = new Date().toISOString();\ndata.status = 'validated';\n\nreturn [{json: data}];"
      }
    },
    {
      "id": "generate_script",
      "type": "n8n-nodes-base.httpRequest",
      "position": [650, 300],
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8000/api/v1/ai/generate-script",
        "authentication": "predefinedCredentialType",
        "nodeCredentialType": "httpBasicAuth",
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "topic",
              "value": "={{$node[\"validate_request\"].json[\"topic\"]}}"
            },
            {
              "name": "style",
              "value": "={{$node[\"validate_request\"].json[\"style\"]}}"
            },
            {
              "name": "optimization_level",
              "value": "={{$node[\"validate_request\"].json[\"optimization_level\"] || 'standard'}}"
            }
          ]
        }
      }
    },
    {
      "id": "check_script_cost",
      "type": "n8n-nodes-base.if",
      "position": [850, 300],
      "parameters": {
        "conditions": {
          "number": [
            {
              "value1": "={{$node[\"generate_script\"].json[\"cost\"]}}",
              "operation": "smaller",
              "value2": 0.5
            }
          ]
        }
      }
    },
    {
      "id": "generate_voice",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1050, 200],
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8000/api/v1/tts/synthesize",
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "text",
              "value": "={{$node[\"generate_script\"].json[\"script\"]}}"
            },
            {
              "name": "voice_provider",
              "value": "google_tts"
            }
          ]
        }
      }
    },
    {
      "id": "generate_video",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1250, 200],
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8000/api/v1/video/render",
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "script",
              "value": "={{$node[\"generate_script\"].json[\"script\"]}}"
            },
            {
              "name": "audio_url",
              "value": "={{$node[\"generate_voice\"].json[\"audio_url\"]}}"
            },
            {
              "name": "style",
              "value": "={{$node[\"validate_request\"].json[\"style\"]}}"
            }
          ]
        }
      }
    },
    {
      "id": "upload_youtube",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1450, 200],
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8000/api/v1/youtube/upload",
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "video_url",
              "value": "={{$node[\"generate_video\"].json[\"video_url\"]}}"
            },
            {
              "name": "title",
              "value": "={{$node[\"generate_script\"].json[\"title\"]}}"
            },
            {
              "name": "description",
              "value": "={{$node[\"generate_script\"].json[\"description\"]}}"
            },
            {
              "name": "channel_id",
              "value": "={{$node[\"validate_request\"].json[\"channel_id\"]}}"
            }
          ]
        }
      }
    },
    {
      "id": "update_database",
      "type": "n8n-nodes-base.postgres",
      "position": [1650, 200],
      "parameters": {
        "operation": "executeQuery",
        "query": "UPDATE videos.video_records SET status = 'published', youtube_video_id = $1, published_at = NOW() WHERE id = $2",
        "additionalFields": {
          "queryParams": "={{$node[\"upload_youtube\"].json[\"youtube_id\"]}},{{$node[\"validate_request\"].json[\"video_id\"]}}"
        }
      }
    },
    {
      "id": "cost_optimization",
      "type": "n8n-nodes-base.function",
      "position": [1050, 400],
      "parameters": {
        "functionCode": "// Switch to economy mode\nconst data = items[0].json;\ndata.optimization_level = 'economy';\ndata.fallback_triggered = true;\nreturn [{json: data}];"
      }
    },
    {
      "id": "error_handler",
      "type": "n8n-nodes-base.function",
      "position": [850, 500],
      "parameters": {
        "functionCode": "// Log error and trigger alerts\nconst error = items[0].json;\n\n// Send to error tracking\nawait $http.post('http://localhost:8000/api/v1/errors', {\n  workflow_id: $executionId,\n  error: error,\n  timestamp: new Date().toISOString()\n});\n\nreturn [{json: {status: 'error', details: error}}];"
      }
    }
  ],
  "connections": {
    "trigger_node": {
      "main": [["validate_request"]]
    },
    "validate_request": {
      "main": [["generate_script"]]
    },
    "generate_script": {
      "main": [["check_script_cost"]]
    },
    "check_script_cost": {
      "main": [
        ["generate_voice"],
        ["cost_optimization"]
      ]
    },
    "generate_voice": {
      "main": [["generate_video"]]
    },
    "generate_video": {
      "main": [["upload_youtube"]]
    },
    "upload_youtube": {
      "main": [["update_database"]]
    },
    "cost_optimization": {
      "main": [["generate_script"]]
    }
  },
  "settings": {
    "executionOrder": "v1",
    "saveDataSuccessExecution": "all",
    "saveExecutionProgress": true,
    "saveManualExecutions": true,
    "callerPolicy": "workflowsFromSameOwner",
    "errorWorkflow": "error_handler_workflow_id"
  }
}
```

### 2.2 Daily Scheduling Workflow

```javascript
// N8N Daily Video Scheduling Workflow
const schedulingWorkflow = {
  name: "Daily_Video_Scheduler",
  trigger: {
    type: "cron",
    expression: "0 6,12,18 * * *", // Run at 6 AM, 12 PM, 6 PM
  },
  
  nodes: [
    {
      name: "Load Active Channels",
      type: "database",
      query: `
        SELECT c.*, u.subscription_tier
        FROM channels.youtube_channels c
        JOIN users.accounts u ON c.user_id = u.id
        WHERE c.status = 'active'
        AND c.automation_enabled = true
        ORDER BY c.health_score DESC
      `
    },
    {
      name: "Calculate Video Distribution",
      type: "function",
      code: `
        // Distribute 50 videos across channels
        const channels = items;
        const totalVideos = 50;
        const distribution = [];
        
        // Prioritize by health score and tier
        channels.forEach(channel => {
          const tierMultiplier = {
            'scale': 3,
            'growth': 2,
            'starter': 1
          }[channel.subscription_tier] || 1;
          
          const videosForChannel = Math.min(
            channel.videos_per_day,
            Math.floor(channel.health_score * tierMultiplier * 2)
          );
          
          distribution.push({
            channel_id: channel.id,
            videos_to_generate: videosForChannel,
            priority: channel.health_score * tierMultiplier
          });
        });
        
        return distribution;
      `
    },
    {
      name: "Queue Video Generation",
      type: "loop",
      forEach: "channel",
      operations: [
        {
          name: "Generate Topics",
          type: "http",
          url: "http://localhost:8000/api/v1/ai/generate-topics",
          body: {
            channel_id: "{{channel.channel_id}}",
            count: "{{channel.videos_to_generate}}",
            niche: "{{channel.niche}}"
          }
        },
        {
          name: "Create Generation Jobs",
          type: "http",
          url: "http://localhost:8000/api/v1/videos/queue",
          body: {
            channel_id: "{{channel.channel_id}}",
            topics: "{{topics}}",
            priority: "{{channel.priority}}"
          }
        }
      ]
    }
  ]
};
```

---

## 3. Video Processing Pipeline

### 3.1 Video Format Specifications

```python
class VideoSpecifications:
    """
    Complete video format and processing specifications
    """
    
    # Output format specifications
    VIDEO_FORMATS = {
        'standard': {
            'resolution': '1920x1080',  # Full HD
            'fps': 30,
            'codec': 'h264',
            'bitrate': '5000k',
            'audio_codec': 'aac',
            'audio_bitrate': '192k',
            'container': 'mp4'
        },
        'short': {
            'resolution': '1080x1920',  # Vertical for Shorts
            'fps': 30,
            'codec': 'h264',
            'bitrate': '4000k',
            'audio_codec': 'aac',
            'audio_bitrate': '192k',
            'container': 'mp4',
            'max_duration': 60  # seconds
        },
        'economy': {
            'resolution': '1280x720',  # HD
            'fps': 24,
            'codec': 'h264',
            'bitrate': '2500k',
            'audio_codec': 'aac',
            'audio_bitrate': '128k',
            'container': 'mp4'
        }
    }
    
    # Thumbnail specifications
    THUMBNAIL_SPECS = {
        'resolution': '1280x720',
        'format': 'jpg',
        'quality': 95,
        'max_file_size': '2MB'
    }
    
    # Video length targets
    VIDEO_LENGTHS = {
        'short': (30, 60),      # 30-60 seconds
        'standard': (300, 600),  # 5-10 minutes
        'long': (600, 1200)      # 10-20 minutes
    }
```

### 3.2 GPU Video Processing Implementation

```python
import subprocess
import os
from pathlib import Path
import torch
import cv2
import numpy as np

class GPUVideoProcessor:
    """
    GPU-accelerated video processing using RTX 5090
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_memory_limit = 28 * 1024 * 1024 * 1024  # 28GB usable
        
    async def render_video(
        self,
        script: str,
        audio_path: str,
        assets: dict,
        style: str = 'standard'
    ) -> dict:
        """
        Render video using GPU acceleration
        """
        
        # Determine rendering complexity
        complexity = self._calculate_complexity(script, style)
        
        if complexity == 'simple':
            return await self._render_simple_video(script, audio_path, assets)
        else:
            return await self._render_complex_video(script, audio_path, assets)
    
    async def _render_simple_video(
        self,
        script: str,
        audio_path: str,
        assets: dict
    ) -> dict:
        """
        CPU-based simple video rendering (slideshow style)
        """
        
        output_path = f"/tmp/videos/{uuid.uuid4()}.mp4"
        
        # Create video using FFmpeg (CPU)
        command = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-loop', '1',  # Loop images
            '-framerate', '1',  # 1 fps for slideshow
            '-i', assets['images'][0],  # Input images
            '-i', audio_path,  # Audio
            '-c:v', 'libx264',  # Video codec
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',  # Audio codec
            '-b:a', '192k',
            '-shortest',  # Match audio duration
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg error: {stderr.decode()}")
        
        return {
            'video_path': output_path,
            'duration': self._get_video_duration(output_path),
            'size': os.path.getsize(output_path),
            'complexity': 'simple'
        }
    
    async def _render_complex_video(
        self,
        script: str,
        audio_path: str,
        assets: dict
    ) -> dict:
        """
        GPU-accelerated complex video rendering
        """
        
        output_path = f"/tmp/videos/{uuid.uuid4()}.mp4"
        
        # Use NVIDIA hardware encoding
        command = [
            'ffmpeg',
            '-y',
            '-hwaccel', 'cuda',  # GPU acceleration
            '-hwaccel_output_format', 'cuda',
            '-i', assets['video_clips'][0],  # Input video
            '-i', audio_path,  # Audio
            '-c:v', 'h264_nvenc',  # NVIDIA encoder
            '-preset', 'p4',  # Quality preset
            '-tune', 'hq',
            '-b:v', '5M',  # Bitrate
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            output_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg GPU error: {stderr.decode()}")
        
        return {
            'video_path': output_path,
            'duration': self._get_video_duration(output_path),
            'size': os.path.getsize(output_path),
            'complexity': 'complex',
            'gpu_used': True
        }
```

### 3.3 Thumbnail Generation

```python
from PIL import Image, ImageDraw, ImageFont
import random

class ThumbnailGenerator:
    """
    AI-powered thumbnail generation
    """
    
    def __init__(self):
        self.templates = self._load_templates()
        self.fonts = self._load_fonts()
        
    async def generate_thumbnail(
        self,
        title: str,
        style: str = 'standard',
        background_image: str = None
    ) -> str:
        """
        Generate engaging thumbnail
        """
        
        # Create base image
        img = Image.new('RGB', (1280, 720), color='white')
        
        if background_image:
            bg = Image.open(background_image)
            bg = bg.resize((1280, 720), Image.LANCZOS)
            img.paste(bg, (0, 0))
        
        # Add overlay
        overlay = Image.new('RGBA', (1280, 720), (0, 0, 0, 100))
        img.paste(overlay, (0, 0), overlay)
        
        # Add text
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(self.fonts['bold'], 72)
        
        # Word wrap title
        words = title.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            test_line = ' '.join(current_line)
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] > 1100:  # Max width with padding
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw text with shadow
        y_offset = (720 - len(lines) * 80) // 2
        for line in lines:
            # Shadow
            draw.text((102, y_offset + 2), line, font=font, fill='black')
            # Main text
            draw.text((100, y_offset), line, font=font, fill='white')
            y_offset += 80
        
        # Save thumbnail
        output_path = f"/tmp/thumbnails/{uuid.uuid4()}.jpg"
        img.save(output_path, 'JPEG', quality=95, optimize=True)
        
        return output_path
```

---

## 4. Security & Authentication

### 4.1 JWT Authentication System

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
import bcrypt

class AuthenticationSystem:
    """
    Complete JWT-based authentication system
    """
    
    def __init__(self):
        self.secret_key = os.environ.get('JWT_SECRET_KEY')
        self.algorithm = 'HS256'
        self.access_token_expire = 15  # minutes
        self.refresh_token_expire = 7  # days
        
    def hash_password(self, password: str) -> str:
        """
        Hash password using bcrypt
        """
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify password against hash
        """
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    
    def create_access_token(self, user_id: str, email: str) -> str:
        """
        Create JWT access token
        """
        payload = {
            'sub': user_id,
            'email': email,
            'type': 'access',
            'exp': datetime.utcnow() + timedelta(minutes=self.access_token_expire),
            'iat': datetime.utcnow(),
            'jti': str(uuid.uuid4())
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """
        Create JWT refresh token
        """
        payload = {
            'sub': user_id,
            'type': 'refresh',
            'exp': datetime.utcnow() + timedelta(days=self.refresh_token_expire),
            'iat': datetime.utcnow(),
            'jti': str(uuid.uuid4())
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ):
        """
        Dependency to get current authenticated user
        """
        token = credentials.credentials
        
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            if payload.get('type') != 'access':
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            user_id = payload.get('sub')
            
            # Get user from database
            user = await self.get_user_by_id(user_id)
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            return user
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
```

### 4.2 API Security Configuration

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

class APISecurityConfig:
    """
    API security configuration and middleware
    """
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.setup_cors()
        self.setup_rate_limiting()
        self.setup_security_headers()
        
    def setup_cors(self):
        """
        Configure CORS settings
        """
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",
                "https://app.ytempire.com"
            ],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-Total-Count", "X-Page", "X-Per-Page"]
        )
    
    def setup_rate_limiting(self):
        """
        Configure rate limiting
        """
        limiter = Limiter(key_func=get_remote_address)
        self.app.state.limiter = limiter
        self.app.add_exception_handler(429, _rate_limit_exceeded_handler)
        
        # Rate limit rules by tier
        self.rate_limits = {
            'free': '60/minute',
            'starter': '100/minute',
            'growth': '300/minute',
            'scale': '1000/minute'
        }
    
    def setup_security_headers(self):
        """
        Add security headers to all responses
        """
        @self.app.middleware("http")
        async def add_security_headers(request: Request, call_next):
            response = await call_next(request)
            
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Content-Security-Policy"] = "default-src 'self'"
            
            return response
```

---

## 5. Content Strategy Algorithm

### 5.1 Niche Selection Algorithm

```python
class NicheSelectionAlgorithm:
    """
    AI-powered niche selection for profitable content
    """
    
    def __init__(self):
        self.profitable_niches = self._load_niche_database()
        self.trend_analyzer = TrendAnalyzer()
        
    async def select_profitable_niches(
        self,
        user_interests: list,
        budget: float,
        experience_level: str
    ) -> list:
        """
        Select 5 profitable niches for user
        """
        
        # Analyze current trends
        trending_topics = await self.trend_analyzer.get_trending_topics()
        
        # Score niches based on multiple factors
        niche_scores = []
        
        for niche in self.profitable_niches:
            score = 0
            
            # Competition score (lower is better)
            competition = await self._analyze_competition(niche['name'])
            score += (1 - competition) * 30
            
            # Monetization potential
            monetization = niche['monetization_potential']
            score += monetization * 25
            
            # Growth rate
            growth = await self._analyze_growth_rate(niche['name'])
            score += growth * 20
            
            # User interest alignment
            interest_match = self._calculate_interest_match(
                niche['name'],
                user_interests
            )
            score += interest_match * 15
            
            # Trend alignment
            trend_score = self._calculate_trend_score(
                niche['name'],
                trending_topics
            )
            score += trend_score * 10
            
            niche_scores.append({
                'niche': niche['name'],
                'score': score,
                'difficulty': niche['difficulty'],
                'estimated_revenue': niche['estimated_monthly_revenue']
            })
        
        # Sort by score and filter by experience level
        niche_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Filter based on experience
        if experience_level == 'beginner':
            niche_scores = [n for n in niche_scores if n['difficulty'] <= 3]
        elif experience_level == 'intermediate':
            niche_scores = [n for n in niche_scores if n['difficulty'] <= 7]
        
        return niche_scores[:5]
    
    async def _analyze_competition(self, niche: str) -> float:
        """
        Analyze competition level (0-1, higher is more competitive)
        """
        # Simplified - would use YouTube API in production
        competition_keywords = {
            'technology': 0.8,
            'finance': 0.9,
            'health': 0.7,
            'education': 0.6,
            'entertainment': 0.85,
            'cooking': 0.65,
            'travel': 0.75,
            'gaming': 0.9,
            'diy': 0.5,
            'pets': 0.6
        }
        
        return competition_keywords.get(niche.lower(), 0.7)
```

### 5.2 Content Calendar Generation

```python
class ContentCalendarGenerator:
    """
    Generates optimized content calendar for channels
    """
    
    def __init__(self):
        self.best_times = self._load_optimal_posting_times()
        
    async def generate_calendar(
        self,
        channel_id: str,
        days: int = 30
    ) -> dict:
        """
        Generate 30-day content calendar
        """
        
        # Get channel configuration
        channel = await self.get_channel_config(channel_id)
        
        calendar = {
            'channel_id': channel_id,
            'start_date': datetime.now().date(),
            'end_date': (datetime.now() + timedelta(days=days)).date(),
            'scheduled_videos': []
        }
        
        # Distribute videos across the month
        videos_per_day = channel['videos_per_day']
        
        for day in range(days):
            date = datetime.now() + timedelta(days=day)
            
            # Skip certain days based on strategy
            if date.weekday() in [0, 6] and channel['niche'] == 'business':
                continue  # Skip Sunday/Monday for business content
            
            # Schedule videos for this day
            for slot in range(min(videos_per_day, 3)):
                publish_time = self._get_optimal_time(
                    channel['niche'],
                    channel['target_audience'],
                    slot
                )
                
                calendar['scheduled_videos'].append({
                    'date': date.date(),
                    'time': publish_time,
                    'topic': None,  # To be generated
                    'status': 'scheduled'
                })
        
        return calendar
    
    def _get_optimal_time(
        self,
        niche: str,
        audience: dict,
        slot: int
    ) -> str:
        """
        Determine optimal posting time
        """
        
        # Best times by niche (simplified)
        niche_times = {
            'technology': ['09:00', '14:00', '19:00'],
            'finance': ['07:00', '12:00', '17:00'],
            'entertainment': ['12:00', '18:00', '21:00'],
            'education': ['08:00', '15:00', '20:00'],
            'cooking': ['11:00', '17:00', '19:00']
        }
        
        times = niche_times.get(niche, ['09:00', '14:00', '20:00'])
        return times[min(slot, len(times) - 1)]
```

---

## Key Implementation Notes for Integration Specialist

### Database Integration
1. **Connection Pooling**: Use asyncpg with connection pool (min=10, max=100)
2. **Migrations**: Use Alembic for schema versioning
3. **Backup**: Daily backups with 7-day retention
4. **Monitoring**: Track slow queries > 1 second

### N8N Workflow Management
1. **Deployment**: Run N8N in Docker container
2. **Scaling**: Maximum 20 concurrent workflows
3. **Error Handling**: All workflows must have error nodes
4. **Monitoring**: Track execution time and success rates

### Video Processing
1. **GPU Utilization**: Monitor VRAM usage, stay under 28GB
2. **Concurrent Jobs**: Max 3 GPU jobs, 4 CPU jobs
3. **Storage**: Clean up temp files after 24 hours
4. **Quality Checks**: Validate all videos before upload

### Security Implementation
1. **Token Rotation**: Refresh tokens every 7 days
2. **Rate Limiting**: Implement per-tier limits
3. **Audit Logging**: Log all API calls and state changes
4. **Encryption**: Use AES-256 for sensitive data

### Content Strategy
1. **Niche Updates**: Refresh niche database weekly
2. **Performance Tracking**: Monitor video performance daily
3. **A/B Testing**: Test different posting times
4. **Optimization**: Adjust strategy based on metrics

---

**Document Status**: Complete
**Next Steps**: Implement database migrations, configure N8N workflows, set up GPU processing
**Support**: Contact Backend Team Lead for clarifications