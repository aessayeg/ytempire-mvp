"""Initial database schema

Revision ID: 001
Revises: 
Create Date: 2025-01-10 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial database schema."""
    
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('username', sa.String(100), nullable=False),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('is_verified', sa.Boolean(), nullable=False, default=False),
        sa.Column('subscription_tier', sa.Enum('FREE', 'BASIC', 'PREMIUM', 'ENTERPRISE', name='subscriptiontier'), nullable=False, default='FREE'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('youtube_token_encrypted', sa.Text(), nullable=True),
        sa.Column('preferences', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default={}),
        sa.Column('usage_stats', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default={}),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username')
    )
    
    # Create indexes for users table
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_users_username', 'users', ['username'])
    op.create_index('ix_users_subscription_tier', 'users', ['subscription_tier'])
    op.create_index('ix_users_created_at', 'users', ['created_at'])

    # Create channels table
    op.create_table(
        'channels',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('youtube_channel_id', sa.String(100), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('category', sa.String(100), nullable=True),
        sa.Column('target_audience', sa.String(255), nullable=True),
        sa.Column('content_style', sa.String(100), nullable=True),
        sa.Column('upload_schedule', sa.String(100), nullable=True),
        sa.Column('branding', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default={}),
        sa.Column('automation_settings', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default={}),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )
    
    # Create indexes for channels table
    op.create_index('ix_channels_user_id', 'channels', ['user_id'])
    op.create_index('ix_channels_youtube_channel_id', 'channels', ['youtube_channel_id'])
    op.create_index('ix_channels_category', 'channels', ['category'])

    # Create videos table
    op.create_table(
        'videos',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('channel_id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('title', sa.String(500), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('script_content', sa.Text(), nullable=True),
        sa.Column('thumbnail_url', sa.String(500), nullable=True),
        sa.Column('video_url', sa.String(500), nullable=True),
        sa.Column('youtube_video_id', sa.String(100), nullable=True),
        sa.Column('status', sa.Enum('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'SCHEDULED', name='videostatus'), nullable=False, default='PENDING'),
        sa.Column('current_stage', sa.String(100), nullable=True),
        sa.Column('priority', sa.Integer(), nullable=False, default=1),
        sa.Column('scheduled_publish_at', sa.DateTime(), nullable=True),
        sa.Column('published_at', sa.DateTime(), nullable=True),
        sa.Column('content_settings', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default={}),
        sa.Column('generation_settings', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default={}),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default={}),
        sa.Column('tags', sa.ARRAY(sa.String(50)), nullable=True),
        sa.Column('category', sa.String(100), nullable=True),
        sa.Column('duration', sa.Integer(), nullable=True),
        sa.Column('total_cost', sa.Numeric(10, 4), nullable=True),
        sa.Column('pipeline_id', sa.String(100), nullable=True),
        sa.Column('pipeline_started_at', sa.DateTime(), nullable=True),
        sa.Column('pipeline_completed_at', sa.DateTime(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, default=0),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['channel_id'], ['channels.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )
    
    # Create indexes for videos table
    op.create_index('ix_videos_channel_id', 'videos', ['channel_id'])
    op.create_index('ix_videos_user_id', 'videos', ['user_id'])
    op.create_index('ix_videos_status', 'videos', ['status'])
    op.create_index('ix_videos_youtube_video_id', 'videos', ['youtube_video_id'])
    op.create_index('ix_videos_created_at', 'videos', ['created_at'])
    op.create_index('ix_videos_scheduled_publish_at', 'videos', ['scheduled_publish_at'])

    # Create channel_analytics table
    op.create_table(
        'channel_analytics',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('channel_id', sa.String(36), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('views', sa.BigInteger(), nullable=False, default=0),
        sa.Column('subscribers', sa.BigInteger(), nullable=False, default=0),
        sa.Column('videos_published', sa.Integer(), nullable=False, default=0),
        sa.Column('watch_time_minutes', sa.BigInteger(), nullable=False, default=0),
        sa.Column('estimated_revenue', sa.Numeric(10, 2), nullable=False, default=0),
        sa.Column('engagement_rate', sa.Float(), nullable=False, default=0.0),
        sa.Column('click_through_rate', sa.Float(), nullable=False, default=0.0),
        sa.Column('average_view_duration', sa.Float(), nullable=False, default=0.0),
        sa.Column('top_countries', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default={}),
        sa.Column('age_demographics', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default={}),
        sa.Column('device_types', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default={}),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['channel_id'], ['channels.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('channel_id', 'date', name='uq_channel_analytics_date')
    )
    
    # Create indexes for channel_analytics table
    op.create_index('ix_channel_analytics_channel_id', 'channel_analytics', ['channel_id'])
    op.create_index('ix_channel_analytics_date', 'channel_analytics', ['date'])

    # Create video_analytics table
    op.create_table(
        'video_analytics',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('video_id', sa.String(36), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('views', sa.BigInteger(), nullable=False, default=0),
        sa.Column('likes', sa.BigInteger(), nullable=False, default=0),
        sa.Column('dislikes', sa.BigInteger(), nullable=False, default=0),
        sa.Column('comments', sa.BigInteger(), nullable=False, default=0),
        sa.Column('shares', sa.BigInteger(), nullable=False, default=0),
        sa.Column('watch_time_minutes', sa.BigInteger(), nullable=False, default=0),
        sa.Column('impressions', sa.BigInteger(), nullable=False, default=0),
        sa.Column('click_through_rate', sa.Float(), nullable=False, default=0.0),
        sa.Column('average_view_duration', sa.Float(), nullable=False, default=0.0),
        sa.Column('estimated_revenue', sa.Numeric(10, 2), nullable=False, default=0),
        sa.Column('engagement_score', sa.Float(), nullable=False, default=0.0),
        sa.Column('trending_score', sa.Float(), nullable=False, default=0.0),
        sa.Column('traffic_sources', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default={}),
        sa.Column('audience_retention', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default={}),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('video_id', 'date', name='uq_video_analytics_date')
    )
    
    # Create indexes for video_analytics table
    op.create_index('ix_video_analytics_video_id', 'video_analytics', ['video_id'])
    op.create_index('ix_video_analytics_date', 'video_analytics', ['date'])
    op.create_index('ix_video_analytics_views', 'video_analytics', ['views'])

    # Create cost_tracking table
    op.create_table(
        'cost_tracking',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('video_id', sa.String(36), nullable=True),
        sa.Column('service_type', sa.Enum('OPENAI_GPT4', 'ELEVENLABS_TTS', 'CONTENT_GENERATION', 'AUDIO_SYNTHESIS', 'VIDEO_COMPILATION', 'THUMBNAIL_GENERATION', 'YOUTUBE_API', 'STORAGE', 'OTHER', name='servicetype'), nullable=False),
        sa.Column('operation', sa.String(100), nullable=False),
        sa.Column('cost_amount', sa.Numeric(10, 4), nullable=False),
        sa.Column('currency', sa.String(3), nullable=False, default='USD'),
        sa.Column('usage_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True, default={}),
        sa.Column('billing_period', sa.String(10), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE')
    )
    
    # Create indexes for cost_tracking table
    op.create_index('ix_cost_tracking_user_id', 'cost_tracking', ['user_id'])
    op.create_index('ix_cost_tracking_video_id', 'cost_tracking', ['video_id'])
    op.create_index('ix_cost_tracking_service_type', 'cost_tracking', ['service_type'])
    op.create_index('ix_cost_tracking_created_at', 'cost_tracking', ['created_at'])
    op.create_index('ix_cost_tracking_billing_period', 'cost_tracking', ['billing_period'])


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table('cost_tracking')
    op.drop_table('video_analytics')
    op.drop_table('channel_analytics')
    op.drop_table('videos')
    op.drop_table('channels')
    op.drop_table('users')
    
    # Drop custom enums
    op.execute('DROP TYPE IF EXISTS subscriptiontier')
    op.execute('DROP TYPE IF EXISTS videostatus')
    op.execute('DROP TYPE IF EXISTS servicetype')