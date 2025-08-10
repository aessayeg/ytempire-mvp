"""Initial migration

Revision ID: 001
Revises: 
Create Date: 2025-01-10

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
    """Create initial tables"""
    
    # Create users table
    op.create_table('users',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('username', sa.String(), nullable=False),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('full_name', sa.String(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('is_superuser', sa.Boolean(), nullable=False, default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    
    # Create channels table
    op.create_table('channels',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('youtube_channel_id', sa.String(), nullable=False),
        sa.Column('channel_name', sa.String(), nullable=False),
        sa.Column('channel_handle', sa.String(), nullable=True),
        sa.Column('niche', sa.String(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_channels_youtube_channel_id'), 'channels', ['youtube_channel_id'], unique=True)
    
    # Create videos table
    op.create_table('videos',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('channel_id', sa.UUID(), nullable=False),
        sa.Column('youtube_video_id', sa.String(), nullable=True),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('script', sa.Text(), nullable=True),
        sa.Column('status', sa.String(), nullable=False, default='pending'),
        sa.Column('generation_cost', sa.Float(), nullable=True),
        sa.Column('view_count', sa.Integer(), nullable=True, default=0),
        sa.Column('like_count', sa.Integer(), nullable=True, default=0),
        sa.Column('comment_count', sa.Integer(), nullable=True, default=0),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['channel_id'], ['channels.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_videos_status'), 'videos', ['status'])
    op.create_index(op.f('ix_videos_youtube_video_id'), 'videos', ['youtube_video_id'])
    
    # Create costs table
    op.create_table('costs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('video_id', sa.UUID(), nullable=True),
        sa.Column('service', sa.String(), nullable=False),
        sa.Column('amount', sa.Float(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_costs_user_id'), 'costs', ['user_id'])
    op.create_index(op.f('ix_costs_created_at'), 'costs', ['created_at'])
    
    # Create api_keys table
    op.create_table('api_keys',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('key_hash', sa.String(), nullable=False),
        sa.Column('last_used', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_api_keys_key_hash'), 'api_keys', ['key_hash'], unique=True)
    
    # Create analytics table
    op.create_table('analytics',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('video_id', sa.UUID(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('views', sa.Integer(), nullable=False, default=0),
        sa.Column('watch_time_minutes', sa.Float(), nullable=True),
        sa.Column('subscribers_gained', sa.Integer(), nullable=True),
        sa.Column('revenue', sa.Float(), nullable=True),
        sa.Column('impressions', sa.Integer(), nullable=True),
        sa.Column('click_through_rate', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_analytics_video_date'), 'analytics', ['video_id', 'date'], unique=True)


def downgrade() -> None:
    """Drop all tables"""
    op.drop_index(op.f('ix_analytics_video_date'), table_name='analytics')
    op.drop_table('analytics')
    op.drop_index(op.f('ix_api_keys_key_hash'), table_name='api_keys')
    op.drop_table('api_keys')
    op.drop_index(op.f('ix_costs_created_at'), table_name='costs')
    op.drop_index(op.f('ix_costs_user_id'), table_name='costs')
    op.drop_table('costs')
    op.drop_index(op.f('ix_videos_youtube_video_id'), table_name='videos')
    op.drop_index(op.f('ix_videos_status'), table_name='videos')
    op.drop_table('videos')
    op.drop_index(op.f('ix_channels_youtube_channel_id'), table_name='channels')
    op.drop_table('channels')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')