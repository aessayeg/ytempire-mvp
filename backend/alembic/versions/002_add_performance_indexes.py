"""Add performance indexes for critical queries

Revision ID: 002_performance_indexes
Revises: 001_initial
Create Date: 2025-08-11

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "002_performance_indexes"
down_revision = "653dd9b7e6c7"
branch_labels = None
depends_on = None


def upgrade():
    """Add indexes to improve query performance from 2101ms to <500ms"""

    # Users table indexes
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_is_active", "users", ["is_active"])
    op.create_index("ix_users_created_at", "users", ["created_at"])

    # Channels table indexes
    op.create_index("ix_channels_user_id", "channels", ["user_id"])
    op.create_index(
        "ix_channels_youtube_channel_id", "channels", ["youtube_channel_id"]
    )
    op.create_index("ix_channels_is_active", "channels", ["is_active"])
    op.create_index("ix_channels_user_active", "channels", ["user_id", "is_active"])

    # Videos table indexes
    op.create_index("ix_videos_channel_id", "videos", ["channel_id"])
    op.create_index("ix_videos_status", "videos", ["status"])
    op.create_index("ix_videos_created_at", "videos", ["created_at"])
    op.create_index("ix_videos_youtube_video_id", "videos", ["youtube_video_id"])
    op.create_index("ix_videos_channel_status", "videos", ["channel_id", "status"])
    op.create_index("ix_videos_channel_created", "videos", ["channel_id", "created_at"])

    # Partial index for pending videos (most queried)
    op.execute(
        """
        CREATE INDEX ix_videos_pending 
        ON videos(status, created_at) 
        WHERE status IN ('pending', 'processing')
    """
    )

    # Analytics table indexes
    op.create_index("ix_analytics_channel_id", "analytics", ["channel_id"])
    op.create_index("ix_analytics_video_id", "analytics", ["video_id"])
    op.create_index("ix_analytics_date", "analytics", ["date"])
    op.create_index("ix_analytics_metric_type", "analytics", ["metric_type"])
    op.create_index("ix_analytics_channel_date", "analytics", ["channel_id", "date"])

    # Costs table indexes
    op.create_index("ix_costs_video_id", "costs", ["video_id"])
    op.create_index("ix_costs_service", "costs", ["service"])
    op.create_index("ix_costs_created_at", "costs", ["created_at"])
    op.create_index("ix_costs_video_service", "costs", ["video_id", "service"])

    # Payments table indexes (if exists)
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'payments') THEN
                CREATE INDEX IF NOT EXISTS ix_payments_user_id ON payments(user_id);
                CREATE INDEX IF NOT EXISTS ix_payments_status ON payments(status);
                CREATE INDEX IF NOT EXISTS ix_payments_created_at ON payments(created_at);
            END IF;
        END $$;
    """
    )

    # YouTube accounts table indexes (if exists)
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'youtube_accounts') THEN
                CREATE INDEX IF NOT EXISTS ix_youtube_accounts_is_active ON youtube_accounts(is_active);
                CREATE INDEX IF NOT EXISTS ix_youtube_accounts_health_score ON youtube_accounts(health_score);
                CREATE INDEX IF NOT EXISTS ix_youtube_accounts_quota_used ON youtube_accounts(quota_used);
                CREATE INDEX IF NOT EXISTS ix_youtube_active_health ON youtube_accounts(is_active, health_score DESC)
                    WHERE is_active = true;
            END IF;
        END $$;
    """
    )

    # API logs table indexes for monitoring
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'api_logs') THEN
                CREATE INDEX IF NOT EXISTS ix_api_logs_endpoint ON api_logs(endpoint);
                CREATE INDEX IF NOT EXISTS ix_api_logs_user_id ON api_logs(user_id);
                CREATE INDEX IF NOT EXISTS ix_api_logs_response_time ON api_logs(response_time);
                CREATE INDEX IF NOT EXISTS ix_api_logs_created_at ON api_logs(created_at);
            END IF;
        END $$;
    """
    )

    # Update table statistics for query planner
    op.execute("ANALYZE users;")
    op.execute("ANALYZE channels;")
    op.execute("ANALYZE videos;")
    op.execute("ANALYZE analytics;")
    op.execute("ANALYZE costs;")


def downgrade():
    """Remove performance indexes"""

    # Drop all indexes created above
    op.drop_index("ix_users_email", "users")
    op.drop_index("ix_users_is_active", "users")
    op.drop_index("ix_users_created_at", "users")

    op.drop_index("ix_channels_user_id", "channels")
    op.drop_index("ix_channels_youtube_channel_id", "channels")
    op.drop_index("ix_channels_is_active", "channels")
    op.drop_index("ix_channels_user_active", "channels")

    op.drop_index("ix_videos_channel_id", "videos")
    op.drop_index("ix_videos_status", "videos")
    op.drop_index("ix_videos_created_at", "videos")
    op.drop_index("ix_videos_youtube_video_id", "videos")
    op.drop_index("ix_videos_channel_status", "videos")
    op.drop_index("ix_videos_channel_created", "videos")
    op.drop_index("ix_videos_pending", "videos")

    op.drop_index("ix_analytics_channel_id", "analytics")
    op.drop_index("ix_analytics_video_id", "analytics")
    op.drop_index("ix_analytics_date", "analytics")
    op.drop_index("ix_analytics_metric_type", "analytics")
    op.drop_index("ix_analytics_channel_date", "analytics")

    op.drop_index("ix_costs_video_id", "costs")
    op.drop_index("ix_costs_service", "costs")
    op.drop_index("ix_costs_created_at", "costs")
    op.drop_index("ix_costs_video_service", "costs")

    # Drop conditional indexes
    op.execute("DROP INDEX IF EXISTS ix_payments_user_id;")
    op.execute("DROP INDEX IF EXISTS ix_payments_status;")
    op.execute("DROP INDEX IF EXISTS ix_payments_created_at;")
    op.execute("DROP INDEX IF EXISTS ix_youtube_accounts_is_active;")
    op.execute("DROP INDEX IF EXISTS ix_youtube_accounts_health_score;")
    op.execute("DROP INDEX IF EXISTS ix_youtube_accounts_quota_used;")
    op.execute("DROP INDEX IF EXISTS ix_youtube_active_health;")
    op.execute("DROP INDEX IF EXISTS ix_api_logs_endpoint;")
    op.execute("DROP INDEX IF EXISTS ix_api_logs_user_id;")
    op.execute("DROP INDEX IF EXISTS ix_api_logs_response_time;")
    op.execute("DROP INDEX IF EXISTS ix_api_logs_created_at;")
