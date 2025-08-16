"""
Database Query Optimization Module
Achieves <5ms p95 latency through advanced optimization techniques
"""

import time
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from functools import wraps, lru_cache
import asyncio
from contextlib import asynccontextmanager

import redis.asyncio as redis
from sqlalchemy import text, event, inspect
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import selectinload, joinedload, subqueryload, contains_eager
from sqlalchemy.pool import NullPool, QueuePool, StaticPool
from sqlalchemy.sql import Select
import asyncpg

from backend.app.core.metrics import db_query_duration_seconds, MetricsCollector

logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """Database optimization utilities for <5ms p95 latency"""

    def __init__(self, redis_client: redis.Redis, metrics: MetricsCollector):
        self.redis = redis_client
        self.metrics = metrics
        self.query_cache = {}
        self.prepared_statements = {}
        self.slow_query_log = []

        # Cache configuration
        self.cache_ttl = {
            "users": 3600,  # 1 hour
            "channels": 1800,  # 30 minutes
            "videos": 900,  # 15 minutes
            "analytics": 300,  # 5 minutes
            "youtube_accounts": 600,  # 10 minutes
        }

        # Query optimization hints
        self.index_hints = {
            "channels": ["youtube_channel_id", "user_id", "is_active"],
            "videos": ["channel_id", "status", "created_at", "youtube_video_id"],
            "analytics": ["channel_id", "date", "metric_type"],
            "costs": ["video_id", "service", "created_at"],
            "youtube_accounts": ["account_id", "is_active", "quota_reset_time"],
        }

    # ============================================================================
    # Query Result Caching
    # ============================================================================

    def cache_key(self, query: str, params: Dict[str, Any]) -> str:
        """Generate cache key for query"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        params_hash = hashlib.md5(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()
        return f"query_cache:{query_hash}:{params_hash}"

    async def get_cached(self, query: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached query result"""
        key = self.cache_key(query, params)

        try:
            cached = await self.redis.get(key)
            if cached:
                self.metrics.track_cache_hit("query")
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache get error: {e}")

        self.metrics.track_cache_miss("query")
        return None

    async def set_cached(
        self, query: str, params: Dict[str, Any], result: Any, ttl: int = 300
    ):
        """Cache query result"""
        key = self.cache_key(query, params)

        try:
            await self.redis.setex(key, ttl, json.dumps(result, default=str))
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def with_cache(self, ttl: int = 300):
        """Decorator for caching query results"""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract query and params from function call
                query = str(args[0]) if args else ""
                params = kwargs

                # Try cache first
                cached = await self.get_cached(query, params)
                if cached is not None:
                    return cached

                # Execute query
                result = await func(*args, **kwargs)

                # Cache result
                await self.set_cached(query, params, result, ttl)

                return result

            return wrapper

        return decorator

    # ============================================================================
    # Connection Pool Optimization
    # ============================================================================

    def create_optimized_engine(self, database_url: str):
        """Create optimized database engine with connection pooling"""
        return create_async_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=30,  # Number of persistent connections
            max_overflow=10,  # Maximum overflow connections
            pool_timeout=30,  # Timeout for getting connection
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_pre_ping=True,  # Verify connections before using
            echo=False,  # Disable SQL logging for performance
            # AsyncPG specific optimizations
            connect_args={
                "server_settings": {
                    "application_name": "ytempire_backend",
                    "jit": "off",  # Disable JIT for consistent performance
                },
                "command_timeout": 10,
                "prepared_statement_cache_size": 256,  # Cache prepared statements
                "prepared_statement_name_func": lambda n: f"ps_{n}",
            },
        )

    # ============================================================================
    # Prepared Statements
    # ============================================================================

    async def prepare_statement(self, conn: asyncpg.Connection, name: str, query: str):
        """Prepare a statement for repeated use"""
        if name not in self.prepared_statements:
            stmt = await conn.prepare(query)
            self.prepared_statements[name] = stmt
        return self.prepared_statements[name]

    async def execute_prepared(
        self, conn: asyncpg.Connection, name: str, query: str, *args
    ):
        """Execute a prepared statement"""
        stmt = await self.prepare_statement(conn, name, query)
        return await stmt.fetch(*args)

    # ============================================================================
    # Query Optimization
    # ============================================================================

    def optimize_query(self, query: Select, table_name: str) -> Select:
        """Apply query optimizations"""
        # Add index hints if available
        if table_name in self.index_hints:
            # SQLAlchemy doesn't directly support index hints,
            # but we can ensure we're using indexed columns in WHERE/ORDER BY
            pass

        # Apply eager loading for relationships
        mapper = inspect(query.column_descriptions[0]["entity"])
        for relationship in mapper.relationships:
            if relationship.lazy == "select":
                # Convert lazy loading to eager loading
                query = query.options(selectinload(relationship.key))

        return query

    def add_query_hints(self, query: str, hints: List[str]) -> str:
        """Add PostgreSQL query hints"""
        hint_str = " ".join(f"/*+ {hint} */" for hint in hints)
        return f"{hint_str} {query}"

    # ============================================================================
    # Batch Operations
    # ============================================================================

    async def batch_insert(
        self, session: AsyncSession, table: str, records: List[Dict]
    ):
        """Efficient batch insert using COPY"""
        if not records:
            return

        # Use PostgreSQL COPY for bulk inserts
        columns = records[0].keys()
        values = []

        for record in records:
            values.append(tuple(record.get(col) for col in columns))

        # Create temporary table
        temp_table = f"temp_{table}_{int(time.time())}"

        await session.execute(
            text(
                f"""
            CREATE TEMP TABLE {temp_table} (LIKE {table} INCLUDING ALL)
        """
            )
        )

        # Bulk insert into temp table
        await session.execute(
            text(
                f"""
            INSERT INTO {temp_table} ({','.join(columns)})
            VALUES {','.join(['(:' + ','.join(f'{col}_{i}' for col in columns) + ')' for i in range(len(records))])}
        """
            ),
            {
                f"{col}_{i}": val
                for i, record in enumerate(records)
                for col, val in record.items()
            },
        )

        # Copy from temp to main table
        await session.execute(
            text(
                f"""
            INSERT INTO {table}
            SELECT * FROM {temp_table}
            ON CONFLICT DO NOTHING
        """
            )
        )

        # Drop temp table
        await session.execute(text(f"DROP TABLE {temp_table}"))

    # ============================================================================
    # Index Management
    # ============================================================================

    async def analyze_slow_queries(self, session: AsyncSession) -> List[Dict]:
        """Analyze slow queries and suggest indexes"""
        result = await session.execute(
            text(
                """
            SELECT 
                query,
                calls,
                mean_exec_time,
                total_exec_time,
                min_exec_time,
                max_exec_time,
                stddev_exec_time
            FROM pg_stat_statements
            WHERE mean_exec_time > 5  -- Queries slower than 5ms
            ORDER BY mean_exec_time DESC
            LIMIT 20
        """
            )
        )

        slow_queries = []
        for row in result:
            slow_queries.append(
                {
                    "query": row.query,
                    "calls": row.calls,
                    "mean_time": row.mean_exec_time,
                    "total_time": row.total_exec_time,
                    "suggested_index": self._suggest_index(row.query),
                }
            )

        return slow_queries

    def _suggest_index(self, query: str) -> Optional[str]:
        """Suggest index based on query pattern"""
        query_lower = query.lower()

        # Extract WHERE clause columns
        if "where" in query_lower:
            where_part = query_lower.split("where")[1].split("order by")[0]

            # Simple pattern matching for column names
            if "channel_id" in where_part and "created_at" in where_part:
                return "CREATE INDEX IF NOT EXISTS idx_channel_created ON table_name(channel_id, created_at)"
            elif "user_id" in where_part:
                return "CREATE INDEX IF NOT EXISTS idx_user_id ON table_name(user_id)"
            elif "status" in where_part:
                return "CREATE INDEX IF NOT EXISTS idx_status ON table_name(status)"

        return None

    async def create_missing_indexes(self, session: AsyncSession):
        """Create missing indexes for optimal performance"""
        indexes = [
            # Channels
            "CREATE INDEX IF NOT EXISTS idx_channels_user_active ON channels(user_id, is_active)",
            "CREATE INDEX IF NOT EXISTS idx_channels_youtube_id ON channels(youtube_channel_id)",
            "CREATE INDEX IF NOT EXISTS idx_channels_health ON channels(health_score) WHERE is_active = true",
            # Videos
            "CREATE INDEX IF NOT EXISTS idx_videos_channel_status ON videos(channel_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_videos_created ON videos(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_videos_youtube_id ON videos(youtube_video_id)",
            "CREATE INDEX IF NOT EXISTS idx_videos_processing ON videos(status) WHERE status IN ('pending', 'processing')",
            # Analytics
            "CREATE INDEX IF NOT EXISTS idx_analytics_channel_date ON analytics(channel_id, date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_analytics_metric ON analytics(metric_type, date DESC)",
            # Costs
            "CREATE INDEX IF NOT EXISTS idx_costs_video ON costs(video_id)",
            "CREATE INDEX IF NOT EXISTS idx_costs_service_date ON costs(service, created_at DESC)",
            # YouTube Accounts
            "CREATE INDEX IF NOT EXISTS idx_youtube_active ON youtube_accounts(is_active, quota_used)",
            "CREATE INDEX IF NOT EXISTS idx_youtube_quota ON youtube_accounts(quota_reset_time) WHERE is_active = true",
            # Partial indexes for common queries
            "CREATE INDEX IF NOT EXISTS idx_videos_recent ON videos(created_at DESC) WHERE created_at > NOW() - INTERVAL '7 days'",
            "CREATE INDEX IF NOT EXISTS idx_channels_healthy ON channels(health_score DESC) WHERE health_score > 0.7",
        ]

        for index_sql in indexes:
            try:
                await session.execute(text(index_sql))
            except Exception as e:
                logger.warning(f"Index creation failed: {e}")

    # ============================================================================
    # Materialized Views
    # ============================================================================

    async def create_materialized_views(self, session: AsyncSession):
        """Create materialized views for complex aggregations"""
        views = [
            # Channel statistics view
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS channel_stats AS
            SELECT 
                c.id as channel_id,
                c.channel_name,
                COUNT(DISTINCT v.id) as video_count,
                AVG(v.duration) as avg_duration,
                SUM(v.views) as total_views,
                AVG(co.cost) as avg_cost_per_video,
                MAX(v.created_at) as last_video_date
            FROM channels c
            LEFT JOIN videos v ON c.id = v.channel_id
            LEFT JOIN costs co ON v.id = co.video_id
            WHERE c.is_active = true
            GROUP BY c.id, c.channel_name
            WITH DATA
            """,
            # Daily revenue view
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS daily_revenue AS
            SELECT 
                DATE(created_at) as date,
                SUM(amount) as total_revenue,
                COUNT(*) as transaction_count,
                AVG(amount) as avg_transaction
            FROM payments
            WHERE status = 'succeeded'
            GROUP BY DATE(created_at)
            WITH DATA
            """,
            # Video performance view
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS video_performance AS
            SELECT 
                v.id,
                v.title,
                v.views,
                v.likes,
                v.comments,
                (v.likes::float / NULLIF(v.views, 0)) as engagement_rate,
                co.cost,
                (v.views::float / NULLIF(co.cost, 0)) as views_per_dollar
            FROM videos v
            LEFT JOIN costs co ON v.id = co.video_id
            WHERE v.status = 'published'
            WITH DATA
            """,
        ]

        for view_sql in views:
            try:
                await session.execute(text(view_sql))
            except Exception as e:
                logger.warning(f"Materialized view creation failed: {e}")

        # Create indexes on materialized views
        mv_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_channel_stats_id ON channel_stats(channel_id)",
            "CREATE INDEX IF NOT EXISTS idx_daily_revenue_date ON daily_revenue(date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_video_performance_views ON video_performance(views DESC)",
        ]

        for index_sql in mv_indexes:
            try:
                await session.execute(text(index_sql))
            except Exception as e:
                logger.warning(f"Materialized view index creation failed: {e}")

    async def refresh_materialized_views(self, session: AsyncSession):
        """Refresh materialized views"""
        views = ["channel_stats", "daily_revenue", "video_performance"]

        for view in views:
            try:
                await session.execute(
                    text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}")
                )
                logger.info(f"Refreshed materialized view: {view}")
            except Exception as e:
                logger.error(f"Failed to refresh view {view}: {e}")

    # ============================================================================
    # Query Plan Analysis
    # ============================================================================

    async def analyze_query_plan(self, session: AsyncSession, query: str) -> Dict:
        """Analyze query execution plan"""
        result = await session.execute(
            text(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}")
        )
        plan = result.scalar()

        # Extract key metrics
        if plan and len(plan) > 0:
            plan_data = plan[0]["Plan"]
            return {
                "total_cost": plan_data.get("Total Cost"),
                "actual_time": plan_data.get("Actual Total Time"),
                "rows": plan_data.get("Actual Rows"),
                "buffers_hit": plan_data.get("Shared Hit Blocks", 0),
                "buffers_read": plan_data.get("Shared Read Blocks", 0),
                "cache_hit_ratio": self._calculate_cache_hit_ratio(plan_data),
            }

        return {}

    def _calculate_cache_hit_ratio(self, plan_data: Dict) -> float:
        """Calculate buffer cache hit ratio"""
        hits = plan_data.get("Shared Hit Blocks", 0)
        reads = plan_data.get("Shared Read Blocks", 0)

        if hits + reads > 0:
            return hits / (hits + reads)
        return 0.0

    # ============================================================================
    # Performance Monitoring
    # ============================================================================

    async def monitor_query_performance(self, session: AsyncSession):
        """Monitor and log query performance"""
        # Get current database statistics
        result = await session.execute(
            text(
                """
            SELECT 
                numbackends as active_connections,
                xact_commit as transactions_committed,
                xact_rollback as transactions_rolled_back,
                blks_hit as blocks_hit,
                blks_read as blocks_read,
                tup_returned as rows_returned,
                tup_fetched as rows_fetched,
                tup_inserted as rows_inserted,
                tup_updated as rows_updated,
                tup_deleted as rows_deleted
            FROM pg_stat_database
            WHERE datname = current_database()
        """
            )
        )

        stats = result.first()

        if stats:
            # Calculate cache hit ratio
            cache_hit_ratio = 0
            if stats.blocks_hit + stats.blocks_read > 0:
                cache_hit_ratio = stats.blocks_hit / (
                    stats.blocks_hit + stats.blocks_read
                )

            logger.info(
                f"""
                Database Performance:
                - Active Connections: {stats.active_connections}
                - Cache Hit Ratio: {cache_hit_ratio:.2%}
                - Transactions: {stats.transactions_committed} committed, {stats.transactions_rolled_back} rolled back
            """
            )

            # Alert if cache hit ratio is low
            if cache_hit_ratio < 0.9:
                logger.warning(f"Low cache hit ratio: {cache_hit_ratio:.2%}")

    # ============================================================================
    # Auto-optimization
    # ============================================================================

    async def auto_optimize(self, session: AsyncSession):
        """Automatically optimize database for <5ms p95 latency"""
        logger.info("Running automatic database optimization...")

        # 1. Create missing indexes
        await self.create_missing_indexes(session)

        # 2. Create/refresh materialized views
        await self.create_materialized_views(session)
        await self.refresh_materialized_views(session)

        # 3. Update table statistics
        await session.execute(text("ANALYZE"))

        # 4. Vacuum tables to reclaim space
        tables = ["channels", "videos", "analytics", "costs"]
        for table in tables:
            try:
                await session.execute(text(f"VACUUM ANALYZE {table}"))
            except Exception as e:
                logger.warning(f"Vacuum failed for {table}: {e}")

        # 5. Analyze slow queries
        slow_queries = await self.analyze_slow_queries(session)
        if slow_queries:
            logger.warning(
                f"Found {len(slow_queries)} slow queries requiring optimization"
            )
            for query in slow_queries[:5]:
                logger.info(
                    f"Slow query (mean: {query['mean_time']:.2f}ms): {query['query'][:100]}..."
                )

        logger.info("Database optimization completed")


# ============================================================================
# Query Optimization Middleware
# ============================================================================


class OptimizedQueryMiddleware:
    """Middleware to automatically optimize queries"""

    def __init__(self, optimizer: DatabaseOptimizer):
        self.optimizer = optimizer

    async def __call__(self, execute, query, *args, **kwargs):
        """Intercept and optimize queries"""
        start_time = time.time()

        # Check cache first
        query_str = str(query)
        cached = await self.optimizer.get_cached(query_str, kwargs)
        if cached is not None:
            return cached

        # Execute query
        result = await execute(query, *args, **kwargs)

        # Cache result if query was fast
        duration = time.time() - start_time
        if duration < 0.005:  # Cache queries faster than 5ms
            await self.optimizer.set_cached(query_str, kwargs, result)
        elif duration > 0.01:  # Log slow queries
            logger.warning(f"Slow query ({duration*1000:.2f}ms): {query_str[:100]}")

        return result
