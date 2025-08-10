# YTEMPIRE Database Schema Testing & Data Validation Guide
**Version 1.0 | January 2025**  
**Owner: QA Engineering Team**  
**Primary Author: QA Engineer**  
**Approved By: Platform Operations Lead**

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Database Architecture Overview](#2-database-architecture-overview)
3. [Schema Testing Strategy](#3-schema-testing-strategy)
4. [Data Validation Framework](#4-data-validation-framework)
5. [Test Implementation](#5-test-implementation)
6. [Performance Testing](#6-performance-testing)
7. [Data Integrity Testing](#7-data-integrity-testing)
8. [Migration Testing](#8-migration-testing)
9. [Test Automation](#9-test-automation)
10. [Monitoring & Alerting](#10-monitoring--alerting)

---

## 1. Executive Summary

This guide provides comprehensive testing strategies for YTEMPIRE's database layer, ensuring data integrity, performance, and reliability for our automated YouTube content platform.

### Core Testing Objectives
- **Data Integrity**: Zero data corruption or loss
- **Performance**: Sub-100ms query response times
- **Scalability**: Support 500+ channels, 100M+ records
- **Reliability**: 99.99% database availability
- **Compliance**: GDPR and YouTube data requirements

### Testing Scope
- PostgreSQL 15 primary database
- Redis 7 caching layer
- Database migrations and schema changes
- Data validation and constraints
- Performance benchmarks
- Backup and recovery procedures

---

## 2. Database Architecture Overview

### 2.1 Technology Stack

```yaml
database_stack:
  primary_database:
    type: PostgreSQL 15
    purpose: Transactional data, user management, content metadata
    size: 100GB (MVP) → 1TB (Scale)
    connections: 200 concurrent
    
  cache_layer:
    type: Redis 7
    purpose: Session management, hot data, rate limiting
    size: 8GB (MVP) → 32GB (Scale)
    persistence: AOF with 1-second fsync
    
  backup_strategy:
    frequency: Daily full + hourly incremental
    retention: 30 days
    recovery_time: <30 minutes
    recovery_point: <15 minutes
```

### 2.2 Core Database Schema

```sql
-- Users and Authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'active',
    subscription_tier VARCHAR(20) DEFAULT 'free',
    
    CONSTRAINT email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'),
    CONSTRAINT username_length CHECK (LENGTH(username) >= 3)
);

-- YouTube Channels
CREATE TABLE channels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    youtube_channel_id VARCHAR(100) UNIQUE NOT NULL,
    channel_name VARCHAR(255) NOT NULL,
    niche VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    monetization_status VARCHAR(20) DEFAULT 'pending',
    
    INDEX idx_user_channels (user_id),
    INDEX idx_channel_status (status)
);

-- Videos
CREATE TABLE videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_id UUID NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    youtube_video_id VARCHAR(100) UNIQUE,
    status VARCHAR(20) DEFAULT 'draft',
    generation_cost DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    published_at TIMESTAMP WITH TIME ZONE,
    view_count BIGINT DEFAULT 0,
    revenue DECIMAL(10,2) DEFAULT 0,
    
    INDEX idx_channel_videos (channel_id),
    INDEX idx_video_status (status),
    INDEX idx_published_date (published_at)
);

-- Analytics
CREATE TABLE analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    views INTEGER DEFAULT 0,
    watch_time_minutes DECIMAL(10,2) DEFAULT 0,
    subscribers_gained INTEGER DEFAULT 0,
    revenue DECIMAL(10,2) DEFAULT 0,
    
    UNIQUE(video_id, date),
    INDEX idx_analytics_date (date),
    INDEX idx_video_analytics (video_id)
);
```

---

## 3. Schema Testing Strategy

### 3.1 Schema Validation Tests

```python
import pytest
import psycopg2
from datetime import datetime
import uuid

class TestDatabaseSchema:
    """Comprehensive schema validation tests"""
    
    @pytest.fixture
    def db_connection(self):
        """Database connection fixture"""
        conn = psycopg2.connect(
            host="localhost",
            database="ytempire_test",
            user="test_user",
            password="test_pass"
        )
        yield conn
        conn.rollback()
        conn.close()
    
    def test_table_existence(self, db_connection):
        """Verify all required tables exist"""
        cursor = db_connection.cursor()
        
        required_tables = [
            'users', 'channels', 'videos', 
            'analytics', 'api_keys', 'workflows'
        ]
        
        for table in required_tables:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                )
            """, (table,))
            
            exists = cursor.fetchone()[0]
            assert exists, f"Table {table} does not exist"
    
    def test_column_constraints(self, db_connection):
        """Test column constraints and data types"""
        cursor = db_connection.cursor()
        
        # Test NOT NULL constraints
        cursor.execute("""
            SELECT column_name, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'users'
            AND column_name IN ('email', 'username', 'password_hash')
        """)
        
        for column, nullable in cursor.fetchall():
            assert nullable == 'NO', f"{column} should not be nullable"
        
        # Test UNIQUE constraints
        cursor.execute("""
            SELECT COUNT(*)
            FROM information_schema.table_constraints
            WHERE table_name = 'users'
            AND constraint_type = 'UNIQUE'
        """)
        
        unique_count = cursor.fetchone()[0]
        assert unique_count >= 2, "Users table should have at least 2 unique constraints"
    
    def test_foreign_key_relationships(self, db_connection):
        """Validate foreign key relationships"""
        cursor = db_connection.cursor()
        
        # Test channels -> users relationship
        cursor.execute("""
            SELECT COUNT(*)
            FROM information_schema.referential_constraints
            WHERE constraint_name LIKE 'channels_user_id_fkey'
        """)
        
        assert cursor.fetchone()[0] == 1, "Missing foreign key: channels.user_id -> users.id"
        
        # Test cascade delete
        cursor.execute("""
            SELECT delete_rule
            FROM information_schema.referential_constraints
            WHERE constraint_name LIKE 'channels_user_id_fkey'
        """)
        
        delete_rule = cursor.fetchone()[0]
        assert delete_rule == 'CASCADE', "Channels should cascade delete with users"
    
    def test_index_performance(self, db_connection):
        """Verify critical indexes exist"""
        cursor = db_connection.cursor()
        
        critical_indexes = [
            ('channels', 'idx_user_channels'),
            ('videos', 'idx_channel_videos'),
            ('videos', 'idx_video_status'),
            ('analytics', 'idx_analytics_date')
        ]
        
        for table, index_name in critical_indexes:
            cursor.execute("""
                SELECT COUNT(*)
                FROM pg_indexes
                WHERE tablename = %s
                AND indexname = %s
            """, (table, index_name))
            
            exists = cursor.fetchone()[0]
            assert exists == 1, f"Missing index: {table}.{index_name}"
```

### 3.2 Data Type Validation

```python
class TestDataTypes:
    """Test data type constraints and validations"""
    
    def test_uuid_generation(self, db_connection):
        """Test UUID primary key generation"""
        cursor = db_connection.cursor()
        
        # Insert without specifying ID
        cursor.execute("""
            INSERT INTO users (email, username, password_hash)
            VALUES ('test@example.com', 'testuser', 'hash123')
            RETURNING id
        """)
        
        user_id = cursor.fetchone()[0]
        
        # Verify UUID format
        try:
            uuid.UUID(str(user_id))
        except ValueError:
            pytest.fail(f"Invalid UUID format: {user_id}")
    
    def test_email_validation(self, db_connection):
        """Test email format constraint"""
        cursor = db_connection.cursor()
        
        # Valid email should succeed
        cursor.execute("""
            INSERT INTO users (email, username, password_hash)
            VALUES ('valid@email.com', 'user1', 'hash')
        """)
        
        # Invalid email should fail
        with pytest.raises(psycopg2.IntegrityError):
            cursor.execute("""
                INSERT INTO users (email, username, password_hash)
                VALUES ('invalid-email', 'user2', 'hash')
            """)
    
    def test_decimal_precision(self, db_connection):
        """Test decimal field precision"""
        cursor = db_connection.cursor()
        
        # Insert video with cost
        cursor.execute("""
            INSERT INTO videos (channel_id, title, generation_cost)
            VALUES (%s, 'Test Video', 0.99)
            RETURNING generation_cost
        """, (str(uuid.uuid4()),))
        
        cost = cursor.fetchone()[0]
        assert float(cost) == 0.99, "Decimal precision not maintained"
    
    def test_timestamp_timezone(self, db_connection):
        """Test timezone-aware timestamps"""
        cursor = db_connection.cursor()
        
        cursor.execute("""
            INSERT INTO users (email, username, password_hash)
            VALUES ('tz@test.com', 'tzuser', 'hash')
            RETURNING created_at
        """)
        
        created_at = cursor.fetchone()[0]
        assert created_at.tzinfo is not None, "Timestamp should be timezone-aware"
```

---

## 4. Data Validation Framework

### 4.1 Business Logic Validation

```python
class DataValidator:
    """Core data validation logic"""
    
    def __init__(self, db_connection):
        self.conn = db_connection
        
    def validate_user_channel_limit(self, user_id: str) -> bool:
        """Ensure users don't exceed channel limits"""
        cursor = self.conn.cursor()
        
        # Get user subscription tier
        cursor.execute("""
            SELECT subscription_tier FROM users WHERE id = %s
        """, (user_id,))
        
        tier = cursor.fetchone()[0]
        
        # Get current channel count
        cursor.execute("""
            SELECT COUNT(*) FROM channels 
            WHERE user_id = %s AND status = 'active'
        """, (user_id,))
        
        channel_count = cursor.fetchone()[0]
        
        # Check limits based on tier
        limits = {
            'free': 1,
            'starter': 5,
            'professional': 20,
            'enterprise': 100
        }
        
        return channel_count < limits.get(tier, 1)
    
    def validate_video_generation_cost(self, video_id: str) -> dict:
        """Validate video generation cost is within bounds"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT generation_cost, created_at
            FROM videos WHERE id = %s
        """, (video_id,))
        
        cost, created = cursor.fetchone()
        
        validation_result = {
            'valid': True,
            'errors': []
        }
        
        # Cost should be between $0.10 and $5.00
        if cost < 0.10:
            validation_result['valid'] = False
            validation_result['errors'].append("Cost below minimum threshold")
        elif cost > 5.00:
            validation_result['valid'] = False
            validation_result['errors'].append("Cost exceeds maximum limit")
            
        return validation_result
    
    def validate_analytics_consistency(self, video_id: str) -> dict:
        """Ensure analytics data is consistent"""
        cursor = self.conn.cursor()
        
        # Get video total views
        cursor.execute("""
            SELECT view_count FROM videos WHERE id = %s
        """, (video_id,))
        
        video_views = cursor.fetchone()[0]
        
        # Get sum of analytics views
        cursor.execute("""
            SELECT SUM(views) FROM analytics WHERE video_id = %s
        """, (video_id,))
        
        analytics_sum = cursor.fetchone()[0] or 0
        
        # Views should match (with small tolerance for processing delay)
        tolerance = 100  # Allow 100 view difference
        is_consistent = abs(video_views - analytics_sum) <= tolerance
        
        return {
            'consistent': is_consistent,
            'video_views': video_views,
            'analytics_sum': analytics_sum,
            'difference': abs(video_views - analytics_sum)
        }
```

### 4.2 Data Quality Tests

```python
class TestDataQuality:
    """Test data quality and consistency"""
    
    def test_no_orphaned_records(self, db_connection):
        """Ensure no orphaned records exist"""
        cursor = db_connection.cursor()
        
        # Check for channels without users
        cursor.execute("""
            SELECT COUNT(*)
            FROM channels c
            LEFT JOIN users u ON c.user_id = u.id
            WHERE u.id IS NULL
        """)
        
        orphaned_channels = cursor.fetchone()[0]
        assert orphaned_channels == 0, f"Found {orphaned_channels} orphaned channels"
        
        # Check for videos without channels
        cursor.execute("""
            SELECT COUNT(*)
            FROM videos v
            LEFT JOIN channels c ON v.channel_id = c.id
            WHERE c.id IS NULL
        """)
        
        orphaned_videos = cursor.fetchone()[0]
        assert orphaned_videos == 0, f"Found {orphaned_videos} orphaned videos"
    
    def test_data_completeness(self, db_connection):
        """Test required fields are populated"""
        cursor = db_connection.cursor()
        
        # Check for videos missing YouTube IDs after publishing
        cursor.execute("""
            SELECT COUNT(*)
            FROM videos
            WHERE status = 'published'
            AND youtube_video_id IS NULL
        """)
        
        missing_ids = cursor.fetchone()[0]
        assert missing_ids == 0, f"Found {missing_ids} published videos without YouTube IDs"
    
    def test_temporal_consistency(self, db_connection):
        """Test time-based data consistency"""
        cursor = db_connection.cursor()
        
        # Published date should be after created date
        cursor.execute("""
            SELECT COUNT(*)
            FROM videos
            WHERE published_at IS NOT NULL
            AND published_at < created_at
        """)
        
        temporal_errors = cursor.fetchone()[0]
        assert temporal_errors == 0, "Found videos published before creation"
```

---

## 5. Test Implementation

### 5.1 Test Environment Setup

```python
# conftest.py - Pytest configuration
import pytest
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import redis
import os

@pytest.fixture(scope="session")
def test_database():
    """Create and configure test database"""
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database="postgres",
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Create test database
    cursor.execute("DROP DATABASE IF EXISTS ytempire_test")
    cursor.execute("CREATE DATABASE ytempire_test")
    
    # Connect to test database
    test_conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database="ytempire_test",
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )
    
    # Run migrations
    with open('schema/migrations/001_initial_schema.sql', 'r') as f:
        test_conn.cursor().execute(f.read())
    test_conn.commit()
    
    yield test_conn
    
    # Cleanup
    test_conn.close()
    cursor.execute("DROP DATABASE ytempire_test")
    conn.close()

@pytest.fixture(scope="function")
def db_session(test_database):
    """Provide transactional database session"""
    test_database.rollback()  # Start fresh
    yield test_database
    test_database.rollback()  # Rollback changes

@pytest.fixture(scope="session")
def redis_client():
    """Provide Redis test client"""
    client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=6379,
        db=1,  # Use separate DB for tests
        decode_responses=True
    )
    client.flushdb()  # Clear test database
    yield client
    client.flushdb()  # Cleanup
```

### 5.2 Test Data Factories

```python
# factories.py - Test data generation
import factory
from factory import fuzzy
import uuid
from datetime import datetime, timedelta
import random

class UserFactory(factory.Factory):
    """Generate test user data"""
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    email = factory.Faker('email')
    username = factory.Faker('user_name')
    password_hash = factory.Faker('sha256')
    created_at = factory.Faker('date_time_this_year', tzinfo=None)
    subscription_tier = factory.fuzzy.FuzzyChoice(
        ['free', 'starter', 'professional', 'enterprise']
    )
    status = 'active'

class ChannelFactory(factory.Factory):
    """Generate test channel data"""
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    user_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    youtube_channel_id = factory.Faker('bothify', text='UC??############')
    channel_name = factory.Faker('company')
    niche = factory.fuzzy.FuzzyChoice([
        'Technology', 'Gaming', 'Education', 
        'Entertainment', 'Music', 'Sports'
    ])
    status = 'active'
    created_at = factory.Faker('date_time_this_year', tzinfo=None)

class VideoFactory(factory.Factory):
    """Generate test video data"""
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    channel_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    title = factory.Faker('sentence', nb_words=8)
    description = factory.Faker('text', max_nb_chars=500)
    youtube_video_id = factory.Faker('bothify', text='???????????')
    status = factory.fuzzy.FuzzyChoice(
        ['draft', 'processing', 'published', 'failed']
    )
    generation_cost = factory.fuzzy.FuzzyDecimal(0.10, 2.00, precision=2)
    view_count = factory.fuzzy.FuzzyInteger(0, 100000)
    revenue = factory.fuzzy.FuzzyDecimal(0, 100, precision=2)

class TestDataGenerator:
    """Generate complex test scenarios"""
    
    @staticmethod
    def create_user_with_channels(conn, num_channels=5):
        """Create user with multiple channels"""
        user = UserFactory()
        cursor = conn.cursor()
        
        # Insert user
        cursor.execute("""
            INSERT INTO users (id, email, username, password_hash, subscription_tier)
            VALUES (%(id)s, %(email)s, %(username)s, %(password_hash)s, %(subscription_tier)s)
            RETURNING id
        """, user)
        
        user_id = cursor.fetchone()[0]
        
        # Create channels
        channels = []
        for _ in range(num_channels):
            channel = ChannelFactory(user_id=user_id)
            cursor.execute("""
                INSERT INTO channels (id, user_id, youtube_channel_id, channel_name, niche)
                VALUES (%(id)s, %(user_id)s, %(youtube_channel_id)s, %(channel_name)s, %(niche)s)
                RETURNING id
            """, channel)
            channels.append(cursor.fetchone()[0])
        
        conn.commit()
        return user_id, channels
    
    @staticmethod
    def create_channel_with_videos(conn, channel_id, num_videos=10):
        """Create channel with video history"""
        cursor = conn.cursor()
        videos = []
        
        for _ in range(num_videos):
            video = VideoFactory(channel_id=channel_id)
            cursor.execute("""
                INSERT INTO videos (id, channel_id, title, description, 
                                  youtube_video_id, status, generation_cost)
                VALUES (%(id)s, %(channel_id)s, %(title)s, %(description)s,
                       %(youtube_video_id)s, %(status)s, %(generation_cost)s)
                RETURNING id
            """, video)
            videos.append(cursor.fetchone()[0])
        
        conn.commit()
        return videos
```

---

## 6. Performance Testing

### 6.1 Query Performance Tests

```python
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

class TestDatabasePerformance:
    """Database performance testing suite"""
    
    def test_query_response_time(self, db_connection):
        """Test individual query performance"""
        cursor = db_connection.cursor()
        
        queries = [
            ("User lookup", "SELECT * FROM users WHERE email = %s", ('test@example.com',)),
            ("Channel list", "SELECT * FROM channels WHERE user_id = %s", (str(uuid.uuid4()),)),
            ("Video analytics", """
                SELECT v.*, SUM(a.views) as total_views
                FROM videos v
                LEFT JOIN analytics a ON v.id = a.video_id
                WHERE v.channel_id = %s
                GROUP BY v.id
            """, (str(uuid.uuid4()),))
        ]
        
        for name, query, params in queries:
            times = []
            
            for _ in range(100):  # Run 100 times
                start = time.perf_counter()
                cursor.execute(query, params)
                cursor.fetchall()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
            
            avg_time = statistics.mean(times)
            p95_time = statistics.quantiles(times, n=20)[18]  # 95th percentile
            
            print(f"{name}: avg={avg_time:.2f}ms, p95={p95_time:.2f}ms")
            assert p95_time < 100, f"{name} p95 exceeds 100ms: {p95_time:.2f}ms"
    
    def test_concurrent_load(self, db_connection):
        """Test database under concurrent load"""
        
        def execute_query(query_num):
            conn = psycopg2.connect(
                host="localhost",
                database="ytempire_test",
                user="test_user",
                password="test_pass"
            )
            cursor = conn.cursor()
            
            start = time.perf_counter()
            cursor.execute("""
                SELECT COUNT(*) FROM videos 
                WHERE status = 'published'
            """)
            result = cursor.fetchone()
            end = time.perf_counter()
            
            conn.close()
            return end - start
        
        # Run 100 concurrent queries
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(execute_query, i) for i in range(100)]
            times = [f.result() for f in futures]
        
        avg_time = statistics.mean(times) * 1000
        max_time = max(times) * 1000
        
        print(f"Concurrent load: avg={avg_time:.2f}ms, max={max_time:.2f}ms")
        assert max_time < 500, f"Max response time too high: {max_time:.2f}ms"
    
    def test_bulk_insert_performance(self, db_connection):
        """Test bulk insert performance"""
        cursor = db_connection.cursor()
        
        # Generate test data
        videos = [VideoFactory() for _ in range(1000)]
        
        # Test bulk insert
        start = time.perf_counter()
        
        cursor.executemany("""
            INSERT INTO videos (id, channel_id, title, description, generation_cost)
            VALUES (%(id)s, %(channel_id)s, %(title)s, %(description)s, %(generation_cost)s)
        """, videos)
        
        db_connection.commit()
        end = time.perf_counter()
        
        insert_time = end - start
        rate = 1000 / insert_time
        
        print(f"Bulk insert: {insert_time:.2f}s for 1000 records ({rate:.0f} records/sec)")
        assert insert_time < 5, f"Bulk insert too slow: {insert_time:.2f}s"
```

### 6.2 Connection Pool Testing

```python
class TestConnectionPool:
    """Test database connection pooling"""
    
    def test_connection_limits(self):
        """Test maximum connection handling"""
        from psycopg2 import pool
        
        # Create connection pool
        connection_pool = pool.ThreadedConnectionPool(
            minconn=5,
            maxconn=20,
            host="localhost",
            database="ytempire_test",
            user="test_user",
            password="test_pass"
        )
        
        connections = []
        
        # Get maximum connections
        for i in range(20):
            conn = connection_pool.getconn()
            connections.append(conn)
        
        # Try to get one more (should wait or fail)
        import threading
        timeout_occurred = False
        
        def get_extra_connection():
            nonlocal timeout_occurred
            try:
                conn = connection_pool.getconn()
                connection_pool.putconn(conn)
            except pool.PoolError:
                timeout_occurred = True
        
        thread = threading.Thread(target=get_extra_connection)
        thread.start()
        thread.join(timeout=1)
        
        # Return connections
        for conn in connections:
            connection_pool.putconn(conn)
        
        connection_pool.closeall()
    
    def test_connection_recovery(self, db_connection):
        """Test connection recovery after failure"""
        cursor = db_connection.cursor()
        
        # Simulate connection drop
        original_conn = db_connection
        
        # Kill connection (simulate network issue)
        cursor.execute("SELECT pg_backend_pid()")
        pid = cursor.fetchone()[0]
        
        # Create new connection to kill the old one
        admin_conn = psycopg2.connect(
            host="localhost",
            database="ytempire_test",
            user="test_user",
            password="test_pass"
        )
        admin_cursor = admin_conn.cursor()
        
        # Terminate connection
        admin_cursor.execute(f"SELECT pg_terminate_backend({pid})")
        admin_conn.commit()
        admin_conn.close()
        
        # Try to use original connection (should fail)
        with pytest.raises(psycopg2.OperationalError):
            cursor.execute("SELECT 1")
        
        # Verify new connection works
        new_conn = psycopg2.connect(
            host="localhost",
            database="ytempire_test",
            user="test_user",
            password="test_pass"
        )
        new_cursor = new_conn.cursor()
        new_cursor.execute("SELECT 1")
        assert new_cursor.fetchone()[0] == 1
        new_conn.close()
```

---

## 7. Data Integrity Testing

### 7.1 Transaction Testing

```python
class TestTransactions:
    """Test transaction integrity and isolation"""
    
    def test_rollback_on_error(self, db_connection):
        """Test transaction rollback on error"""
        cursor = db_connection.cursor()
        
        try:
            # Start transaction
            cursor.execute("BEGIN")
            
            # Insert valid user
            cursor.execute("""
                INSERT INTO users (email, username, password_hash)
                VALUES ('rollback@test.com', 'rollbackuser', 'hash')
            """)
            
            # Insert invalid channel (missing user_id)
            cursor.execute("""
                INSERT INTO channels (youtube_channel_id, channel_name, niche)
                VALUES ('UC123', 'Test Channel', 'Tech')
            """)
            
            # This should fail due to NOT NULL constraint
            cursor.execute("COMMIT")
            
        except psycopg2.IntegrityError:
            cursor.execute("ROLLBACK")
        
        # Verify user was not created
        cursor.execute("""
            SELECT COUNT(*) FROM users WHERE email = 'rollback@test.com'
        """)
        assert cursor.fetchone()[0] == 0, "Transaction not rolled back properly"
    
    def test_isolation_levels(self, db_connection):
        """Test transaction isolation"""
        import threading
        
        def concurrent_update(email_suffix):
            conn = psycopg2.connect(
                host="localhost",
                database="ytempire_test",
                user="test_user",
                password="test_pass"
            )
            cursor = conn.cursor()
            
            cursor.execute("BEGIN ISOLATION LEVEL SERIALIZABLE")
            
            # Read current count
            cursor.execute("SELECT COUNT(*) FROM users")
            count = cursor.fetchone()[0]
            
            # Simulate processing delay
            time.sleep(0.1)
            
            # Insert new user
            cursor.execute("""
                INSERT INTO users (email, username, password_hash)
                VALUES (%s, %s, 'hash')
            """, (f'user{count}@{email_suffix}.com', f'user{count}'))
            
            try:
                cursor.execute("COMMIT")
                conn.close()
                return True
            except psycopg2.extensions.TransactionRollbackError:
                cursor.execute("ROLLBACK")
                conn.close()
                return False
        
        # Run concurrent transactions
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_update, args=(f'test{i}',))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify data consistency
        cursor = db_connection.cursor()
        cursor.execute("SELECT COUNT(DISTINCT username) FROM users")
        unique_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM users")
        total_count = cursor.fetchone()[0]
        
        assert unique_count == total_count, "Duplicate usernames created"
```

### 7.2 Constraint Testing

```python
class TestConstraints:
    """Test database constraints and triggers"""
    
    def test_cascade_delete(self, db_connection):
        """Test cascade delete behavior"""
        cursor = db_connection.cursor()
        
        # Create user with channels and videos
        user_id = str(uuid.uuid4())
        channel_id = str(uuid.uuid4())
        video_id = str(uuid.uuid4())
        
        # Insert test data
        cursor.execute("""
            INSERT INTO users (id, email, username, password_hash)
            VALUES (%s, 'cascade@test.com', 'cascadeuser', 'hash')
        """, (user_id,))
        
        cursor.execute("""
            INSERT INTO channels (id, user_id, youtube_channel_id, channel_name, niche)
            VALUES (%s, %s, 'UC_CASCADE', 'Cascade Channel', 'Test')
        """, (channel_id, user_id))
        
        cursor.execute("""
            INSERT INTO videos (id, channel_id, title)
            VALUES (%s, %s, 'Cascade Video')
        """, (video_id, channel_id))
        
        db_connection.commit()
        
        # Delete user
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        db_connection.commit()
        
        # Verify cascade delete
        cursor.execute("SELECT COUNT(*) FROM channels WHERE id = %s", (channel_id,))
        assert cursor.fetchone()[0] == 0, "Channel not cascade deleted"
        
        cursor.execute("SELECT COUNT(*) FROM videos WHERE id = %s", (video_id,))
        assert cursor.fetchone()[0] == 0, "Video not cascade deleted"
    
    def test_check_constraints(self, db_connection):
        """Test CHECK constraints"""
        cursor = db_connection.cursor()
        
        # Test email format constraint
        with pytest.raises(psycopg2.IntegrityError):
            cursor.execute("""
                INSERT INTO users (email, username, password_hash)
                VALUES ('notanemail', 'user1', 'hash')
            """)
            db_connection.commit()
        
        db_connection.rollback()
        
        # Test username length constraint
        with pytest.raises(psycopg2.IntegrityError):
            cursor.execute("""
                INSERT INTO users (email, username, password_hash)
                VALUES ('valid@email.com', 'ab', 'hash')
            """)
            db_connection.commit()
        
        db_connection.rollback()
    
    def test_unique_constraints(self, db_connection):
        """Test UNIQUE constraints"""
        cursor = db_connection.cursor()
        
        # Insert first user
        cursor.execute("""
            INSERT INTO users (email, username, password_hash)
            VALUES ('unique@test.com', 'uniqueuser', 'hash')
        """)
        db_connection.commit()
        
        # Try to insert duplicate email
        with pytest.raises(psycopg2.IntegrityError) as exc_info:
            cursor.execute("""
                INSERT INTO users (email, username, password_hash)
                VALUES ('unique@test.com', 'different', 'hash')
            """)
            db_connection.commit()
        
        assert 'unique constraint' in str(exc_info.value).lower()
        db_connection.rollback()
        
        # Try to insert duplicate username
        with pytest.raises(psycopg2.IntegrityError) as exc_info:
            cursor.execute("""
                INSERT INTO users (email, username, password_hash)
                VALUES ('different@test.com', 'uniqueuser', 'hash')
            """)
            db_connection.commit()
        
        assert 'unique constraint' in str(exc_info.value).lower()
```

---

## 8. Migration Testing

### 8.1 Migration Validation

```python
class TestMigrations:
    """Test database migration scripts"""
    
    def test_migration_sequence(self):
        """Test migrations run in correct order"""
        import os
        import glob
        
        migration_files = sorted(glob.glob('schema/migrations/*.sql'))
        
        # Verify naming convention (001_name.sql, 002_name.sql, etc.)
        for i, filepath in enumerate(migration_files, 1):
            filename = os.path.basename(filepath)
            expected_prefix = f"{i:03d}_"
            assert filename.startswith(expected_prefix), \
                f"Migration {filename} should start with {expected_prefix}"
    
    def test_migration_rollback(self, db_connection):
        """Test migration rollback capabilities"""
        cursor = db_connection.cursor()
        
        # Apply migration
        migration_up = """
            CREATE TABLE test_migration (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(100)
            );
        """
        
        migration_down = """
            DROP TABLE IF EXISTS test_migration;
        """
        
        # Apply migration
        cursor.execute(migration_up)
        db_connection.commit()
        
        # Verify table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'test_migration'
            )
        """)
        assert cursor.fetchone()[0] == True
        
        # Rollback migration
        cursor.execute(migration_down)
        db_connection.commit()
        
        # Verify table removed
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'test_migration'
            )
        """)
        assert cursor.fetchone()[0] == False
    
    def test_migration_idempotency(self, db_connection):
        """Test migrations are idempotent"""
        cursor = db_connection.cursor()
        
        migration = """
            CREATE TABLE IF NOT EXISTS idempotent_test (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid()
            );
            
            CREATE INDEX IF NOT EXISTS idx_idempotent_test_id 
            ON idempotent_test(id);
        """
        
        # Run migration twice
        cursor.execute(migration)
        db_connection.commit()
        
        cursor.execute(migration)  # Should not error
        db_connection.commit()
        
        # Verify table exists only once
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = 'idempotent_test'
        """)
        assert cursor.fetchone()[0] == 1
```

### 8.2 Data Migration Testing

```python
class TestDataMigration:
    """Test data migration procedures"""
    
    def test_data_transformation(self, db_connection):
        """Test data transformation during migration"""
        cursor = db_connection.cursor()
        
        # Create old schema
        cursor.execute("""
            CREATE TABLE old_users (
                id SERIAL PRIMARY KEY,
                full_name VARCHAR(200),
                email_address VARCHAR(200)
            )
        """)
        
        # Insert test data
        cursor.execute("""
            INSERT INTO old_users (full_name, email_address)
            VALUES 
            ('John Doe', 'john@example.com'),
            ('Jane Smith', 'jane@example.com')
        """)
        
        # Migrate data to new schema
        cursor.execute("""
            INSERT INTO users (email, username, password_hash)
            SELECT 
                email_address,
                LOWER(REPLACE(full_name, ' ', '_')),
                'temp_hash_' || id
            FROM old_users
        """)
        
        db_connection.commit()
        
        # Verify migration
        cursor.execute("""
            SELECT username FROM users WHERE email = 'john@example.com'
        """)
        username = cursor.fetchone()[0]
        assert username == 'john_doe'
    
    def test_large_data_migration(self, db_connection):
        """Test migration performance with large datasets"""
        cursor = db_connection.cursor()
        
        # Create large dataset
        cursor.execute("""
            INSERT INTO users (email, username, password_hash)
            SELECT 
                'user' || generate_series || '@test.com',
                'user' || generate_series,
                'hash' || generate_series
            FROM generate_series(1, 10000)
        """)
        
        start_time = time.perf_counter()
        
        # Perform migration operation (e.g., add computed column)
        cursor.execute("""
            ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name VARCHAR(100);
            UPDATE users SET display_name = UPPER(username);
        """)
        
        db_connection.commit()
        migration_time = time.perf_counter() - start_time
        
        print(f"Migration of 10,000 records took {migration_time:.2f}s")
        assert migration_time < 30, f"Migration too slow: {migration_time:.2f}s"
```

---

## 9. Test Automation

### 9.1 Continuous Integration Tests

```yaml
# .github/workflows/database-tests.yml
name: Database Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  database-tests:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: ytempire_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements-test.txt
      
      - name: Run migrations
        env:
          DATABASE_URL: postgresql://postgres:testpass@localhost/ytempire_test
        run: |
          python scripts/migrate.py
      
      - name: Run schema tests
        run: |
          pytest tests/database/test_schema.py -v
      
      - name: Run data validation tests
        run: |
          pytest tests/database/test_validation.py -v
      
      - name: Run performance tests
        run: |
          pytest tests/database/test_performance.py -v --benchmark
      
      - name: Run integrity tests
        run: |
          pytest tests/database/test_integrity.py -v
      
      - name: Generate coverage report
        run: |
          pytest tests/database/ --cov=database --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### 9.2 Automated Test Execution

```python
# run_database_tests.py - Automated test runner
import subprocess
import sys
import json
from datetime import datetime

class DatabaseTestRunner:
    """Automated database test execution"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'skipped': 0
            }
        }
    
    def run_test_suite(self, suite_name, test_path):
        """Run a specific test suite"""
        print(f"\n{'='*60}")
        print(f"Running {suite_name}")
        print('='*60)
        
        result = subprocess.run(
            ['pytest', test_path, '-v', '--json-report', '--json-report-file=report.json'],
            capture_output=True,
            text=True
        )
        
        # Parse results
        with open('report.json', 'r') as f:
            report = json.load(f)
        
        self.results['tests'][suite_name] = {
            'duration': report['duration'],
            'passed': report['summary']['passed'],
            'failed': report['summary']['failed'],
            'total': report['summary']['total']
        }
        
        # Update summary
        self.results['summary']['total'] += report['summary']['total']
        self.results['summary']['passed'] += report['summary']['passed']
        self.results['summary']['failed'] += report['summary']['failed']
        
        return result.returncode == 0
    
    def run_all_tests(self):
        """Run all database tests"""
        test_suites = [
            ('Schema Tests', 'tests/database/test_schema.py'),
            ('Data Validation', 'tests/database/test_validation.py'),
            ('Performance Tests', 'tests/database/test_performance.py'),
            ('Integrity Tests', 'tests/database/test_integrity.py'),
            ('Migration Tests', 'tests/database/test_migrations.py')
        ]
        
        all_passed = True
        
        for suite_name, test_path in test_suites:
            passed = self.run_test_suite(suite_name, test_path)
            if not passed:
                all_passed = False
        
        # Generate report
        self.generate_report()
        
        return all_passed
    
    def generate_report(self):
        """Generate test report"""
        print("\n" + "="*60)
        print("DATABASE TEST SUMMARY")
        print("="*60)
        
        print(f"\nTotal Tests: {self.results['summary']['total']}")
        print(f"Passed: {self.results['summary']['passed']}")
        print(f"Failed: {self.results['summary']['failed']}")
        
        print("\nTest Suite Results:")
        for suite, data in self.results['tests'].items():
            status = "✅" if data['failed'] == 0 else "❌"
            print(f"{status} {suite}: {data['passed']}/{data['total']} passed ({data['duration']:.2f}s)")
        
        # Save report
        with open('database_test_report.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nDetailed report saved to: database_test_report.json")

if __name__ == "__main__":
    runner = DatabaseTestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)
```

---

## 10. Monitoring & Alerting

### 10.1 Database Monitoring Queries

```sql
-- monitoring_queries.sql - Production monitoring

-- Active connections monitoring
CREATE OR REPLACE VIEW active_connections AS
SELECT 
    pid,
    usename,
    application_name,
    client_addr,
    state,
    query_start,
    state_change,
    query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY query_start;

-- Slow query monitoring
CREATE OR REPLACE VIEW slow_queries AS
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    max_time,
    stddev_time
FROM pg_stat_statements
WHERE mean_time > 100  -- Queries averaging over 100ms
ORDER BY mean_time DESC
LIMIT 20;

-- Table size monitoring
CREATE OR REPLACE VIEW table_sizes AS
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Index usage monitoring
CREATE OR REPLACE VIEW index_usage AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Lock monitoring
CREATE OR REPLACE VIEW blocking_locks AS
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS blocking_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks 
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.DATABASE IS NOT DISTINCT FROM blocked_locks.DATABASE
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.GRANTED;
```

### 10.2 Automated Monitoring Script

```python
# database_monitor.py - Automated database monitoring
import psycopg2
import time
import logging
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

class DatabaseMonitor:
    """Automated database monitoring and alerting"""
    
    def __init__(self, connection_string):
        self.conn_string = connection_string
        self.alerts = []
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def check_connection_count(self):
        """Monitor connection usage"""
        conn = psycopg2.connect(self.conn_string)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                count(*) as current,
                setting::int as max_conn
            FROM pg_stat_activity, pg_settings
            WHERE name = 'max_connections'
            GROUP BY setting
        """)
        
        current, max_conn = cursor.fetchone()
        usage_percent = (current / max_conn) * 100
        
        if usage_percent > 80:
            self.alert(
                f"High connection usage: {current}/{max_conn} ({usage_percent:.1f}%)",
                severity='WARNING'
            )
        
        conn.close()
        return {'current': current, 'max': max_conn, 'usage_percent': usage_percent}
    
    def check_slow_queries(self):
        """Monitor for slow queries"""
        conn = psycopg2.connect(self.conn_string)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                query,
                now() - query_start as duration,
                state
            FROM pg_stat_activity
            WHERE state != 'idle'
            AND now() - query_start > interval '5 seconds'
            ORDER BY duration DESC
            LIMIT 5
        """)
        
        slow_queries = cursor.fetchall()
        
        if slow_queries:
            for query, duration, state in slow_queries:
                self.alert(
                    f"Slow query detected ({duration}): {query[:100]}...",
                    severity='WARNING'
                )
        
        conn.close()
        return slow_queries
    
    def check_table_bloat(self):
        """Monitor table bloat"""
        conn = psycopg2.connect(self.conn_string)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                n_dead_tup,
                n_live_tup,
                round(n_dead_tup::numeric / NULLIF(n_live_tup, 0) * 100, 2) as dead_ratio
            FROM pg_stat_user_tables
            WHERE n_dead_tup > 1000
            AND n_live_tup > 0
            ORDER BY dead_ratio DESC
            LIMIT 5
        """)
        
        bloated_tables = cursor.fetchall()
        
        for schema, table, size, dead, live, ratio in bloated_tables:
            if ratio > 20:  # More than 20% dead tuples
                self.alert(
                    f"Table bloat detected: {schema}.{table} has {ratio}% dead tuples",
                    severity='WARNING'
                )
        
        conn.close()
        return bloated_tables
    
    def check_replication_lag(self):
        """Monitor replication lag (if configured)"""
        conn = psycopg2.connect(self.conn_string)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                client_addr,
                state,
                sent_lsn,
                write_lsn,
                flush_lsn,
                replay_lsn,
                write_lag,
                flush_lag,
                replay_lag
            FROM pg_stat_replication
        """)
        
        replicas = cursor.fetchall()
        
        for replica in replicas:
            if replica[8]:  # replay_lag
                lag_seconds = replica[8].total_seconds()
                if lag_seconds > 10:
                    self.alert(
                        f"High replication lag: {lag_seconds:.1f} seconds for {replica[0]}",
                        severity='CRITICAL'
                    )
        
        conn.close()
        return replicas
    
    def alert(self, message, severity='INFO'):
        """Send alert notification"""
        self.logger.log(
            logging.WARNING if severity == 'WARNING' else logging.CRITICAL,
            message
        )
        
        self.alerts.append({
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'message': message
        })
        
        # Send email for critical alerts
        if severity == 'CRITICAL':
            self.send_email_alert(message)
    
    def send_email_alert(self, message):
        """Send email for critical alerts"""
        # Configure with your SMTP settings
        smtp_config = {
            'host': 'smtp.gmail.com',
            'port': 587,
            'user': 'alerts@ytempire.com',
            'password': 'your_password',
            'to': ['qa-team@ytempire.com']
        }
        
        msg = MIMEText(f"Critical Database Alert:\n\n{message}")
        msg['Subject'] = 'YTEMPIRE Database Alert'
        msg['From'] = smtp_config['user']
        msg['To'] = ', '.join(smtp_config['to'])
        
        try:
            with smtplib.SMTP(smtp_config['host'], smtp_config['port']) as server:
                server.starttls()
                server.login(smtp_config['user'], smtp_config['password'])
                server.send_message(msg)
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def run_monitoring_loop(self, interval=60):
        """Run continuous monitoring"""
        self.logger.info("Starting database monitoring...")
        
        while True:
            try:
                # Run all checks
                self.check_connection_count()
                self.check_slow_queries()
                self.check_table_bloat()
                self.check_replication_lag()
                
                # Clear old alerts
                if len(self.alerts) > 1000:
                    self.alerts = self.alerts[-500:]
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(interval)

if __name__ == "__main__":
    monitor = DatabaseMonitor(
        "postgresql://monitor_user:password@localhost/ytempire"
    )
    monitor.run_monitoring_loop()
```

---

## Appendix A: Quick Reference

### Common Test Commands

```bash
# Run all database tests
pytest tests/database/ -v

# Run specific test suite
pytest tests/database/test_performance.py -v

# Run with coverage
pytest tests/database/ --cov=database --cov-report=html

# Run with benchmarks
pytest tests/database/test_performance.py --benchmark-only

# Run in parallel
pytest tests/database/ -n 4

# Run with specific marker
pytest tests/database/ -m "slow"
```

### Test Database Setup

```bash
# Create test database
createdb ytempire_test

# Run migrations
psql ytempire_test < schema/migrations/*.sql

# Load test data
python scripts/load_test_data.py

# Clean test database
dropdb ytempire_test
```

### Performance Baseline Targets

| Metric | Target | Critical |
|--------|--------|----------|
| Query Response (p95) | <100ms | >500ms |
| Bulk Insert (1000 records) | <5s | >10s |
| Connection Pool | 200 | 250 |
| Transaction Rate | 1000/s | <100/s |
| Deadlock Rate | <0.1% | >1% |

---

## Document Control

- **Version**: 1.0
- **Last Updated**: January 2025
- **Review Schedule**: Weekly during MVP, Monthly post-launch
- **Owner**: QA Engineering Team
- **Next Review**: End of Week 1

**Approval Chain:**
1. QA Engineer (Author) ✅
2. Platform Operations Lead (Review) ✅
3. Backend Team Lead (Technical Review)