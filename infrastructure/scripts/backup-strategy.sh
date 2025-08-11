#!/bin/bash

# YTEmpire Backup Strategy Implementation
# Automated backup script for PostgreSQL, Redis, and application data

set -e

# Configuration
BACKUP_DIR="/backups/ytempire"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Database configuration
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-ytempire}
DB_USER=${DB_USER:-ytempire_user}

# Redis configuration
REDIS_HOST=${REDIS_HOST:-localhost}
REDIS_PORT=${REDIS_PORT:-6379}

# S3 configuration (optional)
S3_BUCKET=${S3_BUCKET:-}
AWS_PROFILE=${AWS_PROFILE:-default}

# Create backup directories
mkdir -p "$BACKUP_DIR/postgresql"
mkdir -p "$BACKUP_DIR/redis"
mkdir -p "$BACKUP_DIR/files"
mkdir -p "$BACKUP_DIR/logs"

# Logging
LOG_FILE="$BACKUP_DIR/logs/backup_${TIMESTAMP}.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo "======================================"
echo "YTEmpire Backup Started: $(date)"
echo "======================================"

# Function to check command success
check_status() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

# 1. PostgreSQL Backup
echo "📦 Backing up PostgreSQL database..."
PGPASSWORD=$DB_PASSWORD pg_dump \
    -h $DB_HOST \
    -p $DB_PORT \
    -U $DB_USER \
    -d $DB_NAME \
    --no-owner \
    --no-acl \
    --format=custom \
    --compress=9 \
    --file="$BACKUP_DIR/postgresql/ytempire_${TIMESTAMP}.dump"
check_status "PostgreSQL backup"

# Create SQL format backup for portability
PGPASSWORD=$DB_PASSWORD pg_dump \
    -h $DB_HOST \
    -p $DB_PORT \
    -U $DB_USER \
    -d $DB_NAME \
    --no-owner \
    --no-acl \
    --format=plain \
    | gzip > "$BACKUP_DIR/postgresql/ytempire_${TIMESTAMP}.sql.gz"
check_status "PostgreSQL SQL backup"

# 2. Redis Backup
echo "📦 Backing up Redis data..."
redis-cli -h $REDIS_HOST -p $REDIS_PORT --rdb "$BACKUP_DIR/redis/dump_${TIMESTAMP}.rdb"
check_status "Redis backup"

# Save Redis AOF if enabled
if redis-cli -h $REDIS_HOST -p $REDIS_PORT CONFIG GET appendonly | grep -q "yes"; then
    cp /var/lib/redis/appendonly.aof "$BACKUP_DIR/redis/appendonly_${TIMESTAMP}.aof" 2>/dev/null || true
fi

# 3. Application Files Backup
echo "📦 Backing up application files..."
# Backup uploaded files
if [ -d "/app/uploads" ]; then
    tar -czf "$BACKUP_DIR/files/uploads_${TIMESTAMP}.tar.gz" -C /app uploads
    check_status "Uploads backup"
fi

# Backup generated videos
if [ -d "/app/videos" ]; then
    tar -czf "$BACKUP_DIR/files/videos_${TIMESTAMP}.tar.gz" -C /app videos
    check_status "Videos backup"
fi

# Backup configuration files
tar -czf "$BACKUP_DIR/files/config_${TIMESTAMP}.tar.gz" \
    /app/.env \
    /app/backend/alembic.ini \
    /app/ml-pipeline/config.yaml \
    /app/infrastructure/docker-compose.yml \
    2>/dev/null || true
check_status "Configuration backup"

# 4. Create backup manifest
echo "📝 Creating backup manifest..."
cat > "$BACKUP_DIR/manifest_${TIMESTAMP}.json" <<EOF
{
    "timestamp": "${TIMESTAMP}",
    "date": "$(date -Iseconds)",
    "type": "full",
    "components": {
        "postgresql": {
            "dump": "postgresql/ytempire_${TIMESTAMP}.dump",
            "sql": "postgresql/ytempire_${TIMESTAMP}.sql.gz",
            "size": "$(du -h $BACKUP_DIR/postgresql/ytempire_${TIMESTAMP}.dump | cut -f1)"
        },
        "redis": {
            "rdb": "redis/dump_${TIMESTAMP}.rdb",
            "size": "$(du -h $BACKUP_DIR/redis/dump_${TIMESTAMP}.rdb | cut -f1)"
        },
        "files": {
            "uploads": "files/uploads_${TIMESTAMP}.tar.gz",
            "videos": "files/videos_${TIMESTAMP}.tar.gz",
            "config": "files/config_${TIMESTAMP}.tar.gz"
        }
    },
    "retention_days": ${RETENTION_DAYS}
}
EOF
check_status "Manifest creation"

# 5. Compress full backup
echo "🗜️ Creating compressed archive..."
cd "$BACKUP_DIR"
tar -czf "ytempire_backup_${TIMESTAMP}.tar.gz" \
    "postgresql/ytempire_${TIMESTAMP}".* \
    "redis/dump_${TIMESTAMP}.rdb" \
    "files/*_${TIMESTAMP}.tar.gz" \
    "manifest_${TIMESTAMP}.json"
check_status "Archive creation"

# 6. Upload to S3 (if configured)
if [ -n "$S3_BUCKET" ]; then
    echo "☁️ Uploading to S3..."
    aws s3 cp \
        "ytempire_backup_${TIMESTAMP}.tar.gz" \
        "s3://${S3_BUCKET}/backups/ytempire_backup_${TIMESTAMP}.tar.gz" \
        --profile $AWS_PROFILE \
        --storage-class STANDARD_IA
    check_status "S3 upload"
    
    # Copy manifest for easy listing
    aws s3 cp \
        "manifest_${TIMESTAMP}.json" \
        "s3://${S3_BUCKET}/backups/manifests/manifest_${TIMESTAMP}.json" \
        --profile $AWS_PROFILE
fi

# 7. Cleanup old backups
echo "🧹 Cleaning up old backups..."
find "$BACKUP_DIR" -name "ytempire_backup_*.tar.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR/postgresql" -name "*.dump" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR/postgresql" -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR/redis" -name "*.rdb" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR/files" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
check_status "Cleanup"

# 8. Verify backup integrity
echo "✔️ Verifying backup integrity..."
tar -tzf "ytempire_backup_${TIMESTAMP}.tar.gz" > /dev/null
check_status "Backup verification"

# 9. Generate backup report
BACKUP_SIZE=$(du -h "ytempire_backup_${TIMESTAMP}.tar.gz" | cut -f1)
echo ""
echo "======================================"
echo "📊 Backup Summary"
echo "======================================"
echo "Timestamp: ${TIMESTAMP}"
echo "Total Size: ${BACKUP_SIZE}"
echo "Location: $BACKUP_DIR/ytempire_backup_${TIMESTAMP}.tar.gz"
if [ -n "$S3_BUCKET" ]; then
    echo "S3 Location: s3://${S3_BUCKET}/backups/ytempire_backup_${TIMESTAMP}.tar.gz"
fi
echo "Retention: ${RETENTION_DAYS} days"
echo "======================================"
echo "✅ Backup completed successfully!"
echo "======================================