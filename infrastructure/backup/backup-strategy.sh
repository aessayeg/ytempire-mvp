#!/bin/bash

# YTEmpire Backup Strategy Implementation
# Automated backup for PostgreSQL, Redis, and application files

set -e

# Configuration
BACKUP_DIR="/backup/ytempire"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Database Configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-ytempire}"
DB_USER="${DB_USER:-ytempire_user}"

# S3 Configuration
S3_BUCKET="${S3_BUCKET:-ytempire-backups}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Create backup directories
mkdir -p "$BACKUP_DIR/daily"
mkdir -p "$BACKUP_DIR/weekly"
mkdir -p "$BACKUP_DIR/monthly"

# Function: Backup PostgreSQL
backup_postgres() {
    echo "Starting PostgreSQL backup..."
    
    PGPASSWORD="$DB_PASSWORD" pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -Fc \
        -f "$BACKUP_DIR/daily/postgres_${TIMESTAMP}.dump"
    
    echo "PostgreSQL backup completed"
}

# Function: Backup Redis
backup_redis() {
    echo "Starting Redis backup..."
    
    redis-cli --rdb "$BACKUP_DIR/daily/redis_${TIMESTAMP}.rdb"
    
    echo "Redis backup completed"
}

# Function: Backup application files
backup_files() {
    echo "Starting application files backup..."
    
    tar -czf "$BACKUP_DIR/daily/app_files_${TIMESTAMP}.tar.gz" \
        /app/uploads \
        /app/config \
        /app/ml-models \
        --exclude='*.log' \
        --exclude='node_modules' \
        --exclude='__pycache__'
    
    echo "Application files backup completed"
}

# Function: Upload to S3
upload_to_s3() {
    echo "Uploading backups to S3..."
    
    aws s3 cp "$BACKUP_DIR/daily/" \
        "s3://$S3_BUCKET/daily/" \
        --recursive \
        --exclude "*" \
        --include "*_${TIMESTAMP}.*" \
        --storage-class STANDARD_IA
    
    echo "S3 upload completed"
}

# Function: Cleanup old backups
cleanup_old_backups() {
    echo "Cleaning up old backups..."
    
    # Local cleanup
    find "$BACKUP_DIR/daily" -type f -mtime +$RETENTION_DAYS -delete
    
    # S3 cleanup
    aws s3 ls "s3://$S3_BUCKET/daily/" | \
        while read -r line; do
            createDate=$(echo "$line" | awk '{print $1" "$2}')
            createDate=$(date -d "$createDate" +%s)
            olderThan=$(date -d "$RETENTION_DAYS days ago" +%s)
            if [[ $createDate -lt $olderThan ]]; then
                fileName=$(echo "$line" | awk '{print $4}')
                aws s3 rm "s3://$S3_BUCKET/daily/$fileName"
            fi
        done
    
    echo "Cleanup completed"
}

# Function: Verify backup
verify_backup() {
    echo "Verifying backups..."
    
    # Test PostgreSQL backup
    pg_restore --list "$BACKUP_DIR/daily/postgres_${TIMESTAMP}.dump" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "PostgreSQL backup verified"
    else
        echo "PostgreSQL backup verification failed"
        exit 1
    fi
    
    # Test tar archive
    tar -tzf "$BACKUP_DIR/daily/app_files_${TIMESTAMP}.tar.gz" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "Application files backup verified"
    else
        echo "Application files backup verification failed"
        exit 1
    fi
}

# Main execution
main() {
    echo "=== YTEmpire Backup Started at $TIMESTAMP ==="
    
    backup_postgres
    backup_redis
    backup_files
    verify_backup
    upload_to_s3
    cleanup_old_backups
    
    echo "=== YTEmpire Backup Completed Successfully ==="
}

# Run main function
main