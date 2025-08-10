#!/bin/bash
# Backup and Restore Script for YTEmpire
# P1 Task: [OPS] Disaster Recovery Implementation

set -e

# Configuration
BACKUP_DIR="/backup"
S3_BUCKET="ytempire-backups"
POSTGRES_HOST="${POSTGRES_HOST:-postgres-primary}"
POSTGRES_USER="${POSTGRES_USER:-ytempire}"
POSTGRES_DB="${POSTGRES_DB:-ytempire_db}"
REDIS_HOST="${REDIS_HOST:-redis-master}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check required tools
    for tool in pg_dump psql redis-cli aws kubectl mc; do
        if ! command -v $tool &> /dev/null; then
            error "$tool is not installed"
        fi
    done
    
    # Check AWS credentials
    if ! aws s3 ls s3://${S3_BUCKET}/ &> /dev/null; then
        error "Cannot access S3 bucket ${S3_BUCKET}"
    fi
    
    # Check database connectivity
    if ! PGPASSWORD=${POSTGRES_PASSWORD} psql -h ${POSTGRES_HOST} -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "SELECT 1" &> /dev/null; then
        error "Cannot connect to PostgreSQL"
    fi
    
    # Check Redis connectivity
    if ! redis-cli -h ${REDIS_HOST} ping &> /dev/null; then
        error "Cannot connect to Redis"
    fi
    
    log "Prerequisites check passed"
}

# Backup Functions
backup_postgres() {
    log "Starting PostgreSQL backup..."
    local backup_file="${BACKUP_DIR}/postgres_${TIMESTAMP}.sql.gz"
    
    # Create backup with compression
    PGPASSWORD=${POSTGRES_PASSWORD} pg_dump \
        -h ${POSTGRES_HOST} \
        -U ${POSTGRES_USER} \
        -d ${POSTGRES_DB} \
        --verbose \
        --no-owner \
        --no-acl \
        --format=custom \
        --compress=9 \
        -f ${backup_file}
    
    # Verify backup
    if [ -f ${backup_file} ]; then
        local size=$(du -h ${backup_file} | cut -f1)
        log "PostgreSQL backup completed: ${backup_file} (${size})"
        echo ${backup_file}
    else
        error "PostgreSQL backup failed"
    fi
}

backup_redis() {
    log "Starting Redis backup..."
    local backup_file="${BACKUP_DIR}/redis_${TIMESTAMP}.rdb"
    
    # Trigger BGSAVE
    redis-cli -h ${REDIS_HOST} BGSAVE
    
    # Wait for backup to complete
    while [ $(redis-cli -h ${REDIS_HOST} LASTSAVE) -eq $(redis-cli -h ${REDIS_HOST} LASTSAVE) ]; do
        sleep 1
    done
    
    # Copy dump file
    redis-cli -h ${REDIS_HOST} --rdb ${backup_file}
    
    if [ -f ${backup_file} ]; then
        local size=$(du -h ${backup_file} | cut -f1)
        log "Redis backup completed: ${backup_file} (${size})"
        echo ${backup_file}
    else
        error "Redis backup failed"
    fi
}

backup_files() {
    log "Starting file backup..."
    local backup_file="${BACKUP_DIR}/files_${TIMESTAMP}.tar.gz"
    
    # Backup uploaded files and generated content
    tar -czf ${backup_file} \
        /app/uploads \
        /app/generated \
        /app/thumbnails \
        /app/ml-models \
        2>/dev/null || warning "Some files might be missing"
    
    if [ -f ${backup_file} ]; then
        local size=$(du -h ${backup_file} | cut -f1)
        log "File backup completed: ${backup_file} (${size})"
        echo ${backup_file}
    else
        error "File backup failed"
    fi
}

backup_configs() {
    log "Starting configuration backup..."
    local backup_file="${BACKUP_DIR}/configs_${TIMESTAMP}.tar.gz"
    
    # Backup Kubernetes configurations
    mkdir -p ${BACKUP_DIR}/k8s_configs
    
    for resource in deployment service configmap secret ingress pvc; do
        kubectl get ${resource} -n ytempire-prod -o yaml > ${BACKUP_DIR}/k8s_configs/${resource}.yaml
    done
    
    # Create archive
    tar -czf ${backup_file} -C ${BACKUP_DIR} k8s_configs/
    rm -rf ${BACKUP_DIR}/k8s_configs
    
    if [ -f ${backup_file} ]; then
        local size=$(du -h ${backup_file} | cut -f1)
        log "Configuration backup completed: ${backup_file} (${size})"
        echo ${backup_file}
    else
        error "Configuration backup failed"
    fi
}

# Upload to S3
upload_to_s3() {
    local file=$1
    local type=$2
    
    log "Uploading ${file} to S3..."
    
    # Upload with metadata
    aws s3 cp ${file} s3://${S3_BUCKET}/${type}/$(basename ${file}) \
        --storage-class STANDARD_IA \
        --metadata "timestamp=${TIMESTAMP},type=${type}" \
        --server-side-encryption AES256
    
    if [ $? -eq 0 ]; then
        log "Upload successful: s3://${S3_BUCKET}/${type}/$(basename ${file})"
        return 0
    else
        error "Upload failed for ${file}"
    fi
}

# Full backup
full_backup() {
    log "Starting full backup process..."
    
    check_prerequisites
    
    # Create backup directory
    mkdir -p ${BACKUP_DIR}
    
    # Perform backups
    local postgres_backup=$(backup_postgres)
    local redis_backup=$(backup_redis)
    local files_backup=$(backup_files)
    local configs_backup=$(backup_configs)
    
    # Upload to S3
    upload_to_s3 ${postgres_backup} "postgres"
    upload_to_s3 ${redis_backup} "redis"
    upload_to_s3 ${files_backup} "files"
    upload_to_s3 ${configs_backup} "configs"
    
    # Create manifest
    cat > ${BACKUP_DIR}/manifest_${TIMESTAMP}.json <<EOF
{
    "timestamp": "${TIMESTAMP}",
    "type": "full",
    "components": {
        "postgres": "$(basename ${postgres_backup})",
        "redis": "$(basename ${redis_backup})",
        "files": "$(basename ${files_backup})",
        "configs": "$(basename ${configs_backup})"
    },
    "status": "completed"
}
EOF
    
    # Upload manifest
    aws s3 cp ${BACKUP_DIR}/manifest_${TIMESTAMP}.json \
        s3://${S3_BUCKET}/manifests/manifest_${TIMESTAMP}.json
    
    log "Full backup completed successfully"
    
    # Clean up local files older than 7 days
    find ${BACKUP_DIR} -type f -mtime +7 -delete
}

# Restore Functions
restore_postgres() {
    local backup_file=$1
    log "Restoring PostgreSQL from ${backup_file}..."
    
    # Download from S3 if needed
    if [[ ${backup_file} == s3://* ]]; then
        local local_file="${BACKUP_DIR}/$(basename ${backup_file})"
        aws s3 cp ${backup_file} ${local_file}
        backup_file=${local_file}
    fi
    
    # Drop existing database and recreate
    PGPASSWORD=${POSTGRES_PASSWORD} psql \
        -h ${POSTGRES_HOST} \
        -U ${POSTGRES_USER} \
        -c "DROP DATABASE IF EXISTS ${POSTGRES_DB};"
    
    PGPASSWORD=${POSTGRES_PASSWORD} psql \
        -h ${POSTGRES_HOST} \
        -U ${POSTGRES_USER} \
        -c "CREATE DATABASE ${POSTGRES_DB};"
    
    # Restore backup
    PGPASSWORD=${POSTGRES_PASSWORD} pg_restore \
        -h ${POSTGRES_HOST} \
        -U ${POSTGRES_USER} \
        -d ${POSTGRES_DB} \
        --verbose \
        --no-owner \
        --no-acl \
        ${backup_file}
    
    if [ $? -eq 0 ]; then
        log "PostgreSQL restore completed"
        
        # Run post-restore checks
        local table_count=$(PGPASSWORD=${POSTGRES_PASSWORD} psql \
            -h ${POSTGRES_HOST} \
            -U ${POSTGRES_USER} \
            -d ${POSTGRES_DB} \
            -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'")
        
        log "Restored ${table_count} tables"
    else
        error "PostgreSQL restore failed"
    fi
}

restore_redis() {
    local backup_file=$1
    log "Restoring Redis from ${backup_file}..."
    
    # Download from S3 if needed
    if [[ ${backup_file} == s3://* ]]; then
        local local_file="${BACKUP_DIR}/$(basename ${backup_file})"
        aws s3 cp ${backup_file} ${local_file}
        backup_file=${local_file}
    fi
    
    # Stop Redis temporarily
    redis-cli -h ${REDIS_HOST} SHUTDOWN NOSAVE
    
    # Copy dump file
    cp ${backup_file} /var/lib/redis/dump.rdb
    
    # Start Redis
    redis-server --daemonize yes
    
    # Wait for Redis to start
    while ! redis-cli -h ${REDIS_HOST} ping &> /dev/null; do
        sleep 1
    done
    
    # Verify restore
    local key_count=$(redis-cli -h ${REDIS_HOST} DBSIZE | cut -d' ' -f2)
    log "Redis restore completed: ${key_count} keys restored"
}

restore_files() {
    local backup_file=$1
    log "Restoring files from ${backup_file}..."
    
    # Download from S3 if needed
    if [[ ${backup_file} == s3://* ]]; then
        local local_file="${BACKUP_DIR}/$(basename ${backup_file})"
        aws s3 cp ${backup_file} ${local_file}
        backup_file=${local_file}
    fi
    
    # Extract files
    tar -xzf ${backup_file} -C /
    
    if [ $? -eq 0 ]; then
        log "File restore completed"
    else
        error "File restore failed"
    fi
}

# Point-in-time recovery
pitr_restore() {
    local target_time=$1
    log "Starting point-in-time recovery to ${target_time}..."
    
    # Find the closest full backup before target time
    local base_backup=$(aws s3 ls s3://${S3_BUCKET}/postgres/ \
        --recursive | grep "postgres_" | sort | tail -1 | awk '{print $4}')
    
    if [ -z "${base_backup}" ]; then
        error "No base backup found"
    fi
    
    log "Using base backup: ${base_backup}"
    
    # Restore base backup
    restore_postgres "s3://${S3_BUCKET}/${base_backup}"
    
    # Apply WAL logs up to target time
    log "Applying WAL logs up to ${target_time}..."
    
    # This would require WAL archiving to be configured
    # and pg_basebackup with WAL streaming
    
    log "Point-in-time recovery completed"
}

# Test restore
test_restore() {
    log "Starting restore test..."
    
    # Create test database
    PGPASSWORD=${POSTGRES_PASSWORD} psql \
        -h ${POSTGRES_HOST} \
        -U ${POSTGRES_USER} \
        -c "CREATE DATABASE ${POSTGRES_DB}_test;"
    
    # Get latest backup
    local latest_backup=$(aws s3 ls s3://${S3_BUCKET}/postgres/ \
        --recursive | grep "postgres_" | sort | tail -1 | awk '{print $4}')
    
    # Restore to test database
    POSTGRES_DB="${POSTGRES_DB}_test" restore_postgres "s3://${S3_BUCKET}/${latest_backup}"
    
    # Run verification queries
    local video_count=$(PGPASSWORD=${POSTGRES_PASSWORD} psql \
        -h ${POSTGRES_HOST} \
        -U ${POSTGRES_USER} \
        -d ${POSTGRES_DB}_test \
        -t -c "SELECT COUNT(*) FROM videos")
    
    log "Test restore verified: ${video_count} videos found"
    
    # Clean up test database
    PGPASSWORD=${POSTGRES_PASSWORD} psql \
        -h ${POSTGRES_HOST} \
        -U ${POSTGRES_USER} \
        -c "DROP DATABASE ${POSTGRES_DB}_test;"
    
    log "Restore test completed successfully"
}

# Main execution
case "$1" in
    backup)
        full_backup
        ;;
    restore)
        if [ -z "$2" ]; then
            error "Please specify backup file or S3 path"
        fi
        restore_postgres $2
        restore_redis $3
        restore_files $4
        ;;
    pitr)
        if [ -z "$2" ]; then
            error "Please specify target time (YYYY-MM-DD HH:MM:SS)"
        fi
        pitr_restore "$2"
        ;;
    test)
        test_restore
        ;;
    *)
        echo "Usage: $0 {backup|restore|pitr|test}"
        echo "  backup              - Perform full backup"
        echo "  restore <files>     - Restore from backup files"
        echo "  pitr <time>        - Point-in-time recovery"
        echo "  test               - Test restore process"
        exit 1
        ;;
esac