#!/usr/bin/env python3
"""
Disaster Recovery and Backup Management System
Comprehensive backup and restore functionality for YTEmpire MVP
"""

import os
import json
import shutil
import asyncio
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import boto3
import psycopg2
from redis import Redis
from dataclasses import dataclass
from enum import Enum
import aiofiles
import tarfile
import gzip

logger = logging.getLogger(__name__)

class BackupType(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"

class BackupStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

@dataclass
class BackupConfig:
    """Backup configuration settings"""
    # Database settings
    db_host: str = os.getenv("DATABASE_HOST", "localhost")
    db_port: int = int(os.getenv("DATABASE_PORT", "5432"))
    db_name: str = os.getenv("DATABASE_NAME", "ytempire")
    db_user: str = os.getenv("DATABASE_USER", "postgres")
    db_password: str = os.getenv("DATABASE_PASSWORD", "password")
    
    # Redis settings
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: str = os.getenv("REDIS_PASSWORD", "")
    
    # Storage settings
    local_backup_dir: str = "/var/backups/ytempire"
    s3_bucket: str = os.getenv("BACKUP_S3_BUCKET", "ytempire-backups")
    s3_region: str = os.getenv("AWS_REGION", "us-east-1")
    
    # Retention settings
    local_retention_days: int = 7
    s3_retention_days: int = 90
    
    # Schedule settings
    full_backup_interval_hours: int = 24
    incremental_backup_interval_hours: int = 4
    
    # Encryption
    encryption_key: str = os.getenv("BACKUP_ENCRYPTION_KEY", "")
    
    # Monitoring
    webhook_url: str = os.getenv("BACKUP_WEBHOOK_URL", "")
    slack_webhook: str = os.getenv("SLACK_WEBHOOK_URL", "")

@dataclass
class BackupRecord:
    """Backup operation record"""
    id: str
    backup_type: BackupType
    status: BackupStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    error_message: Optional[str] = None
    components: List[str] = None
    metadata: Dict[str, Any] = None

class DatabaseBackupManager:
    """Manages PostgreSQL database backups"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        
    async def create_full_backup(self, backup_path: str) -> Dict[str, Any]:
        """Create a full database backup"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            db_backup_file = f"{backup_path}/db_full_{timestamp}.sql.gz"
            
            # Create pg_dump command
            dump_cmd = [
                "pg_dump",
                f"--host={self.config.db_host}",
                f"--port={self.config.db_port}",
                f"--username={self.config.db_user}",
                f"--dbname={self.config.db_name}",
                "--format=custom",
                "--compress=9",
                "--no-password",
                "--verbose"
            ]
            
            # Set password environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = self.config.db_password
            
            # Execute backup
            with gzip.open(db_backup_file, 'wb') as f:
                process = await asyncio.create_subprocess_exec(
                    *dump_cmd,
                    stdout=f,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                _, stderr = await process.communicate()
                
            if process.returncode == 0:
                file_size = os.path.getsize(db_backup_file)
                checksum = await self._calculate_checksum(db_backup_file)
                
                return {
                    "status": "success",
                    "file_path": db_backup_file,
                    "file_size": file_size,
                    "checksum": checksum
                }
            else:
                error = stderr.decode() if stderr else "Unknown pg_dump error"
                logger.error(f"Database backup failed: {error}")
                return {"status": "failed", "error": error}
                
        except Exception as e:
            logger.error(f"Database backup error: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def restore_database(self, backup_file: str) -> Dict[str, Any]:
        """Restore database from backup file"""
        try:
            # Validate backup file
            if not os.path.exists(backup_file):
                return {"status": "failed", "error": "Backup file not found"}
            
            # Create restore command
            restore_cmd = [
                "pg_restore",
                f"--host={self.config.db_host}",
                f"--port={self.config.db_port}",
                f"--username={self.config.db_user}",
                f"--dbname={self.config.db_name}",
                "--clean",
                "--if-exists",
                "--no-owner",
                "--no-privileges",
                "--verbose",
                backup_file
            ]
            
            # Set password environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = self.config.db_password
            
            # Execute restore
            process = await asyncio.create_subprocess_exec(
                *restore_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return {"status": "success", "output": stdout.decode()}
            else:
                error = stderr.decode() if stderr else "Unknown pg_restore error"
                logger.error(f"Database restore failed: {error}")
                return {"status": "failed", "error": error}
                
        except Exception as e:
            logger.error(f"Database restore error: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of file"""
        import hashlib
        hash_sha256 = hashlib.sha256()
        async with aiofiles.open(file_path, 'rb') as f:
            async for chunk in f:
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

class RedisBackupManager:
    """Manages Redis data backups"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.redis = Redis(
            host=config.redis_host,
            port=config.redis_port,
            password=config.redis_password if config.redis_password else None,
            decode_responses=True
        )
    
    async def create_backup(self, backup_path: str) -> Dict[str, Any]:
        """Create Redis backup"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            redis_backup_file = f"{backup_path}/redis_{timestamp}.rdb.gz"
            
            # Trigger Redis save
            self.redis.bgsave()
            
            # Wait for save to complete
            while self.redis.lastsave() == self.redis.lastsave():
                await asyncio.sleep(1)
            
            # Copy RDB file
            rdb_path = "/var/lib/redis/dump.rdb"  # Default Redis RDB path
            if os.path.exists(rdb_path):
                with open(rdb_path, 'rb') as f_in:
                    with gzip.open(redis_backup_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                file_size = os.path.getsize(redis_backup_file)
                return {
                    "status": "success",
                    "file_path": redis_backup_file,
                    "file_size": file_size
                }
            else:
                return {"status": "failed", "error": "Redis RDB file not found"}
                
        except Exception as e:
            logger.error(f"Redis backup error: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def restore_redis(self, backup_file: str) -> Dict[str, Any]:
        """Restore Redis from backup"""
        try:
            if not os.path.exists(backup_file):
                return {"status": "failed", "error": "Backup file not found"}
            
            # Stop Redis temporarily
            subprocess.run(["systemctl", "stop", "redis"], check=True)
            
            # Extract and copy RDB file
            rdb_path = "/var/lib/redis/dump.rdb"
            with gzip.open(backup_file, 'rb') as f_in:
                with open(rdb_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Set proper ownership
            subprocess.run(["chown", "redis:redis", rdb_path], check=True)
            
            # Restart Redis
            subprocess.run(["systemctl", "start", "redis"], check=True)
            
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Redis restore error: {e}")
            # Ensure Redis is restarted even on error
            try:
                subprocess.run(["systemctl", "start", "redis"])
            except:
                pass
            return {"status": "failed", "error": str(e)}

class FileSystemBackupManager:
    """Manages application file system backups"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.backup_paths = [
            "/app/uploads",
            "/app/generated_videos",
            "/app/thumbnails",
            "/app/config",
            "/var/log/ytempire"
        ]
    
    async def create_backup(self, backup_path: str) -> Dict[str, Any]:
        """Create file system backup"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fs_backup_file = f"{backup_path}/filesystem_{timestamp}.tar.gz"
            
            with tarfile.open(fs_backup_file, 'w:gz') as tar:
                for path in self.backup_paths:
                    if os.path.exists(path):
                        tar.add(path, arcname=os.path.basename(path))
            
            file_size = os.path.getsize(fs_backup_file)
            return {
                "status": "success",
                "file_path": fs_backup_file,
                "file_size": file_size
            }
            
        except Exception as e:
            logger.error(f"Filesystem backup error: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def restore_filesystem(self, backup_file: str) -> Dict[str, Any]:
        """Restore filesystem from backup"""
        try:
            if not os.path.exists(backup_file):
                return {"status": "failed", "error": "Backup file not found"}
            
            # Extract backup
            with tarfile.open(backup_file, 'r:gz') as tar:
                tar.extractall(path="/tmp/restore")
            
            # Copy files back to original locations
            restore_base = "/tmp/restore"
            for path in self.backup_paths:
                src_path = os.path.join(restore_base, os.path.basename(path))
                if os.path.exists(src_path):
                    if os.path.exists(path):
                        shutil.rmtree(path)
                    shutil.move(src_path, path)
            
            # Cleanup
            shutil.rmtree(restore_base, ignore_errors=True)
            
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Filesystem restore error: {e}")
            return {"status": "failed", "error": str(e)}

class S3BackupStorage:
    """Manages S3 backup storage"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.s3_client = boto3.client(
            's3',
            region_name=config.s3_region
        )
    
    async def upload_backup(self, local_file: str, s3_key: str) -> Dict[str, Any]:
        """Upload backup to S3"""
        try:
            # Upload file
            self.s3_client.upload_file(
                local_file,
                self.config.s3_bucket,
                s3_key,
                ExtraArgs={'StorageClass': 'STANDARD_IA'}  # Cost-effective storage class
            )
            
            # Verify upload
            response = self.s3_client.head_object(
                Bucket=self.config.s3_bucket,
                Key=s3_key
            )
            
            return {
                "status": "success",
                "s3_key": s3_key,
                "size": response['ContentLength'],
                "etag": response['ETag']
            }
            
        except Exception as e:
            logger.error(f"S3 upload error: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def download_backup(self, s3_key: str, local_file: str) -> Dict[str, Any]:
        """Download backup from S3"""
        try:
            self.s3_client.download_file(
                self.config.s3_bucket,
                s3_key,
                local_file
            )
            
            return {"status": "success", "local_file": local_file}
            
        except Exception as e:
            logger.error(f"S3 download error: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def cleanup_old_backups(self) -> Dict[str, Any]:
        """Clean up old backups based on retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.s3_retention_days)
            
            # List all objects
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.config.s3_bucket)
            
            deleted_count = 0
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                            self.s3_client.delete_object(
                                Bucket=self.config.s3_bucket,
                                Key=obj['Key']
                            )
                            deleted_count += 1
            
            return {"status": "success", "deleted_count": deleted_count}
            
        except Exception as e:
            logger.error(f"S3 cleanup error: {e}")
            return {"status": "failed", "error": str(e)}

class DisasterRecoveryManager:
    """Main disaster recovery and backup manager"""
    
    def __init__(self, config: BackupConfig = None):
        self.config = config or BackupConfig()
        self.db_manager = DatabaseBackupManager(self.config)
        self.redis_manager = RedisBackupManager(self.config)
        self.fs_manager = FileSystemBackupManager(self.config)
        self.s3_storage = S3BackupStorage(self.config)
        
        # Ensure backup directory exists
        Path(self.config.local_backup_dir).mkdir(parents=True, exist_ok=True)
    
    async def create_full_backup(self) -> BackupRecord:
        """Create a comprehensive full backup"""
        backup_id = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        record = BackupRecord(
            id=backup_id,
            backup_type=BackupType.FULL,
            status=BackupStatus.RUNNING,
            start_time=datetime.now(),
            components=["database", "redis", "filesystem"]
        )
        
        try:
            backup_dir = f"{self.config.local_backup_dir}/{backup_id}"
            Path(backup_dir).mkdir(parents=True, exist_ok=True)
            
            # Backup database
            logger.info("Starting database backup...")
            db_result = await self.db_manager.create_full_backup(backup_dir)
            
            # Backup Redis
            logger.info("Starting Redis backup...")
            redis_result = await self.redis_manager.create_backup(backup_dir)
            
            # Backup filesystem
            logger.info("Starting filesystem backup...")
            fs_result = await self.fs_manager.create_backup(backup_dir)
            
            # Check if all components succeeded
            all_success = all(
                result.get("status") == "success"
                for result in [db_result, redis_result, fs_result]
            )
            
            if all_success:
                # Create manifest
                manifest = {
                    "backup_id": backup_id,
                    "timestamp": datetime.now().isoformat(),
                    "type": "full",
                    "components": {
                        "database": db_result,
                        "redis": redis_result,
                        "filesystem": fs_result
                    }
                }
                
                manifest_file = f"{backup_dir}/manifest.json"
                with open(manifest_file, 'w') as f:
                    json.dump(manifest, f, indent=2, default=str)
                
                # Upload to S3
                s3_key = f"backups/{backup_id}"
                await self._upload_directory_to_s3(backup_dir, s3_key)
                
                record.status = BackupStatus.COMPLETED
                record.end_time = datetime.now()
                record.file_path = backup_dir
                record.metadata = manifest
                
            else:
                record.status = BackupStatus.PARTIAL
                record.error_message = "Some components failed"
            
        except Exception as e:
            logger.error(f"Full backup failed: {e}")
            record.status = BackupStatus.FAILED
            record.error_message = str(e)
        finally:
            record.end_time = datetime.now()
            await self._notify_backup_completion(record)
        
        return record
    
    async def restore_full_backup(self, backup_id: str) -> Dict[str, Any]:
        """Restore from a full backup"""
        try:
            logger.info(f"Starting restore from backup: {backup_id}")
            
            # Download from S3 if not available locally
            local_backup_dir = f"{self.config.local_backup_dir}/{backup_id}"
            if not os.path.exists(local_backup_dir):
                s3_key = f"backups/{backup_id}"
                await self._download_directory_from_s3(s3_key, local_backup_dir)
            
            # Read manifest
            manifest_file = f"{local_backup_dir}/manifest.json"
            if not os.path.exists(manifest_file):
                return {"status": "failed", "error": "Backup manifest not found"}
            
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            results = {}
            
            # Restore database
            if "database" in manifest["components"]:
                db_file = manifest["components"]["database"]["file_path"]
                results["database"] = await self.db_manager.restore_database(db_file)
            
            # Restore Redis
            if "redis" in manifest["components"]:
                redis_file = manifest["components"]["redis"]["file_path"]
                results["redis"] = await self.redis_manager.restore_redis(redis_file)
            
            # Restore filesystem
            if "filesystem" in manifest["components"]:
                fs_file = manifest["components"]["filesystem"]["file_path"]
                results["filesystem"] = await self.fs_manager.restore_filesystem(fs_file)
            
            # Check overall success
            all_success = all(
                result.get("status") == "success"
                for result in results.values()
            )
            
            return {
                "status": "success" if all_success else "partial",
                "components": results
            }
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def schedule_backups(self):
        """Schedule automated backups"""
        while True:
            try:
                now = datetime.now()
                
                # Check if full backup is needed
                last_full = await self._get_last_backup_time(BackupType.FULL)
                if (not last_full or 
                    now - last_full > timedelta(hours=self.config.full_backup_interval_hours)):
                    
                    logger.info("Starting scheduled full backup")
                    await self.create_full_backup()
                
                # Check if incremental backup is needed
                last_incremental = await self._get_last_backup_time(BackupType.INCREMENTAL)
                if (not last_incremental or
                    now - last_incremental > timedelta(hours=self.config.incremental_backup_interval_hours)):
                    
                    logger.info("Starting scheduled incremental backup")
                    # Note: Incremental backup implementation would go here
                
                # Cleanup old backups
                await self._cleanup_local_backups()
                await self.s3_storage.cleanup_old_backups()
                
            except Exception as e:
                logger.error(f"Backup scheduler error: {e}")
            
            # Sleep for 1 hour before next check
            await asyncio.sleep(3600)
    
    async def _upload_directory_to_s3(self, local_dir: str, s3_key: str):
        """Upload entire directory to S3"""
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, local_dir)
                s3_file_key = f"{s3_key}/{relative_path}"
                await self.s3_storage.upload_backup(local_file, s3_file_key)
    
    async def _download_directory_from_s3(self, s3_key: str, local_dir: str):
        """Download directory from S3"""
        # Implementation would list S3 objects and download each file
        pass
    
    async def _get_last_backup_time(self, backup_type: BackupType) -> Optional[datetime]:
        """Get timestamp of last backup of given type"""
        # Implementation would check backup records/database
        return None
    
    async def _cleanup_local_backups(self):
        """Clean up old local backups"""
        cutoff_date = datetime.now() - timedelta(days=self.config.local_retention_days)
        
        for backup_dir in Path(self.config.local_backup_dir).iterdir():
            if backup_dir.is_dir() and backup_dir.stat().st_mtime < cutoff_date.timestamp():
                shutil.rmtree(backup_dir, ignore_errors=True)
    
    async def _notify_backup_completion(self, record: BackupRecord):
        """Send notifications about backup completion"""
        if self.config.webhook_url:
            # Send webhook notification
            pass
        
        if self.config.slack_webhook:
            # Send Slack notification
            pass

# CLI Interface
async def main():
    """Command-line interface for backup operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YTEmpire Disaster Recovery Manager")
    parser.add_argument("action", choices=["backup", "restore", "schedule", "status"])
    parser.add_argument("--backup-id", help="Backup ID for restore operations")
    parser.add_argument("--type", choices=["full", "incremental"], default="full")
    
    args = parser.parse_args()
    
    config = BackupConfig()
    manager = DisasterRecoveryManager(config)
    
    if args.action == "backup":
        if args.type == "full":
            result = await manager.create_full_backup()
            print(f"Backup completed: {result.id} - Status: {result.status.value}")
        
    elif args.action == "restore":
        if not args.backup_id:
            print("Error: --backup-id required for restore")
            return
        
        result = await manager.restore_full_backup(args.backup_id)
        print(f"Restore completed: Status: {result['status']}")
        
    elif args.action == "schedule":
        print("Starting backup scheduler...")
        await manager.schedule_backups()
        
    elif args.action == "status":
        print("Backup system status:")
        print(f"Local backup directory: {config.local_backup_dir}")
        print(f"S3 bucket: {config.s3_bucket}")

if __name__ == "__main__":
    asyncio.run(main())