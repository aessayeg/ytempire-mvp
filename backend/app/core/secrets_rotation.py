"""
Secrets rotation system for YTEmpire.
Automatically rotates API keys, passwords, and other sensitive credentials.
"""

import os
import secrets
import hashlib
import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import redis.asyncio as redis
from pydantic import BaseModel, Field
import httpx

from app.core.audit_logging import audit_logger, AuditEventType, AuditSeverity

logger = logging.getLogger(__name__)

# ============================================================================
# Secret Types and Configuration
# ============================================================================


class SecretType(str, Enum):
    """Types of secrets that can be rotated"""

    API_KEY = "api_key"
    DATABASE_PASSWORD = "database_password"
    JWT_SECRET = "jwt_secret"
    ENCRYPTION_KEY = "encryption_key"
    OAUTH_SECRET = "oauth_secret"
    WEBHOOK_SECRET = "webhook_secret"
    SERVICE_ACCOUNT_KEY = "service_account_key"
    TLS_CERTIFICATE = "tls_certificate"
    SSH_KEY = "ssh_key"
    SMTP_PASSWORD = "smtp_password"


class RotationPolicy(BaseModel):
    """Secret rotation policy"""

    secret_type: SecretType
    rotation_interval_days: int = 90
    grace_period_hours: int = 24
    max_versions: int = 3
    auto_rotate: bool = True
    notify_before_days: int = 7
    complexity_requirements: Dict[str, Any] = Field(default_factory=dict)


class SecretMetadata(BaseModel):
    """Metadata for a secret"""

    secret_id: str
    secret_type: SecretType
    version: int
    created_at: datetime
    expires_at: datetime
    rotated_at: Optional[datetime] = None
    rotated_by: Optional[str] = None
    status: str = "active"  # active, rotating, expired, revoked
    checksum: str
    tags: Dict[str, str] = Field(default_factory=dict)


# ============================================================================
# Secret Generation
# ============================================================================


class SecretGenerator:
    """Generate secure secrets based on type and requirements"""

    @staticmethod
    def generate_api_key(length: int = 32) -> str:
        """Generate a secure API key"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def generate_password(
        length: int = 24,
        include_uppercase: bool = True,
        include_lowercase: bool = True,
        include_digits: bool = True,
        include_special: bool = True,
    ) -> str:
        """Generate a secure password"""
        import string

        characters = ""
        if include_lowercase:
            characters += string.ascii_lowercase
        if include_uppercase:
            characters += string.ascii_uppercase
        if include_digits:
            characters += string.digits
        if include_special:
            characters += string.punctuation

        if not characters:
            characters = string.ascii_letters + string.digits

        return "".join(secrets.choice(characters) for _ in range(length))

    @staticmethod
    def generate_jwt_secret(length: int = 64) -> str:
        """Generate a JWT secret"""
        return secrets.token_hex(length)

    @staticmethod
    def generate_encryption_key() -> bytes:
        """Generate an encryption key"""
        return Fernet.generate_key()

    @staticmethod
    def generate_oauth_secret(length: int = 48) -> str:
        """Generate an OAuth client secret"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def generate_webhook_secret(length: int = 32) -> str:
        """Generate a webhook signing secret"""
        return secrets.token_hex(length)

    @staticmethod
    def generate_ssh_key() -> Tuple[str, str]:
        """Generate SSH key pair"""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        # Generate private key
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)

        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption(),
        )

        # Get public key
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH,
        )

        return private_pem.decode(), public_pem.decode()


# ============================================================================
# Secret Storage
# ============================================================================


class SecretVault:
    """Secure storage for secrets"""

    def __init__(
        self,
        master_key: Optional[str] = None,
        redis_client: Optional[redis.Redis] = None,
        db_session: Optional[AsyncSession] = None,
    ):
        self.redis_client = redis_client
        self.db_session = db_session

        # Initialize encryption
        if master_key:
            self.fernet = Fernet(
                master_key.encode() if isinstance(master_key, str) else master_key
            )
        else:
            # Generate from environment or use default (NOT for production!)
            key = os.getenv("MASTER_ENCRYPTION_KEY", Fernet.generate_key().decode())
            self.fernet = Fernet(key.encode() if isinstance(key, str) else key)

    async def store_secret(
        self,
        secret_id: str,
        secret_value: str,
        secret_type: SecretType,
        metadata: Optional[SecretMetadata] = None,
    ) -> bool:
        """Store a secret securely"""
        try:
            # Encrypt the secret
            encrypted_value = self.fernet.encrypt(secret_value.encode())

            # Generate metadata if not provided
            if not metadata:
                metadata = SecretMetadata(
                    secret_id=secret_id,
                    secret_type=secret_type,
                    version=1,
                    created_at=datetime.now(timezone.utc),
                    expires_at=datetime.now(timezone.utc) + timedelta(days=90),
                    checksum=hashlib.sha256(secret_value.encode()).hexdigest(),
                )

            # Store in Redis with TTL
            if self.redis_client:
                ttl = int(
                    (metadata.expires_at - datetime.now(timezone.utc)).total_seconds()
                )
                await self.redis_client.setex(
                    f"secret:{secret_id}:value", ttl, encrypted_value
                )
                await self.redis_client.setex(
                    f"secret:{secret_id}:metadata", ttl, metadata.json()
                )

            # Store in database for persistence
            if self.db_session:
                await self.db_session.execute(
                    text(
                        """
                        INSERT INTO secrets (
                            secret_id, secret_type, encrypted_value,
                            version, created_at, expires_at, metadata
                        ) VALUES (
                            :secret_id, :secret_type, :encrypted_value,
                            :version, :created_at, :expires_at, :metadata
                        )
                        ON CONFLICT (secret_id) DO UPDATE SET
                            encrypted_value = :encrypted_value,
                            version = secrets.version + 1,
                            updated_at = NOW()
                    """
                    ),
                    {
                        "secret_id": secret_id,
                        "secret_type": secret_type.value,
                        "encrypted_value": encrypted_value.decode(),
                        "version": metadata.version,
                        "created_at": metadata.created_at,
                        "expires_at": metadata.expires_at,
                        "metadata": metadata.json(),
                    },
                )
                await self.db_session.commit()

            logger.info(f"Secret {secret_id} stored successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to store secret {secret_id}: {e}")
            return False

    async def retrieve_secret(self, secret_id: str) -> Optional[str]:
        """Retrieve and decrypt a secret"""
        try:
            # Try Redis first
            if self.redis_client:
                encrypted_value = await self.redis_client.get(
                    f"secret:{secret_id}:value"
                )
                if encrypted_value:
                    return self.fernet.decrypt(encrypted_value).decode()

            # Fallback to database
            if self.db_session:
                result = await self.db_session.execute(
                    text(
                        """
                        SELECT encrypted_value FROM secrets
                        WHERE secret_id = :secret_id
                        AND expires_at > NOW()
                        AND status = 'active'
                    """
                    ),
                    {"secret_id": secret_id},
                )
                row = result.first()
                if row:
                    return self.fernet.decrypt(row.encrypted_value.encode()).decode()

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_id}: {e}")
            return None

    async def delete_secret(self, secret_id: str) -> bool:
        """Delete a secret"""
        try:
            # Delete from Redis
            if self.redis_client:
                await self.redis_client.delete(f"secret:{secret_id}:value")
                await self.redis_client.delete(f"secret:{secret_id}:metadata")

            # Mark as deleted in database (soft delete)
            if self.db_session:
                await self.db_session.execute(
                    text(
                        """
                        UPDATE secrets
                        SET status = 'deleted', deleted_at = NOW()
                        WHERE secret_id = :secret_id
                    """
                    ),
                    {"secret_id": secret_id},
                )
                await self.db_session.commit()

            logger.info(f"Secret {secret_id} deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to delete secret {secret_id}: {e}")
            return False


# ============================================================================
# Secrets Rotation Manager
# ============================================================================


class SecretsRotationManager:
    """Manages automatic rotation of secrets"""

    def __init__(
        self,
        vault: SecretVault,
        redis_client: Optional[redis.Redis] = None,
        db_session: Optional[AsyncSession] = None,
    ):
        self.vault = vault
        self.redis_client = redis_client
        self.db_session = db_session
        self.generator = SecretGenerator()
        self.rotation_policies: Dict[
            SecretType, RotationPolicy
        ] = self._load_default_policies()
        self._rotation_tasks: Dict[str, asyncio.Task] = {}

    def _load_default_policies(self) -> Dict[SecretType, RotationPolicy]:
        """Load default rotation policies"""
        return {
            SecretType.API_KEY: RotationPolicy(
                secret_type=SecretType.API_KEY,
                rotation_interval_days=90,
                grace_period_hours=48,
            ),
            SecretType.DATABASE_PASSWORD: RotationPolicy(
                secret_type=SecretType.DATABASE_PASSWORD,
                rotation_interval_days=60,
                grace_period_hours=24,
            ),
            SecretType.JWT_SECRET: RotationPolicy(
                secret_type=SecretType.JWT_SECRET,
                rotation_interval_days=30,
                grace_period_hours=12,
            ),
            SecretType.ENCRYPTION_KEY: RotationPolicy(
                secret_type=SecretType.ENCRYPTION_KEY,
                rotation_interval_days=180,
                grace_period_hours=72,
            ),
            SecretType.OAUTH_SECRET: RotationPolicy(
                secret_type=SecretType.OAUTH_SECRET,
                rotation_interval_days=365,
                grace_period_hours=168,  # 1 week
            ),
            SecretType.WEBHOOK_SECRET: RotationPolicy(
                secret_type=SecretType.WEBHOOK_SECRET,
                rotation_interval_days=90,
                grace_period_hours=24,
            ),
            SecretType.TLS_CERTIFICATE: RotationPolicy(
                secret_type=SecretType.TLS_CERTIFICATE,
                rotation_interval_days=365,
                grace_period_hours=720,  # 30 days
                notify_before_days=30,
            ),
            SecretType.SSH_KEY: RotationPolicy(
                secret_type=SecretType.SSH_KEY,
                rotation_interval_days=180,
                grace_period_hours=48,
            ),
        }

    async def rotate_secret(
        self, secret_id: str, secret_type: SecretType, user_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """Rotate a specific secret"""
        try:
            logger.info(f"Starting rotation for secret {secret_id}")

            # Generate new secret based on type
            new_secret = await self._generate_new_secret(secret_type)

            # Get old secret for rollback if needed
            old_secret = await self.vault.retrieve_secret(secret_id)

            # Create metadata
            metadata = SecretMetadata(
                secret_id=secret_id,
                secret_type=secret_type,
                version=await self._get_next_version(secret_id),
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc)
                + timedelta(
                    days=self.rotation_policies[secret_type].rotation_interval_days
                ),
                rotated_at=datetime.now(timezone.utc),
                rotated_by=user_id or "system",
                checksum=hashlib.sha256(new_secret.encode()).hexdigest(),
            )

            # Store new secret
            success = await self.vault.store_secret(
                secret_id=secret_id,
                secret_value=new_secret,
                secret_type=secret_type,
                metadata=metadata,
            )

            if success:
                # Archive old secret
                if old_secret:
                    await self._archive_old_secret(
                        secret_id, old_secret, metadata.version - 1
                    )

                # Update dependent services
                await self._update_dependent_services(
                    secret_id, secret_type, new_secret
                )

                # Log audit event
                await audit_logger.log_event(
                    event_type=AuditEventType.CONFIG_CHANGE,
                    action=f"Secret rotated: {secret_id}",
                    result="success",
                    severity=AuditSeverity.MEDIUM,
                    user_id=user_id,
                    metadata={
                        "secret_type": secret_type.value,
                        "version": metadata.version,
                    },
                )

                logger.info(f"Secret {secret_id} rotated successfully")
                return True, new_secret

            return False, None

        except Exception as e:
            logger.error(f"Failed to rotate secret {secret_id}: {e}")

            # Log audit event
            await audit_logger.log_event(
                event_type=AuditEventType.CONFIG_CHANGE,
                action=f"Secret rotation failed: {secret_id}",
                result="failure",
                severity=AuditSeverity.HIGH,
                user_id=user_id,
                reason=str(e),
            )

            return False, None

    async def _generate_new_secret(self, secret_type: SecretType) -> str:
        """Generate new secret based on type"""
        if secret_type == SecretType.API_KEY:
            return self.generator.generate_api_key()
        elif secret_type == SecretType.DATABASE_PASSWORD:
            return self.generator.generate_password()
        elif secret_type == SecretType.JWT_SECRET:
            return self.generator.generate_jwt_secret()
        elif secret_type == SecretType.ENCRYPTION_KEY:
            return self.generator.generate_encryption_key().decode()
        elif secret_type == SecretType.OAUTH_SECRET:
            return self.generator.generate_oauth_secret()
        elif secret_type == SecretType.WEBHOOK_SECRET:
            return self.generator.generate_webhook_secret()
        elif secret_type == SecretType.SSH_KEY:
            private, public = self.generator.generate_ssh_key()
            return json.dumps({"private": private, "public": public})
        else:
            return self.generator.generate_api_key()

    async def _get_next_version(self, secret_id: str) -> int:
        """Get next version number for secret"""
        if self.db_session:
            result = await self.db_session.execute(
                text(
                    """
                    SELECT MAX(version) as max_version
                    FROM secrets
                    WHERE secret_id = :secret_id
                """
                ),
                {"secret_id": secret_id},
            )
            row = result.first()
            if row and row.max_version:
                return row.max_version + 1
        return 1

    async def _archive_old_secret(self, secret_id: str, old_secret: str, version: int):
        """Archive old version of secret"""
        try:
            if self.db_session:
                await self.db_session.execute(
                    text(
                        """
                        INSERT INTO secret_archives (
                            secret_id, version, encrypted_value, archived_at
                        ) VALUES (
                            :secret_id, :version, :encrypted_value, NOW()
                        )
                    """
                    ),
                    {
                        "secret_id": secret_id,
                        "version": version,
                        "encrypted_value": self.vault.fernet.encrypt(
                            old_secret.encode()
                        ).decode(),
                    },
                )
                await self.db_session.commit()
        except Exception as e:
            logger.error(f"Failed to archive old secret: {e}")

    async def _update_dependent_services(
        self, secret_id: str, secret_type: SecretType, new_secret: str
    ):
        """Update dependent services with new secret"""
        # This would integrate with your service discovery and configuration management
        # For example, updating environment variables, Kubernetes secrets, etc.

        if secret_type == SecretType.DATABASE_PASSWORD:
            # Update database connection pools
            pass
        elif secret_type == SecretType.API_KEY:
            # Update API clients
            pass
        elif secret_type == SecretType.JWT_SECRET:
            # Trigger token refresh for active sessions
            pass

        logger.info(f"Updated dependent services for {secret_id}")

    async def check_expiring_secrets(self) -> List[Dict]:
        """Check for secrets that are about to expire"""
        expiring = []

        if self.db_session:
            for secret_type, policy in self.rotation_policies.items():
                notify_date = datetime.now(timezone.utc) + timedelta(
                    days=policy.notify_before_days
                )

                result = await self.db_session.execute(
                    text(
                        """
                        SELECT secret_id, expires_at, version
                        FROM secrets
                        WHERE secret_type = :secret_type
                        AND status = 'active'
                        AND expires_at <= :notify_date
                    """
                    ),
                    {"secret_type": secret_type.value, "notify_date": notify_date},
                )

                for row in result:
                    expiring.append(
                        {
                            "secret_id": row.secret_id,
                            "secret_type": secret_type.value,
                            "expires_at": row.expires_at.isoformat(),
                            "days_until_expiry": (
                                row.expires_at - datetime.now(timezone.utc)
                            ).days,
                        }
                    )

        return expiring

    async def start_auto_rotation(self):
        """Start automatic rotation scheduler"""
        logger.info("Starting automatic secrets rotation")

        # Schedule rotation checks
        task = asyncio.create_task(self._rotation_scheduler())
        self._rotation_tasks["scheduler"] = task

    async def stop_auto_rotation(self):
        """Stop automatic rotation"""
        logger.info("Stopping automatic secrets rotation")

        for task in self._rotation_tasks.values():
            task.cancel()

        self._rotation_tasks.clear()

    async def _rotation_scheduler(self):
        """Background task for automatic rotation"""
        while True:
            try:
                # Check for expiring secrets
                expiring = await self.check_expiring_secrets()

                for secret in expiring:
                    if secret["days_until_expiry"] <= 0:
                        # Rotate immediately
                        await self.rotate_secret(
                            secret["secret_id"], SecretType(secret["secret_type"])
                        )
                    elif secret["days_until_expiry"] <= 7:
                        # Send notification
                        logger.warning(
                            f"Secret {secret['secret_id']} expires in {secret['days_until_expiry']} days"
                        )

                # Sleep for 1 hour
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rotation scheduler: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute


# ============================================================================
# Global Instance
# ============================================================================

# Initialize with environment configuration
vault = SecretVault()
rotation_manager = SecretsRotationManager(vault)
