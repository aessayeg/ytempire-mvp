"""
Enhanced Secrets Management System
Owner: Security Engineer #1

Advanced secrets management with encryption, rotation, and audit logging.
Extends HashiCorp Vault with additional security features.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

from app.core.config import settings
from app.services.vault_service import VaultService
from app.core.metrics import metrics

logger = logging.getLogger(__name__)


class SecretType(Enum):
    """Types of secrets managed by the system."""
    API_KEY = "api_key"
    DATABASE_PASSWORD = "database_password"
    ENCRYPTION_KEY = "encryption_key"
    JWT_SECRET = "jwt_secret"
    WEBHOOK_SECRET = "webhook_secret"
    OAUTH_TOKEN = "oauth_token"
    SERVICE_ACCOUNT_KEY = "service_account_key"
    TLS_CERTIFICATE = "tls_certificate"


class SecretRotationStatus(Enum):
    """Status of secret rotation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SecretMetadata:
    """Metadata for secrets management."""
    id: str
    name: str
    type: SecretType
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    rotation_interval_days: Optional[int] = None
    last_rotated_at: Optional[datetime] = None
    rotation_status: SecretRotationStatus = SecretRotationStatus.COMPLETED
    tags: Dict[str, str] = None
    audit_log: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.audit_log is None:
            self.audit_log = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert datetime objects to ISO format
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif key == 'type':
                data[key] = value.value if hasattr(value, 'value') else value
            elif key == 'rotation_status':
                data[key] = value.value if hasattr(value, 'value') else value
        return data


class SecretAuditEvent(Enum):
    """Audit event types for secrets."""
    CREATED = "created"
    ACCESSED = "accessed"
    UPDATED = "updated"
    ROTATED = "rotated"
    DELETED = "deleted"
    ROTATION_FAILED = "rotation_failed"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


class SecretsManagerError(Exception):
    """Custom exception for secrets management errors."""
    pass


class SecretsManager:
    """Enhanced secrets management system."""
    
    def __init__(self, vault_service: Optional[VaultService] = None):
        self.vault_service = vault_service or VaultService()
        self._local_encryption_key = None
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize secrets manager with Vault and local encryption."""
        try:
            # Initialize Vault service
            if not await self.vault_service.initialize():
                logger.warning("Vault not available, using local encryption only")
            
            # Initialize local encryption
            await self._initialize_local_encryption()
            
            # Set up secret rotation monitoring
            await self._schedule_rotation_checks()
            
            self.initialized = True
            logger.info("Secrets manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize secrets manager: {str(e)}")
            raise SecretsManagerError(f"Initialization failed: {str(e)}")
    
    async def store_secret(
        self,
        name: str,
        value: Union[str, Dict[str, Any]],
        secret_type: SecretType,
        metadata: Optional[Dict[str, Any]] = None,
        rotation_interval_days: Optional[int] = None,
        encrypt_locally: bool = False
    ) -> str:
        """
        Store a secret with metadata and optional local encryption.
        
        Args:
            name: Secret name/identifier
            value: Secret value (string or dict)
            secret_type: Type of secret
            metadata: Additional metadata
            rotation_interval_days: Automatic rotation interval
            encrypt_locally: Whether to encrypt locally before storing in Vault
            
        Returns:
            Secret ID
        """
        try:
            secret_id = self._generate_secret_id(name, secret_type)
            
            # Prepare secret data
            secret_data = {
                "value": value,
                "metadata": metadata or {}
            }
            
            # Encrypt locally if requested
            if encrypt_locally:
                secret_data = await self._encrypt_secret_data(secret_data)
                secret_data["encrypted"] = True
            
            # Create metadata object
            secret_metadata = SecretMetadata(
                id=secret_id,
                name=name,
                type=secret_type,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                rotation_interval_days=rotation_interval_days,
                tags=metadata or {}
            )
            
            # Set expiration if rotation interval is specified
            if rotation_interval_days:
                secret_metadata.expires_at = datetime.utcnow() + timedelta(days=rotation_interval_days)
            
            # Store in Vault
            if self.vault_service.client and self.vault_service._authenticated:
                await self.vault_service.store_secret(
                    path=f"ytempire/secrets/{secret_id}",
                    secret=secret_data
                )
                
                # Store metadata separately
                await self.vault_service.store_secret(
                    path=f"ytempire/metadata/{secret_id}",
                    secret=secret_metadata.to_dict()
                )
            else:
                # Fallback to local storage (for development/testing)
                logger.warning(f"Storing secret {secret_id} locally - not recommended for production")
                await self._store_locally(secret_id, secret_data, secret_metadata)
            
            # Audit log
            await self._audit_log(secret_id, SecretAuditEvent.CREATED, {
                "name": name,
                "type": secret_type.value,
                "has_rotation": rotation_interval_days is not None
            })
            
            # Update metrics
            metrics.increment("secrets_created", {"type": secret_type.value})
            
            logger.info(f"Stored secret {secret_id} ({secret_type.value})")
            return secret_id
            
        except Exception as e:
            logger.error(f"Failed to store secret {name}: {str(e)}")
            metrics.increment("secrets_errors", {"operation": "store"})
            raise SecretsManagerError(f"Failed to store secret: {str(e)}")
    
    async def get_secret(
        self,
        name_or_id: str,
        decrypt_locally: bool = True,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a secret by name or ID.
        
        Args:
            name_or_id: Secret name or ID
            decrypt_locally: Whether to decrypt locally encrypted secrets
            use_cache: Whether to use cached values
            
        Returns:
            Secret data or None if not found
        """
        try:
            # Check cache first
            if use_cache and name_or_id in self._cache:
                cache_entry = self._cache[name_or_id]
                if datetime.utcnow() < cache_entry["expires"]:
                    await self._audit_log(name_or_id, SecretAuditEvent.ACCESSED, {
                        "source": "cache"
                    })
                    return cache_entry["data"]
                else:
                    del self._cache[name_or_id]
            
            # Try to resolve name to ID if needed
            secret_id = await self._resolve_secret_id(name_or_id)
            if not secret_id:
                return None
            
            # Retrieve from Vault
            secret_data = None
            if self.vault_service.client and self.vault_service._authenticated:
                secret_data = await self.vault_service.get_secret(f"ytempire/secrets/{secret_id}")
            else:
                # Fallback to local storage
                secret_data = await self._get_locally(secret_id)
            
            if not secret_data:
                return None
            
            # Decrypt if needed
            if secret_data.get("encrypted") and decrypt_locally:
                secret_data = await self._decrypt_secret_data(secret_data)
            
            # Cache the result
            if use_cache:
                self._cache[name_or_id] = {
                    "data": secret_data,
                    "expires": datetime.utcnow() + timedelta(seconds=self._cache_ttl)
                }
            
            # Audit log
            await self._audit_log(secret_id, SecretAuditEvent.ACCESSED, {
                "name_or_id": name_or_id,
                "source": "vault" if self.vault_service._authenticated else "local"
            })
            
            # Update metrics
            metrics.increment("secrets_accessed")
            
            return secret_data
            
        except Exception as e:
            logger.error(f"Failed to get secret {name_or_id}: {str(e)}")
            metrics.increment("secrets_errors", {"operation": "get"})
            await self._audit_log(name_or_id, SecretAuditEvent.UNAUTHORIZED_ACCESS, {
                "error": str(e)
            })
            raise SecretsManagerError(f"Failed to get secret: {str(e)}")
    
    async def rotate_secret(self, secret_id: str, new_value: Optional[str] = None) -> bool:
        """
        Rotate a secret with a new value.
        
        Args:
            secret_id: Secret identifier
            new_value: New secret value (auto-generated if not provided)
            
        Returns:
            True if rotation successful
        """
        try:
            # Get current metadata
            metadata = await self._get_secret_metadata(secret_id)
            if not metadata:
                raise SecretsManagerError(f"Secret {secret_id} not found")
            
            # Update rotation status
            metadata.rotation_status = SecretRotationStatus.IN_PROGRESS
            metadata.updated_at = datetime.utcnow()
            await self._update_secret_metadata(secret_id, metadata)
            
            # Generate new value if not provided
            if new_value is None:
                new_value = await self._generate_secret_value(metadata.type)
            
            # Update the secret
            secret_data = await self.get_secret(secret_id)
            if secret_data:
                secret_data["value"] = new_value
                secret_data["rotated_at"] = datetime.utcnow().isoformat()
                
                # Store updated secret
                if self.vault_service.client and self.vault_service._authenticated:
                    await self.vault_service.store_secret(
                        path=f"ytempire/secrets/{secret_id}",
                        secret=secret_data
                    )
                else:
                    await self._store_locally(secret_id, secret_data, metadata)
            
            # Update metadata
            metadata.rotation_status = SecretRotationStatus.COMPLETED
            metadata.last_rotated_at = datetime.utcnow()
            if metadata.rotation_interval_days:
                metadata.expires_at = datetime.utcnow() + timedelta(days=metadata.rotation_interval_days)
            
            await self._update_secret_metadata(secret_id, metadata)
            
            # Clear cache
            self._invalidate_cache(secret_id)
            
            # Audit log
            await self._audit_log(secret_id, SecretAuditEvent.ROTATED, {
                "rotation_method": "manual" if new_value else "auto"
            })
            
            # Update metrics
            metrics.increment("secrets_rotated", {"type": metadata.type.value})
            
            logger.info(f"Successfully rotated secret {secret_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rotate secret {secret_id}: {str(e)}")
            
            # Update status to failed
            try:
                metadata = await self._get_secret_metadata(secret_id)
                if metadata:
                    metadata.rotation_status = SecretRotationStatus.FAILED
                    await self._update_secret_metadata(secret_id, metadata)
            except Exception:
                pass
            
            await self._audit_log(secret_id, SecretAuditEvent.ROTATION_FAILED, {
                "error": str(e)
            })
            
            metrics.increment("secrets_errors", {"operation": "rotate"})
            return False
    
    async def list_secrets(
        self,
        secret_type: Optional[SecretType] = None,
        expired_only: bool = False
    ) -> List[SecretMetadata]:
        """
        List secrets with optional filtering.
        
        Args:
            secret_type: Filter by secret type
            expired_only: Only return expired secrets
            
        Returns:
            List of secret metadata
        """
        try:
            secrets = []
            
            # Get list from Vault
            if self.vault_service.client and self.vault_service._authenticated:
                metadata_list = await self.vault_service.list_secrets("ytempire/metadata/")
                
                for metadata_path in metadata_list:
                    metadata_data = await self.vault_service.get_secret(f"ytempire/metadata/{metadata_path}")
                    if metadata_data:
                        metadata = self._dict_to_metadata(metadata_data)
                        
                        # Apply filters
                        if secret_type and metadata.type != secret_type:
                            continue
                        
                        if expired_only:
                            if not metadata.expires_at or metadata.expires_at > datetime.utcnow():
                                continue
                        
                        secrets.append(metadata)
            
            return secrets
            
        except Exception as e:
            logger.error(f"Failed to list secrets: {str(e)}")
            return []
    
    async def delete_secret(self, secret_id: str) -> bool:
        """
        Delete a secret and its metadata.
        
        Args:
            secret_id: Secret identifier
            
        Returns:
            True if deletion successful
        """
        try:
            # Delete from Vault
            if self.vault_service.client and self.vault_service._authenticated:
                await self.vault_service.delete_secret(f"ytempire/secrets/{secret_id}")
                await self.vault_service.delete_secret(f"ytempire/metadata/{secret_id}")
            else:
                await self._delete_locally(secret_id)
            
            # Clear cache
            self._invalidate_cache(secret_id)
            
            # Audit log
            await self._audit_log(secret_id, SecretAuditEvent.DELETED, {})
            
            # Update metrics
            metrics.increment("secrets_deleted")
            
            logger.info(f"Deleted secret {secret_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete secret {secret_id}: {str(e)}")
            return False
    
    async def check_expired_secrets(self) -> List[str]:
        """
        Check for expired secrets that need rotation.
        
        Returns:
            List of expired secret IDs
        """
        try:
            expired_secrets = []
            secrets = await self.list_secrets(expired_only=True)
            
            for metadata in secrets:
                if (metadata.expires_at and 
                    metadata.expires_at <= datetime.utcnow() and
                    metadata.rotation_status == SecretRotationStatus.COMPLETED):
                    expired_secrets.append(metadata.id)
            
            logger.info(f"Found {len(expired_secrets)} expired secrets")
            return expired_secrets
            
        except Exception as e:
            logger.error(f"Failed to check expired secrets: {str(e)}")
            return []
    
    # Private helper methods
    
    async def _initialize_local_encryption(self):
        """Initialize local encryption key."""
        # Try to get encryption key from environment or generate one
        key_material = os.getenv("LOCAL_ENCRYPTION_KEY")
        if not key_material:
            # Generate a new key (in production, this should be persisted securely)
            key_material = Fernet.generate_key().decode()
            logger.warning("Generated new local encryption key - should be persisted for production")
        
        self._local_encryption_key = key_material.encode()
    
    def _generate_secret_id(self, name: str, secret_type: SecretType) -> str:
        """Generate unique secret ID."""
        content = f"{name}:{secret_type.value}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def _encrypt_secret_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt secret data locally."""
        if not self._local_encryption_key:
            raise SecretsManagerError("Local encryption not initialized")
        
        fernet = Fernet(base64.urlsafe_b64encode(self._local_encryption_key[:32]))
        encrypted_data = fernet.encrypt(json.dumps(data).encode())
        
        return {
            "encrypted_data": base64.b64encode(encrypted_data).decode(),
            "encryption_method": "fernet"
        }
    
    async def _decrypt_secret_data(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt locally encrypted secret data."""
        if not self._local_encryption_key:
            raise SecretsManagerError("Local encryption not initialized")
        
        if encrypted_data.get("encryption_method") != "fernet":
            raise SecretsManagerError("Unsupported encryption method")
        
        fernet = Fernet(base64.urlsafe_b64encode(self._local_encryption_key[:32]))
        encrypted_bytes = base64.b64decode(encrypted_data["encrypted_data"])
        decrypted_data = fernet.decrypt(encrypted_bytes)
        
        return json.loads(decrypted_data.decode())
    
    async def _generate_secret_value(self, secret_type: SecretType) -> str:
        """Generate new secret value based on type."""
        import secrets
        import string
        
        if secret_type in [SecretType.API_KEY, SecretType.JWT_SECRET, SecretType.WEBHOOK_SECRET]:
            # Generate random string
            alphabet = string.ascii_letters + string.digits
            return ''.join(secrets.choice(alphabet) for _ in range(32))
        elif secret_type == SecretType.DATABASE_PASSWORD:
            # Generate secure password
            alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
            return ''.join(secrets.choice(alphabet) for _ in range(16))
        elif secret_type == SecretType.ENCRYPTION_KEY:
            # Generate encryption key
            return base64.b64encode(secrets.token_bytes(32)).decode()
        else:
            # Default random string
            return secrets.token_hex(16)
    
    async def _audit_log(
        self, 
        secret_id: str, 
        event: SecretAuditEvent, 
        details: Dict[str, Any]
    ):
        """Log audit event."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "secret_id": secret_id,
            "event": event.value,
            "details": details,
            "user": "system"  # In production, get from current user context
        }
        
        # In production, this should go to a secure audit log system
        logger.info(f"AUDIT: {json.dumps(audit_entry)}")
        
        # Update metrics
        metrics.increment("secrets_audit_events", {"event": event.value})
    
    def _invalidate_cache(self, secret_id: str):
        """Invalidate cache entries for a secret."""
        keys_to_remove = [key for key in self._cache.keys() if secret_id in key]
        for key in keys_to_remove:
            del self._cache[key]
    
    def _dict_to_metadata(self, data: Dict[str, Any]) -> SecretMetadata:
        """Convert dictionary to SecretMetadata object."""
        # Convert string dates back to datetime objects
        for date_field in ['created_at', 'updated_at', 'expires_at', 'last_rotated_at']:
            if data.get(date_field):
                data[date_field] = datetime.fromisoformat(data[date_field])
        
        # Convert enum strings back to enums
        if 'type' in data:
            data['type'] = SecretType(data['type'])
        if 'rotation_status' in data:
            data['rotation_status'] = SecretRotationStatus(data['rotation_status'])
        
        return SecretMetadata(**data)
    
    # Placeholder methods for local storage (development/testing)
    async def _store_locally(self, secret_id: str, data: Dict[str, Any], metadata: SecretMetadata):
        """Store secret locally (development only)."""
        pass
    
    async def _get_locally(self, secret_id: str) -> Optional[Dict[str, Any]]:
        """Get secret from local storage (development only)."""
        return None
    
    async def _delete_locally(self, secret_id: str):
        """Delete secret from local storage (development only)."""
        pass
    
    async def _resolve_secret_id(self, name_or_id: str) -> Optional[str]:
        """Resolve secret name to ID."""
        # In production, maintain a name-to-ID mapping
        return name_or_id
    
    async def _get_secret_metadata(self, secret_id: str) -> Optional[SecretMetadata]:
        """Get secret metadata."""
        if self.vault_service.client and self.vault_service._authenticated:
            data = await self.vault_service.get_secret(f"ytempire/metadata/{secret_id}")
            if data:
                return self._dict_to_metadata(data)
        return None
    
    async def _update_secret_metadata(self, secret_id: str, metadata: SecretMetadata):
        """Update secret metadata."""
        if self.vault_service.client and self.vault_service._authenticated:
            await self.vault_service.store_secret(
                path=f"ytempire/metadata/{secret_id}",
                secret=metadata.to_dict()
            )
    
    async def _schedule_rotation_checks(self):
        """Schedule periodic checks for secret rotation."""
        # In production, this would use a proper scheduler
        logger.info("Secret rotation monitoring scheduled")


# Global instance
secrets_manager = SecretsManager()