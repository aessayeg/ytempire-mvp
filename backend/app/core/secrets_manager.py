"""
Secrets Management Configuration
Secure handling of sensitive configuration and API keys
"""
import os
import json
import base64
from typing import Any, Dict, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SecretsManager:
    """
    Centralized secrets management with encryption
    In production, integrate with HashiCorp Vault, AWS Secrets Manager, or Azure Key Vault
    """

    def __init__(self, master_key: Optional[str] = None):
        """Initialize secrets manager with master key"""
        self.master_key = master_key or os.environ.get("MASTER_KEY", "")

        if not self.master_key:
            # Generate a master key if not provided (for development only)
            logger.warning("No master key provided, generating one (DEVELOPMENT ONLY)")
            self.master_key = Fernet.generate_key().decode()
            logger.info(f"Generated master key: {self.master_key}")

        # Initialize Fernet cipher
        self._init_cipher()

        # Secrets storage (in production, use external service)
        self.secrets_file = Path("secrets.enc")
        self.secrets_cache = {}

        # Load existing secrets
        self._load_secrets()

    def _init_cipher(self):
        """Initialize encryption cipher"""
        try:
            # Derive key from master key
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"ytempire_salt_v1",  # In production, use random salt
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
            self.cipher = Fernet(key)
        except Exception as e:
            logger.error(f"Failed to initialize cipher: {e}")
            raise

    def _load_secrets(self):
        """Load encrypted secrets from file"""
        if self.secrets_file.exists():
            try:
                with open(self.secrets_file, "rb") as f:
                    encrypted_data = f.read()
                    decrypted_data = self.cipher.decrypt(encrypted_data)
                    self.secrets_cache = json.loads(decrypted_data.decode())
                logger.info("Secrets loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load secrets: {e}")
                self.secrets_cache = {}
        else:
            self.secrets_cache = self._get_default_secrets()

    def _save_secrets(self):
        """Save encrypted secrets to file"""
        try:
            data = json.dumps(self.secrets_cache).encode()
            encrypted_data = self.cipher.encrypt(data)
            with open(self.secrets_file, "wb") as f:
                f.write(encrypted_data)
            logger.info("Secrets saved successfully")
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")

    def _get_default_secrets(self) -> Dict[str, Any]:
        """Get default secrets configuration"""
        return {
            "api_keys": {
                "openai": os.environ.get("OPENAI_API_KEY", ""),
                "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
                "elevenlabs": os.environ.get("ELEVENLABS_API_KEY", ""),
                "youtube": os.environ.get("YOUTUBE_API_KEY", ""),
                "stripe": os.environ.get("STRIPE_SECRET_KEY", ""),
                "sendgrid": os.environ.get("SENDGRID_API_KEY", ""),
            },
            "database": {
                "password": os.environ.get("POSTGRES_PASSWORD", "admin"),
                "connection_string": os.environ.get("DATABASE_URL", ""),
            },
            "jwt": {
                "secret_key": os.environ.get(
                    "JWT_SECRET_KEY", Fernet.generate_key().decode()
                ),
                "refresh_secret": os.environ.get(
                    "JWT_REFRESH_SECRET", Fernet.generate_key().decode()
                ),
            },
            "encryption": {
                "data_key": Fernet.generate_key().decode(),
                "file_key": Fernet.generate_key().decode(),
            },
            "oauth": {
                "google_client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
                "google_client_secret": os.environ.get("GOOGLE_CLIENT_SECRET", ""),
                "youtube_client_id": os.environ.get("YOUTUBE_CLIENT_ID", ""),
                "youtube_client_secret": os.environ.get("YOUTUBE_CLIENT_SECRET", ""),
            },
            "webhooks": {
                "stripe_webhook_secret": os.environ.get("STRIPE_WEBHOOK_SECRET", ""),
                "github_webhook_secret": os.environ.get("GITHUB_WEBHOOK_SECRET", ""),
            },
        }

    def get_secret(self, key_path: str, default: Any = None) -> Any:
        """
        Get a secret value by path (e.g., 'api_keys.openai')
        """
        try:
            keys = key_path.split(".")
            value = self.secrets_cache

            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    return default

            return value if value is not None else default
        except Exception as e:
            logger.error(f"Failed to get secret {key_path}: {e}")
            return default

    def set_secret(self, key_path: str, value: Any):
        """
        Set a secret value by path
        """
        try:
            keys = key_path.split(".")
            current = self.secrets_cache

            # Navigate to the parent
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the value
            current[keys[-1]] = value

            # Save to file
            self._save_secrets()

            logger.info(f"Secret {key_path} updated")
        except Exception as e:
            logger.error(f"Failed to set secret {key_path}: {e}")
            raise

    def rotate_secret(self, key_path: str) -> str:
        """
        Rotate a secret (generate new value)
        """
        try:
            new_value = Fernet.generate_key().decode()
            self.set_secret(key_path, new_value)
            logger.info(f"Secret {key_path} rotated")
            return new_value
        except Exception as e:
            logger.error(f"Failed to rotate secret {key_path}: {e}")
            raise

    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a service"""
        return self.get_secret(f"api_keys.{service}")

    def get_database_password(self) -> str:
        """Get database password"""
        return self.get_secret("database.password", "admin")

    def get_jwt_secret(self) -> str:
        """Get JWT secret key"""
        return self.get_secret("jwt.secret_key", "default-jwt-secret")

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            key = self.get_secret("encryption.data_key")
            if not key:
                key = self.rotate_secret("encryption.data_key")

            cipher = Fernet(key.encode() if isinstance(key, str) else key)
            encrypted = cipher.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            key = self.get_secret("encryption.data_key")
            if not key:
                raise ValueError("Encryption key not found")

            cipher = Fernet(key.encode() if isinstance(key, str) else key)
            decrypted = cipher.decrypt(base64.b64decode(encrypted_data))
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise

    def validate_secrets(self) -> Dict[str, bool]:
        """Validate that all required secrets are configured"""
        validation_results = {}

        required_secrets = [
            "api_keys.openai",
            "api_keys.youtube",
            "database.password",
            "jwt.secret_key",
        ]

        for secret_path in required_secrets:
            value = self.get_secret(secret_path)
            validation_results[secret_path] = bool(value and value != "")

        return validation_results

    def export_public_config(self) -> Dict[str, Any]:
        """Export non-sensitive configuration"""
        return {
            "environment": os.environ.get("ENVIRONMENT", "development"),
            "api_version": "v1",
            "features": {
                "video_generation": True,
                "multi_account": True,
                "analytics": True,
                "payments": True,
            },
            "limits": {
                "max_upload_size": 100 * 1024 * 1024,  # 100MB
                "max_video_duration": 600,  # 10 minutes
                "max_channels_per_user": 5,
            },
        }


# Global instance
secrets_manager = SecretsManager()


# Environment variable wrapper for backward compatibility
class SecureConfig:
    """Secure configuration with secrets management"""

    @property
    def OPENAI_API_KEY(self) -> str:
        return secrets_manager.get_api_key("openai") or ""

    @property
    def DATABASE_PASSWORD(self) -> str:
        return secrets_manager.get_database_password()

    @property
    def JWT_SECRET(self) -> str:
        return secrets_manager.get_jwt_secret()

    @property
    def STRIPE_API_KEY(self) -> str:
        return secrets_manager.get_api_key("stripe") or ""

    def get_secret(self, name: str) -> Any:
        """Get any secret by name"""
        return secrets_manager.get_secret(name)

    def validate(self) -> bool:
        """Validate all secrets are configured"""
        results = secrets_manager.validate_secrets()
        all_valid = all(results.values())

        if not all_valid:
            missing = [k for k, v in results.items() if not v]
            logger.warning(f"Missing secrets: {missing}")

        return all_valid


# Global secure config instance
secure_config = SecureConfig()
