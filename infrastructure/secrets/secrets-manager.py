#!/usr/bin/env python3
"""
YTEmpire Secrets Management System
Handles secure storage and retrieval of sensitive configuration
"""
import os
import json
import base64
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import boto3
from botocore.exceptions import ClientError
import hvac
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecretsManager:
    """Unified secrets management interface"""
    
    def __init__(self, backend: str = "local"):
        """
        Initialize secrets manager
        
        Args:
            backend: Storage backend ('local', 'aws', 'vault')
        """
        self.backend = backend
        self.encryption_key = self._get_or_create_encryption_key()
        
        if backend == "aws":
            self.aws_client = boto3.client('secretsmanager', region_name='us-east-1')
        elif backend == "vault":
            self.vault_client = hvac.Client(
                url=os.getenv('VAULT_URL', 'http://localhost:8200'),
                token=os.getenv('VAULT_TOKEN')
            )
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create local encryption key"""
        key_file = '.encryption.key'
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            password = os.getenv('ENCRYPTION_PASSWORD', 'ytempire-default-password').encode()
            salt = os.urandom(16)
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            
            # Save key
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Restrict permissions
            os.chmod(key_file, 0o600)
            
            return key
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a secret value"""
        f = Fernet(self.encryption_key)
        return f.encrypt(value.encode()).decode()
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a secret value"""
        f = Fernet(self.encryption_key)
        return f.decrypt(encrypted_value.encode()).decode()
    
    def store_secret(self, key: str, value: Dict[str, Any], encrypted: bool = True) -> bool:
        """
        Store a secret
        
        Args:
            key: Secret identifier
            value: Secret data
            encrypted: Whether to encrypt the value
        
        Returns:
            Success status
        """
        try:
            if self.backend == "local":
                return self._store_local(key, value, encrypted)
            elif self.backend == "aws":
                return self._store_aws(key, value)
            elif self.backend == "vault":
                return self._store_vault(key, value)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
        except Exception as e:
            logger.error(f"Failed to store secret {key}: {e}")
            return False
    
    def retrieve_secret(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a secret
        
        Args:
            key: Secret identifier
        
        Returns:
            Secret data or None if not found
        """
        try:
            if self.backend == "local":
                return self._retrieve_local(key)
            elif self.backend == "aws":
                return self._retrieve_aws(key)
            elif self.backend == "vault":
                return self._retrieve_vault(key)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
        except Exception as e:
            logger.error(f"Failed to retrieve secret {key}: {e}")
            return None
    
    def _store_local(self, key: str, value: Dict[str, Any], encrypted: bool) -> bool:
        """Store secret locally"""
        secrets_dir = '.secrets'
        os.makedirs(secrets_dir, exist_ok=True)
        
        file_path = os.path.join(secrets_dir, f"{key}.json")
        
        if encrypted:
            # Encrypt sensitive fields
            for field in ['password', 'api_key', 'secret', 'token']:
                if field in value:
                    value[field] = self.encrypt_value(value[field])
        
        with open(file_path, 'w') as f:
            json.dump(value, f, indent=2)
        
        # Restrict permissions
        os.chmod(file_path, 0o600)
        
        logger.info(f"Stored secret locally: {key}")
        return True
    
    def _retrieve_local(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve secret from local storage"""
        file_path = os.path.join('.secrets', f"{key}.json")
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, 'r') as f:
            value = json.load(f)
        
        # Decrypt sensitive fields
        for field in ['password', 'api_key', 'secret', 'token']:
            if field in value and value[field].startswith('gAAAAA'):  # Fernet encrypted
                value[field] = self.decrypt_value(value[field])
        
        return value
    
    def _store_aws(self, key: str, value: Dict[str, Any]) -> bool:
        """Store secret in AWS Secrets Manager"""
        try:
            self.aws_client.create_secret(
                Name=f"ytempire/{key}",
                SecretString=json.dumps(value),
                Tags=[
                    {'Key': 'Application', 'Value': 'YTEmpire'},
                    {'Key': 'Environment', 'Value': os.getenv('ENVIRONMENT', 'development')},
                ]
            )
            logger.info(f"Stored secret in AWS: {key}")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceExistsException':
                # Update existing secret
                self.aws_client.update_secret(
                    SecretId=f"ytempire/{key}",
                    SecretString=json.dumps(value)
                )
                logger.info(f"Updated secret in AWS: {key}")
                return True
            raise
    
    def _retrieve_aws(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve secret from AWS Secrets Manager"""
        try:
            response = self.aws_client.get_secret_value(SecretId=f"ytempire/{key}")
            return json.loads(response['SecretString'])
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                return None
            raise
    
    def _store_vault(self, key: str, value: Dict[str, Any]) -> bool:
        """Store secret in HashiCorp Vault"""
        self.vault_client.secrets.kv.v2.create_or_update_secret(
            path=f"ytempire/{key}",
            secret=value
        )
        logger.info(f"Stored secret in Vault: {key}")
        return True
    
    def _retrieve_vault(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve secret from HashiCorp Vault"""
        try:
            response = self.vault_client.secrets.kv.v2.read_secret_version(
                path=f"ytempire/{key}"
            )
            return response['data']['data']
        except Exception:
            return None
    
    def rotate_secret(self, key: str, new_value: Dict[str, Any]) -> bool:
        """
        Rotate a secret
        
        Args:
            key: Secret identifier
            new_value: New secret data
        
        Returns:
            Success status
        """
        # Store old version with timestamp
        old_value = self.retrieve_secret(key)
        if old_value:
            import time
            backup_key = f"{key}_backup_{int(time.time())}"
            self.store_secret(backup_key, old_value)
        
        # Store new version
        return self.store_secret(key, new_value)
    
    def list_secrets(self) -> list:
        """List all available secrets"""
        if self.backend == "local":
            secrets_dir = '.secrets'
            if not os.path.exists(secrets_dir):
                return []
            return [f.replace('.json', '') for f in os.listdir(secrets_dir) if f.endswith('.json')]
        elif self.backend == "aws":
            response = self.aws_client.list_secrets(
                Filters=[{'Key': 'tag-key', 'Values': ['Application']}]
            )
            return [s['Name'].replace('ytempire/', '') for s in response['SecretList']]
        elif self.backend == "vault":
            response = self.vault_client.secrets.kv.v2.list_secrets(path='ytempire')
            return response['data']['keys']
        return []


class SecretsInitializer:
    """Initialize default secrets for YTEmpire"""
    
    def __init__(self, manager: SecretsManager):
        self.manager = manager
    
    def initialize_defaults(self):
        """Initialize default secrets if not present"""
        
        # Database credentials
        if not self.manager.retrieve_secret('database'):
            self.manager.store_secret('database', {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME', 'ytempire'),
                'username': os.getenv('DB_USER', 'ytempire_user'),
                'password': os.getenv('DB_PASSWORD', 'change_me_in_production')
            })
        
        # Redis credentials
        if not self.manager.retrieve_secret('redis'):
            self.manager.store_secret('redis', {
                'host': os.getenv('REDIS_HOST', 'localhost'),
                'port': os.getenv('REDIS_PORT', '6379'),
                'password': os.getenv('REDIS_PASSWORD', '')
            })
        
        # API Keys
        if not self.manager.retrieve_secret('api_keys'):
            self.manager.store_secret('api_keys', {
                'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
                'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY', ''),
                'elevenlabs_api_key': os.getenv('ELEVENLABS_API_KEY', ''),
                'youtube_api_key': os.getenv('YOUTUBE_API_KEY', ''),
                'youtube_client_id': os.getenv('YOUTUBE_CLIENT_ID', ''),
                'youtube_client_secret': os.getenv('YOUTUBE_CLIENT_SECRET', '')
            })
        
        # JWT Settings
        if not self.manager.retrieve_secret('jwt'):
            self.manager.store_secret('jwt', {
                'secret_key': os.getenv('JWT_SECRET_KEY', Fernet.generate_key().decode()),
                'algorithm': 'RS256',
                'access_token_expire_minutes': 30,
                'refresh_token_expire_days': 7
            })
        
        # Stripe Settings
        if not self.manager.retrieve_secret('stripe'):
            self.manager.store_secret('stripe', {
                'publishable_key': os.getenv('STRIPE_PUBLISHABLE_KEY', ''),
                'secret_key': os.getenv('STRIPE_SECRET_KEY', ''),
                'webhook_secret': os.getenv('STRIPE_WEBHOOK_SECRET', '')
            })
        
        # AWS Settings
        if not self.manager.retrieve_secret('aws'):
            self.manager.store_secret('aws', {
                'access_key_id': os.getenv('AWS_ACCESS_KEY_ID', ''),
                'secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY', ''),
                'region': os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
                's3_bucket': os.getenv('S3_BUCKET', 'ytempire-videos')
            })
        
        logger.info("Default secrets initialized")


def get_secret(key: str, field: Optional[str] = None) -> Any:
    """
    Convenience function to retrieve a secret
    
    Args:
        key: Secret identifier
        field: Optional specific field to retrieve
    
    Returns:
        Secret value
    """
    backend = os.getenv('SECRETS_BACKEND', 'local')
    manager = SecretsManager(backend=backend)
    
    secret = manager.retrieve_secret(key)
    if secret and field:
        return secret.get(field)
    return secret


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Initialize secrets manager
    backend = os.getenv('SECRETS_BACKEND', 'local')
    manager = SecretsManager(backend=backend)
    
    # Initialize default secrets
    initializer = SecretsInitializer(manager)
    initializer.initialize_defaults()
    
    # List all secrets
    secrets = manager.list_secrets()
    print(f"Available secrets: {secrets}")
    
    # Example: Retrieve database credentials
    db_config = get_secret('database')
    print(f"Database config: {db_config}")