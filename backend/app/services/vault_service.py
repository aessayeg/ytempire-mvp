"""
HashiCorp Vault Service
Owner: Security Engineer #1
"""

import hvac
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
import asyncio
from datetime import datetime, timedelta
import base64

from app.core.config import settings

logger = logging.getLogger(__name__)


class VaultService:
    """HashiCorp Vault client service."""
    
    def __init__(self, vault_url: str = "http://localhost:8200"):
        self.vault_url = vault_url
        self.client: Optional[hvac.Client] = None
        self._authenticated = False
        self._token_expires_at: Optional[datetime] = None
        
    async def initialize(self, auth_method: str = "userpass", **auth_kwargs) -> bool:
        """Initialize Vault client with authentication."""
        try:
            self.client = hvac.Client(url=self.vault_url)
            
            # Check if Vault is accessible
            if not self.client.sys.is_initialized():
                logger.error("Vault is not initialized")
                return False
            
            if self.client.sys.is_sealed():
                logger.error("Vault is sealed")
                return False
            
            # Authenticate based on method
            if auth_method == "userpass":
                return await self._authenticate_userpass(**auth_kwargs)
            elif auth_method == "token":
                return await self._authenticate_token(**auth_kwargs)
            else:
                logger.error(f"Unsupported auth method: {auth_method}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Vault client: {str(e)}")
            return False
    
    async def _authenticate_userpass(self, username: str, password: str) -> bool:
        """Authenticate using username/password."""
        try:
            auth_response = self.client.auth.userpass.login(
                username=username,
                password=password
            )
            
            if auth_response['auth']['client_token']:
                self.client.token = auth_response['auth']['client_token']
                self._authenticated = True
                
                # Calculate token expiration
                lease_duration = auth_response['auth'].get('lease_duration', 3600)
                self._token_expires_at = datetime.utcnow() + timedelta(seconds=lease_duration)
                
                logger.info(f"Successfully authenticated to Vault as {username}")
                return True
                
        except Exception as e:
            logger.error(f"Vault userpass authentication failed: {str(e)}")
        
        return False
    
    async def _authenticate_token(self, token: str) -> bool:
        """Authenticate using token."""
        try:
            self.client.token = token
            
            # Verify token is valid
            token_info = self.client.auth.token.lookup_self()
            if token_info:
                self._authenticated = True
                
                # Calculate token expiration
                ttl = token_info.get('data', {}).get('ttl', 3600)
                self._token_expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                
                logger.info("Successfully authenticated to Vault with token")
                return True
                
        except Exception as e:
            logger.error(f"Vault token authentication failed: {str(e)}")
        
        return False
    
    async def _ensure_authenticated(self) -> bool:
        """Ensure client is authenticated and token is valid."""
        if not self._authenticated or not self.client:
            return False
        
        # Check token expiration
        if self._token_expires_at and datetime.utcnow() >= self._token_expires_at:
            logger.warning("Vault token expired, re-authentication required")
            self._authenticated = False
            return False
        
        return True
    
    async def get_secret(self, path: str, key: Optional[str] = None) -> Optional[Any]:
        """Get secret from Vault KV store."""
        if not await self._ensure_authenticated():
            logger.error("Not authenticated to Vault")
            return None
        
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            secret_data = response['data']['data']
            
            if key:
                return secret_data.get(key)
            
            return secret_data
            
        except Exception as e:
            logger.error(f"Failed to get secret from {path}: {str(e)}")
            return None
    
    async def set_secret(self, path: str, secret: Dict[str, Any]) -> bool:
        """Set secret in Vault KV store."""
        if not await self._ensure_authenticated():
            logger.error("Not authenticated to Vault")
            return False
        
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=secret
            )
            
            logger.info(f"Successfully stored secret at {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set secret at {path}: {str(e)}")
            return False
    
    async def delete_secret(self, path: str) -> bool:
        """Delete secret from Vault KV store."""
        if not await self._ensure_authenticated():
            logger.error("Not authenticated to Vault")
            return False
        
        try:
            self.client.secrets.kv.v2.delete_latest_version_of_secret(path=path)
            logger.info(f"Successfully deleted secret at {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete secret at {path}: {str(e)}")
            return False
    
    async def encrypt_data(self, plaintext: str, key_name: str = "ytempire") -> Optional[str]:
        """Encrypt data using Vault Transit engine."""
        if not await self._ensure_authenticated():
            logger.error("Not authenticated to Vault")
            return None
        
        try:
            # Encode plaintext to base64
            encoded_plaintext = base64.b64encode(plaintext.encode()).decode()
            
            response = self.client.secrets.transit.encrypt_data(
                name=key_name,
                plaintext=encoded_plaintext
            )
            
            return response['data']['ciphertext']
            
        except Exception as e:
            logger.error(f"Failed to encrypt data: {str(e)}")
            return None
    
    async def decrypt_data(self, ciphertext: str, key_name: str = "ytempire") -> Optional[str]:
        """Decrypt data using Vault Transit engine."""
        if not await self._ensure_authenticated():
            logger.error("Not authenticated to Vault")
            return None
        
        try:
            response = self.client.secrets.transit.decrypt_data(
                name=key_name,
                ciphertext=ciphertext
            )
            
            # Decode from base64
            decoded_plaintext = base64.b64decode(response['data']['plaintext']).decode()
            return decoded_plaintext
            
        except Exception as e:
            logger.error(f"Failed to decrypt data: {str(e)}")
            return None
    
    async def get_database_credentials(self, role: str = "ytempire-app-role") -> Optional[Dict[str, str]]:
        """Get dynamic database credentials."""
        if not await self._ensure_authenticated():
            logger.error("Not authenticated to Vault")
            return None
        
        try:
            response = self.client.secrets.database.generate_credentials(name=role)
            
            return {
                'username': response['data']['username'],
                'password': response['data']['password'],
                'lease_id': response['lease_id'],
                'lease_duration': response['lease_duration']
            }
            
        except Exception as e:
            logger.error(f"Failed to get database credentials: {str(e)}")
            return None
    
    async def revoke_lease(self, lease_id: str) -> bool:
        """Revoke a Vault lease."""
        if not await self._ensure_authenticated():
            logger.error("Not authenticated to Vault")
            return False
        
        try:
            self.client.sys.revoke_lease(lease_id=lease_id)
            logger.info(f"Successfully revoked lease: {lease_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke lease {lease_id}: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Vault health status."""
        try:
            if not self.client:
                return {"healthy": False, "error": "Client not initialized"}
            
            health = self.client.sys.read_health_status()
            
            return {
                "healthy": True,
                "initialized": health.get("initialized", False),
                "sealed": health.get("sealed", True),
                "standby": health.get("standby", False),
                "version": health.get("version", "unknown"),
                "authenticated": self._authenticated
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }


# Global Vault service instance
vault_service = VaultService(settings.VAULT_URL if hasattr(settings, 'VAULT_URL') else "http://localhost:8200")


async def get_vault_service() -> VaultService:
    """Get initialized Vault service."""
    if not vault_service._authenticated:
        # Try to authenticate with app credentials
        success = await vault_service.initialize(
            auth_method="userpass",
            username="ytempire-app",
            password="ytempire-app-secret"
        )
        
        if not success:
            logger.warning("Failed to authenticate to Vault")
    
    return vault_service


# Utility functions for common operations
async def get_app_config() -> Dict[str, Any]:
    """Get application configuration from Vault."""
    vault = await get_vault_service()
    return await vault.get_secret("ytempire/app/config") or {}


async def get_api_keys() -> Dict[str, Any]:
    """Get API keys from Vault."""
    vault = await get_vault_service()
    return await vault.get_secret("ytempire/api-keys/external") or {}


async def get_database_config() -> Dict[str, Any]:
    """Get database configuration from Vault."""
    vault = await get_vault_service()
    return await vault.get_secret("ytempire/database/config") or {}


async def encrypt_sensitive_data(data: str) -> Optional[str]:
    """Encrypt sensitive data using Vault."""
    vault = await get_vault_service()
    return await vault.encrypt_data(data)


async def decrypt_sensitive_data(encrypted_data: str) -> Optional[str]:
    """Decrypt sensitive data using Vault."""
    vault = await get_vault_service()
    return await vault.decrypt_data(encrypted_data)