#!/usr/bin/env python3
"""
Data Encryption Manager for YTEmpire MVP
Comprehensive encryption for data at rest and in transit
"""

import os
import json
import base64
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import aiofiles

# Cryptographic libraries
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from cryptography.x509.oid import NameOID
import cryptography.hazmat.primitives.serialization as crypto_serialization

import sqlalchemy
from sqlalchemy_utils.types.encrypted.encrypted_type import AesEngine
import redis

logger = logging.getLogger(__name__)

class EncryptionType(Enum):
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    FERNET = "fernet"
    CHACHA20_POLY1305 = "chacha20_poly1305"

class KeyType(Enum):
    MASTER = "master"
    DATA = "data"
    BACKUP = "backup"
    TRANSIT = "transit"
    DATABASE = "database"
    FILES = "files"

@dataclass
class EncryptionKey:
    """Encryption key metadata"""
    key_id: str
    key_type: KeyType
    encryption_type: EncryptionType
    created_at: datetime
    expires_at: Optional[datetime]
    version: int
    is_active: bool
    metadata: Dict[str, Any]

@dataclass
class EncryptedData:
    """Encrypted data container"""
    encrypted_data: bytes
    key_id: str
    encryption_type: EncryptionType
    iv_or_nonce: Optional[bytes]
    auth_tag: Optional[bytes]
    metadata: Dict[str, Any]

class KeyManager:
    """Manages encryption keys with rotation and versioning"""
    
    def __init__(self, key_store_path: str = "/etc/ytempire/keys"):
        self.key_store_path = key_store_path
        self.keys: Dict[str, EncryptionKey] = {}
        self.key_data: Dict[str, bytes] = {}
        
        # Ensure key store directory exists
        os.makedirs(key_store_path, mode=0o700, exist_ok=True)
        
        # Master key for key encryption (KEK - Key Encryption Key)
        self.master_key = self._load_or_create_master_key()
    
    def _load_or_create_master_key(self) -> bytes:
        """Load or create master key for key encryption"""
        master_key_path = os.path.join(self.key_store_path, "master.key")
        
        if os.path.exists(master_key_path):
            try:
                with open(master_key_path, "rb") as f:
                    encrypted_master = f.read()
                
                # In production, this would come from a secure source like:
                # - Hardware Security Module (HSM)
                # - AWS KMS, Azure Key Vault, Google Cloud KMS
                # - Environment variable from secure secret management
                passphrase = os.getenv("YTEMPIRE_MASTER_PASSPHRASE", "default_dev_passphrase").encode()
                
                # Derive key from passphrase
                salt = encrypted_master[:32]  # First 32 bytes are salt
                kdf = Scrypt(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    n=2**14,
                    r=8,
                    p=1,
                    backend=default_backend()
                )
                key = kdf.derive(passphrase)
                
                # Decrypt master key
                f = Fernet(base64.urlsafe_b64encode(key))
                master_key = f.decrypt(encrypted_master[32:])
                
                logger.info("Master key loaded successfully")
                return master_key
                
            except Exception as e:
                logger.error(f"Failed to load master key: {e}")
                raise
        else:
            # Create new master key
            master_key = secrets.token_bytes(32)  # 256-bit key
            
            # Encrypt master key with passphrase
            passphrase = os.getenv("YTEMPIRE_MASTER_PASSPHRASE", "default_dev_passphrase").encode()
            salt = secrets.token_bytes(32)
            
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                n=2**14,
                r=8,
                p=1,
                backend=default_backend()
            )
            key = kdf.derive(passphrase)
            
            f = Fernet(base64.urlsafe_b64encode(key))
            encrypted_master = f.encrypt(master_key)
            
            # Save encrypted master key
            with open(master_key_path, "wb") as f:
                f.write(salt + encrypted_master)
            
            # Set secure permissions
            os.chmod(master_key_path, 0o600)
            
            logger.info("New master key created and saved")
            return master_key
    
    def generate_key(self, key_type: KeyType, encryption_type: EncryptionType, 
                    expires_in_days: Optional[int] = None) -> str:
        """Generate a new encryption key"""
        key_id = f"{key_type.value}_{encryption_type.value}_{secrets.token_hex(8)}"
        
        # Generate key based on encryption type
        if encryption_type == EncryptionType.AES_256_GCM:
            key_data = secrets.token_bytes(32)  # 256 bits
        elif encryption_type == EncryptionType.AES_256_CBC:
            key_data = secrets.token_bytes(32)  # 256 bits
        elif encryption_type == EncryptionType.FERNET:
            key_data = Fernet.generate_key()
        elif encryption_type == EncryptionType.CHACHA20_POLY1305:
            key_data = secrets.token_bytes(32)  # 256 bits
        elif encryption_type in [EncryptionType.RSA_2048, EncryptionType.RSA_4096]:
            # Generate RSA key pair
            key_size = 2048 if encryption_type == EncryptionType.RSA_2048 else 4096
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
            key_data = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            raise ValueError(f"Unsupported encryption type: {encryption_type}")
        
        # Create key metadata
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        key_metadata = EncryptionKey(
            key_id=key_id,
            key_type=key_type,
            encryption_type=encryption_type,
            created_at=datetime.now(),
            expires_at=expires_at,
            version=1,
            is_active=True,
            metadata={}
        )
        
        # Store key (encrypted with master key)
        self.keys[key_id] = key_metadata
        self.key_data[key_id] = key_data
        
        # Persist to disk
        self._save_key(key_id, key_data, key_metadata)
        
        logger.info(f"Generated new {encryption_type.value} key: {key_id}")
        return key_id
    
    def _save_key(self, key_id: str, key_data: bytes, key_metadata: EncryptionKey):
        """Save encrypted key to disk"""
        key_file_path = os.path.join(self.key_store_path, f"{key_id}.key")
        metadata_file_path = os.path.join(self.key_store_path, f"{key_id}.meta")
        
        # Encrypt key data with master key
        f = Fernet(base64.urlsafe_b64encode(self.master_key))
        encrypted_key_data = f.encrypt(key_data)
        
        # Save encrypted key
        with open(key_file_path, "wb") as f:
            f.write(encrypted_key_data)
        
        # Save metadata
        with open(metadata_file_path, "w") as f:
            json.dump(asdict(key_metadata), f, default=str, indent=2)
        
        # Set secure permissions
        os.chmod(key_file_path, 0o600)
        os.chmod(metadata_file_path, 0o600)
    
    def get_key(self, key_id: str) -> Optional[bytes]:
        """Get decrypted key data"""
        if key_id in self.key_data:
            return self.key_data[key_id]
        
        # Load from disk
        key_file_path = os.path.join(self.key_store_path, f"{key_id}.key")
        if os.path.exists(key_file_path):
            with open(key_file_path, "rb") as f:
                encrypted_key_data = f.read()
            
            # Decrypt with master key
            f = Fernet(base64.urlsafe_b64encode(self.master_key))
            key_data = f.decrypt(encrypted_key_data)
            
            # Cache in memory
            self.key_data[key_id] = key_data
            return key_data
        
        return None
    
    def rotate_key(self, old_key_id: str) -> str:
        """Rotate an encryption key"""
        old_key = self.keys.get(old_key_id)
        if not old_key:
            raise ValueError(f"Key {old_key_id} not found")
        
        # Generate new key with same parameters
        new_key_id = self.generate_key(
            old_key.key_type,
            old_key.encryption_type,
            expires_in_days=365 if old_key.expires_at else None
        )
        
        # Update version
        self.keys[new_key_id].version = old_key.version + 1
        
        # Deactivate old key
        old_key.is_active = False
        
        logger.info(f"Rotated key {old_key_id} to {new_key_id}")
        return new_key_id
    
    def list_keys(self, key_type: Optional[KeyType] = None, 
                 active_only: bool = True) -> List[EncryptionKey]:
        """List encryption keys with optional filtering"""
        keys = []
        for key in self.keys.values():
            if key_type and key.key_type != key_type:
                continue
            if active_only and not key.is_active:
                continue
            keys.append(key)
        
        return sorted(keys, key=lambda k: k.created_at, reverse=True)

class DataEncryption:
    """Handles data encryption and decryption operations"""
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
    
    def encrypt_data(self, data: Union[str, bytes], key_id: str, 
                    additional_data: Optional[bytes] = None) -> EncryptedData:
        """Encrypt data with specified key"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        key = self.key_manager.get_key(key_id)
        if not key:
            raise ValueError(f"Key {key_id} not found")
        
        key_metadata = self.key_manager.keys[key_id]
        encryption_type = key_metadata.encryption_type
        
        if encryption_type == EncryptionType.AES_256_GCM:
            return self._encrypt_aes_gcm(data, key, key_id, additional_data)
        elif encryption_type == EncryptionType.AES_256_CBC:
            return self._encrypt_aes_cbc(data, key, key_id)
        elif encryption_type == EncryptionType.FERNET:
            return self._encrypt_fernet(data, key, key_id)
        elif encryption_type == EncryptionType.CHACHA20_POLY1305:
            return self._encrypt_chacha20(data, key, key_id, additional_data)
        elif encryption_type in [EncryptionType.RSA_2048, EncryptionType.RSA_4096]:
            return self._encrypt_rsa(data, key, key_id)
        else:
            raise ValueError(f"Unsupported encryption type: {encryption_type}")
    
    def decrypt_data(self, encrypted_data: EncryptedData, 
                    additional_data: Optional[bytes] = None) -> bytes:
        """Decrypt data"""
        key = self.key_manager.get_key(encrypted_data.key_id)
        if not key:
            raise ValueError(f"Key {encrypted_data.key_id} not found")
        
        encryption_type = encrypted_data.encryption_type
        
        if encryption_type == EncryptionType.AES_256_GCM:
            return self._decrypt_aes_gcm(encrypted_data, key, additional_data)
        elif encryption_type == EncryptionType.AES_256_CBC:
            return self._decrypt_aes_cbc(encrypted_data, key)
        elif encryption_type == EncryptionType.FERNET:
            return self._decrypt_fernet(encrypted_data, key)
        elif encryption_type == EncryptionType.CHACHA20_POLY1305:
            return self._decrypt_chacha20(encrypted_data, key, additional_data)
        elif encryption_type in [EncryptionType.RSA_2048, EncryptionType.RSA_4096]:
            return self._decrypt_rsa(encrypted_data, key)
        else:
            raise ValueError(f"Unsupported encryption type: {encryption_type}")
    
    def _encrypt_aes_gcm(self, data: bytes, key: bytes, key_id: str, 
                        additional_data: Optional[bytes]) -> EncryptedData:
        """Encrypt with AES-256-GCM"""
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        if additional_data:
            encryptor.authenticate_additional_data(additional_data)
        
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return EncryptedData(
            encrypted_data=ciphertext,
            key_id=key_id,
            encryption_type=EncryptionType.AES_256_GCM,
            iv_or_nonce=iv,
            auth_tag=encryptor.tag,
            metadata={'additional_data_len': len(additional_data) if additional_data else 0}
        )
    
    def _decrypt_aes_gcm(self, encrypted_data: EncryptedData, key: bytes, 
                        additional_data: Optional[bytes]) -> bytes:
        """Decrypt AES-256-GCM data"""
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(encrypted_data.iv_or_nonce, encrypted_data.auth_tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        if additional_data:
            decryptor.authenticate_additional_data(additional_data)
        
        return decryptor.update(encrypted_data.encrypted_data) + decryptor.finalize()
    
    def _encrypt_aes_cbc(self, data: bytes, key: bytes, key_id: str) -> EncryptedData:
        """Encrypt with AES-256-CBC"""
        iv = secrets.token_bytes(16)  # 128-bit IV
        
        # Add PKCS7 padding
        pad_len = 16 - (len(data) % 16)
        padded_data = data + bytes([pad_len] * pad_len)
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        return EncryptedData(
            encrypted_data=ciphertext,
            key_id=key_id,
            encryption_type=EncryptionType.AES_256_CBC,
            iv_or_nonce=iv,
            auth_tag=None,
            metadata={}
        )
    
    def _decrypt_aes_cbc(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt AES-256-CBC data"""
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(encrypted_data.iv_or_nonce),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_data.encrypted_data) + decryptor.finalize()
        
        # Remove PKCS7 padding
        pad_len = padded_data[-1]
        return padded_data[:-pad_len]
    
    def _encrypt_fernet(self, data: bytes, key: bytes, key_id: str) -> EncryptedData:
        """Encrypt with Fernet (AES-128 in CBC mode + HMAC)"""
        f = Fernet(key)
        ciphertext = f.encrypt(data)
        
        return EncryptedData(
            encrypted_data=ciphertext,
            key_id=key_id,
            encryption_type=EncryptionType.FERNET,
            iv_or_nonce=None,
            auth_tag=None,
            metadata={}
        )
    
    def _decrypt_fernet(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt Fernet data"""
        f = Fernet(key)
        return f.decrypt(encrypted_data.encrypted_data)
    
    def _encrypt_chacha20(self, data: bytes, key: bytes, key_id: str,
                         additional_data: Optional[bytes]) -> EncryptedData:
        """Encrypt with ChaCha20-Poly1305"""
        nonce = secrets.token_bytes(12)  # 96-bit nonce
        
        cipher = Cipher(
            algorithms.ChaCha20(key, nonce),
            modes.GCM(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        if additional_data:
            encryptor.authenticate_additional_data(additional_data)
        
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return EncryptedData(
            encrypted_data=ciphertext,
            key_id=key_id,
            encryption_type=EncryptionType.CHACHA20_POLY1305,
            iv_or_nonce=nonce,
            auth_tag=encryptor.tag,
            metadata={'additional_data_len': len(additional_data) if additional_data else 0}
        )
    
    def _decrypt_chacha20(self, encrypted_data: EncryptedData, key: bytes,
                         additional_data: Optional[bytes]) -> bytes:
        """Decrypt ChaCha20-Poly1305 data"""
        cipher = Cipher(
            algorithms.ChaCha20(key, encrypted_data.iv_or_nonce),
            modes.GCM(encrypted_data.iv_or_nonce, encrypted_data.auth_tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        if additional_data:
            decryptor.authenticate_additional_data(additional_data)
        
        return decryptor.update(encrypted_data.encrypted_data) + decryptor.finalize()
    
    def _encrypt_rsa(self, data: bytes, key: bytes, key_id: str) -> EncryptedData:
        """Encrypt with RSA"""
        private_key = serialization.load_pem_private_key(
            key, password=None, backend=default_backend()
        )
        public_key = private_key.public_key()
        
        # RSA encryption is limited by key size, use OAEP padding
        ciphertext = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return EncryptedData(
            encrypted_data=ciphertext,
            key_id=key_id,
            encryption_type=EncryptionType.RSA_2048,  # Will be overridden by actual type
            iv_or_nonce=None,
            auth_tag=None,
            metadata={}
        )
    
    def _decrypt_rsa(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt RSA data"""
        private_key = serialization.load_pem_private_key(
            key, password=None, backend=default_backend()
        )
        
        plaintext = private_key.decrypt(
            encrypted_data.encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return plaintext

class DatabaseEncryption:
    """Database-specific encryption functionality"""
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
        self.data_encryption = DataEncryption(key_manager)
        
        # Generate database encryption key if it doesn't exist
        db_keys = key_manager.list_keys(KeyType.DATABASE)
        if not db_keys:
            self.db_key_id = key_manager.generate_key(
                KeyType.DATABASE, 
                EncryptionType.AES_256_GCM,
                expires_in_days=365
            )
        else:
            self.db_key_id = db_keys[0].key_id
    
    def create_encrypted_column_type(self, secret_key: Optional[str] = None):
        """Create SQLAlchemy encrypted column type"""
        from sqlalchemy_utils import EncryptedType
        from sqlalchemy import String
        
        if not secret_key:
            # Use our managed key
            key_data = self.key_manager.get_key(self.db_key_id)
            secret_key = base64.urlsafe_b64encode(key_data[:32]).decode()
        
        return EncryptedType(String, secret_key, AesEngine, 'pkcs5')
    
    def encrypt_database_field(self, value: str, field_name: str = "") -> str:
        """Encrypt a database field value"""
        additional_data = field_name.encode() if field_name else None
        
        encrypted = self.data_encryption.encrypt_data(
            value, self.db_key_id, additional_data
        )
        
        # Serialize for database storage
        return base64.b64encode(self._serialize_encrypted_data(encrypted)).decode()
    
    def decrypt_database_field(self, encrypted_value: str, field_name: str = "") -> str:
        """Decrypt a database field value"""
        additional_data = field_name.encode() if field_name else None
        
        # Deserialize from database
        serialized = base64.b64decode(encrypted_value.encode())
        encrypted = self._deserialize_encrypted_data(serialized)
        
        decrypted = self.data_encryption.decrypt_data(encrypted, additional_data)
        return decrypted.decode('utf-8')
    
    def _serialize_encrypted_data(self, encrypted: EncryptedData) -> bytes:
        """Serialize encrypted data for storage"""
        data = {
            'encrypted_data': base64.b64encode(encrypted.encrypted_data).decode(),
            'key_id': encrypted.key_id,
            'encryption_type': encrypted.encryption_type.value,
            'iv_or_nonce': base64.b64encode(encrypted.iv_or_nonce).decode() if encrypted.iv_or_nonce else None,
            'auth_tag': base64.b64encode(encrypted.auth_tag).decode() if encrypted.auth_tag else None,
            'metadata': encrypted.metadata
        }
        return json.dumps(data).encode()
    
    def _deserialize_encrypted_data(self, serialized: bytes) -> EncryptedData:
        """Deserialize encrypted data from storage"""
        data = json.loads(serialized.decode())
        
        return EncryptedData(
            encrypted_data=base64.b64decode(data['encrypted_data']),
            key_id=data['key_id'],
            encryption_type=EncryptionType(data['encryption_type']),
            iv_or_nonce=base64.b64decode(data['iv_or_nonce']) if data['iv_or_nonce'] else None,
            auth_tag=base64.b64decode(data['auth_tag']) if data['auth_tag'] else None,
            metadata=data['metadata']
        )

class FileEncryption:
    """File system encryption functionality"""
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
        self.data_encryption = DataEncryption(key_manager)
        
        # Generate file encryption key if it doesn't exist
        file_keys = key_manager.list_keys(KeyType.FILES)
        if not file_keys:
            self.file_key_id = key_manager.generate_key(
                KeyType.FILES,
                EncryptionType.AES_256_GCM,
                expires_in_days=365
            )
        else:
            self.file_key_id = file_keys[0].key_id
    
    async def encrypt_file(self, input_path: str, output_path: str, 
                          chunk_size: int = 8192) -> Dict[str, Any]:
        """Encrypt a file"""
        file_metadata = {
            'original_name': os.path.basename(input_path),
            'original_size': os.path.getsize(input_path),
            'encryption_timestamp': datetime.now().isoformat(),
            'key_id': self.file_key_id
        }
        
        # Generate file-specific additional data
        additional_data = json.dumps(file_metadata, sort_keys=True).encode()
        
        async with aiofiles.open(input_path, 'rb') as infile:
            async with aiofiles.open(output_path, 'wb') as outfile:
                # Write metadata header
                metadata_json = json.dumps(file_metadata).encode()
                metadata_length = len(metadata_json)
                await outfile.write(metadata_length.to_bytes(4, 'big'))
                await outfile.write(metadata_json)
                
                # Encrypt file in chunks
                total_encrypted = 0
                while True:
                    chunk = await infile.read(chunk_size)
                    if not chunk:
                        break
                    
                    # Encrypt chunk
                    encrypted_chunk = self.data_encryption.encrypt_data(
                        chunk, self.file_key_id, additional_data
                    )
                    
                    # Serialize and write encrypted chunk
                    serialized = self._serialize_encrypted_data(encrypted_chunk)
                    chunk_length = len(serialized)
                    await outfile.write(chunk_length.to_bytes(4, 'big'))
                    await outfile.write(serialized)
                    total_encrypted += len(chunk)
        
        return {
            'encrypted_file': output_path,
            'original_size': file_metadata['original_size'],
            'encrypted_size': os.path.getsize(output_path),
            'key_id': self.file_key_id
        }
    
    async def decrypt_file(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Decrypt a file"""
        async with aiofiles.open(input_path, 'rb') as infile:
            # Read metadata header
            metadata_length_bytes = await infile.read(4)
            metadata_length = int.from_bytes(metadata_length_bytes, 'big')
            metadata_json = await infile.read(metadata_length)
            file_metadata = json.loads(metadata_json.decode())
            
            additional_data = json.dumps(file_metadata, sort_keys=True).encode()
            
            async with aiofiles.open(output_path, 'wb') as outfile:
                total_decrypted = 0
                while True:
                    # Read chunk length
                    chunk_length_bytes = await infile.read(4)
                    if len(chunk_length_bytes) != 4:
                        break
                    
                    chunk_length = int.from_bytes(chunk_length_bytes, 'big')
                    
                    # Read encrypted chunk
                    encrypted_chunk_data = await infile.read(chunk_length)
                    encrypted_chunk = self._deserialize_encrypted_data(encrypted_chunk_data)
                    
                    # Decrypt chunk
                    decrypted_chunk = self.data_encryption.decrypt_data(
                        encrypted_chunk, additional_data
                    )
                    
                    await outfile.write(decrypted_chunk)
                    total_decrypted += len(decrypted_chunk)
        
        return {
            'decrypted_file': output_path,
            'original_metadata': file_metadata,
            'decrypted_size': total_decrypted
        }
    
    def _serialize_encrypted_data(self, encrypted: EncryptedData) -> bytes:
        """Serialize encrypted data for file storage"""
        data = {
            'encrypted_data': base64.b64encode(encrypted.encrypted_data).decode(),
            'key_id': encrypted.key_id,
            'encryption_type': encrypted.encryption_type.value,
            'iv_or_nonce': base64.b64encode(encrypted.iv_or_nonce).decode() if encrypted.iv_or_nonce else None,
            'auth_tag': base64.b64encode(encrypted.auth_tag).decode() if encrypted.auth_tag else None,
            'metadata': encrypted.metadata
        }
        return json.dumps(data).encode()
    
    def _deserialize_encrypted_data(self, serialized: bytes) -> EncryptedData:
        """Deserialize encrypted data from file storage"""
        data = json.loads(serialized.decode())
        
        return EncryptedData(
            encrypted_data=base64.b64decode(data['encrypted_data']),
            key_id=data['key_id'],
            encryption_type=EncryptionType(data['encryption_type']),
            iv_or_nonce=base64.b64decode(data['iv_or_nonce']) if data['iv_or_nonce'] else None,
            auth_tag=base64.b64decode(data['auth_tag']) if data['auth_tag'] else None,
            metadata=data['metadata']
        )

class TLSManager:
    """Manages TLS certificates for encryption in transit"""
    
    def __init__(self, cert_path: str = "/etc/ytempire/certs"):
        self.cert_path = cert_path
        os.makedirs(cert_path, mode=0o700, exist_ok=True)
    
    def generate_self_signed_certificate(self, domain: str, 
                                       validity_days: int = 365) -> Tuple[str, str]:
        """Generate self-signed certificate for development"""
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Generate certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "YTEmpire"),
            x509.NameAttribute(NameOID.COMMON_NAME, domain),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.now()
        ).not_valid_after(
            datetime.now() + timedelta(days=validity_days)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(domain),
                x509.DNSName(f"*.{domain}"),
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256(), default_backend())
        
        # Save certificate and key
        cert_file = os.path.join(self.cert_path, f"{domain}.crt")
        key_file = os.path.join(self.cert_path, f"{domain}.key")
        
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        with open(key_file, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        # Set secure permissions
        os.chmod(cert_file, 0o644)
        os.chmod(key_file, 0o600)
        
        logger.info(f"Generated self-signed certificate for {domain}")
        return cert_file, key_file
    
    def create_combined_pem(self, cert_file: str, key_file: str, 
                           output_file: str) -> str:
        """Combine certificate and key for HAProxy"""
        with open(output_file, 'w') as combined:
            with open(cert_file, 'r') as cert:
                combined.write(cert.read())
            with open(key_file, 'r') as key:
                combined.write(key.read())
        
        os.chmod(output_file, 0o600)
        return output_file

class EncryptionManager:
    """Main encryption management system"""
    
    def __init__(self, key_store_path: str = "/etc/ytempire/keys",
                 cert_path: str = "/etc/ytempire/certs"):
        self.key_manager = KeyManager(key_store_path)
        self.data_encryption = DataEncryption(self.key_manager)
        self.db_encryption = DatabaseEncryption(self.key_manager)
        self.file_encryption = FileEncryption(self.key_manager)
        self.tls_manager = TLSManager(cert_path)
        
        # Initialize default keys
        self._initialize_default_keys()
    
    def _initialize_default_keys(self):
        """Initialize default encryption keys for different purposes"""
        # Transit encryption key
        transit_keys = self.key_manager.list_keys(KeyType.TRANSIT)
        if not transit_keys:
            self.key_manager.generate_key(
                KeyType.TRANSIT,
                EncryptionType.AES_256_GCM,
                expires_in_days=90
            )
        
        # Backup encryption key
        backup_keys = self.key_manager.list_keys(KeyType.BACKUP)
        if not backup_keys:
            self.key_manager.generate_key(
                KeyType.BACKUP,
                EncryptionType.AES_256_GCM,
                expires_in_days=365
            )
    
    def get_encryption_status(self) -> Dict[str, Any]:
        """Get comprehensive encryption status"""
        return {
            'key_counts': {
                key_type.value: len(self.key_manager.list_keys(key_type))
                for key_type in KeyType
            },
            'active_keys': [
                {
                    'key_id': key.key_id,
                    'type': key.key_type.value,
                    'encryption': key.encryption_type.value,
                    'created': key.created_at.isoformat(),
                    'expires': key.expires_at.isoformat() if key.expires_at else None
                }
                for key in self.key_manager.list_keys(active_only=True)
            ],
            'tls_certificates': self._get_certificate_status()
        }
    
    def _get_certificate_status(self) -> List[Dict[str, Any]]:
        """Get TLS certificate status"""
        certificates = []
        for cert_file in os.listdir(self.tls_manager.cert_path):
            if cert_file.endswith('.crt'):
                cert_path = os.path.join(self.tls_manager.cert_path, cert_file)
                try:
                    with open(cert_path, 'rb') as f:
                        cert_data = f.read()
                    
                    cert = x509.load_pem_x509_certificate(cert_data, default_backend())
                    
                    certificates.append({
                        'file': cert_file,
                        'subject': cert.subject.rfc4514_string(),
                        'not_valid_before': cert.not_valid_before.isoformat(),
                        'not_valid_after': cert.not_valid_after.isoformat(),
                        'is_expired': cert.not_valid_after < datetime.now()
                    })
                except Exception as e:
                    logger.error(f"Failed to read certificate {cert_file}: {e}")
        
        return certificates
    
    async def rotate_all_keys(self):
        """Rotate all encryption keys (maintenance operation)"""
        rotated_keys = []
        for key in self.key_manager.list_keys(active_only=True):
            if key.expires_at and key.expires_at < datetime.now() + timedelta(days=30):
                new_key_id = self.key_manager.rotate_key(key.key_id)
                rotated_keys.append((key.key_id, new_key_id))
        
        return rotated_keys

# CLI Interface and utility functions
async def main():
    """Command-line interface for encryption management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YTEmpire Encryption Manager")
    parser.add_argument("action", choices=["init", "status", "generate-key", "rotate-key", "encrypt-file", "decrypt-file", "generate-cert"])
    parser.add_argument("--key-type", choices=[kt.value for kt in KeyType], help="Key type")
    parser.add_argument("--encryption-type", choices=[et.value for et in EncryptionType], help="Encryption type")
    parser.add_argument("--input", help="Input file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--domain", help="Domain for certificate generation")
    parser.add_argument("--key-id", help="Key ID for operations")
    
    args = parser.parse_args()
    
    # Initialize encryption manager
    manager = EncryptionManager()
    
    if args.action == "init":
        print("Encryption system initialized successfully")
        
    elif args.action == "status":
        status = manager.get_encryption_status()
        print(json.dumps(status, indent=2))
        
    elif args.action == "generate-key":
        if not args.key_type or not args.encryption_type:
            print("ERROR: --key-type and --encryption-type required")
            return
        
        key_id = manager.key_manager.generate_key(
            KeyType(args.key_type),
            EncryptionType(args.encryption_type)
        )
        print(f"Generated key: {key_id}")
        
    elif args.action == "rotate-key":
        if not args.key_id:
            print("ERROR: --key-id required")
            return
        
        new_key_id = manager.key_manager.rotate_key(args.key_id)
        print(f"Rotated {args.key_id} to {new_key_id}")
        
    elif args.action == "encrypt-file":
        if not args.input or not args.output:
            print("ERROR: --input and --output required")
            return
        
        result = await manager.file_encryption.encrypt_file(args.input, args.output)
        print(f"Encrypted file: {result}")
        
    elif args.action == "decrypt-file":
        if not args.input or not args.output:
            print("ERROR: --input and --output required")
            return
        
        result = await manager.file_encryption.decrypt_file(args.input, args.output)
        print(f"Decrypted file: {result}")
        
    elif args.action == "generate-cert":
        if not args.domain:
            print("ERROR: --domain required")
            return
        
        cert_file, key_file = manager.tls_manager.generate_self_signed_certificate(args.domain)
        print(f"Generated certificate: {cert_file}, {key_file}")

if __name__ == "__main__":
    import ipaddress  # Add this import at the top
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())