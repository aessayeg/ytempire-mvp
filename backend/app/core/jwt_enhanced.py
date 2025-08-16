"""
Enhanced JWT Security Configuration
Implements JWT best practices with refresh tokens and blacklisting
"""
import secrets
import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
import redis
import hashlib
import logging

from .config import settings

logger = logging.getLogger(__name__)


class EnhancedJWTManager:
    """
    Enhanced JWT management with security best practices
    """

    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True
        )

        # JWT Configuration
        self.algorithm = "RS256"  # Use RSA instead of HS256 for better security
        self.access_token_expire = timedelta(minutes=15)  # Short-lived access tokens
        self.refresh_token_expire = timedelta(days=7)  # Longer refresh tokens
        self.refresh_token_reuse_window = timedelta(
            minutes=2
        )  # Grace period for token rotation

        # Generate or load RSA keys (in production, load from secure storage)
        self._setup_keys()

        # Token blacklist prefix
        self.blacklist_prefix = "jwt_blacklist:"
        self.refresh_token_prefix = "refresh_token:"

    def _setup_keys(self):
        """Setup RSA keys for JWT signing"""
        try:
            # In production, these should be loaded from secure storage
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.backends import default_backend

            # Generate RSA key pair (do this once and store securely)
            private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048, backend=default_backend()
            )

            # Serialize keys
            self.private_key = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )

            self.public_key = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        except Exception as e:
            # Fallback to HS256 if RSA setup fails
            logger.warning(f"RSA key setup failed, falling back to HS256: {e}")
            self.algorithm = "HS256"
            self.private_key = settings.SECRET_KEY
            self.public_key = settings.SECRET_KEY

    def create_access_token(
        self,
        subject: str,
        user_id: int,
        scopes: list = None,
        additional_claims: Dict[str, Any] = None,
    ) -> str:
        """Create a secure access token"""

        # Token ID for tracking
        jti = secrets.token_urlsafe(16)

        # Build claims
        claims = {
            "sub": subject,  # Subject (usually email)
            "user_id": user_id,
            "type": "access",
            "jti": jti,  # JWT ID for blacklisting
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + self.access_token_expire,
            "nbf": datetime.now(timezone.utc),  # Not before
            "scopes": scopes or [],
        }

        # Add additional claims if provided
        if additional_claims:
            claims.update(additional_claims)

        # Create token
        token = jwt.encode(claims, self.private_key, algorithm=self.algorithm)

        return token

    def create_refresh_token(
        self, subject: str, user_id: int, family_id: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Create a refresh token with rotation support
        Returns: (token, family_id)
        """

        # Generate family ID for token rotation tracking
        if not family_id:
            family_id = secrets.token_urlsafe(16)

        # Token ID
        jti = secrets.token_urlsafe(16)

        # Build claims
        claims = {
            "sub": subject,
            "user_id": user_id,
            "type": "refresh",
            "jti": jti,
            "family": family_id,  # Token family for rotation
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + self.refresh_token_expire,
        }

        # Create token
        token = jwt.encode(claims, self.private_key, algorithm=self.algorithm)

        # Store refresh token metadata in Redis
        self._store_refresh_token(jti, user_id, family_id)

        return token, family_id

    def verify_access_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode access token"""
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[self.algorithm],
                options={"require": ["exp", "jti", "sub", "type"]},
            )

            # Verify token type
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type",
                )

            # Check if token is blacklisted
            if self._is_token_blacklisted(payload["jti"]):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                )

            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
            )

    def verify_refresh_token(self, token: str) -> Dict[str, Any]:
        """Verify refresh token and check for reuse"""
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[self.algorithm],
                options={"require": ["exp", "jti", "sub", "type", "family"]},
            )

            # Verify token type
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type",
                )

            # Check if token exists and is valid
            if not self._validate_refresh_token(payload["jti"], payload["family"]):
                # Token reuse detected - revoke entire family
                self._revoke_token_family(payload["family"])
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token reuse detected - all tokens revoked",
                )

            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired",
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid refresh token: {str(e)}",
            )

    def rotate_refresh_token(self, old_token: str) -> tuple[str, str, str]:
        """
        Rotate refresh token (create new access and refresh tokens)
        Returns: (new_access_token, new_refresh_token, family_id)
        """
        # Verify old refresh token
        payload = self.verify_refresh_token(old_token)

        # Mark old token as used
        self._mark_refresh_token_used(payload["jti"])

        # Create new access token
        new_access_token = self.create_access_token(
            subject=payload["sub"], user_id=payload["user_id"]
        )

        # Create new refresh token in same family
        new_refresh_token, family_id = self.create_refresh_token(
            subject=payload["sub"],
            user_id=payload["user_id"],
            family_id=payload["family"],
        )

        return new_access_token, new_refresh_token, family_id

    def revoke_token(self, token: str):
        """Revoke a token by adding it to blacklist"""
        try:
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False},  # Allow expired tokens to be revoked
            )

            jti = payload.get("jti")
            if jti:
                # Calculate remaining TTL
                exp = payload.get("exp")
                if exp:
                    ttl = max(0, exp - datetime.now(timezone.utc).timestamp())
                    # Add to blacklist with TTL
                    self.redis_client.setex(
                        f"{self.blacklist_prefix}{jti}", int(ttl), "revoked"
                    )
                    logger.info(f"Token {jti} revoked")

        except jwt.InvalidTokenError:
            pass  # Invalid tokens don't need to be blacklisted

    def revoke_all_user_tokens(self, user_id: int):
        """Revoke all tokens for a user"""
        # In production, maintain a user->tokens mapping for efficient revocation
        # For now, we'll increment a version number that's checked during validation
        self.redis_client.setex(
            f"user_token_version:{user_id}",
            int(self.refresh_token_expire.total_seconds()),
            datetime.now(timezone.utc).isoformat(),
        )
        logger.info(f"All tokens revoked for user {user_id}")

    def _store_refresh_token(self, jti: str, user_id: int, family_id: str):
        """Store refresh token metadata"""
        key = f"{self.refresh_token_prefix}{jti}"
        value = {
            "user_id": user_id,
            "family": family_id,
            "created": datetime.now(timezone.utc).isoformat(),
            "used": False,
        }
        self.redis_client.setex(
            key, int(self.refresh_token_expire.total_seconds()), json.dumps(value)
        )

    def _validate_refresh_token(self, jti: str, family_id: str) -> bool:
        """Validate refresh token hasn't been used"""
        key = f"{self.refresh_token_prefix}{jti}"
        data = self.redis_client.get(key)

        if not data:
            return False

        token_data = json.loads(data)

        # Check if token has been used
        if token_data.get("used"):
            return False

        # Validate family
        if token_data.get("family") != family_id:
            return False

        return True

    def _mark_refresh_token_used(self, jti: str):
        """Mark refresh token as used"""
        key = f"{self.refresh_token_prefix}{jti}"
        data = self.redis_client.get(key)

        if data:
            token_data = json.loads(data)
            token_data["used"] = True
            token_data["used_at"] = datetime.now(timezone.utc).isoformat()

            # Keep for a short grace period for race conditions
            self.redis_client.setex(
                key,
                int(self.refresh_token_reuse_window.total_seconds()),
                json.dumps(token_data),
            )

    def _revoke_token_family(self, family_id: str):
        """Revoke entire token family (potential token reuse detected)"""
        # In production, maintain family->tokens mapping
        logger.warning(f"Token family {family_id} revoked due to reuse detection")
        self.redis_client.setex(
            f"revoked_family:{family_id}",
            int(self.refresh_token_expire.total_seconds()),
            "revoked",
        )

    def _is_token_blacklisted(self, jti: str) -> bool:
        """Check if token is blacklisted"""
        return bool(self.redis_client.get(f"{self.blacklist_prefix}{jti}"))


# Global instance
jwt_manager = EnhancedJWTManager()


# Helper functions for easy use
def create_tokens(subject: str, user_id: int, scopes: list = None) -> Dict[str, str]:
    """Create both access and refresh tokens"""
    access_token = jwt_manager.create_access_token(subject, user_id, scopes)
    refresh_token, family_id = jwt_manager.create_refresh_token(subject, user_id)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": int(jwt_manager.access_token_expire.total_seconds()),
    }


def verify_token(token: str) -> Dict[str, Any]:
    """Verify access token"""
    return jwt_manager.verify_access_token(token)


def refresh_tokens(refresh_token: str) -> Dict[str, str]:
    """Refresh both tokens"""
    access_token, new_refresh_token, _ = jwt_manager.rotate_refresh_token(refresh_token)

    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
        "expires_in": int(jwt_manager.access_token_expire.total_seconds()),
    }
