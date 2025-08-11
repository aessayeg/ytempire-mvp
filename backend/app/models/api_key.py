"""
API Key Model for External Service Management
"""
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base


class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Key Information
    service = Column(String, nullable=False)  # openai, elevenlabs, youtube, etc.
    key_name = Column(String)
    encrypted_key = Column(String, nullable=False)
    
    # Usage Limits
    usage_limit = Column(JSON)  # {"daily": 1000, "monthly": 30000}
    current_usage = Column(JSON)  # {"daily": 100, "monthly": 2500}
    
    # Status
    is_active = Column(Boolean, default=True)
    is_valid = Column(Boolean, default=True)
    last_validated_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
