"""
Cost Tracking Model
"""
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
import uuid
from app.db.base import Base


class Cost(Base):
    __tablename__ = "costs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    video_id = Column(String, ForeignKey("videos.id"), nullable=True)
    
    # Cost Details
    service_type = Column(String, nullable=False)  # openai, elevenlabs, google-tts, etc.
    service_name = Column(String)  # gpt-4, claude-3, voice-synthesis, etc.
    operation = Column(String)  # script-generation, voice-synthesis, thumbnail, etc.
    
    # Metrics
    amount = Column(Float, nullable=False)
    tokens_used = Column(JSON)  # {"input": 1000, "output": 500}
    characters_used = Column(Float)
    api_calls = Column(Float, default=1)
    
    # Tracking
    request_id = Column(String)
    response_time_ms = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="costs")
    video = relationship("Video", back_populates="costs")