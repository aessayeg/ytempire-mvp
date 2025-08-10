"""
Database Base Class
"""
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Import all models here for Alembic
from app.models.user import User
from app.models.channel import Channel
from app.models.video import Video
from app.models.cost import Cost
from app.models.analytics import Analytics