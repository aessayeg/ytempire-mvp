"""
Database Seed Data Script
Owner: Backend Team Lead
"""

import asyncio
import sys
from pathlib import Path
import uuid
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import AsyncSessionLocal, init_database
from app.models.user import User, SubscriptionTier
from app.models.channel import Channel
from app.models.video import Video, VideoStatus
from app.models.analytics import ChannelAnalytics, VideoAnalytics, CostTracking, ServiceType
from app.services.auth_service import AuthService
from app.repositories.user_repository import UserRepository
from app.schemas.auth import UserRegister


async def create_seed_users(session):
    """Create seed users."""
    print("Creating seed users...")
    
    user_repo = UserRepository(session)
    auth_service = AuthService(user_repo)
    
    users_data = [
        {
            "email": "admin@ytempire.com",
            "username": "admin",
            "password": "admin123456",
            "full_name": "YTEmpire Admin",
            "subscription_tier": "ENTERPRISE"
        },
        {
            "email": "demo@ytempire.com",
            "username": "demo",
            "password": "demo123456",
            "full_name": "Demo User",
            "subscription_tier": "PREMIUM"
        },
        {
            "email": "test@ytempire.com",
            "username": "testuser",
            "password": "test123456",
            "full_name": "Test User",
            "subscription_tier": "BASIC"
        },
        {
            "email": "free@ytempire.com",
            "username": "freeuser",
            "password": "free123456",
            "full_name": "Free User",
            "subscription_tier": "FREE"
        }
    ]
    
    created_users = []
    
    for user_data in users_data:
        try:
            # Check if user already exists
            existing_user = await user_repo.get_by_email(user_data["email"])
            if existing_user:
                print(f"User {user_data['email']} already exists, skipping...")
                created_users.append(existing_user)
                continue
            
            user_register = UserRegister(
                email=user_data["email"],
                username=user_data["username"],
                password=user_data["password"],
                full_name=user_data["full_name"],
                subscription_tier=user_data["subscription_tier"]
            )
            
            user = await auth_service.register_user(user_register)
            
            # Verify user if it's admin or demo
            if user_data["username"] in ["admin", "demo"]:
                await user_repo.verify_user(user.id)
            
            created_users.append(user)
            print(f"Created user: {user_data['email']}")
            
        except Exception as e:
            print(f"Error creating user {user_data['email']}: {str(e)}")
    
    return created_users


async def create_seed_channels(session, users):
    """Create seed channels."""
    print("Creating seed channels...")
    
    channels_data = [
        {
            "user": users[0],  # Admin
            "name": "YTEmpire Official",
            "description": "Official YTEmpire channel showcasing AI-powered video creation",
            "category": "technology",
            "target_audience": "content creators",
            "content_style": "professional",
            "upload_schedule": "daily"
        },
        {
            "user": users[1],  # Demo
            "name": "Tech Reviews Central",
            "description": "Latest technology reviews and unboxings",
            "category": "technology",
            "target_audience": "tech enthusiasts",
            "content_style": "informative",
            "upload_schedule": "3x_weekly"
        },
        {
            "user": users[1],  # Demo (second channel)
            "name": "Coding Tutorials Hub",
            "description": "Programming tutorials and coding tips",
            "category": "education",
            "target_audience": "developers",
            "content_style": "educational",
            "upload_schedule": "weekly"
        },
        {
            "user": users[2],  # Test
            "name": "Lifestyle & Wellness",
            "description": "Health, fitness, and lifestyle content",
            "category": "lifestyle",
            "target_audience": "health conscious",
            "content_style": "casual",
            "upload_schedule": "weekly"
        }
    ]
    
    created_channels = []
    
    for channel_data in channels_data:
        channel = Channel(
            id=str(uuid.uuid4()),
            user_id=channel_data["user"].id,
            name=channel_data["name"],
            description=channel_data["description"],
            category=channel_data["category"],
            target_audience=channel_data["target_audience"],
            content_style=channel_data["content_style"],
            upload_schedule=channel_data["upload_schedule"],
            branding={
                "primary_color": "#FF0000",
                "secondary_color": "#000000",
                "logo_url": f"https://example.com/logos/{channel_data['name'].lower().replace(' ', '_')}.png"
            },
            automation_settings={
                "auto_publish": True,
                "seo_optimization": True,
                "thumbnail_generation": True
            },
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        session.add(channel)
        created_channels.append(channel)
        print(f"Created channel: {channel_data['name']}")
    
    await session.commit()
    return created_channels


async def create_seed_videos(session, channels):
    """Create seed videos."""
    print("Creating seed videos...")
    
    videos_data = [
        {
            "channel": channels[0],
            "title": "Welcome to YTEmpire: AI-Powered Video Creation",
            "description": "Discover how YTEmpire revolutionizes content creation with AI automation",
            "status": VideoStatus.COMPLETED,
            "category": "introduction",
            "duration": 180
        },
        {
            "channel": channels[1],
            "title": "iPhone 15 Pro Review: Is It Worth the Upgrade?",
            "description": "Comprehensive review of Apple's latest flagship smartphone",
            "status": VideoStatus.COMPLETED,
            "category": "review",
            "duration": 420
        },
        {
            "channel": channels[1],
            "title": "M3 MacBook Pro Unboxing and First Impressions",
            "description": "Unboxing and initial thoughts on Apple's new M3 MacBook Pro",
            "status": VideoStatus.PROCESSING,
            "category": "unboxing",
            "duration": 300
        },
        {
            "channel": channels[2],
            "title": "Python FastAPI Tutorial: Build REST APIs in Minutes",
            "description": "Learn to build modern REST APIs using Python and FastAPI",
            "status": VideoStatus.COMPLETED,
            "category": "tutorial",
            "duration": 600
        },
        {
            "channel": channels[2],
            "title": "React Hooks Explained: useState and useEffect",
            "description": "Master React hooks with practical examples and best practices",
            "status": VideoStatus.PENDING,
            "category": "tutorial",
            "duration": 480
        },
        {
            "channel": channels[3],
            "title": "10-Minute Morning Workout Routine",
            "description": "Quick and effective morning exercises to start your day",
            "status": VideoStatus.COMPLETED,
            "category": "fitness",
            "duration": 600
        }
    ]
    
    created_videos = []
    
    for i, video_data in enumerate(videos_data):
        video = Video(
            id=str(uuid.uuid4()),
            channel_id=video_data["channel"].id,
            user_id=video_data["channel"].user_id,
            title=video_data["title"],
            description=video_data["description"],
            status=video_data["status"],
            category=video_data["category"],
            duration=video_data["duration"],
            priority=1,
            tags=["ytempire", video_data["category"], "ai-generated"],
            content_settings={
                "voice_type": "professional",
                "background_music": True,
                "captions": True
            },
            generation_settings={
                "model_version": "gpt-4",
                "voice_model": "elevenlabs-v1",
                "video_style": "modern"
            },
            metadata={
                "seed_data": True,
                "created_by": "seed_script"
            },
            total_cost=round(2.50 + (i * 0.25), 2),
            created_at=datetime.utcnow() - timedelta(days=i),
            updated_at=datetime.utcnow()
        )
        
        if video_data["status"] == VideoStatus.COMPLETED:
            video.youtube_video_id = f"yt_video_{i + 1}"
            video.published_at = datetime.utcnow() - timedelta(days=i)
            video.pipeline_completed_at = datetime.utcnow() - timedelta(days=i)
        
        session.add(video)
        created_videos.append(video)
        print(f"Created video: {video_data['title']}")
    
    await session.commit()
    return created_videos


async def create_seed_analytics(session, channels, videos):
    """Create seed analytics data."""
    print("Creating seed analytics...")
    
    # Channel analytics
    for i, channel in enumerate(channels):
        for days_ago in range(30):  # Last 30 days
            date = datetime.utcnow().date() - timedelta(days=days_ago)
            
            analytics = ChannelAnalytics(
                id=str(uuid.uuid4()),
                channel_id=channel.id,
                date=date,
                views=1000 + (i * 500) + (days_ago * 50),
                subscribers=5000 + (i * 1000) - (days_ago * 10),
                videos_published=1 if days_ago % 7 == 0 else 0,
                watch_time_minutes=8000 + (i * 2000) + (days_ago * 100),
                estimated_revenue=round(10.50 + (i * 5.25) + (days_ago * 0.75), 2),
                engagement_rate=0.05 + (i * 0.01),
                click_through_rate=0.08 + (i * 0.005),
                average_view_duration=120.0 + (i * 30),
                top_countries={"US": 45, "CA": 15, "UK": 12, "DE": 8, "AU": 7},
                age_demographics={"18-24": 25, "25-34": 35, "35-44": 20, "45-54": 15, "55+": 5},
                device_types={"mobile": 60, "desktop": 30, "tablet": 10},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            session.add(analytics)
    
    # Video analytics
    for i, video in enumerate(videos):
        if video.status == VideoStatus.COMPLETED:
            for days_ago in range(min(14, i + 1)):  # Up to 14 days of data
                date = datetime.utcnow().date() - timedelta(days=days_ago)
                
                analytics = VideoAnalytics(
                    id=str(uuid.uuid4()),
                    video_id=video.id,
                    date=date,
                    views=500 + (i * 200) - (days_ago * 25),
                    likes=25 + (i * 10) - (days_ago * 1),
                    dislikes=2 + (days_ago // 3),
                    comments=15 + (i * 5) - (days_ago * 1),
                    shares=5 + (i * 2),
                    watch_time_minutes=400 + (i * 150) - (days_ago * 20),
                    impressions=2000 + (i * 500) - (days_ago * 100),
                    click_through_rate=0.12 + (i * 0.01),
                    average_view_duration=video.duration * 0.6,  # 60% retention
                    estimated_revenue=round(2.75 + (i * 1.25) - (days_ago * 0.15), 2),
                    engagement_score=0.15 + (i * 0.02),
                    trending_score=0.25 + (i * 0.05),
                    traffic_sources={"search": 40, "suggested": 30, "external": 20, "direct": 10},
                    audience_retention={
                        "0-25%": 100,
                        "25-50%": 75,
                        "50-75%": 50,
                        "75-100%": 30
                    },
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                session.add(analytics)
    
    await session.commit()
    print("Created analytics data")


async def create_seed_cost_tracking(session, users, videos):
    """Create seed cost tracking data."""
    print("Creating seed cost tracking...")
    
    service_types = [
        ServiceType.OPENAI_GPT4,
        ServiceType.ELEVENLABS_TTS,
        ServiceType.CONTENT_GENERATION,
        ServiceType.AUDIO_SYNTHESIS,
        ServiceType.VIDEO_COMPILATION
    ]
    
    for video in videos:
        if video.status == VideoStatus.COMPLETED:
            # Create cost entries for each service type
            for service_type in service_types:
                cost = CostTracking(
                    id=str(uuid.uuid4()),
                    user_id=video.user_id,
                    video_id=video.id,
                    service_type=service_type,
                    operation=f"{service_type.value.lower()}_operation",
                    cost_amount=round(0.25 + (hash(str(video.id + service_type.value)) % 100) / 100, 2),
                    currency="USD",
                    usage_data={
                        "tokens": 1000,
                        "characters": 5000,
                        "duration_seconds": 180
                    },
                    billing_period="2024-01",
                    created_at=video.created_at
                )
                session.add(cost)
    
    await session.commit()
    print("Created cost tracking data")


async def main():
    """Main seed function."""
    print("Starting database seed...")
    
    # Initialize database
    await init_database()
    
    async with AsyncSessionLocal() as session:
        try:
            # Create seed data
            users = await create_seed_users(session)
            channels = await create_seed_channels(session, users)
            videos = await create_seed_videos(session, channels)
            await create_seed_analytics(session, channels, videos)
            await create_seed_cost_tracking(session, users, videos)
            
            print("\n✅ Database seeding completed successfully!")
            print(f"Created {len(users)} users")
            print(f"Created {len(channels)} channels") 
            print(f"Created {len(videos)} videos")
            print("Created analytics and cost tracking data")
            
        except Exception as e:
            await session.rollback()
            print(f"\n❌ Error during seeding: {str(e)}")
            raise
        finally:
            await session.close()


if __name__ == "__main__":
    asyncio.run(main())