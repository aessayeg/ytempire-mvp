"""
Test Data Generators using Factory Boy
Generates realistic test data for all models
"""
import factory
from factory import fuzzy
from faker import Faker
from datetime import datetime, timedelta
import random
import uuid
from typing import Dict, Any, List

fake = Faker()


class UserFactory(factory.Factory):
    """Factory for generating User test data"""

    class Meta:
        model = dict

    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    email = factory.LazyAttribute(lambda o: fake.unique.email())
    name = factory.LazyAttribute(lambda o: fake.name())
    password_hash = factory.LazyFunction(lambda: fake.sha256())
    is_active = True
    is_verified = factory.LazyAttribute(
        lambda o: random.choice([True, True, True, False])
    )
    is_admin = factory.LazyAttribute(
        lambda o: random.choice([False, False, False, True])
    )
    subscription_tier = factory.LazyAttribute(
        lambda o: random.choice(["free", "starter", "pro", "enterprise"])
    )
    api_key = factory.LazyFunction(lambda: fake.uuid4())
    created_at = factory.LazyAttribute(
        lambda o: fake.date_time_between(start_date="-1y", end_date="now")
    )
    updated_at = factory.LazyFunction(datetime.utcnow)

    @factory.post_generation
    def channels(self, create, extracted, **kwargs):
        if extracted:
            self.channels = extracted
        else:
            self.channels = ChannelFactory.create_batch(random.randint(1, 3))


class ChannelFactory(factory.Factory):
    """Factory for generating Channel test data"""

    class Meta:
        model = dict

    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    user_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    name = factory.LazyAttribute(
        lambda o: f"{fake.catch_phrase()} {random.choice(['Channel', 'TV', 'Show', 'Stream'])}"
    )
    youtube_channel_id = factory.LazyAttribute(lambda o: f"UC{fake.lexify('?' * 22)}")
    description = factory.LazyAttribute(lambda o: fake.text(max_nb_chars=200))
    category = factory.LazyAttribute(
        lambda o: random.choice(
            [
                "Technology",
                "Gaming",
                "Education",
                "Entertainment",
                "Music",
                "Sports",
                "News",
                "Comedy",
                "Science",
                "Travel",
            ]
        )
    )
    subscriber_count = factory.LazyAttribute(lambda o: random.randint(100, 1000000))
    video_count = factory.LazyAttribute(lambda o: random.randint(10, 1000))
    view_count = factory.LazyAttribute(lambda o: random.randint(10000, 100000000))
    is_monetized = factory.LazyAttribute(lambda o: random.choice([True, True, False]))
    is_active = True
    auto_upload = factory.LazyAttribute(lambda o: random.choice([True, False]))
    upload_schedule = factory.LazyAttribute(
        lambda o: random.choice(
            [
                "Daily at 2:00 PM",
                "Mon, Wed, Fri at 10:00 AM",
                "Weekly on Sunday",
                "Twice daily at 9:00 AM and 6:00 PM",
            ]
        )
    )
    api_quota_used = factory.LazyAttribute(lambda o: random.randint(0, 9000))
    api_quota_limit = 10000
    created_at = factory.LazyAttribute(
        lambda o: fake.date_time_between(start_date="-6m", end_date="now")
    )


class VideoFactory(factory.Factory):
    """Factory for generating Video test data"""

    class Meta:
        model = dict

    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    channel_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    user_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    title = factory.LazyAttribute(
        lambda o: fake.sentence(nb_words=random.randint(5, 10)).rstrip(".")
    )
    description = factory.LazyAttribute(lambda o: fake.text(max_nb_chars=500))
    script = factory.LazyAttribute(lambda o: fake.text(max_nb_chars=2000))
    youtube_video_id = factory.LazyAttribute(lambda o: fake.lexify("?" * 11))
    status = factory.LazyAttribute(
        lambda o: random.choice(
            [
                "queued",
                "processing",
                "completed",
                "failed",
                "scheduled",
                "uploaded",
                "draft",
            ]
        )
    )
    progress = factory.LazyAttribute(
        lambda o: 100 if o.status == "completed" else random.randint(0, 99)
    )
    duration_seconds = factory.LazyAttribute(lambda o: random.randint(60, 1800))
    thumbnail_url = factory.LazyAttribute(lambda o: fake.image_url())
    video_url = factory.LazyAttribute(lambda o: fake.url())

    # Metrics
    view_count = factory.LazyAttribute(lambda o: random.randint(0, 1000000))
    like_count = factory.LazyAttribute(lambda o: random.randint(0, 50000))
    comment_count = factory.LazyAttribute(lambda o: random.randint(0, 5000))
    estimated_revenue = factory.LazyAttribute(
        lambda o: round(random.uniform(0, 1000), 2)
    )

    # Scheduling
    scheduled_date = factory.LazyAttribute(
        lambda o: fake.future_datetime(end_date="+30d")
        if o.status == "scheduled"
        else None
    )
    uploaded_at = factory.LazyAttribute(
        lambda o: fake.past_datetime(start_date="-7d")
        if o.status == "uploaded"
        else None
    )

    # Processing details
    processing_stage = factory.LazyAttribute(
        lambda o: random.choice(
            [
                "Script generation",
                "Voice synthesis",
                "Video editing",
                "Thumbnail creation",
                "Final rendering",
                "Quality check",
            ]
        )
        if o.status == "processing"
        else None
    )
    error_message = factory.LazyAttribute(
        lambda o: fake.sentence() if o.status == "failed" else None
    )

    # Tags and metadata
    tags = factory.LazyAttribute(
        lambda o: [fake.word() for _ in range(random.randint(3, 10))]
    )
    language = factory.LazyAttribute(
        lambda o: random.choice(["en", "es", "fr", "de", "ja"])
    )

    created_at = factory.LazyAttribute(
        lambda o: fake.date_time_between(start_date="-30d", end_date="now")
    )
    updated_at = factory.LazyFunction(datetime.utcnow)


class CostFactory(factory.Factory):
    """Factory for generating Cost tracking test data"""

    class Meta:
        model = dict

    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    user_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    video_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    service_type = factory.LazyAttribute(
        lambda o: random.choice(["openai", "elevenlabs", "google", "aws", "azure"])
    )
    service_name = factory.LazyAttribute(
        lambda o: {
            "openai": random.choice(["gpt-4", "gpt-3.5-turbo", "dall-e-3"]),
            "elevenlabs": "voice-synthesis",
            "google": random.choice(["translate", "tts", "vision"]),
            "aws": random.choice(["s3", "ec2", "lambda"]),
            "azure": random.choice(["cognitive-services", "storage"]),
        }.get(o.service_type, "unknown")
    )
    operation = factory.LazyAttribute(
        lambda o: random.choice(
            [
                "script-generation",
                "voice-synthesis",
                "thumbnail-generation",
                "video-processing",
                "translation",
                "transcription",
            ]
        )
    )
    amount = factory.LazyAttribute(lambda o: round(random.uniform(0.01, 10.0), 4))
    units = factory.LazyAttribute(lambda o: random.randint(1, 10000))
    unit_cost = factory.LazyAttribute(
        lambda o: o.amount / o.units if o.units > 0 else 0
    )
    tokens_used = factory.LazyAttribute(
        lambda o: {
            "input": random.randint(100, 5000),
            "output": random.randint(100, 2000),
        }
        if "gpt" in o.service_name
        else None
    )
    characters_used = factory.LazyAttribute(
        lambda o: random.randint(100, 10000) if o.service_type == "elevenlabs" else None
    )
    api_calls = factory.LazyAttribute(lambda o: random.randint(1, 10))
    request_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    response_time_ms = factory.LazyAttribute(lambda o: random.randint(50, 5000))
    created_at = factory.LazyAttribute(
        lambda o: fake.date_time_between(start_date="-7d", end_date="now")
    )


class AnalyticsFactory(factory.Factory):
    """Factory for generating Analytics test data"""

    class Meta:
        model = dict

    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    video_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    channel_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    date = factory.LazyAttribute(
        lambda o: fake.date_between(start_date="-30d", end_date="today")
    )

    # Metrics
    views = factory.LazyAttribute(lambda o: random.randint(0, 100000))
    watch_time_minutes = factory.LazyAttribute(lambda o: random.randint(0, 50000))
    average_view_duration = factory.LazyAttribute(lambda o: random.uniform(0.1, 15.0))
    likes = factory.LazyAttribute(
        lambda o: random.randint(0, o.views // 10 if o.views > 0 else 0)
    )
    dislikes = factory.LazyAttribute(
        lambda o: random.randint(0, o.views // 100 if o.views > 0 else 0)
    )
    comments = factory.LazyAttribute(
        lambda o: random.randint(0, o.views // 20 if o.views > 0 else 0)
    )
    shares = factory.LazyAttribute(
        lambda o: random.randint(0, o.views // 50 if o.views > 0 else 0)
    )

    # Demographics
    age_groups = factory.LazyAttribute(
        lambda o: {
            "13-17": random.randint(5, 15),
            "18-24": random.randint(20, 35),
            "25-34": random.randint(25, 40),
            "35-44": random.randint(10, 25),
            "45-54": random.randint(5, 15),
            "55+": random.randint(5, 15),
        }
    )
    gender_distribution = factory.LazyAttribute(
        lambda o: {
            "male": random.randint(40, 60),
            "female": random.randint(35, 55),
            "other": random.randint(0, 5),
        }
    )

    # Traffic sources
    traffic_sources = factory.LazyAttribute(
        lambda o: {
            "search": random.randint(20, 40),
            "suggested": random.randint(30, 50),
            "browse": random.randint(10, 25),
            "external": random.randint(5, 15),
            "direct": random.randint(5, 15),
        }
    )

    # Revenue
    estimated_revenue = factory.LazyAttribute(
        lambda o: round(o.views * random.uniform(0.001, 0.01), 2)
    )
    ad_impressions = factory.LazyAttribute(lambda o: o.views * random.randint(1, 3))
    cpm = factory.LazyAttribute(lambda o: round(random.uniform(1.0, 10.0), 2))

    created_at = factory.LazyFunction(datetime.utcnow)


class TestDataGenerator:
    """Main class for generating test data"""

    @staticmethod
    def generate_user(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a single user with optional overrides"""
        return UserFactory.create(**(overrides or {}))

    @staticmethod
    def generate_users(count: int = 10) -> List[Dict[str, Any]]:
        """Generate multiple users"""
        return UserFactory.create_batch(count)

    @staticmethod
    def generate_channel(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a single channel"""
        return ChannelFactory.create(**(overrides or {}))

    @staticmethod
    def generate_channels(count: int = 5) -> List[Dict[str, Any]]:
        """Generate multiple channels"""
        return ChannelFactory.create_batch(count)

    @staticmethod
    def generate_video(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a single video"""
        return VideoFactory.create(**(overrides or {}))

    @staticmethod
    def generate_videos(count: int = 20) -> List[Dict[str, Any]]:
        """Generate multiple videos"""
        return VideoFactory.create_batch(count)

    @staticmethod
    def generate_cost_record(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a single cost record"""
        return CostFactory.create(**(overrides or {}))

    @staticmethod
    def generate_cost_records(count: int = 50) -> List[Dict[str, Any]]:
        """Generate multiple cost records"""
        return CostFactory.create_batch(count)

    @staticmethod
    def generate_analytics(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate analytics data"""
        return AnalyticsFactory.create(**(overrides or {}))

    @staticmethod
    def generate_complete_dataset() -> Dict[str, Any]:
        """Generate a complete dataset for testing"""
        users = UserFactory.create_batch(5)
        channels = []
        videos = []
        costs = []
        analytics = []

        for user in users:
            # Create channels for each user
            user_channels = ChannelFactory.create_batch(
                random.randint(1, 3), user_id=user["id"]
            )
            channels.extend(user_channels)

            # Create videos for each channel
            for channel in user_channels:
                channel_videos = VideoFactory.create_batch(
                    random.randint(5, 15), channel_id=channel["id"], user_id=user["id"]
                )
                videos.extend(channel_videos)

                # Create costs for each video
                for video in channel_videos:
                    video_costs = CostFactory.create_batch(
                        random.randint(3, 7), video_id=video["id"], user_id=user["id"]
                    )
                    costs.extend(video_costs)

                    # Create analytics for completed videos
                    if video["status"] == "completed":
                        video_analytics = AnalyticsFactory.create_batch(
                            random.randint(7, 30),
                            video_id=video["id"],
                            channel_id=channel["id"],
                        )
                        analytics.extend(video_analytics)

        return {
            "users": users,
            "channels": channels,
            "videos": videos,
            "costs": costs,
            "analytics": analytics,
        }

    @staticmethod
    def generate_time_series_data(days: int = 30) -> List[Dict[str, Any]]:
        """Generate time series data for testing charts and trends"""
        data = []
        base_date = datetime.utcnow() - timedelta(days=days)

        for i in range(days):
            date = base_date + timedelta(days=i)
            data.append(
                {
                    "date": date.isoformat(),
                    "views": random.randint(10000, 100000) + i * 100,
                    "revenue": round(random.uniform(50, 500) + i * 2, 2),
                    "costs": round(random.uniform(20, 100), 2),
                    "videos_created": random.randint(1, 10),
                    "new_subscribers": random.randint(50, 500),
                    "engagement_rate": round(random.uniform(2, 15), 2),
                }
            )

        return data


# Singleton instance
test_data_generator = TestDataGenerator()
