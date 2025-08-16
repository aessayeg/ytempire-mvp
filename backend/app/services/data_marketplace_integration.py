"""
Data Marketplace Integration Service
Enables integration with external data marketplaces and data exchange platforms
"""

import json
import logging
import hashlib
import hmac
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import asyncio
import uuid
from decimal import Decimal

import httpx
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update

from app.core.config import settings
from app.models.channel import Channel
from app.models.video import Video

logger = logging.getLogger(__name__)


class MarketplaceProvider(Enum):
    """Supported data marketplace providers"""

    AWS_DATA_EXCHANGE = "aws_data_exchange"
    AZURE_MARKETPLACE = "azure_marketplace"
    GOOGLE_ANALYTICS_HUB = "google_analytics_hub"
    SNOWFLAKE_MARKETPLACE = "snowflake_marketplace"
    DATABRICKS_MARKETPLACE = "databricks_marketplace"
    KAGGLE = "kaggle"
    QUANDL = "quandl"
    RAPID_API = "rapid_api"
    DATA_WORLD = "data_world"
    SOCRATA = "socrata"


class DataCategory(Enum):
    """Categories of data available in marketplaces"""

    SOCIAL_MEDIA = "social_media"
    VIDEO_ANALYTICS = "video_analytics"
    TRENDING_TOPICS = "trending_topics"
    COMPETITOR_DATA = "competitor_data"
    AUDIENCE_INSIGHTS = "audience_insights"
    MARKET_RESEARCH = "market_research"
    SEO_DATA = "seo_data"
    ADVERTISING_DATA = "advertising_data"
    DEMOGRAPHIC_DATA = "demographic_data"
    CONTENT_PERFORMANCE = "content_performance"


class DataFormat(Enum):
    """Supported data formats"""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    AVRO = "avro"
    API = "api"
    STREAMING = "streaming"
    BATCH = "batch"


@dataclass
class DataProduct:
    """Represents a data product in the marketplace"""

    id: str
    name: str
    provider: MarketplaceProvider
    category: DataCategory
    description: str
    price_model: str  # 'free', 'subscription', 'pay_per_use', 'one_time'
    price: Decimal
    format: DataFormat
    update_frequency: str  # 'real_time', 'hourly', 'daily', 'weekly', 'monthly'
    data_schema: Dict[str, Any]
    sample_data: Optional[Dict[str, Any]] = None
    rating: float = 0.0
    reviews_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class DataSubscription:
    """Represents a subscription to a data product"""

    id: str
    product_id: str
    provider: MarketplaceProvider
    status: str  # 'active', 'paused', 'cancelled', 'expired'
    start_date: datetime
    end_date: Optional[datetime]
    usage_limit: Optional[int]
    usage_count: int = 0
    last_sync: Optional[datetime] = None
    credentials: Dict[str, str] = field(default_factory=dict)


@dataclass
class DataTransaction:
    """Represents a data transaction"""

    id: str
    subscription_id: str
    timestamp: datetime
    data_size: int  # bytes
    records_count: int
    cost: Decimal
    status: str  # 'pending', 'completed', 'failed'
    error_message: Optional[str] = None


class DataMarketplaceIntegration:
    """Service for integrating with data marketplaces"""

    def __init__(self):
        self.providers: Dict[MarketplaceProvider, Dict[str, Any]] = {}
        self.products: Dict[str, DataProduct] = {}
        self.subscriptions: Dict[str, DataSubscription] = {}
        self.transactions: List[DataTransaction] = []
        self.data_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._initialize_providers()
        self._load_sample_products()

    def _initialize_providers(self):
        """Initialize marketplace provider configurations"""
        # AWS Data Exchange
        self.providers[MarketplaceProvider.AWS_DATA_EXCHANGE] = {
            "base_url": "https://dataexchange.amazonaws.com",
            "auth_type": "aws_signature",
            "credentials": {
                "access_key": settings.AWS_ACCESS_KEY
                if hasattr(settings, "AWS_ACCESS_KEY")
                else "",
                "secret_key": settings.AWS_SECRET_KEY
                if hasattr(settings, "AWS_SECRET_KEY")
                else "",
                "region": "us-east-1",
            },
            "rate_limit": 100,  # requests per minute
        }

        # Google Analytics Hub
        self.providers[MarketplaceProvider.GOOGLE_ANALYTICS_HUB] = {
            "base_url": "https://analyticshub.googleapis.com",
            "auth_type": "oauth2",
            "credentials": {
                "client_id": settings.GOOGLE_CLIENT_ID
                if hasattr(settings, "GOOGLE_CLIENT_ID")
                else "",
                "client_secret": settings.GOOGLE_CLIENT_SECRET
                if hasattr(settings, "GOOGLE_CLIENT_SECRET")
                else "",
            },
            "rate_limit": 200,
        }

        # Rapid API
        self.providers[MarketplaceProvider.RAPID_API] = {
            "base_url": "https://rapidapi.com",
            "auth_type": "api_key",
            "credentials": {
                "api_key": settings.RAPID_API_KEY
                if hasattr(settings, "RAPID_API_KEY")
                else ""
            },
            "rate_limit": 500,
        }

        # Snowflake Marketplace
        self.providers[MarketplaceProvider.SNOWFLAKE_MARKETPLACE] = {
            "base_url": "https://marketplace.snowflake.com",
            "auth_type": "snowflake",
            "credentials": {
                "account": settings.SNOWFLAKE_ACCOUNT
                if hasattr(settings, "SNOWFLAKE_ACCOUNT")
                else "",
                "username": settings.SNOWFLAKE_USERNAME
                if hasattr(settings, "SNOWFLAKE_USERNAME")
                else "",
                "password": settings.SNOWFLAKE_PASSWORD
                if hasattr(settings, "SNOWFLAKE_PASSWORD")
                else "",
            },
            "rate_limit": 150,
        }

    def _load_sample_products(self):
        """Load sample data products"""
        # YouTube Analytics Data
        self.register_product(
            DataProduct(
                id="yt_analytics_001",
                name="YouTube Channel Analytics Pro",
                provider=MarketplaceProvider.RAPID_API,
                category=DataCategory.VIDEO_ANALYTICS,
                description="Comprehensive YouTube channel and video analytics including engagement metrics",
                price_model="subscription",
                price=Decimal("99.99"),
                format=DataFormat.API,
                update_frequency="hourly",
                data_schema={
                    "channel_id": "string",
                    "subscribers": "integer",
                    "total_views": "integer",
                    "avg_view_duration": "float",
                    "engagement_rate": "float",
                    "top_videos": "array",
                },
                rating=4.5,
                reviews_count=234,
            )
        )

        # Trending Topics Data
        self.register_product(
            DataProduct(
                id="trends_001",
                name="Global Trending Topics Feed",
                provider=MarketplaceProvider.AWS_DATA_EXCHANGE,
                category=DataCategory.TRENDING_TOPICS,
                description="Real-time trending topics across social media platforms",
                price_model="pay_per_use",
                price=Decimal("0.01"),  # per request
                format=DataFormat.STREAMING,
                update_frequency="real_time",
                data_schema={
                    "topic": "string",
                    "platform": "string",
                    "trend_score": "float",
                    "volume": "integer",
                    "sentiment": "float",
                    "related_keywords": "array",
                },
                rating=4.8,
                reviews_count=512,
            )
        )

        # Competitor Analysis Data
        self.register_product(
            DataProduct(
                id="competitor_001",
                name="YouTube Competitor Intelligence",
                provider=MarketplaceProvider.SNOWFLAKE_MARKETPLACE,
                category=DataCategory.COMPETITOR_DATA,
                description="Detailed competitor channel analysis and benchmarking data",
                price_model="subscription",
                price=Decimal("299.99"),
                format=DataFormat.BATCH,
                update_frequency="daily",
                data_schema={
                    "competitor_channel_id": "string",
                    "channel_name": "string",
                    "content_strategy": "object",
                    "posting_schedule": "array",
                    "engagement_metrics": "object",
                    "growth_rate": "float",
                },
                rating=4.6,
                reviews_count=89,
            )
        )

        # Audience Insights Data
        self.register_product(
            DataProduct(
                id="audience_001",
                name="YouTube Audience Demographics & Psychographics",
                provider=MarketplaceProvider.GOOGLE_ANALYTICS_HUB,
                category=DataCategory.AUDIENCE_INSIGHTS,
                description="Detailed audience analysis including demographics and interests",
                price_model="subscription",
                price=Decimal("149.99"),
                format=DataFormat.API,
                update_frequency="weekly",
                data_schema={
                    "channel_id": "string",
                    "demographics": {
                        "age_groups": "object",
                        "gender": "object",
                        "location": "object",
                        "devices": "object",
                    },
                    "interests": "array",
                    "viewing_patterns": "object",
                },
                rating=4.7,
                reviews_count=167,
            )
        )

        # SEO Data
        self.register_product(
            DataProduct(
                id="seo_001",
                name="Video SEO Optimization Data",
                provider=MarketplaceProvider.RAPID_API,
                category=DataCategory.SEO_DATA,
                description="SEO metrics and recommendations for video content",
                price_model="pay_per_use",
                price=Decimal("0.05"),
                format=DataFormat.JSON,
                update_frequency="daily",
                data_schema={
                    "keyword": "string",
                    "search_volume": "integer",
                    "competition": "float",
                    "cpc": "float",
                    "related_keywords": "array",
                    "trending_score": "float",
                },
                rating=4.4,
                reviews_count=321,
            )
        )

    def register_product(self, product: DataProduct):
        """Register a new data product"""
        self.products[product.id] = product
        logger.info(f"Registered data product: {product.name} (ID: {product.id})")

    async def browse_products(
        self,
        category: Optional[DataCategory] = None,
        provider: Optional[MarketplaceProvider] = None,
        price_range: Optional[Tuple[Decimal, Decimal]] = None,
    ) -> List[DataProduct]:
        """Browse available data products"""
        products = list(self.products.values())

        # Filter by category
        if category:
            products = [p for p in products if p.category == category]

        # Filter by provider
        if provider:
            products = [p for p in products if p.provider == provider]

        # Filter by price range
        if price_range:
            min_price, max_price = price_range
            products = [p for p in products if min_price <= p.price <= max_price]

        # Sort by rating
        products.sort(key=lambda x: x.rating, reverse=True)

        return products

    async def subscribe_to_product(
        self, product_id: str, duration_days: Optional[int] = None
    ) -> str:
        """Subscribe to a data product"""
        if product_id not in self.products:
            raise ValueError(f"Product {product_id} not found")

        product = self.products[product_id]
        subscription_id = str(uuid.uuid4())

        # Create subscription
        subscription = DataSubscription(
            id=subscription_id,
            product_id=product_id,
            provider=product.provider,
            status="active",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=duration_days)
            if duration_days
            else None,
            usage_limit=1000 if product.price_model == "pay_per_use" else None,
        )

        # Store credentials based on provider
        if product.provider in self.providers:
            subscription.credentials = self.providers[product.provider].get(
                "credentials", {}
            )

        self.subscriptions[subscription_id] = subscription

        logger.info(
            f"Created subscription {subscription_id} for product {product.name}"
        )
        return subscription_id

    async def fetch_data(
        self,
        subscription_id: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fetch data from a subscribed product"""
        if subscription_id not in self.subscriptions:
            raise ValueError(f"Subscription {subscription_id} not found")

        subscription = self.subscriptions[subscription_id]

        # Check subscription status
        if subscription.status != "active":
            raise ValueError(f"Subscription {subscription_id} is not active")

        # Check usage limits
        if (
            subscription.usage_limit
            and subscription.usage_count >= subscription.usage_limit
        ):
            raise ValueError(f"Usage limit exceeded for subscription {subscription_id}")

        product = self.products[subscription.product_id]

        # Check cache
        cache_key = f"{subscription_id}:{json.dumps(filters or {})}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        # Fetch data based on product type
        data = await self._fetch_product_data(product, filters, limit)

        # Record transaction
        transaction = DataTransaction(
            id=str(uuid.uuid4()),
            subscription_id=subscription_id,
            timestamp=datetime.now(),
            data_size=len(json.dumps(data)),
            records_count=len(data.get("records", [])),
            cost=self._calculate_cost(product, len(data.get("records", []))),
            status="completed",
        )
        self.transactions.append(transaction)

        # Update subscription usage
        subscription.usage_count += 1
        subscription.last_sync = datetime.now()

        # Cache data
        self._cache_data(cache_key, data)

        return data

    async def _fetch_product_data(
        self,
        product: DataProduct,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fetch data for a specific product"""
        # Simulate data fetching based on product category
        if product.category == DataCategory.VIDEO_ANALYTICS:
            return await self._fetch_video_analytics(filters, limit)
        elif product.category == DataCategory.TRENDING_TOPICS:
            return await self._fetch_trending_topics(filters, limit)
        elif product.category == DataCategory.COMPETITOR_DATA:
            return await self._fetch_competitor_data(filters, limit)
        elif product.category == DataCategory.AUDIENCE_INSIGHTS:
            return await self._fetch_audience_insights(filters, limit)
        elif product.category == DataCategory.SEO_DATA:
            return await self._fetch_seo_data(filters, limit)
        else:
            return {"records": [], "metadata": {"source": product.name}}

    async def _fetch_video_analytics(
        self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Fetch video analytics data"""
        limit = limit or 100
        records = []

        for i in range(limit):
            records.append(
                {
                    "channel_id": f"UC{uuid.uuid4().hex[:22]}",
                    "channel_name": f"Channel_{i+1}",
                    "subscribers": np.random.randint(1000, 1000000),
                    "total_views": np.random.randint(10000, 10000000),
                    "avg_view_duration": np.random.uniform(60, 600),
                    "engagement_rate": np.random.uniform(0.01, 0.15),
                    "top_videos": [
                        {
                            "video_id": f"vid_{j}",
                            "title": f"Video Title {j}",
                            "views": np.random.randint(1000, 100000),
                        }
                        for j in range(5)
                    ],
                    "growth_rate": np.random.uniform(-0.1, 0.3),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return {
            "records": records,
            "metadata": {
                "source": "YouTube Analytics Pro",
                "update_time": datetime.now().isoformat(),
                "record_count": len(records),
            },
        }

    async def _fetch_trending_topics(
        self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Fetch trending topics data"""
        limit = limit or 50
        topics = [
            "AI Technology",
            "Climate Change",
            "Space Exploration",
            "Crypto Currency",
            "Mental Health",
            "Remote Work",
            "Electric Vehicles",
            "Sustainable Living",
            "Gaming",
            "Fashion Trends",
            "Food Recipes",
            "Travel Destinations",
            "Fitness Tips",
            "Music Releases",
            "Movie Reviews",
            "Tech Reviews",
        ]

        records = []
        for i in range(min(limit, len(topics))):
            records.append(
                {
                    "topic": topics[i],
                    "platform": np.random.choice(
                        ["youtube", "twitter", "tiktok", "instagram"]
                    ),
                    "trend_score": np.random.uniform(60, 100),
                    "volume": np.random.randint(1000, 1000000),
                    "sentiment": np.random.uniform(-1, 1),
                    "related_keywords": [f"keyword_{j}" for j in range(5)],
                    "geo_distribution": {
                        "US": np.random.uniform(0.2, 0.4),
                        "UK": np.random.uniform(0.1, 0.2),
                        "CA": np.random.uniform(0.05, 0.15),
                        "AU": np.random.uniform(0.05, 0.1),
                        "Other": np.random.uniform(0.3, 0.5),
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return {
            "records": records,
            "metadata": {
                "source": "Global Trending Topics Feed",
                "update_time": datetime.now().isoformat(),
                "platforms_covered": ["youtube", "twitter", "tiktok", "instagram"],
            },
        }

    async def _fetch_competitor_data(
        self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Fetch competitor analysis data"""
        limit = limit or 20
        records = []

        for i in range(limit):
            records.append(
                {
                    "competitor_channel_id": f"UC{uuid.uuid4().hex[:22]}",
                    "channel_name": f"Competitor_{i+1}",
                    "content_strategy": {
                        "primary_category": np.random.choice(
                            ["Education", "Entertainment", "Gaming", "Tech"]
                        ),
                        "video_length_avg": np.random.uniform(5, 20),
                        "posting_frequency": np.random.uniform(1, 7),
                        "content_types": ["tutorials", "reviews", "vlogs"],
                    },
                    "posting_schedule": {
                        "best_days": ["Tuesday", "Thursday", "Saturday"],
                        "best_times": ["10:00", "14:00", "19:00"],
                    },
                    "engagement_metrics": {
                        "avg_views": np.random.randint(5000, 500000),
                        "avg_likes": np.random.randint(100, 10000),
                        "avg_comments": np.random.randint(10, 1000),
                        "engagement_rate": np.random.uniform(0.02, 0.12),
                    },
                    "growth_rate": np.random.uniform(-0.05, 0.25),
                    "strengths": ["consistent posting", "high quality", "good SEO"],
                    "weaknesses": ["low engagement", "poor thumbnails"],
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return {
            "records": records,
            "metadata": {
                "source": "YouTube Competitor Intelligence",
                "analysis_depth": "comprehensive",
                "update_frequency": "daily",
            },
        }

    async def _fetch_audience_insights(
        self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Fetch audience insights data"""
        records = [
            {
                "channel_id": filters.get("channel_id", "UC_default"),
                "demographics": {
                    "age_groups": {
                        "13-17": 0.08,
                        "18-24": 0.25,
                        "25-34": 0.35,
                        "35-44": 0.20,
                        "45-54": 0.08,
                        "55+": 0.04,
                    },
                    "gender": {"male": 0.60, "female": 0.38, "other": 0.02},
                    "location": {
                        "US": 0.35,
                        "UK": 0.15,
                        "CA": 0.10,
                        "AU": 0.08,
                        "IN": 0.12,
                        "Other": 0.20,
                    },
                    "devices": {
                        "mobile": 0.65,
                        "desktop": 0.25,
                        "tablet": 0.08,
                        "tv": 0.02,
                    },
                },
                "interests": [
                    "Technology",
                    "Gaming",
                    "Education",
                    "Entertainment",
                    "Sports",
                    "Music",
                    "Travel",
                    "Food",
                ],
                "viewing_patterns": {
                    "peak_hours": ["20:00-22:00", "12:00-13:00"],
                    "avg_session_duration": 25.5,
                    "videos_per_session": 3.2,
                    "return_rate": 0.68,
                },
                "psychographics": {
                    "personality_traits": ["curious", "tech-savvy", "social"],
                    "values": ["innovation", "creativity", "community"],
                    "lifestyle": ["digital-first", "mobile-centric"],
                },
                "timestamp": datetime.now().isoformat(),
            }
        ]

        return {
            "records": records,
            "metadata": {
                "source": "YouTube Audience Demographics & Psychographics",
                "confidence_score": 0.85,
                "sample_size": 10000,
            },
        }

    async def _fetch_seo_data(
        self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Fetch SEO data"""
        limit = limit or 100
        keywords = [
            "how to",
            "tutorial",
            "review",
            "best",
            "top 10",
            "guide",
            "tips",
            "tricks",
            "explained",
            "vs",
            "comparison",
            "unboxing",
            "reaction",
            "challenge",
        ]

        records = []
        for i in range(min(limit, len(keywords) * 10)):
            base_keyword = np.random.choice(keywords)
            records.append(
                {
                    "keyword": f'{base_keyword} {np.random.choice(["2024", "ultimate", "beginner", "advanced"])}',
                    "search_volume": np.random.randint(1000, 100000),
                    "competition": np.random.uniform(0.1, 1.0),
                    "cpc": np.random.uniform(0.1, 5.0),
                    "difficulty": np.random.uniform(0.1, 1.0),
                    "related_keywords": [
                        f"{base_keyword} {suffix}"
                        for suffix in [
                            "tips",
                            "guide",
                            "tutorial",
                            "examples",
                            "best practices",
                        ]
                    ][:3],
                    "trending_score": np.random.uniform(0, 100),
                    "seasonal_trend": {
                        "Jan": np.random.uniform(0.7, 1.0),
                        "Feb": np.random.uniform(0.7, 1.0),
                        "Mar": np.random.uniform(0.7, 1.0),
                        "Apr": np.random.uniform(0.7, 1.0),
                        "May": np.random.uniform(0.7, 1.0),
                        "Jun": np.random.uniform(0.7, 1.0),
                        "Jul": np.random.uniform(0.7, 1.0),
                        "Aug": np.random.uniform(0.7, 1.0),
                        "Sep": np.random.uniform(0.7, 1.0),
                        "Oct": np.random.uniform(0.7, 1.0),
                        "Nov": np.random.uniform(0.7, 1.0),
                        "Dec": np.random.uniform(0.7, 1.0),
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return {
            "records": records,
            "metadata": {
                "source": "Video SEO Optimization Data",
                "regions": ["US", "UK", "CA", "AU"],
                "language": "en",
            },
        }

    def _calculate_cost(self, product: DataProduct, record_count: int) -> Decimal:
        """Calculate cost for data usage"""
        if product.price_model == "pay_per_use":
            return product.price * Decimal(record_count)
        elif product.price_model == "subscription":
            return Decimal("0")  # Already paid via subscription
        else:
            return product.price

    def _get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if available"""
        if cache_key in self.data_cache:
            data, cached_time = self.data_cache[cache_key]
            # Cache valid for 1 hour
            if datetime.now() - cached_time < timedelta(hours=1):
                return data
        return None

    def _cache_data(self, cache_key: str, data: Dict[str, Any]):
        """Cache fetched data"""
        self.data_cache[cache_key] = (data, datetime.now())

        # Clean old cache entries
        cutoff_time = datetime.now() - timedelta(hours=2)
        self.data_cache = {
            k: v for k, v in self.data_cache.items() if v[1] > cutoff_time
        }

    async def sync_to_warehouse(
        self, subscription_id: str, warehouse_config: Dict[str, Any]
    ) -> bool:
        """Sync marketplace data to data warehouse"""
        try:
            # Fetch latest data
            data = await self.fetch_data(subscription_id)

            # Transform data for warehouse
            df = pd.DataFrame(data["records"])

            # Add metadata columns
            df["source_subscription"] = subscription_id
            df["sync_timestamp"] = datetime.now()
            df["data_version"] = data["metadata"].get("version", "1.0")

            # Simulate warehouse sync (would actually write to warehouse)
            logger.info(
                f"Synced {len(df)} records to warehouse for subscription {subscription_id}"
            )

            return True
        except Exception as e:
            logger.error(f"Failed to sync to warehouse: {e}")
            return False

    def get_subscription_status(self, subscription_id: str) -> Dict[str, Any]:
        """Get subscription status and usage"""
        if subscription_id not in self.subscriptions:
            raise ValueError(f"Subscription {subscription_id} not found")

        subscription = self.subscriptions[subscription_id]
        product = self.products[subscription.product_id]

        # Calculate usage percentage
        usage_pct = 0
        if subscription.usage_limit:
            usage_pct = (subscription.usage_count / subscription.usage_limit) * 100

        # Calculate costs
        total_cost = sum(
            t.cost
            for t in self.transactions
            if t.subscription_id == subscription_id and t.status == "completed"
        )

        return {
            "subscription_id": subscription_id,
            "product_name": product.name,
            "status": subscription.status,
            "start_date": subscription.start_date.isoformat(),
            "end_date": subscription.end_date.isoformat()
            if subscription.end_date
            else None,
            "usage_count": subscription.usage_count,
            "usage_limit": subscription.usage_limit,
            "usage_percentage": usage_pct,
            "total_cost": float(total_cost),
            "last_sync": subscription.last_sync.isoformat()
            if subscription.last_sync
            else None,
        }

    def get_marketplace_analytics(self) -> Dict[str, Any]:
        """Get analytics on marketplace usage"""
        # Calculate metrics
        total_subscriptions = len(self.subscriptions)
        active_subscriptions = sum(
            1 for s in self.subscriptions.values() if s.status == "active"
        )
        total_transactions = len(self.transactions)
        successful_transactions = sum(
            1 for t in self.transactions if t.status == "completed"
        )

        # Calculate costs
        total_cost = sum(t.cost for t in self.transactions if t.status == "completed")

        # Data volume
        total_data_size = sum(
            t.data_size for t in self.transactions if t.status == "completed"
        )
        total_records = sum(
            t.records_count for t in self.transactions if t.status == "completed"
        )

        # Popular categories
        category_usage = {}
        for sub in self.subscriptions.values():
            product = self.products.get(sub.product_id)
            if product:
                category = product.category.value
                category_usage[category] = category_usage.get(category, 0) + 1

        return {
            "total_api_calls": successful_transactions,  # Add this for compatibility
            "subscriptions": {
                "total": total_subscriptions,
                "active": active_subscriptions,
                "inactive": total_subscriptions - active_subscriptions,
            },
            "transactions": {
                "total": total_transactions,
                "successful": successful_transactions,
                "failed": total_transactions - successful_transactions,
                "success_rate": (successful_transactions / total_transactions * 100)
                if total_transactions > 0
                else 0,
            },
            "costs": {
                "total": float(total_cost),
                "average_per_transaction": float(total_cost / successful_transactions)
                if successful_transactions > 0
                else 0,
            },
            "data_volume": {
                "total_size_mb": total_data_size / (1024 * 1024),
                "total_records": total_records,
                "avg_records_per_transaction": total_records / successful_transactions
                if successful_transactions > 0
                else 0,
            },
            "popular_categories": category_usage,
            "providers": {
                provider.value: sum(
                    1 for p in self.products.values() if p.provider == provider
                )
                for provider in MarketplaceProvider
            },
        }

    def export_data(
        self,
        subscription_id: str,
        format: str = "csv",
        output_path: Optional[str] = None,
    ) -> Union[str, bytes]:
        """Export fetched data in various formats"""
        # Get latest cached data
        cache_key = f"{subscription_id}:{json.dumps({})}"
        data = self._get_cached_data(cache_key)

        if not data:
            raise ValueError(f"No cached data found for subscription {subscription_id}")

        df = pd.DataFrame(data["records"])

        if format == "csv":
            return df.to_csv(index=False)
        elif format == "json":
            return df.to_json(orient="records", indent=2)
        elif format == "parquet":
            import io

            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            return buffer.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")


# Singleton instance
data_marketplace = DataMarketplaceIntegration()
