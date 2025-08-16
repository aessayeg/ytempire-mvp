"""
Third-Party Integrations Service
Manages integrations with external services and APIs
"""

import asyncio
import logging
import hashlib
import hmac
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json
import httpx
from pydantic import BaseModel, Field

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.core.config import settings
from app.services.advanced_error_recovery import advanced_recovery, RecoveryStrategy

logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """Types of integrations"""

    WEBHOOK = "webhook"
    REST_API = "rest_api"
    OAUTH = "oauth"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    SDK = "sdk"


class IntegrationStatus(Enum):
    """Integration connection status"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    AUTHENTICATING = "authenticating"


@dataclass
class IntegrationConfig:
    """Configuration for a third-party integration"""

    name: str
    type: IntegrationType
    base_url: str
    auth_type: str  # "api_key", "oauth2", "basic", "bearer"
    credentials: Dict[str, str]

    # Rate limiting
    rate_limit: int = 100  # requests per minute
    burst_limit: int = 10

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0

    # Timeout
    timeout: float = 30.0

    # Headers
    default_headers: Dict[str, str] = field(default_factory=dict)

    # Webhooks
    webhook_secret: Optional[str] = None
    webhook_events: List[str] = field(default_factory=list)

    # OAuth
    oauth_authorize_url: Optional[str] = None
    oauth_token_url: Optional[str] = None
    oauth_scopes: List[str] = field(default_factory=list)

    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class WebhookEvent(BaseModel):
    """Webhook event model"""

    id: str
    integration: str
    event_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    signature: Optional[str] = None
    verified: bool = False


class ThirdPartyIntegrationService:
    """Service for managing third-party integrations"""

    def __init__(self):
        self.integrations: Dict[str, IntegrationConfig] = {}
        self.connections: Dict[str, httpx.AsyncClient] = {}
        self.rate_limiters: Dict[str, "RateLimiter"] = {}
        self.webhook_handlers: Dict[str, List[Callable]] = {}
        self.oauth_tokens: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False

    async def initialize(self):
        """Initialize the integration service"""
        # Register default integrations
        await self._register_default_integrations()

        # Initialize error recovery
        await advanced_recovery.initialize()

        self.is_initialized = True
        logger.info("Third-party integration service initialized")

    async def shutdown(self):
        """Shutdown all connections"""
        for client in self.connections.values():
            await client.aclose()

        await advanced_recovery.shutdown()

    async def _register_default_integrations(self):
        """Register default integrations"""
        # Slack integration
        if hasattr(settings, "SLACK_WEBHOOK_URL"):
            await self.register_integration(
                IntegrationConfig(
                    name="slack",
                    type=IntegrationType.WEBHOOK,
                    base_url=settings.SLACK_WEBHOOK_URL,
                    auth_type="none",
                    credentials={},
                    webhook_events=["video_generated", "error", "user_signup"],
                )
            )

        # Discord integration
        if hasattr(settings, "DISCORD_WEBHOOK_URL"):
            await self.register_integration(
                IntegrationConfig(
                    name="discord",
                    type=IntegrationType.WEBHOOK,
                    base_url=settings.DISCORD_WEBHOOK_URL,
                    auth_type="none",
                    credentials={},
                    webhook_events=["video_published", "milestone_reached"],
                )
            )

        # Zapier integration
        if hasattr(settings, "ZAPIER_WEBHOOK_URL"):
            await self.register_integration(
                IntegrationConfig(
                    name="zapier",
                    type=IntegrationType.WEBHOOK,
                    base_url=settings.ZAPIER_WEBHOOK_URL,
                    auth_type="api_key",
                    credentials={"api_key": getattr(settings, "ZAPIER_API_KEY", "")},
                    webhook_events=["all"],
                )
            )

        # Make.com (Integromat) integration
        if hasattr(settings, "MAKE_WEBHOOK_URL"):
            await self.register_integration(
                IntegrationConfig(
                    name="make",
                    type=IntegrationType.WEBHOOK,
                    base_url=settings.MAKE_WEBHOOK_URL,
                    auth_type="none",
                    credentials={},
                    webhook_events=["all"],
                )
            )

        # Airtable integration
        if hasattr(settings, "AIRTABLE_API_KEY"):
            await self.register_integration(
                IntegrationConfig(
                    name="airtable",
                    type=IntegrationType.REST_API,
                    base_url="https://api.airtable.com/v0",
                    auth_type="bearer",
                    credentials={"token": settings.AIRTABLE_API_KEY},
                    rate_limit=5,  # Airtable has strict rate limits
                    custom_settings={
                        "base_id": getattr(settings, "AIRTABLE_BASE_ID", ""),
                        "table_name": getattr(
                            settings, "AIRTABLE_TABLE_NAME", "Videos"
                        ),
                    },
                )
            )

        # Google Sheets integration
        if hasattr(settings, "GOOGLE_SERVICE_ACCOUNT"):
            await self.register_integration(
                IntegrationConfig(
                    name="google_sheets",
                    type=IntegrationType.REST_API,
                    base_url="https://sheets.googleapis.com/v4",
                    auth_type="oauth2",
                    credentials=json.loads(settings.GOOGLE_SERVICE_ACCOUNT),
                    oauth_scopes=["https://www.googleapis.com/auth/spreadsheets"],
                )
            )

        # Notion integration
        if hasattr(settings, "NOTION_API_KEY"):
            await self.register_integration(
                IntegrationConfig(
                    name="notion",
                    type=IntegrationType.REST_API,
                    base_url="https://api.notion.com/v1",
                    auth_type="bearer",
                    credentials={"token": settings.NOTION_API_KEY},
                    default_headers={"Notion-Version": "2022-06-28"},
                )
            )

        # Trello integration
        if hasattr(settings, "TRELLO_API_KEY"):
            await self.register_integration(
                IntegrationConfig(
                    name="trello",
                    type=IntegrationType.REST_API,
                    base_url="https://api.trello.com/1",
                    auth_type="api_key",
                    credentials={
                        "key": settings.TRELLO_API_KEY,
                        "token": getattr(settings, "TRELLO_TOKEN", ""),
                    },
                )
            )

        # HubSpot integration
        if hasattr(settings, "HUBSPOT_API_KEY"):
            await self.register_integration(
                IntegrationConfig(
                    name="hubspot",
                    type=IntegrationType.REST_API,
                    base_url="https://api.hubapi.com",
                    auth_type="bearer",
                    credentials={"token": settings.HUBSPOT_API_KEY},
                    rate_limit=100,
                )
            )

        # Mailchimp integration
        if hasattr(settings, "MAILCHIMP_API_KEY"):
            dc = settings.MAILCHIMP_API_KEY.split("-")[-1]  # Data center
            await self.register_integration(
                IntegrationConfig(
                    name="mailchimp",
                    type=IntegrationType.REST_API,
                    base_url=f"https://{dc}.api.mailchimp.com/3.0",
                    auth_type="basic",
                    credentials={
                        "username": "anystring",
                        "password": settings.MAILCHIMP_API_KEY,
                    },
                )
            )

    async def register_integration(self, config: IntegrationConfig):
        """Register a new integration"""
        self.integrations[config.name] = config

        # Create HTTP client
        headers = config.default_headers.copy()

        # Add authentication headers
        if config.auth_type == "bearer":
            headers["Authorization"] = f"Bearer {config.credentials.get('token', '')}"
        elif config.auth_type == "api_key":
            headers["X-API-Key"] = config.credentials.get("api_key", "")

        self.connections[config.name] = httpx.AsyncClient(
            base_url=config.base_url, headers=headers, timeout=config.timeout
        )

        # Create rate limiter
        self.rate_limiters[config.name] = RateLimiter(
            config.rate_limit, config.burst_limit
        )

        # Register with error recovery
        advanced_recovery.register_circuit_breaker(f"integration_{config.name}")
        advanced_recovery.register_bulkhead(f"integration_{config.name}")

        logger.info(f"Registered integration: {config.name}")

    async def call_api(
        self,
        integration_name: str,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Call a third-party API with error recovery"""
        if integration_name not in self.integrations:
            raise ValueError(f"Integration {integration_name} not registered")

        config = self.integrations[integration_name]
        client = self.connections[integration_name]
        rate_limiter = self.rate_limiters[integration_name]

        # Check rate limit
        await rate_limiter.acquire()

        # Build request
        request_headers = headers or {}

        # Add OAuth token if available
        if config.auth_type == "oauth2" and integration_name in self.oauth_tokens:
            token_data = self.oauth_tokens[integration_name]
            request_headers["Authorization"] = f"Bearer {token_data['access_token']}"

        # Execute with error recovery
        async def make_request():
            response = await client.request(
                method=method,
                url=endpoint,
                json=data,
                params=params,
                headers=request_headers,
            )
            response.raise_for_status()
            return response.json()

        try:
            result = await advanced_recovery.with_circuit_breaker(
                make_request, f"integration_{integration_name}"
            )

            logger.info(f"API call successful: {integration_name} {method} {endpoint}")
            return result

        except Exception as e:
            logger.error(
                f"API call failed: {integration_name} {method} {endpoint}: {e}"
            )
            raise

    async def send_webhook(
        self, integration_name: str, event_type: str, payload: Dict[str, Any]
    ):
        """Send webhook to integration"""
        if integration_name not in self.integrations:
            raise ValueError(f"Integration {integration_name} not registered")

        config = self.integrations[integration_name]

        if config.type != IntegrationType.WEBHOOK:
            raise ValueError(f"Integration {integration_name} is not a webhook")

        # Check if event type is supported
        if config.webhook_events and "all" not in config.webhook_events:
            if event_type not in config.webhook_events:
                logger.debug(
                    f"Event {event_type} not configured for {integration_name}"
                )
                return

        # Create webhook event
        event = WebhookEvent(
            id=hashlib.md5(
                f"{integration_name}{event_type}{datetime.now()}".encode()
            ).hexdigest(),
            integration=integration_name,
            event_type=event_type,
            payload=payload,
            timestamp=datetime.now(),
        )

        # Add signature if configured
        if config.webhook_secret:
            event.signature = self._generate_webhook_signature(
                config.webhook_secret, event.dict()
            )

        # Send webhook
        try:
            await self.call_api(integration_name, "POST", "", data=event.dict())
            logger.info(f"Webhook sent: {integration_name} - {event_type}")

        except Exception as e:
            logger.error(
                f"Failed to send webhook: {integration_name} - {event_type}: {e}"
            )

    async def verify_webhook(
        self, integration_name: str, signature: str, payload: Dict[str, Any]
    ) -> bool:
        """Verify incoming webhook signature"""
        if integration_name not in self.integrations:
            return False

        config = self.integrations[integration_name]

        if not config.webhook_secret:
            return True  # No signature verification configured

        expected_signature = self._generate_webhook_signature(
            config.webhook_secret, payload
        )

        return hmac.compare_digest(signature, expected_signature)

    def _generate_webhook_signature(self, secret: str, payload: Dict[str, Any]) -> str:
        """Generate webhook signature"""
        message = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"

    def register_webhook_handler(
        self, integration_name: str, event_type: str, handler: Callable
    ):
        """Register a handler for incoming webhooks"""
        key = f"{integration_name}:{event_type}"
        if key not in self.webhook_handlers:
            self.webhook_handlers[key] = []

        self.webhook_handlers[key].append(handler)
        logger.info(f"Registered webhook handler: {integration_name} - {event_type}")

    async def handle_incoming_webhook(
        self,
        integration_name: str,
        event_type: str,
        payload: Dict[str, Any],
        signature: Optional[str] = None,
    ):
        """Handle incoming webhook"""
        # Verify signature if provided
        if signature:
            if not await self.verify_webhook(integration_name, signature, payload):
                logger.warning(f"Invalid webhook signature from {integration_name}")
                raise ValueError("Invalid webhook signature")

        # Find handlers
        handlers = []
        handlers.extend(
            self.webhook_handlers.get(f"{integration_name}:{event_type}", [])
        )
        handlers.extend(self.webhook_handlers.get(f"{integration_name}:*", []))
        handlers.extend(self.webhook_handlers.get(f"*:{event_type}", []))
        handlers.extend(self.webhook_handlers.get("*:*", []))

        if not handlers:
            logger.warning(
                f"No handlers for webhook: {integration_name} - {event_type}"
            )
            return

        # Execute handlers
        for handler in handlers:
            try:
                await handler(integration_name, event_type, payload)
            except Exception as e:
                logger.error(f"Webhook handler failed: {e}")

    async def sync_to_airtable(
        self, data: List[Dict[str, Any]], table_name: Optional[str] = None
    ):
        """Sync data to Airtable"""
        if "airtable" not in self.integrations:
            logger.warning("Airtable integration not configured")
            return

        config = self.integrations["airtable"]
        base_id = config.custom_settings.get("base_id")
        table = table_name or config.custom_settings.get("table_name", "Videos")

        # Prepare records
        records = [{"fields": record} for record in data]

        # Send to Airtable
        await self.call_api(
            "airtable", "POST", f"/{base_id}/{table}", data={"records": records}
        )

        logger.info(f"Synced {len(data)} records to Airtable")

    async def sync_to_google_sheets(
        self, spreadsheet_id: str, range_name: str, values: List[List[Any]]
    ):
        """Sync data to Google Sheets"""
        if "google_sheets" not in self.integrations:
            logger.warning("Google Sheets integration not configured")
            return

        await self.call_api(
            "google_sheets",
            "PUT",
            f"/spreadsheets/{spreadsheet_id}/values/{range_name}",
            params={"valueInputOption": "USER_ENTERED"},
            data={"range": range_name, "values": values},
        )

        logger.info(f"Synced {len(values)} rows to Google Sheets")

    async def create_notion_page(
        self,
        parent_id: str,
        properties: Dict[str, Any],
        content: Optional[List[Dict[str, Any]]] = None,
    ):
        """Create a page in Notion"""
        if "notion" not in self.integrations:
            logger.warning("Notion integration not configured")
            return

        data = {"parent": {"database_id": parent_id}, "properties": properties}

        if content:
            data["children"] = content

        result = await self.call_api("notion", "POST", "/pages", data=data)

        logger.info(f"Created Notion page: {result.get('id')}")
        return result

    async def create_trello_card(
        self,
        list_id: str,
        name: str,
        desc: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ):
        """Create a card in Trello"""
        if "trello" not in self.integrations:
            logger.warning("Trello integration not configured")
            return

        config = self.integrations["trello"]
        params = {
            "key": config.credentials["key"],
            "token": config.credentials["token"],
            "idList": list_id,
            "name": name,
        }

        if desc:
            params["desc"] = desc

        if labels:
            params["idLabels"] = ",".join(labels)

        result = await self.call_api("trello", "POST", "/cards", params=params)

        logger.info(f"Created Trello card: {result.get('id')}")
        return result

    async def add_hubspot_contact(self, email: str, properties: Dict[str, Any]):
        """Add a contact to HubSpot"""
        if "hubspot" not in self.integrations:
            logger.warning("HubSpot integration not configured")
            return

        data = {"properties": {"email": email, **properties}}

        result = await self.call_api(
            "hubspot", "POST", "/crm/v3/objects/contacts", data=data
        )

        logger.info(f"Added HubSpot contact: {result.get('id')}")
        return result

    async def subscribe_to_mailchimp(
        self, list_id: str, email: str, merge_fields: Optional[Dict[str, Any]] = None
    ):
        """Subscribe email to Mailchimp list"""
        if "mailchimp" not in self.integrations:
            logger.warning("Mailchimp integration not configured")
            return

        data = {"email_address": email, "status": "subscribed"}

        if merge_fields:
            data["merge_fields"] = merge_fields

        # Use MD5 hash of email for member ID
        member_id = hashlib.md5(email.lower().encode()).hexdigest()

        result = await self.call_api(
            "mailchimp", "PUT", f"/lists/{list_id}/members/{member_id}", data=data
        )

        logger.info(f"Subscribed {email} to Mailchimp")
        return result

    def get_integration_status(self, integration_name: str) -> IntegrationStatus:
        """Get the status of an integration"""
        if integration_name not in self.integrations:
            return IntegrationStatus.INACTIVE

        # Check circuit breaker status
        breaker_status = advanced_recovery.get_circuit_breaker_status()
        breaker_key = f"integration_{integration_name}"

        if breaker_key in breaker_status:
            if breaker_status[breaker_key]["state"] == "OPEN":
                return IntegrationStatus.ERROR

        # Check rate limiter
        if integration_name in self.rate_limiters:
            limiter = self.rate_limiters[integration_name]
            if limiter.is_rate_limited():
                return IntegrationStatus.RATE_LIMITED

        return IntegrationStatus.ACTIVE

    def get_all_integrations_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all integrations"""
        status = {}

        for name in self.integrations:
            status[name] = {
                "status": self.get_integration_status(name).value,
                "type": self.integrations[name].type.value,
                "rate_limit": self.integrations[name].rate_limit,
            }

        return status


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, rate_limit: int, burst_limit: int):
        self.rate_limit = rate_limit  # requests per minute
        self.burst_limit = burst_limit
        self.requests: List[datetime] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        async with self._lock:
            now = datetime.now()

            # Remove old requests
            minute_ago = now - timedelta(minutes=1)
            self.requests = [r for r in self.requests if r > minute_ago]

            # Check rate limit
            if len(self.requests) >= self.rate_limit:
                # Calculate wait time
                oldest = self.requests[0]
                wait_time = (oldest + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    return await self.acquire()

            # Check burst limit
            if len(self.requests) >= self.burst_limit:
                second_ago = now - timedelta(seconds=1)
                recent = [r for r in self.requests if r > second_ago]
                if len(recent) >= self.burst_limit:
                    await asyncio.sleep(1)
                    return await self.acquire()

            self.requests.append(now)

    def is_rate_limited(self) -> bool:
        """Check if currently rate limited"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        recent_requests = [r for r in self.requests if r > minute_ago]
        return len(recent_requests) >= self.rate_limit


# Singleton instance
third_party_service = ThirdPartyIntegrationService()
