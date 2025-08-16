"""
API endpoints for third-party integrations
"""

from fastapi import APIRouter, HTTPException, Depends, Header, Request
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl

from app.core.security import get_current_user
from app.models.user import User
from app.services.third_party_integrations import (
    third_party_service,
    IntegrationType,
    IntegrationStatus,
    IntegrationConfig,
    WebhookEvent,
)

router = APIRouter()


class IntegrationRequest(BaseModel):
    """Integration configuration request"""

    name: str
    type: str  # webhook, rest_api, oauth, graphql, websocket, sdk
    base_url: HttpUrl
    auth_type: str  # api_key, oauth2, basic, bearer
    credentials: Dict[str, str] = {}
    rate_limit: int = Field(default=100, ge=1, le=1000)
    timeout: float = Field(default=30.0, ge=1.0, le=120.0)
    webhook_events: List[str] = []
    custom_settings: Dict[str, Any] = {}


class WebhookPayload(BaseModel):
    """Webhook payload model"""

    event_type: str
    payload: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = {}


class SlackMessage(BaseModel):
    """Slack message model"""

    channel: Optional[str] = None
    text: str
    attachments: Optional[List[Dict[str, Any]]] = None
    blocks: Optional[List[Dict[str, Any]]] = None


class AirtableRecord(BaseModel):
    """Airtable record model"""

    fields: Dict[str, Any]


@router.get("/status")
async def get_integrations_status(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get status of all configured integrations

    Returns the connection status and configuration for all integrations
    """
    try:
        if not third_party_service.is_initialized:
            await third_party_service.initialize()

        status = third_party_service.get_all_integrations_status()

        return {
            "timestamp": datetime.now().isoformat(),
            "integrations": status,
            "total": len(status),
            "active": sum(1 for s in status.values() if s["status"] == "active"),
            "error": sum(1 for s in status.values() if s["status"] == "error"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register")
async def register_integration(
    config: IntegrationRequest, current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Register a new third-party integration

    - **name**: Unique name for the integration
    - **type**: Type of integration (webhook, rest_api, oauth, etc.)
    - **base_url**: Base URL for the integration
    - **auth_type**: Authentication type
    - **credentials**: Authentication credentials
    """
    try:
        if not third_party_service.is_initialized:
            await third_party_service.initialize()

        integration_config = IntegrationConfig(
            name=config.name,
            type=IntegrationType(config.type),
            base_url=str(config.base_url),
            auth_type=config.auth_type,
            credentials=config.credentials,
            rate_limit=config.rate_limit,
            timeout=config.timeout,
            webhook_events=config.webhook_events,
            custom_settings=config.custom_settings,
        )

        await third_party_service.register_integration(integration_config)

        return {
            "status": "success",
            "message": f"Integration '{config.name}' registered successfully",
            "config": config.dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook/send")
async def send_webhook(
    integration_name: str,
    event: WebhookPayload,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Send a webhook to a specific integration

    - **integration_name**: Name of the integration to send to
    - **event**: Webhook event payload
    """
    try:
        await third_party_service.send_webhook(
            integration_name, event.event_type, event.payload
        )

        return {
            "status": "success",
            "message": f"Webhook sent to {integration_name}",
            "event_type": event.event_type,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook/receive/{integration_name}")
async def receive_webhook(
    integration_name: str,
    request: Request,
    event_type: str = Header(None, alias="X-Event-Type"),
    signature: str = Header(None, alias="X-Signature"),
) -> Dict[str, Any]:
    """
    Receive incoming webhook from external service

    - **integration_name**: Name of the integration sending the webhook
    """
    try:
        payload = await request.json()

        # Handle the incoming webhook
        await third_party_service.handle_incoming_webhook(
            integration_name, event_type or "unknown", payload, signature
        )

        return {"status": "success", "message": "Webhook received and processed"}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/slack/message")
async def send_slack_message(
    message: SlackMessage, current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Send a message to Slack

    - **channel**: Slack channel (optional)
    - **text**: Message text
    - **attachments**: Message attachments (optional)
    - **blocks**: Message blocks for rich formatting (optional)
    """
    try:
        payload = {"text": message.text}

        if message.channel:
            payload["channel"] = message.channel
        if message.attachments:
            payload["attachments"] = message.attachments
        if message.blocks:
            payload["blocks"] = message.blocks

        await third_party_service.send_webhook("slack", "message", payload)

        return {"status": "success", "message": "Slack message sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/discord/message")
async def send_discord_message(
    content: str,
    embed: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Send a message to Discord

    - **content**: Message content
    - **embed**: Rich embed object (optional)
    """
    try:
        payload = {"content": content}

        if embed:
            payload["embeds"] = [embed]

        await third_party_service.send_webhook("discord", "message", payload)

        return {"status": "success", "message": "Discord message sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/airtable/sync")
async def sync_to_airtable(
    records: List[AirtableRecord],
    table_name: Optional[str] = None,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Sync records to Airtable

    - **records**: List of records to sync
    - **table_name**: Airtable table name (optional)
    """
    try:
        data = [record.fields for record in records]
        await third_party_service.sync_to_airtable(data, table_name)

        return {
            "status": "success",
            "message": f"Synced {len(records)} records to Airtable",
            "count": len(records),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/google-sheets/sync")
async def sync_to_google_sheets(
    spreadsheet_id: str,
    range_name: str,
    values: List[List[Any]],
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Sync data to Google Sheets

    - **spreadsheet_id**: Google Sheets spreadsheet ID
    - **range_name**: Range to update (e.g., 'Sheet1!A1:D10')
    - **values**: 2D array of values to write
    """
    try:
        await third_party_service.sync_to_google_sheets(
            spreadsheet_id, range_name, values
        )

        return {
            "status": "success",
            "message": f"Synced {len(values)} rows to Google Sheets",
            "rows": len(values),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/notion/page")
async def create_notion_page(
    parent_id: str,
    title: str,
    properties: Dict[str, Any],
    content: Optional[List[Dict[str, Any]]] = None,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Create a page in Notion

    - **parent_id**: Parent database or page ID
    - **title**: Page title
    - **properties**: Page properties
    - **content**: Page content blocks (optional)
    """
    try:
        # Add title to properties
        properties["title"] = {"title": [{"text": {"content": title}}]}

        result = await third_party_service.create_notion_page(
            parent_id, properties, content
        )

        return {
            "status": "success",
            "message": "Notion page created",
            "page_id": result.get("id") if result else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trello/card")
async def create_trello_card(
    list_id: str,
    name: str,
    desc: Optional[str] = None,
    labels: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Create a card in Trello

    - **list_id**: Trello list ID
    - **name**: Card name
    - **desc**: Card description (optional)
    - **labels**: Card labels (optional)
    """
    try:
        result = await third_party_service.create_trello_card(
            list_id, name, desc, labels
        )

        return {
            "status": "success",
            "message": "Trello card created",
            "card_id": result.get("id") if result else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hubspot/contact")
async def add_hubspot_contact(
    email: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    company: Optional[str] = None,
    properties: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Add a contact to HubSpot

    - **email**: Contact email
    - **first_name**: First name (optional)
    - **last_name**: Last name (optional)
    - **company**: Company name (optional)
    - **properties**: Additional properties (optional)
    """
    try:
        contact_properties = properties or {}

        if first_name:
            contact_properties["firstname"] = first_name
        if last_name:
            contact_properties["lastname"] = last_name
        if company:
            contact_properties["company"] = company

        result = await third_party_service.add_hubspot_contact(
            email, contact_properties
        )

        return {
            "status": "success",
            "message": "HubSpot contact added",
            "contact_id": result.get("id") if result else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mailchimp/subscribe")
async def subscribe_to_mailchimp(
    list_id: str,
    email: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Subscribe email to Mailchimp list

    - **list_id**: Mailchimp list ID
    - **email**: Email to subscribe
    - **first_name**: First name (optional)
    - **last_name**: Last name (optional)
    - **tags**: Tags to apply (optional)
    """
    try:
        merge_fields = {}

        if first_name:
            merge_fields["FNAME"] = first_name
        if last_name:
            merge_fields["LNAME"] = last_name

        result = await third_party_service.subscribe_to_mailchimp(
            list_id, email, merge_fields if merge_fields else None
        )

        return {
            "status": "success",
            "message": f"Subscribed {email} to Mailchimp",
            "email": email,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/call")
async def call_integration_api(
    integration_name: str,
    method: str,
    endpoint: str,
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Make a generic API call to an integration

    - **integration_name**: Name of the integration
    - **method**: HTTP method (GET, POST, PUT, DELETE, etc.)
    - **endpoint**: API endpoint path
    - **data**: Request body (optional)
    - **params**: Query parameters (optional)
    - **headers**: Additional headers (optional)
    """
    try:
        result = await third_party_service.call_api(
            integration_name, method, endpoint, data, params, headers
        )

        return {"status": "success", "data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/notifications")
async def send_batch_notifications(
    event_type: str,
    payload: Dict[str, Any],
    integrations: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Send notifications to multiple integrations

    - **event_type**: Type of event
    - **payload**: Event payload
    - **integrations**: List of integrations to notify (optional, defaults to all)
    """
    try:
        if integrations is None:
            integrations = list(third_party_service.integrations.keys())

        results = {}
        errors = {}

        for integration in integrations:
            try:
                await third_party_service.send_webhook(integration, event_type, payload)
                results[integration] = "success"
            except Exception as e:
                errors[integration] = str(e)

        return {
            "status": "completed",
            "event_type": event_type,
            "successful": results,
            "failed": errors,
            "total": len(integrations),
            "success_count": len(results),
            "error_count": len(errors),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
