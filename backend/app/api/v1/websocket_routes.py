"""
WebSocket API Routes
Handles WebSocket connections and real-time communication
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import uuid
import logging
from app.websocket.websocket_manager import manager, VideoGenerationNotifier
from app.core.security import verify_token

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

async def get_current_user_ws(token: str) -> Optional[dict]:
    """Verify WebSocket authentication token"""
    try:
        # Verify JWT token
        payload = verify_token(token)
        return payload
    except Exception as e:
        logger.error(f"WebSocket auth error: {e}")
        return None

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """Main WebSocket endpoint"""
    client_id = str(uuid.uuid4())
    user_data = None
    
    # Authenticate if token provided
    if token:
        user_data = await get_current_user_ws(token)
        if not user_data:
            await websocket.close(code=1008, reason="Invalid authentication")
            return
    
    # Connect client
    user_id = user_data.get("sub") if user_data else None
    await manager.connect(websocket, client_id, user_id)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            await manager.handle_client_message(client_id, data)
            
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await manager.disconnect(client_id)

@router.websocket("/ws/channel/{channel_id}")
async def channel_websocket(
    websocket: WebSocket,
    channel_id: str,
    token: Optional[str] = Query(None)
):
    """Channel-specific WebSocket endpoint"""
    client_id = str(uuid.uuid4())
    
    # Authenticate if token provided
    user_data = None
    if token:
        user_data = await get_current_user_ws(token)
        if not user_data:
            await websocket.close(code=1008, reason="Invalid authentication")
            return
    
    # Connect and subscribe to channel
    user_id = user_data.get("sub") if user_data else None
    await manager.connect(websocket, client_id, user_id)
    await manager.subscribe_to_channel(client_id, channel_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            await manager.handle_client_message(client_id, data)
            
    except WebSocketDisconnect:
        await manager.unsubscribe_from_channel(client_id, channel_id)
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Channel WebSocket error: {e}")
        await manager.disconnect(client_id)

@router.on_event("startup")
async def startup_event():
    """Initialize WebSocket manager on startup"""
    await manager.initialize()
    logger.info("WebSocket manager initialized")

@router.on_event("shutdown")
async def shutdown_event():
    """Clean up WebSocket manager on shutdown"""
    await manager.cleanup()
    logger.info("WebSocket manager cleaned up")