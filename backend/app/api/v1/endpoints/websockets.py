"""
WebSocket Endpoints for Real-time Communication
Handles video generation updates, analytics streaming, and live notifications
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Optional, Any
import json
import logging
import asyncio
from datetime import datetime

from app.services.websocket_manager import ConnectionManager
from app.core.auth import verify_token
from app.services.video_generation_orchestrator import video_orchestrator
from app.services.realtime_analytics_service import realtime_analytics_service
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

# Initialize WebSocket manager
ws_manager = ConnectionManager()


async def get_current_user_from_ws(token: str) -> Dict[str, Any]:
    """Verify user from WebSocket token"""
    try:
        payload = verify_token(token)
        return payload
    except Exception as e:
        logger.error(f"WebSocket auth failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )


@router.websocket("/video-updates/{channel_id}")
async def video_generation_updates(
    websocket: WebSocket,
    channel_id: str,
    token: Optional[str] = None
):
    """
    WebSocket endpoint for real-time video generation updates
    
    Args:
        channel_id: Channel ID to monitor
        token: JWT authentication token
    """
    await websocket.accept()
    
    # Authenticate user
    if token:
        try:
            user = await get_current_user_from_ws(token)
            user_id = user.get("sub")
        except:
            await websocket.send_json({
                "type": "error",
                "message": "Authentication failed"
            })
            await websocket.close(code=1008)
            return
    else:
        # Allow unauthenticated for demo, but limit functionality
        user_id = "anonymous"
    
    # Add connection to manager
    connection_id = f"{user_id}:{channel_id}"
    await ws_manager.connect(websocket, connection_id)
    
    try:
        # Send initial connection success
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "channel_id": channel_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_json()
                message_type = data.get("type")
                
                if message_type == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                elif message_type == "subscribe":
                    # Subscribe to specific video generation
                    video_id = data.get("video_id")
                    if video_id:
                        await websocket.send_json({
                            "type": "subscribed",
                            "video_id": video_id,
                            "message": f"Subscribed to video {video_id} updates"
                        })
                        
                elif message_type == "get_status":
                    # Get current generation status
                    video_id = data.get("video_id")
                    if video_id:
                        # This would integrate with video generation service
                        status = await video_orchestrator.get_video_status(video_id)
                        await websocket.send_json({
                            "type": "status_update",
                            "video_id": video_id,
                            "status": status
                        })
                        
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    finally:
        # Remove connection
        await ws_manager.disconnect(connection_id)


@router.websocket("/analytics-stream")
async def analytics_stream(
    websocket: WebSocket,
    token: Optional[str] = None
):
    """
    WebSocket endpoint for real-time analytics streaming
    
    Streams live analytics data including:
    - Video performance metrics
    - Channel statistics
    - Cost tracking
    - User engagement
    """
    await websocket.accept()
    
    # Authenticate user
    if token:
        try:
            user = await get_current_user_from_ws(token)
            user_id = user.get("sub")
        except:
            await websocket.send_json({
                "type": "error",
                "message": "Authentication failed"
            })
            await websocket.close(code=1008)
            return
    else:
        user_id = "anonymous"
    
    connection_id = f"analytics:{user_id}"
    await ws_manager.connect(websocket, connection_id)
    
    try:
        # Send initial connection
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "stream": "analytics",
            "timestamp": datetime.now().isoformat()
        })
        
        # Start streaming analytics
        async def stream_analytics():
            while True:
                try:
                    # Get real-time analytics
                    analytics = await realtime_analytics_service.get_realtime_metrics(user_id)
                    
                    await websocket.send_json({
                        "type": "analytics_update",
                        "data": analytics,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Stream every 5 seconds
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logger.error(f"Analytics streaming error: {e}")
                    break
        
        # Start analytics streaming task
        stream_task = asyncio.create_task(stream_analytics())
        
        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_json()
                message_type = data.get("type")
                
                if message_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    
                elif message_type == "filter":
                    # Update analytics filter
                    filters = data.get("filters", {})
                    await websocket.send_json({
                        "type": "filter_applied",
                        "filters": filters
                    })
                    
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        logger.info(f"Analytics stream disconnected: {connection_id}")
    finally:
        # Cancel streaming task
        if 'stream_task' in locals():
            stream_task.cancel()
        await ws_manager.disconnect(connection_id)


@router.websocket("/notifications")
async def notifications_stream(
    websocket: WebSocket,
    token: Optional[str] = None
):
    """
    WebSocket endpoint for real-time notifications
    
    Delivers:
    - System alerts
    - Video completion notifications
    - Cost warnings
    - Error notifications
    """
    await websocket.accept()
    
    # Authenticate
    if token:
        try:
            user = await get_current_user_from_ws(token)
            user_id = user.get("sub")
        except:
            await websocket.send_json({
                "type": "error",
                "message": "Authentication failed"
            })
            await websocket.close(code=1008)
            return
    else:
        user_id = "anonymous"
    
    connection_id = f"notifications:{user_id}"
    await ws_manager.connect(websocket, connection_id)
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "stream": "notifications",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_json()
                
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    
                elif data.get("type") == "acknowledge":
                    # Acknowledge notification receipt
                    notification_id = data.get("notification_id")
                    logger.info(f"Notification {notification_id} acknowledged by {user_id}")
                    
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        logger.info(f"Notifications disconnected: {connection_id}")
    finally:
        await ws_manager.disconnect(connection_id)


@router.get("/ws/connections")
async def get_active_connections(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get list of active WebSocket connections (admin only)
    
    Returns:
        List of active connection IDs
    """
    # Verify admin access
    token = credentials.credentials
    user = verify_token(token)
    
    # Check if user is admin (implement your admin check logic)
    # if not user.get("is_admin"):
    #     raise HTTPException(status_code=403, detail="Admin access required")
    
    connections = await ws_manager.get_active_connections()
    return {
        "total": len(connections),
        "connections": connections
    }


@router.post("/ws/broadcast")
async def broadcast_message(
    message: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Broadcast message to all connected WebSocket clients (admin only)
    
    Args:
        message: Message to broadcast
        
    Returns:
        Broadcast status
    """
    # Verify admin access
    token = credentials.credentials
    user = verify_token(token)
    
    # Broadcast message
    await ws_manager.broadcast(json.dumps(message))
    
    return {
        "status": "success",
        "message": "Message broadcasted",
        "recipients": await ws_manager.get_active_connections()
    }