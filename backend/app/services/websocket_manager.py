"""
WebSocket Manager for YTEmpire
Real-time communication for video generation status and metrics
"""
from typing import Dict, List, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
from datetime import datetime
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types"""
    VIDEO_STATUS = "video_status"
    METRIC_UPDATE = "metric_update"
    NOTIFICATION = "notification"
    CHANNEL_UPDATE = "channel_update"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting"""
    
    def __init__(self):
        # Store active connections by user_id
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Store connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        # Room/channel subscriptions
        self.room_subscriptions: Dict[str, Set[str]] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        
        # Add to active connections
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        
        # Store metadata
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "connected_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        logger.info(f"WebSocket connected for user {user_id}")
        
        # Send initial connection message
        await self.send_personal_message(
            user_id,
            {
                "type": MessageType.NOTIFICATION.value,
                "data": {
                    "message": "Connected to YTEmpire real-time updates",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
    
    async def disconnect(self, websocket: WebSocket, user_id: str):
        """Remove a WebSocket connection"""
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
                
            # Clean up empty lists
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        # Remove metadata
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
        
        # Remove from room subscriptions
        for room_id, subscribers in self.room_subscriptions.items():
            if user_id in subscribers:
                subscribers.remove(user_id)
        
        logger.info(f"WebSocket disconnected for user {user_id}")
    
    async def send_personal_message(self, user_id: str, message: Dict[str, Any]):
        """Send message to a specific user (all their connections)"""
        if user_id in self.active_connections:
            disconnected = []
            
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending message to {user_id}: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                await self.disconnect(conn, user_id)
    
    async def broadcast(self, message: Dict[str, Any], exclude_user: Optional[str] = None):
        """Broadcast message to all connected users"""
        disconnected = []
        
        for user_id, connections in self.active_connections.items():
            if exclude_user and user_id == exclude_user:
                continue
                
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {user_id}: {e}")
                    disconnected.append((connection, user_id))
        
        # Clean up disconnected connections
        for conn, uid in disconnected:
            await self.disconnect(conn, uid)
    
    async def send_to_room(self, room_id: str, message: Dict[str, Any], exclude_user: Optional[str] = None):
        """Send message to all users in a specific room/channel"""
        if room_id not in self.room_subscriptions:
            return
        
        for user_id in self.room_subscriptions[room_id]:
            if exclude_user and user_id == exclude_user:
                continue
            
            await self.send_personal_message(user_id, message)
    
    async def join_room(self, user_id: str, room_id: str):
        """Add user to a room/channel"""
        if room_id not in self.room_subscriptions:
            self.room_subscriptions[room_id] = set()
        
        self.room_subscriptions[room_id].add(user_id)
        
        # Notify room members
        await self.send_to_room(
            room_id,
            {
                "type": MessageType.NOTIFICATION.value,
                "data": {
                    "message": f"User {user_id} joined the room",
                    "room_id": room_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            },
            exclude_user=user_id
        )
    
    async def leave_room(self, user_id: str, room_id: str):
        """Remove user from a room/channel"""
        if room_id in self.room_subscriptions:
            if user_id in self.room_subscriptions[room_id]:
                self.room_subscriptions[room_id].remove(user_id)
                
                # Clean up empty rooms
                if not self.room_subscriptions[room_id]:
                    del self.room_subscriptions[room_id]
                
                # Notify room members
                await self.send_to_room(
                    room_id,
                    {
                        "type": MessageType.NOTIFICATION.value,
                        "data": {
                            "message": f"User {user_id} left the room",
                            "room_id": room_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    }
                )
    
    async def send_video_status_update(self, user_id: str, video_id: str, status: str, progress: float, metadata: Optional[Dict[str, Any]] = None):
        """Send video generation status update"""
        message = {
            "type": MessageType.VIDEO_STATUS.value,
            "data": {
                "video_id": video_id,
                "status": status,
                "progress": progress,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
        }
        
        await self.send_personal_message(user_id, message)
    
    async def send_metric_update(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Broadcast metric update to all connected users"""
        message = {
            "type": MessageType.METRIC_UPDATE.value,
            "data": {
                "metric_name": metric_name,
                "value": value,
                "labels": labels or {},
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        await self.broadcast(message)
    
    async def send_channel_update(self, channel_id: str, update_type: str, data: Dict[str, Any]):
        """Send channel-specific update to subscribers"""
        room_id = f"channel:{channel_id}"
        
        message = {
            "type": MessageType.CHANNEL_UPDATE.value,
            "data": {
                "channel_id": channel_id,
                "update_type": update_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        await self.send_to_room(room_id, message)
    
    async def handle_incoming_message(self, websocket: WebSocket, user_id: str, data: Dict[str, Any]):
        """Process incoming WebSocket messages"""
        message_type = data.get("type")
        
        if message_type == MessageType.PING.value:
            # Respond to ping
            await websocket.send_json({
                "type": MessageType.PONG.value,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        elif message_type == "subscribe":
            # Subscribe to a room/channel
            room_id = data.get("room_id")
            if room_id:
                await self.join_room(user_id, room_id)
        
        elif message_type == "unsubscribe":
            # Unsubscribe from a room/channel
            room_id = data.get("room_id")
            if room_id:
                await self.leave_room(user_id, room_id)
        
        else:
            # Handle custom message types
            logger.info(f"Received message from {user_id}: {message_type}")
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return sum(len(connections) for connections in self.active_connections.values())
    
    def get_user_count(self) -> int:
        """Get number of connected users"""
        return len(self.active_connections)
    
    def get_room_info(self) -> Dict[str, int]:
        """Get information about active rooms"""
        return {
            room_id: len(subscribers)
            for room_id, subscribers in self.room_subscriptions.items()
        }


# Global connection manager instance
manager = ConnectionManager()


class WebSocketEndpoint:
    """WebSocket endpoint handler"""
    
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
    
    async def websocket_endpoint(self, websocket: WebSocket, user_id: str):
        """Handle WebSocket connection lifecycle"""
        await self.manager.connect(websocket, user_id)
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_json()
                
                # Process message
                await self.manager.handle_incoming_message(websocket, user_id, data)
                
        except WebSocketDisconnect:
            await self.manager.disconnect(websocket, user_id)
        except Exception as e:
            logger.error(f"WebSocket error for user {user_id}: {e}")
            await self.manager.disconnect(websocket, user_id)


# Create endpoint instance
ws_endpoint = WebSocketEndpoint(manager)


# Background task for periodic updates
async def send_periodic_metrics():
    """Send periodic metric updates to all connected clients"""
    while True:
        await asyncio.sleep(30)  # Send updates every 30 seconds
        
        # Example metrics
        metrics = {
            "active_connections": manager.get_connection_count(),
            "connected_users": manager.get_user_count(),
            "active_rooms": len(manager.room_subscriptions)
        }
        
        await manager.send_metric_update(
            metric_name="system.stats",
            value=1,
            labels=metrics
        )