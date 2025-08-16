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
    COLLABORATION = "collaboration"
    PRESENCE = "presence"
    CURSOR = "cursor"
    TYPING = "typing"
    SYNC = "sync"
    BROADCAST = "broadcast"
    PRIVATE_MESSAGE = "private_message"


class ConnectionManager:
    """Enhanced WebSocket manager for real-time collaboration"""

    def __init__(self):
        # Store active connections by user_id
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Store connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        # Room/channel subscriptions
        self.room_subscriptions: Dict[str, Set[str]] = {}
        # User presence tracking
        self.user_presence: Dict[str, Dict[str, Any]] = {}
        # Collaboration sessions
        self.collaboration_sessions: Dict[str, Dict[str, Any]] = {}
        # Message history for rooms (last 100 messages)
        self.room_history: Dict[str, List[Dict[str, Any]]] = {}
        # Typing indicators
        self.typing_status: Dict[str, Set[str]] = {}
        # User cursor positions for collaborative editing
        self.cursor_positions: Dict[str, Dict[str, Any]] = {}

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Accept and register a new WebSocket connection with enhanced features"""
        await websocket.accept()

        # Add to active connections
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)

        # Store metadata
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "connected_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "client_id": metadata.get("client_id") if metadata else None,
        }

        # Update user presence
        self.user_presence[user_id] = {
            "status": "online",
            "last_seen": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "active_rooms": set(),
        }

        logger.info(f"WebSocket connected for user {user_id}")

        # Send initial connection message with session info
        await self.send_personal_message(
            user_id,
            {
                "type": MessageType.NOTIFICATION.value,
                "data": {
                    "message": "Connected to YTEmpire real-time updates",
                    "timestamp": datetime.utcnow().isoformat(),
                    "session_id": websocket.__hash__(),
                    "features": [
                        "collaboration",
                        "presence",
                        "typing",
                        "cursor_tracking",
                    ],
                },
            },
        )

        # Broadcast presence update
        await self.broadcast_presence_update(user_id, "online")

    async def disconnect(self, websocket: WebSocket, user_id: str):
        """Remove a WebSocket connection with cleanup"""
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)

            # Clean up empty lists
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

                # Update presence to offline
                if user_id in self.user_presence:
                    self.user_presence[user_id]["status"] = "offline"
                    self.user_presence[user_id][
                        "last_seen"
                    ] = datetime.utcnow().isoformat()

                # Broadcast presence update
                await self.broadcast_presence_update(user_id, "offline")

        # Remove metadata
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]

        # Remove from room subscriptions and typing status
        for room_id, subscribers in list(self.room_subscriptions.items()):
            if user_id in subscribers:
                subscribers.remove(user_id)
                # Clear typing status
                if (
                    room_id in self.typing_status
                    and user_id in self.typing_status[room_id]
                ):
                    self.typing_status[room_id].remove(user_id)
                    await self.broadcast_typing_status(room_id)

        # Remove cursor position
        if user_id in self.cursor_positions:
            del self.cursor_positions[user_id]

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

    async def broadcast(
        self, message: Dict[str, Any], exclude_user: Optional[str] = None
    ):
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

    async def send_to_room(
        self, room_id: str, message: Dict[str, Any], exclude_user: Optional[str] = None
    ):
        """Send message to all users in a specific room/channel"""
        if room_id not in self.room_subscriptions:
            return

        for user_id in self.room_subscriptions[room_id]:
            if exclude_user and user_id == exclude_user:
                continue

            await self.send_personal_message(user_id, message)

    async def join_room(self, user_id: str, room_id: str):
        """Add user to a room with enhanced collaboration features"""
        if room_id not in self.room_subscriptions:
            self.room_subscriptions[room_id] = set()
            self.room_history[room_id] = []
            self.typing_status[room_id] = set()

        self.room_subscriptions[room_id].add(user_id)

        # Update user presence
        if user_id in self.user_presence:
            self.user_presence[user_id]["active_rooms"].add(room_id)

        # Send room history to new member
        if room_id in self.room_history:
            await self.send_personal_message(
                user_id,
                {
                    "type": MessageType.SYNC.value,
                    "data": {
                        "room_id": room_id,
                        "history": self.room_history[room_id][-50:],  # Last 50 messages
                        "members": list(self.room_subscriptions[room_id]),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                },
            )

        # Notify room members
        await self.send_to_room(
            room_id,
            {
                "type": MessageType.PRESENCE.value,
                "data": {
                    "action": "joined",
                    "user_id": user_id,
                    "room_id": room_id,
                    "members": list(self.room_subscriptions[room_id]),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            },
            exclude_user=user_id,
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
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    },
                )

    async def send_video_status_update(
        self,
        user_id: str,
        video_id: str,
        status: str,
        progress: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Send video generation status update"""
        message = {
            "type": MessageType.VIDEO_STATUS.value,
            "data": {
                "video_id": video_id,
                "status": status,
                "progress": progress,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {},
            },
        }

        await self.send_personal_message(user_id, message)

    async def send_metric_update(
        self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None
    ):
        """Broadcast metric update to all connected users"""
        message = {
            "type": MessageType.METRIC_UPDATE.value,
            "data": {
                "metric_name": metric_name,
                "value": value,
                "labels": labels or {},
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

        await self.broadcast(message)

    async def send_channel_update(
        self, channel_id: str, update_type: str, data: Dict[str, Any]
    ):
        """Send channel-specific update to subscribers"""
        room_id = f"channel:{channel_id}"

        message = {
            "type": MessageType.CHANNEL_UPDATE.value,
            "data": {
                "channel_id": channel_id,
                "update_type": update_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

        await self.send_to_room(room_id, message)

    async def handle_incoming_message(
        self, websocket: WebSocket, user_id: str, data: Dict[str, Any]
    ):
        """Process incoming WebSocket messages with collaboration features"""
        message_type = data.get("type")

        if message_type == MessageType.PING.value:
            # Respond to ping
            await websocket.send_json(
                {
                    "type": MessageType.PONG.value,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

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

        elif message_type == MessageType.COLLABORATION.value:
            # Handle collaboration message
            await self.handle_collaboration_message(user_id, data.get("data", {}))

        elif message_type == MessageType.TYPING.value:
            # Handle typing indicator
            room_id = data.get("room_id")
            is_typing = data.get("is_typing", False)
            if room_id:
                await self.update_typing_status(user_id, room_id, is_typing)

        elif message_type == MessageType.CURSOR.value:
            # Handle cursor position update
            await self.update_cursor_position(user_id, data.get("data", {}))

        elif message_type == MessageType.BROADCAST.value:
            # Broadcast message to room
            room_id = data.get("room_id")
            if room_id and user_id in self.room_subscriptions.get(room_id, set()):
                await self.broadcast_room_message(
                    user_id, room_id, data.get("data", {})
                )

        elif message_type == MessageType.PRIVATE_MESSAGE.value:
            # Send private message to another user
            target_user = data.get("target_user")
            message_data = data.get("data", {})
            if target_user:
                await self.send_private_message(user_id, target_user, message_data)

        else:
            # Handle custom message types
            logger.info(f"Received message from {user_id}: {message_type}")

    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return sum(len(connections) for connections in self.active_connections.values())

    def get_user_count(self) -> int:
        """Get number of connected users"""
        return len(self.active_connections)

    def get_room_info(self) -> Dict[str, Any]:
        """Get detailed information about active rooms"""
        return {
            room_id: {
                "member_count": len(subscribers),
                "members": list(subscribers),
                "typing_users": list(self.typing_status.get(room_id, set())),
                "message_count": len(self.room_history.get(room_id, [])),
            }
            for room_id, subscribers in self.room_subscriptions.items()
        }

    async def broadcast_presence_update(self, user_id: str, status: str):
        """Broadcast user presence update to all relevant rooms"""
        presence_data = {
            "type": MessageType.PRESENCE.value,
            "data": {
                "user_id": user_id,
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

        # Send to all rooms the user is in
        for room_id, subscribers in self.room_subscriptions.items():
            if user_id in subscribers:
                await self.send_to_room(room_id, presence_data, exclude_user=user_id)

    async def update_typing_status(self, user_id: str, room_id: str, is_typing: bool):
        """Update and broadcast typing status"""
        if room_id not in self.typing_status:
            self.typing_status[room_id] = set()

        if is_typing:
            self.typing_status[room_id].add(user_id)
        else:
            self.typing_status[room_id].discard(user_id)

        await self.broadcast_typing_status(room_id)

    async def broadcast_typing_status(self, room_id: str):
        """Broadcast current typing status for a room"""
        typing_users = list(self.typing_status.get(room_id, set()))

        await self.send_to_room(
            room_id,
            {
                "type": MessageType.TYPING.value,
                "data": {
                    "room_id": room_id,
                    "typing_users": typing_users,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            },
        )

    async def update_cursor_position(self, user_id: str, data: Dict[str, Any]):
        """Update and broadcast cursor position for collaborative editing"""
        self.cursor_positions[user_id] = {
            "position": data.get("position"),
            "selection": data.get("selection"),
            "document_id": data.get("document_id"),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Broadcast to users in the same document/room
        room_id = data.get("room_id") or f"doc:{data.get('document_id')}"
        if room_id:
            await self.send_to_room(
                room_id,
                {
                    "type": MessageType.CURSOR.value,
                    "data": {
                        "user_id": user_id,
                        "cursor": self.cursor_positions[user_id],
                    },
                },
                exclude_user=user_id,
            )

    async def handle_collaboration_message(self, user_id: str, data: Dict[str, Any]):
        """Handle real-time collaboration messages"""
        action = data.get("action")
        room_id = data.get("room_id")

        if not room_id:
            return

        collaboration_data = {
            "type": MessageType.COLLABORATION.value,
            "data": {
                "user_id": user_id,
                "action": action,
                "payload": data.get("payload"),
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

        # Store in room history
        if room_id not in self.room_history:
            self.room_history[room_id] = []

        self.room_history[room_id].append(collaboration_data)

        # Keep only last 100 messages
        if len(self.room_history[room_id]) > 100:
            self.room_history[room_id] = self.room_history[room_id][-100:]

        # Broadcast to room members
        await self.send_to_room(room_id, collaboration_data, exclude_user=user_id)

    async def broadcast_room_message(
        self, user_id: str, room_id: str, data: Dict[str, Any]
    ):
        """Broadcast a message to all room members"""
        message = {
            "type": MessageType.BROADCAST.value,
            "data": {
                "sender_id": user_id,
                "room_id": room_id,
                "content": data,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

        # Store in room history
        if room_id not in self.room_history:
            self.room_history[room_id] = []

        self.room_history[room_id].append(message)

        # Broadcast to room
        await self.send_to_room(room_id, message)

    async def send_private_message(
        self, sender_id: str, target_id: str, data: Dict[str, Any]
    ):
        """Send a private message between users"""
        message = {
            "type": MessageType.PRIVATE_MESSAGE.value,
            "data": {
                "sender_id": sender_id,
                "content": data,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

        await self.send_personal_message(target_id, message)

    async def create_collaboration_session(
        self, session_id: str, owner_id: str, metadata: Dict[str, Any]
    ):
        """Create a new collaboration session"""
        self.collaboration_sessions[session_id] = {
            "owner_id": owner_id,
            "created_at": datetime.utcnow().isoformat(),
            "participants": [owner_id],
            "metadata": metadata,
            "state": {},
        }

        return session_id

    async def join_collaboration_session(self, session_id: str, user_id: str):
        """Join an existing collaboration session"""
        if session_id in self.collaboration_sessions:
            session = self.collaboration_sessions[session_id]
            if user_id not in session["participants"]:
                session["participants"].append(user_id)

            # Send session state to new participant
            await self.send_personal_message(
                user_id,
                {
                    "type": MessageType.SYNC.value,
                    "data": {
                        "session_id": session_id,
                        "state": session["state"],
                        "participants": session["participants"],
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                },
            )

            return True
        return False

    def get_user_presence(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user presence information"""
        return self.user_presence.get(user_id)

    def get_online_users(self) -> List[str]:
        """Get list of online users"""
        return [
            user_id
            for user_id, presence in self.user_presence.items()
            if presence.get("status") == "online"
        ]


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
            "active_rooms": len(manager.room_subscriptions),
        }

        await manager.send_metric_update(
            metric_name="system.stats", value=1, labels=metrics
        )
