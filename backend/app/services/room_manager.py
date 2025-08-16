"""
Room Manager Service for WebSocket Collaboration
Handles real-time collaboration rooms, user presence, and message broadcasting
"""

import logging
import json
import asyncio
from typing import Dict, Set, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import uuid

from fastapi import WebSocket
from redis import asyncio as aioredis
from app.core.config import settings
from app.services.websocket_manager import ConnectionManager

logger = logging.getLogger(__name__)


class Room:
    """Represents a collaboration room"""

    def __init__(self, room_id: str, room_type: str = "channel", metadata: Dict = None):
        self.id = room_id
        self.type = room_type
        self.created_at = datetime.utcnow()
        self.metadata = metadata or {}
        self.users: Set[str] = set()
        self.active_connections: Dict[str, WebSocket] = {}
        self.message_history: List[Dict] = []
        self.max_history = 100

    def add_user(self, user_id: str, websocket: WebSocket):
        """Add user to room"""
        self.users.add(user_id)
        self.active_connections[user_id] = websocket

    def remove_user(self, user_id: str):
        """Remove user from room"""
        self.users.discard(user_id)
        self.active_connections.pop(user_id, None)

    def get_user_count(self) -> int:
        """Get number of users in room"""
        return len(self.users)

    def add_message(self, message: Dict):
        """Add message to history"""
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)

    def to_dict(self) -> Dict:
        """Convert room to dictionary"""
        return {
            "id": self.id,
            "type": self.type,
            "created_at": self.created_at.isoformat(),
            "user_count": self.get_user_count(),
            "users": list(self.users),
            "metadata": self.metadata,
        }


class RoomManager:
    """
    Manages WebSocket collaboration rooms for real-time features
    """

    def __init__(self):
        self.rooms: Dict[str, Room] = {}
        self.user_rooms: Dict[str, Set[str]] = defaultdict(set)
        self.connection_manager = ConnectionManager()
        self.redis_client: Optional[aioredis.Redis] = None
        self.pubsub_task: Optional[asyncio.Task] = None
        self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis connection for pub/sub"""
        try:
            self.redis_client = aioredis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {str(e)}")

    async def create_room(
        self, room_id: str, room_type: str = "channel", metadata: Dict = None
    ) -> Room:
        """
        Create a new collaboration room

        Args:
            room_id: Unique room identifier
            room_type: Type of room (channel, video, team, etc.)
            metadata: Additional room metadata

        Returns:
            Created room instance
        """
        if room_id in self.rooms:
            return self.rooms[room_id]

        room = Room(room_id, room_type, metadata)
        self.rooms[room_id] = room

        # Store in Redis for persistence
        if self.redis_client:
            await self.redis_client.hset(
                f"room:{room_id}",
                mapping={
                    "type": room_type,
                    "created_at": room.created_at.isoformat(),
                    "metadata": json.dumps(metadata or {}),
                },
            )
            await self.redis_client.expire(f"room:{room_id}", 86400)  # 24 hour TTL

        logger.info(f"Created room {room_id} of type {room_type}")
        return room

    async def join_room(self, room_id: str, user_id: str, websocket: WebSocket) -> bool:
        """
        Add user to a room

        Args:
            room_id: Room to join
            user_id: User identifier
            websocket: User's WebSocket connection

        Returns:
            Success status
        """
        # Create room if it doesn't exist
        if room_id not in self.rooms:
            await self.create_room(room_id)

        room = self.rooms[room_id]
        room.add_user(user_id, websocket)
        self.user_rooms[user_id].add(room_id)

        # Update Redis
        if self.redis_client:
            await self.redis_client.sadd(f"room:{room_id}:users", user_id)
            await self.redis_client.hset(
                f"user:{user_id}:rooms", room_id, datetime.utcnow().isoformat()
            )

        # Notify other users
        await self.broadcast_to_room(
            room_id,
            {
                "type": "user_joined",
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "user_count": room.get_user_count(),
            },
            exclude_user=user_id,
        )

        # Send room state to joining user
        await self.send_room_state(room_id, user_id)

        logger.info(f"User {user_id} joined room {room_id}")
        return True

    async def leave_room(self, room_id: str, user_id: str) -> bool:
        """
        Remove user from a room

        Args:
            room_id: Room to leave
            user_id: User identifier

        Returns:
            Success status
        """
        if room_id not in self.rooms:
            return False

        room = self.rooms[room_id]
        room.remove_user(user_id)
        self.user_rooms[user_id].discard(room_id)

        # Update Redis
        if self.redis_client:
            await self.redis_client.srem(f"room:{room_id}:users", user_id)
            await self.redis_client.hdel(f"user:{user_id}:rooms", room_id)

        # Notify other users
        await self.broadcast_to_room(
            room_id,
            {
                "type": "user_left",
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "user_count": room.get_user_count(),
            },
        )

        # Clean up empty room
        if room.get_user_count() == 0:
            await self.cleanup_room(room_id)

        logger.info(f"User {user_id} left room {room_id}")
        return True

    async def broadcast_to_room(
        self, room_id: str, message: Dict[str, Any], exclude_user: Optional[str] = None
    ):
        """
        Broadcast message to all users in a room

        Args:
            room_id: Target room
            message: Message to broadcast
            exclude_user: User to exclude from broadcast
        """
        if room_id not in self.rooms:
            return

        room = self.rooms[room_id]

        # Add to message history
        room.add_message(message)

        # Broadcast to all connections
        for user_id, websocket in room.active_connections.items():
            if user_id != exclude_user:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send message to {user_id}: {str(e)}")
                    # Remove failed connection
                    await self.leave_room(room_id, user_id)

        # Publish to Redis for cross-server broadcasting
        if self.redis_client:
            await self.redis_client.publish(f"room:{room_id}", json.dumps(message))

    async def send_to_user_in_room(
        self, room_id: str, user_id: str, message: Dict[str, Any]
    ):
        """
        Send message to specific user in a room

        Args:
            room_id: Target room
            user_id: Target user
            message: Message to send
        """
        if room_id not in self.rooms:
            return

        room = self.rooms[room_id]
        if user_id in room.active_connections:
            try:
                await room.active_connections[user_id].send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to {user_id}: {str(e)}")
                await self.leave_room(room_id, user_id)

    async def send_room_state(self, room_id: str, user_id: str):
        """Send current room state to a user"""
        if room_id not in self.rooms:
            return

        room = self.rooms[room_id]
        state = {
            "type": "room_state",
            "room": room.to_dict(),
            "message_history": room.message_history[-20:],  # Last 20 messages
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self.send_to_user_in_room(room_id, user_id, state)

    async def handle_room_message(
        self, room_id: str, user_id: str, message: Dict[str, Any]
    ):
        """
        Handle incoming message from user in room

        Args:
            room_id: Source room
            user_id: Source user
            message: Message content
        """
        # Add metadata
        message["room_id"] = room_id
        message["user_id"] = user_id
        message["timestamp"] = datetime.utcnow().isoformat()

        # Handle different message types
        message_type = message.get("type", "message")

        if message_type == "message":
            # Regular chat message
            await self.broadcast_to_room(room_id, message)

        elif message_type == "typing":
            # Typing indicator
            await self.broadcast_to_room(
                room_id,
                {
                    "type": "typing",
                    "user_id": user_id,
                    "is_typing": message.get("is_typing", False),
                },
                exclude_user=user_id,
            )

        elif message_type == "video_update":
            # Video generation update
            await self.broadcast_to_room(room_id, message)

        elif message_type == "cursor_position":
            # Collaborative cursor position
            await self.broadcast_to_room(
                room_id,
                {
                    "type": "cursor_position",
                    "user_id": user_id,
                    "position": message.get("position"),
                },
                exclude_user=user_id,
            )

    async def cleanup_room(self, room_id: str):
        """Clean up empty room"""
        if room_id in self.rooms:
            room = self.rooms[room_id]
            if room.get_user_count() == 0:
                del self.rooms[room_id]

                # Clean Redis
                if self.redis_client:
                    await self.redis_client.delete(f"room:{room_id}")
                    await self.redis_client.delete(f"room:{room_id}:users")

                logger.info(f"Cleaned up empty room {room_id}")

    async def get_room_info(self, room_id: str) -> Optional[Dict]:
        """Get room information"""
        if room_id not in self.rooms:
            # Try to load from Redis
            if self.redis_client:
                room_data = await self.redis_client.hgetall(f"room:{room_id}")
                if room_data:
                    return {
                        "id": room_id,
                        "type": room_data.get("type"),
                        "created_at": room_data.get("created_at"),
                        "metadata": json.loads(room_data.get("metadata", "{}")),
                        "user_count": 0,
                    }
            return None

        return self.rooms[room_id].to_dict()

    async def list_user_rooms(self, user_id: str) -> List[str]:
        """List all rooms a user is in"""
        return list(self.user_rooms.get(user_id, set()))

    async def list_active_rooms(self) -> List[Dict]:
        """List all active rooms"""
        return [room.to_dict() for room in self.rooms.values()]

    async def disconnect_user(self, user_id: str):
        """Disconnect user from all rooms"""
        rooms = list(self.user_rooms.get(user_id, set()))
        for room_id in rooms:
            await self.leave_room(room_id, user_id)

    async def start_pubsub(self):
        """Start Redis pub/sub listener for cross-server communication"""
        if not self.redis_client:
            return

        async def pubsub_listener():
            pubsub = self.redis_client.pubsub()
            await pubsub.psubscribe("room:*")

            async for message in pubsub.listen():
                if message["type"] == "pmessage":
                    room_id = message["channel"].split(":", 1)[1]
                    if room_id in self.rooms:
                        try:
                            data = json.loads(message["data"])
                            # Broadcast to local connections
                            for user_id, ws in self.rooms[
                                room_id
                            ].active_connections.items():
                                await ws.send_json(data)
                        except Exception as e:
                            logger.error(f"Pubsub error: {str(e)}")

        self.pubsub_task = asyncio.create_task(pubsub_listener())

    async def stop_pubsub(self):
        """Stop Redis pub/sub listener"""
        if self.pubsub_task:
            self.pubsub_task.cancel()
            try:
                await self.pubsub_task
            except asyncio.CancelledError:
                pass


# Singleton instance
room_manager = RoomManager()
