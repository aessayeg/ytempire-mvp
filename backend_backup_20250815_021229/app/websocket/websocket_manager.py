"""
WebSocket Foundation for Real-time Communication
Handles real-time updates for video generation, metrics, and notifications
"""
from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
from datetime import datetime
import logging
from enum import Enum
from dataclasses import dataclass, asdict
import redis.asyncio as redis
from collections import defaultdict

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """WebSocket message types"""
    # Video generation events
    VIDEO_GENERATION_STARTED = "video_generation_started"
    VIDEO_GENERATION_PROGRESS = "video_generation_progress"
    VIDEO_GENERATION_COMPLETED = "video_generation_completed"
    VIDEO_GENERATION_FAILED = "video_generation_failed"
    
    # Channel events
    CHANNEL_UPDATE = "channel_update"
    CHANNEL_ANALYTICS = "channel_analytics"
    
    # System events
    NOTIFICATION = "notification"
    METRICS_UPDATE = "metrics_update"
    COST_UPDATE = "cost_update"
    
    # User events
    USER_STATUS = "user_status"
    HEARTBEAT = "heartbeat"
    
    # Queue events
    QUEUE_UPDATE = "queue_update"
    JOB_STATUS = "job_status"

@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: MessageType
    data: Dict[str, Any]
    timestamp: str = None
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_json(self) -> str:
        """Convert message to JSON"""
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "channel_id": self.channel_id
        })

class ConnectionManager:
    """Manages WebSocket connections and message routing"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/4"):
        # Connection storage
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        self.channel_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Redis for pub/sub across multiple servers
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
    async def initialize(self):
        """Initialize Redis connection and start background tasks"""
        self.redis_client = await redis.from_url(self.redis_url)
        self.pubsub = self.redis_client.pubsub()
        
        # Start message listener
        task = asyncio.create_task(self._redis_listener())
        self.background_tasks.add(task)
        
        # Start heartbeat monitor
        task = asyncio.create_task(self._heartbeat_monitor())
        self.background_tasks.add(task)
        
        logger.info("WebSocket manager initialized")
        
    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        user_id: Optional[str] = None
    ):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        
        # Store connection
        self.active_connections[client_id] = websocket
        
        if user_id:
            self.user_connections[user_id].add(client_id)
            
        # Send connection confirmation
        welcome_msg = WebSocketMessage(
            type=MessageType.USER_STATUS,
            data={
                "status": "connected",
                "client_id": client_id,
                "server_time": datetime.utcnow().isoformat()
            },
            user_id=user_id
        )
        await self.send_personal_message(welcome_msg, client_id)
        
        # Subscribe to Redis channel for this client
        if self.pubsub:
            await self.pubsub.subscribe(f"ws:client:{client_id}")
            if user_id:
                await self.pubsub.subscribe(f"ws:user:{user_id}")
                
        logger.info(f"Client {client_id} connected (user: {user_id})")
        
    async def disconnect(self, client_id: str):
        """Handle client disconnection"""
        if client_id in self.active_connections:
            # Remove from connection storage
            del self.active_connections[client_id]
            
            # Remove from user connections
            for user_id, connections in self.user_connections.items():
                if client_id in connections:
                    connections.remove(client_id)
                    if not connections:
                        del self.user_connections[user_id]
                    break
                    
            # Remove from channel subscriptions
            for channel_id, subscribers in self.channel_subscriptions.items():
                if client_id in subscribers:
                    subscribers.remove(client_id)
                    
            # Unsubscribe from Redis channels
            if self.pubsub:
                await self.pubsub.unsubscribe(f"ws:client:{client_id}")
                
            logger.info(f"Client {client_id} disconnected")
            
    async def subscribe_to_channel(self, client_id: str, channel_id: str):
        """Subscribe a client to channel updates"""
        self.channel_subscriptions[channel_id].add(client_id)
        
        if self.pubsub:
            await self.pubsub.subscribe(f"ws:channel:{channel_id}")
            
        logger.info(f"Client {client_id} subscribed to channel {channel_id}")
        
    async def unsubscribe_from_channel(self, client_id: str, channel_id: str):
        """Unsubscribe a client from channel updates"""
        if client_id in self.channel_subscriptions[channel_id]:
            self.channel_subscriptions[channel_id].remove(client_id)
            
        logger.info(f"Client {client_id} unsubscribed from channel {channel_id}")
        
    async def send_personal_message(self, message: WebSocketMessage, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_text(message.to_json())
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {e}")
                await self.disconnect(client_id)
                
    async def send_user_message(self, message: WebSocketMessage, user_id: str):
        """Send message to all connections of a user"""
        if user_id in self.user_connections:
            for client_id in self.user_connections[user_id]:
                await self.send_personal_message(message, client_id)
                
    async def send_channel_message(self, message: WebSocketMessage, channel_id: str):
        """Send message to all subscribers of a channel"""
        if channel_id in self.channel_subscriptions:
            for client_id in self.channel_subscriptions[channel_id]:
                await self.send_personal_message(message, client_id)
                
        # Also publish to Redis for other servers
        if self.redis_client:
            await self.redis_client.publish(
                f"ws:channel:{channel_id}",
                message.to_json()
            )
            
    async def broadcast(self, message: WebSocketMessage):
        """Broadcast message to all connected clients"""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message.to_json())
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)
                
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
            
    async def handle_client_message(self, client_id: str, data: str):
        """Handle incoming message from client"""
        try:
            message = json.loads(data)
            message_type = message.get("type")
            
            if message_type == "heartbeat":
                # Respond to heartbeat
                pong = WebSocketMessage(
                    type=MessageType.HEARTBEAT,
                    data={"status": "pong", "timestamp": datetime.utcnow().isoformat()}
                )
                await self.send_personal_message(pong, client_id)
                
            elif message_type == "subscribe_channel":
                channel_id = message.get("channel_id")
                if channel_id:
                    await self.subscribe_to_channel(client_id, channel_id)
                    
            elif message_type == "unsubscribe_channel":
                channel_id = message.get("channel_id")
                if channel_id:
                    await self.unsubscribe_from_channel(client_id, channel_id)
                    
            else:
                # Handle other message types
                logger.info(f"Received message from {client_id}: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from client {client_id}: {data}")
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            
    async def _redis_listener(self):
        """Listen for Redis pub/sub messages"""
        if not self.pubsub:
            return
            
        try:
            while True:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True)
                if message:
                    channel = message["channel"].decode()
                    data = message["data"].decode()
                    
                    # Route message based on channel
                    if channel.startswith("ws:client:"):
                        client_id = channel.replace("ws:client:", "")
                        if client_id in self.active_connections:
                            await self.active_connections[client_id].send_text(data)
                            
                    elif channel.startswith("ws:user:"):
                        user_id = channel.replace("ws:user:", "")
                        if user_id in self.user_connections:
                            for client_id in self.user_connections[user_id]:
                                if client_id in self.active_connections:
                                    await self.active_connections[client_id].send_text(data)
                                    
                    elif channel.startswith("ws:channel:"):
                        channel_id = channel.replace("ws:channel:", "")
                        if channel_id in self.channel_subscriptions:
                            for client_id in self.channel_subscriptions[channel_id]:
                                if client_id in self.active_connections:
                                    await self.active_connections[client_id].send_text(data)
                                    
                await asyncio.sleep(0.01)
                
        except asyncio.CancelledError:
            logger.info("Redis listener cancelled")
        except Exception as e:
            logger.error(f"Redis listener error: {e}")
            
    async def _heartbeat_monitor(self):
        """Monitor connection health with heartbeats"""
        try:
            while True:
                # Send heartbeat to all connections
                heartbeat = WebSocketMessage(
                    type=MessageType.HEARTBEAT,
                    data={"status": "ping"}
                )
                
                disconnected = []
                for client_id, websocket in self.active_connections.items():
                    try:
                        await websocket.send_text(heartbeat.to_json())
                    except:
                        disconnected.append(client_id)
                        
                # Clean up disconnected clients
                for client_id in disconnected:
                    await self.disconnect(client_id)
                    
                # Wait before next heartbeat
                await asyncio.sleep(30)
                
        except asyncio.CancelledError:
            logger.info("Heartbeat monitor cancelled")
        except Exception as e:
            logger.error(f"Heartbeat monitor error: {e}")
            
    async def cleanup(self):
        """Clean up resources"""
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
            
        # Close all WebSocket connections
        for client_id in list(self.active_connections.keys()):
            await self.disconnect(client_id)
            
        logger.info("WebSocket manager cleaned up")


# Global connection manager instance
manager = ConnectionManager()


class VideoGenerationNotifier:
    """Handles video generation progress notifications"""
    
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        
    async def notify_generation_started(
        self,
        job_id: str,
        user_id: str,
        channel_id: str,
        video_details: Dict[str, Any]
    ):
        """Notify that video generation has started"""
        message = WebSocketMessage(
            type=MessageType.VIDEO_GENERATION_STARTED,
            data={
                "job_id": job_id,
                "video_details": video_details,
                "estimated_time": "5-10 minutes"
            },
            user_id=user_id,
            channel_id=channel_id
        )
        
        await self.manager.send_user_message(message, user_id)
        await self.manager.send_channel_message(message, channel_id)
        
    async def notify_generation_progress(
        self,
        job_id: str,
        user_id: str,
        channel_id: str,
        progress: int,
        current_step: str
    ):
        """Notify video generation progress"""
        message = WebSocketMessage(
            type=MessageType.VIDEO_GENERATION_PROGRESS,
            data={
                "job_id": job_id,
                "progress": progress,
                "current_step": current_step
            },
            user_id=user_id,
            channel_id=channel_id
        )
        
        await self.manager.send_user_message(message, user_id)
        
    async def notify_generation_completed(
        self,
        job_id: str,
        user_id: str,
        channel_id: str,
        video_data: Dict[str, Any]
    ):
        """Notify that video generation is complete"""
        message = WebSocketMessage(
            type=MessageType.VIDEO_GENERATION_COMPLETED,
            data={
                "job_id": job_id,
                "video_data": video_data,
                "success": True
            },
            user_id=user_id,
            channel_id=channel_id
        )
        
        await self.manager.send_user_message(message, user_id)
        await self.manager.send_channel_message(message, channel_id)
        
    async def notify_generation_failed(
        self,
        job_id: str,
        user_id: str,
        channel_id: str,
        error: str
    ):
        """Notify that video generation failed"""
        message = WebSocketMessage(
            type=MessageType.VIDEO_GENERATION_FAILED,
            data={
                "job_id": job_id,
                "error": error,
                "success": False
            },
            user_id=user_id,
            channel_id=channel_id
        )
        
        await self.manager.send_user_message(message, user_id)