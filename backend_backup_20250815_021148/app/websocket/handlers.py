"""
WebSocket Handlers for Different Real-time Features
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

from app.services.websocket_manager import ConnectionManager
from app.services.room_manager import room_manager
from app.services.notification_service import notification_service

logger = logging.getLogger(__name__)


class BaseWebSocketHandler:
    """Base handler for WebSocket connections"""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
    
    async def handle_connect(self, websocket: WebSocket, client_id: str):
        """Handle new WebSocket connection"""
        await self.connection_manager.connect(websocket, client_id)
        await websocket.send_json({
            'type': 'connection',
            'status': 'connected',
            'client_id': client_id,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def handle_disconnect(self, client_id: str):
        """Handle WebSocket disconnection"""
        await self.connection_manager.disconnect(client_id)
    
    async def handle_message(self, client_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        raise NotImplementedError


class VideoUpdateHandler(BaseWebSocketHandler):
    """Handler for video generation updates"""
    
    async def handle_message(self, client_id: str, message: Dict[str, Any]):
        """Handle video-related messages"""
        message_type = message.get('type')
        
        if message_type == 'subscribe_video':
            # Subscribe to video updates
            video_id = message.get('video_id')
            if video_id:
                await self.subscribe_to_video(client_id, video_id)
        
        elif message_type == 'unsubscribe_video':
            # Unsubscribe from video updates
            video_id = message.get('video_id')
            if video_id:
                await self.unsubscribe_from_video(client_id, video_id)
        
        elif message_type == 'video_progress':
            # Send video generation progress
            await self.send_video_progress(client_id, message)
    
    async def subscribe_to_video(self, client_id: str, video_id: str):
        """Subscribe client to video updates"""
        # Store subscription in Redis or memory
        await self.connection_manager.send_to_client(client_id, {
            'type': 'subscription',
            'resource': 'video',
            'video_id': video_id,
            'status': 'subscribed',
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def unsubscribe_from_video(self, client_id: str, video_id: str):
        """Unsubscribe client from video updates"""
        await self.connection_manager.send_to_client(client_id, {
            'type': 'subscription',
            'resource': 'video',
            'video_id': video_id,
            'status': 'unsubscribed',
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def send_video_progress(self, client_id: str, progress_data: Dict[str, Any]):
        """Send video generation progress update"""
        update = {
            'type': 'video_progress',
            'video_id': progress_data.get('video_id'),
            'stage': progress_data.get('stage'),
            'progress': progress_data.get('progress', 0),
            'message': progress_data.get('message'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to all subscribers of this video
        await self.connection_manager.broadcast(update)
    
    async def send_video_complete(self, video_id: str, result: Dict[str, Any]):
        """Send video completion notification"""
        update = {
            'type': 'video_complete',
            'video_id': video_id,
            'success': result.get('success', False),
            'youtube_url': result.get('youtube_url'),
            'error': result.get('error'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.connection_manager.broadcast(update)


class ChannelUpdateHandler(BaseWebSocketHandler):
    """Handler for channel-related updates"""
    
    async def handle_message(self, client_id: str, message: Dict[str, Any]):
        """Handle channel-related messages"""
        message_type = message.get('type')
        
        if message_type == 'subscribe_channel':
            channel_id = message.get('channel_id')
            if channel_id:
                await self.subscribe_to_channel(client_id, channel_id)
        
        elif message_type == 'channel_analytics':
            await self.send_channel_analytics(client_id, message)
        
        elif message_type == 'channel_status':
            await self.send_channel_status(client_id, message)
    
    async def subscribe_to_channel(self, client_id: str, channel_id: str):
        """Subscribe to channel updates"""
        await self.connection_manager.send_to_client(client_id, {
            'type': 'subscription',
            'resource': 'channel',
            'channel_id': channel_id,
            'status': 'subscribed',
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def send_channel_analytics(self, client_id: str, analytics: Dict[str, Any]):
        """Send channel analytics update"""
        update = {
            'type': 'channel_analytics',
            'channel_id': analytics.get('channel_id'),
            'views': analytics.get('views'),
            'subscribers': analytics.get('subscribers'),
            'revenue': analytics.get('revenue'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.connection_manager.send_to_client(client_id, update)
    
    async def send_channel_status(self, client_id: str, status: Dict[str, Any]):
        """Send channel status update"""
        update = {
            'type': 'channel_status',
            'channel_id': status.get('channel_id'),
            'health_score': status.get('health_score'),
            'quota_used': status.get('quota_used'),
            'videos_today': status.get('videos_today'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.connection_manager.send_to_client(client_id, update)


class CollaborationHandler(BaseWebSocketHandler):
    """Handler for real-time collaboration features"""
    
    async def handle_connect(self, websocket: WebSocket, client_id: str, room_id: Optional[str] = None):
        """Handle collaboration connection"""
        await super().handle_connect(websocket, client_id)
        
        if room_id:
            # Auto-join room if provided
            await room_manager.join_room(room_id, client_id, websocket)
    
    async def handle_disconnect(self, client_id: str):
        """Handle collaboration disconnection"""
        # Leave all rooms
        await room_manager.disconnect_user(client_id)
        await super().handle_disconnect(client_id)
    
    async def handle_message(self, client_id: str, message: Dict[str, Any]):
        """Handle collaboration messages"""
        message_type = message.get('type')
        
        if message_type == 'join_room':
            room_id = message.get('room_id')
            if room_id:
                websocket = self.connection_manager.get_websocket(client_id)
                if websocket:
                    await room_manager.join_room(room_id, client_id, websocket)
        
        elif message_type == 'leave_room':
            room_id = message.get('room_id')
            if room_id:
                await room_manager.leave_room(room_id, client_id)
        
        elif message_type == 'room_message':
            room_id = message.get('room_id')
            if room_id:
                await room_manager.handle_room_message(room_id, client_id, message)
        
        elif message_type == 'typing':
            room_id = message.get('room_id')
            if room_id:
                await room_manager.handle_room_message(room_id, client_id, {
                    'type': 'typing',
                    'is_typing': message.get('is_typing', False)
                })
        
        elif message_type == 'cursor_position':
            room_id = message.get('room_id')
            if room_id:
                await room_manager.handle_room_message(room_id, client_id, {
                    'type': 'cursor_position',
                    'position': message.get('position')
                })


class NotificationHandler(BaseWebSocketHandler):
    """Handler for real-time notifications"""
    
    async def handle_message(self, client_id: str, message: Dict[str, Any]):
        """Handle notification-related messages"""
        message_type = message.get('type')
        
        if message_type == 'mark_read':
            notification_id = message.get('notification_id')
            if notification_id:
                await self.mark_notification_read(client_id, notification_id)
        
        elif message_type == 'get_unread':
            await self.send_unread_notifications(client_id)
    
    async def mark_notification_read(self, client_id: str, notification_id: str):
        """Mark notification as read"""
        await notification_service.mark_as_read(notification_id)
        
        await self.connection_manager.send_to_client(client_id, {
            'type': 'notification_read',
            'notification_id': notification_id,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def send_unread_notifications(self, client_id: str):
        """Send unread notifications to client"""
        unread = await notification_service.get_unread_notifications(client_id)
        
        await self.connection_manager.send_to_client(client_id, {
            'type': 'unread_notifications',
            'count': len(unread),
            'notifications': unread,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def send_notification(self, user_id: str, notification: Dict[str, Any]):
        """Send real-time notification to user"""
        await self.connection_manager.send_to_user(user_id, {
            'type': 'notification',
            'title': notification.get('title'),
            'message': notification.get('message'),
            'notification_type': notification.get('type', 'info'),
            'timestamp': datetime.utcnow().isoformat()
        })