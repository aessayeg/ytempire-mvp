"""
WebSocket Middleware for Authentication and Authorization
"""

import logging
import jwt
from typing import Optional, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect, status
from app.core.config import settings
from app.core.security import decode_token

logger = logging.getLogger(__name__)


class WebSocketAuthMiddleware:
    """
    Middleware for WebSocket authentication
    """
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
    
    async def authenticate(
        self,
        websocket: WebSocket,
        token: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Authenticate WebSocket connection
        
        Args:
            websocket: WebSocket connection
            token: JWT token from query params or headers
        
        Returns:
            User data if authenticated, None otherwise
        """
        try:
            # Try to get token from query params first
            if not token:
                token = websocket.query_params.get('token')
            
            # Try to get from headers
            if not token:
                headers = dict(websocket.headers)
                auth_header = headers.get('authorization', '')
                if auth_header.startswith('Bearer '):
                    token = auth_header[7:]
            
            if not token:
                await self.reject_connection(
                    websocket,
                    "No authentication token provided"
                )
                return None
            
            # Decode token
            try:
                payload = decode_token(token)
                user_id = payload.get('sub')
                
                if not user_id:
                    await self.reject_connection(
                        websocket,
                        "Invalid token payload"
                    )
                    return None
                
                return {
                    'user_id': user_id,
                    'email': payload.get('email'),
                    'roles': payload.get('roles', []),
                    'token': token
                }
                
            except jwt.ExpiredSignatureError:
                await self.reject_connection(
                    websocket,
                    "Token has expired"
                )
                return None
            
            except jwt.InvalidTokenError as e:
                await self.reject_connection(
                    websocket,
                    f"Invalid token: {str(e)}"
                )
                return None
        
        except Exception as e:
            logger.error(f"WebSocket authentication error: {str(e)}")
            await self.reject_connection(
                websocket,
                "Authentication failed"
            )
            return None
    
    async def reject_connection(
        self,
        websocket: WebSocket,
        reason: str
    ):
        """
        Reject WebSocket connection with error message
        
        Args:
            websocket: WebSocket connection
            reason: Rejection reason
        """
        await websocket.accept()
        await websocket.send_json({
            'type': 'error',
            'error': 'authentication_failed',
            'message': reason
        })
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    
    async def authorize_resource(
        self,
        user_data: Dict[str, Any],
        resource_type: str,
        resource_id: str,
        action: str = "read"
    ) -> bool:
        """
        Authorize user access to resource
        
        Args:
            user_data: Authenticated user data
            resource_type: Type of resource (video, channel, etc.)
            resource_id: Resource ID
            action: Action to perform (read, write, delete)
        
        Returns:
            True if authorized, False otherwise
        """
        # Admin users have full access
        if 'admin' in user_data.get('roles', []):
            return True
        
        user_id = user_data.get('user_id')
        
        # Check resource-specific permissions
        if resource_type == 'video':
            return await self.check_video_permission(user_id, resource_id, action)
        elif resource_type == 'channel':
            return await self.check_channel_permission(user_id, resource_id, action)
        elif resource_type == 'room':
            return await self.check_room_permission(user_id, resource_id, action)
        
        # Default deny
        return False
    
    async def check_video_permission(
        self,
        user_id: str,
        video_id: str,
        action: str
    ) -> bool:
        """Check video access permission"""
        from app.db.session import AsyncSessionLocal
        from app.models.video import Video
        
        async with AsyncSessionLocal() as db:
            video = await db.get(Video, video_id)
            if not video:
                return False
            
            # Check ownership through channel
            from app.models.channel import Channel
            channel = await db.get(Channel, video.channel_id)
            if not channel:
                return False
            
            # Owner has full access
            if channel.user_id == user_id:
                return True
            
            # Collaborators have read access
            if action == "read" and user_id in (channel.collaborators or []):
                return True
            
            return False
    
    async def check_channel_permission(
        self,
        user_id: str,
        channel_id: str,
        action: str
    ) -> bool:
        """Check channel access permission"""
        from app.db.session import AsyncSessionLocal
        from app.models.channel import Channel
        
        async with AsyncSessionLocal() as db:
            channel = await db.get(Channel, channel_id)
            if not channel:
                return False
            
            # Owner has full access
            if channel.user_id == user_id:
                return True
            
            # Collaborators have limited access
            if user_id in (channel.collaborators or []):
                return action in ["read", "write"]
            
            return False
    
    async def check_room_permission(
        self,
        user_id: str,
        room_id: str,
        action: str
    ) -> bool:
        """Check room access permission"""
        # For now, authenticated users can join any room
        # You can implement more sophisticated logic here
        return True


class RateLimitMiddleware:
    """
    Rate limiting for WebSocket connections
    """
    
    def __init__(self, max_connections_per_user: int = 5):
        self.max_connections_per_user = max_connections_per_user
        self.user_connections: Dict[str, int] = {}
    
    async def check_rate_limit(
        self,
        user_id: str
    ) -> bool:
        """
        Check if user has exceeded connection limit
        
        Args:
            user_id: User ID
        
        Returns:
            True if within limits, False if exceeded
        """
        current_connections = self.user_connections.get(user_id, 0)
        
        if current_connections >= self.max_connections_per_user:
            logger.warning(f"User {user_id} exceeded WebSocket connection limit")
            return False
        
        return True
    
    def increment_connections(self, user_id: str):
        """Increment user connection count"""
        self.user_connections[user_id] = self.user_connections.get(user_id, 0) + 1
    
    def decrement_connections(self, user_id: str):
        """Decrement user connection count"""
        if user_id in self.user_connections:
            self.user_connections[user_id] -= 1
            if self.user_connections[user_id] <= 0:
                del self.user_connections[user_id]


# Singleton instances
websocket_auth = WebSocketAuthMiddleware()
rate_limiter = RateLimitMiddleware()