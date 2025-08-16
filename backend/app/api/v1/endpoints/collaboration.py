"""
Real-time Collaboration API Endpoints
Handles collaborative editing, presence, and team features
"""

from typing import List, Dict, Any, Optional
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field
from datetime import datetime
import json
import uuid
import asyncio

from app.api.v1.endpoints.auth import get_current_verified_user
from app.models.user import User
from app.services.websocket_manager import manager
from app.db.session import get_db
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


class CollaborationSessionRequest(BaseModel):
    """Request to create a collaboration session"""

    name: str = Field(..., description="Session name")
    type: str = Field(..., description="Session type: document, dashboard, channel")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    invite_users: Optional[List[str]] = Field(None, description="User IDs to invite")


class CollaborationActionRequest(BaseModel):
    """Request for collaboration action"""

    session_id: str
    action: str = Field(..., description="Action type: edit, comment, annotate")
    payload: Dict[str, Any]


class PresenceUpdateRequest(BaseModel):
    """Request to update user presence"""

    status: str = Field(..., description="Status: online, away, busy, offline")
    custom_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RoomMessageRequest(BaseModel):
    """Request to send message to a room"""

    room_id: str
    message: str
    message_type: str = Field(
        default="text", description="Message type: text, action, system"
    )
    metadata: Optional[Dict[str, Any]] = None


@router.websocket("/ws/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket, user_id: str, db: AsyncSession = Depends(get_db)
):
    """
    Main WebSocket endpoint for real-time collaboration
    """
    # Verify user authentication (simplified for WebSocket)
    # In production, implement proper WebSocket authentication

    await manager.connect(websocket, user_id, {"source": "collaboration_api"})

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)

            # Process message
            await manager.handle_incoming_message(websocket, user_id, message)

    except WebSocketDisconnect:
        await manager.disconnect(websocket, user_id)
    except Exception as e:
        await manager.disconnect(websocket, user_id)
        raise


@router.post("/sessions")
async def create_collaboration_session(
    request: CollaborationSessionRequest,
    current_user: User = Depends(get_current_verified_user),
):
    """Create a new collaboration session"""
    try:
        session_id = f"collab_{uuid.uuid4().hex[:12]}"

        # Create session in manager
        await manager.create_collaboration_session(
            session_id=session_id,
            owner_id=str(current_user.id),
            metadata={
                "name": request.name,
                "type": request.type,
                "created_at": datetime.utcnow().isoformat(),
                **request.metadata,
            },
        )

        # Auto-join the creator to the session
        await manager.join_collaboration_session(session_id, str(current_user.id))

        # Invite other users if specified
        if request.invite_users:
            for invited_user_id in request.invite_users:
                await manager.send_personal_message(
                    invited_user_id,
                    {
                        "type": "invitation",
                        "data": {
                            "session_id": session_id,
                            "session_name": request.name,
                            "invited_by": str(current_user.id),
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    },
                )

        return {
            "success": True,
            "session_id": session_id,
            "message": f"Collaboration session '{request.name}' created successfully",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create collaboration session: {str(e)}",
        )


@router.post("/sessions/{session_id}/join")
async def join_collaboration_session(
    session_id: str, current_user: User = Depends(get_current_verified_user)
):
    """Join an existing collaboration session"""
    try:
        success = await manager.join_collaboration_session(
            session_id=session_id, user_id=str(current_user.id)
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session '{session_id}' not found",
            )

        return {
            "success": True,
            "session_id": session_id,
            "message": "Successfully joined collaboration session",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to join session: {str(e)}",
        )


@router.post("/sessions/{session_id}/action")
async def send_collaboration_action(
    session_id: str,
    request: CollaborationActionRequest,
    current_user: User = Depends(get_current_verified_user),
):
    """Send a collaboration action to session participants"""
    try:
        # Send collaboration action through WebSocket
        await manager.handle_collaboration_message(
            user_id=str(current_user.id),
            data={
                "room_id": f"session:{session_id}",
                "action": request.action,
                "payload": request.payload,
            },
        )

        return {"success": True, "message": "Collaboration action sent successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send collaboration action: {str(e)}",
        )


@router.post("/rooms/{room_id}/join")
async def join_room(
    room_id: str, current_user: User = Depends(get_current_verified_user)
):
    """Join a collaboration room"""
    try:
        await manager.join_room(str(current_user.id), room_id)

        return {
            "success": True,
            "room_id": room_id,
            "message": f"Successfully joined room '{room_id}'",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to join room: {str(e)}",
        )


@router.post("/rooms/{room_id}/leave")
async def leave_room(
    room_id: str, current_user: User = Depends(get_current_verified_user)
):
    """Leave a collaboration room"""
    try:
        await manager.leave_room(str(current_user.id), room_id)

        return {
            "success": True,
            "room_id": room_id,
            "message": f"Successfully left room '{room_id}'",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to leave room: {str(e)}",
        )


@router.post("/rooms/{room_id}/message")
async def send_room_message(
    room_id: str,
    request: RoomMessageRequest,
    current_user: User = Depends(get_current_verified_user),
):
    """Send a message to a room"""
    try:
        await manager.broadcast_room_message(
            user_id=str(current_user.id),
            room_id=room_id,
            data={
                "message": request.message,
                "type": request.message_type,
                "metadata": request.metadata,
            },
        )

        return {"success": True, "message": "Message sent successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send message: {str(e)}",
        )


@router.post("/presence/update")
async def update_presence(
    request: PresenceUpdateRequest,
    current_user: User = Depends(get_current_verified_user),
):
    """Update user presence status"""
    try:
        user_id = str(current_user.id)

        # Update presence in manager
        if user_id in manager.user_presence:
            manager.user_presence[user_id]["status"] = request.status
            manager.user_presence[user_id]["custom_message"] = request.custom_message
            manager.user_presence[user_id]["last_seen"] = datetime.utcnow().isoformat()

            if request.metadata:
                manager.user_presence[user_id]["metadata"].update(request.metadata)

        # Broadcast presence update
        await manager.broadcast_presence_update(user_id, request.status)

        return {
            "success": True,
            "status": request.status,
            "message": "Presence updated successfully",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update presence: {str(e)}",
        )


@router.get("/presence/online")
async def get_online_users(current_user: User = Depends(get_current_verified_user)):
    """Get list of online users"""
    try:
        online_users = manager.get_online_users()

        # Get detailed presence info for each user
        presence_info = []
        for user_id in online_users:
            presence = manager.get_user_presence(user_id)
            if presence:
                presence_info.append(
                    {
                        "user_id": user_id,
                        "status": presence.get("status"),
                        "last_seen": presence.get("last_seen"),
                        "custom_message": presence.get("custom_message"),
                        "active_rooms": list(presence.get("active_rooms", set())),
                    }
                )

        return {
            "success": True,
            "online_count": len(online_users),
            "users": presence_info,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get online users: {str(e)}",
        )


@router.get("/rooms")
async def get_active_rooms(current_user: User = Depends(get_current_verified_user)):
    """Get information about active collaboration rooms"""
    try:
        room_info = manager.get_room_info()

        return {"success": True, "total_rooms": len(room_info), "rooms": room_info}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get room info: {str(e)}",
        )


@router.get("/stats")
async def get_collaboration_stats(
    current_user: User = Depends(get_current_verified_user),
):
    """Get real-time collaboration statistics"""
    try:
        stats = {
            "active_connections": manager.get_connection_count(),
            "online_users": manager.get_user_count(),
            "active_rooms": len(manager.room_subscriptions),
            "active_sessions": len(manager.collaboration_sessions),
            "rooms": manager.get_room_info(),
            "timestamp": datetime.utcnow().isoformat(),
        }

        return {"success": True, **stats}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}",
        )


@router.post("/typing/{room_id}")
async def update_typing_status(
    room_id: str,
    is_typing: bool,
    current_user: User = Depends(get_current_verified_user),
):
    """Update typing status in a room"""
    try:
        await manager.update_typing_status(
            user_id=str(current_user.id), room_id=room_id, is_typing=is_typing
        )

        return {"success": True, "message": "Typing status updated"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update typing status: {str(e)}",
        )


@router.post("/cursor/update")
async def update_cursor_position(
    document_id: str,
    position: Dict[str, Any],
    selection: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_verified_user),
):
    """Update cursor position for collaborative editing"""
    try:
        await manager.update_cursor_position(
            user_id=str(current_user.id),
            data={
                "document_id": document_id,
                "position": position,
                "selection": selection,
                "room_id": f"doc:{document_id}",
            },
        )

        return {"success": True, "message": "Cursor position updated"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update cursor position: {str(e)}",
        )
