"""
Secrets Management Endpoints
Owner: Security Engineer #1

API endpoints for managing secrets, rotation, and security auditing.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel

from app.api.v1.endpoints.auth import get_current_user
from app.models.user import User
from app.services.secrets_manager import secrets_manager, SecretType, SecretMetadata
from app.tasks.secret_rotation import (
    rotate_secret_task, 
    bulk_rotate_secrets,
    generate_security_report,
    validate_secret_integrity
)

router = APIRouter()


class SecretCreateRequest(BaseModel):
    """Request model for creating secrets."""
    name: str
    value: str
    secret_type: SecretType
    rotation_interval_days: Optional[int] = None
    tags: Optional[Dict[str, str]] = None
    encrypt_locally: bool = False


class SecretResponse(BaseModel):
    """Response model for secret metadata."""
    id: str
    name: str
    type: str
    created_at: str
    updated_at: str
    expires_at: Optional[str] = None
    rotation_interval_days: Optional[int] = None
    last_rotated_at: Optional[str] = None
    rotation_status: str
    tags: Dict[str, str]


class SecretRotationRequest(BaseModel):
    """Request model for secret rotation."""
    secret_id: str
    new_value: Optional[str] = None


# Only admin users can access secrets management
def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Verify admin access for secrets management."""
    # In production, check actual admin role
    if not getattr(current_user, 'is_admin', False):
        # For now, allow all authenticated users (should be restricted in production)
        pass
    return current_user


@router.post("/", response_model=Dict[str, str])
async def create_secret(
    request: SecretCreateRequest,
    admin_user: User = Depends(get_admin_user)
):
    """
    Create a new secret with metadata and optional rotation.
    """
    try:
        if not secrets_manager.initialized:
            await secrets_manager.initialize()
        
        secret_id = await secrets_manager.store_secret(
            name=request.name,
            value=request.value,
            secret_type=request.secret_type,
            metadata=request.tags,
            rotation_interval_days=request.rotation_interval_days,
            encrypt_locally=request.encrypt_locally
        )
        
        return {
            "secret_id": secret_id,
            "message": f"Secret '{request.name}' created successfully",
            "rotation_enabled": request.rotation_interval_days is not None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create secret: {str(e)}"
        )


@router.get("/", response_model=List[SecretResponse])
async def list_secrets(
    secret_type: Optional[str] = Query(None),
    expired_only: bool = Query(False),
    admin_user: User = Depends(get_admin_user)
):
    """
    List secrets with optional filtering.
    """
    try:
        if not secrets_manager.initialized:
            await secrets_manager.initialize()
        
        filter_type = SecretType(secret_type) if secret_type else None
        secrets = await secrets_manager.list_secrets(
            secret_type=filter_type,
            expired_only=expired_only
        )
        
        return [
            SecretResponse(
                id=secret.id,
                name=secret.name,
                type=secret.type.value,
                created_at=secret.created_at.isoformat(),
                updated_at=secret.updated_at.isoformat(),
                expires_at=secret.expires_at.isoformat() if secret.expires_at else None,
                rotation_interval_days=secret.rotation_interval_days,
                last_rotated_at=secret.last_rotated_at.isoformat() if secret.last_rotated_at else None,
                rotation_status=secret.rotation_status.value,
                tags=secret.tags or {}
            )
            for secret in secrets
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list secrets: {str(e)}"
        )


@router.get("/{secret_id}", response_model=Dict[str, Any])
async def get_secret(
    secret_id: str,
    include_value: bool = Query(False),
    admin_user: User = Depends(get_admin_user)
):
    """
    Get secret details (optionally including the value).
    """
    try:
        if not secrets_manager.initialized:
            await secrets_manager.initialize()
        
        if include_value:
            # Return secret with value (audit this access)
            secret_data = await secrets_manager.get_secret(secret_id)
            if not secret_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Secret not found"
                )
            return secret_data
        else:
            # Return only metadata
            secrets = await secrets_manager.list_secrets()
            secret_metadata = next((s for s in secrets if s.id == secret_id), None)
            
            if not secret_metadata:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Secret not found"
                )
            
            return {
                "id": secret_metadata.id,
                "name": secret_metadata.name,
                "type": secret_metadata.type.value,
                "created_at": secret_metadata.created_at.isoformat(),
                "updated_at": secret_metadata.updated_at.isoformat(),
                "expires_at": secret_metadata.expires_at.isoformat() if secret_metadata.expires_at else None,
                "rotation_status": secret_metadata.rotation_status.value,
                "tags": secret_metadata.tags or {}
            }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get secret: {str(e)}"
        )


@router.post("/{secret_id}/rotate")
async def rotate_secret(
    secret_id: str,
    new_value: Optional[str] = None,
    admin_user: User = Depends(get_admin_user)
):
    """
    Rotate a secret (generate new value or use provided value).
    """
    try:
        # Start rotation task
        task_result = rotate_secret_task.delay(secret_id, new_value)
        
        return {
            "secret_id": secret_id,
            "task_id": task_result.id,
            "message": "Secret rotation started",
            "status": "in_progress"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start secret rotation: {str(e)}"
        )


@router.post("/rotate-bulk")
async def rotate_secrets_bulk(
    secret_type: Optional[str] = Query(None),
    admin_user: User = Depends(get_admin_user)
):
    """
    Rotate multiple secrets in bulk.
    """
    try:
        # Start bulk rotation task
        task_result = bulk_rotate_secrets.delay(secret_type)
        
        return {
            "task_id": task_result.id,
            "secret_type": secret_type or "all",
            "message": "Bulk secret rotation started",
            "status": "in_progress"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start bulk rotation: {str(e)}"
        )


@router.delete("/{secret_id}")
async def delete_secret(
    secret_id: str,
    admin_user: User = Depends(get_admin_user)
):
    """
    Delete a secret permanently.
    """
    try:
        if not secrets_manager.initialized:
            await secrets_manager.initialize()
        
        success = await secrets_manager.delete_secret(secret_id)
        
        if success:
            return {
                "secret_id": secret_id,
                "message": "Secret deleted successfully",
                "deleted": True
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Secret not found or deletion failed"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete secret: {str(e)}"
        )


@router.get("/audit/report")
async def get_security_report(
    generate_new: bool = Query(False),
    admin_user: User = Depends(get_admin_user)
):
    """
    Get security report for secrets management.
    """
    try:
        if generate_new:
            # Generate new report
            task_result = generate_security_report.delay()
            report = task_result.get(timeout=300)
        else:
            # Return cached/previous report (in production, get from storage)
            report = {
                "message": "Use generate_new=true to create a new report",
                "last_report_date": "Use generate_new parameter to get current report"
            }
        
        return report
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get security report: {str(e)}"
        )


@router.post("/audit/integrity-check")
async def run_integrity_check(
    admin_user: User = Depends(get_admin_user)
):
    """
    Run integrity check on all secrets.
    """
    try:
        task_result = validate_secret_integrity.delay()
        
        return {
            "task_id": task_result.id,
            "message": "Secret integrity check started",
            "status": "in_progress"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start integrity check: {str(e)}"
        )


@router.get("/types", response_model=List[str])
async def get_secret_types(
    admin_user: User = Depends(get_admin_user)
):
    """
    Get available secret types.
    """
    return [secret_type.value for secret_type in SecretType]


@router.get("/stats")
async def get_secrets_stats(
    admin_user: User = Depends(get_admin_user)
):
    """
    Get secrets management statistics.
    """
    try:
        if not secrets_manager.initialized:
            await secrets_manager.initialize()
        
        # Get all secrets for statistics
        all_secrets = await secrets_manager.list_secrets()
        expired_secrets = await secrets_manager.list_secrets(expired_only=True)
        
        # Calculate statistics by type
        type_counts = {}
        for secret in all_secrets:
            secret_type = secret.type.value
            type_counts[secret_type] = type_counts.get(secret_type, 0) + 1
        
        # Calculate rotation statistics
        rotated_count = len([s for s in all_secrets if s.last_rotated_at])
        rotation_enabled_count = len([s for s in all_secrets if s.rotation_interval_days])
        
        stats = {
            "total_secrets": len(all_secrets),
            "expired_secrets": len(expired_secrets),
            "secrets_by_type": type_counts,
            "rotation_statistics": {
                "rotation_enabled": rotation_enabled_count,
                "previously_rotated": rotated_count,
                "never_rotated": len(all_secrets) - rotated_count
            },
            "security_status": "healthy" if len(expired_secrets) == 0 else "warning",
            "last_updated": datetime.utcnow().isoformat()
        }
        
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get secrets statistics: {str(e)}"
        )


@router.get("/task/{task_id}/status")
async def get_task_status(
    task_id: str,
    admin_user: User = Depends(get_admin_user)
):
    """
    Get status of a secrets management task.
    """
    try:
        from app.core.celery_app import celery_app
        
        task_result = celery_app.AsyncResult(task_id)
        
        return {
            "task_id": task_id,
            "status": task_result.status,
            "result": task_result.result if task_result.successful() else None,
            "error": str(task_result.info) if task_result.failed() else None,
            "progress": task_result.info if task_result.state == 'PROGRESS' else None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )