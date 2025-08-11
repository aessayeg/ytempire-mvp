"""
GPU Resource Management API Endpoints
Provides GPU allocation, monitoring, and management capabilities
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from app.db.session import get_db
from app.api.v1.endpoints.auth import get_current_verified_user
from app.models.user import User
from app.services.gpu_resource_service import gpu_service, TaskPriority

logger = logging.getLogger(__name__)

router = APIRouter()


class GPUAllocationRequest(BaseModel):
    """Request to allocate GPU resources"""
    task_id: str
    task_type: str = Field(..., description="Type of task: video_generation, model_training, inference")
    memory_required: int = Field(..., ge=1, description="Required GPU memory in bytes")
    priority: int = Field(default=1, ge=0, le=3, description="Task priority (0=low, 3=critical)")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional task metadata")


class GPUAllocationResponse(BaseModel):
    """GPU allocation response"""
    success: bool
    device_id: Optional[int]
    message: str
    allocation_id: Optional[str]
    estimated_wait_time: Optional[int]  # seconds


class GPUReleaseRequest(BaseModel):
    """Request to release GPU resources"""
    task_id: str
    success: bool = True
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class GPUDeviceStatus(BaseModel):
    """GPU device status response"""
    device_id: int
    name: str
    status: str
    memory_total: int
    memory_used: int
    memory_free: int
    memory_utilization: float
    gpu_utilization: float
    temperature: float
    power_draw: float
    reserved_by: Optional[str]
    reserved_until: Optional[datetime]
    health_status: str
    total_tasks_completed: int
    total_tasks_failed: int
    processes: List[Dict[str, Any]]


class GPUTaskInfo(BaseModel):
    """GPU task information"""
    task_id: str
    task_type: str
    user_id: str
    priority: int
    status: str
    device_id: Optional[int]
    memory_required: int
    estimated_duration: Optional[int]
    created_at: datetime
    allocated_at: Optional[datetime]
    started_at: Optional[datetime]
    queue_position: Optional[int]


class GPUMetricsResponse(BaseModel):
    """GPU metrics response"""
    device_id: int
    timestamp: datetime
    gpu_utilization: float
    memory_utilization: float
    temperature: float
    power_draw: float
    process_count: int
    efficiency_score: float


class GPUStatsResponse(BaseModel):
    """GPU statistics response"""
    total_devices: int
    available_devices: int
    busy_devices: int
    error_devices: int
    total_memory_gb: float
    available_memory_gb: float
    avg_utilization: float
    avg_temperature: float
    total_tasks_today: int
    successful_tasks_today: int
    failed_tasks_today: int
    avg_task_duration_minutes: float
    current_queue_length: int


@router.post("/allocate", response_model=GPUAllocationResponse)
async def allocate_gpu(
    request: GPUAllocationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> GPUAllocationResponse:
    """
    Allocate GPU resources for a task
    """
    try:
        device_id = await gpu_service.allocate_gpu(
            db=db,
            task_id=request.task_id,
            task_type=request.task_type,
            user_id=str(current_user.id),
            memory_required=request.memory_required,
            priority=request.priority,
            estimated_duration=request.estimated_duration,
            metadata=request.metadata
        )
        
        if device_id is not None:
            return GPUAllocationResponse(
                success=True,
                device_id=device_id,
                message=f"GPU {device_id} allocated successfully",
                allocation_id=request.task_id
            )
        else:
            # Task queued - estimate wait time
            queue = await gpu_service.get_task_queue(db, limit=100)
            queue_position = len([t for t in queue if t["priority"] >= request.priority]) + 1
            estimated_wait = queue_position * 300  # Rough estimate: 5 minutes per task
            
            return GPUAllocationResponse(
                success=False,
                device_id=None,
                message="No GPU available - task queued",
                estimated_wait_time=estimated_wait
            )
            
    except Exception as e:
        logger.error(f"Failed to allocate GPU for task {request.task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to allocate GPU: {str(e)}"
        )


@router.post("/release")
async def release_gpu(
    request: GPUReleaseRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, str]:
    """
    Release GPU resources for a completed task
    """
    try:
        await gpu_service.release_gpu(
            db=db,
            task_id=request.task_id,
            success=request.success,
            result_data=request.result_data,
            error_message=request.error_message
        )
        
        return {"status": "released", "task_id": request.task_id}
        
    except Exception as e:
        logger.error(f"Failed to release GPU for task {request.task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to release GPU: {str(e)}"
        )


@router.get("/devices", response_model=List[GPUDeviceStatus])
async def get_gpu_devices(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> List[GPUDeviceStatus]:
    """
    Get status of all GPU devices
    """
    try:
        devices = await gpu_service.get_device_status(db)
        
        return [
            GPUDeviceStatus(
                device_id=device["device_id"],
                name=device["name"],
                status=device["status"],
                memory_total=device["memory_total"],
                memory_used=device["memory_used"],
                memory_free=device["memory_free"],
                memory_utilization=(device["memory_used"] / device["memory_total"]) * 100,
                gpu_utilization=device["utilization_gpu"],
                temperature=device["temperature"],
                power_draw=device["power_draw"],
                reserved_by=device["reserved_by"],
                reserved_until=datetime.fromisoformat(device["reserved_until"]) if device["reserved_until"] else None,
                health_status=device["health_status"],
                total_tasks_completed=device["total_tasks_completed"],
                total_tasks_failed=device["total_tasks_failed"],
                processes=device["processes"]
            )
            for device in devices
        ]
        
    except Exception as e:
        logger.error(f"Failed to get GPU devices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch GPU device status"
        )


@router.get("/queue", response_model=List[GPUTaskInfo])
async def get_task_queue(
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> List[GPUTaskInfo]:
    """
    Get current GPU task queue
    """
    try:
        queue = await gpu_service.get_task_queue(db, limit)
        
        task_list = []
        for i, task in enumerate(queue):
            task_info = GPUTaskInfo(
                task_id=task["task_id"],
                task_type=task["task_type"],
                user_id=task["user_id"],
                priority=task["priority"],
                status=task["status"],
                device_id=task["device_id"],
                memory_required=task["memory_required"],
                estimated_duration=task["estimated_duration"],
                created_at=datetime.fromisoformat(task["created_at"]),
                allocated_at=datetime.fromisoformat(task["allocated_at"]) if task["allocated_at"] else None,
                started_at=datetime.fromisoformat(task["started_at"]) if task["started_at"] else None,
                queue_position=i + 1 if task["status"] == "pending" else None
            )
            task_list.append(task_info)
            
        return task_list
        
    except Exception as e:
        logger.error(f"Failed to get task queue: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch task queue"
        )


@router.get("/metrics", response_model=List[GPUMetricsResponse])
async def get_gpu_metrics(
    device_id: Optional[int] = Query(None, description="Specific device ID"),
    hours: int = Query(24, ge=1, le=168, description="Hours of metrics to retrieve"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> List[GPUMetricsResponse]:
    """
    Get GPU performance metrics
    """
    try:
        metrics_data = await gpu_service.get_gpu_metrics(db, device_id, hours)
        
        return [
            GPUMetricsResponse(
                device_id=metric["device_id"],
                timestamp=datetime.fromisoformat(metric["timestamp"]),
                gpu_utilization=metric["gpu_utilization"],
                memory_utilization=metric["memory_utilization"],
                temperature=metric["temperature"],
                power_draw=metric["power_draw"],
                process_count=metric["process_count"],
                efficiency_score=metric["efficiency_score"]
            )
            for metric in metrics_data
        ]
        
    except Exception as e:
        logger.error(f"Failed to get GPU metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch GPU metrics"
        )


@router.get("/stats", response_model=GPUStatsResponse)
async def get_gpu_statistics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> GPUStatsResponse:
    """
    Get comprehensive GPU statistics
    """
    try:
        # Get device statuses
        devices = await gpu_service.get_device_status(db)
        
        # Calculate statistics
        total_devices = len(devices)
        available_devices = len([d for d in devices if d["status"] == "available"])
        busy_devices = len([d for d in devices if d["status"] == "busy"])
        error_devices = len([d for d in devices if d["status"] == "error"])
        
        total_memory = sum(d["memory_total"] for d in devices) / (1024**3)  # GB
        available_memory = sum(d["memory_free"] for d in devices) / (1024**3)  # GB
        
        avg_utilization = sum(d["utilization_gpu"] for d in devices) / max(total_devices, 1)
        avg_temperature = sum(d["temperature"] for d in devices) / max(total_devices, 1)
        
        # Get task statistics (simplified)
        queue = await gpu_service.get_task_queue(db, limit=1000)
        current_queue_length = len([t for t in queue if t["status"] == "pending"])
        
        return GPUStatsResponse(
            total_devices=total_devices,
            available_devices=available_devices,
            busy_devices=busy_devices,
            error_devices=error_devices,
            total_memory_gb=round(total_memory, 1),
            available_memory_gb=round(available_memory, 1),
            avg_utilization=round(avg_utilization, 1),
            avg_temperature=round(avg_temperature, 1),
            total_tasks_today=0,  # Would query from database
            successful_tasks_today=0,  # Would query from database
            failed_tasks_today=0,  # Would query from database
            avg_task_duration_minutes=0,  # Would calculate from completed tasks
            current_queue_length=current_queue_length
        )
        
    except Exception as e:
        logger.error(f"Failed to get GPU statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch GPU statistics"
        )


@router.post("/reserve/{device_id}")
async def reserve_gpu(
    device_id: int,
    duration_minutes: int = Query(60, ge=1, le=480, description="Reservation duration in minutes"),
    purpose: str = Query(..., description="Purpose of reservation"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Reserve GPU device for exclusive use
    """
    try:
        # This would implement device reservation logic
        # For now, return a mock response
        
        reservation_id = f"res_{current_user.id}_{device_id}_{int(datetime.utcnow().timestamp())}"
        
        return {
            "reservation_id": reservation_id,
            "device_id": device_id,
            "reserved_until": (datetime.utcnow() + timedelta(minutes=duration_minutes)).isoformat(),
            "purpose": purpose,
            "message": f"GPU {device_id} reserved for {duration_minutes} minutes"
        }
        
    except Exception as e:
        logger.error(f"Failed to reserve GPU {device_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reserve GPU {device_id}"
        )


@router.delete("/reserve/{reservation_id}")
async def cancel_reservation(
    reservation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, str]:
    """
    Cancel GPU reservation
    """
    try:
        # This would implement reservation cancellation logic
        return {
            "status": "cancelled",
            "reservation_id": reservation_id
        }
        
    except Exception as e:
        logger.error(f"Failed to cancel reservation {reservation_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel reservation"
        )


@router.get("/health")
async def gpu_health_check(
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    GPU service health check
    """
    try:
        devices = await gpu_service.get_device_status(db)
        
        return {
            "status": "healthy",
            "gpu_service_available": gpu_service.nvidia_available,
            "devices_count": len(devices),
            "monitoring_active": gpu_service.monitoring_active,
            "scheduler_active": gpu_service.task_scheduler_active,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"GPU health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }