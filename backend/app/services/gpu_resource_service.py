"""
Enhanced GPU Resource Management Service
Manages GPU allocation, monitoring, and optimization with database persistence
"""
import asyncio
import json
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager
import uuid

try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    # Note: Will log after logger is initialized

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from app.models.gpu_resources import (
    GPUDevice as GPUDeviceModel, 
    GPUTask as GPUTaskModel,
    GPUMetrics, 
    GPUResourcePool, 
    GPUAllocation
)
from app.db.session import get_db
from app.services.websocket_manager import ConnectionManager
from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)

# Log pynvml availability
if not NVIDIA_AVAILABLE:
    logger.warning("pynvml not available - GPU management will be limited")


class GPUStatus:
    """GPU status constants"""
    AVAILABLE = "available"
    BUSY = "busy"
    RESERVED = "reserved"
    ERROR = "error"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class TaskStatus:
    """Task status constants"""
    PENDING = "pending"
    ALLOCATED = "allocated"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority:
    """Task priority constants"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class GPUInfo:
    """Current GPU information"""
    device_id: int
    name: str
    memory_total: int
    memory_used: int
    memory_free: int
    utilization_gpu: float
    utilization_memory: float
    temperature: float
    power_draw: float
    processes: List[Dict[str, Any]]
    status: str


class EnhancedGPUResourceService:
    """
    Advanced GPU resource management with database persistence
    """
    
    def __init__(self, websocket_manager: Optional[ConnectionManager] = None):
        self.websocket_manager = websocket_manager
        self.nvidia_available = NVIDIA_AVAILABLE
        self.monitoring_active = False
        self.task_scheduler_active = False
        self.devices_cache = {}
        
        # Configuration
        self.monitoring_interval = 30  # seconds
        self.health_check_interval = 300  # seconds
        self.metrics_retention_days = 30
        
    async def initialize(self, db: AsyncSession):
        """Initialize GPU resource service"""
        try:
            if not self.nvidia_available:
                logger.info("GPU management disabled - NVIDIA drivers not available")
                return
                
            # Initialize NVIDIA Management Library
            pynvml.nvmlInit()
            
            # Discover and register devices
            await self.discover_and_register_devices(db)
            
            # Start monitoring
            asyncio.create_task(self.monitor_devices())
            asyncio.create_task(self.health_check_loop())
            asyncio.create_task(self.task_scheduler_loop())
            asyncio.create_task(self.cleanup_expired_allocations())
            
            logger.info("Enhanced GPU Resource Service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU service: {e}")
            self.nvidia_available = False
            
    async def discover_and_register_devices(self, db: AsyncSession):
        """Discover and register GPU devices"""
        if not self.nvidia_available:
            return
            
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get device information
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get additional device info
                try:
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_capability = f"{major}.{minor}"
                except:
                    compute_capability = "unknown"
                    
                try:
                    driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                except:
                    driver_version = "unknown"
                    
                try:
                    cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                    cuda_version = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
                except:
                    cuda_version = "unknown"
                
                # Check if device already exists
                existing = await db.execute(
                    select(GPUDeviceModel).where(GPUDeviceModel.device_id == i)
                )
                device = existing.scalar_one_or_none()
                
                if device:
                    # Update existing device
                    device.name = name
                    device.memory_total = memory_info.total
                    device.compute_capability = compute_capability
                    device.driver_version = driver_version
                    device.cuda_version = cuda_version
                    device.last_seen_at = datetime.utcnow()
                    device.is_enabled = True
                else:
                    # Create new device
                    device = GPUDeviceModel(
                        device_id=i,
                        name=name,
                        memory_total=memory_info.total,
                        compute_capability=compute_capability,
                        driver_version=driver_version,
                        cuda_version=cuda_version,
                        status=GPUStatus.AVAILABLE
                    )
                    db.add(device)
                    
                await db.commit()
                logger.info(f"Registered GPU {i}: {name} ({memory_info.total / 1024**3:.1f}GB)")
                
        except Exception as e:
            logger.error(f"Failed to discover devices: {e}")
            
    async def get_gpu_info(self, device_id: int) -> Optional[GPUInfo]:
        """Get current GPU information"""
        if not self.nvidia_available:
            return None
            
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            
            # Get basic info
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Get temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = 0.0
                
            # Get power draw
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            except:
                power = 0.0
                
            # Get running processes
            processes = []
            try:
                process_info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for proc in process_info:
                    try:
                        # Get process name if available
                        process_name = psutil.Process(proc.pid).name()
                    except:
                        process_name = "unknown"
                        
                    processes.append({
                        "pid": proc.pid,
                        "name": process_name,
                        "memory_used": proc.usedGpuMemory
                    })
            except:
                pass
                
            # Determine status
            if utilization.gpu < 10:
                status = GPUStatus.AVAILABLE
            elif utilization.gpu < 80:
                status = GPUStatus.BUSY
            else:
                status = GPUStatus.BUSY
                
            return GPUInfo(
                device_id=device_id,
                name=name,
                memory_total=memory_info.total,
                memory_used=memory_info.used,
                memory_free=memory_info.free,
                utilization_gpu=utilization.gpu,
                utilization_memory=(memory_info.used / memory_info.total) * 100,
                temperature=temperature,
                power_draw=power,
                processes=processes,
                status=status
            )
            
        except Exception as e:
            logger.error(f"Failed to get GPU {device_id} info: {e}")
            return None
            
    async def allocate_gpu(
        self,
        db: AsyncSession,
        task_id: str,
        task_type: str,
        user_id: str,
        memory_required: int,
        priority: int = TaskPriority.NORMAL,
        estimated_duration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """
        Allocate GPU for a task
        
        Returns:
            Device ID if allocated, None if no device available
        """
        try:
            # Create task record
            task = GPUTaskModel(
                task_id=task_id,
                task_type=task_type,
                user_id=user_id,
                memory_required=memory_required,
                priority=priority,
                estimated_duration=estimated_duration,
                metadata=metadata or {},
                status=TaskStatus.PENDING
            )
            
            db.add(task)
            await db.commit()
            await db.refresh(task)
            
            # Find suitable device
            device_id = await self._find_suitable_device(db, memory_required, priority)
            
            if device_id is not None:
                # Allocate device
                await self._allocate_device_to_task(db, device_id, task)
                
                # Create allocation record
                allocation = GPUAllocation(
                    device_id=device_id,
                    allocated_to=task_id,
                    allocation_type="task",
                    memory_allocated=memory_required,
                    expires_at=datetime.utcnow() + timedelta(hours=1),  # Default 1 hour
                    purpose=f"{task_type} task"
                )
                
                db.add(allocation)
                await db.commit()
                
                # Notify via WebSocket
                if self.websocket_manager:
                    await self.websocket_manager.send_to_user(user_id, {
                        "type": "gpu_allocated",
                        "data": {
                            "task_id": task_id,
                            "device_id": device_id,
                            "memory_allocated": memory_required
                        }
                    })
                    
                logger.info(f"Allocated GPU {device_id} to task {task_id}")
                return device_id
            else:
                # No device available - queue task
                logger.warning(f"No GPU available for task {task_id} - queued")
                return None
                
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to allocate GPU for task {task_id}: {e}")
            return None
            
    async def release_gpu(
        self,
        db: AsyncSession,
        task_id: str,
        success: bool = True,
        result_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        """Release GPU allocation for completed task"""
        try:
            # Get task
            result = await db.execute(
                select(GPUTaskModel).where(GPUTaskModel.task_id == task_id)
            )
            task = result.scalar_one_or_none()
            
            if not task:
                logger.warning(f"Task {task_id} not found for GPU release")
                return
                
            # Update task completion
            task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.success = success
            task.result_data = result_data
            task.error_message = error_message
            
            if task.started_at:
                task.actual_duration = int(
                    (task.completed_at - task.started_at).total_seconds()
                )
                
            # Release device
            if task.device_id is not None:
                await self._release_device(db, task.device_id, task_id)
                
                # Update device statistics
                device_result = await db.execute(
                    select(GPUDeviceModel).where(GPUDeviceModel.device_id == task.device_id)
                )
                device = device_result.scalar_one_or_none()
                
                if device:
                    if success:
                        device.total_tasks_completed += 1
                    else:
                        device.total_tasks_failed += 1
                        
            await db.commit()
            
            logger.info(f"Released GPU for task {task_id} (success: {success})")
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to release GPU for task {task_id}: {e}")
            
    async def get_device_status(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """Get status of all GPU devices"""
        try:
            result = await db.execute(
                select(GPUDeviceModel).where(GPUDeviceModel.is_enabled == True)
                .order_by(GPUDeviceModel.device_id)
            )
            devices = result.scalars().all()
            
            device_statuses = []
            
            for device in devices:
                # Get current GPU info if available
                current_info = await self.get_gpu_info(device.device_id)
                
                status = {
                    "device_id": device.device_id,
                    "name": device.name,
                    "status": device.status,
                    "memory_total": device.memory_total,
                    "memory_used": current_info.memory_used if current_info else device.memory_used,
                    "memory_free": current_info.memory_free if current_info else device.memory_free,
                    "utilization_gpu": current_info.utilization_gpu if current_info else device.utilization_gpu,
                    "temperature": current_info.temperature if current_info else device.temperature,
                    "power_draw": current_info.power_draw if current_info else device.power_draw,
                    "reserved_by": device.reserved_by,
                    "reserved_until": device.reserved_until.isoformat() if device.reserved_until else None,
                    "health_status": device.health_status,
                    "total_tasks_completed": device.total_tasks_completed,
                    "total_tasks_failed": device.total_tasks_failed,
                    "processes": current_info.processes if current_info else []
                }
                
                device_statuses.append(status)
                
            return device_statuses
            
        except Exception as e:
            logger.error(f"Failed to get device status: {e}")
            return []
            
    async def get_task_queue(self, db: AsyncSession, limit: int = 50) -> List[Dict[str, Any]]:
        """Get current GPU task queue"""
        try:
            result = await db.execute(
                select(GPUTaskModel).where(
                    GPUTaskModel.status.in_([TaskStatus.PENDING, TaskStatus.ALLOCATED, TaskStatus.RUNNING])
                ).order_by(desc(GPUTaskModel.priority), GPUTaskModel.created_at)
                .limit(limit)
            )
            tasks = result.scalars().all()
            
            task_queue = []
            for task in tasks:
                task_info = {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "user_id": task.user_id,
                    "priority": task.priority,
                    "status": task.status,
                    "device_id": task.device_id,
                    "memory_required": task.memory_required,
                    "estimated_duration": task.estimated_duration,
                    "created_at": task.created_at.isoformat(),
                    "allocated_at": task.allocated_at.isoformat() if task.allocated_at else None,
                    "started_at": task.started_at.isoformat() if task.started_at else None
                }
                task_queue.append(task_info)
                
            return task_queue
            
        except Exception as e:
            logger.error(f"Failed to get task queue: {e}")
            return []
            
    async def get_gpu_metrics(
        self,
        db: AsyncSession,
        device_id: Optional[int] = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get GPU performance metrics"""
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            query = select(GPUMetrics).where(GPUMetrics.timestamp >= since)
            
            if device_id is not None:
                query = query.where(GPUMetrics.device_id == device_id)
                
            query = query.order_by(GPUMetrics.timestamp)
            
            result = await db.execute(query)
            metrics = result.scalars().all()
            
            metrics_data = []
            for metric in metrics:
                metrics_data.append({
                    "device_id": metric.device_id,
                    "timestamp": metric.timestamp.isoformat(),
                    "gpu_utilization": metric.gpu_utilization,
                    "memory_utilization": metric.memory_utilization,
                    "temperature": metric.temperature,
                    "power_draw": metric.power_draw,
                    "process_count": metric.process_count,
                    "efficiency_score": metric.efficiency_score
                })
                
            return metrics_data
            
        except Exception as e:
            logger.error(f"Failed to get GPU metrics: {e}")
            return []
            
    # Background monitoring tasks
    async def monitor_devices(self):
        """Background task to monitor GPU devices"""
        if not self.nvidia_available:
            return
            
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                async with get_db() as db:
                    # Get registered devices
                    result = await db.execute(
                        select(GPUDeviceModel).where(GPUDeviceModel.is_enabled == True)
                    )
                    devices = result.scalars().all()
                    
                    for device in devices:
                        # Get current GPU info
                        current_info = await self.get_gpu_info(device.device_id)
                        
                        if current_info:
                            # Update device status
                            device.memory_used = current_info.memory_used
                            device.memory_free = current_info.memory_free
                            device.utilization_gpu = current_info.utilization_gpu
                            device.utilization_memory = current_info.utilization_memory
                            device.temperature = current_info.temperature
                            device.power_draw = current_info.power_draw
                            device.status = current_info.status
                            device.last_seen_at = datetime.utcnow()
                            
                            # Store metrics
                            metrics = GPUMetrics(
                                device_id=device.device_id,
                                timestamp=datetime.utcnow(),
                                gpu_utilization=current_info.utilization_gpu,
                                memory_utilization=current_info.utilization_memory,
                                memory_used=current_info.memory_used,
                                memory_free=current_info.memory_free,
                                temperature=current_info.temperature,
                                power_draw=current_info.power_draw,
                                process_count=len(current_info.processes),
                                efficiency_score=self._calculate_efficiency_score(current_info)
                            )
                            
                            db.add(metrics)
                            
                    await db.commit()
                    
            except Exception as e:
                logger.error(f"Error in device monitoring: {e}")
                
            await asyncio.sleep(self.monitoring_interval)
            
    async def health_check_loop(self):
        """Background health checking"""
        while self.monitoring_active:
            try:
                async with get_db() as db:
                    await self._perform_health_checks(db)
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                
            await asyncio.sleep(self.health_check_interval)
            
    async def task_scheduler_loop(self):
        """Background task scheduler"""
        self.task_scheduler_active = True
        
        while self.task_scheduler_active:
            try:
                async with get_db() as db:
                    await self._schedule_pending_tasks(db)
            except Exception as e:
                logger.error(f"Error in task scheduler: {e}")
                
            await asyncio.sleep(30)  # Check every 30 seconds
            
    async def cleanup_expired_allocations(self):
        """Clean up expired allocations"""
        while self.monitoring_active:
            try:
                async with get_db() as db:
                    # Find expired allocations
                    now = datetime.utcnow()
                    
                    result = await db.execute(
                        select(GPUAllocation).where(
                            and_(
                                GPUAllocation.status == "active",
                                GPUAllocation.expires_at < now
                            )
                        )
                    )
                    expired_allocations = result.scalars().all()
                    
                    for allocation in expired_allocations:
                        allocation.status = "expired"
                        allocation.released_at = now
                        
                        # Release device
                        await self._release_device(db, allocation.device_id, allocation.allocated_to)
                        
                    await db.commit()
                    
                    if expired_allocations:
                        logger.info(f"Cleaned up {len(expired_allocations)} expired GPU allocations")
                        
            except Exception as e:
                logger.error(f"Error cleaning up allocations: {e}")
                
            await asyncio.sleep(300)  # Check every 5 minutes
            
    # Private helper methods
    async def _find_suitable_device(
        self,
        db: AsyncSession,
        memory_required: int,
        priority: int
    ) -> Optional[int]:
        """Find suitable GPU device for allocation"""
        result = await db.execute(
            select(GPUDeviceModel).where(
                and_(
                    GPUDeviceModel.is_enabled == True,
                    GPUDeviceModel.status.in_([GPUStatus.AVAILABLE, GPUStatus.BUSY]),
                    GPUDeviceModel.memory_free >= memory_required + (512 * 1024 * 1024)  # 512MB buffer
                )
            ).order_by(
                GPUDeviceModel.utilization_gpu,  # Prefer less utilized devices
                GPUDeviceModel.device_id
            )
        )
        
        suitable_devices = result.scalars().all()
        
        if suitable_devices:
            return suitable_devices[0].device_id
            
        return None
        
    async def _allocate_device_to_task(self, db: AsyncSession, device_id: int, task: GPUTaskModel):
        """Allocate device to task"""
        task.device_id = device_id
        task.status = TaskStatus.ALLOCATED
        task.allocated_at = datetime.utcnow()
        
        # Update device status
        await db.execute(
            update(GPUDeviceModel)
            .where(GPUDeviceModel.device_id == device_id)
            .values(
                status=GPUStatus.BUSY,
                reserved_by=task.task_id,
                reserved_until=datetime.utcnow() + timedelta(hours=1)
            )
        )
        
    async def _release_device(self, db: AsyncSession, device_id: int, task_id: str):
        """Release device allocation"""
        await db.execute(
            update(GPUDeviceModel)
            .where(GPUDeviceModel.device_id == device_id)
            .values(
                status=GPUStatus.AVAILABLE,
                reserved_by=None,
                reserved_until=None
            )
        )
        
        # Mark allocations as released
        await db.execute(
            update(GPUAllocation)
            .where(
                and_(
                    GPUAllocation.device_id == device_id,
                    GPUAllocation.allocated_to == task_id,
                    GPUAllocation.status == "active"
                )
            )
            .values(
                status="released",
                released_at=datetime.utcnow()
            )
        )
        
    async def _schedule_pending_tasks(self, db: AsyncSession):
        """Schedule pending tasks to available devices"""
        # Get pending tasks ordered by priority
        result = await db.execute(
            select(GPUTaskModel).where(GPUTaskModel.status == TaskStatus.PENDING)
            .order_by(desc(GPUTaskModel.priority), GPUTaskModel.created_at)
            .limit(20)  # Process up to 20 tasks at once
        )
        pending_tasks = result.scalars().all()
        
        for task in pending_tasks:
            device_id = await self._find_suitable_device(
                db, task.memory_required, task.priority
            )
            
            if device_id is not None:
                await self._allocate_device_to_task(db, device_id, task)
                
                # Create allocation record
                allocation = GPUAllocation(
                    device_id=device_id,
                    allocated_to=task.task_id,
                    allocation_type="task",
                    memory_allocated=task.memory_required,
                    expires_at=datetime.utcnow() + timedelta(hours=2),
                    purpose=f"{task.task_type} task"
                )
                
                db.add(allocation)
                
                logger.info(f"Scheduled task {task.task_id} to GPU {device_id}")
                
        await db.commit()
        
    async def _perform_health_checks(self, db: AsyncSession):
        """Perform health checks on devices"""
        result = await db.execute(
            select(GPUDeviceModel).where(GPUDeviceModel.is_enabled == True)
        )
        devices = result.scalars().all()
        
        for device in devices:
            health_status = "healthy"
            health_message = None
            
            # Check temperature
            if device.temperature > 85:
                health_status = "critical"
                health_message = f"High temperature: {device.temperature}°C"
            elif device.temperature > 75:
                health_status = "warning"
                health_message = f"Elevated temperature: {device.temperature}°C"
                
            # Check if device is responding
            current_info = await self.get_gpu_info(device.device_id)
            if current_info is None:
                health_status = "critical"
                health_message = "Device not responding"
                device.status = GPUStatus.ERROR
                device.error_count += 1
            else:
                device.error_count = 0
                
            device.health_status = health_status
            device.health_message = health_message
            device.last_health_check = datetime.utcnow()
            
        await db.commit()
        
    def _calculate_efficiency_score(self, gpu_info: GPUInfo) -> float:
        """Calculate GPU efficiency score (0-100)"""
        # Simple efficiency calculation
        # High utilization with reasonable temperature = good efficiency
        utilization_score = min(gpu_info.utilization_gpu, 90) / 90 * 70  # Max 70 points
        
        # Temperature penalty
        temp_penalty = 0
        if gpu_info.temperature > 80:
            temp_penalty = (gpu_info.temperature - 80) * 2
            
        # Memory utilization bonus
        memory_bonus = min(gpu_info.utilization_memory, 80) / 80 * 30  # Max 30 points
        
        efficiency = max(0, utilization_score + memory_bonus - temp_penalty)
        
        return min(100, efficiency)


# Global instance
gpu_service = EnhancedGPUResourceService()