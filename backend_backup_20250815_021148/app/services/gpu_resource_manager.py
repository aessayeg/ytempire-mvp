"""
GPU Resource Management Service
Handles GPU allocation, monitoring, and optimization for ML workloads
"""
import asyncio
import psutil
import pynvml
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import redis.asyncio as redis
from contextlib import asynccontextmanager
import numpy as np

logger = logging.getLogger(__name__)


class GPUStatus(str, Enum):
    """GPU status states"""
    AVAILABLE = "available"
    BUSY = "busy"
    RESERVED = "reserved"
    ERROR = "error"
    OFFLINE = "offline"


class TaskPriority(int, Enum):
    """Task priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class GPUDevice:
    """GPU device information"""
    device_id: int
    name: str
    memory_total: int  # bytes
    memory_used: int
    memory_free: int
    utilization: float  # percentage
    temperature: float  # celsius
    power_draw: float  # watts
    processes: List[Dict[str, Any]]
    status: GPUStatus
    reserved_by: Optional[str] = None
    reserved_until: Optional[datetime] = None
    

@dataclass
class GPUTask:
    """GPU task for execution"""
    task_id: str
    task_type: str  # video_generation, model_training, inference
    priority: TaskPriority
    memory_required: int  # bytes
    estimated_duration: int  # seconds
    user_id: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    device_id: Optional[int] = None
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)


class GPUResourceManager:
    """
    Manages GPU resources for optimal utilization
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.devices: Dict[int, GPUDevice] = {}
        self.task_queue: List[GPUTask] = []
        self.active_tasks: Dict[str, GPUTask] = {}
        self.monitoring_interval = 5  # seconds
        self.initialized = False
        
        # Memory thresholds
        self.memory_reserve_mb = 512  # Reserve 512MB for system
        self.max_utilization = 90  # Max 90% utilization
        
    async def initialize(self):
        """Initialize GPU resource manager"""
        try:
            # Initialize NVIDIA Management Library
            pynvml.nvmlInit()
            
            # Connect to Redis
            self.redis_client = await redis.from_url(self.redis_url)
            
            # Discover GPU devices
            await self.discover_devices()
            
            # Start monitoring task
            asyncio.create_task(self.monitor_devices())
            
            # Start task scheduler
            asyncio.create_task(self.task_scheduler())
            
            self.initialized = True
            logger.info(f"GPU Resource Manager initialized with {len(self.devices)} devices")
            
        except pynvml.NVMLError as e:
            logger.error(f"NVIDIA GPU not available: {e}")
            # Continue without GPU support
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize GPU manager: {e}")
            raise
            
    async def discover_devices(self):
        """Discover available GPU devices"""
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get device info
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = 0
                    
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                except:
                    power = 0
                
                # Get running processes
                processes = []
                try:
                    process_info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for proc in process_info:
                        processes.append({
                            "pid": proc.pid,
                            "memory_used": proc.usedGpuMemory
                        })
                except:
                    pass
                
                device = GPUDevice(
                    device_id=i,
                    name=name,
                    memory_total=memory_info.total,
                    memory_used=memory_info.used,
                    memory_free=memory_info.free,
                    utilization=utilization.gpu,
                    temperature=temperature,
                    power_draw=power,
                    processes=processes,
                    status=GPUStatus.AVAILABLE if utilization.gpu < 50 else GPUStatus.BUSY
                )
                
                self.devices[i] = device
                logger.info(f"Discovered GPU {i}: {name} ({memory_info.free / 1024**3:.1f}GB free)")
                
        except pynvml.NVMLError as e:
            logger.warning(f"No NVIDIA GPUs found: {e}")
            
    async def monitor_devices(self):
        """Continuously monitor GPU devices"""
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                for device_id in self.devices:
                    await self.update_device_status(device_id)
                    
                # Check for stuck tasks
                await self.check_stuck_tasks()
                
                # Store metrics in Redis
                await self.store_metrics()
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                
    async def update_device_status(self, device_id: int):
        """Update status of a specific GPU device"""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            device = self.devices[device_id]
            device.memory_used = memory_info.used
            device.memory_free = memory_info.free
            device.utilization = utilization.gpu
            
            try:
                device.temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                pass
                
            # Update status based on utilization
            if device.reserved_by:
                device.status = GPUStatus.RESERVED
            elif utilization.gpu < 10:
                device.status = GPUStatus.AVAILABLE
            elif utilization.gpu < 80:
                device.status = GPUStatus.BUSY
            else:
                device.status = GPUStatus.BUSY
                
        except Exception as e:
            logger.error(f"Failed to update device {device_id}: {e}")
            self.devices[device_id].status = GPUStatus.ERROR
            
    async def allocate_gpu(
        self,
        task: GPUTask,
        prefer_device: Optional[int] = None
    ) -> Optional[int]:
        """
        Allocate GPU for a task
        
        Returns:
            Device ID if allocated, None if no device available
        """
        # Find suitable device
        suitable_devices = []
        
        for device_id, device in self.devices.items():
            if device.status in [GPUStatus.AVAILABLE, GPUStatus.BUSY]:
                # Check memory requirements
                available_memory = device.memory_free - (self.memory_reserve_mb * 1024 * 1024)
                
                if available_memory >= task.memory_required:
                    # Check utilization
                    if device.utilization < self.max_utilization:
                        suitable_devices.append((device_id, device.utilization))
        
        if not suitable_devices:
            logger.warning(f"No suitable GPU for task {task.task_id}")
            return None
            
        # Sort by utilization (prefer less busy devices)
        suitable_devices.sort(key=lambda x: x[1])
        
        # Prefer specified device if available
        if prefer_device is not None and prefer_device in [d[0] for d in suitable_devices]:
            selected_device = prefer_device
        else:
            selected_device = suitable_devices[0][0]
            
        # Reserve device
        device = self.devices[selected_device]
        device.reserved_by = task.task_id
        device.reserved_until = datetime.utcnow() + timedelta(seconds=task.estimated_duration)
        
        # Update task
        task.device_id = selected_device
        task.status = "allocated"
        task.started_at = datetime.utcnow()
        
        # Store allocation in Redis
        await self.store_allocation(task)
        
        logger.info(f"Allocated GPU {selected_device} to task {task.task_id}")
        return selected_device
        
    async def release_gpu(self, task_id: str):
        """Release GPU allocation"""
        # Find device with this task
        for device_id, device in self.devices.items():
            if device.reserved_by == task_id:
                device.reserved_by = None
                device.reserved_until = None
                
                # Update task status
                if task_id in self.active_tasks:
                    task = self.active_tasks[task_id]
                    task.completed_at = datetime.utcnow()
                    task.status = "completed"
                    del self.active_tasks[task_id]
                    
                logger.info(f"Released GPU {device_id} from task {task_id}")
                break
                
    async def queue_task(self, task: GPUTask) -> str:
        """Add task to GPU queue"""
        # Add to queue
        self.task_queue.append(task)
        
        # Sort by priority
        self.task_queue.sort(key=lambda x: (-x.priority.value, x.created_at))
        
        # Store in Redis
        await self.store_task_in_redis(task)
        
        logger.info(f"Queued GPU task {task.task_id} with priority {task.priority.name}")
        return task.task_id
        
    async def task_scheduler(self):
        """Schedule queued tasks to available GPUs"""
        while True:
            try:
                await asyncio.sleep(2)  # Check every 2 seconds
                
                if not self.task_queue:
                    continue
                    
                # Process queue
                tasks_to_remove = []
                
                for task in self.task_queue:
                    # Try to allocate GPU
                    device_id = await self.allocate_gpu(task)
                    
                    if device_id is not None:
                        # Move to active tasks
                        self.active_tasks[task.task_id] = task
                        tasks_to_remove.append(task)
                        
                        # Trigger task execution
                        asyncio.create_task(self.execute_task(task, device_id))
                        
                # Remove allocated tasks from queue
                for task in tasks_to_remove:
                    self.task_queue.remove(task)
                    
            except Exception as e:
                logger.error(f"Task scheduler error: {e}")
                
    async def execute_task(self, task: GPUTask, device_id: int):
        """Execute task on allocated GPU"""
        try:
            logger.info(f"Executing task {task.task_id} on GPU {device_id}")
            
            # Simulate task execution
            # In production, this would launch actual GPU workload
            if task.task_type == "video_generation":
                await self.execute_video_generation(task, device_id)
            elif task.task_type == "model_training":
                await self.execute_model_training(task, device_id)
            elif task.task_type == "inference":
                await self.execute_inference(task, device_id)
                
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            task.status = "failed"
        finally:
            # Release GPU
            await self.release_gpu(task.task_id)
            
    async def execute_video_generation(self, task: GPUTask, device_id: int):
        """Execute video generation task"""
        # Set CUDA device
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        
        # Simulate video generation
        duration = task.metadata.get('duration_minutes', 5) * 60
        await asyncio.sleep(min(duration / 10, 30))  # Simulate processing
        
        logger.info(f"Video generation task {task.task_id} completed")
        
    async def execute_model_training(self, task: GPUTask, device_id: int):
        """Execute model training task"""
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        
        # Simulate training
        epochs = task.metadata.get('epochs', 10)
        await asyncio.sleep(min(epochs * 2, 60))  # Simulate training
        
        logger.info(f"Model training task {task.task_id} completed")
        
    async def execute_inference(self, task: GPUTask, device_id: int):
        """Execute inference task"""
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        
        # Simulate inference
        batch_size = task.metadata.get('batch_size', 32)
        await asyncio.sleep(min(batch_size / 10, 10))  # Simulate inference
        
        logger.info(f"Inference task {task.task_id} completed")
        
    async def get_status(self) -> Dict[str, Any]:
        """Get current GPU status"""
        status = {
            "devices": [],
            "queue_length": len(self.task_queue),
            "active_tasks": len(self.active_tasks),
            "total_memory": 0,
            "used_memory": 0,
            "free_memory": 0
        }
        
        for device_id, device in self.devices.items():
            device_info = {
                "id": device_id,
                "name": device.name,
                "status": device.status.value,
                "utilization": device.utilization,
                "memory": {
                    "total": device.memory_total,
                    "used": device.memory_used,
                    "free": device.memory_free
                },
                "temperature": device.temperature,
                "power": device.power_draw,
                "reserved_by": device.reserved_by
            }
            status["devices"].append(device_info)
            
            status["total_memory"] += device.memory_total
            status["used_memory"] += device.memory_used
            status["free_memory"] += device.memory_free
            
        return status
        
    async def get_queue_status(self) -> List[Dict[str, Any]]:
        """Get queue status"""
        queue_status = []
        
        for task in self.task_queue:
            queue_status.append({
                "task_id": task.task_id,
                "task_type": task.task_type,
                "priority": task.priority.name,
                "memory_required": task.memory_required,
                "estimated_duration": task.estimated_duration,
                "created_at": task.created_at.isoformat(),
                "position": self.task_queue.index(task) + 1
            })
            
        return queue_status
        
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued or active task"""
        # Check queue
        for task in self.task_queue:
            if task.task_id == task_id:
                self.task_queue.remove(task)
                logger.info(f"Cancelled queued task {task_id}")
                return True
                
        # Check active tasks
        if task_id in self.active_tasks:
            # Mark for cancellation
            self.active_tasks[task_id].status = "cancelled"
            await self.release_gpu(task_id)
            logger.info(f"Cancelled active task {task_id}")
            return True
            
        return False
        
    async def get_estimated_wait_time(self, memory_required: int) -> int:
        """Get estimated wait time for a task with given memory requirements"""
        # Calculate based on current queue and active tasks
        total_time = 0
        
        for task in self.task_queue:
            if task.memory_required <= memory_required:
                total_time += task.estimated_duration
                
        # Add time for active tasks
        for task in self.active_tasks.values():
            if task.started_at:
                elapsed = (datetime.utcnow() - task.started_at).total_seconds()
                remaining = max(0, task.estimated_duration - elapsed)
                total_time += remaining
                
        return int(total_time)
        
    async def optimize_allocation(self):
        """Optimize GPU allocation based on current workload"""
        # Analyze task patterns
        task_types = {}
        for task in self.active_tasks.values():
            task_types[task.task_type] = task_types.get(task.task_type, 0) + 1
            
        # Adjust allocation strategy based on workload
        if task_types.get("video_generation", 0) > task_types.get("inference", 0):
            # Prioritize memory for video generation
            self.memory_reserve_mb = 1024
        else:
            # Optimize for inference throughput
            self.memory_reserve_mb = 256
            
    async def check_stuck_tasks(self):
        """Check for stuck tasks and handle them"""
        now = datetime.utcnow()
        
        for task_id, task in self.active_tasks.items():
            if task.started_at:
                elapsed = (now - task.started_at).total_seconds()
                
                # Check if task exceeded estimated time by 2x
                if elapsed > task.estimated_duration * 2:
                    logger.warning(f"Task {task_id} appears stuck (elapsed: {elapsed}s)")
                    
                    # Release and requeue
                    await self.release_gpu(task_id)
                    task.status = "requeued"
                    self.task_queue.append(task)
                    
    async def store_metrics(self):
        """Store GPU metrics in Redis"""
        if not self.redis_client:
            return
            
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "devices": {}
        }
        
        for device_id, device in self.devices.items():
            metrics["devices"][device_id] = {
                "utilization": device.utilization,
                "memory_used": device.memory_used,
                "temperature": device.temperature,
                "power": device.power_draw,
                "status": device.status.value
            }
            
        await self.redis_client.setex(
            f"gpu:metrics:{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            3600,  # 1 hour TTL
            json.dumps(metrics)
        )
        
    async def store_allocation(self, task: GPUTask):
        """Store task allocation in Redis"""
        if not self.redis_client:
            return
            
        allocation = {
            "task_id": task.task_id,
            "device_id": task.device_id,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "estimated_duration": task.estimated_duration,
            "task_type": task.task_type,
            "user_id": task.user_id
        }
        
        await self.redis_client.setex(
            f"gpu:allocation:{task.task_id}",
            task.estimated_duration + 3600,  # Duration + 1 hour
            json.dumps(allocation)
        )
        
    async def store_task_in_redis(self, task: GPUTask):
        """Store task in Redis"""
        if not self.redis_client:
            return
            
        task_data = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "priority": task.priority.value,
            "memory_required": task.memory_required,
            "estimated_duration": task.estimated_duration,
            "user_id": task.user_id,
            "created_at": task.created_at.isoformat(),
            "status": task.status,
            "metadata": task.metadata
        }
        
        await self.redis_client.setex(
            f"gpu:task:{task.task_id}",
            86400,  # 24 hours
            json.dumps(task_data)
        )
        
    @asynccontextmanager
    async def gpu_context(self, task: GPUTask):
        """Context manager for GPU allocation"""
        device_id = None
        try:
            # Allocate GPU
            device_id = await self.allocate_gpu(task)
            if device_id is None:
                raise RuntimeError("No GPU available")
                
            yield device_id
            
        finally:
            # Release GPU
            if device_id is not None:
                await self.release_gpu(task.task_id)


# Global instance
gpu_manager = GPUResourceManager()