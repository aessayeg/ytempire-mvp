"""
Real-time Inference Pipeline Service
Production model serving with TorchServe integration and optimizations
"""
import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import torch
import aiohttp
from collections import deque
import redis.asyncio as redis
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.core.cache import cache_service

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Supported model types"""
    CONTENT_GENERATION = "content_generation"
    QUALITY_SCORING = "quality_scoring"
    TREND_PREDICTION = "trend_prediction"
    ENGAGEMENT_PREDICTION = "engagement_prediction"
    THUMBNAIL_RANKING = "thumbnail_ranking"
    REVENUE_FORECAST = "revenue_forecast"
    USER_BEHAVIOR = "user_behavior"
    SENTIMENT_ANALYSIS = "sentiment_analysis"


class InferenceStatus(str, Enum):
    """Inference request status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CACHED = "cached"


@dataclass
class InferenceRequest:
    """Inference request structure"""
    request_id: str
    model_type: ModelType
    model_version: str
    input_data: Dict[str, Any]
    timestamp: datetime
    priority: int = 5  # 1-10, higher is more priority
    timeout_seconds: float = 30.0
    batch_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    """Inference response structure"""
    request_id: str
    model_type: ModelType
    model_version: str
    output: Union[Dict[str, Any], List[Any]]
    confidence: float
    latency_ms: float
    status: InferenceStatus
    timestamp: datetime
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelEndpoint:
    """Model endpoint configuration"""
    model_type: ModelType
    version: str
    url: str
    is_active: bool
    max_batch_size: int
    timeout_seconds: float
    fallback_endpoint: Optional[str] = None
    health_check_url: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class BatchRequest:
    """Batch of requests for processing"""
    batch_id: str
    requests: List[InferenceRequest]
    created_at: datetime
    model_type: ModelType
    size: int


class InferencePipeline:
    """Real-time inference pipeline with batching and caching"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.model_endpoints: Dict[ModelType, ModelEndpoint] = {}
        self.request_queues: Dict[ModelType, deque] = {}
        self.batch_processors: Dict[ModelType, asyncio.Task] = {}
        self.cache_ttl = 3600  # 1 hour cache for predictions
        self.max_batch_size = 32
        self.batch_timeout_ms = 100  # Max wait time for batching
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.torchserve_base_url = settings.TORCHSERVE_URL or "http://localhost:8080"
        self.management_url = settings.TORCHSERVE_MGMT_URL or "http://localhost:8081"
        
        # Performance tracking
        self.latency_history: Dict[ModelType, deque] = {}
        self.success_rate: Dict[ModelType, float] = {}
        self.request_count: Dict[ModelType, int] = {}
        
    async def initialize(self):
        """Initialize pipeline connections and model endpoints"""
        try:
            # Connect to Redis
            self.redis_client = await redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Initialize model endpoints
            await self._initialize_model_endpoints()
            
            # Initialize request queues
            for model_type in ModelType:
                self.request_queues[model_type] = deque()
                self.latency_history[model_type] = deque(maxlen=100)
                self.success_rate[model_type] = 1.0
                self.request_count[model_type] = 0
                
            # Start batch processors
            for model_type in ModelType:
                self.batch_processors[model_type] = asyncio.create_task(
                    self._batch_processor(model_type)
                )
                
            logger.info("Inference pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize inference pipeline: {e}")
            raise
            
    async def predict(
        self,
        model_type: ModelType,
        input_data: Dict[str, Any],
        model_version: Optional[str] = None,
        priority: int = 5,
        use_cache: bool = True
    ) -> InferenceResponse:
        """Make a prediction using specified model"""
        request_id = self._generate_request_id()
        
        # Check cache first
        if use_cache:
            cached_response = await self._get_cached_prediction(
                model_type, input_data
            )
            if cached_response:
                cached_response.status = InferenceStatus.CACHED
                return cached_response
                
        # Get model endpoint
        endpoint = self.model_endpoints.get(model_type)
        if not endpoint or not endpoint.is_active:
            raise ValueError(f"No active endpoint for model type {model_type}")
            
        # Create request
        request = InferenceRequest(
            request_id=request_id,
            model_type=model_type,
            model_version=model_version or endpoint.version,
            input_data=input_data,
            timestamp=datetime.utcnow(),
            priority=priority
        )
        
        # Add to queue for batching
        self.request_queues[model_type].append(request)
        
        # Wait for response
        response = await self._wait_for_response(request_id, request.timeout_seconds)
        
        # Cache successful response
        if response.status == InferenceStatus.COMPLETED and use_cache:
            await self._cache_prediction(model_type, input_data, response)
            
        return response
        
    async def predict_batch(
        self,
        model_type: ModelType,
        input_batch: List[Dict[str, Any]],
        model_version: Optional[str] = None
    ) -> List[InferenceResponse]:
        """Make batch predictions"""
        endpoint = self.model_endpoints.get(model_type)
        if not endpoint or not endpoint.is_active:
            raise ValueError(f"No active endpoint for model type {model_type}")
            
        # Create batch request
        batch_id = self._generate_batch_id()
        requests = []
        
        for input_data in input_batch:
            request = InferenceRequest(
                request_id=self._generate_request_id(),
                model_type=model_type,
                model_version=model_version or endpoint.version,
                input_data=input_data,
                timestamp=datetime.utcnow(),
                batch_size=len(input_batch)
            )
            requests.append(request)
            
        # Process batch directly
        responses = await self._process_batch(
            BatchRequest(
                batch_id=batch_id,
                requests=requests,
                created_at=datetime.utcnow(),
                model_type=model_type,
                size=len(requests)
            )
        )
        
        return responses
        
    async def _batch_processor(self, model_type: ModelType):
        """Background task to process request batches"""
        while True:
            try:
                # Collect requests for batching
                batch = await self._collect_batch(model_type)
                
                if batch:
                    # Process batch
                    await self._process_batch(batch)
                    
                await asyncio.sleep(self.batch_timeout_ms / 1000)
                
            except Exception as e:
                logger.error(f"Batch processor error for {model_type}: {e}")
                await asyncio.sleep(1)
                
    async def _collect_batch(self, model_type: ModelType) -> Optional[BatchRequest]:
        """Collect requests into a batch"""
        queue = self.request_queues[model_type]
        
        if not queue:
            return None
            
        batch_requests = []
        batch_start_time = time.time()
        
        # Collect up to max_batch_size or until timeout
        while len(batch_requests) < self.max_batch_size:
            if not queue:
                # Wait a bit for more requests
                if (time.time() - batch_start_time) * 1000 < self.batch_timeout_ms:
                    await asyncio.sleep(0.01)
                    continue
                else:
                    break
                    
            # Sort by priority and get highest priority request
            if queue:
                request = queue.popleft()
                batch_requests.append(request)
                
        if not batch_requests:
            return None
            
        return BatchRequest(
            batch_id=self._generate_batch_id(),
            requests=batch_requests,
            created_at=datetime.utcnow(),
            model_type=model_type,
            size=len(batch_requests)
        )
        
    async def _process_batch(self, batch: BatchRequest) -> List[InferenceResponse]:
        """Process a batch of requests"""
        endpoint = self.model_endpoints.get(batch.model_type)
        if not endpoint:
            return self._create_failed_responses(
                batch.requests, "No endpoint available"
            )
            
        start_time = time.time()
        responses = []
        
        try:
            # Prepare batch input
            batch_input = self._prepare_batch_input(batch.requests)
            
            # Call TorchServe endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{endpoint.url}/predictions/{batch.model_type.value}",
                    json=batch_input,
                    timeout=aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
                ) as response:
                    if response.status == 200:
                        predictions = await response.json()
                        
                        # Create responses
                        latency_ms = (time.time() - start_time) * 1000
                        
                        for i, request in enumerate(batch.requests):
                            pred = predictions[i] if i < len(predictions) else {}
                            
                            resp = InferenceResponse(
                                request_id=request.request_id,
                                model_type=request.model_type,
                                model_version=request.model_version,
                                output=pred.get("prediction", {}),
                                confidence=pred.get("confidence", 0.0),
                                latency_ms=latency_ms,
                                status=InferenceStatus.COMPLETED,
                                timestamp=datetime.utcnow()
                            )
                            responses.append(resp)
                            
                            # Store response
                            await self._store_response(resp)
                            
                        # Update metrics
                        self._update_metrics(batch.model_type, latency_ms, True)
                        
                    else:
                        error_msg = f"TorchServe returned status {response.status}"
                        responses = self._create_failed_responses(
                            batch.requests, error_msg
                        )
                        self._update_metrics(batch.model_type, 0, False)
                        
        except asyncio.TimeoutError:
            responses = self._create_failed_responses(
                batch.requests, "Request timeout", InferenceStatus.TIMEOUT
            )
            self._update_metrics(batch.model_type, 0, False)
            
            # Try fallback endpoint if available
            if endpoint.fallback_endpoint:
                logger.info(f"Trying fallback endpoint for {batch.model_type}")
                # Implement fallback logic here
                
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            responses = self._create_failed_responses(batch.requests, str(e))
            self._update_metrics(batch.model_type, 0, False)
            
        return responses
        
    async def _initialize_model_endpoints(self):
        """Initialize model endpoints from TorchServe"""
        try:
            # Register default models
            default_models = {
                ModelType.CONTENT_GENERATION: {
                    "url": f"{self.torchserve_base_url}",
                    "version": "1.0",
                    "max_batch_size": 8,
                    "timeout_seconds": 60
                },
                ModelType.QUALITY_SCORING: {
                    "url": f"{self.torchserve_base_url}",
                    "version": "1.0",
                    "max_batch_size": 32,
                    "timeout_seconds": 10
                },
                ModelType.TREND_PREDICTION: {
                    "url": f"{self.torchserve_base_url}",
                    "version": "1.0",
                    "max_batch_size": 16,
                    "timeout_seconds": 20
                },
                ModelType.ENGAGEMENT_PREDICTION: {
                    "url": f"{self.torchserve_base_url}",
                    "version": "1.0",
                    "max_batch_size": 32,
                    "timeout_seconds": 10
                },
                ModelType.THUMBNAIL_RANKING: {
                    "url": f"{self.torchserve_base_url}",
                    "version": "1.0",
                    "max_batch_size": 16,
                    "timeout_seconds": 15
                }
            }
            
            for model_type, config in default_models.items():
                self.model_endpoints[model_type] = ModelEndpoint(
                    model_type=model_type,
                    version=config["version"],
                    url=config["url"],
                    is_active=True,
                    max_batch_size=config["max_batch_size"],
                    timeout_seconds=config["timeout_seconds"],
                    health_check_url=f"{self.management_url}/models/{model_type.value}"
                )
                
            # Check model health
            await self._check_model_health()
            
        except Exception as e:
            logger.error(f"Failed to initialize model endpoints: {e}")
            # Use fallback configuration
            
    async def _check_model_health(self):
        """Check health of all model endpoints"""
        for model_type, endpoint in self.model_endpoints.items():
            if endpoint.health_check_url:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            endpoint.health_check_url,
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                endpoint.is_active = data.get("status") == "Healthy"
                            else:
                                endpoint.is_active = False
                                
                except Exception as e:
                    logger.warning(f"Health check failed for {model_type}: {e}")
                    endpoint.is_active = False
                    
    async def register_model(
        self,
        model_type: ModelType,
        model_path: str,
        handler: str = "default",
        batch_size: int = 8,
        max_workers: int = 2
    ) -> bool:
        """Register a new model with TorchServe"""
        try:
            # Create model archive
            model_config = {
                "model_name": model_type.value,
                "model_path": model_path,
                "handler": handler,
                "batch_size": batch_size,
                "max_workers": max_workers
            }
            
            # Register with TorchServe management API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.management_url}/models",
                    data=model_config
                ) as response:
                    if response.status == 200:
                        logger.info(f"Registered model {model_type.value}")
                        
                        # Update endpoint
                        self.model_endpoints[model_type] = ModelEndpoint(
                            model_type=model_type,
                            version="1.0",
                            url=self.torchserve_base_url,
                            is_active=True,
                            max_batch_size=batch_size,
                            timeout_seconds=30
                        )
                        return True
                    else:
                        logger.error(f"Failed to register model: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Model registration error: {e}")
            return False
            
    async def update_model(
        self,
        model_type: ModelType,
        new_model_path: str,
        version: str
    ) -> bool:
        """Update model version (A/B testing support)"""
        try:
            # Register new version
            success = await self.register_model(
                model_type=model_type,
                model_path=new_model_path,
                batch_size=self.model_endpoints[model_type].max_batch_size
            )
            
            if success:
                # Update version
                self.model_endpoints[model_type].version = version
                
                # Gradual rollout can be implemented here
                logger.info(f"Updated {model_type.value} to version {version}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Model update error: {e}")
            return False
            
    async def get_model_metrics(
        self,
        model_type: Optional[ModelType] = None
    ) -> Dict[str, Any]:
        """Get model performance metrics"""
        if model_type:
            return self._get_single_model_metrics(model_type)
        else:
            metrics = {}
            for mt in ModelType:
                metrics[mt.value] = self._get_single_model_metrics(mt)
            return metrics
            
    def _get_single_model_metrics(self, model_type: ModelType) -> Dict[str, Any]:
        """Get metrics for a single model"""
        latencies = list(self.latency_history.get(model_type, []))
        
        return {
            "model_type": model_type.value,
            "endpoint": self.model_endpoints.get(model_type, {}).url if model_type in self.model_endpoints else None,
            "version": self.model_endpoints.get(model_type, {}).version if model_type in self.model_endpoints else None,
            "is_active": self.model_endpoints.get(model_type, {}).is_active if model_type in self.model_endpoints else False,
            "request_count": self.request_count.get(model_type, 0),
            "success_rate": self.success_rate.get(model_type, 0),
            "avg_latency_ms": np.mean(latencies) if latencies else 0,
            "p50_latency_ms": np.percentile(latencies, 50) if latencies else 0,
            "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
            "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0
        }
        
    def _prepare_batch_input(self, requests: List[InferenceRequest]) -> Dict[str, Any]:
        """Prepare batch input for TorchServe"""
        # Combine all input data into batch format
        batch_input = {
            "instances": [req.input_data for req in requests],
            "metadata": {
                "batch_size": len(requests),
                "model_version": requests[0].model_version if requests else "1.0"
            }
        }
        return batch_input
        
    def _create_failed_responses(
        self,
        requests: List[InferenceRequest],
        error_message: str,
        status: InferenceStatus = InferenceStatus.FAILED
    ) -> List[InferenceResponse]:
        """Create failed response objects"""
        responses = []
        for request in requests:
            response = InferenceResponse(
                request_id=request.request_id,
                model_type=request.model_type,
                model_version=request.model_version,
                output={},
                confidence=0.0,
                latency_ms=0.0,
                status=status,
                timestamp=datetime.utcnow(),
                error_message=error_message
            )
            responses.append(response)
            
        return responses
        
    async def _wait_for_response(
        self,
        request_id: str,
        timeout_seconds: float
    ) -> InferenceResponse:
        """Wait for inference response"""
        start_time = time.time()
        
        while (time.time() - start_time) < timeout_seconds:
            # Check for response in Redis
            response_data = await self.redis_client.get(f"inference:response:{request_id}")
            
            if response_data:
                response_dict = json.loads(response_data)
                # Convert timestamp
                response_dict['timestamp'] = datetime.fromisoformat(response_dict['timestamp'])
                return InferenceResponse(**response_dict)
                
            await asyncio.sleep(0.1)
            
        # Timeout
        return InferenceResponse(
            request_id=request_id,
            model_type=ModelType.CONTENT_GENERATION,  # Default
            model_version="1.0",
            output={},
            confidence=0.0,
            latency_ms=(time.time() - start_time) * 1000,
            status=InferenceStatus.TIMEOUT,
            timestamp=datetime.utcnow(),
            error_message="Request timeout"
        )
        
    async def _store_response(self, response: InferenceResponse):
        """Store response in Redis"""
        key = f"inference:response:{response.request_id}"
        await self.redis_client.setex(
            key,
            300,  # 5 minutes TTL
            json.dumps(asdict(response), default=str)
        )
        
    async def _get_cached_prediction(
        self,
        model_type: ModelType,
        input_data: Dict[str, Any]
    ) -> Optional[InferenceResponse]:
        """Get cached prediction if available"""
        cache_key = self._generate_cache_key(model_type, input_data)
        cached_data = await cache_service.get(cache_key)
        
        if cached_data:
            # Convert to response object
            cached_data['timestamp'] = datetime.fromisoformat(cached_data['timestamp'])
            return InferenceResponse(**cached_data)
            
        return None
        
    async def _cache_prediction(
        self,
        model_type: ModelType,
        input_data: Dict[str, Any],
        response: InferenceResponse
    ):
        """Cache prediction result"""
        cache_key = self._generate_cache_key(model_type, input_data)
        await cache_service.set(
            cache_key,
            asdict(response, default=str),
            self.cache_ttl
        )
        
    def _generate_cache_key(
        self,
        model_type: ModelType,
        input_data: Dict[str, Any]
    ) -> str:
        """Generate cache key for prediction"""
        # Create deterministic key from input
        input_str = json.dumps(input_data, sort_keys=True)
        input_hash = hash(input_str)
        return f"inference:cache:{model_type.value}:{input_hash}"
        
    def _update_metrics(
        self,
        model_type: ModelType,
        latency_ms: float,
        success: bool
    ):
        """Update performance metrics"""
        self.request_count[model_type] = self.request_count.get(model_type, 0) + 1
        
        if latency_ms > 0:
            self.latency_history[model_type].append(latency_ms)
            
        # Update success rate (exponential moving average)
        alpha = 0.1  # Smoothing factor
        current_rate = self.success_rate.get(model_type, 1.0)
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
        self.success_rate[model_type] = new_rate
        
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        return f"req_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        
    def _generate_batch_id(self) -> str:
        """Generate unique batch ID"""
        return f"batch_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"


# Singleton instance
inference_pipeline = InferencePipeline()