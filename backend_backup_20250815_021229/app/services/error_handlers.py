"""
Service-specific error handlers with fallback chains and recovery strategies
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from datetime import datetime

from app.core.exceptions import (
    OpenAIException,
    ElevenLabsException,
    YouTubeAPIException,
    ExternalServiceException,
    ThresholdExceededException,
    QuotaExceededException
)

# Import the error handling framework
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from misc.error_handling_framework import (
    ErrorHandler,
    ServiceErrorHandler,
    ErrorSeverity,
    ErrorContext,
    RecoveryStrategy,
    with_retry,
    with_circuit_breaker,
    with_fallback
)

logger = logging.getLogger(__name__)

# Initialize global error handler
error_handler = ErrorHandler()

class AIServiceErrorHandler(ServiceErrorHandler):
    """Error handler for AI services with fallback chain"""
    
    def __init__(self):
        super().__init__("ai_services", error_handler)
        self.fallback_chain = {
            "gpt-4": "gpt-3.5-turbo",
            "gpt-3.5-turbo": "claude-2",
            "claude-2": None  # No fallback after Claude
        }
    
    async def handle_openai_error(
        self,
        error: Exception,
        model: str,
        prompt: str,
        **kwargs
    ) -> Optional[str]:
        """Handle OpenAI API errors with fallback"""
        
        # Check if it's a quota/rate limit error
        if "rate_limit" in str(error).lower() or "quota" in str(error).lower():
            severity = ErrorSeverity.HIGH
            strategy = RecoveryStrategy.FALLBACK
        else:
            severity = ErrorSeverity.MEDIUM
            strategy = RecoveryStrategy.RETRY
        
        context = ErrorContext(
            error_type=type(error),
            error_message=str(error),
            severity=severity,
            service="openai",
            operation="completion",
            timestamp=datetime.now(),
            metadata={
                "model": model,
                "prompt_length": len(prompt),
                "fallback_model": self.fallback_chain.get(model)
            }
        )
        
        if strategy == RecoveryStrategy.FALLBACK:
            fallback_model = self.fallback_chain.get(model)
            if fallback_model:
                logger.info(f"Falling back from {model} to {fallback_model}")
                # Return fallback model to use
                return fallback_model
        
        # Let the error handler manage retry logic
        return await error_handler.handle_error(error, context, strategy)
    
    @with_retry(max_retries=3, exceptions=(OpenAIException,))
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60)
    async def call_openai_with_fallback(
        self,
        prompt: str,
        model: str = "gpt-4",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Call OpenAI with automatic fallback on failure"""
        
        current_model = model
        last_error = None
        
        while current_model:
            try:
                # Simulate OpenAI call (replace with actual implementation)
                from app.services.ai_services import openai_service
                response = await openai_service.generate(
                    prompt=prompt,
                    model=current_model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                logger.info(f"Successfully used model: {current_model}")
                return {
                    "response": response,
                    "model_used": current_model,
                    "fallback_used": current_model != model
                }
                
            except Exception as e:
                last_error = e
                fallback_model = await self.handle_openai_error(
                    e, current_model, prompt
                )
                
                if fallback_model:
                    current_model = fallback_model
                else:
                    break
        
        # All models failed
        raise OpenAIException(f"All AI models failed. Last error: {last_error}")

class YouTubeServiceErrorHandler(ServiceErrorHandler):
    """Error handler for YouTube API with account rotation"""
    
    def __init__(self):
        super().__init__("youtube", error_handler)
        self.account_health_scores = {}
        self.account_rotation_index = 0
        self.total_accounts = 15
    
    async def handle_youtube_error(
        self,
        error: Exception,
        account_id: int,
        operation: str
    ) -> Optional[int]:
        """Handle YouTube API errors with account rotation"""
        
        error_str = str(error).lower()
        
        # Quota exceeded - rotate account
        if "quota" in error_str or "limit" in error_str:
            return await self._rotate_account(account_id)
        
        # Authentication error - mark account as unhealthy
        if "auth" in error_str or "forbidden" in error_str:
            await self._mark_account_unhealthy(account_id)
            return await self._rotate_account(account_id)
        
        # Rate limit - wait and retry
        if "rate" in error_str:
            await asyncio.sleep(5)  # Wait 5 seconds
            return account_id  # Retry with same account
        
        # Other errors - use standard retry
        context = ErrorContext(
            error_type=type(error),
            error_message=str(error),
            severity=ErrorSeverity.MEDIUM,
            service="youtube",
            operation=operation,
            timestamp=datetime.now(),
            metadata={"account_id": account_id}
        )
        
        await error_handler.handle_error(error, context, RecoveryStrategy.RETRY)
        return account_id
    
    async def _rotate_account(self, current_account: int) -> int:
        """Rotate to next healthy YouTube account"""
        
        # Find next healthy account
        for _ in range(self.total_accounts):
            self.account_rotation_index = (self.account_rotation_index + 1) % self.total_accounts
            next_account = self.account_rotation_index
            
            if self._is_account_healthy(next_account):
                logger.info(f"Rotating from account {current_account} to {next_account}")
                return next_account
        
        # No healthy accounts available
        raise YouTubeAPIException("All YouTube accounts are exhausted or unhealthy")
    
    async def _mark_account_unhealthy(self, account_id: int):
        """Mark an account as unhealthy"""
        self.account_health_scores[account_id] = 0
        logger.warning(f"YouTube account {account_id} marked as unhealthy")
    
    def _is_account_healthy(self, account_id: int) -> bool:
        """Check if account is healthy"""
        return self.account_health_scores.get(account_id, 100) > 50
    
    @with_retry(max_retries=5, exceptions=(YouTubeAPIException,))
    async def upload_video_with_rotation(
        self,
        video_path: str,
        metadata: Dict[str, Any],
        account_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Upload video with automatic account rotation on failure"""
        
        if account_id is None:
            account_id = self.account_rotation_index
        
        max_attempts = self.total_accounts
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                # Simulate YouTube upload (replace with actual implementation)
                from app.services.youtube_multi_account import youtube_service
                result = await youtube_service.upload_video(
                    video_path=video_path,
                    metadata=metadata,
                    account_id=account_id
                )
                
                logger.info(f"Successfully uploaded video using account {account_id}")
                return result
                
            except Exception as e:
                last_error = e
                new_account = await self.handle_youtube_error(e, account_id, "upload")
                
                if new_account != account_id:
                    account_id = new_account
                else:
                    # Same account, error was handled
                    continue
        
        raise YouTubeAPIException(f"Failed to upload after {max_attempts} attempts: {last_error}")

class PaymentServiceErrorHandler(ServiceErrorHandler):
    """Error handler for payment services with idempotency"""
    
    def __init__(self):
        super().__init__("payment", error_handler)
        self.idempotency_keys = {}
        self.transaction_logs = []
    
    async def handle_payment_error(
        self,
        error: Exception,
        transaction_id: str,
        operation: str
    ) -> bool:
        """Handle payment errors with transaction safety"""
        
        error_str = str(error).lower()
        
        # Network errors - safe to retry
        if "network" in error_str or "timeout" in error_str:
            return True  # Can retry
        
        # Duplicate transaction - check idempotency
        if "duplicate" in error_str:
            return await self._check_idempotency(transaction_id)
        
        # Insufficient funds - don't retry
        if "insufficient" in error_str or "declined" in error_str:
            logger.error(f"Payment declined for transaction {transaction_id}")
            return False  # Don't retry
        
        # Other errors - log and decide
        context = ErrorContext(
            error_type=type(error),
            error_message=str(error),
            severity=ErrorSeverity.HIGH,
            service="payment",
            operation=operation,
            timestamp=datetime.now(),
            metadata={"transaction_id": transaction_id}
        )
        
        await error_handler.handle_error(error, context, RecoveryStrategy.COMPENSATE)
        return False
    
    async def _check_idempotency(self, transaction_id: str) -> bool:
        """Check if transaction was already processed"""
        if transaction_id in self.idempotency_keys:
            logger.info(f"Transaction {transaction_id} already processed")
            return False
        return True
    
    async def _rollback_transaction(self, transaction_id: str):
        """Rollback a failed transaction"""
        logger.info(f"Rolling back transaction {transaction_id}")
        
        # Log the rollback
        self.transaction_logs.append({
            "transaction_id": transaction_id,
            "action": "rollback",
            "timestamp": datetime.now()
        })
        
        # Implement actual rollback logic
        # This would involve database updates, refunds, etc.
    
    @with_retry(max_retries=2, exceptions=(Exception,))
    async def process_payment_with_safety(
        self,
        amount: float,
        currency: str,
        customer_id: str,
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process payment with transaction safety"""
        
        transaction_id = idempotency_key or f"txn_{datetime.now().timestamp()}"
        
        # Check idempotency
        if not await self._check_idempotency(transaction_id):
            # Transaction already processed
            return {"status": "duplicate", "transaction_id": transaction_id}
        
        try:
            # Record idempotency key
            self.idempotency_keys[transaction_id] = datetime.now()
            
            # Simulate payment processing (replace with actual implementation)
            from app.services.payment_service_enhanced import payment_service
            result = await payment_service.charge(
                amount=amount,
                currency=currency,
                customer_id=customer_id,
                idempotency_key=transaction_id
            )
            
            # Log successful transaction
            self.transaction_logs.append({
                "transaction_id": transaction_id,
                "action": "success",
                "amount": amount,
                "timestamp": datetime.now()
            })
            
            return result
            
        except Exception as e:
            can_retry = await self.handle_payment_error(e, transaction_id, "charge")
            
            if not can_retry:
                # Rollback if needed
                await self._rollback_transaction(transaction_id)
                
                # Remove idempotency key so it can be retried later
                self.idempotency_keys.pop(transaction_id, None)
            
            raise

class VideoProcessingErrorHandler(ServiceErrorHandler):
    """Error handler for video processing with checkpoint recovery"""
    
    def __init__(self):
        super().__init__("video_processing", error_handler)
        self.checkpoints = {}
        self.processing_stages = [
            "script_generation",
            "voice_synthesis",
            "image_generation",
            "video_assembly",
            "quality_check",
            "upload"
        ]
    
    async def handle_processing_error(
        self,
        error: Exception,
        video_id: str,
        stage: str
    ) -> Optional[str]:
        """Handle video processing errors with checkpoint recovery"""
        
        # Save checkpoint
        await self._save_checkpoint(video_id, stage)
        
        error_str = str(error).lower()
        
        # Memory errors - reduce quality/resolution
        if "memory" in error_str or "resource" in error_str:
            logger.warning(f"Resource error at {stage}, reducing quality")
            return "reduce_quality"
        
        # File errors - check paths and permissions
        if "file" in error_str or "path" in error_str:
            logger.error(f"File error at {stage}: {error}")
            return "check_files"
        
        # Timeout - resume from checkpoint
        if "timeout" in error_str:
            logger.info(f"Timeout at {stage}, will resume from checkpoint")
            return "resume"
        
        # Other errors
        context = ErrorContext(
            error_type=type(error),
            error_message=str(error),
            severity=ErrorSeverity.HIGH,
            service="video_processing",
            operation=stage,
            timestamp=datetime.now(),
            metadata={
                "video_id": video_id,
                "stage": stage,
                "checkpoint": self.checkpoints.get(video_id)
            }
        )
        
        await error_handler.handle_error(error, context, RecoveryStrategy.RETRY)
        return "retry"
    
    async def _save_checkpoint(self, video_id: str, stage: str):
        """Save processing checkpoint"""
        if video_id not in self.checkpoints:
            self.checkpoints[video_id] = {}
        
        self.checkpoints[video_id][stage] = {
            "timestamp": datetime.now(),
            "status": "completed"
        }
        
        logger.info(f"Checkpoint saved for video {video_id} at stage {stage}")
    
    async def _get_resume_stage(self, video_id: str) -> Optional[str]:
        """Get stage to resume from"""
        if video_id not in self.checkpoints:
            return self.processing_stages[0]
        
        completed_stages = self.checkpoints[video_id].keys()
        
        for i, stage in enumerate(self.processing_stages):
            if stage not in completed_stages:
                return stage
        
        return None  # All stages completed
    
    @with_retry(max_retries=3)
    async def process_video_with_recovery(
        self,
        video_id: str,
        config: Dict[str, Any],
        resume: bool = False
    ) -> Dict[str, Any]:
        """Process video with checkpoint recovery"""
        
        # Determine starting stage
        if resume:
            start_stage = await self._get_resume_stage(video_id)
            if not start_stage:
                logger.info(f"Video {video_id} already completed")
                return {"status": "completed", "video_id": video_id}
        else:
            start_stage = self.processing_stages[0]
        
        start_index = self.processing_stages.index(start_stage)
        
        # Process each stage
        for stage in self.processing_stages[start_index:]:
            try:
                logger.info(f"Processing video {video_id} - Stage: {stage}")
                
                # Simulate stage processing (replace with actual implementation)
                if stage == "script_generation":
                    # Generate script
                    pass
                elif stage == "voice_synthesis":
                    # Synthesize voice
                    pass
                elif stage == "image_generation":
                    # Generate images
                    pass
                elif stage == "video_assembly":
                    # Assemble video
                    pass
                elif stage == "quality_check":
                    # Check quality
                    pass
                elif stage == "upload":
                    # Upload video
                    pass
                
                # Save checkpoint after successful stage
                await self._save_checkpoint(video_id, stage)
                
            except Exception as e:
                action = await self.handle_processing_error(e, video_id, stage)
                
                if action == "reduce_quality":
                    # Reduce quality settings and retry
                    config["quality"] = "medium"
                    continue
                elif action == "resume":
                    # Will resume from last checkpoint
                    return await self.process_video_with_recovery(
                        video_id, config, resume=True
                    )
                else:
                    raise
        
        return {
            "status": "completed",
            "video_id": video_id,
            "checkpoints": self.checkpoints.get(video_id, {})
        }

# Initialize service error handlers
ai_error_handler = AIServiceErrorHandler()
youtube_error_handler = YouTubeServiceErrorHandler()
payment_error_handler = PaymentServiceErrorHandler()
video_error_handler = VideoProcessingErrorHandler()

# Export for use in services
__all__ = [
    'ai_error_handler',
    'youtube_error_handler',
    'payment_error_handler',
    'video_error_handler',
    'error_handler'
]