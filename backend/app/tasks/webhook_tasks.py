"""
Webhook Background Tasks
Handles webhook delivery, retries, and verification
"""
import asyncio
import logging
from typing import Dict, Any
from celery import Task

from app.core.celery_app import celery_app
from app.db.session import get_db
from app.services.webhook_service import webhook_service
import uuid

logger = logging.getLogger(__name__)


class WebhookTask(Task):
    """Base webhook task with error handling"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Webhook task {task_id} failed: {exc}")
        
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Webhook task {task_id} completed successfully")


@celery_app.task(bind=True, base=WebhookTask, max_retries=3, default_retry_delay=60)
def deliver_webhook(self, delivery_id: str):
    """
    Deliver webhook to endpoint
    """
    try:
        async def _deliver():
            async with get_db() as db:
                success = await webhook_service.deliver_webhook(
                    db=db,
                    delivery_id=uuid.UUID(delivery_id)
                )
                return success
                
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            success = loop.run_until_complete(_deliver())
            if not success:
                logger.warning(f"Webhook delivery {delivery_id} failed")
            return success
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Error delivering webhook {delivery_id}: {exc}")
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@celery_app.task(bind=True, base=WebhookTask)
def send_verification(self, webhook_id: str, payload: Dict[str, Any]):
    """
    Send webhook verification
    """
    try:
        async def _verify():
            async with get_db() as db:
                # Create a temporary delivery for verification
                from app.models.webhook import WebhookDelivery, Webhook
                from sqlalchemy import select
                
                # Get webhook
                result = await db.execute(
                    select(Webhook).where(Webhook.id == uuid.UUID(webhook_id))
                )
                webhook = result.scalar_one_or_none()
                
                if not webhook:
                    logger.error(f"Webhook {webhook_id} not found for verification")
                    return False
                    
                # Create temporary delivery
                delivery = WebhookDelivery(
                    webhook_id=webhook.id,
                    event_id=f"verification_{webhook_id}",
                    event_type="webhook.verification",
                    payload=payload["event"]["data"],
                    max_attempts=1
                )
                
                db.add(delivery)
                await db.commit()
                await db.refresh(delivery)
                
                # Deliver verification
                success = await webhook_service.deliver_webhook(db, delivery.id)
                
                if success:
                    logger.info(f"Webhook verification sent to {webhook.url}")
                else:
                    logger.warning(f"Webhook verification failed for {webhook.url}")
                    # Mark webhook as failed verification
                    webhook.status = "inactive"
                    webhook.last_error_message = "Verification failed"
                    await db.commit()
                    
                return success
                
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(_verify())
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Error sending webhook verification {webhook_id}: {exc}")
        return False


@celery_app.task(bind=True, base=WebhookTask)
def cleanup_old_deliveries(self, days_old: int = 30):
    """
    Clean up old webhook deliveries to save storage
    """
    try:
        async def _cleanup():
            async with get_db() as db:
                from app.models.webhook import WebhookDelivery
                from sqlalchemy import delete
                from datetime import datetime, timedelta
                
                cutoff_date = datetime.utcnow() - timedelta(days=days_old)
                
                # Delete old successful deliveries (keep failures for debugging)
                result = await db.execute(
                    delete(WebhookDelivery).where(
                        and_(
                            WebhookDelivery.delivered_at < cutoff_date,
                            WebhookDelivery.success == True
                        )
                    ).returning(WebhookDelivery.id)
                )
                
                deleted_count = len(result.fetchall())
                await db.commit()
                
                logger.info(f"Cleaned up {deleted_count} old webhook deliveries")
                return deleted_count
                
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(_cleanup())
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Error cleaning up webhook deliveries: {exc}")
        return 0


@celery_app.task(bind=True, base=WebhookTask)
def process_webhook_event_batch(self, event_data_list: list):
    """
    Process multiple webhook events in batch for better performance
    """
    try:
        async def _process_batch():
            async with get_db() as db:
                processed_count = 0
                
                for event_data in event_data_list:
                    try:
                        event_id = await webhook_service.trigger_event(
                            db=db,
                            event_type=event_data["event_type"],
                            source_type=event_data["source_type"],
                            source_id=event_data["source_id"],
                            user_id=event_data["user_id"],
                            data=event_data["data"],
                            metadata=event_data.get("metadata")
                        )
                        processed_count += 1
                        logger.debug(f"Processed webhook event {event_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process webhook event: {e}")
                        continue
                        
                logger.info(f"Processed {processed_count}/{len(event_data_list)} webhook events")
                return processed_count
                
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(_process_batch())
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Error processing webhook event batch: {exc}")
        return 0


@celery_app.task(bind=True, base=WebhookTask)
def update_webhook_stats(self):
    """
    Update webhook statistics and health status
    """
    try:
        async def _update_stats():
            async with get_db() as db:
                from app.models.webhook import Webhook
                from sqlalchemy import select, update, func
                from datetime import datetime, timedelta
                
                # Get all active webhooks
                result = await db.execute(
                    select(Webhook).where(Webhook.active == True)
                )
                webhooks = result.scalars().all()
                
                updated_count = 0
                
                for webhook in webhooks:
                    # Calculate recent success rate
                    last_24h = datetime.utcnow() - timedelta(hours=24)
                    
                    delivery_stats = await db.execute(
                        select(
                            func.count().label('total'),
                            func.count().filter_by(success=True).label('successful')
                        ).select_from(WebhookDelivery).where(
                            and_(
                                WebhookDelivery.webhook_id == webhook.id,
                                WebhookDelivery.delivered_at >= last_24h
                            )
                        )
                    )
                    
                    stats = delivery_stats.first()
                    
                    # Update webhook status based on recent performance
                    if stats.total > 0:
                        success_rate = stats.successful / stats.total
                        
                        if success_rate < 0.5 and webhook.failure_count > 10:
                            # Suspend webhook if success rate is too low
                            webhook.status = "suspended"
                            webhook.last_error_message = f"Suspended due to low success rate: {success_rate:.2%}"
                        elif webhook.status == "suspended" and success_rate > 0.8:
                            # Reactivate if performance improves
                            webhook.status = "active"
                            webhook.last_error_message = None
                            
                    updated_count += 1
                    
                await db.commit()
                
                logger.info(f"Updated stats for {updated_count} webhooks")
                return updated_count
                
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(_update_stats())
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Error updating webhook stats: {exc}")
        return 0


# Schedule periodic tasks
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup periodic webhook maintenance tasks"""
    
    # Clean up old deliveries daily
    sender.add_periodic_task(
        86400.0,  # 24 hours
        cleanup_old_deliveries.s(),
        name='cleanup_webhook_deliveries'
    )
    
    # Update webhook stats every hour
    sender.add_periodic_task(
        3600.0,  # 1 hour
        update_webhook_stats.s(),
        name='update_webhook_stats'
    )