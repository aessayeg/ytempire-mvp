"""
Subscription Service for YTEmpire
Handles subscription management, billing cycles, and usage tracking
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
import stripe
import asyncio

from sqlalchemy import select, func, and_, or_
from app.db.session import AsyncSessionLocal
from app.models.user import User
from app.models.subscription import Subscription, SubscriptionPlan, SubscriptionStatus
from app.models.billing import Invoice, PaymentHistory
from app.models.cost import Cost
from app.services.payment_service_enhanced import PaymentService
from app.services.invoice_generator import invoice_generator
from app.services.notification_service import notification_service
from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY


class SubscriptionService:
    """
    Manages user subscriptions and billing cycles
    """

    def __init__(self):
        self.payment_service = PaymentService()
        self.plans = {
            "starter": {
                "name": "Starter",
                "price": Decimal("29.99"),
                "videos_per_month": 30,
                "channels": 1,
                "features": ["Basic analytics", "Email support"],
            },
            "growth": {
                "name": "Growth",
                "price": Decimal("99.99"),
                "videos_per_month": 150,
                "channels": 3,
                "features": [
                    "Advanced analytics",
                    "Priority support",
                    "Custom thumbnails",
                ],
            },
            "scale": {
                "name": "Scale",
                "price": Decimal("299.99"),
                "videos_per_month": 500,
                "channels": 5,
                "features": [
                    "Full analytics",
                    "Dedicated support",
                    "API access",
                    "White label",
                ],
            },
            "enterprise": {
                "name": "Enterprise",
                "price": Decimal("999.99"),
                "videos_per_month": -1,  # Unlimited
                "channels": -1,  # Unlimited
                "features": [
                    "Everything in Scale",
                    "Custom integrations",
                    "SLA",
                    "Training",
                ],
            },
        }

    async def create_subscription(
        self, user_id: str, plan_id: str, payment_method_id: str, trial_days: int = 0
    ) -> Dict[str, Any]:
        """
        Create a new subscription for user

        Args:
            user_id: User ID
            plan_id: Subscription plan ID
            payment_method_id: Stripe payment method ID
            trial_days: Number of trial days

        Returns:
            Subscription details
        """
        try:
            if plan_id not in self.plans:
                return {"success": False, "error": "Invalid plan ID"}

            plan = self.plans[plan_id]

            async with AsyncSessionLocal() as db:
                # Check if user already has active subscription
                existing = await db.execute(
                    select(Subscription).where(
                        and_(
                            Subscription.user_id == user_id,
                            Subscription.status.in_(
                                [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
                            ),
                        )
                    )
                )

                if existing.scalar():
                    return {
                        "success": False,
                        "error": "User already has an active subscription",
                    }

                # Get user
                user = await db.get(User, user_id)
                if not user:
                    return {"success": False, "error": "User not found"}

                # Create Stripe subscription
                stripe_sub = await self._create_stripe_subscription(
                    user.email, plan_id, payment_method_id, trial_days
                )

                if not stripe_sub:
                    return {
                        "success": False,
                        "error": "Failed to create Stripe subscription",
                    }

                # Calculate dates
                start_date = datetime.utcnow()
                if trial_days > 0:
                    trial_end = start_date + timedelta(days=trial_days)
                    next_billing = trial_end
                    status = SubscriptionStatus.TRIALING
                else:
                    trial_end = None
                    next_billing = start_date + timedelta(days=30)
                    status = SubscriptionStatus.ACTIVE

                # Create subscription record
                subscription = Subscription(
                    user_id=user_id,
                    plan_id=plan_id,
                    plan_name=plan["name"],
                    stripe_subscription_id=stripe_sub["id"],
                    stripe_customer_id=stripe_sub["customer"],
                    status=status,
                    monthly_price=plan["price"],
                    video_limit=plan["videos_per_month"],
                    channel_limit=plan["channels"],
                    features=plan["features"],
                    current_period_start=start_date,
                    current_period_end=next_billing,
                    trial_end=trial_end,
                    next_billing_date=next_billing,
                )

                db.add(subscription)
                await db.commit()
                await db.refresh(subscription)

                # Send welcome notification
                await notification_service.send_notification(
                    user_id=user_id,
                    title="Subscription Created",
                    message=f"Welcome to YTEmpire {plan['name']} plan!",
                    type="success",
                )

                return {
                    "success": True,
                    "subscription_id": str(subscription.id),
                    "plan": plan["name"],
                    "status": status.value,
                    "next_billing": next_billing.isoformat(),
                }

        except Exception as e:
            logger.error(f"Subscription creation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _create_stripe_subscription(
        self, email: str, plan_id: str, payment_method_id: str, trial_days: int
    ) -> Optional[Dict]:
        """Create subscription in Stripe"""
        try:
            # Create or get customer
            customers = stripe.Customer.list(email=email, limit=1)
            if customers.data:
                customer = customers.data[0]
            else:
                customer = stripe.Customer.create(
                    email=email,
                    payment_method=payment_method_id,
                    invoice_settings={"default_payment_method": payment_method_id},
                )

            # Attach payment method
            stripe.PaymentMethod.attach(payment_method_id, customer=customer.id)

            # Create subscription
            subscription = stripe.Subscription.create(
                customer=customer.id,
                items=[{"price": settings.STRIPE_PRICE_IDS.get(plan_id)}],
                trial_period_days=trial_days if trial_days > 0 else None,
                payment_settings={
                    "payment_method_types": ["card"],
                    "save_default_payment_method": "on_subscription",
                },
            )

            return subscription

        except stripe.error.StripeError as e:
            logger.error(f"Stripe subscription creation failed: {str(e)}")
            return None

    async def cancel_subscription(
        self,
        subscription_id: str,
        immediate: bool = False,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Cancel a subscription

        Args:
            subscription_id: Subscription ID
            immediate: Cancel immediately vs end of period
            reason: Cancellation reason

        Returns:
            Cancellation result
        """
        try:
            async with AsyncSessionLocal() as db:
                subscription = await db.get(Subscription, subscription_id)
                if not subscription:
                    return {"success": False, "error": "Subscription not found"}

                # Cancel in Stripe
                if subscription.stripe_subscription_id:
                    stripe.Subscription.modify(
                        subscription.stripe_subscription_id,
                        cancel_at_period_end=not immediate,
                    )

                # Update subscription status
                if immediate:
                    subscription.status = SubscriptionStatus.CANCELLED
                    subscription.cancelled_at = datetime.utcnow()
                else:
                    subscription.status = SubscriptionStatus.PENDING_CANCELLATION
                    subscription.cancel_at_period_end = True

                subscription.cancellation_reason = reason
                await db.commit()

                # Send notification
                await notification_service.send_notification(
                    user_id=subscription.user_id,
                    title="Subscription Cancelled",
                    message=f"Your subscription will end on {subscription.current_period_end}",
                    type="info",
                )

                return {
                    "success": True,
                    "subscription_id": subscription_id,
                    "cancelled_at": datetime.utcnow().isoformat(),
                    "effective_date": subscription.current_period_end.isoformat(),
                }

        except Exception as e:
            logger.error(f"Subscription cancellation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def upgrade_subscription(
        self, subscription_id: str, new_plan_id: str, prorate: bool = True
    ) -> Dict[str, Any]:
        """
        Upgrade subscription to a higher plan

        Args:
            subscription_id: Current subscription ID
            new_plan_id: New plan ID
            prorate: Whether to prorate the change

        Returns:
            Upgrade result
        """
        try:
            if new_plan_id not in self.plans:
                return {"success": False, "error": "Invalid plan ID"}

            async with AsyncSessionLocal() as db:
                subscription = await db.get(Subscription, subscription_id)
                if not subscription:
                    return {"success": False, "error": "Subscription not found"}

                old_plan = self.plans[subscription.plan_id]
                new_plan = self.plans[new_plan_id]

                # Check if it's actually an upgrade
                if new_plan["price"] <= old_plan["price"]:
                    return {"success": False, "error": "New plan must be higher tier"}

                # Update Stripe subscription
                if subscription.stripe_subscription_id:
                    stripe.Subscription.modify(
                        subscription.stripe_subscription_id,
                        items=[
                            {
                                "id": subscription.stripe_subscription_id,
                                "price": settings.STRIPE_PRICE_IDS.get(new_plan_id),
                            }
                        ],
                        proration_behavior="create_prorations" if prorate else "none",
                    )

                # Update subscription
                subscription.plan_id = new_plan_id
                subscription.plan_name = new_plan["name"]
                subscription.monthly_price = new_plan["price"]
                subscription.video_limit = new_plan["videos_per_month"]
                subscription.channel_limit = new_plan["channels"]
                subscription.features = new_plan["features"]
                subscription.upgraded_at = datetime.utcnow()

                await db.commit()

                # Send notification
                await notification_service.send_notification(
                    user_id=subscription.user_id,
                    title="Subscription Upgraded",
                    message=f"Your plan has been upgraded to {new_plan['name']}",
                    type="success",
                )

                return {
                    "success": True,
                    "subscription_id": subscription_id,
                    "new_plan": new_plan["name"],
                    "upgraded_at": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            logger.error(f"Subscription upgrade failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def check_usage_limits(self, user_id: str) -> Dict[str, Any]:
        """
        Check user's usage against subscription limits

        Args:
            user_id: User ID

        Returns:
            Usage status and limits
        """
        async with AsyncSessionLocal() as db:
            # Get active subscription
            subscription = await db.execute(
                select(Subscription).where(
                    and_(
                        Subscription.user_id == user_id,
                        Subscription.status == SubscriptionStatus.ACTIVE,
                    )
                )
            )
            sub = subscription.scalar_one_or_none()

            if not sub:
                return {
                    "has_subscription": False,
                    "can_create_video": False,
                    "can_add_channel": False,
                }

            # Count videos in current period
            video_count = await db.execute(
                select(func.count()).where(
                    and_(
                        Cost.user_id == user_id,
                        Cost.operation == "video_generation",
                        Cost.created_at >= sub.current_period_start,
                    )
                )
            )
            videos_used = video_count.scalar() or 0

            # Count active channels
            from app.models.channel import Channel

            channel_count = await db.execute(
                select(func.count()).where(
                    and_(Channel.user_id == user_id, Channel.is_active == True)
                )
            )
            channels_used = channel_count.scalar() or 0

            return {
                "has_subscription": True,
                "plan": sub.plan_name,
                "videos_used": videos_used,
                "video_limit": sub.video_limit if sub.video_limit > 0 else "Unlimited",
                "can_create_video": sub.video_limit < 0
                or videos_used < sub.video_limit,
                "channels_used": channels_used,
                "channel_limit": sub.channel_limit
                if sub.channel_limit > 0
                else "Unlimited",
                "can_add_channel": sub.channel_limit < 0
                or channels_used < sub.channel_limit,
                "period_end": sub.current_period_end.isoformat(),
            }

    async def process_billing_cycle(self):
        """Process billing for all subscriptions (scheduled task)"""
        try:
            async with AsyncSessionLocal() as db:
                # Get subscriptions due for billing
                due_subscriptions = await db.execute(
                    select(Subscription).where(
                        and_(
                            Subscription.status == SubscriptionStatus.ACTIVE,
                            Subscription.next_billing_date <= datetime.utcnow(),
                        )
                    )
                )

                for subscription in due_subscriptions.scalars():
                    try:
                        # Generate invoice
                        invoice_result = await invoice_generator.generate_invoice(
                            user_id=subscription.user_id,
                            subscription_id=str(subscription.id),
                            billing_period_start=subscription.current_period_start,
                            billing_period_end=subscription.current_period_end,
                        )

                        if invoice_result["success"]:
                            # Process payment
                            payment_result = (
                                await self.payment_service.charge_subscription(
                                    subscription_id=str(subscription.id),
                                    amount=float(subscription.monthly_price),
                                )
                            )

                            if payment_result["success"]:
                                # Update subscription dates
                                subscription.current_period_start = (
                                    subscription.current_period_end
                                )
                                subscription.current_period_end = (
                                    subscription.current_period_end + timedelta(days=30)
                                )
                                subscription.next_billing_date = (
                                    subscription.current_period_end
                                )
                                subscription.last_payment_date = datetime.utcnow()

                                # Reset usage counters
                                subscription.current_period_video_count = 0
                            else:
                                # Payment failed
                                subscription.payment_failed_count += 1
                                if subscription.payment_failed_count >= 3:
                                    subscription.status = SubscriptionStatus.SUSPENDED

                                # Send payment failure notification
                                await notification_service.send_notification(
                                    user_id=subscription.user_id,
                                    title="Payment Failed",
                                    message="Your subscription payment failed. Please update your payment method.",
                                    type="error",
                                )

                        await db.commit()

                    except Exception as e:
                        logger.error(
                            f"Billing cycle processing failed for subscription {subscription.id}: {str(e)}"
                        )

        except Exception as e:
            logger.error(f"Billing cycle processing failed: {str(e)}")

    async def get_subscription_status(self, user_id: str) -> Dict[str, Any]:
        """
        Get detailed subscription status for user

        Args:
            user_id: User ID

        Returns:
            Subscription status and details
        """
        async with AsyncSessionLocal() as db:
            subscription = await db.execute(
                select(Subscription)
                .where(Subscription.user_id == user_id)
                .order_by(Subscription.created_at.desc())
            )
            sub = subscription.scalar_one_or_none()

            if not sub:
                return {
                    "has_subscription": False,
                    "available_plans": list(self.plans.keys()),
                }

            # Get usage stats
            usage = await self.check_usage_limits(user_id)

            # Get payment history
            payments = await db.execute(
                select(PaymentHistory)
                .where(PaymentHistory.user_id == user_id)
                .order_by(PaymentHistory.created_at.desc())
                .limit(5)
            )

            return {
                "has_subscription": True,
                "subscription_id": str(sub.id),
                "plan": sub.plan_name,
                "status": sub.status.value,
                "monthly_price": float(sub.monthly_price),
                "current_period": {
                    "start": sub.current_period_start.isoformat(),
                    "end": sub.current_period_end.isoformat(),
                },
                "next_billing_date": sub.next_billing_date.isoformat()
                if sub.next_billing_date
                else None,
                "usage": usage,
                "features": sub.features,
                "payment_history": [
                    {
                        "date": payment.created_at.isoformat(),
                        "amount": float(payment.amount),
                        "status": payment.status,
                    }
                    for payment in payments.scalars()
                ],
                "can_upgrade": sub.plan_id != "enterprise",
                "cancel_at_period_end": sub.cancel_at_period_end,
            }


# Singleton instance
subscription_service = SubscriptionService()
