"""
Enhanced Payment Service
Comprehensive Stripe integration with subscription management, usage tracking, and billing
"""
import stripe
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import os
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from app.models.payments import (
    PaymentCustomer,
    Subscription,
    Payment,
    Invoice,
    PaymentMethod,
    UsageRecord,
    BillingAlert,
)
from app.models.user import User
from app.services.webhook_service import webhook_service, WebhookEvents
from app.services.websocket_manager import ConnectionManager

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")


class PlanLimits:
    """Subscription plan limits"""

    BASIC = {
        "videos_monthly": 50,
        "storage_gb": 25,
        "api_calls_monthly": 5000,
        "channels": 3,
        "video_length_minutes": 10,
    }

    PRO = {
        "videos_monthly": 200,
        "storage_gb": 100,
        "api_calls_monthly": 25000,
        "channels": 10,
        "video_length_minutes": 30,
    }

    ENTERPRISE = {
        "videos_monthly": 1000,
        "storage_gb": 500,
        "api_calls_monthly": 100000,
        "channels": 50,
        "video_length_minutes": 60,
    }


class PaymentStatus:
    """Payment status constants"""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    REFUNDED = "refunded"
    DISPUTED = "disputed"


class SubscriptionStatus:
    """Subscription status constants"""

    ACTIVE = "active"
    CANCELED = "canceled"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    PAST_DUE = "past_due"
    TRIALING = "trialing"
    UNPAID = "unpaid"


class EnhancedPaymentService:
    """
    Comprehensive payment service with Stripe integration
    """

    def __init__(self, websocket_manager: Optional[ConnectionManager] = None):
        self.websocket_manager = websocket_manager
        self.webhook_secret = STRIPE_WEBHOOK_SECRET

    async def get_or_create_customer(
        self,
        db: AsyncSession,
        user_id: str,
        email: str,
        name: Optional[str] = None,
        phone: Optional[str] = None,
        address: Optional[Dict[str, str]] = None,
    ) -> PaymentCustomer:
        """
        Get existing customer or create new Stripe customer
        """
        try:
            # Check if customer exists
            result = await db.execute(
                select(PaymentCustomer).where(PaymentCustomer.user_id == user_id)
            )
            customer = result.scalar_one_or_none()

            if customer:
                return customer

            # Create Stripe customer
            stripe_customer = stripe.Customer.create(
                email=email,
                name=name,
                phone=phone,
                address=address,
                metadata={"user_id": user_id},
            )

            # Create database record
            customer = PaymentCustomer(
                user_id=user_id,
                stripe_customer_id=stripe_customer.id,
                email=email,
                name=name,
                phone=phone,
                address=address or {},
            )

            db.add(customer)
            await db.commit()
            await db.refresh(customer)

            logger.info(f"Created customer {stripe_customer.id} for user {user_id}")
            return customer

        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to create customer for user {user_id}: {e}")
            raise

    async def create_checkout_session(
        self,
        db: AsyncSession,
        user_id: str,
        price_id: str,
        success_url: str,
        cancel_url: str,
        trial_days: Optional[int] = None,
        coupon_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Create Stripe checkout session for subscription
        """
        try:
            # Get user details
            user_result = await db.execute(select(User).where(User.id == user_id))
            user = user_result.scalar_one_or_none()
            if not user:
                raise ValueError("User not found")

            # Get or create customer
            customer = await self.get_or_create_customer(
                db=db, user_id=user_id, email=user.email, name=user.full_name
            )

            # Prepare session parameters
            session_params = {
                "customer": customer.stripe_customer_id,
                "payment_method_types": ["card"],
                "line_items": [{"price": price_id, "quantity": 1}],
                "mode": "subscription",
                "success_url": success_url,
                "cancel_url": cancel_url,
                "metadata": {"user_id": user_id, **(metadata or {})},
                "subscription_data": {"metadata": {"user_id": user_id}},
            }

            # Add trial period if specified
            if trial_days and trial_days > 0:
                session_params["subscription_data"]["trial_period_days"] = trial_days

            # Add coupon if provided
            if coupon_id:
                session_params["discounts"] = [{"coupon": coupon_id}]

            # Create checkout session
            session = stripe.checkout.Session.create(**session_params)

            return {"checkout_url": session.url, "session_id": session.id}

        except Exception as e:
            logger.error(f"Failed to create checkout session: {e}")
            raise

    async def create_subscription(
        self,
        db: AsyncSession,
        user_id: str,
        price_id: str,
        payment_method_id: str,
        trial_days: Optional[int] = None,
    ) -> Subscription:
        """
        Create subscription directly with payment method
        """
        try:
            # Get customer
            result = await db.execute(
                select(PaymentCustomer).where(PaymentCustomer.user_id == user_id)
            )
            customer = result.scalar_one_or_none()
            if not customer:
                raise ValueError("Customer not found")

            # Attach payment method to customer
            stripe.PaymentMethod.attach(
                payment_method_id, customer=customer.stripe_customer_id
            )

            # Set as default payment method
            stripe.Customer.modify(
                customer.stripe_customer_id,
                invoice_settings={"default_payment_method": payment_method_id},
            )

            # Create subscription
            subscription_params = {
                "customer": customer.stripe_customer_id,
                "items": [{"price": price_id}],
                "default_payment_method": payment_method_id,
                "expand": ["latest_invoice.payment_intent"],
                "metadata": {"user_id": user_id},
            }

            if trial_days and trial_days > 0:
                subscription_params["trial_period_days"] = trial_days

            stripe_subscription = stripe.Subscription.create(**subscription_params)

            # Get price and product info
            price = stripe.Price.retrieve(price_id)
            product = stripe.Product.retrieve(price.product)

            # Create database record
            subscription = await self._create_subscription_record(
                db=db,
                user_id=user_id,
                customer_id=customer.id,
                stripe_subscription=stripe_subscription,
                price=price,
                product=product,
            )

            return subscription

        except Exception as e:
            logger.error(f"Failed to create subscription: {e}")
            raise

    async def cancel_subscription(
        self,
        db: AsyncSession,
        user_id: str,
        subscription_id: str,
        cancel_at_period_end: bool = True,
        reason: Optional[str] = None,
    ) -> Subscription:
        """
        Cancel user subscription
        """
        try:
            # Get subscription
            result = await db.execute(
                select(Subscription).where(
                    and_(
                        Subscription.user_id == user_id,
                        Subscription.stripe_subscription_id == subscription_id,
                    )
                )
            )
            subscription = result.scalar_one_or_none()
            if not subscription:
                raise ValueError("Subscription not found")

            # Cancel in Stripe
            if cancel_at_period_end:
                stripe_subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True,
                    metadata={"cancel_reason": reason or "user_requested"},
                )
            else:
                stripe_subscription = stripe.Subscription.delete(subscription_id)

            # Update database record
            subscription.cancel_at_period_end = cancel_at_period_end
            subscription.cancel_reason = reason

            if not cancel_at_period_end:
                subscription.status = SubscriptionStatus.CANCELED
                subscription.canceled_at = datetime.utcnow()

            await db.commit()

            # Send notification
            if self.websocket_manager:
                await self.websocket_manager.send_to_user(
                    user_id,
                    {
                        "type": "subscription_canceled",
                        "data": {
                            "subscription_id": subscription_id,
                            "cancel_at_period_end": cancel_at_period_end,
                        },
                    },
                )

            return subscription

        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to cancel subscription: {e}")
            raise

    async def track_usage(
        self,
        db: AsyncSession,
        user_id: str,
        usage_type: str,
        quantity: float,
        unit: str = "count",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Track usage for billing purposes
        """
        try:
            # Get active subscription
            subscription = await self.get_active_subscription(db, user_id)
            if not subscription:
                logger.warning(f"No active subscription for user {user_id}")
                return

            # Calculate billing period
            now = datetime.utcnow()
            period_start = subscription.current_period_start
            period_end = subscription.current_period_end

            # Get unit price (would be configured based on plan and usage type)
            unit_price = await self._get_usage_unit_price(
                subscription.plan_tier, usage_type
            )
            total_cost = Decimal(str(quantity)) * unit_price

            # Create usage record
            usage_record = UsageRecord(
                user_id=user_id,
                subscription_id=subscription.id,
                usage_type=usage_type,
                quantity=quantity,
                unit=unit,
                billing_period_start=period_start,
                billing_period_end=period_end,
                unit_price=unit_price,
                total_cost=total_cost,
                resource_type=resource_type,
                resource_id=resource_id,
                metadata=metadata or {},
            )

            db.add(usage_record)

            # Update subscription usage counters
            if usage_type == "videos":
                subscription.videos_generated_current_period += int(quantity)
            elif usage_type == "api_calls":
                subscription.api_calls_current_period += int(quantity)
            elif usage_type == "storage":
                subscription.storage_used_gb += quantity

            await db.commit()

            # Check limits and create alerts if needed
            await self._check_usage_limits(db, subscription)

        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to track usage: {e}")

    async def get_usage_summary(
        self,
        db: AsyncSession,
        user_id: str,
        billing_period_start: Optional[datetime] = None,
        billing_period_end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get usage summary for billing period
        """
        try:
            # Get subscription
            subscription = await self.get_active_subscription(db, user_id)
            if not subscription:
                return {}

            # Use current billing period if not specified
            if not billing_period_start:
                billing_period_start = subscription.current_period_start
            if not billing_period_end:
                billing_period_end = subscription.current_period_end

            # Get usage records
            result = await db.execute(
                select(
                    UsageRecord.usage_type,
                    func.sum(UsageRecord.quantity).label("total_quantity"),
                    func.sum(UsageRecord.total_cost).label("total_cost"),
                )
                .where(
                    and_(
                        UsageRecord.subscription_id == subscription.id,
                        UsageRecord.billing_period_start >= billing_period_start,
                        UsageRecord.billing_period_end <= billing_period_end,
                    )
                )
                .group_by(UsageRecord.usage_type)
            )

            usage_summary = {}
            total_cost = Decimal("0")

            for row in result:
                usage_summary[row.usage_type] = {
                    "quantity": float(row.total_quantity),
                    "cost": float(row.total_cost or 0),
                }
                total_cost += row.total_cost or 0

            # Get plan limits
            plan_limits = self._get_plan_limits(subscription.plan_tier)

            return {
                "billing_period": {
                    "start": billing_period_start.isoformat(),
                    "end": billing_period_end.isoformat(),
                },
                "subscription": {
                    "plan_name": subscription.plan_name,
                    "plan_tier": subscription.plan_tier,
                    "status": subscription.status,
                },
                "usage": usage_summary,
                "limits": plan_limits,
                "total_cost": float(total_cost),
                "currency": subscription.currency,
            }

        except Exception as e:
            logger.error(f"Failed to get usage summary: {e}")
            return {}

    async def process_webhook(
        self, db: AsyncSession, payload: bytes, signature: str
    ) -> Dict[str, str]:
        """
        Process Stripe webhook events
        """
        try:
            # Verify webhook signature
            event = stripe.Webhook.construct_event(
                payload, signature, self.webhook_secret
            )

            logger.info(f"Processing Stripe webhook: {event['type']}")

            # Handle different event types
            if event["type"] == "checkout.session.completed":
                await self._handle_checkout_completed(db, event["data"]["object"])
            elif event["type"] == "customer.subscription.created":
                await self._handle_subscription_created(db, event["data"]["object"])
            elif event["type"] == "customer.subscription.updated":
                await self._handle_subscription_updated(db, event["data"]["object"])
            elif event["type"] == "customer.subscription.deleted":
                await self._handle_subscription_deleted(db, event["data"]["object"])
            elif event["type"] == "invoice.payment_succeeded":
                await self._handle_payment_succeeded(db, event["data"]["object"])
            elif event["type"] == "invoice.payment_failed":
                await self._handle_payment_failed(db, event["data"]["object"])
            elif event["type"] == "customer.subscription.trial_will_end":
                await self._handle_trial_will_end(db, event["data"]["object"])
            else:
                logger.info(f"Unhandled webhook event: {event['type']}")

            return {"status": "success", "event_type": event["type"]}

        except ValueError as e:
            logger.error(f"Invalid webhook signature: {e}")
            raise
        except Exception as e:
            logger.error(f"Webhook processing failed: {e}")
            raise

    async def get_active_subscription(
        self, db: AsyncSession, user_id: str
    ) -> Optional[Subscription]:
        """
        Get user's active subscription
        """
        result = await db.execute(
            select(Subscription)
            .where(
                and_(
                    Subscription.user_id == user_id,
                    Subscription.status.in_(
                        [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
                    ),
                )
            )
            .order_by(desc(Subscription.created_at))
        )
        return result.scalar_one_or_none()

    async def get_billing_history(
        self, db: AsyncSession, user_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get user's billing history
        """
        result = await db.execute(
            select(Invoice)
            .where(Invoice.user_id == user_id)
            .order_by(desc(Invoice.created_at))
            .limit(limit)
        )
        invoices = result.scalars().all()

        billing_history = []
        for invoice in invoices:
            billing_history.append(
                {
                    "invoice_id": str(invoice.id),
                    "invoice_number": invoice.invoice_number,
                    "amount": invoice.total,
                    "currency": invoice.currency,
                    "status": invoice.status,
                    "period_start": invoice.period_start.isoformat(),
                    "period_end": invoice.period_end.isoformat(),
                    "paid_at": invoice.paid_at.isoformat() if invoice.paid_at else None,
                    "invoice_pdf_url": invoice.invoice_pdf_url,
                    "hosted_invoice_url": invoice.hosted_invoice_url,
                }
            )

        return billing_history

    # Private helper methods
    async def _create_subscription_record(
        self,
        db: AsyncSession,
        user_id: str,
        customer_id: uuid.UUID,
        stripe_subscription: stripe.Subscription,
        price: stripe.Price,
        product: stripe.Product,
    ) -> Subscription:
        """Create subscription database record"""
        plan_limits = self._get_plan_limits(product.metadata.get("tier", "basic"))

        subscription = Subscription(
            user_id=user_id,
            customer_id=customer_id,
            stripe_subscription_id=stripe_subscription.id,
            stripe_price_id=price.id,
            stripe_product_id=product.id,
            status=stripe_subscription.status,
            plan_name=product.name,
            plan_tier=product.metadata.get("tier", "basic"),
            billing_cycle="monthly"
            if price.recurring.interval == "month"
            else "yearly",
            unit_amount=price.unit_amount,
            currency=price.currency,
            current_period_start=datetime.fromtimestamp(
                stripe_subscription.current_period_start
            ),
            current_period_end=datetime.fromtimestamp(
                stripe_subscription.current_period_end
            ),
            trial_start=datetime.fromtimestamp(stripe_subscription.trial_start)
            if stripe_subscription.trial_start
            else None,
            trial_end=datetime.fromtimestamp(stripe_subscription.trial_end)
            if stripe_subscription.trial_end
            else None,
            video_limit_monthly=plan_limits["videos_monthly"],
            storage_limit_gb=plan_limits["storage_gb"],
            api_calls_limit_monthly=plan_limits["api_calls_monthly"],
        )

        db.add(subscription)
        await db.commit()
        await db.refresh(subscription)

        return subscription

    async def _get_usage_unit_price(self, plan_tier: str, usage_type: str) -> Decimal:
        """Get unit price for usage type and plan"""
        # This would be configured in database or config
        pricing = {
            "basic": {
                "videos": Decimal("0.10"),
                "api_calls": Decimal("0.001"),
                "storage": Decimal("0.05"),
            },
            "pro": {
                "videos": Decimal("0.08"),
                "api_calls": Decimal("0.0008"),
                "storage": Decimal("0.04"),
            },
            "enterprise": {
                "videos": Decimal("0.05"),
                "api_calls": Decimal("0.0005"),
                "storage": Decimal("0.02"),
            },
        }

        return pricing.get(plan_tier, pricing["basic"]).get(usage_type, Decimal("0"))

    def _get_plan_limits(self, plan_tier: str) -> Dict[str, Any]:
        """Get plan limits based on tier"""
        if plan_tier == "pro":
            return PlanLimits.PRO
        elif plan_tier == "enterprise":
            return PlanLimits.ENTERPRISE
        else:
            return PlanLimits.BASIC

    async def _check_usage_limits(self, db: AsyncSession, subscription: Subscription):
        """Check usage limits and create alerts"""
        # Check video limit
        if (
            subscription.videos_generated_current_period
            >= subscription.video_limit_monthly * 0.8
        ):
            await self._create_usage_alert(
                db=db,
                user_id=subscription.user_id,
                alert_type="video_limit_warning",
                threshold_value=subscription.video_limit_monthly * 0.8,
                current_value=subscription.videos_generated_current_period,
            )

        # Check API calls limit
        if (
            subscription.api_calls_current_period
            >= subscription.api_calls_limit_monthly * 0.8
        ):
            await self._create_usage_alert(
                db=db,
                user_id=subscription.user_id,
                alert_type="api_limit_warning",
                threshold_value=subscription.api_calls_limit_monthly * 0.8,
                current_value=subscription.api_calls_current_period,
            )

    async def _create_usage_alert(
        self,
        db: AsyncSession,
        user_id: str,
        alert_type: str,
        threshold_value: float,
        current_value: float,
    ):
        """Create usage alert"""
        alert = BillingAlert(
            user_id=user_id,
            alert_type=alert_type,
            severity="warning",
            title=f"Usage limit warning: {alert_type}",
            message=f"You've reached {(current_value/threshold_value)*100:.1f}% of your limit",
            threshold_value=threshold_value,
            current_value=current_value,
        )

        db.add(alert)
        await db.commit()

        # Send notification
        if self.websocket_manager:
            await self.websocket_manager.send_to_user(
                user_id,
                {
                    "type": "usage_alert",
                    "data": {"alert_type": alert_type, "message": alert.message},
                },
            )

    # Webhook handlers
    async def _handle_checkout_completed(
        self, db: AsyncSession, session: Dict[str, Any]
    ):
        """Handle checkout.session.completed webhook"""
        user_id = session["metadata"].get("user_id")
        if not user_id:
            return

        logger.info(f"Checkout completed for user {user_id}")

    async def _handle_subscription_created(
        self, db: AsyncSession, subscription: Dict[str, Any]
    ):
        """Handle customer.subscription.created webhook"""
        user_id = subscription["metadata"].get("user_id")
        if not user_id:
            return

        # Trigger webhook event
        await webhook_service.trigger_event(
            db=db,
            event_type=WebhookEvents.SUBSCRIPTION_UPDATED,
            source_type="subscription",
            source_id=subscription["id"],
            user_id=user_id,
            data={
                "subscription_id": subscription["id"],
                "status": subscription["status"],
                "action": "created",
            },
        )

    async def _handle_subscription_updated(
        self, db: AsyncSession, subscription: Dict[str, Any]
    ):
        """Handle customer.subscription.updated webhook"""
        user_id = subscription["metadata"].get("user_id")
        if not user_id:
            return

        # Update database record
        result = await db.execute(
            select(Subscription).where(
                Subscription.stripe_subscription_id == subscription["id"]
            )
        )
        db_subscription = result.scalar_one_or_none()

        if db_subscription:
            db_subscription.status = subscription["status"]
            db_subscription.current_period_start = datetime.fromtimestamp(
                subscription["current_period_start"]
            )
            db_subscription.current_period_end = datetime.fromtimestamp(
                subscription["current_period_end"]
            )
            await db.commit()

    async def _handle_subscription_deleted(
        self, db: AsyncSession, subscription: Dict[str, Any]
    ):
        """Handle customer.subscription.deleted webhook"""
        user_id = subscription["metadata"].get("user_id")
        if not user_id:
            return

        # Update database record
        await db.execute(
            update(Subscription)
            .where(Subscription.stripe_subscription_id == subscription["id"])
            .values(status=SubscriptionStatus.CANCELED, canceled_at=datetime.utcnow())
        )
        await db.commit()

    async def _handle_payment_succeeded(
        self, db: AsyncSession, invoice: Dict[str, Any]
    ):
        """Handle invoice.payment_succeeded webhook"""
        # Create payment record and update invoice
        pass

    async def _handle_payment_failed(self, db: AsyncSession, invoice: Dict[str, Any]):
        """Handle invoice.payment_failed webhook"""
        # Create failed payment record and send alerts
        pass

    async def _handle_trial_will_end(
        self, db: AsyncSession, subscription: Dict[str, Any]
    ):
        """Handle customer.subscription.trial_will_end webhook"""
        user_id = subscription["metadata"].get("user_id")
        if not user_id and self.websocket_manager:
            await self.websocket_manager.send_to_user(
                user_id,
                {
                    "type": "trial_ending",
                    "data": {
                        "subscription_id": subscription["id"],
                        "trial_end": subscription["trial_end"],
                    },
                },
            )

    # Week 2 Enhancement: Subscription Upgrade/Downgrade

    async def upgrade_downgrade_subscription(
        self,
        db: AsyncSession,
        user_id: str,
        new_price_id: str,
        proration_behavior: str = "create_prorations",
    ) -> Dict[str, Any]:
        """
        Upgrade or downgrade subscription with proration

        Args:
            user_id: User ID
            new_price_id: New Stripe price ID
            proration_behavior: How to handle proration (create_prorations, none, always_invoice)
        """
        try:
            # Get current subscription
            result = await db.execute(
                select(Subscription).where(
                    and_(
                        Subscription.user_id == user_id,
                        Subscription.status == SubscriptionStatus.ACTIVE,
                    )
                )
            )
            subscription = result.scalar_one_or_none()

            if not subscription:
                raise ValueError("No active subscription found")

            # Update Stripe subscription
            stripe_subscription = stripe.Subscription.modify(
                subscription.stripe_subscription_id,
                items=[
                    {
                        "id": subscription.stripe_subscription_item_id,
                        "price": new_price_id,
                    }
                ],
                proration_behavior=proration_behavior,
                metadata={
                    "user_id": user_id,
                    "change_type": "plan_change",
                    "previous_price": subscription.stripe_price_id,
                },
            )

            # Update database
            subscription.stripe_price_id = new_price_id
            subscription.plan_name = stripe_subscription["items"]["data"][0]["price"][
                "nickname"
            ]
            subscription.plan_amount = (
                stripe_subscription["items"]["data"][0]["price"]["unit_amount"] / 100
            )
            subscription.updated_at = datetime.utcnow()

            await db.commit()

            # Send notification
            if self.websocket_manager:
                await self.websocket_manager.send_to_user(
                    user_id,
                    {
                        "type": "subscription_updated",
                        "data": {
                            "subscription_id": subscription.id,
                            "new_plan": subscription.plan_name,
                            "status": "success",
                        },
                    },
                )

            return {
                "subscription_id": subscription.id,
                "status": stripe_subscription["status"],
                "new_plan": subscription.plan_name,
                "proration_amount": self._calculate_proration(stripe_subscription),
            }

        except Exception as e:
            logger.error(f"Subscription upgrade/downgrade failed: {str(e)}")
            raise

    async def add_payment_method(
        self,
        db: AsyncSession,
        user_id: str,
        payment_method_id: str,
        set_as_default: bool = True,
    ) -> PaymentMethod:
        """Add new payment method for user"""
        try:
            # Get customer
            result = await db.execute(
                select(PaymentCustomer).where(PaymentCustomer.user_id == user_id)
            )
            customer = result.scalar_one_or_none()

            if not customer:
                raise ValueError("Customer not found")

            # Attach payment method to customer
            payment_method = stripe.PaymentMethod.attach(
                payment_method_id, customer=customer.stripe_customer_id
            )

            # Set as default if requested
            if set_as_default:
                stripe.Customer.modify(
                    customer.stripe_customer_id,
                    invoice_settings={"default_payment_method": payment_method_id},
                )

            # Save to database
            db_payment_method = PaymentMethod(
                user_id=user_id,
                stripe_payment_method_id=payment_method_id,
                type=payment_method["type"],
                last4=payment_method["card"]["last4"]
                if payment_method["type"] == "card"
                else None,
                brand=payment_method["card"]["brand"]
                if payment_method["type"] == "card"
                else None,
                exp_month=payment_method["card"]["exp_month"]
                if payment_method["type"] == "card"
                else None,
                exp_year=payment_method["card"]["exp_year"]
                if payment_method["type"] == "card"
                else None,
                is_default=set_as_default,
            )

            # Update other payment methods if setting as default
            if set_as_default:
                await db.execute(
                    update(PaymentMethod)
                    .where(PaymentMethod.user_id == user_id)
                    .values(is_default=False)
                )

            db.add(db_payment_method)
            await db.commit()

            return db_payment_method

        except Exception as e:
            logger.error(f"Add payment method failed: {str(e)}")
            raise

    async def remove_payment_method(
        self, db: AsyncSession, user_id: str, payment_method_id: str
    ) -> Dict[str, str]:
        """Remove payment method"""
        try:
            # Detach from Stripe
            stripe.PaymentMethod.detach(payment_method_id)

            # Remove from database
            await db.execute(
                delete(PaymentMethod).where(
                    and_(
                        PaymentMethod.user_id == user_id,
                        PaymentMethod.stripe_payment_method_id == payment_method_id,
                    )
                )
            )
            await db.commit()

            return {"status": "success", "message": "Payment method removed"}

        except Exception as e:
            logger.error(f"Remove payment method failed: {str(e)}")
            raise

    async def generate_invoice(
        self,
        db: AsyncSession,
        user_id: str,
        items: List[Dict[str, Any]],
        send_invoice: bool = True,
    ) -> Invoice:
        """Generate custom invoice for additional charges"""
        try:
            # Get customer
            result = await db.execute(
                select(PaymentCustomer).where(PaymentCustomer.user_id == user_id)
            )
            customer = result.scalar_one_or_none()

            if not customer:
                raise ValueError("Customer not found")

            # Create invoice items
            total_amount = 0
            for item in items:
                stripe.InvoiceItem.create(
                    customer=customer.stripe_customer_id,
                    amount=int(item["amount"] * 100),  # Convert to cents
                    currency=item.get("currency", "usd"),
                    description=item["description"],
                )
                total_amount += item["amount"]

            # Create invoice
            stripe_invoice = stripe.Invoice.create(
                customer=customer.stripe_customer_id,
                auto_advance=send_invoice,  # Auto-finalize and send
                metadata={"user_id": user_id, "type": "custom_invoice"},
            )

            # Save to database
            db_invoice = Invoice(
                user_id=user_id,
                stripe_invoice_id=stripe_invoice["id"],
                amount_paid=0,
                amount_due=total_amount,
                amount_remaining=total_amount,
                currency=stripe_invoice["currency"],
                status="draft" if not send_invoice else "open",
                invoice_pdf=stripe_invoice.get("invoice_pdf"),
                hosted_invoice_url=stripe_invoice.get("hosted_invoice_url"),
                period_start=datetime.fromtimestamp(stripe_invoice["period_start"]),
                period_end=datetime.fromtimestamp(stripe_invoice["period_end"]),
            )

            db.add(db_invoice)
            await db.commit()

            # Send invoice if requested
            if send_invoice:
                stripe.Invoice.send_invoice(stripe_invoice["id"])

            return db_invoice

        except Exception as e:
            logger.error(f"Invoice generation failed: {str(e)}")
            raise

    async def get_invoice_history(
        self,
        db: AsyncSession,
        user_id: str,
        limit: int = 10,
        starting_after: Optional[str] = None,
    ) -> List[Invoice]:
        """Get invoice history with pagination"""
        query = (
            select(Invoice)
            .where(Invoice.user_id == user_id)
            .order_by(desc(Invoice.created_at))
            .limit(limit)
        )

        if starting_after:
            query = query.where(Invoice.id > starting_after)

        result = await db.execute(query)
        return result.scalars().all()

    async def process_usage_overage(
        self, db: AsyncSession, user_id: str, usage_type: str, overage_amount: float
    ) -> Dict[str, Any]:
        """Process usage-based billing for overages"""
        try:
            # Get active subscription
            subscription = await self.get_active_subscription(db, user_id)

            if not subscription:
                raise ValueError("No active subscription")

            # Get overage pricing
            overage_price = await self._get_usage_unit_price(
                subscription.plan_tier, usage_type
            )

            # Calculate cost
            overage_cost = overage_amount * float(overage_price)

            # Create usage record
            stripe_usage = stripe.SubscriptionItem.create_usage_record(
                subscription.stripe_subscription_item_id,
                quantity=int(overage_amount),
                timestamp=int(datetime.utcnow().timestamp()),
                action="increment",
                metadata={
                    "user_id": user_id,
                    "usage_type": usage_type,
                    "overage": "true",
                },
            )

            # Save to database
            usage_record = UsageRecord(
                subscription_id=subscription.id,
                usage_type=usage_type,
                quantity=overage_amount,
                unit_price=overage_price,
                total_cost=Decimal(str(overage_cost)),
                metadata={"overage": True},
            )

            db.add(usage_record)
            await db.commit()

            # Send alert
            if overage_cost > 10:  # Alert for significant overages
                await self._create_usage_alert(
                    db,
                    subscription.id,
                    "overage_charge",
                    f"Usage overage charge of ${overage_cost:.2f} for {usage_type}",
                )

            return {
                "status": "success",
                "usage_type": usage_type,
                "overage_amount": overage_amount,
                "cost": overage_cost,
                "billed": True,
            }

        except Exception as e:
            logger.error(f"Process overage failed: {str(e)}")
            raise

    def _calculate_proration(self, subscription: Dict[str, Any]) -> float:
        """Calculate proration amount from subscription update"""
        # This would parse the upcoming invoice to get proration amount
        try:
            upcoming = stripe.Invoice.upcoming(
                customer=subscription["customer"], subscription=subscription["id"]
            )

            proration_amount = 0
            for line in upcoming["lines"]["data"]:
                if line.get("proration"):
                    proration_amount += line["amount"]

            return proration_amount / 100  # Convert from cents
        except:
            return 0.0

    async def update_billing_address(
        self, db: AsyncSession, user_id: str, address: Dict[str, str]
    ) -> Dict[str, str]:
        """Update customer billing address"""
        try:
            # Get customer
            result = await db.execute(
                select(PaymentCustomer).where(PaymentCustomer.user_id == user_id)
            )
            customer = result.scalar_one_or_none()

            if not customer:
                raise ValueError("Customer not found")

            # Update in Stripe
            stripe.Customer.modify(customer.stripe_customer_id, address=address)

            # Update in database
            customer.address = address
            customer.updated_at = datetime.utcnow()
            await db.commit()

            return {"status": "success", "message": "Billing address updated"}

        except Exception as e:
            logger.error(f"Update billing address failed: {str(e)}")
            raise


# Global instance
payment_service = EnhancedPaymentService()
