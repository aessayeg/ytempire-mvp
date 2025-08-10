"""
Payment Gateway Service - Stripe Integration
Handles payments, subscriptions, and billing
"""
import os
import stripe
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
from decimal import Decimal

logger = logging.getLogger(__name__)

# Stripe configuration
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "sk_test_placeholder")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "whsec_placeholder")

class PlanTier(Enum):
    """Subscription plan tiers"""
    FREE = "free"
    STARTER = "starter"      # $19/month
    GROWTH = "growth"        # $49/month  
    PROFESSIONAL = "pro"     # $99/month
    ENTERPRISE = "enterprise" # Custom pricing

class PaymentStatus(Enum):
    """Payment status types"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"

@dataclass
class PlanDetails:
    """Subscription plan details"""
    tier: PlanTier
    name: str
    price_monthly: Decimal
    price_yearly: Decimal
    stripe_price_id_monthly: str
    stripe_price_id_yearly: str
    features: Dict[str, Any]
    limits: Dict[str, int]

# Plan configurations
PLANS = {
    PlanTier.FREE: PlanDetails(
        tier=PlanTier.FREE,
        name="Free Trial",
        price_monthly=Decimal("0.00"),
        price_yearly=Decimal("0.00"),
        stripe_price_id_monthly="",
        stripe_price_id_yearly="",
        features={
            "videos_per_month": 3,
            "channels": 1,
            "analytics": "basic",
            "support": "community",
            "api_access": False,
            "custom_voices": False,
            "priority_processing": False
        },
        limits={
            "max_video_duration": 300,  # 5 minutes
            "max_storage_gb": 1,
            "max_api_calls": 0
        }
    ),
    PlanTier.STARTER: PlanDetails(
        tier=PlanTier.STARTER,
        name="Starter",
        price_monthly=Decimal("19.00"),
        price_yearly=Decimal("190.00"),  # ~17% discount
        stripe_price_id_monthly="price_starter_monthly",
        stripe_price_id_yearly="price_starter_yearly",
        features={
            "videos_per_month": 30,
            "channels": 3,
            "analytics": "standard",
            "support": "email",
            "api_access": False,
            "custom_voices": True,
            "priority_processing": False
        },
        limits={
            "max_video_duration": 900,  # 15 minutes
            "max_storage_gb": 10,
            "max_api_calls": 100
        }
    ),
    PlanTier.GROWTH: PlanDetails(
        tier=PlanTier.GROWTH,
        name="Growth",
        price_monthly=Decimal("49.00"),
        price_yearly=Decimal("470.00"),  # ~20% discount
        stripe_price_id_monthly="price_growth_monthly",
        stripe_price_id_yearly="price_growth_yearly",
        features={
            "videos_per_month": 100,
            "channels": 10,
            "analytics": "advanced",
            "support": "priority",
            "api_access": True,
            "custom_voices": True,
            "priority_processing": True
        },
        limits={
            "max_video_duration": 1800,  # 30 minutes
            "max_storage_gb": 50,
            "max_api_calls": 1000
        }
    ),
    PlanTier.PROFESSIONAL: PlanDetails(
        tier=PlanTier.PROFESSIONAL,
        name="Professional",
        price_monthly=Decimal("99.00"),
        price_yearly=Decimal("950.00"),  # ~20% discount
        stripe_price_id_monthly="price_pro_monthly",
        stripe_price_id_yearly="price_pro_yearly",
        features={
            "videos_per_month": 500,
            "channels": 50,
            "analytics": "advanced",
            "support": "dedicated",
            "api_access": True,
            "custom_voices": True,
            "priority_processing": True,
            "white_label": True,
            "custom_branding": True
        },
        limits={
            "max_video_duration": 3600,  # 60 minutes
            "max_storage_gb": 200,
            "max_api_calls": 10000
        }
    )
}

class PaymentGateway:
    """Main payment gateway for Stripe operations"""
    
    def __init__(self):
        self.stripe = stripe
        
    async def create_customer(
        self,
        user_id: str,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Create a Stripe customer"""
        try:
            customer = self.stripe.Customer.create(
                email=email,
                name=name,
                metadata={
                    "user_id": user_id,
                    **(metadata or {})
                }
            )
            logger.info(f"Created Stripe customer {customer.id} for user {user_id}")
            return customer.id
            
        except stripe.error.StripeError as e:
            logger.error(f"Error creating customer: {e}")
            raise
            
    async def create_subscription(
        self,
        customer_id: str,
        plan_tier: PlanTier,
        billing_cycle: str = "monthly"
    ) -> Dict[str, Any]:
        """Create a subscription for a customer"""
        try:
            plan = PLANS[plan_tier]
            
            # Get the appropriate price ID
            if billing_cycle == "monthly":
                price_id = plan.stripe_price_id_monthly
            else:
                price_id = plan.stripe_price_id_yearly
                
            # Create subscription
            subscription = self.stripe.Subscription.create(
                customer=customer_id,
                items=[{"price": price_id}],
                payment_behavior="default_incomplete",
                payment_settings={"save_default_payment_method": "on_subscription"},
                expand=["latest_invoice.payment_intent"],
                metadata={
                    "plan_tier": plan_tier.value,
                    "billing_cycle": billing_cycle
                }
            )
            
            logger.info(f"Created subscription {subscription.id} for customer {customer_id}")
            
            return {
                "subscription_id": subscription.id,
                "status": subscription.status,
                "client_secret": subscription.latest_invoice.payment_intent.client_secret,
                "current_period_end": subscription.current_period_end,
                "plan_tier": plan_tier.value,
                "billing_cycle": billing_cycle
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Error creating subscription: {e}")
            raise
            
    async def update_subscription(
        self,
        subscription_id: str,
        new_plan_tier: PlanTier,
        billing_cycle: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update an existing subscription"""
        try:
            subscription = self.stripe.Subscription.retrieve(subscription_id)
            plan = PLANS[new_plan_tier]
            
            # Determine price ID
            if billing_cycle:
                price_id = plan.stripe_price_id_monthly if billing_cycle == "monthly" else plan.stripe_price_id_yearly
            else:
                # Keep current billing cycle
                current_billing = subscription.items.data[0].price.recurring.interval
                price_id = plan.stripe_price_id_monthly if current_billing == "month" else plan.stripe_price_id_yearly
                
            # Update subscription
            updated_subscription = self.stripe.Subscription.modify(
                subscription_id,
                items=[{
                    "id": subscription.items.data[0].id,
                    "price": price_id
                }],
                proration_behavior="always_invoice",
                metadata={
                    "plan_tier": new_plan_tier.value,
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Updated subscription {subscription_id} to {new_plan_tier.value}")
            
            return {
                "subscription_id": updated_subscription.id,
                "status": updated_subscription.status,
                "plan_tier": new_plan_tier.value,
                "current_period_end": updated_subscription.current_period_end
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Error updating subscription: {e}")
            raise
            
    async def cancel_subscription(
        self,
        subscription_id: str,
        immediate: bool = False
    ) -> Dict[str, Any]:
        """Cancel a subscription"""
        try:
            if immediate:
                # Cancel immediately
                subscription = self.stripe.Subscription.delete(subscription_id)
            else:
                # Cancel at end of billing period
                subscription = self.stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True
                )
                
            logger.info(f"Cancelled subscription {subscription_id} (immediate: {immediate})")
            
            return {
                "subscription_id": subscription.id,
                "status": subscription.status,
                "cancel_at": subscription.cancel_at,
                "canceled_at": subscription.canceled_at
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Error cancelling subscription: {e}")
            raise
            
    async def create_payment_intent(
        self,
        amount: Decimal,
        currency: str = "usd",
        customer_id: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a one-time payment intent"""
        try:
            intent = self.stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Convert to cents
                currency=currency,
                customer=customer_id,
                description=description,
                metadata=metadata or {},
                automatic_payment_methods={"enabled": True}
            )
            
            logger.info(f"Created payment intent {intent.id} for ${amount}")
            
            return {
                "payment_intent_id": intent.id,
                "client_secret": intent.client_secret,
                "amount": amount,
                "currency": currency,
                "status": intent.status
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Error creating payment intent: {e}")
            raise
            
    async def process_webhook(
        self,
        payload: bytes,
        signature: str
    ) -> Dict[str, Any]:
        """Process Stripe webhook events"""
        try:
            # Verify webhook signature
            event = self.stripe.Webhook.construct_event(
                payload,
                signature,
                STRIPE_WEBHOOK_SECRET
            )
            
            logger.info(f"Processing webhook event: {event.type}")
            
            # Handle different event types
            if event.type == "payment_intent.succeeded":
                return await self._handle_payment_success(event.data.object)
                
            elif event.type == "payment_intent.payment_failed":
                return await self._handle_payment_failed(event.data.object)
                
            elif event.type == "customer.subscription.created":
                return await self._handle_subscription_created(event.data.object)
                
            elif event.type == "customer.subscription.updated":
                return await self._handle_subscription_updated(event.data.object)
                
            elif event.type == "customer.subscription.deleted":
                return await self._handle_subscription_deleted(event.data.object)
                
            else:
                logger.info(f"Unhandled webhook event type: {event.type}")
                return {"status": "unhandled", "event_type": event.type}
                
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Invalid webhook signature: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
            raise
            
    async def _handle_payment_success(self, payment_intent):
        """Handle successful payment"""
        logger.info(f"Payment succeeded: {payment_intent.id}")
        
        return {
            "status": "success",
            "payment_intent_id": payment_intent.id,
            "amount": payment_intent.amount / 100,
            "customer_id": payment_intent.customer
        }
        
    async def _handle_payment_failed(self, payment_intent):
        """Handle failed payment"""
        logger.warning(f"Payment failed: {payment_intent.id}")
        
        return {
            "status": "failed",
            "payment_intent_id": payment_intent.id,
            "error": payment_intent.last_payment_error.message if payment_intent.last_payment_error else None
        }
        
    async def _handle_subscription_created(self, subscription):
        """Handle new subscription"""
        logger.info(f"Subscription created: {subscription.id}")
        
        return {
            "status": "created",
            "subscription_id": subscription.id,
            "customer_id": subscription.customer,
            "plan_tier": subscription.metadata.get("plan_tier")
        }
        
    async def _handle_subscription_updated(self, subscription):
        """Handle subscription update"""
        logger.info(f"Subscription updated: {subscription.id}")
        
        return {
            "status": "updated",
            "subscription_id": subscription.id,
            "customer_id": subscription.customer,
            "plan_tier": subscription.metadata.get("plan_tier")
        }
        
    async def _handle_subscription_deleted(self, subscription):
        """Handle subscription cancellation"""
        logger.info(f"Subscription deleted: {subscription.id}")
        
        return {
            "status": "deleted",
            "subscription_id": subscription.id,
            "customer_id": subscription.customer
        }
        
    async def get_customer_billing(
        self,
        customer_id: str
    ) -> Dict[str, Any]:
        """Get customer billing information"""
        try:
            # Get customer
            customer = self.stripe.Customer.retrieve(customer_id)
            
            # Get active subscriptions
            subscriptions = self.stripe.Subscription.list(
                customer=customer_id,
                status="active"
            )
            
            # Get recent invoices
            invoices = self.stripe.Invoice.list(
                customer=customer_id,
                limit=10
            )
            
            return {
                "customer": {
                    "id": customer.id,
                    "email": customer.email
                },
                "subscriptions": [
                    {
                        "id": sub.id,
                        "status": sub.status,
                        "current_period_end": sub.current_period_end
                    }
                    for sub in subscriptions.data
                ],
                "invoices": [
                    {
                        "id": inv.id,
                        "amount": inv.amount_paid / 100,
                        "status": inv.status,
                        "created": inv.created
                    }
                    for inv in invoices.data
                ]
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Error getting customer billing: {e}")
            raise