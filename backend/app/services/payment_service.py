"""
Payment Service for YTEmpire
Handles Stripe integration for subscriptions and payments
"""
import stripe
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
from decimal import Decimal
import json

from app.core.config import settings
from app.models.user import User
from app.models.subscription import Subscription
from app.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY


class PaymentService:
    """Handles all payment-related operations"""
    
    # Subscription tiers and pricing
    SUBSCRIPTION_TIERS = {
        "free": {
            "price": 0,
            "stripe_price_id": None,
            "channels_limit": 1,
            "videos_per_day": 3,
            "features": ["Basic analytics", "Standard support"]
        },
        "starter": {
            "price": 29.99,
            "stripe_price_id": "price_starter_monthly",
            "channels_limit": 3,
            "videos_per_day": 10,
            "features": ["Advanced analytics", "Priority support", "Custom thumbnails"]
        },
        "pro": {
            "price": 99.99,
            "stripe_price_id": "price_pro_monthly",
            "channels_limit": 10,
            "videos_per_day": 50,
            "features": ["All starter features", "API access", "Bulk operations", "Advanced AI models"]
        },
        "enterprise": {
            "price": 499.99,
            "stripe_price_id": "price_enterprise_monthly",
            "channels_limit": 999,
            "videos_per_day": 999,
            "features": ["All pro features", "Dedicated support", "Custom integrations", "SLA"]
        }
    }
    
    async def create_customer(self, user: User) -> str:
        """Create a Stripe customer for a user"""
        try:
            customer = stripe.Customer.create(
                email=user.email,
                name=user.full_name or user.username,
                metadata={
                    "user_id": str(user.id),
                    "username": user.username
                }
            )
            
            # Update user with Stripe customer ID
            async with AsyncSessionLocal() as db:
                user.stripe_customer_id = customer.id
                db.add(user)
                await db.commit()
            
            logger.info(f"Created Stripe customer {customer.id} for user {user.id}")
            return customer.id
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating customer: {e}")
            raise
    
    async def create_subscription(
        self,
        user: User,
        tier: str,
        payment_method_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create or update a subscription for a user"""
        
        if tier not in self.SUBSCRIPTION_TIERS:
            raise ValueError(f"Invalid subscription tier: {tier}")
        
        tier_config = self.SUBSCRIPTION_TIERS[tier]
        
        # Free tier doesn't need Stripe subscription
        if tier == "free":
            return await self._handle_free_tier(user)
        
        # Ensure user has a Stripe customer ID
        if not user.stripe_customer_id:
            await self.create_customer(user)
        
        try:
            # Attach payment method if provided
            if payment_method_id:
                stripe.PaymentMethod.attach(
                    payment_method_id,
                    customer=user.stripe_customer_id
                )
                
                # Set as default payment method
                stripe.Customer.modify(
                    user.stripe_customer_id,
                    invoice_settings={
                        "default_payment_method": payment_method_id
                    }
                )
            
            # Check for existing subscription
            existing_subscriptions = stripe.Subscription.list(
                customer=user.stripe_customer_id,
                status="active",
                limit=1
            )
            
            if existing_subscriptions.data:
                # Update existing subscription
                subscription = stripe.Subscription.modify(
                    existing_subscriptions.data[0].id,
                    items=[{
                        "id": existing_subscriptions.data[0]["items"]["data"][0].id,
                        "price": tier_config["stripe_price_id"]
                    }],
                    proration_behavior="create_prorations"
                )
            else:
                # Create new subscription
                subscription = stripe.Subscription.create(
                    customer=user.stripe_customer_id,
                    items=[{"price": tier_config["stripe_price_id"]}],
                    payment_behavior="default_incomplete",
                    payment_settings={
                        "save_default_payment_method": "on_subscription"
                    },
                    expand=["latest_invoice.payment_intent"]
                )
            
            # Update user subscription in database
            await self._update_user_subscription(user, tier, subscription.id)
            
            return {
                "subscription_id": subscription.id,
                "status": subscription.status,
                "tier": tier,
                "current_period_end": datetime.fromtimestamp(subscription.current_period_end),
                "client_secret": subscription.latest_invoice.payment_intent.client_secret if subscription.latest_invoice else None
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating subscription: {e}")
            raise
    
    async def cancel_subscription(self, user: User) -> bool:
        """Cancel a user's subscription"""
        if not user.stripe_subscription_id:
            return False
        
        try:
            # Cancel at period end to allow user to use service until end of billing period
            subscription = stripe.Subscription.modify(
                user.stripe_subscription_id,
                cancel_at_period_end=True
            )
            
            logger.info(f"Cancelled subscription {subscription.id} for user {user.id}")
            
            # Update database
            async with AsyncSessionLocal() as db:
                user_sub = await db.get(Subscription, user.id)
                if user_sub:
                    user_sub.cancel_at_period_end = True
                    await db.commit()
            
            return True
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error cancelling subscription: {e}")
            return False
    
    async def resume_subscription(self, user: User) -> bool:
        """Resume a cancelled subscription"""
        if not user.stripe_subscription_id:
            return False
        
        try:
            subscription = stripe.Subscription.modify(
                user.stripe_subscription_id,
                cancel_at_period_end=False
            )
            
            logger.info(f"Resumed subscription {subscription.id} for user {user.id}")
            
            # Update database
            async with AsyncSessionLocal() as db:
                user_sub = await db.get(Subscription, user.id)
                if user_sub:
                    user_sub.cancel_at_period_end = False
                    await db.commit()
            
            return True
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error resuming subscription: {e}")
            return False
    
    async def get_subscription_status(self, user: User) -> Dict[str, Any]:
        """Get current subscription status for a user"""
        if not user.stripe_subscription_id:
            return {
                "tier": "free",
                "status": "active",
                "channels_limit": self.SUBSCRIPTION_TIERS["free"]["channels_limit"],
                "videos_per_day": self.SUBSCRIPTION_TIERS["free"]["videos_per_day"]
            }
        
        try:
            subscription = stripe.Subscription.retrieve(user.stripe_subscription_id)
            
            # Determine tier from price ID
            tier = "free"
            for tier_name, config in self.SUBSCRIPTION_TIERS.items():
                if config["stripe_price_id"] == subscription["items"]["data"][0].price.id:
                    tier = tier_name
                    break
            
            return {
                "tier": tier,
                "status": subscription.status,
                "current_period_end": datetime.fromtimestamp(subscription.current_period_end),
                "cancel_at_period_end": subscription.cancel_at_period_end,
                "channels_limit": self.SUBSCRIPTION_TIERS[tier]["channels_limit"],
                "videos_per_day": self.SUBSCRIPTION_TIERS[tier]["videos_per_day"],
                "features": self.SUBSCRIPTION_TIERS[tier]["features"]
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error getting subscription: {e}")
            return {
                "tier": "free",
                "status": "error",
                "error": str(e)
            }
    
    async def get_payment_methods(self, user: User) -> List[Dict[str, Any]]:
        """Get saved payment methods for a user"""
        if not user.stripe_customer_id:
            return []
        
        try:
            payment_methods = stripe.PaymentMethod.list(
                customer=user.stripe_customer_id,
                type="card"
            )
            
            return [
                {
                    "id": pm.id,
                    "brand": pm.card.brand,
                    "last4": pm.card.last4,
                    "exp_month": pm.card.exp_month,
                    "exp_year": pm.card.exp_year,
                    "is_default": pm.id == user.default_payment_method_id
                }
                for pm in payment_methods.data
            ]
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error getting payment methods: {e}")
            return []
    
    async def add_payment_method(self, user: User, payment_method_id: str) -> bool:
        """Add a payment method to a user's account"""
        if not user.stripe_customer_id:
            await self.create_customer(user)
        
        try:
            stripe.PaymentMethod.attach(
                payment_method_id,
                customer=user.stripe_customer_id
            )
            
            logger.info(f"Added payment method {payment_method_id} for user {user.id}")
            return True
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error adding payment method: {e}")
            return False
    
    async def remove_payment_method(self, user: User, payment_method_id: str) -> bool:
        """Remove a payment method from a user's account"""
        try:
            stripe.PaymentMethod.detach(payment_method_id)
            
            logger.info(f"Removed payment method {payment_method_id} for user {user.id}")
            return True
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error removing payment method: {e}")
            return False
    
    async def get_invoices(self, user: User, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent invoices for a user"""
        if not user.stripe_customer_id:
            return []
        
        try:
            invoices = stripe.Invoice.list(
                customer=user.stripe_customer_id,
                limit=limit
            )
            
            return [
                {
                    "id": invoice.id,
                    "number": invoice.number,
                    "amount": invoice.amount_paid / 100,  # Convert from cents
                    "currency": invoice.currency,
                    "status": invoice.status,
                    "created": datetime.fromtimestamp(invoice.created),
                    "pdf_url": invoice.invoice_pdf,
                    "hosted_url": invoice.hosted_invoice_url
                }
                for invoice in invoices.data
            ]
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error getting invoices: {e}")
            return []
    
    async def create_checkout_session(
        self,
        user: User,
        tier: str,
        success_url: str,
        cancel_url: str
    ) -> str:
        """Create a Stripe Checkout session for subscription"""
        if tier not in self.SUBSCRIPTION_TIERS or tier == "free":
            raise ValueError(f"Invalid subscription tier for checkout: {tier}")
        
        tier_config = self.SUBSCRIPTION_TIERS[tier]
        
        # Ensure user has a Stripe customer ID
        if not user.stripe_customer_id:
            await self.create_customer(user)
        
        try:
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{
                    "price": tier_config["stripe_price_id"],
                    "quantity": 1
                }],
                mode="subscription",
                customer=user.stripe_customer_id,
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={
                    "user_id": str(user.id),
                    "tier": tier
                }
            )
            
            logger.info(f"Created checkout session {session.id} for user {user.id}")
            return session.url
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating checkout session: {e}")
            raise
    
    async def handle_webhook(self, payload: str, signature: str) -> bool:
        """Handle Stripe webhook events"""
        try:
            event = stripe.Webhook.construct_event(
                payload,
                signature,
                settings.STRIPE_WEBHOOK_SECRET
            )
        except ValueError:
            logger.error("Invalid webhook payload")
            return False
        except stripe.error.SignatureVerificationError:
            logger.error("Invalid webhook signature")
            return False
        
        # Handle different event types
        if event["type"] == "checkout.session.completed":
            await self._handle_checkout_completed(event["data"]["object"])
        
        elif event["type"] == "customer.subscription.updated":
            await self._handle_subscription_updated(event["data"]["object"])
        
        elif event["type"] == "customer.subscription.deleted":
            await self._handle_subscription_deleted(event["data"]["object"])
        
        elif event["type"] == "invoice.payment_succeeded":
            await self._handle_payment_succeeded(event["data"]["object"])
        
        elif event["type"] == "invoice.payment_failed":
            await self._handle_payment_failed(event["data"]["object"])
        
        logger.info(f"Processed webhook event: {event['type']}")
        return True
    
    async def _handle_free_tier(self, user: User) -> Dict[str, Any]:
        """Handle free tier subscription"""
        async with AsyncSessionLocal() as db:
            user.subscription_tier = "free"
            user.channels_limit = self.SUBSCRIPTION_TIERS["free"]["channels_limit"]
            user.videos_per_day_limit = self.SUBSCRIPTION_TIERS["free"]["videos_per_day"]
            db.add(user)
            await db.commit()
        
        return {
            "tier": "free",
            "status": "active",
            "channels_limit": user.channels_limit,
            "videos_per_day": user.videos_per_day_limit
        }
    
    async def _update_user_subscription(self, user: User, tier: str, subscription_id: str):
        """Update user subscription in database"""
        async with AsyncSessionLocal() as db:
            user.subscription_tier = tier
            user.stripe_subscription_id = subscription_id
            user.channels_limit = self.SUBSCRIPTION_TIERS[tier]["channels_limit"]
            user.videos_per_day_limit = self.SUBSCRIPTION_TIERS[tier]["videos_per_day"]
            
            # Create or update subscription record
            subscription = await db.get(Subscription, user.id)
            if not subscription:
                subscription = Subscription(
                    user_id=user.id,
                    tier=tier,
                    stripe_subscription_id=subscription_id,
                    status="active",
                    started_at=datetime.utcnow()
                )
                db.add(subscription)
            else:
                subscription.tier = tier
                subscription.stripe_subscription_id = subscription_id
                subscription.status = "active"
            
            db.add(user)
            await db.commit()
    
    async def _handle_checkout_completed(self, session: Dict[str, Any]):
        """Handle successful checkout"""
        user_id = session["metadata"]["user_id"]
        tier = session["metadata"]["tier"]
        
        async with AsyncSessionLocal() as db:
            user = await db.get(User, user_id)
            if user:
                await self._update_user_subscription(
                    user,
                    tier,
                    session["subscription"]
                )
    
    async def _handle_subscription_updated(self, subscription: Dict[str, Any]):
        """Handle subscription update"""
        # Update user's subscription status
        pass
    
    async def _handle_subscription_deleted(self, subscription: Dict[str, Any]):
        """Handle subscription cancellation"""
        # Downgrade user to free tier
        pass
    
    async def _handle_payment_succeeded(self, invoice: Dict[str, Any]):
        """Handle successful payment"""
        # Record payment in database
        pass
    
    async def _handle_payment_failed(self, invoice: Dict[str, Any]):
        """Handle failed payment"""
        # Notify user and potentially suspend service
        pass


# Global payment service instance
payment_service = PaymentService()