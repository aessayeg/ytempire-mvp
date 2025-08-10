"""
Payment System API Endpoints
Handles Stripe integration for subscriptions and payments
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from datetime import datetime
import stripe
import os
import logging

from app.db.session import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.services.payment_service import PaymentService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_API_KEY")


class CreateCheckoutSessionRequest(BaseModel):
    """Request for creating a checkout session"""
    price_id: str
    success_url: str
    cancel_url: str
    metadata: Optional[Dict[str, str]] = None


class CreateSubscriptionRequest(BaseModel):
    """Request for creating a subscription"""
    price_id: str
    payment_method_id: str
    trial_days: Optional[int] = 0


class UpdatePaymentMethodRequest(BaseModel):
    """Request for updating payment method"""
    payment_method_id: str


class SubscriptionResponse(BaseModel):
    """Response for subscription details"""
    subscription_id: str
    status: str
    current_period_start: datetime
    current_period_end: datetime
    plan_name: str
    plan_amount: float
    currency: str
    cancel_at_period_end: bool


@router.post("/checkout-session")
async def create_checkout_session(
    request: CreateCheckoutSessionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create a Stripe checkout session for subscription
    """
    try:
        payment_service = PaymentService(db)
        
        # Create or get Stripe customer
        customer_id = await payment_service.get_or_create_customer(
            user_id=current_user.id,
            email=current_user.email,
            name=current_user.full_name
        )
        
        # Create checkout session
        session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=['card'],
            line_items=[{
                'price': request.price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url=request.success_url,
            cancel_url=request.cancel_url,
            metadata={
                'user_id': str(current_user.id),
                **(request.metadata or {})
            }
        )
        
        return {
            "checkout_url": session.url,
            "session_id": session.id
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Payment error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Payment processing failed"
        )


@router.post("/subscription")
async def create_subscription(
    request: CreateSubscriptionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> SubscriptionResponse:
    """
    Create a subscription directly (for users with saved payment methods)
    """
    try:
        payment_service = PaymentService(db)
        
        # Get or create customer
        customer_id = await payment_service.get_or_create_customer(
            user_id=current_user.id,
            email=current_user.email,
            name=current_user.full_name
        )
        
        # Attach payment method to customer
        stripe.PaymentMethod.attach(
            request.payment_method_id,
            customer=customer_id
        )
        
        # Set as default payment method
        stripe.Customer.modify(
            customer_id,
            invoice_settings={
                'default_payment_method': request.payment_method_id
            }
        )
        
        # Create subscription
        subscription = stripe.Subscription.create(
            customer=customer_id,
            items=[{'price': request.price_id}],
            trial_period_days=request.trial_days,
            metadata={'user_id': str(current_user.id)}
        )
        
        # Update user subscription in database
        await payment_service.update_user_subscription(
            user_id=current_user.id,
            subscription_id=subscription.id,
            status=subscription.status,
            plan_id=request.price_id
        )
        
        # Get price details
        price = stripe.Price.retrieve(request.price_id)
        
        return SubscriptionResponse(
            subscription_id=subscription.id,
            status=subscription.status,
            current_period_start=datetime.fromtimestamp(subscription.current_period_start),
            current_period_end=datetime.fromtimestamp(subscription.current_period_end),
            plan_name=price.nickname or "Subscription",
            plan_amount=price.unit_amount / 100,
            currency=price.currency,
            cancel_at_period_end=subscription.cancel_at_period_end
        )
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/subscription")
async def get_subscription(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Optional[SubscriptionResponse]:
    """
    Get current user's subscription details
    """
    try:
        payment_service = PaymentService(db)
        subscription_data = await payment_service.get_user_subscription(current_user.id)
        
        if not subscription_data:
            return None
        
        # Get details from Stripe
        subscription = stripe.Subscription.retrieve(subscription_data['subscription_id'])
        price = stripe.Price.retrieve(subscription['items']['data'][0]['price']['id'])
        
        return SubscriptionResponse(
            subscription_id=subscription.id,
            status=subscription.status,
            current_period_start=datetime.fromtimestamp(subscription.current_period_start),
            current_period_end=datetime.fromtimestamp(subscription.current_period_end),
            plan_name=price.nickname or "Subscription",
            plan_amount=price.unit_amount / 100,
            currency=price.currency,
            cancel_at_period_end=subscription.cancel_at_period_end
        )
        
    except Exception as e:
        logger.error(f"Error fetching subscription: {e}")
        return None


@router.post("/subscription/cancel")
async def cancel_subscription(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Cancel the current subscription at period end
    """
    try:
        payment_service = PaymentService(db)
        subscription_data = await payment_service.get_user_subscription(current_user.id)
        
        if not subscription_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active subscription found"
            )
        
        # Cancel at period end
        subscription = stripe.Subscription.modify(
            subscription_data['subscription_id'],
            cancel_at_period_end=True
        )
        
        # Update database
        await payment_service.update_user_subscription(
            user_id=current_user.id,
            subscription_id=subscription.id,
            status='canceling',
            plan_id=subscription_data['plan_id']
        )
        
        return {
            "status": "success",
            "message": "Subscription will be canceled at period end",
            "cancel_at": datetime.fromtimestamp(subscription.current_period_end).isoformat()
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/subscription/resume")
async def resume_subscription(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Resume a subscription that was set to cancel
    """
    try:
        payment_service = PaymentService(db)
        subscription_data = await payment_service.get_user_subscription(current_user.id)
        
        if not subscription_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No subscription found"
            )
        
        # Resume subscription
        subscription = stripe.Subscription.modify(
            subscription_data['subscription_id'],
            cancel_at_period_end=False
        )
        
        # Update database
        await payment_service.update_user_subscription(
            user_id=current_user.id,
            subscription_id=subscription.id,
            status='active',
            plan_id=subscription_data['plan_id']
        )
        
        return {
            "status": "success",
            "message": "Subscription resumed successfully"
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/payment-methods")
async def get_payment_methods(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Get user's saved payment methods
    """
    try:
        payment_service = PaymentService(db)
        customer_id = await payment_service.get_stripe_customer_id(current_user.id)
        
        if not customer_id:
            return []
        
        # Get payment methods from Stripe
        payment_methods = stripe.PaymentMethod.list(
            customer=customer_id,
            type="card"
        )
        
        return [
            {
                "id": pm.id,
                "brand": pm.card.brand,
                "last4": pm.card.last4,
                "exp_month": pm.card.exp_month,
                "exp_year": pm.card.exp_year,
                "is_default": pm.id == stripe.Customer.retrieve(customer_id).invoice_settings.default_payment_method
            }
            for pm in payment_methods.data
        ]
        
    except Exception as e:
        logger.error(f"Error fetching payment methods: {e}")
        return []


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Handle Stripe webhook events
    """
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    payment_service = PaymentService(db)
    
    # Handle different event types
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        await payment_service.handle_checkout_completed(session)
        
    elif event['type'] == 'customer.subscription.updated':
        subscription = event['data']['object']
        await payment_service.handle_subscription_updated(subscription)
        
    elif event['type'] == 'customer.subscription.deleted':
        subscription = event['data']['object']
        await payment_service.handle_subscription_deleted(subscription)
        
    elif event['type'] == 'invoice.payment_failed':
        invoice = event['data']['object']
        await payment_service.handle_payment_failed(invoice)
    
    return {"status": "success"}


@router.get("/plans")
async def get_subscription_plans() -> List[Dict[str, Any]]:
    """
    Get available subscription plans
    """
    try:
        # Get all prices from Stripe
        prices = stripe.Price.list(active=True, expand=['data.product'])
        
        plans = []
        for price in prices.data:
            if price.type == 'recurring':
                plans.append({
                    "price_id": price.id,
                    "product_id": price.product.id,
                    "name": price.product.name,
                    "description": price.product.description,
                    "amount": price.unit_amount / 100,
                    "currency": price.currency,
                    "interval": price.recurring.interval,
                    "interval_count": price.recurring.interval_count,
                    "features": price.product.metadata.get('features', '').split(',') if price.product.metadata.get('features') else [],
                    "recommended": price.product.metadata.get('recommended', 'false') == 'true'
                })
        
        return sorted(plans, key=lambda x: x['amount'])
        
    except Exception as e:
        logger.error(f"Error fetching plans: {e}")
        return []