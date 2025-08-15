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
from app.api.v1.endpoints.auth import get_current_verified_user
from app.models.user import User
from app.services.payment_service_enhanced import payment_service, PaymentService, SubscriptionStatus

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
    plan_tier: str
    plan_amount: float
    currency: str
    cancel_at_period_end: bool
    trial_end: Optional[datetime] = None
    usage_limits: Dict[str, Any]
    current_usage: Dict[str, Any]


class UsageTrackingRequest(BaseModel):
    """Request for tracking usage"""
    usage_type: str = Field(..., description="Type of usage: videos, api_calls, storage")
    quantity: float = Field(..., gt=0, description="Quantity of usage")
    unit: str = Field(default="count", description="Unit of measurement")
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CancelSubscriptionRequest(BaseModel):
    """Request to cancel subscription"""
    cancel_at_period_end: bool = Field(default=True, description="Cancel at end of billing period")
    reason: Optional[str] = Field(None, description="Cancellation reason")


class WebhookRequest(BaseModel):
    """Stripe webhook request"""
    payload: bytes
    signature: str


class UpgradeDowngradeRequest(BaseModel):
    """Request for subscription upgrade/downgrade"""
    new_price_id: str = Field(..., description="New Stripe price ID")
    proration_behavior: str = Field(default="create_prorations", description="How to handle proration")


class PaymentMethodRequest(BaseModel):
    """Request for payment method operations"""
    payment_method_id: str = Field(..., description="Stripe payment method ID")
    set_as_default: bool = Field(default=True, description="Set as default payment method")


class InvoiceGenerationRequest(BaseModel):
    """Request for generating custom invoice"""
    items: List[Dict[str, Any]] = Field(..., description="Invoice line items")
    send_invoice: bool = Field(default=True, description="Send invoice immediately")


@router.post("/checkout-session")
async def create_checkout_session(
    request: CreateCheckoutSessionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Create a Stripe checkout session for subscription
    """
    try:
        # Create checkout session using enhanced service
        result = await payment_service.create_checkout_session(
            db=db,
            user_id=str(current_user.id),
            price_id=request.price_id,
            success_url=request.success_url,
            cancel_url=request.cancel_url,
            metadata=request.metadata
        )
        
        return result
        
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


@router.post("/subscription", response_model=SubscriptionResponse)
async def create_subscription(
    request: CreateSubscriptionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> SubscriptionResponse:
    """
    Create a new subscription with payment method
    """
    try:
        subscription = await payment_service.create_subscription(
            db=db,
            user_id=str(current_user.id),
            price_id=request.price_id,
            payment_method_id=request.payment_method_id,
            trial_days=request.trial_days
        )
        
        return SubscriptionResponse(
            subscription_id=subscription.stripe_subscription_id,
            status=subscription.status,
            current_period_start=subscription.current_period_start,
            current_period_end=subscription.current_period_end,
            plan_name=subscription.plan_name,
            plan_tier=subscription.plan_tier,
            plan_amount=subscription.unit_amount / 100,  # Convert from cents
            currency=subscription.currency,
            cancel_at_period_end=subscription.cancel_at_period_end,
            trial_end=subscription.trial_end,
            usage_limits={
                "videos_monthly": subscription.video_limit_monthly,
                "storage_gb": subscription.storage_limit_gb,
                "api_calls_monthly": subscription.api_calls_limit_monthly
            },
            current_usage={
                "videos_generated": subscription.videos_generated_current_period,
                "storage_used_gb": subscription.storage_used_gb,
                "api_calls": subscription.api_calls_current_period
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to create subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create subscription"
        )


@router.get("/subscription", response_model=SubscriptionResponse)
async def get_current_subscription(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> SubscriptionResponse:
    """
    Get current user's active subscription
    """
    try:
        subscription = await payment_service.get_active_subscription(
            db=db,
            user_id=str(current_user.id)
        )
        
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active subscription found"
            )
            
        return SubscriptionResponse(
            subscription_id=subscription.stripe_subscription_id,
            status=subscription.status,
            current_period_start=subscription.current_period_start,
            current_period_end=subscription.current_period_end,
            plan_name=subscription.plan_name,
            plan_tier=subscription.plan_tier,
            plan_amount=subscription.unit_amount / 100,
            currency=subscription.currency,
            cancel_at_period_end=subscription.cancel_at_period_end,
            trial_end=subscription.trial_end,
            usage_limits={
                "videos_monthly": subscription.video_limit_monthly,
                "storage_gb": subscription.storage_limit_gb,
                "api_calls_monthly": subscription.api_calls_limit_monthly
            },
            current_usage={
                "videos_generated": subscription.videos_generated_current_period,
                "storage_used_gb": subscription.storage_used_gb,
                "api_calls": subscription.api_calls_current_period
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve subscription"
        )


@router.put("/subscription/upgrade-downgrade")
async def upgrade_downgrade_subscription(
    request: UpgradeDowngradeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Upgrade or downgrade user's subscription tier
    """
    try:
        result = await payment_service.upgrade_downgrade_subscription(
            db=db,
            user_id=str(current_user.id),
            new_price_id=request.new_price_id,
            proration_behavior=request.proration_behavior
        )
        
        return {
            "status": "success",
            "subscription_id": result["subscription_id"],
            "new_plan": result["new_plan"],
            "effective_date": result["effective_date"],
            "proration_amount": result.get("proration_amount", 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to upgrade/downgrade subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upgrade/downgrade subscription"
        )


@router.post("/subscription/cancel")
async def cancel_subscription(
    request: CancelSubscriptionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Cancel user's subscription
    """
    try:
        # Get current subscription
        subscription = await payment_service.get_active_subscription(
            db=db,
            user_id=str(current_user.id)
        )
        
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active subscription found"
            )
            
        # Cancel subscription
        canceled_subscription = await payment_service.cancel_subscription(
            db=db,
            user_id=str(current_user.id),
            subscription_id=subscription.stripe_subscription_id,
            cancel_at_period_end=request.cancel_at_period_end,
            reason=request.reason
        )
        
        return {
            "status": "canceled",
            "subscription_id": canceled_subscription.stripe_subscription_id,
            "cancel_at_period_end": canceled_subscription.cancel_at_period_end,
            "canceled_at": canceled_subscription.canceled_at.isoformat() if canceled_subscription.canceled_at else None,
            "access_until": canceled_subscription.current_period_end.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel subscription"
        )


@router.post("/usage/track")
async def track_usage(
    request: UsageTrackingRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, str]:
    """
    Track usage for billing purposes
    """
    try:
        await payment_service.track_usage(
            db=db,
            user_id=str(current_user.id),
            usage_type=request.usage_type,
            quantity=request.quantity,
            unit=request.unit,
            resource_type=request.resource_type,
            resource_id=request.resource_id,
            metadata=request.metadata
        )
        
        return {
            "status": "tracked",
            "usage_type": request.usage_type,
            "quantity": str(request.quantity)
        }
        
    except Exception as e:
        logger.error(f"Failed to track usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to track usage"
        )


@router.get("/usage/summary")
async def get_usage_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Get current billing period usage summary
    """
    try:
        summary = await payment_service.get_usage_summary(
            db=db,
            user_id=str(current_user.id)
        )
        
        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No subscription found"
            )
            
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get usage summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage summary"
        )


@router.get("/billing/history")
async def get_billing_history(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> List[Dict[str, Any]]:
    """
    Get billing history
    """
    try:
        history = await payment_service.get_billing_history(
            db=db,
            user_id=str(current_user.id),
            limit=limit
        )
        
        return history
        
    except Exception as e:
        logger.error(f"Failed to get billing history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve billing history"
        )


@router.post("/invoices/generate")
async def generate_invoice(
    request: InvoiceGenerationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Generate a custom invoice
    """
    try:
        result = await payment_service.generate_invoice(
            db=db,
            user_id=str(current_user.id),
            items=request.items,
            send_invoice=request.send_invoice
        )
        
        return {
            "status": "success",
            "invoice_id": result["invoice_id"],
            "invoice_url": result.get("invoice_url"),
            "amount_due": result["amount_due"],
            "due_date": result["due_date"],
            "sent": result.get("sent", False)
        }
        
    except Exception as e:
        logger.error(f"Failed to generate invoice: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate invoice"
        )


@router.get("/invoices")
async def get_invoices(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> List[Dict[str, Any]]:
    """
    Get invoice history
    """
    try:
        invoices = await payment_service.get_invoice_history(
            db=db,
            user_id=str(current_user.id),
            limit=limit
        )
        
        return invoices
        
    except Exception as e:
        logger.error(f"Failed to get invoices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve invoices"
        )


@router.post("/usage/overage")
async def process_usage_overage(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Process usage overage charges for the current billing period
    """
    try:
        result = await payment_service.process_usage_overage(
            db=db,
            user_id=str(current_user.id)
        )
        
        return {
            "status": "success",
            "overage_detected": result.get("overage_detected", False),
            "overage_amount": result.get("overage_amount", 0),
            "invoice_created": result.get("invoice_created", False),
            "invoice_id": result.get("invoice_id"),
            "details": result.get("details", {})
        }
        
    except Exception as e:
        logger.error(f"Failed to process usage overage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process usage overage"
        )


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Handle Stripe webhooks
    """
    try:
        payload = await request.body()
        signature = request.headers.get("stripe-signature")
        
        if not signature:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing Stripe signature"
            )
            
        result = await payment_service.process_webhook(
            db=db,
            payload=payload,
            signature=signature
        )
        
        return result
        
    except stripe.error.SignatureVerificationError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook signature"
        )
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook processing failed"
        )


@router.get("/plans")
async def get_available_plans() -> List[Dict[str, Any]]:
    """
    Get available subscription plans
    """
    try:
        # Get plans from Stripe
        prices = stripe.Price.list(
            active=True,
            type="recurring",
            expand=["data.product"]
        )
        
        plans = []
        for price in prices.data:
            product = price.product
            
            plan = {
                "price_id": price.id,
                "product_id": product.id,
                "name": product.name,
                "description": product.description,
                "tier": product.metadata.get("tier", "basic"),
                "amount": price.unit_amount,
                "currency": price.currency,
                "interval": price.recurring.interval,
                "interval_count": price.recurring.interval_count,
                "features": product.metadata.get("features", "").split(",") if product.metadata.get("features") else [],
                "limits": {
                    "videos_monthly": int(product.metadata.get("videos_monthly", 50)),
                    "storage_gb": int(product.metadata.get("storage_gb", 25)),
                    "api_calls_monthly": int(product.metadata.get("api_calls_monthly", 5000)),
                    "channels": int(product.metadata.get("channels", 3)),
                    "video_length_minutes": int(product.metadata.get("video_length_minutes", 10))
                }
            }
            
            plans.append(plan)
            
        # Sort by amount
        plans.sort(key=lambda x: x["amount"])
        
        return plans
        
    except Exception as e:
        logger.error(f"Failed to get plans: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve plans"
        )


@router.post("/subscription")
async def create_subscription(
    request: CreateSubscriptionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
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
    current_user: User = Depends(get_current_verified_user)
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
    current_user: User = Depends(get_current_verified_user)
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
    current_user: User = Depends(get_current_verified_user)
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


@router.post("/payment-methods")
async def add_payment_method(
    request: PaymentMethodRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Add a new payment method
    """
    try:
        result = await payment_service.add_payment_method(
            db=db,
            user_id=str(current_user.id),
            payment_method_id=request.payment_method_id,
            set_as_default=request.set_as_default
        )
        
        return {
            "status": "success",
            "payment_method_id": result["payment_method_id"],
            "is_default": result["is_default"],
            "message": "Payment method added successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to add payment method: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add payment method"
        )


@router.delete("/payment-methods/{payment_method_id}")
async def remove_payment_method(
    payment_method_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Remove a payment method
    """
    try:
        result = await payment_service.remove_payment_method(
            db=db,
            user_id=str(current_user.id),
            payment_method_id=payment_method_id
        )
        
        return {
            "status": "success",
            "removed": result["removed"],
            "message": "Payment method removed successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to remove payment method: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove payment method"
        )


@router.get("/payment-methods")
async def get_payment_methods(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
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