# YTEMPIRE Integration Specialist Guide - Part 2: Payment, Webhooks & Testing

**Document Version**: 2.0  
**Date**: January 2025  
**Author**: Backend Team Lead  
**Audience**: Integration Specialist  
**Scope**: Payment Systems, Third-Party Services, Webhooks, and Testing

---

## Table of Contents
1. [Payment System Architecture](#1-payment-system-architecture)
2. [Third-Party Services Integration](#2-third-party-services-integration)
3. [Webhook Events Handling](#3-webhook-events-handling)
4. [Integration Testing Protocols](#4-integration-testing-protocols)
5. [Operational Procedures](#5-operational-procedures)

---

## 1. Payment System Architecture

### 1.1 Stripe Integration Overview

As the Integration Specialist, you'll manage the complete payment infrastructure using Stripe. This includes subscription management, payment processing, and webhook handling for our three-tier pricing model.

```python
import stripe
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import json

class StripePaymentIntegration:
    """
    Complete Stripe payment integration for YTEMPIRE
    Manages subscriptions, one-time payments, and customer lifecycle
    """
    
    def __init__(self):
        # Initialize Stripe with API key
        stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')
        self.webhook_secret = os.environ.get('STRIPE_WEBHOOK_SECRET')
        self.api_version = '2023-10-16'
        
        # Product configuration
        self.subscription_tiers = {
            'starter': {
                'price_id': 'price_starter_monthly',
                'amount': 9700,  # $97.00 in cents
                'channels': 5,
                'videos_per_day': 15,
                'features': ['basic_analytics', 'email_support']
            },
            'growth': {
                'price_id': 'price_growth_monthly',
                'amount': 29700,  # $297.00 in cents
                'channels': 10,
                'videos_per_day': 30,
                'features': ['advanced_analytics', 'priority_support', 'api_access']
            },
            'scale': {
                'price_id': 'price_scale_monthly',
                'amount': 79700,  # $797.00 in cents
                'channels': 25,
                'videos_per_day': 75,
                'features': ['enterprise_analytics', 'dedicated_support', 'api_access', 'custom_integrations']
            }
        }
        
        # Payment retry configuration
        self.retry_config = {
            'max_attempts': 4,
            'retry_intervals': [3, 5, 7],  # Days between retries
            'grace_period': 7  # Days before suspension
        }
```

### 1.2 Customer Management

```python
class StripeCustomerManager:
    """
    Manages Stripe customer creation and updates
    """
    
    def __init__(self):
        self.stripe_integration = StripePaymentIntegration()
        
    async def create_customer(self, user_data: Dict) -> Dict:
        """
        Create a new Stripe customer
        
        Args:
            user_data: {
                'email': str,
                'name': str,
                'user_id': str,  # Internal YTEMPIRE user ID
                'phone': str (optional),
                'address': dict (optional)
            }
        
        Returns:
            Customer creation result with Stripe customer ID
        """
        
        try:
            # Create customer in Stripe
            customer = stripe.Customer.create(
                email=user_data['email'],
                name=user_data.get('name'),
                phone=user_data.get('phone'),
                address=user_data.get('address'),
                metadata={
                    'user_id': user_data['user_id'],
                    'signup_date': datetime.utcnow().isoformat(),
                    'platform': 'YTEMPIRE',
                    'source': user_data.get('source', 'organic')
                }
            )
            
            # Store customer mapping in database
            await self._store_customer_mapping(
                user_id=user_data['user_id'],
                stripe_customer_id=customer.id
            )
            
            # Log customer creation
            await self._log_customer_event(
                event_type='customer_created',
                customer_id=customer.id,
                metadata=user_data
            )
            
            return {
                'success': True,
                'customer_id': customer.id,
                'customer': customer.to_dict(),
                'created_at': datetime.utcnow()
            }
            
        except stripe.error.StripeError as e:
            return await self._handle_stripe_error(e, 'customer_creation')
    
    async def update_customer(self, customer_id: str, updates: Dict) -> Dict:
        """
        Update existing Stripe customer
        """
        
        try:
            customer = stripe.Customer.modify(
                customer_id,
                **updates
            )
            
            return {
                'success': True,
                'customer': customer.to_dict()
            }
            
        except stripe.error.StripeError as e:
            return await self._handle_stripe_error(e, 'customer_update')
```

### 1.3 Subscription Management

```python
class StripeSubscriptionManager:
    """
    Handles all subscription lifecycle operations
    """
    
    def __init__(self):
        self.stripe_integration = StripePaymentIntegration()
        
    async def create_subscription(
        self,
        customer_id: str,
        plan: str,
        payment_method_id: Optional[str] = None,
        trial_days: int = 0
    ) -> Dict:
        """
        Create a new subscription for a customer
        
        Args:
            customer_id: Stripe customer ID
            plan: 'starter', 'growth', or 'scale'
            payment_method_id: Payment method to use
            trial_days: Number of trial days (0 for no trial)
        
        Returns:
            Subscription creation result
        """
        
        tier = self.stripe_integration.subscription_tiers.get(plan)
        if not tier:
            raise ValueError(f"Invalid plan: {plan}")
        
        try:
            # Prepare subscription data
            subscription_data = {
                'customer': customer_id,
                'items': [{'price': tier['price_id']}],
                'payment_behavior': 'default_incomplete',
                'payment_settings': {
                    'save_default_payment_method': 'on_subscription'
                },
                'metadata': {
                    'plan': plan,
                    'channels': str(tier['channels']),
                    'videos_per_day': str(tier['videos_per_day'])
                },
                'expand': ['latest_invoice.payment_intent']
            }
            
            # Add payment method if provided
            if payment_method_id:
                subscription_data['default_payment_method'] = payment_method_id
            
            # Add trial period if specified
            if trial_days > 0:
                subscription_data['trial_period_days'] = trial_days
            
            # Create subscription
            subscription = stripe.Subscription.create(**subscription_data)
            
            # Store subscription details
            await self._store_subscription(subscription, plan)
            
            # Activate user features
            await self._activate_plan_features(customer_id, plan)
            
            return {
                'success': True,
                'subscription_id': subscription.id,
                'status': subscription.status,
                'current_period_end': subscription.current_period_end,
                'client_secret': subscription.latest_invoice.payment_intent.client_secret
                    if subscription.latest_invoice and subscription.latest_invoice.payment_intent
                    else None
            }
            
        except stripe.error.StripeError as e:
            return await self._handle_stripe_error(e, 'subscription_creation')
    
    async def update_subscription(
        self,
        subscription_id: str,
        new_plan: str,
        prorate: bool = True
    ) -> Dict:
        """
        Update subscription (upgrade/downgrade)
        """
        
        new_tier = self.stripe_integration.subscription_tiers.get(new_plan)
        if not new_tier:
            raise ValueError(f"Invalid plan: {new_plan}")
        
        try:
            # Retrieve current subscription
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            # Update subscription
            updated_subscription = stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=False,
                proration_behavior='create_prorations' if prorate else 'none',
                items=[{
                    'id': subscription['items']['data'][0].id,
                    'price': new_tier['price_id']
                }],
                metadata={
                    'plan': new_plan,
                    'channels': str(new_tier['channels']),
                    'videos_per_day': str(new_tier['videos_per_day']),
                    'updated_at': datetime.utcnow().isoformat()
                }
            )
            
            # Update user features
            await self._update_plan_features(
                subscription.customer,
                new_plan
            )
            
            return {
                'success': True,
                'subscription': updated_subscription.to_dict(),
                'new_plan': new_plan
            }
            
        except stripe.error.StripeError as e:
            return await self._handle_stripe_error(e, 'subscription_update')
    
    async def cancel_subscription(
        self,
        subscription_id: str,
        immediately: bool = False,
        reason: Optional[str] = None
    ) -> Dict:
        """
        Cancel a subscription
        """
        
        try:
            if immediately:
                # Cancel immediately
                subscription = stripe.Subscription.delete(subscription_id)
            else:
                # Cancel at period end
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True,
                    metadata={'cancellation_reason': reason or 'user_requested'}
                )
            
            # Update user access
            await self._handle_subscription_cancellation(subscription)
            
            return {
                'success': True,
                'subscription': subscription.to_dict(),
                'cancelled_immediately': immediately
            }
            
        except stripe.error.StripeError as e:
            return await self._handle_stripe_error(e, 'subscription_cancellation')
```

### 1.4 Payment Processing

```python
class PaymentProcessor:
    """
    Handles payment processing with comprehensive error handling
    """
    
    def __init__(self):
        self.stripe_integration = StripePaymentIntegration()
        
    async def process_payment(self, payment_data: Dict) -> Dict:
        """
        Process a one-time payment
        
        Args:
            payment_data: {
                'amount': int (in cents),
                'currency': str,
                'customer_id': str,
                'payment_method_id': str,
                'description': str,
                'metadata': dict
            }
        """
        
        try:
            # Create payment intent
            payment_intent = stripe.PaymentIntent.create(
                amount=payment_data['amount'],
                currency=payment_data.get('currency', 'usd'),
                customer=payment_data['customer_id'],
                payment_method=payment_data.get('payment_method_id'),
                description=payment_data.get('description'),
                metadata=payment_data.get('metadata', {}),
                confirm=True if payment_data.get('payment_method_id') else False,
                automatic_payment_methods={
                    'enabled': True,
                    'allow_redirects': 'never'
                } if not payment_data.get('payment_method_id') else None
            )
            
            # Handle payment status
            return await self._handle_payment_status(payment_intent)
            
        except stripe.error.CardError as e:
            # Card was declined
            return {
                'success': False,
                'error': str(e.user_message),
                'error_type': 'card_declined',
                'decline_code': e.code
            }
            
        except stripe.error.StripeError as e:
            return await self._handle_stripe_error(e, 'payment_processing')
    
    async def _handle_payment_status(self, payment_intent) -> Dict:
        """
        Handle different payment intent statuses
        """
        
        if payment_intent.status == 'succeeded':
            await self._record_successful_payment(payment_intent)
            return {
                'success': True,
                'payment_intent_id': payment_intent.id,
                'status': 'succeeded',
                'amount': payment_intent.amount
            }
            
        elif payment_intent.status == 'requires_action':
            return {
                'success': False,
                'requires_action': True,
                'client_secret': payment_intent.client_secret,
                'status': 'requires_action',
                'action_type': '3d_secure'
            }
            
        elif payment_intent.status == 'processing':
            return {
                'success': False,
                'status': 'processing',
                'payment_intent_id': payment_intent.id,
                'retry_after': 60  # Check again in 60 seconds
            }
            
        else:
            return {
                'success': False,
                'status': payment_intent.status,
                'error': 'Payment failed'
            }
```

### 1.5 Refund Management

```python
class RefundManager:
    """
    Handles refunds and disputes
    """
    
    def __init__(self):
        self.refund_policy = {
            'full_refund_period_days': 14,
            'partial_refund_period_days': 30,
            'auto_approve_threshold_cents': 10000  # $100
        }
        
    async def process_refund(
        self,
        payment_intent_id: str,
        amount: Optional[int] = None,
        reason: str = 'requested_by_customer',
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Process a refund request
        """
        
        try:
            # Get payment details
            payment_intent = stripe.PaymentIntent.retrieve(payment_intent_id)
            
            # Check refund eligibility
            eligibility = await self._check_refund_eligibility(payment_intent)
            
            if not eligibility['eligible']:
                return {
                    'success': False,
                    'reason': eligibility['reason']
                }
            
            # Calculate refund amount if not specified
            if amount is None:
                amount = payment_intent.amount
            
            # Create refund
            refund = stripe.Refund.create(
                payment_intent=payment_intent_id,
                amount=amount,
                reason=reason,
                metadata=metadata or {}
            )
            
            # Update user account for refund
            await self._adjust_user_account_for_refund(
                payment_intent.customer,
                amount
            )
            
            return {
                'success': True,
                'refund_id': refund.id,
                'amount': amount,
                'status': refund.status,
                'reason': reason
            }
            
        except stripe.error.StripeError as e:
            return await self._handle_stripe_error(e, 'refund_processing')
```

---

## 2. Third-Party Services Integration

### 2.1 OpenAI Integration

```python
import openai
from typing import Dict, Optional
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

class OpenAIIntegration:
    """
    OpenAI API integration for content generation
    Implements cost tracking and fallback strategies
    """
    
    def __init__(self):
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        
        # Model configurations with cost tracking
        self.model_configs = {
            'economy': {
                'model': 'gpt-3.5-turbo',
                'max_tokens': 1500,
                'temperature': 0.7,
                'cost_per_1k_tokens': 0.002,
                'quality': 'basic'
            },
            'standard': {
                'model': 'gpt-3.5-turbo',
                'max_tokens': 2000,
                'temperature': 0.8,
                'cost_per_1k_tokens': 0.002,
                'quality': 'good'
            },
            'premium': {
                'model': 'gpt-4',
                'max_tokens': 2500,
                'temperature': 0.7,
                'cost_per_1k_tokens': 0.03,
                'quality': 'excellent'
            }
        }
        
        self.daily_budget = 50.00  # $50 per day limit
        self.cost_tracker = CostTracker()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_content(
        self,
        prompt: str,
        content_type: str = 'script',
        optimization_level: str = 'standard'
    ) -> Dict:
        """
        Generate content using OpenAI API
        
        Args:
            prompt: The prompt for content generation
            content_type: 'script', 'description', 'title', 'tags'
            optimization_level: 'economy', 'standard', 'premium'
        """
        
        config = self.model_configs[optimization_level]
        
        # Check daily budget
        current_spend = await self.cost_tracker.get_daily_spend('openai')
        if current_spend >= self.daily_budget:
            # Fallback to economy mode
            config = self.model_configs['economy']
            optimization_level = 'economy'
        
        try:
            # Prepare messages based on content type
            messages = self._prepare_messages(prompt, content_type)
            
            # Make API call
            response = await openai.ChatCompletion.acreate(
                model=config['model'],
                messages=messages,
                max_tokens=config['max_tokens'],
                temperature=config['temperature'],
                presence_penalty=0.0,
                frequency_penalty=0.0
            )
            
            # Extract content
            generated_content = response.choices[0].message.content
            
            # Calculate and track cost
            tokens_used = response.usage.total_tokens
            cost = (tokens_used / 1000) * config['cost_per_1k_tokens']
            await self.cost_tracker.track_cost('openai', cost)
            
            return {
                'success': True,
                'content': generated_content,
                'model': config['model'],
                'tokens_used': tokens_used,
                'cost': cost,
                'optimization_level': optimization_level,
                'quality': config['quality']
            }
            
        except openai.error.RateLimitError as e:
            # Handle rate limiting
            await asyncio.sleep(60)  # Wait 1 minute
            raise  # Retry will handle
            
        except openai.error.APIError as e:
            # Log error and use fallback
            return await self._use_template_fallback(prompt, content_type)
    
    def _prepare_messages(self, prompt: str, content_type: str) -> List[Dict]:
        """
        Prepare messages for different content types
        """
        
        system_messages = {
            'script': "You are an expert YouTube script writer. Create engaging, informative scripts.",
            'description': "You are an SEO expert. Write compelling YouTube descriptions.",
            'title': "You are a YouTube title expert. Create click-worthy but accurate titles.",
            'tags': "You are a YouTube SEO specialist. Generate relevant tags for maximum reach."
        }
        
        return [
            {"role": "system", "content": system_messages.get(content_type, system_messages['script'])},
            {"role": "user", "content": prompt}
        ]
```

### 2.2 Text-to-Speech Integration

```python
from google.cloud import texttospeech
import aiohttp
from abc import ABC, abstractmethod

class TTSProvider(ABC):
    """
    Abstract base class for TTS providers
    """
    
    @abstractmethod
    async def synthesize(self, text: str, voice_settings: Optional[Dict] = None) -> Dict:
        pass
    
    @abstractmethod
    async def is_healthy(self) -> bool:
        pass


class GoogleTTSProvider(TTSProvider):
    """
    Google Cloud Text-to-Speech provider
    Primary TTS service due to cost efficiency
    """
    
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()
        self.cost_per_1k_chars = 0.016  # Neural voices
        
        # Default voice configuration
        self.default_voice = texttospeech.VoiceSelectionParams(
            language_code='en-US',
            name='en-US-Neural2-J',
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
        
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )
    
    async def synthesize(self, text: str, voice_settings: Optional[Dict] = None) -> Dict:
        """
        Synthesize speech using Google TTS
        """
        
        try:
            # Use custom voice settings if provided
            voice = voice_settings or self.default_voice
            
            # Create synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Perform synthesis
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=self.audio_config
            )
            
            # Calculate cost
            char_count = len(text)
            cost = (char_count / 1000) * self.cost_per_1k_chars
            
            return {
                'success': True,
                'audio_data': response.audio_content,
                'cost': cost,
                'char_count': char_count,
                'provider': 'google_tts'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'provider': 'google_tts'
            }
    
    async def is_healthy(self) -> bool:
        """
        Check if Google TTS is available
        """
        
        try:
            # Test with minimal text
            result = await self.synthesize("test")
            return result['success']
        except:
            return False


class ElevenLabsProvider(TTSProvider):
    """
    ElevenLabs TTS provider
    Premium fallback for high-quality voices
    """
    
    def __init__(self):
        self.api_key = os.environ.get('ELEVENLABS_API_KEY')
        self.base_url = 'https://api.elevenlabs.io/v1'
        self.default_voice_id = '21m00Tcm4TlvDq8ikWAM'
        self.cost_per_1k_chars = 0.30
        self.monthly_character_limit = 100000
    
    async def synthesize(self, text: str, voice_settings: Optional[Dict] = None) -> Dict:
        """
        Synthesize speech using ElevenLabs
        """
        
        url = f"{self.base_url}/text-to-speech/{self.default_voice_id}"
        
        headers = {
            'Accept': 'audio/mpeg',
            'xi-api-key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        data = {
            'text': text,
            'model_id': 'eleven_monolingual_v1',
            'voice_settings': voice_settings or {
                'stability': 0.5,
                'similarity_boost': 0.5
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    audio_content = await response.read()
                    
                    # Calculate cost
                    char_count = len(text)
                    cost = (char_count / 1000) * self.cost_per_1k_chars
                    
                    return {
                        'success': True,
                        'audio_data': audio_content,
                        'cost': cost,
                        'char_count': char_count,
                        'provider': 'elevenlabs'
                    }
                else:
                    return {
                        'success': False,
                        'error': f"API error: {response.status}",
                        'provider': 'elevenlabs'
                    }
    
    async def is_healthy(self) -> bool:
        """
        Check if ElevenLabs is available
        """
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/user",
                    headers={'xi-api-key': self.api_key}
                ) as response:
                    return response.status == 200
        except:
            return False


class TTSManager:
    """
    Manages multiple TTS providers with fallback
    """
    
    def __init__(self):
        self.providers = {
            'google': GoogleTTSProvider(),
            'elevenlabs': ElevenLabsProvider()
        }
        self.provider_priority = ['google', 'elevenlabs']  # Order of preference
    
    async def synthesize_speech(
        self,
        text: str,
        preferred_provider: Optional[str] = None
    ) -> Dict:
        """
        Synthesize speech with automatic fallback
        """
        
        # Use preferred provider if specified and available
        if preferred_provider and preferred_provider in self.providers:
            provider = self.providers[preferred_provider]
            if await provider.is_healthy():
                result = await provider.synthesize(text)
                if result['success']:
                    return result
        
        # Try providers in priority order
        for provider_name in self.provider_priority:
            provider = self.providers[provider_name]
            
            if await provider.is_healthy():
                result = await provider.synthesize(text)
                if result['success']:
                    return result
        
        # All providers failed
        return {
            'success': False,
            'error': 'All TTS providers failed'
        }
```

### 2.3 Stock Media Integration

```python
class StockMediaManager:
    """
    Manages integration with stock media providers
    """
    
    def __init__(self):
        self.pexels_api_key = os.environ.get('PEXELS_API_KEY')
        self.pixabay_api_key = os.environ.get('PIXABAY_API_KEY')
        self.cache_dir = '/var/cache/stock_media'
        
    async def search_videos(
        self,
        keywords: List[str],
        count: int = 10,
        min_duration: int = 5,
        max_duration: int = 60
    ) -> List[Dict]:
        """
        Search for stock videos across providers
        """
        
        results = []
        
        # Search Pexels
        pexels_results = await self._search_pexels(keywords, count)
        results.extend(pexels_results)
        
        # If not enough results, search Pixabay
        if len(results) < count:
            pixabay_results = await self._search_pixabay(
                keywords,
                count - len(results)
            )
            results.extend(pixabay_results)
        
        return results
    
    async def _search_pexels(self, keywords: List[str], count: int) -> List[Dict]:
        """
        Search Pexels for videos
        """
        
        url = "https://api.pexels.com/videos/search"
        headers = {'Authorization': self.pexels_api_key}
        params = {
            'query': ' '.join(keywords),
            'per_page': count
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    videos = []
                    for video in data.get('videos', []):
                        videos.append({
                            'id': f"pexels_{video['id']}",
                            'url': video['video_files'][0]['link'],
                            'thumbnail': video['image'],
                            'duration': video['duration'],
                            'width': video['width'],
                            'height': video['height'],
                            'provider': 'pexels'
                        })
                    
                    return videos
                
                return []
```

---

## 3. Webhook Events Handling

### 3.1 Webhook Management System

```python
from fastapi import Request, HTTPException
import hmac
import hashlib
from typing import Dict, Callable

class WebhookManager:
    """
    Centralized webhook handling for all services
    """
    
    def __init__(self):
        self.handlers = {
            'stripe': StripeWebhookHandler(),
            'youtube': YouTubeWebhookHandler(),
            'n8n': N8NWebhookHandler()
        }
        self.event_store = WebhookEventStore()
    
    async def handle_webhook(
        self,
        service: str,
        request: Request
    ) -> Dict:
        """
        Main webhook entry point
        """
        
        # Get appropriate handler
        handler = self.handlers.get(service)
        if not handler:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown webhook service: {service}"
            )
        
        # Verify webhook signature
        if not await handler.verify_signature(request):
            raise HTTPException(
                status_code=401,
                detail="Invalid webhook signature"
            )
        
        # Parse event
        event = await handler.parse_event(request)
        
        # Check for duplicate (idempotency)
        if await self.event_store.is_duplicate(event['id']):
            return {
                'status': 'duplicate',
                'event_id': event['id']
            }
        
        # Store event
        await self.event_store.store_event(event)
        
        # Process event
        try:
            result = await handler.process_event(event)
            
            # Mark as processed
            await self.event_store.mark_processed(event['id'])
            
            return {
                'status': 'success',
                'event_id': event['id'],
                'result': result
            }
            
        except Exception as e:
            # Log error
            await self.event_store.mark_failed(event['id'], str(e))
            
            # Determine if retry is needed
            if handler.should_retry(event, e):
                await self._schedule_retry(service, event)
            
            raise
```

### 3.2 Stripe Webhook Handler

```python
class StripeWebhookHandler:
    """
    Handles all Stripe webhook events
    """
    
    def __init__(self):
        self.webhook_secret = os.environ.get('STRIPE_WEBHOOK_SECRET')
        
        # Map event types to handlers
        self.event_handlers = {
            'customer.subscription.created': self._handle_subscription_created,
            'customer.subscription.updated': self._handle_subscription_updated,
            'customer.subscription.deleted': self._handle_subscription_deleted,
            'invoice.payment_succeeded': self._handle_payment_succeeded,
            'invoice.payment_failed': self._handle_payment_failed,
            'payment_intent.succeeded': self._handle_payment_intent_succeeded,
            'payment_intent.payment_failed': self._handle_payment_intent_failed,
            'charge.dispute.created': self._handle_dispute_created
        }
    
    async def verify_signature(self, request: Request) -> bool:
        """
        Verify Stripe webhook signature
        """
        
        payload = await request.body()
        sig_header = request.headers.get('Stripe-Signature')
        
        if not sig_header:
            return False
        
        try:
            stripe.Webhook.construct_event(
                payload,
                sig_header,
                self.webhook_secret
            )
            return True
        except:
            return False
    
    async def parse_event(self, request: Request) -> Dict:
        """
        Parse Stripe webhook event
        """
        
        payload = await request.body()
        sig_header = request.headers.get('Stripe-Signature')
        
        event = stripe.Webhook.construct_event(
            payload,
            sig_header,
            self.webhook_secret
        )
        
        return {
            'id': event['id'],
            'type': event['type'],
            'data': event['data']['object'],
            'created': event['created']
        }
    
    async def process_event(self, event: Dict) -> Dict:
        """
        Process Stripe webhook event
        """
        
        event_type = event['type']
        handler = self.event_handlers.get(event_type)
        
        if not handler:
            return {'status': 'unhandled', 'event_type': event_type}
        
        return await handler(event['data'])
    
    async def _handle_subscription_created(self, subscription: Dict) -> Dict:
        """
        Handle new subscription creation
        """
        
        # Activate user features
        await self._activate_subscription_features(subscription)
        
        # Send welcome email
        await self._send_welcome_email(subscription['customer'])
        
        # Log event
        await self._log_subscription_event('created', subscription)
        
        return {
            'action': 'subscription_activated',
            'subscription_id': subscription['id']
        }
    
    async def _handle_payment_failed(self, invoice: Dict) -> Dict:
        """
        Handle failed payment
        """
        
        attempt_count = invoice.get('attempt_count', 1)
        
        if attempt_count <= 3:
            # Schedule retry
            await self._schedule_payment_retry(invoice)
            action = 'retry_scheduled'
        else:
            # Suspend subscription
            await self._suspend_subscription(invoice['subscription'])
            action = 'subscription_suspended'
        
        # Notify user
        await self._notify_payment_failure(invoice, action)
        
        return {
            'action': action,
            'attempt': attempt_count
        }
    
    def should_retry(self, event: Dict, error: Exception) -> bool:
        """
        Determine if event should be retried
        """
        
        # Don't retry certain event types
        non_retryable = ['charge.dispute.created', 'customer.deleted']
        if event['type'] in non_retryable:
            return False
        
        # Retry on temporary errors
        if isinstance(error, (ConnectionError, TimeoutError)):
            return True
        
        return False
```

### 3.3 Custom Webhook Implementation

```python
class N8NWebhookHandler:
    """
    Handles N8N workflow webhooks
    """
    
    def __init__(self):
        self.webhook_secret = os.environ.get('N8N_WEBHOOK_SECRET')
    
    async def verify_signature(self, request: Request) -> bool:
        """
        Verify N8N webhook signature
        """
        
        payload = await request.body()
        signature = request.headers.get('X-N8N-Signature')
        
        if not signature:
            return False
        
        expected = hmac.new(
            self.webhook_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected)
    
    async def parse_event(self, request: Request) -> Dict:
        """
        Parse N8N webhook event
        """
        
        data = await request.json()
        
        return {
            'id': data.get('execution_id'),
            'type': 'n8n_workflow',
            'workflow_id': data.get('workflow_id'),
            'status': data.get('status'),
            'data': data.get('data', {}),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def process_event(self, event: Dict) -> Dict:
        """
        Process N8N workflow event
        """
        
        status = event['status']
        
        if status == 'success':
            # Update video status
            await self._update_video_status(
                event['data'].get('video_id'),
                'completed'
            )
            return {'action': 'workflow_completed'}
            
        elif status == 'error':
            # Handle workflow error
            await self._handle_workflow_error(event)
            return {'action': 'error_handled'}
        
        return {'action': 'no_action'}
```

---

## 4. Integration Testing Protocols

### 4.1 Test Environment Setup

```python
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json

class IntegrationTestEnvironment:
    """
    Complete test environment for integration testing
    """
    
    def __init__(self):
        self.test_config = self._load_test_config()
        self.mock_services = {}
        self.test_data = {}
    
    def _load_test_config(self) -> Dict:
        """
        Load test configuration
        """
        
        return {
            'stripe_test_key': os.environ.get('STRIPE_TEST_KEY'),
            'youtube_test_accounts': [
                'test_account_01',
                'test_account_02',
                'test_account_03'
            ],
            'test_api_keys': {
                'openai': 'test_openai_key',
                'elevenlabs': 'test_elevenlabs_key',
                'pexels': 'test_pexels_key'
            }
        }
    
    async def setup(self):
        """
        Set up test environment
        """
        
        # Initialize mock services
        self.mock_services = {
            'stripe': await self._setup_stripe_mock(),
            'youtube': await self._setup_youtube_mock(),
            'openai': await self._setup_openai_mock(),
            'tts': await self._setup_tts_mock()
        }
        
        # Load test data
        self.test_data = await self._load_test_data()
    
    async def teardown(self):
        """
        Clean up test environment
        """
        
        # Clean up test data
        await self._cleanup_test_data()
        
        # Reset mocks
        for mock in self.mock_services.values():
            mock.reset()
```

### 4.2 Payment System Tests

```python
@pytest.mark.asyncio
class TestPaymentIntegration:
    """
    Test payment system integration
    """
    
    async def test_subscription_lifecycle(self):
        """
        Test complete subscription lifecycle
        """
        
        # Initialize
        stripe_integration = StripePaymentIntegration()
        customer_manager = StripeCustomerManager()
        subscription_manager = StripeSubscriptionManager()
        
        # Create customer
        customer_result = await customer_manager.create_customer({
            'email': 'test@example.com',
            'name': 'Test User',
            'user_id': 'test_user_123'
        })
        
        assert customer_result['success'] == True
        customer_id = customer_result['customer_id']
        
        # Create subscription
        subscription_result = await subscription_manager.create_subscription(
            customer_id=customer_id,
            plan='starter',
            trial_days=14
        )
        
        assert subscription_result['success'] == True
        assert subscription_result['status'] in ['trialing', 'active']
        
        # Update subscription
        update_result = await subscription_manager.update_subscription(
            subscription_id=subscription_result['subscription_id'],
            new_plan='growth'
        )
        
        assert update_result['success'] == True
        assert update_result['new_plan'] == 'growth'
        
        # Cancel subscription
        cancel_result = await subscription_manager.cancel_subscription(
            subscription_id=subscription_result['subscription_id'],
            immediately=False
        )
        
        assert cancel_result['success'] == True
    
    async def test_payment_processing(self):
        """
        Test payment processing with different scenarios
        """
        
        processor = PaymentProcessor()
        
        # Test successful payment
        result = await processor.process_payment({
            'amount': 10000,  # $100
            'currency': 'usd',
            'customer_id': 'cus_test123',
            'payment_method_id': 'pm_card_visa',
            'description': 'Test payment'
        })
        
        assert result['success'] == True
        assert result['status'] == 'succeeded'
    
    async def test_webhook_handling(self):
        """
        Test webhook signature verification and processing
        """
        
        handler = StripeWebhookHandler()
        
        # Create test event
        test_event = {
            'id': 'evt_test123',
            'type': 'customer.subscription.created',
            'data': {
                'object': {
                    'id': 'sub_test123',
                    'customer': 'cus_test123',
                    'status': 'active'
                }
            }
        }
        
        # Process event
        result = await handler.process_event(test_event)
        
        assert result['action'] == 'subscription_activated'
```

### 4.3 Third-Party Service Tests

```python
@pytest.mark.asyncio
class TestThirdPartyServices:
    """
    Test third-party service integrations
    """
    
    async def test_openai_integration(self):
        """
        Test OpenAI content generation
        """
        
        openai_integration = OpenAIIntegration()
        
        # Test script generation
        result = await openai_integration.generate_content(
            prompt="Create a script about test topic",
            content_type='script',
            optimization_level='economy'
        )
        
        assert result['success'] == True
        assert len(result['content']) > 0
        assert result['cost'] < 0.10  # Should be under $0.10
    
    async def test_tts_fallback(self):
        """
        Test TTS provider fallback
        """
        
        tts_manager = TTSManager()
        
        # Mock Google TTS failure
        with patch.object(
            tts_manager.providers['google'],
            'is_healthy',
            return_value=False
        ):
            # Should fallback to ElevenLabs
            result = await tts_manager.synthesize_speech("Test text")
            
            assert result['success'] == True
            assert result['provider'] == 'elevenlabs'
    
    async def test_stock_media_search(self):
        """
        Test stock media search
        """
        
        stock_manager = StockMediaManager()
        
        # Search for videos
        results = await stock_manager.search_videos(
            keywords=['technology', 'computer'],
            count=5
        )
        
        assert len(results) > 0
        assert results[0]['provider'] in ['pexels', 'pixabay']
```

### 4.4 End-to-End Integration Tests

```python
@pytest.mark.asyncio
class TestEndToEndIntegration:
    """
    End-to-end integration tests
    """
    
    async def test_complete_payment_flow(self):
        """
        Test complete payment flow from customer creation to subscription
        """
        
        # Create customer
        customer_manager = StripeCustomerManager()
        customer = await customer_manager.create_customer({
            'email': 'e2e_test@example.com',
            'name': 'E2E Test User',
            'user_id': 'e2e_user_123'
        })
        
        # Add payment method
        payment_method = stripe.PaymentMethod.create(
            type='card',
            card={
                'number': '4242424242424242',
                'exp_month': 12,
                'exp_year': 2025,
                'cvc': '123'
            }
        )
        
        # Attach to customer
        payment_method.attach(customer=customer['customer_id'])
        
        # Create subscription
        subscription_manager = StripeSubscriptionManager()
        subscription = await subscription_manager.create_subscription(
            customer_id=customer['customer_id'],
            plan='starter',
            payment_method_id=payment_method.id
        )
        
        assert subscription['success'] == True
        assert subscription['status'] == 'active'
    
    async def test_webhook_to_action_flow(self):
        """
        Test webhook event triggering appropriate actions
        """
        
        webhook_manager = WebhookManager()
        
        # Simulate payment failed webhook
        mock_request = Mock()
        mock_request.body = AsyncMock(return_value=json.dumps({
            'id': 'evt_test',
            'type': 'invoice.payment_failed',
            'data': {
                'object': {
                    'customer': 'cus_test',
                    'subscription': 'sub_test',
                    'attempt_count': 1
                }
            }
        }).encode())
        
        mock_request.headers = {
            'Stripe-Signature': 'test_signature'
        }
        
        # Process webhook
        with patch.object(
            webhook_manager.handlers['stripe'],
            'verify_signature',
            return_value=True
        ):
            result = await webhook_manager.handle_webhook('stripe', mock_request)
            
            assert result['status'] == 'success'
```

---

## 5. Operational Procedures

### 5.1 Daily Operations Checklist

```yaml
daily_operations:
  morning_tasks:  # 9:00 AM
    payment_system:
      - Check Stripe dashboard for failed payments
      - Review overnight subscription changes
      - Verify webhook processing queue
      - Check for any disputes or chargebacks
    
    third_party_services:
      - Verify OpenAI API key validity
      - Check TTS provider quotas
      - Test one API endpoint for each service
      - Review service health dashboards
    
    webhooks:
      - Check webhook event queue
      - Review failed webhook retries
      - Verify signature keys are valid
      - Clear processed events older than 7 days
  
  afternoon_tasks:  # 2:00 PM
    monitoring:
      - Review payment success rates
      - Check API usage and costs
      - Monitor webhook latency
      - Verify fallback activation rates
    
    testing:
      - Run integration test suite
      - Test one payment flow manually
      - Verify webhook endpoints
      - Check service failover
  
  evening_tasks:  # 6:00 PM
    reporting:
      - Generate daily payment report
      - Calculate API costs for the day
      - Review webhook processing metrics
      - Update documentation if needed
```

### 5.2 Monitoring and Alerts

```python
class IntegrationMonitoring:
    """
    Monitoring configuration for all integrations
    """
    
    ALERT_THRESHOLDS = {
        'payment_failure_rate': {
            'warning': 0.05,  # 5%
            'critical': 0.10   # 10%
        },
        'webhook_latency': {
            'warning': 5000,   # 5 seconds
            'critical': 10000  # 10 seconds
        },
        'api_error_rate': {
            'warning': 0.02,   # 2%
            'critical': 0.05   # 5%
        },
        'daily_cost': {
            'warning': 40.00,  # $40
            'critical': 50.00  # $50
        }
    }
    
    HEALTH_CHECKS = {
        'stripe': {
            'endpoint': 'https://api.stripe.com/v1/charges',
            'frequency': 300,  # 5 minutes
            'timeout': 5
        },
        'openai': {
            'endpoint': 'https://api.openai.com/v1/models',
            'frequency': 600,  # 10 minutes
            'timeout': 10
        },
        'google_tts': {
            'endpoint': 'custom_health_check',
            'frequency': 600,
            'timeout': 5
        }
    }
```

### 5.3 Emergency Procedures

```yaml
emergency_procedures:
  payment_system_down:
    severity: CRITICAL
    steps:
      1. Switch to offline mode for new signups
      2. Queue all payment attempts
      3. Notify finance team
      4. Contact Stripe support if needed
      5. Process queued payments when restored
  
  all_tts_providers_failed:
    severity: HIGH
    steps:
      1. Activate emergency TTS cache
      2. Use pre-generated common phrases
      3. Pause new video generation
      4. Contact provider support
      5. Consider manual voiceover option
  
  webhook_signature_mismatch:
    severity: HIGH
    steps:
      1. Immediately block affected endpoint
      2. Rotate webhook secrets
      3. Review recent webhook events
      4. Check for security breach
      5. Re-enable with new secrets
```

---

## Key Performance Indicators

### Integration Health Metrics

- **Payment Success Rate**: > 95%
- **Webhook Processing Time**: < 2 seconds
- **API Availability**: > 99.9%
- **Cost per Operation**: < targets
- **Fallback Activation Rate**: < 5%
- **Test Coverage**: > 80%

### Cost Targets

- **OpenAI per script**: < $0.50
- **TTS per video**: < $0.20
- **Total per video**: < $3.00
- **Daily API spend**: < $50.00

---

**Document Status**: Complete - Version 2.0  
**Implementation Priority**: Start with Stripe webhook setup  
**Next Review**: Weekly operational review every Friday  
**Support Contact**: Backend Team Lead or CTO for escalations