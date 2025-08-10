# Third-Party API Integration Handbook

**Document Version**: 1.0  
**For**: Integration Specialist  
**Classification**: CONFIDENTIAL - API KEYS SENSITIVE  
**Last Updated**: January 2025

---

## 🎯 Integration Overview

### Your API Arsenal
You'll integrate and optimize 7 critical third-party services that power YTEMPIRE's content generation pipeline. Each integration must be cost-optimized, reliable, and scalable.

### Cost Budget Distribution (Per Video)
```
Total Budget: $3.00 (Hard Limit)
├── OpenAI (Script): $0.20-0.40
├── TTS (Voice): $0.10-0.20
├── Stock Media: $0.00-0.10
├── Processing: $0.20-0.30
├── Storage/CDN: $0.10
└── Buffer: $1.90-2.40
```

---

## 🤖 OpenAI Integration

### API Configuration

```python
# OpenAI API Setup
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

class OpenAIIntegration:
    """Your OpenAI integration manager"""
    
    def __init__(self):
        self.api_key = os.environ['OPENAI_API_KEY']
        self.organization = os.environ.get('OPENAI_ORG_ID')
        openai.api_key = self.api_key
        
        # Cost tracking
        self.cost_per_1k_tokens = {
            'gpt-3.5-turbo': 0.002,
            'gpt-4': 0.03,
            'gpt-4-turbo': 0.01
        }
        
        # Default configuration
        self.default_config = {
            'model': 'gpt-3.5-turbo',  # Primary model
            'temperature': 0.7,
            'max_tokens': 2000,
            'top_p': 0.9,
            'frequency_penalty': 0.3,
            'presence_penalty': 0.3
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def generate_script(self, topic: str, style: str, length: int = 800) -> dict:
        """Generate video script with cost optimization"""
        
        # Check cache first
        cache_key = f"script:{hashlib.md5(f'{topic}:{style}'.encode()).hexdigest()}"
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Craft optimized prompt
        prompt = self.craft_prompt(topic, style, length)
        
        # Make API call
        start_time = time.time()
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.default_config['model'],
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.default_config['temperature'],
                max_tokens=self.default_config['max_tokens']
            )
            
            # Calculate cost
            tokens_used = response['usage']['total_tokens']
            cost = (tokens_used / 1000) * self.cost_per_1k_tokens[self.default_config['model']]
            
            # Process response
            script = response['choices'][0]['message']['content']
            
            result = {
                'script': script,
                'tokens_used': tokens_used,
                'cost': cost,
                'model': self.default_config['model'],
                'generation_time': time.time() - start_time
            }
            
            # Cache result
            await self.redis.set(cache_key, json.dumps(result), ex=3600)
            
            # Track cost
            await self.track_cost('openai', cost)
            
            return result
            
        except openai.error.RateLimitError:
            # Switch to fallback or wait
            await self.handle_rate_limit()
            raise
        except openai.error.APIError as e:
            # Log and retry
            await self.log_api_error(e)
            raise
    
    def craft_prompt(self, topic: str, style: str, length: int) -> str:
        """Craft cost-effective prompt"""
        
        return f"""Create a {length}-word YouTube video script about {topic}.
Style: {style}
Include:
- Engaging hook (first 5 seconds)
- 3-5 main points
- Call to action
- Natural speech patterns
Format: Natural paragraphs, no timestamps."""
    
    def get_system_prompt(self) -> str:
        """System prompt for consistency"""
        
        return """You are a YouTube content expert creating engaging scripts.
Focus on retention, clear value, and natural speech.
Avoid: Excessive adjectives, filler content, repetition.
Goal: Maximum engagement in minimum words."""

### Cost Optimization Strategies

```python
class OpenAICostOptimizer:
    """Reduce OpenAI costs by 60%"""
    
    def __init__(self):
        self.optimization_strategies = {
            'caching': 0.30,  # 30% reduction
            'prompt_optimization': 0.15,  # 15% reduction
            'model_selection': 0.10,  # 10% reduction
            'batching': 0.05  # 5% reduction
        }
    
    async def optimize_request(self, request: dict) -> dict:
        """Apply all optimization strategies"""
        
        # 1. Check cache (30% savings)
        cached = await self.check_cache(request)
        if cached:
            return cached
        
        # 2. Optimize prompt (15% savings)
        request['prompt'] = self.minimize_prompt(request['prompt'])
        
        # 3. Select cheapest viable model (10% savings)
        request['model'] = self.select_optimal_model(request)
        
        # 4. Batch if possible (5% savings)
        if self.can_batch(request):
            return await self.add_to_batch(request)
        
        return request
    
    def minimize_prompt(self, prompt: str) -> str:
        """Reduce prompt tokens without losing quality"""
        
        # Remove redundant instructions
        optimizations = [
            (r'\s+', ' '),  # Multiple spaces to single
            (r'Please\s+', ''),  # Remove politeness
            (r'Could you\s+', ''),  # Direct instructions
            (r'I would like\s+', ''),  # Be concise
        ]
        
        for pattern, replacement in optimizations:
            prompt = re.sub(pattern, replacement, prompt)
        
        return prompt.strip()
    
    def select_optimal_model(self, request: dict) -> str:
        """Choose cheapest model that meets requirements"""
        
        complexity = self.assess_complexity(request['prompt'])
        
        if complexity < 0.3:
            return 'gpt-3.5-turbo'  # $0.002/1k tokens
        elif complexity < 0.7:
            return 'gpt-3.5-turbo-16k'  # For longer context
        else:
            # Only use GPT-4 when absolutely necessary
            return 'gpt-4-turbo-preview'  # $0.01/1k tokens
```

---

## 🎤 Text-to-Speech Integration

### Primary: Google Cloud TTS

```python
from google.cloud import texttospeech
import asyncio

class GoogleTTSIntegration:
    """Primary TTS - lowest cost, good quality"""
    
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()
        
        # Voice configuration
        self.voice_config = {
            'primary': {
                'language_code': 'en-US',
                'name': 'en-US-Neural2-J',  # Male, energetic
                'speaking_rate': 1.0,
                'pitch': 0.0
            },
            'alternative': {
                'language_code': 'en-US',
                'name': 'en-US-Neural2-H',  # Female, professional
                'speaking_rate': 1.0,
                'pitch': 0.0
            }
        }
        
        # Cost: $0.016 per 1000 characters (Neural2)
        self.cost_per_1k_chars = 0.016
    
    async def synthesize_speech(self, text: str, voice_type: str = 'primary') -> dict:
        """Generate speech from text"""
        
        # Check cache for common phrases
        cache_key = f"tts:google:{hashlib.md5(text.encode()).hexdigest()}"
        cached = await self.redis.get(cache_key)
        if cached:
            return {'audio_path': cached, 'cost': 0, 'cached': True}
        
        # Prepare synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Configure voice
        voice_cfg = self.voice_config[voice_type]
        voice = texttospeech.VoiceSelectionParams(
            language_code=voice_cfg['language_code'],
            name=voice_cfg['name']
        )
        
        # Configure audio
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=voice_cfg['speaking_rate'],
            pitch=voice_cfg['pitch']
        )
        
        # Synthesize
        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Save audio
        audio_path = f"/tmp/audio_{uuid.uuid4()}.mp3"
        with open(audio_path, 'wb') as f:
            f.write(response.audio_content)
        
        # Calculate cost
        char_count = len(text)
        cost = (char_count / 1000) * self.cost_per_1k_chars
        
        # Cache if common phrase
        if self.is_common_phrase(text):
            await self.redis.set(cache_key, audio_path, ex=86400)
        
        return {
            'audio_path': audio_path,
            'duration': self.estimate_duration(text),
            'cost': cost,
            'chars': char_count,
            'service': 'google_tts'
        }
```

### Fallback: ElevenLabs

```python
import elevenlabs
from elevenlabs import Voice, VoiceSettings

class ElevenLabsIntegration:
    """Premium TTS fallback - better quality, higher cost"""
    
    def __init__(self):
        self.api_key = os.environ['ELEVENLABS_API_KEY']
        elevenlabs.set_api_key(self.api_key)
        
        # Voice IDs
        self.voices = {
            'adam': '21m00Tcm4TlvDq8ikWAM',
            'bella': 'EXAVITQu4vr4xnSDxMaL',
            'josh': 'TxGEqnHWrfWFTfGW9XjX'
        }
        
        # Cost: $0.30 per 1000 characters
        self.cost_per_1k_chars = 0.30
        self.monthly_quota = 100000  # characters
        
    async def synthesize_speech(self, text: str, voice: str = 'adam') -> dict:
        """Generate premium speech - use sparingly"""
        
        # Check quota
        usage = await self.check_monthly_usage()
        if usage > self.monthly_quota * 0.9:
            raise Exception("ElevenLabs quota nearly exhausted")
        
        # Generate audio
        audio = elevenlabs.generate(
            text=text,
            voice=Voice(
                voice_id=self.voices[voice],
                settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.0,
                    use_speaker_boost=True
                )
            ),
            model="eleven_monolingual_v1"
        )
        
        # Save audio
        audio_path = f"/tmp/audio_premium_{uuid.uuid4()}.mp3"
        elevenlabs.save(audio, audio_path)
        
        # Calculate cost
        char_count = len(text)
        cost = (char_count / 1000) * self.cost_per_1k_chars
        
        # Update usage
        await self.update_monthly_usage(char_count)
        
        return {
            'audio_path': audio_path,
            'cost': cost,
            'chars': char_count,
            'service': 'elevenlabs',
            'quality': 'premium'
        }
```

### TTS Optimization Strategy

```python
class TTSOptimizer:
    """Optimize TTS costs and quality"""
    
    def __init__(self):
        self.google_tts = GoogleTTSIntegration()
        self.elevenlabs = ElevenLabsIntegration()
        
        # Common phrases to pre-generate and cache
        self.common_phrases = [
            "Don't forget to like and subscribe",
            "Hit the notification bell",
            "Leave a comment below",
            "Thanks for watching",
            "Let's dive in",
            "Before we begin"
        ]
    
    async def select_tts_service(self, text: str, video_priority: int) -> str:
        """Choose TTS service based on priority and cost"""
        
        # High priority videos get better voice
        if video_priority >= 8:
            # Check ElevenLabs quota
            if await self.elevenlabs.check_quota():
                return 'elevenlabs'
        
        # Default to Google TTS
        return 'google_tts'
    
    async def pre_cache_common_phrases(self):
        """Pre-generate common phrases to save costs"""
        
        for phrase in self.common_phrases:
            # Generate with Google TTS (cheapest)
            result = await self.google_tts.synthesize_speech(phrase)
            
            # Store in permanent cache
            cache_key = f"tts:permanent:{hashlib.md5(phrase.encode()).hexdigest()}"
            await self.redis.set(cache_key, result['audio_path'])
            
            print(f"Cached: {phrase} - Saved ${result['cost']}")
```

---

## 💳 Stripe Payment Integration

### Payment Processing Setup

```python
import stripe
from typing import Optional

class StripeIntegration:
    """Handle all payment operations"""
    
    def __init__(self):
        self.api_key = os.environ['STRIPE_SECRET_KEY']
        stripe.api_key = self.api_key
        
        # Webhook endpoint secret
        self.webhook_secret = os.environ['STRIPE_WEBHOOK_SECRET']
        
        # Product/Price IDs
        self.price_ids = {
            'starter': 'price_1234567890starter',
            'growth': 'price_1234567890growth',
            'scale': 'price_1234567890scale'
        }
    
    async def create_checkout_session(self, plan: str, customer_email: str) -> str:
        """Create Stripe checkout session"""
        
        try:
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': self.price_ids[plan],
                    'quantity': 1
                }],
                mode='subscription',
                success_url='https://ytempire.com/success?session_id={CHECKOUT_SESSION_ID}',
                cancel_url='https://ytempire.com/cancel',
                customer_email=customer_email,
                metadata={
                    'plan': plan,
                    'source': 'web_checkout'
                }
            )
            
            return session.url
            
        except stripe.error.StripeError as e:
            await self.log_stripe_error(e)
            raise
    
    async def handle_webhook(self, payload: bytes, sig_header: str) -> dict:
        """Process Stripe webhooks"""
        
        try:
            # Verify webhook signature
            event = stripe.Webhook.construct_event(
                payload, sig_header, self.webhook_secret
            )
            
            # Handle different event types
            if event['type'] == 'checkout.session.completed':
                await self.handle_checkout_complete(event['data']['object'])
                
            elif event['type'] == 'invoice.payment_succeeded':
                await self.handle_payment_success(event['data']['object'])
                
            elif event['type'] == 'invoice.payment_failed':
                await self.handle_payment_failure(event['data']['object'])
                
            elif event['type'] == 'customer.subscription.deleted':
                await self.handle_subscription_cancelled(event['data']['object'])
            
            return {'status': 'success', 'event_type': event['type']}
            
        except ValueError:
            # Invalid payload
            raise Exception("Invalid webhook payload")
        except stripe.error.SignatureVerificationError:
            # Invalid signature
            raise Exception("Invalid webhook signature")
    
    async def handle_checkout_complete(self, session: dict):
        """Process successful checkout"""
        
        customer_email = session['customer_email']
        subscription_id = session['subscription']
        plan = session['metadata']['plan']
        
        # Create user account
        user = await self.create_user_account(
            email=customer_email,
            subscription_id=subscription_id,
            plan=plan
        )
        
        # Send welcome email
        await self.send_welcome_email(user)
        
        # Initialize YouTube channels
        await self.initialize_channels(user)
    
    async def update_payment_method(self, customer_id: str, payment_method_id: str):
        """Update customer's payment method"""
        
        try:
            # Attach payment method to customer
            stripe.PaymentMethod.attach(
                payment_method_id,
                customer=customer_id
            )
            
            # Set as default
            stripe.Customer.modify(
                customer_id,
                invoice_settings={
                    'default_payment_method': payment_method_id
                }
            )
            
            return {'status': 'success'}
            
        except stripe.error.StripeError as e:
            await self.log_stripe_error(e)
            raise
```

---

## 🎬 Stock Media APIs

### Pexels Integration

```python
import aiohttp

class PexelsIntegration:
    """Free stock videos and images"""
    
    def __init__(self):
        self.api_key = os.environ['PEXELS_API_KEY']
        self.base_url = 'https://api.pexels.com/v1'
        self.video_url = 'https://api.pexels.com/videos'
        
        # Rate limiting
        self.rate_limit = 200  # requests per hour
        self.request_count = 0
        self.reset_time = None
    
    async def search_videos(self, query: str, per_page: int = 10) -> list:
        """Search for stock videos"""
        
        # Check rate limit
        if not await self.check_rate_limit():
            # Fallback to cached videos
            return await self.get_cached_videos(query)
        
        headers = {'Authorization': self.api_key}
        params = {
            'query': query,
            'per_page': per_page,
            'orientation': 'landscape'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f'{self.video_url}/search',
                headers=headers,
                params=params
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Process videos
                    videos = []
                    for video in data['videos']:
                        processed = {
                            'id': video['id'],
                            'url': video['video_files'][0]['link'],
                            'duration': video['duration'],
                            'width': video['width'],
                            'height': video['height'],
                            'thumbnail': video['image']
                        }
                        videos.append(processed)
                    
                    # Cache results
                    await self.cache_search_results(query, videos)
                    
                    return videos
                    
                elif response.status == 429:
                    # Rate limited
                    return await self.get_cached_videos(query)
                    
                else:
                    raise Exception(f"Pexels API error: {response.status}")
    
    async def download_video(self, video_url: str, video_id: str) -> str:
        """Download video to local storage"""
        
        local_path = f"/tmp/stock_video_{video_id}.mp4"
        
        # Check if already downloaded
        if os.path.exists(local_path):
            return local_path
        
        async with aiohttp.ClientSession() as session:
            async with session.get(video_url) as response:
                if response.status == 200:
                    with open(local_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    
                    return local_path
                else:
                    raise Exception(f"Download failed: {response.status}")
```

### Pixabay Integration

```python
class PixabayIntegration:
    """Alternative free stock media"""
    
    def __init__(self):
        self.api_key = os.environ['PIXABAY_API_KEY']
        self.base_url = 'https://pixabay.com/api'
        
        # Higher rate limit than Pexels
        self.rate_limit = 5000  # requests per hour
    
    async def search_videos(self, query: str, min_duration: int = 5) -> list:
        """Search Pixabay for videos"""
        
        params = {
            'key': self.api_key,
            'q': query,
            'video_type': 'all',
            'min_duration': min_duration,
            'per_page': 20
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f'{self.base_url}/videos/',
                params=params
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    videos = []
                    for hit in data['hits']:
                        # Get medium quality (balance between quality and size)
                        video_url = hit['videos']['medium']['url']
                        
                        processed = {
                            'id': hit['id'],
                            'url': video_url,
                            'duration': hit['duration'],
                            'tags': hit['tags'],
                            'downloads': hit['downloads']
                        }
                        videos.append(processed)
                    
                    return videos
                else:
                    raise Exception(f"Pixabay API error: {response.status}")
```

---

## 🔄 Circuit Breaker Implementation

### Universal Circuit Breaker

```python
from enum import Enum
from datetime import datetime, timedelta
import asyncio

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Prevent cascading failures in API calls"""
    
    def __init__(self, service_name: str, failure_threshold: int = 5, 
                 recovery_timeout: int = 60, expected_exception: type = Exception):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception(f"Circuit breaker OPEN for {self.service_name}")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to retry"""
        
        return (
            self.last_failure_time and
            datetime.now() >= self.last_failure_time + timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        """Reset failure count on success"""
        
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Increment failure count and possibly open circuit"""
        
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            print(f"Circuit breaker OPENED for {self.service_name}")
```

### Service-Specific Circuit Breakers

```python
class APICircuitBreakers:
    """Manage circuit breakers for all services"""
    
    def __init__(self):
        self.breakers = {
            'openai': CircuitBreaker('openai', failure_threshold=3, recovery_timeout=30),
            'google_tts': CircuitBreaker('google_tts', failure_threshold=5, recovery_timeout=60),
            'elevenlabs': CircuitBreaker('elevenlabs', failure_threshold=2, recovery_timeout=120),
            'youtube': CircuitBreaker('youtube', failure_threshold=5, recovery_timeout=60),
            'stripe': CircuitBreaker('stripe', failure_threshold=2, recovery_timeout=300),
            'pexels': CircuitBreaker('pexels', failure_threshold=10, recovery_timeout=3600),
            'pixabay': CircuitBreaker('pixabay', failure_threshold=10, recovery_timeout=1800)
        }
    
    async def execute_with_breaker(self, service: str, func, *args, **kwargs):
        """Execute API call with appropriate circuit breaker"""
        
        if service not in self.breakers:
            # No circuit breaker, execute directly
            return await func(*args, **kwargs)
        
        breaker = self.breakers[service]
        return await breaker.call(func, *args, **kwargs)
    
    def get_status(self) -> dict:
        """Get status of all circuit breakers"""
        
        return {
            service: {
                'state': breaker.state.value,
                'failure_count': breaker.failure_count,
                'last_failure': breaker.last_failure_time
            }
            for service, breaker in self.breakers.items()
        }
```

---

## 📊 Cost Tracking & Optimization

### Unified Cost Tracker

```python
class UnifiedCostTracker:
    """Track costs across all APIs"""
    
    def __init__(self):
        self.redis = Redis()
        self.cost_limits = {
            'per_video_warning': 2.50,
            'per_video_critical': 3.00,
            'daily_limit': 150.00
        }
    
    async def track_api_cost(self, service: str, operation: str, cost: float, 
                             video_id: Optional[str] = None):
        """Track cost for any API operation"""
        
        # Track by video if applicable
        if video_id:
            video_key = f"cost:video:{video_id}"
            await self.redis.hincrbyfloat(video_key, service, cost)
            
            # Check video cost threshold
            total = await self.get_video_total_cost(video_id)
            if total > self.cost_limits['per_video_warning']:
                await self.alert_cost_warning(video_id, total)
            if total > self.cost_limits['per_video_critical']:
                await self.stop_video_processing(video_id, total)
        
        # Track daily total
        daily_key = f"cost:daily:{datetime.now().strftime('%Y%m%d')}"
        await self.redis.hincrbyfloat(daily_key, service, cost)
        
        # Track by service
        service_key = f"cost:service:{service}:{datetime.now().strftime('%Y%m')}"
        await self.redis.incrbyfloat(service_key, cost)
        
        # Log to database
        await self.log_cost_to_db(service, operation, cost, video_id)
    
    async def get_cost_breakdown(self, video_id: str) -> dict:
        """Get detailed cost breakdown for a video"""
        
        video_key = f"cost:video:{video_id}"
        costs = await self.redis.hgetall(video_key)
        
        breakdown = {
            'openai': float(costs.get('openai', 0)),
            'google_tts': float(costs.get('google_tts', 0)),
            'elevenlabs': float(costs.get('elevenlabs', 0)),
            'processing': float(costs.get('processing', 0)),
            'storage': float(costs.get('storage', 0))
        }
        
        breakdown['total'] = sum(breakdown.values())
        breakdown['status'] = self.get_cost_status(breakdown['total'])
        
        return breakdown
    
    def get_cost_status(self, total: float) -> str:
        """Determine cost status"""
        
        if total <= 1.00:
            return 'excellent'
        elif total <= 2.00:
            return 'good'
        elif total <= 2.50:
            return 'warning'
        elif total <= 3.00:
            return 'critical'
        else:
            return 'exceeded'
```

---

## 🚨 Emergency Procedures

### API Failure Recovery

```python
class APIFailureRecovery:
    """Handle API failures gracefully"""
    
    def __init__(self):
        self.fallback_chains = {
            'openai': ['gpt-3.5-turbo', 'cached_content', 'template'],
            'tts': ['google_tts', 'elevenlabs', 'cached_audio'],
            'stock_media': ['pexels', 'pixabay', 'cached_media', 'generated']
        }
    
    async def execute_with_fallback(self, primary_func, fallback_chain: list, 
                                   *args, **kwargs):
        """Execute with automatic fallback"""
        
        errors = []
        
        # Try primary
        try:
            return await primary_func(*args, **kwargs)
        except Exception as e:
            errors.append(f"Primary failed: {e}")
        
        # Try fallbacks
        for fallback in fallback_chain:
            try:
                fallback_func = self.get_fallback_function(fallback)
                result = await fallback_func(*args, **kwargs)
                
                # Log fallback usage
                await self.log_fallback_usage(primary_func.__name__, fallback)
                
                return result
                
            except Exception as e:
                errors.append(f"{fallback} failed: {e}")
                continue
        
        # All failed
        raise Exception(f"All fallbacks exhausted: {errors}")
```

---

## 🔑 Security & Credentials

### API Key Management

```yaml
api_keys_storage:
  development:
    location: ".env file"
    encryption: "None (local only)"
    rotation: "Manual"
  
  production:
    location: "Environment variables"
    encryption: "At rest"
    rotation: "Monthly automated"
    backup: "Secure vault"

key_rotation_schedule:
  openai: "Monthly"
  google_cloud: "Quarterly"
  elevenlabs: "Monthly"
  stripe: "Annually"
  pexels: "Annually"
  pixabay: "Annually"
```

### Security Best Practices

```python
class APISecurityManager:
    """Manage API security"""
    
    def __init__(self):
        self.key_rotation_schedule = {
            'openai': 30,  # days
            'google_cloud': 90,
            'elevenlabs': 30,
            'stripe': 365
        }
    
    async def check_key_rotation(self):
        """Check if any keys need rotation"""
        
        for service, days in self.key_rotation_schedule.items():
            last_rotation = await self.get_last_rotation(service)
            
            if datetime.now() - last_rotation > timedelta(days=days):
                await self.alert_key_rotation_needed(service)
    
    def sanitize_api_response(self, response: dict) -> dict:
        """Remove sensitive data from API responses"""
        
        sensitive_fields = [
            'api_key', 'secret', 'token', 'password',
            'client_secret', 'webhook_secret'
        ]
        
        sanitized = response.copy()
        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = '***REDACTED***'
        
        return sanitized
```

---

## 📈 Performance Monitoring

### API Performance Metrics

```python
class APIPerformanceMonitor:
    """Monitor API performance and reliability"""
    
    def __init__(self):
        self.metrics = {
            'response_times': defaultdict(list),
            'error_rates': defaultdict(int),
            'success_rates': defaultdict(int),
            'costs': defaultdict(float)
        }
    
    async def track_api_call(self, service: str, operation: str, 
                            duration: float, success: bool, cost: float = 0):
        """Track API call metrics"""
        
        key = f"{service}:{operation}"
        
        # Track response time
        self.metrics['response_times'][key].append(duration)
        
        # Track success/error
        if success:
            self.metrics['success_rates'][key] += 1
        else:
            self.metrics['error_rates'][key] += 1
        
        # Track cost
        self.metrics['costs'][key] += cost
        
        # Calculate percentiles
        if len(self.metrics['response_times'][key]) >= 100:
            await self.calculate_and_store_percentiles(key)
    
    def get_api_health_score(self, service: str) -> float:
        """Calculate health score for an API service"""
        
        error_rate = self.get_error_rate(service)
        avg_response_time = self.get_avg_response_time(service)
        
        # Health score formula
        health = 100
        health -= error_rate * 100  # Each 1% error = -1 point
        health -= max(0, (avg_response_time - 1000) / 100)  # Over 1s = penalties
        
        return max(0, min(100, health))
```

---

## 🚀 Quick Implementation Guide

### Week 1 Checklist

```yaml
day_1:
  - [ ] Set up all API credentials
  - [ ] Test each API with simple call
  - [ ] Implement circuit breakers
  - [ ] Set up cost tracking

day_2:
  - [ ] OpenAI script generation working
  - [ ] Google TTS synthesis working
  - [ ] Cost tracking per operation
  - [ ] Cache implementation

day_3:
  - [ ] Stripe webhook handler
  - [ ] Payment processing test
  - [ ] Stock media search working
  - [ ] Download and cache media

day_4:
  - [ ] Fallback chains implemented
  - [ ] Error handling complete
  - [ ] Performance monitoring active
  - [ ] API health dashboard

day_5:
  - [ ] Full integration test
  - [ ] Cost optimization verified
  - [ ] All circuit breakers tested
  - [ ] Documentation complete
```

---

**Remember**: Every API call costs money. Every optimization saves money. Every fallback prevents failure. Master these integrations, and you master YTEMPIRE's efficiency!