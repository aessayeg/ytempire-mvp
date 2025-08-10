# 8. EXTERNAL INTEGRATIONS - YTEMPIRE

**Version 2.0 | January 2025**  
**Document Status: Consolidated & Standardized**  
**Last Updated: January 2025**

---

## 8.1 YouTube API Integration

### Account Management Strategy

```yaml
Total Accounts: 15
Active Accounts: 10 (accounts 1-10)
Reserve Accounts: 5 (accounts 11-15)

Quota Management:
  Daily Limit: 10,000 units per account
  Upload Cost: ~1,600 units per video
  Safe Limit: 9,000 units (90% threshold)
  Videos per Account: 5 per day maximum

Rotation Strategy:
  Primary: Round-robin across active accounts
  Fallback: Use reserve accounts when active exhausted
  Recovery: 24-hour cooldown for exhausted accounts
  Health Monitoring: Track consecutive failures
```

### OAuth2 Setup

```python
# YouTube OAuth2 Configuration
YOUTUBE_OAUTH_CONFIG = {
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "redirect_uri": "http://localhost:8000/auth/youtube/callback",
    "scope": [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube",
        "https://www.googleapis.com/auth/youtube.force-ssl"
    ]
}

# Account credential storage structure
YOUTUBE_ACCOUNT = {
    "account_id": 1,
    "email": "ytempire.account1@gmail.com",
    "client_id": "...",
    "client_secret": "...",
    "refresh_token": "...",
    "access_token": "...",
    "token_expiry": "2025-01-15T10:30:00Z"
}
```

### YouTube API Client

```python
# app/services/youtube_client.py
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class YouTubeClient:
    def __init__(self, account_id: int):
        self.account_id = account_id
        self.credentials = self._load_credentials()
        self.service = self._build_service()
        self.quota_used = 0
        self.last_reset = datetime.utcnow().date()
    
    def _load_credentials(self) -> Credentials:
        """Load and refresh credentials if needed"""
        creds_data = self._get_account_credentials(self.account_id)
        
        creds = Credentials(
            token=creds_data['access_token'],
            refresh_token=creds_data['refresh_token'],
            token_uri=YOUTUBE_OAUTH_CONFIG['token_uri'],
            client_id=creds_data['client_id'],
            client_secret=creds_data['client_secret'],
            scopes=YOUTUBE_OAUTH_CONFIG['scope']
        )
        
        # Refresh if expired
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            self._save_credentials(creds)
        
        return creds
    
    def _build_service(self):
        """Build YouTube API service"""
        return build('youtube', 'v3', credentials=self.credentials)
    
    def upload_video(self, video_data: Dict) -> Dict:
        """Upload video with retry logic"""
        try:
            # Check quota
            if self.quota_used >= 9000:
                raise Exception("Quota limit reached for account")
            
            # Prepare video metadata
            body = {
                'snippet': {
                    'title': video_data['title'],
                    'description': video_data['description'],
                    'tags': video_data.get('tags', []),
                    'categoryId': video_data.get('category_id', '22'),
                    'defaultLanguage': 'en',
                    'defaultAudioLanguage': 'en'
                },
                'status': {
                    'privacyStatus': video_data.get('privacy', 'public'),
                    'selfDeclaredMadeForKids': False,
                    'embeddable': True,
                    'publicStatsViewable': True
                }
            }
            
            # Upload video
            media = MediaFileUpload(
                video_data['file_path'],
                mimetype='video/mp4',
                resumable=True,
                chunksize=50 * 1024 * 1024  # 50MB chunks
            )
            
            request = self.service.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            # Execute upload with progress tracking
            response = self._execute_resumable_upload(request)
            
            # Update quota tracking
            self.quota_used += 1600
            
            return {
                'youtube_id': response['id'],
                'youtube_url': f"https://youtube.com/watch?v={response['id']}",
                'published_at': response['snippet']['publishedAt']
            }
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            if e.resp.status == 403:
                # Quota exceeded or permission issue
                raise Exception(f"YouTube API quota exceeded for account {self.account_id}")
            raise
    
    def get_channel_analytics(self, channel_id: str, start_date: str, end_date: str) -> Dict:
        """Get channel analytics using YouTube Analytics API"""
        analytics = build('youtubeAnalytics', 'v2', credentials=self.credentials)
        
        try:
            response = analytics.reports().query(
                ids=f"channel=={channel_id}",
                startDate=start_date,
                endDate=end_date,
                metrics='views,estimatedMinutesWatched,averageViewDuration,subscribersGained,likes,shares,comments',
                dimensions='day'
            ).execute()
            
            return response
            
        except HttpError as e:
            logger.error(f"Analytics API error: {e}")
            return {}
    
    def update_video_metadata(self, video_id: str, updates: Dict) -> Dict:
        """Update video title, description, tags"""
        try:
            # Get current video data
            video = self.service.videos().list(
                part='snippet',
                id=video_id
            ).execute()
            
            if not video['items']:
                raise Exception(f"Video {video_id} not found")
            
            # Update snippet
            snippet = video['items'][0]['snippet']
            snippet.update(updates)
            
            # Update video
            response = self.service.videos().update(
                part='snippet',
                body={
                    'id': video_id,
                    'snippet': snippet
                }
            ).execute()
            
            self.quota_used += 50  # Update costs ~50 units
            
            return response
            
        except HttpError as e:
            logger.error(f"Update error: {e}")
            raise
```

---

## 8.2 AI Services (OpenAI, ElevenLabs)

### OpenAI Integration

```python
# app/services/ai/openai_service.py
import openai
from typing import Dict, List, Optional
from decimal import Decimal
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken

class OpenAIService:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.models = {
            "premium": "gpt-4-turbo-preview",
            "standard": "gpt-3.5-turbo",
            "fallback": "gpt-3.5-turbo-16k"
        }
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_script(self, 
                            topic: str, 
                            style: str, 
                            duration: int,
                            quality: str = "standard") -> Dict:
        """Generate video script with intelligent prompting"""
        
        model = self.models[quality]
        
        # Build sophisticated prompt
        system_prompt = """You are an expert YouTube scriptwriter who creates engaging, 
        retention-optimized scripts. Your scripts hook viewers in the first 5 seconds, 
        maintain engagement throughout, and drive action."""
        
        user_prompt = f"""
        Create a YouTube video script with these specifications:
        
        Topic: {topic}
        Style: {style}
        Duration: {duration} seconds (approximately {duration // 60} minutes)
        
        Structure Requirements:
        1. HOOK (0-5 seconds): Compelling question or statement
        2. INTRO (5-15 seconds): What viewer will learn/gain
        3. MAIN CONTENT: 
           - 3-5 key points
           - Examples and stories
           - Visual cues for editing
        4. CALL TO ACTION: Subscribe reminder mid-video
        5. CONCLUSION: Summary and end screen setup
        
        Tone: Conversational, energetic, authentic
        Pace: Dynamic with variation
        
        Include:
        - [VISUAL] tags for suggested visuals
        - [PAUSE] for dramatic effect
        - Natural speech patterns
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=4000,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.3
            )
            
            script = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            cost = self._calculate_cost(model, tokens_used)
            
            # Post-process script
            processed_script = self._process_script(script)
            
            return {
                "script": processed_script,
                "tokens": tokens_used,
                "cost": cost,
                "model": model,
                "word_count": len(processed_script.split()),
                "estimated_duration": self._estimate_duration(processed_script)
            }
            
        except openai.error.RateLimitError:
            # Fallback to cheaper model
            if quality != "fallback":
                return await self.generate_script(topic, style, duration, "fallback")
            raise
    
    async def improve_script(self, script: str, feedback: str) -> Dict:
        """Improve existing script based on feedback"""
        prompt = f"""
        Improve this YouTube script based on the feedback:
        
        Original Script:
        {script}
        
        Feedback:
        {feedback}
        
        Maintain the same structure and duration, but address the feedback points.
        """
        
        response = await openai.ChatCompletion.acreate(
            model=self.models["standard"],
            messages=[
                {"role": "system", "content": "You are an expert script editor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        return {
            "improved_script": response.choices[0].message.content,
            "cost": self._calculate_cost(self.models["standard"], response.usage.total_tokens)
        }
    
    def _calculate_cost(self, model: str, tokens: int) -> Decimal:
        """Calculate cost based on token usage"""
        rates = {
            "gpt-4-turbo-preview": 0.03,  # per 1K tokens
            "gpt-3.5-turbo": 0.002,
            "gpt-3.5-turbo-16k": 0.003
        }
        
        rate = rates.get(model, 0.002)
        return Decimal(str(tokens * rate / 1000))
    
    def _process_script(self, script: str) -> str:
        """Process script for better TTS compatibility"""
        # Remove excessive punctuation
        script = script.replace("...", " ")
        script = script.replace("!!", "!")
        
        # Standardize visual cues
        import re
        script = re.sub(r'\[([^\]]+)\]', r'[VISUAL: \1]', script)
        
        return script
    
    def _estimate_duration(self, script: str) -> int:
        """Estimate speaking duration in seconds"""
        # Average speaking rate: 150 words per minute
        words = len(script.split())
        return int(words / 150 * 60)
```

### ElevenLabs Integration

```python
# app/services/ai/elevenlabs_service.py
import aiohttp
import asyncio
from typing import Dict, Optional
import base64
from decimal import Decimal

class ElevenLabsService:
    def __init__(self):
        self.api_key = settings.ELEVENLABS_API_KEY
        self.base_url = "https://api.elevenlabs.io/v1"
        self.voices = {
            "professional": "21m00Tcm4TlvDq8ikWAM",  # Rachel
            "casual": "AZnzlk1XvdvUeBnXmlld",        # Domi  
            "energetic": "EXAVITQu4vr4xnSDxMaL",     # Bella
            "calm": "ErXwobaYiN019PkySvjV"           # Antoni
        }
    
    async def text_to_speech(self, 
                            text: str, 
                            voice_style: str = "professional",
                            optimize_streaming: bool = False) -> Dict:
        """Convert text to speech with voice selection"""
        
        voice_id = self.voices.get(voice_style, self.voices["professional"])
        
        # Split long texts into chunks for better processing
        chunks = self._split_text(text, max_chars=5000)
        audio_parts = []
        total_cost = Decimal("0")
        
        async with aiohttp.ClientSession() as session:
            for chunk in chunks:
                url = f"{self.base_url}/text-to-speech/{voice_id}"
                
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": self.api_key
                }
                
                data = {
                    "text": chunk,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.4,
                        "use_speaker_boost": True
                    }
                }
                
                if optimize_streaming:
                    data["optimize_streaming_latency"] = 3
                
                async with session.post(url, json=data, headers=headers) as response:
                    if response.status == 200:
                        audio_data = await response.read()
                        audio_parts.append(audio_data)
                        
                        # Calculate cost (approximately $0.30 per 1000 characters)
                        char_count = len(chunk)
                        cost = Decimal(str(char_count * 0.0003))
                        total_cost += cost
                    else:
                        error = await response.text()
                        raise Exception(f"ElevenLabs API error: {error}")
        
        # Combine audio parts
        combined_audio = b''.join(audio_parts)
        
        # Save to file
        file_path = f"/storage/audio/{uuid4()}.mp3"
        with open(file_path, 'wb') as f:
            f.write(combined_audio)
        
        # Get duration
        duration = self._get_audio_duration(file_path)
        
        return {
            "audio_path": file_path,
            "duration_seconds": duration,
            "cost": total_cost,
            "voice_used": voice_style,
            "character_count": sum(len(chunk) for chunk in chunks)
        }
    
    async def get_voice_settings(self, voice_id: str) -> Dict:
        """Get current voice settings"""
        url = f"{self.base_url}/voices/{voice_id}/settings"
        
        async with aiohttp.ClientSession() as session:
            headers = {"xi-api-key": self.api_key}
            async with session.get(url, headers=headers) as response:
                return await response.json()
    
    def _split_text(self, text: str, max_chars: int = 5000) -> List[str]:
        """Split text into chunks at sentence boundaries"""
        chunks = []
        current_chunk = ""
        
        sentences = text.split('. ')
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chars:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_audio_duration(self, file_path: str) -> int:
        """Get audio duration in seconds"""
        from mutagen.mp3 import MP3
        audio = MP3(file_path)
        return int(audio.info.length)
```

---

## 8.3 Payment Processing (Stripe)

### Stripe Configuration

```python
# app/services/stripe_service.py
import stripe
from typing import Dict, Optional, List
from decimal import Decimal
from datetime import datetime

stripe.api_key = settings.STRIPE_API_KEY

class StripeService:
    def __init__(self):
        self.products = {
            "starter": "prod_starter123",
            "growth": "prod_growth456",
            "scale": "prod_scale789"
        }
        
        self.prices = {
            "starter_monthly": "price_starter_monthly",
            "starter_yearly": "price_starter_yearly",
            "growth_monthly": "price_growth_monthly",
            "growth_yearly": "price_growth_yearly",
            "scale_monthly": "price_scale_monthly",
            "scale_yearly": "price_scale_yearly"
        }
    
    async def create_customer(self, user_data: Dict) -> Dict:
        """Create Stripe customer for new user"""
        try:
            customer = stripe.Customer.create(
                email=user_data['email'],
                name=user_data.get('full_name'),
                metadata={
                    'user_id': str(user_data['id']),
                    'username': user_data['username']
                }
            )
            
            return {
                'customer_id': customer.id,
                'created': customer.created
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe customer creation failed: {e}")
            raise
    
    async def create_subscription(self, 
                                customer_id: str, 
                                tier: str,
                                billing_period: str = "monthly") -> Dict:
        """Create subscription for customer"""
        price_id = self.prices[f"{tier}_{billing_period}"]
        
        try:
            subscription = stripe.Subscription.create(
                customer=customer_id,
                items=[{'price': price_id}],
                payment_behavior='default_incomplete',
                expand=['latest_invoice.payment_intent'],
                metadata={
                    'tier': tier,
                    'billing_period': billing_period
                }
            )
            
            return {
                'subscription_id': subscription.id,
                'status': subscription.status,
                'client_secret': subscription.latest_invoice.payment_intent.client_secret,
                'current_period_end': subscription.current_period_end
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Subscription creation failed: {e}")
            raise
    
    async def cancel_subscription(self, subscription_id: str, immediate: bool = False) -> Dict:
        """Cancel subscription"""
        try:
            if immediate:
                subscription = stripe.Subscription.delete(subscription_id)
            else:
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True
                )
            
            return {
                'status': 'cancelled' if immediate else 'pending_cancellation',
                'cancel_at': subscription.cancel_at
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Subscription cancellation failed: {e}")
            raise
    
    async def create_payment_intent(self, amount: int, currency: str = "usd") -> Dict:
        """Create payment intent for one-time payment"""
        try:
            intent = stripe.PaymentIntent.create(
                amount=amount,  # Amount in cents
                currency=currency,
                automatic_payment_methods={'enabled': True}
            )
            
            return {
                'payment_intent_id': intent.id,
                'client_secret': intent.client_secret,
                'amount': intent.amount
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Payment intent creation failed: {e}")
            raise
    
    async def get_invoices(self, customer_id: str, limit: int = 10) -> List[Dict]:
        """Get customer invoices"""
        try:
            invoices = stripe.Invoice.list(
                customer=customer_id,
                limit=limit
            )
            
            return [{
                'id': invoice.id,
                'amount': invoice.amount_paid / 100,  # Convert to dollars
                'status': invoice.status,
                'date': datetime.fromtimestamp(invoice.created),
                'pdf_url': invoice.hosted_invoice_url
            } for invoice in invoices]
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to fetch invoices: {e}")
            return []
```

---

## 8.4 N8N Workflow Integration

### N8N Workflow Configuration

```yaml
# N8N Workflow Definitions
workflows:
  video_generation:
    trigger: webhook
    nodes:
      - receive_request
      - validate_input
      - queue_generation
      - generate_script
      - generate_voice
      - create_video
      - quality_check
      - upload_youtube
      - notify_completion
    
  quality_monitoring:
    trigger: cron (every hour)
    nodes:
      - fetch_pending_videos
      - check_quality_scores
      - flag_low_quality
      - trigger_regeneration
    
  cost_optimization:
    trigger: webhook
    nodes:
      - analyze_costs
      - identify_savings
      - switch_providers
      - update_configuration
```

### N8N Integration Service

```python
# app/services/n8n_service.py
import aiohttp
from typing import Dict, List, Optional
import json

class N8NService:
    def __init__(self):
        self.base_url = "http://localhost:5678"
        self.auth = aiohttp.BasicAuth('admin', 'password')
        self.workflows = {
            "video_generation": "workflow_001",
            "quality_check": "workflow_002",
            "youtube_upload": "workflow_003",
            "cost_analysis": "workflow_004"
        }
    
    async def trigger_workflow(self, 
                              workflow_name: str, 
                              data: Dict) -> Dict:
        """Trigger N8N workflow via webhook"""
        workflow_id = self.workflows.get(workflow_name)
        if not workflow_id:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        url = f"{self.base_url}/webhook/{workflow_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, auth=self.auth) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'workflow_id': workflow_id,
                        'execution_id': result.get('executionId'),
                        'data': result
                    }
                else:
                    error = await response.text()
                    raise Exception(f"N8N workflow trigger failed: {error}")
    
    async def get_execution_status(self, execution_id: str) -> Dict:
        """Get workflow execution status"""
        url = f"{self.base_url}/executions/{execution_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, auth=self.auth) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {'status': 'unknown'}
    
    async def setup_video_generation_workflow(self, video_data: Dict) -> Dict:
        """Setup complete video generation workflow"""
        # Trigger main workflow
        result = await self.trigger_workflow('video_generation', {
            'video_id': video_data['id'],
            'topic': video_data['topic'],
            'style': video_data['style'],
            'duration': video_data['duration_target'],
            'channel_id': video_data['channel_id']
        })
        
        return result
```

---

## 8.5 Third-party APIs

### Stock Media APIs

```python
# app/services/media/stock_service.py
class StockMediaService:
    def __init__(self):
        self.providers = {
            'pexels': PexelsClient(),
            'pixabay': PixabayClient(),
            'unsplash': UnsplashClient()
        }
    
    async def search_videos(self, query: str, count: int = 5) -> List[Dict]:
        """Search for stock videos across providers"""
        results = []
        
        for provider_name, client in self.providers.items():
            try:
                videos = await client.search_videos(query, count)
                results.extend(videos)
            except Exception as e:
                logger.error(f"Error searching {provider_name}: {e}")
        
        # Sort by relevance and quality
        results.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return results[:count]
    
    async def download_video(self, url: str, video_id: str) -> str:
        """Download stock video"""
        file_path = f"/storage/stock/{video_id}_{uuid4()}.mp4"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                with open(file_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
        
        return file_path
```

### Email Service (SendGrid)

```python
# app/services/email_service.py
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content

class EmailService:
    def __init__(self):
        self.sg = sendgrid.SendGridAPIClient(api_key=settings.SENDGRID_API_KEY)
        self.from_email = Email("noreply@ytempire.com", "YTEMPIRE")
    
    async def send_video_completed(self, user_email: str, video_data: Dict):
        """Send video completion notification"""
        to_email = To(user_email)
        subject = f"Your video '{video_data['title']}' is ready!"
        
        content = Content(
            "text/html",
            f"""
            <h2>Video Generation Complete!</h2>
            <p>Your video has been successfully generated and uploaded to YouTube.</p>
            
            <h3>Video Details:</h3>
            <ul>
                <li><strong>Title:</strong> {video_data['title']}</li>
                <li><strong>Duration:</strong> {video_data['duration']} seconds</li>
                <li><strong>Quality Score:</strong> {video_data['quality_score']}/1.0</li>
                <li><strong>Cost:</strong> ${video_data['cost']}</li>
                <li><strong>YouTube URL:</strong> <a href="{video_data['youtube_url']}">{video_data['youtube_url']}</a></li>
            </ul>
            
            <p>Log in to your dashboard to view analytics and manage your video.</p>
            """
        )
        
        mail = Mail(self.from_email, to_email, subject, content)
        
        try:
            response = self.sg.client.mail.send.post(request_body=mail.get())
            return {'success': True, 'status_code': response.status_code}
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return {'success': False, 'error': str(e)}
```

### Analytics Services

```python
# app/services/analytics/mixpanel_service.py
from mixpanel import Mixpanel

class AnalyticsService:
    def __init__(self):
        self.mp = Mixpanel(settings.MIXPANEL_TOKEN)
    
    def track_video_generated(self, user_id: str, video_data: Dict):
        """Track video generation event"""
        self.mp.track(user_id, 'Video Generated', {
            'video_id': video_data['id'],
            'channel_id': video_data['channel_id'],
            'cost': float(video_data['cost']),
            'duration': video_data['duration'],
            'quality_score': video_data['quality_score'],
            'style': video_data['style']
        })
    
    def track_revenue(self, user_id: str, amount: float, source: str):
        """Track revenue event"""
        self.mp.track(user_id, 'Revenue Generated', {
            'amount': amount,
            'source': source,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Update user profile
        self.mp.people_increment(user_id, {
            'total_revenue': amount,
            'videos_monetized': 1
        })
```

---

## Integration Best Practices

### Rate Limiting Strategy

```python
# app/core/rate_limiter.py
from typing import Dict
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self):
        self.limits = {
            'openai': {'calls': 10000, 'period': 60},  # per minute
            'elevenlabs': {'calls': 100, 'period': 60},
            'youtube': {'calls': 100, 'period': 60},
            'stripe': {'calls': 100, 'period': 1}
        }
        self.calls = defaultdict(list)
    
    async def check_rate_limit(self, service: str) -> bool:
        """Check if rate limit allows call"""
        if service not in self.limits:
            return True
        
        limit = self.limits[service]
        now = time.time()
        
        # Clean old calls
        self.calls[service] = [
            call_time for call_time in self.calls[service]
            if now - call_time < limit['period']
        ]
        
        # Check limit
        if len(self.calls[service]) >= limit['calls']:
            return False
        
        # Record call
        self.calls[service].append(now)
        return True
    
    async def wait_if_needed(self, service: str):
        """Wait if rate limit exceeded"""
        while not await self.check_rate_limit(service):
            await asyncio.sleep(1)
```

### Error Recovery

```python
# app/core/recovery.py
class IntegrationRecovery:
    def __init__(self):
        self.fallback_services = {
            'openai': ['anthropic', 'cohere'],
            'elevenlabs': ['azure_tts', 'google_tts'],
            'stability': ['dalle', 'midjourney']
        }
    
    async def with_fallback(self, primary_service: str, operation, *args, **kwargs):
        """Execute operation with fallback services"""
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            logger.error(f"{primary_service} failed: {e}")
            
            # Try fallback services
            for fallback in self.fallback_services.get(primary_service, []):
                try:
                    fallback_operation = getattr(self, f"{fallback}_operation")
                    return await fallback_operation(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback {fallback} failed: {fallback_error}")
            
            raise Exception(f"All services failed for operation")
```

---

## Document Control

- **Version**: 2.0
- **Last Updated**: January 2025
- **Next Review**: February 2025
- **Owner**: API Development Engineer
- **Approved By**: Integration Specialist

---

## Navigation

- [← Previous: Implementation Guides](./7-implementation-guides.md)
- [→ Next: Operations & Deployment](./9-operations-deployment.md)