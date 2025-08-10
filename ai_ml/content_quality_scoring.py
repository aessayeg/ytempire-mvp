"""
Content Quality Scoring System
Automated quality assessment for YouTube video content
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, GPT2LMHeadModel, GPT2Tokenizer
)
import cv2
import librosa
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import asyncio
import aiohttp
import redis.asyncio as redis
from datetime import datetime, timedelta
import json
import logging
from prometheus_client import Histogram, Counter, Gauge

# Metrics
quality_score_histogram = Histogram('content_quality_score', 'Content quality scores', ['content_type'])
quality_check_duration = Histogram('quality_check_duration', 'Time to check quality', ['check_type'])
low_quality_detected = Counter('low_quality_content', 'Low quality content detected', ['reason'])
content_processed = Counter('content_processed_total', 'Total content processed', ['content_type'])

logger = logging.getLogger(__name__)

@dataclass
class ContentQualityScore:
    """Container for content quality scores"""
    overall_score: float
    title_score: float
    description_score: float
    script_score: float
    thumbnail_score: float
    audio_quality_score: float
    video_quality_score: float
    engagement_prediction: float
    seo_score: float
    originality_score: float
    brand_alignment_score: float
    toxicity_score: float
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    passed: bool = False

@dataclass
class VideoContent:
    """Video content data structure"""
    video_id: str
    title: str
    description: str
    script: str
    tags: List[str]
    thumbnail_path: Optional[str] = None
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    category: Optional[str] = None
    target_audience: Optional[str] = None
    language: str = 'en'

class ContentQualityScorer:
    """Advanced content quality scoring system"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            'nlptown/bert-base-multilingual-uncased-sentiment'
        )
        self.toxicity_pipeline = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # GPT-2 for perplexity calculation
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Initialize Redis for caching
        self.redis_client = None
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # Quality thresholds
        self.quality_thresholds = {
            'overall': 0.7,
            'title': 0.75,
            'description': 0.65,
            'script': 0.7,
            'thumbnail': 0.8,
            'audio': 0.6,
            'video': 0.7,
            'engagement': 0.5,
            'seo': 0.7,
            'originality': 0.6,
            'brand_alignment': 0.7,
            'toxicity': 0.2  # Lower is better
        }
        
        # SEO keywords database
        self.trending_keywords = set()
        self.high_cpm_keywords = set()
        
    async def initialize(self):
        """Initialize async components"""
        self.redis_client = await redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=True
        )
        await self._load_keyword_databases()
    
    async def score_content(self, content: VideoContent) -> ContentQualityScore:
        """
        Comprehensive content quality scoring
        
        Args:
            content: VideoContent object with all content details
        
        Returns:
            ContentQualityScore with detailed scoring and recommendations
        """
        start_time = datetime.utcnow()
        
        # Check cache first
        cached_score = await self._get_cached_score(content.video_id)
        if cached_score:
            return cached_score
        
        # Initialize scores
        scores = ContentQualityScore(
            overall_score=0.0,
            title_score=0.0,
            description_score=0.0,
            script_score=0.0,
            thumbnail_score=0.0,
            audio_quality_score=0.0,
            video_quality_score=0.0,
            engagement_prediction=0.0,
            seo_score=0.0,
            originality_score=0.0,
            brand_alignment_score=0.0,
            toxicity_score=0.0
        )
        
        # Score individual components
        scores.title_score = await self._score_title(content.title)
        scores.description_score = await self._score_description(content.description)
        scores.script_score = await self._score_script(content.script)
        scores.seo_score = await self._calculate_seo_score(content)
        scores.originality_score = await self._check_originality(content)
        scores.toxicity_score = await self._check_toxicity(content)
        scores.brand_alignment_score = await self._check_brand_alignment(content)
        
        # Score media quality if available
        if content.thumbnail_path:
            scores.thumbnail_score = await self._score_thumbnail(content.thumbnail_path)
        
        if content.video_path:
            scores.video_quality_score = await self._score_video_quality(content.video_path)
        
        if content.audio_path:
            scores.audio_quality_score = await self._score_audio_quality(content.audio_path)
        
        # Predict engagement
        scores.engagement_prediction = await self._predict_engagement(content, scores)
        
        # Calculate overall score
        scores.overall_score = self._calculate_overall_score(scores)
        
        # Generate recommendations
        scores.recommendations = self._generate_recommendations(scores)
        
        # Determine if content passes quality checks
        scores.passed = self._check_quality_pass(scores)
        
        # Add detailed metrics
        scores.detailed_metrics = await self._gather_detailed_metrics(content, scores)
        
        # Cache the score
        await self._cache_score(content.video_id, scores)
        
        # Update metrics
        quality_score_histogram.labels(content_type=content.category or 'unknown').observe(scores.overall_score)
        content_processed.labels(content_type=content.category or 'unknown').inc()
        
        if not scores.passed:
            low_quality_detected.labels(reason=scores.recommendations[0] if scores.recommendations else 'unknown').inc()
        
        # Log processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        quality_check_duration.labels(check_type='full').observe(processing_time)
        
        return scores
    
    async def _score_title(self, title: str) -> float:
        """Score title quality"""
        score = 1.0
        
        # Length check (optimal: 50-60 characters)
        title_length = len(title)
        if title_length < 30:
            score *= 0.7
        elif title_length > 100:
            score *= 0.8
        elif 50 <= title_length <= 60:
            score *= 1.1
        
        # Capitalization check
        if title.isupper():
            score *= 0.9  # All caps is usually bad
        elif title[0].islower():
            score *= 0.95  # Should start with capital
        
        # Emoji usage (moderate is good)
        emoji_count = sum(1 for c in title if ord(c) > 127462)
        if emoji_count == 0:
            score *= 0.95
        elif emoji_count > 3:
            score *= 0.85
        
        # Clickbait detection
        clickbait_words = ['shocking', 'unbelievable', 'you won\'t believe', 'gone wrong']
        for word in clickbait_words:
            if word.lower() in title.lower():
                score *= 0.8
        
        # Power words (good for engagement)
        power_words = ['ultimate', 'essential', 'complete', 'proven', 'exclusive']
        for word in power_words:
            if word.lower() in title.lower():
                score *= 1.05
        
        # Question format (good for engagement)
        if '?' in title:
            score *= 1.05
        
        # Number usage (lists perform well)
        import re
        if re.search(r'\d+', title):
            score *= 1.05
        
        return min(1.0, score)
    
    async def _score_description(self, description: str) -> float:
        """Score description quality"""
        score = 1.0
        
        # Length check (optimal: 200-350 words)
        word_count = len(description.split())
        if word_count < 50:
            score *= 0.6
        elif word_count > 500:
            score *= 0.9
        elif 200 <= word_count <= 350:
            score *= 1.1
        
        # Check for timestamps
        if re.search(r'\d{1,2}:\d{2}', description):
            score *= 1.1  # Timestamps are good
        
        # Check for links
        if 'http' in description or 'www.' in description:
            score *= 1.05  # Links to resources are good
        
        # Check for hashtags
        hashtag_count = description.count('#')
        if 3 <= hashtag_count <= 10:
            score *= 1.05
        elif hashtag_count > 15:
            score *= 0.9
        
        # Check for call-to-action
        cta_phrases = ['subscribe', 'like', 'comment', 'share', 'follow']
        for phrase in cta_phrases:
            if phrase.lower() in description.lower():
                score *= 1.02
        
        # Paragraph structure
        if '\n\n' in description:
            score *= 1.05  # Good formatting
        
        return min(1.0, score)
    
    async def _score_script(self, script: str) -> float:
        """Score script quality using advanced NLP"""
        if not script:
            return 0.5
        
        score = 1.0
        
        # Calculate perplexity (lower is better)
        perplexity = await self._calculate_perplexity(script)
        if perplexity < 50:
            score *= 1.1
        elif perplexity > 200:
            score *= 0.8
        
        # Readability score (Flesch Reading Ease)
        readability = self._calculate_readability(script)
        if 60 <= readability <= 80:  # Optimal range
            score *= 1.1
        elif readability < 30:
            score *= 0.8
        
        # Sentiment analysis
        sentiment_score = await self._analyze_sentiment(script)
        if 0.6 <= sentiment_score <= 0.8:  # Positive but not overly so
            score *= 1.05
        
        # Structure analysis
        sentences = script.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s])
        if 10 <= avg_sentence_length <= 20:
            score *= 1.05
        
        # Vocabulary diversity
        words = script.lower().split()
        vocabulary_diversity = len(set(words)) / len(words) if words else 0
        if vocabulary_diversity > 0.5:
            score *= 1.05
        
        return min(1.0, score)
    
    async def _calculate_perplexity(self, text: str) -> float:
        """Calculate text perplexity using GPT-2"""
        encodings = self.gpt2_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.gpt2_model(**encodings, labels=encodings['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        sentences = text.split('.')
        words = text.split()
        syllables = sum([self._count_syllables(word) for word in words])
        
        if not sentences or not words:
            return 50.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        # Flesch Reading Ease formula
        score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        
        return max(0, min(100, score))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        if word.endswith('e'):
            syllable_count -= 1
        
        if syllable_count == 0:
            syllable_count = 1
        
        return syllable_count
    
    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze text sentiment"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Convert 5-star rating to 0-1 scale
            weighted_score = sum([(i+1) * score.item() for i, score in enumerate(scores[0])]) / 5
        
        return weighted_score
    
    async def _score_thumbnail(self, thumbnail_path: str) -> float:
        """Score thumbnail quality"""
        try:
            # Load image
            image = cv2.imread(thumbnail_path)
            if image is None:
                return 0.5
            
            score = 1.0
            height, width = image.shape[:2]
            
            # Resolution check (1280x720 is optimal)
            if width >= 1280 and height >= 720:
                score *= 1.1
            elif width < 640 or height < 360:
                score *= 0.7
            
            # Aspect ratio (16:9 is optimal)
            aspect_ratio = width / height
            if 1.7 <= aspect_ratio <= 1.8:
                score *= 1.05
            
            # Color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].mean()
            brightness = hsv[:, :, 2].mean()
            
            # Good saturation and brightness
            if 100 <= saturation <= 200 and 100 <= brightness <= 200:
                score *= 1.05
            
            # Edge detection (more edges = more detail)
            edges = cv2.Canny(image, 100, 200)
            edge_ratio = np.count_nonzero(edges) / (width * height)
            if 0.05 <= edge_ratio <= 0.15:
                score *= 1.05
            
            # Face detection (faces in thumbnails perform well)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1.1, 4)
            if len(faces) > 0:
                score *= 1.1
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error scoring thumbnail: {e}")
            return 0.5
    
    async def _score_video_quality(self, video_path: str) -> float:
        """Score video quality"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            score = 1.0
            
            # FPS check (30+ is good)
            if fps >= 60:
                score *= 1.1
            elif fps >= 30:
                score *= 1.05
            elif fps < 24:
                score *= 0.8
            
            # Resolution check
            if width >= 1920 and height >= 1080:  # Full HD or better
                score *= 1.1
            elif width >= 1280 and height >= 720:  # HD
                score *= 1.05
            elif width < 854 or height < 480:
                score *= 0.7
            
            # Sample frames for quality analysis
            sample_frames = []
            frame_interval = int(frame_count / 10)  # Sample 10 frames
            
            for i in range(0, int(frame_count), frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    sample_frames.append(frame)
            
            # Check for blur in sampled frames
            blur_scores = []
            for frame in sample_frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_scores.append(laplacian_var)
            
            avg_blur = np.mean(blur_scores)
            if avg_blur > 500:  # Sharp
                score *= 1.05
            elif avg_blur < 100:  # Blurry
                score *= 0.8
            
            cap.release()
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error scoring video quality: {e}")
            return 0.5
    
    async def _score_audio_quality(self, audio_path: str) -> float:
        """Score audio quality"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            score = 1.0
            
            # Sample rate check
            if sr >= 44100:
                score *= 1.05
            elif sr < 22050:
                score *= 0.8
            
            # Check for clipping
            clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
            if clipping_ratio > 0.01:
                score *= 0.8
            
            # Signal-to-noise ratio (simplified)
            noise_floor = np.percentile(np.abs(audio), 10)
            signal_peak = np.percentile(np.abs(audio), 90)
            snr = 20 * np.log10(signal_peak / (noise_floor + 1e-10))
            
            if snr > 40:
                score *= 1.05
            elif snr < 20:
                score *= 0.8
            
            # Check for silence
            silence_threshold = 0.01
            silence_ratio = np.sum(np.abs(audio) < silence_threshold) / len(audio)
            if silence_ratio > 0.3:
                score *= 0.9
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error scoring audio quality: {e}")
            return 0.5
    
    async def _calculate_seo_score(self, content: VideoContent) -> float:
        """Calculate SEO optimization score"""
        score = 0.0
        
        # Keyword presence in title
        title_keywords = set(content.title.lower().split())
        trending_matches = len(title_keywords & self.trending_keywords)
        high_cpm_matches = len(title_keywords & self.high_cpm_keywords)
        
        score += min(0.3, trending_matches * 0.1)
        score += min(0.3, high_cpm_matches * 0.15)
        
        # Tags optimization
        if content.tags:
            if 5 <= len(content.tags) <= 15:
                score += 0.2
            tag_keywords = set(' '.join(content.tags).lower().split())
            score += min(0.2, len(tag_keywords & self.trending_keywords) * 0.05)
        
        # Description keyword density
        if content.description:
            desc_words = content.description.lower().split()
            keyword_density = len([w for w in desc_words if w in self.trending_keywords]) / len(desc_words)
            if 0.02 <= keyword_density <= 0.05:
                score += 0.2
        
        return min(1.0, score)
    
    async def _check_originality(self, content: VideoContent) -> float:
        """Check content originality"""
        # Simplified originality check
        # In production, this would check against a database of existing content
        
        # Check title uniqueness
        title_hash = hash(content.title.lower())
        
        # Check if title exists in cache (simulating database check)
        cached_titles = await self.redis_client.smembers('title_hashes')
        if str(title_hash) in cached_titles:
            return 0.3  # Low originality
        
        # Add to cache
        await self.redis_client.sadd('title_hashes', str(title_hash))
        await self.redis_client.expire('title_hashes', 86400 * 30)  # 30 days
        
        return 0.9  # High originality (new content)
    
    async def _check_toxicity(self, content: VideoContent) -> float:
        """Check content for toxicity"""
        combined_text = f"{content.title} {content.description} {content.script[:500]}"
        
        try:
            results = self.toxicity_pipeline(combined_text)
            # Get the toxicity score (higher is more toxic)
            toxic_score = max([r['score'] for r in results if r['label'] == 'TOXIC'], default=0)
            return toxic_score
        except Exception as e:
            logger.error(f"Error checking toxicity: {e}")
            return 0.0
    
    async def _check_brand_alignment(self, content: VideoContent) -> float:
        """Check alignment with brand guidelines"""
        # This would be customized based on specific brand guidelines
        score = 1.0
        
        # Check for prohibited words
        prohibited_words = ['competitor1', 'controversial_topic', 'banned_word']
        combined_text = f"{content.title} {content.description}".lower()
        
        for word in prohibited_words:
            if word in combined_text:
                score *= 0.5
        
        # Check for required elements
        required_elements = ['brand_name', 'tagline']
        for element in required_elements:
            if element not in combined_text:
                score *= 0.9
        
        return score
    
    async def _predict_engagement(self, content: VideoContent, scores: ContentQualityScore) -> float:
        """Predict potential engagement rate"""
        # Simplified engagement prediction based on quality scores
        features = [
            scores.title_score,
            scores.description_score,
            scores.script_score,
            scores.thumbnail_score,
            scores.seo_score,
            1.0 - scores.toxicity_score  # Inverse of toxicity
        ]
        
        # Weighted average
        weights = [0.25, 0.1, 0.2, 0.25, 0.15, 0.05]
        engagement_score = sum(f * w for f, w in zip(features, weights))
        
        # Add category-specific boost
        high_engagement_categories = ['entertainment', 'gaming', 'tech']
        if content.category and content.category.lower() in high_engagement_categories:
            engagement_score *= 1.1
        
        return min(1.0, engagement_score)
    
    def _calculate_overall_score(self, scores: ContentQualityScore) -> float:
        """Calculate weighted overall score"""
        weights = {
            'title': 0.15,
            'description': 0.1,
            'script': 0.2,
            'thumbnail': 0.15,
            'seo': 0.15,
            'originality': 0.1,
            'engagement': 0.1,
            'toxicity': 0.05  # Inverse weight
        }
        
        overall = (
            scores.title_score * weights['title'] +
            scores.description_score * weights['description'] +
            scores.script_score * weights['script'] +
            scores.thumbnail_score * weights['thumbnail'] +
            scores.seo_score * weights['seo'] +
            scores.originality_score * weights['originality'] +
            scores.engagement_prediction * weights['engagement'] +
            (1.0 - scores.toxicity_score) * weights['toxicity']
        )
        
        # Add media quality if available
        media_count = 0
        media_score = 0
        
        if scores.video_quality_score > 0:
            media_score += scores.video_quality_score
            media_count += 1
        
        if scores.audio_quality_score > 0:
            media_score += scores.audio_quality_score
            media_count += 1
        
        if media_count > 0:
            overall = overall * 0.9 + (media_score / media_count) * 0.1
        
        return min(1.0, overall)
    
    def _generate_recommendations(self, scores: ContentQualityScore) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if scores.title_score < self.quality_thresholds['title']:
            recommendations.append("Improve title: Consider optimal length (50-60 chars), add power words or numbers")
        
        if scores.description_score < self.quality_thresholds['description']:
            recommendations.append("Enhance description: Add timestamps, links, and optimize length (200-350 words)")
        
        if scores.script_score < self.quality_thresholds['script']:
            recommendations.append("Refine script: Improve readability and structure")
        
        if scores.thumbnail_score < self.quality_thresholds['thumbnail']:
            recommendations.append("Optimize thumbnail: Ensure HD resolution, good colors, and clear subject")
        
        if scores.seo_score < self.quality_thresholds['seo']:
            recommendations.append("Improve SEO: Add trending keywords and optimize tags")
        
        if scores.toxicity_score > self.quality_thresholds['toxicity']:
            recommendations.append("Review content for potentially toxic or controversial elements")
        
        if scores.originality_score < self.quality_thresholds['originality']:
            recommendations.append("Enhance originality: Content appears similar to existing videos")
        
        if scores.engagement_prediction < self.quality_thresholds['engagement']:
            recommendations.append("Boost engagement potential: Improve title, thumbnail, and content hook")
        
        return recommendations
    
    def _check_quality_pass(self, scores: ContentQualityScore) -> bool:
        """Check if content passes quality thresholds"""
        return (
            scores.overall_score >= self.quality_thresholds['overall'] and
            scores.toxicity_score <= self.quality_thresholds['toxicity'] and
            scores.originality_score >= self.quality_thresholds['originality']
        )
    
    async def _gather_detailed_metrics(self, content: VideoContent, scores: ContentQualityScore) -> Dict:
        """Gather detailed metrics for analysis"""
        return {
            'word_counts': {
                'title': len(content.title.split()),
                'description': len(content.description.split()),
                'script': len(content.script.split())
            },
            'character_counts': {
                'title': len(content.title),
                'description': len(content.description)
            },
            'tag_count': len(content.tags) if content.tags else 0,
            'has_media': {
                'thumbnail': content.thumbnail_path is not None,
                'video': content.video_path is not None,
                'audio': content.audio_path is not None
            },
            'category': content.category,
            'language': content.language,
            'processing_timestamp': datetime.utcnow().isoformat()
        }
    
    async def _get_cached_score(self, video_id: str) -> Optional[ContentQualityScore]:
        """Get cached quality score"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(f"quality_score:{video_id}")
            if cached_data:
                return ContentQualityScore(**json.loads(cached_data))
        except Exception as e:
            logger.error(f"Error getting cached score: {e}")
        
        return None
    
    async def _cache_score(self, video_id: str, score: ContentQualityScore):
        """Cache quality score"""
        if not self.redis_client:
            return
        
        try:
            # Convert to dict, handling non-serializable fields
            score_dict = {
                'overall_score': score.overall_score,
                'title_score': score.title_score,
                'description_score': score.description_score,
                'script_score': score.script_score,
                'thumbnail_score': score.thumbnail_score,
                'audio_quality_score': score.audio_quality_score,
                'video_quality_score': score.video_quality_score,
                'engagement_prediction': score.engagement_prediction,
                'seo_score': score.seo_score,
                'originality_score': score.originality_score,
                'brand_alignment_score': score.brand_alignment_score,
                'toxicity_score': score.toxicity_score,
                'detailed_metrics': score.detailed_metrics,
                'recommendations': score.recommendations,
                'passed': score.passed
            }
            
            await self.redis_client.setex(
                f"quality_score:{video_id}",
                3600,  # 1 hour cache
                json.dumps(score_dict)
            )
        except Exception as e:
            logger.error(f"Error caching score: {e}")
    
    async def _load_keyword_databases(self):
        """Load trending and high-CPM keywords"""
        # In production, these would be loaded from a database or API
        self.trending_keywords = {
            'viral', 'trending', '2024', 'latest', 'new', 'best',
            'top', 'amazing', 'incredible', 'must', 'watch', 'see'
        }
        
        self.high_cpm_keywords = {
            'insurance', 'mortgage', 'attorney', 'credit', 'lawyer',
            'donate', 'degree', 'hosting', 'claim', 'conference',
            'trading', 'software', 'recovery', 'transfer', 'gas',
            'electricity', 'classes', 'rehab', 'treatment', 'cord'
        }

# Example usage
async def main():
    # Initialize scorer
    scorer = ContentQualityScorer()
    await scorer.initialize()
    
    # Example content
    content = VideoContent(
        video_id="test_video_001",
        title="10 Amazing Python Tips You Need to Know in 2024",
        description="In this video, we'll explore 10 incredible Python tips...",
        script="Welcome to this tutorial where we'll dive deep into Python...",
        tags=["python", "programming", "tutorial", "tips", "2024"],
        category="tech"
    )
    
    # Score content
    quality_score = await scorer.score_content(content)
    
    print(f"Overall Score: {quality_score.overall_score:.2f}")
    print(f"Passed: {quality_score.passed}")
    print(f"Recommendations: {quality_score.recommendations}")

if __name__ == "__main__":
    asyncio.run(main())