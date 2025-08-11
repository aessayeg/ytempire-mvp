"""
Content Quality Scoring System
Evaluates and scores generated content quality
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import re
from textstat import flesch_reading_ease, flesch_kincaid_grade
import spacy
from transformers import pipeline
import asyncio
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Load NLP models
try:
    nlp = spacy.load("en_core_web_sm")
    sentiment_analyzer = pipeline("sentiment-analysis")
    toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")
except Exception as e:
    logger.warning(f"Could not load NLP models: {e}")
    nlp = None
    sentiment_analyzer = None
    toxicity_classifier = None


class QualityDimension(Enum):
    """Quality scoring dimensions"""
    RELEVANCE = "relevance"
    ENGAGEMENT = "engagement"
    CLARITY = "clarity"
    ORIGINALITY = "originality"
    ACCURACY = "accuracy"
    VIRALITY = "virality"
    BRAND_SAFETY = "brand_safety"
    SEO_OPTIMIZATION = "seo_optimization"
    EMOTIONAL_IMPACT = "emotional_impact"
    TECHNICAL_QUALITY = "technical_quality"


@dataclass
class QualityScore:
    """Quality score result"""
    overall_score: float
    dimension_scores: Dict[QualityDimension, float]
    feedback: List[str]
    recommendations: List[str]
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]


class ContentQualityScorer:
    """Main content quality scoring system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.weights = self._initialize_weights()
        self.thresholds = self._initialize_thresholds()
        self.scaler = MinMaxScaler()
        
    def _initialize_weights(self) -> Dict[QualityDimension, float]:
        """Initialize dimension weights"""
        return {
            QualityDimension.RELEVANCE: 0.20,
            QualityDimension.ENGAGEMENT: 0.15,
            QualityDimension.CLARITY: 0.10,
            QualityDimension.ORIGINALITY: 0.10,
            QualityDimension.ACCURACY: 0.10,
            QualityDimension.VIRALITY: 0.10,
            QualityDimension.BRAND_SAFETY: 0.10,
            QualityDimension.SEO_OPTIMIZATION: 0.05,
            QualityDimension.EMOTIONAL_IMPACT: 0.05,
            QualityDimension.TECHNICAL_QUALITY: 0.05
        }
    
    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize quality thresholds"""
        return {
            "minimum_overall": 0.7,
            "excellent": 0.9,
            "good": 0.8,
            "acceptable": 0.7,
            "poor": 0.5
        }
    
    async def score_content(
        self,
        content: Dict[str, Any],
        content_type: str = "video"
    ) -> QualityScore:
        """Score content quality across all dimensions"""
        dimension_scores = {}
        feedback = []
        recommendations = []
        
        # Score each dimension
        if content_type == "video":
            dimension_scores = await self._score_video_content(content)
        elif content_type == "script":
            dimension_scores = await self._score_script_content(content)
        elif content_type == "thumbnail":
            dimension_scores = await self._score_thumbnail(content)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores)
        
        # Generate feedback and recommendations
        feedback = self._generate_feedback(dimension_scores, overall_score)
        recommendations = self._generate_recommendations(dimension_scores)
        
        # Calculate confidence
        confidence = self._calculate_confidence(dimension_scores)
        
        return QualityScore(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            feedback=feedback,
            recommendations=recommendations,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            metadata={
                "content_type": content_type,
                "scoring_version": "1.0.0"
            }
        )
    
    async def _score_video_content(self, content: Dict[str, Any]) -> Dict[QualityDimension, float]:
        """Score video content"""
        scores = {}
        
        # Extract content components
        script = content.get("script", "")
        title = content.get("title", "")
        description = content.get("description", "")
        tags = content.get("tags", [])
        duration = content.get("duration", 0)
        thumbnail = content.get("thumbnail", {})
        
        # Score relevance
        scores[QualityDimension.RELEVANCE] = self._score_relevance(
            script, title, tags, content.get("target_audience", "")
        )
        
        # Score engagement potential
        scores[QualityDimension.ENGAGEMENT] = self._score_engagement(
            title, thumbnail, duration, script
        )
        
        # Score clarity
        scores[QualityDimension.CLARITY] = self._score_clarity(script)
        
        # Score originality
        scores[QualityDimension.ORIGINALITY] = await self._score_originality(
            script, title, content.get("existing_content", [])
        )
        
        # Score accuracy
        scores[QualityDimension.ACCURACY] = self._score_accuracy(
            script, content.get("fact_checks", [])
        )
        
        # Score virality potential
        scores[QualityDimension.VIRALITY] = self._score_virality(
            title, tags, thumbnail, content.get("trend_alignment", 0)
        )
        
        # Score brand safety
        scores[QualityDimension.BRAND_SAFETY] = self._score_brand_safety(
            script, title, description
        )
        
        # Score SEO optimization
        scores[QualityDimension.SEO_OPTIMIZATION] = self._score_seo(
            title, description, tags
        )
        
        # Score emotional impact
        scores[QualityDimension.EMOTIONAL_IMPACT] = self._score_emotional_impact(script)
        
        # Score technical quality
        scores[QualityDimension.TECHNICAL_QUALITY] = self._score_technical_quality(
            content.get("video_quality", {}),
            content.get("audio_quality", {})
        )
        
        return scores
    
    async def _score_script_content(self, content: Dict[str, Any]) -> Dict[QualityDimension, float]:
        """Score script content"""
        script = content.get("text", "")
        scores = {}
        
        scores[QualityDimension.CLARITY] = self._score_clarity(script)
        scores[QualityDimension.ENGAGEMENT] = self._score_script_engagement(script)
        scores[QualityDimension.ORIGINALITY] = await self._score_originality(script, "", [])
        scores[QualityDimension.BRAND_SAFETY] = self._score_brand_safety(script, "", "")
        scores[QualityDimension.EMOTIONAL_IMPACT] = self._score_emotional_impact(script)
        
        # Default scores for non-applicable dimensions
        scores[QualityDimension.RELEVANCE] = 0.8
        scores[QualityDimension.ACCURACY] = 0.8
        scores[QualityDimension.VIRALITY] = 0.7
        scores[QualityDimension.SEO_OPTIMIZATION] = 0.7
        scores[QualityDimension.TECHNICAL_QUALITY] = 1.0
        
        return scores
    
    async def _score_thumbnail(self, content: Dict[str, Any]) -> Dict[QualityDimension, float]:
        """Score thumbnail quality"""
        scores = {}
        
        # Thumbnail-specific scoring
        scores[QualityDimension.ENGAGEMENT] = content.get("click_appeal", 0.7)
        scores[QualityDimension.CLARITY] = content.get("visual_clarity", 0.8)
        scores[QualityDimension.BRAND_SAFETY] = content.get("appropriate", 1.0)
        scores[QualityDimension.TECHNICAL_QUALITY] = content.get("resolution_quality", 0.9)
        
        # Default scores for non-applicable dimensions
        scores[QualityDimension.RELEVANCE] = 0.8
        scores[QualityDimension.ORIGINALITY] = 0.7
        scores[QualityDimension.ACCURACY] = 1.0
        scores[QualityDimension.VIRALITY] = 0.7
        scores[QualityDimension.SEO_OPTIMIZATION] = 0.5
        scores[QualityDimension.EMOTIONAL_IMPACT] = 0.6
        
        return scores
    
    def _score_relevance(
        self,
        script: str,
        title: str,
        tags: List[str],
        target_audience: str
    ) -> float:
        """Score content relevance"""
        score = 0.0
        
        # Keyword matching
        keywords = set(tags)
        script_words = set(script.lower().split())
        title_words = set(title.lower().split())
        
        keyword_coverage = len(keywords.intersection(script_words)) / max(len(keywords), 1)
        title_relevance = len(keywords.intersection(title_words)) / max(len(keywords), 1)
        
        score = (keyword_coverage * 0.6 + title_relevance * 0.4)
        
        # Audience alignment bonus
        if target_audience and target_audience.lower() in script.lower():
            score = min(1.0, score + 0.1)
        
        return min(1.0, max(0.0, score))
    
    def _score_engagement(
        self,
        title: str,
        thumbnail: Dict[str, Any],
        duration: int,
        script: str
    ) -> float:
        """Score engagement potential"""
        score = 0.0
        
        # Title engagement factors
        title_score = 0.0
        if len(title) > 10 and len(title) < 60:
            title_score += 0.3
        if any(word in title.lower() for word in ["how", "why", "what", "best", "top"]):
            title_score += 0.2
        if "?" in title or "!" in title:
            title_score += 0.1
        
        # Duration optimization (8-12 minutes is optimal)
        duration_score = 0.0
        if 480 <= duration <= 720:
            duration_score = 1.0
        elif 300 <= duration <= 900:
            duration_score = 0.7
        else:
            duration_score = 0.4
        
        # Hook strength (first 30 seconds)
        hook_score = 0.7  # Default
        if script:
            first_paragraph = script.split('\n')[0] if script else ""
            if len(first_paragraph) > 50 and len(first_paragraph) < 200:
                hook_score = 0.9
        
        score = (title_score * 0.4 + duration_score * 0.3 + hook_score * 0.3)
        
        return min(1.0, max(0.0, score))
    
    def _score_clarity(self, text: str) -> float:
        """Score text clarity and readability"""
        if not text:
            return 0.5
        
        try:
            # Readability scores
            reading_ease = flesch_reading_ease(text)
            grade_level = flesch_kincaid_grade(text)
            
            # Target: 60-70 reading ease (fairly easy)
            # Target: 7-9 grade level
            ease_score = 0.0
            if 60 <= reading_ease <= 70:
                ease_score = 1.0
            elif 50 <= reading_ease <= 80:
                ease_score = 0.8
            elif 40 <= reading_ease <= 90:
                ease_score = 0.6
            else:
                ease_score = 0.4
            
            grade_score = 0.0
            if 7 <= grade_level <= 9:
                grade_score = 1.0
            elif 5 <= grade_level <= 11:
                grade_score = 0.8
            else:
                grade_score = 0.5
            
            # Sentence structure
            sentences = text.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s])
            
            structure_score = 0.0
            if 15 <= avg_sentence_length <= 20:
                structure_score = 1.0
            elif 10 <= avg_sentence_length <= 25:
                structure_score = 0.8
            else:
                structure_score = 0.5
            
            return (ease_score * 0.4 + grade_score * 0.3 + structure_score * 0.3)
            
        except Exception as e:
            logger.error(f"Error calculating clarity score: {e}")
            return 0.7
    
    async def _score_originality(
        self,
        script: str,
        title: str,
        existing_content: List[str]
    ) -> float:
        """Score content originality"""
        # Simplified originality check
        # In production, would use more sophisticated plagiarism detection
        
        if not existing_content:
            return 0.8  # Default score when no comparison data
        
        # Check for exact matches
        script_lower = script.lower()
        for existing in existing_content:
            if existing.lower() in script_lower or script_lower in existing.lower():
                return 0.2  # High similarity detected
        
        # Check title uniqueness
        title_lower = title.lower()
        for existing in existing_content:
            if title_lower in existing.lower():
                return 0.4  # Title too similar
        
        return 0.85  # Appears original
    
    def _score_accuracy(self, script: str, fact_checks: List[Dict[str, Any]]) -> float:
        """Score factual accuracy"""
        if not fact_checks:
            return 0.8  # Default when no fact checking available
        
        accurate_facts = sum(1 for check in fact_checks if check.get("verified", False))
        total_facts = len(fact_checks)
        
        if total_facts == 0:
            return 0.8
        
        accuracy_rate = accurate_facts / total_facts
        return accuracy_rate
    
    def _score_virality(
        self,
        title: str,
        tags: List[str],
        thumbnail: Dict[str, Any],
        trend_alignment: float
    ) -> float:
        """Score viral potential"""
        score = trend_alignment  # Start with trend alignment score
        
        # Viral title patterns
        viral_patterns = [
            r'\d+',  # Numbers in title
            r'(never|always|everyone|no one)',  # Absolutes
            r'(shocking|amazing|unbelievable|insane)',  # Strong emotions
            r'(hack|trick|secret)',  # Insider knowledge
            r'[\?\!]',  # Questions or exclamations
        ]
        
        title_virality = sum(0.1 for pattern in viral_patterns 
                           if re.search(pattern, title.lower()))
        
        # Tag optimization
        tag_score = min(1.0, len(tags) / 10) if tags else 0
        
        # Thumbnail appeal
        thumbnail_score = thumbnail.get("click_appeal", 0.7)
        
        final_score = (
            score * 0.4 +
            title_virality * 0.3 +
            tag_score * 0.15 +
            thumbnail_score * 0.15
        )
        
        return min(1.0, max(0.0, final_score))
    
    def _score_brand_safety(self, script: str, title: str, description: str) -> float:
        """Score brand safety and appropriateness"""
        if not toxicity_classifier:
            return 0.9  # Default safe score
        
        try:
            # Check for toxic content
            combined_text = f"{title} {description} {script}"[:512]  # Limit length
            toxicity_results = toxicity_classifier(combined_text)
            
            # Get toxicity score
            toxic_score = 0.0
            for result in toxicity_results:
                if result['label'] == 'TOXIC':
                    toxic_score = result['score']
                    break
            
            # Invert for safety score
            safety_score = 1.0 - toxic_score
            
            # Check for sensitive topics
            sensitive_keywords = [
                'politics', 'religion', 'violence', 'adult', 'gambling',
                'alcohol', 'drugs', 'controversial'
            ]
            
            text_lower = combined_text.lower()
            sensitivity_penalty = sum(0.05 for keyword in sensitive_keywords 
                                     if keyword in text_lower)
            
            final_score = max(0.0, safety_score - sensitivity_penalty)
            return final_score
            
        except Exception as e:
            logger.error(f"Error in brand safety scoring: {e}")
            return 0.8
    
    def _score_seo(self, title: str, description: str, tags: List[str]) -> float:
        """Score SEO optimization"""
        score = 0.0
        
        # Title optimization
        title_score = 0.0
        if 30 <= len(title) <= 60:
            title_score += 0.5
        if any(char.isdigit() for char in title):
            title_score += 0.2
        if ':' in title or '-' in title:
            title_score += 0.3
        
        # Description optimization
        desc_score = 0.0
        if description:
            if 100 <= len(description) <= 200:
                desc_score = 1.0
            elif 50 <= len(description) <= 300:
                desc_score = 0.7
            else:
                desc_score = 0.4
        
        # Tags optimization
        tag_score = 0.0
        if tags:
            if 5 <= len(tags) <= 15:
                tag_score = 1.0
            elif 3 <= len(tags) <= 20:
                tag_score = 0.7
            else:
                tag_score = 0.4
        
        score = (title_score * 0.4 + desc_score * 0.3 + tag_score * 0.3)
        return min(1.0, max(0.0, score))
    
    def _score_emotional_impact(self, text: str) -> float:
        """Score emotional impact and sentiment"""
        if not sentiment_analyzer or not text:
            return 0.7  # Default neutral score
        
        try:
            # Analyze sentiment
            sentiment_results = sentiment_analyzer(text[:512])
            
            # Look for strong emotional content
            emotion_keywords = {
                'positive': ['amazing', 'incredible', 'fantastic', 'wonderful', 'excellent'],
                'negative': ['terrible', 'horrible', 'awful', 'disappointing', 'worst'],
                'surprise': ['shocking', 'surprising', 'unexpected', 'unbelievable'],
                'curiosity': ['mysterious', 'secret', 'unknown', 'hidden', 'discover']
            }
            
            text_lower = text.lower()
            emotion_score = 0.0
            
            for category, keywords in emotion_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    emotion_score += 0.2
            
            # Combine sentiment and emotion scores
            sentiment_score = sentiment_results[0]['score'] if sentiment_results else 0.5
            
            final_score = min(1.0, (sentiment_score * 0.5 + emotion_score * 0.5))
            return final_score
            
        except Exception as e:
            logger.error(f"Error in emotional impact scoring: {e}")
            return 0.7
    
    def _score_technical_quality(
        self,
        video_quality: Dict[str, Any],
        audio_quality: Dict[str, Any]
    ) -> float:
        """Score technical quality"""
        video_score = video_quality.get("score", 0.8)
        audio_score = audio_quality.get("score", 0.8)
        
        # Check specific quality metrics
        resolution_score = 1.0 if video_quality.get("resolution", "1080p") in ["1080p", "4K"] else 0.7
        fps_score = 1.0 if video_quality.get("fps", 30) >= 30 else 0.6
        bitrate_score = 1.0 if video_quality.get("bitrate", 5000) >= 4000 else 0.7
        
        # Audio quality checks
        audio_clarity = audio_quality.get("clarity", 0.8)
        audio_levels = audio_quality.get("levels_ok", True)
        audio_level_score = 1.0 if audio_levels else 0.5
        
        final_score = (
            video_score * 0.3 +
            audio_score * 0.3 +
            resolution_score * 0.15 +
            fps_score * 0.1 +
            bitrate_score * 0.1 +
            audio_clarity * 0.05
        )
        
        return min(1.0, max(0.0, final_score))
    
    def _score_script_engagement(self, script: str) -> float:
        """Score script engagement potential"""
        if not script:
            return 0.5
        
        # Story structure elements
        has_hook = bool(re.search(r'^.{20,100}[\?\!]', script))
        has_conflict = any(word in script.lower() for word in 
                         ['problem', 'challenge', 'issue', 'struggle', 'difficult'])
        has_resolution = any(word in script.lower() for word in 
                           ['solution', 'solve', 'fix', 'answer', 'resolve'])
        has_cta = any(phrase in script.lower() for phrase in 
                    ['subscribe', 'like', 'comment', 'share', 'click'])
        
        structure_score = sum([has_hook, has_conflict, has_resolution, has_cta]) * 0.25
        
        # Pacing (variation in sentence length)
        sentences = [s.strip() for s in script.split('.') if s.strip()]
        if sentences:
            lengths = [len(s.split()) for s in sentences]
            variation = np.std(lengths) / np.mean(lengths) if lengths else 0
            pacing_score = min(1.0, variation / 0.5)  # Optimal variation around 0.5
        else:
            pacing_score = 0.5
        
        return (structure_score * 0.7 + pacing_score * 0.3)
    
    def _calculate_overall_score(
        self,
        dimension_scores: Dict[QualityDimension, float]
    ) -> float:
        """Calculate weighted overall score"""
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            weight = self.weights.get(dimension, 0.1)
            total_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        return 0.0
    
    def _generate_feedback(
        self,
        dimension_scores: Dict[QualityDimension, float],
        overall_score: float
    ) -> List[str]:
        """Generate feedback based on scores"""
        feedback = []
        
        # Overall feedback
        if overall_score >= self.thresholds["excellent"]:
            feedback.append("Excellent content quality! Ready for publication.")
        elif overall_score >= self.thresholds["good"]:
            feedback.append("Good content quality with minor improvements possible.")
        elif overall_score >= self.thresholds["acceptable"]:
            feedback.append("Acceptable quality but improvements recommended.")
        else:
            feedback.append("Quality below standards. Significant improvements needed.")
        
        # Dimension-specific feedback
        for dimension, score in dimension_scores.items():
            if score < 0.6:
                feedback.append(f"⚠️ {dimension.value}: Needs improvement (score: {score:.2f})")
            elif score > 0.9:
                feedback.append(f"✅ {dimension.value}: Excellent (score: {score:.2f})")
        
        return feedback
    
    def _generate_recommendations(
        self,
        dimension_scores: Dict[QualityDimension, float]
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Check each dimension
        for dimension, score in dimension_scores.items():
            if score < 0.7:
                if dimension == QualityDimension.CLARITY:
                    recommendations.append("Simplify language and sentence structure")
                elif dimension == QualityDimension.ENGAGEMENT:
                    recommendations.append("Add more compelling hooks and calls-to-action")
                elif dimension == QualityDimension.SEO_OPTIMIZATION:
                    recommendations.append("Optimize title length and add more relevant tags")
                elif dimension == QualityDimension.VIRALITY:
                    recommendations.append("Align content with current trends")
                elif dimension == QualityDimension.BRAND_SAFETY:
                    recommendations.append("Review content for sensitive topics")
                elif dimension == QualityDimension.ORIGINALITY:
                    recommendations.append("Add more unique perspectives or insights")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_confidence(
        self,
        dimension_scores: Dict[QualityDimension, float]
    ) -> float:
        """Calculate scoring confidence"""
        # Higher variance in scores = lower confidence
        scores = list(dimension_scores.values())
        if not scores:
            return 0.5
        
        variance = np.var(scores)
        # Inverse relationship with variance
        confidence = max(0.5, min(1.0, 1.0 - (variance * 2)))
        
        return confidence


# Quality monitoring and improvement
class QualityMonitor:
    """Monitor and track quality trends"""
    
    def __init__(self):
        self.history: List[QualityScore] = []
        self.thresholds = {
            "minimum": 0.7,
            "target": 0.85
        }
    
    def add_score(self, score: QualityScore):
        """Add score to history"""
        self.history.append(score)
        
        # Keep only last 1000 scores
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
    
    def get_trends(self) -> Dict[str, Any]:
        """Analyze quality trends"""
        if not self.history:
            return {}
        
        recent_scores = [s.overall_score for s in self.history[-100:]]
        older_scores = [s.overall_score for s in self.history[-200:-100]] if len(self.history) > 100 else []
        
        trends = {
            "current_average": np.mean(recent_scores),
            "previous_average": np.mean(older_scores) if older_scores else None,
            "trend": "improving" if older_scores and np.mean(recent_scores) > np.mean(older_scores) else "stable",
            "below_threshold": sum(1 for s in recent_scores if s < self.thresholds["minimum"]),
            "above_target": sum(1 for s in recent_scores if s > self.thresholds["target"])
        }
        
        return trends
    
    def get_dimension_insights(self) -> Dict[QualityDimension, Dict[str, float]]:
        """Get insights per quality dimension"""
        if not self.history:
            return {}
        
        insights = {}
        recent = self.history[-100:]
        
        for dimension in QualityDimension:
            scores = [s.dimension_scores.get(dimension, 0) for s in recent]
            if scores:
                insights[dimension] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "improving": self._is_improving(dimension)
                }
        
        return insights
    
    def _is_improving(self, dimension: QualityDimension) -> bool:
        """Check if dimension is improving"""
        if len(self.history) < 20:
            return False
        
        recent = [s.dimension_scores.get(dimension, 0) for s in self.history[-10:]]
        older = [s.dimension_scores.get(dimension, 0) for s in self.history[-20:-10]]
        
        return np.mean(recent) > np.mean(older)


# Global instances
quality_scorer = ContentQualityScorer()
quality_monitor = QualityMonitor()