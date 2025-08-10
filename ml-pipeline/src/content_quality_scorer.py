"""
Content Quality Scoring System for YTEmpire
Evaluates and scores video content quality
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    overall_score: float
    script_quality: float
    voice_quality: float
    visual_quality: float
    thumbnail_quality: float
    seo_score: float
    engagement_potential: float
    recommendations: List[str]


class ContentQualityScorer:
    """Scores content quality across multiple dimensions"""
    
    def score_content(self, content: Dict[str, Any]) -> QualityScore:
        """Calculate comprehensive quality score"""
        
        script_score = self._score_script(content.get('script', ''))
        voice_score = self._score_voice(content.get('voice_params', {}))
        visual_score = self._score_visuals(content.get('visuals', []))
        thumbnail_score = self._score_thumbnail(content.get('thumbnail', {}))
        seo_score = self._score_seo(content)
        engagement_score = self._score_engagement_potential(content)
        
        # Weighted overall score
        overall = (
            script_score * 0.25 +
            voice_score * 0.15 +
            visual_score * 0.20 +
            thumbnail_score * 0.15 +
            seo_score * 0.15 +
            engagement_score * 0.10
        )
        
        recommendations = self._generate_recommendations(
            script_score, voice_score, visual_score, 
            thumbnail_score, seo_score, engagement_score
        )
        
        return QualityScore(
            overall_score=overall,
            script_quality=script_score,
            voice_quality=voice_score,
            visual_quality=visual_score,
            thumbnail_quality=thumbnail_score,
            seo_score=seo_score,
            engagement_potential=engagement_score,
            recommendations=recommendations
        )
    
    def _score_script(self, script: str) -> float:
        """Score script quality with comprehensive metrics"""
        if not script:
            return 0.0
        
        score = 0.0
        words = script.split()
        word_count = len(words)
        sentences = script.split('.')
        
        # Length optimization (15% weight)
        if 1500 <= word_count <= 2500:
            score += 0.15
        elif 1000 <= word_count < 1500 or 2500 < word_count <= 3000:
            score += 0.10
        elif 500 <= word_count < 1000 or 3000 < word_count <= 4000:
            score += 0.05
        
        # Hook quality analysis (20% weight)
        hook = ' '.join(words[:50]) if len(words) >= 50 else script
        hook_keywords = ['discover', 'learn', 'secret', 'amazing', 'how to', 'revealed', 
                        'ultimate', 'proven', 'exclusive', 'breakthrough', 'transform']
        hook_score = sum(1 for word in hook_keywords if word in hook.lower()) / len(hook_keywords)
        score += min(0.20, hook_score * 0.20)
        
        # Engagement elements (15% weight)
        engagement_phrases = ['subscribe', 'like', 'comment', 'share', 'notification bell',
                            'join', 'follow', 'let me know', 'what do you think']
        engagement_count = sum(1 for phrase in engagement_phrases if phrase in script.lower())
        score += min(0.15, (engagement_count / 5) * 0.15)
        
        # Structure analysis (20% weight)
        paragraphs = script.split('\n\n')
        if len(paragraphs) >= 5:  # Well-structured
            score += 0.20
        elif len(paragraphs) >= 3:
            score += 0.15
        elif len(paragraphs) >= 2:
            score += 0.10
        
        # Readability score (15% weight)
        avg_sentence_length = word_count / max(len(sentences), 1)
        if 15 <= avg_sentence_length <= 20:  # Optimal readability
            score += 0.15
        elif 10 <= avg_sentence_length < 15 or 20 < avg_sentence_length <= 25:
            score += 0.10
        else:
            score += 0.05
        
        # Keyword density (10% weight)
        # Check for relevant keywords without stuffing
        common_words = set(words)
        unique_ratio = len(common_words) / max(word_count, 1)
        if 0.4 <= unique_ratio <= 0.6:  # Good vocabulary diversity
            score += 0.10
        elif 0.3 <= unique_ratio < 0.4 or 0.6 < unique_ratio <= 0.7:
            score += 0.07
        else:
            score += 0.03
        
        # Emotional appeal (5% weight)
        emotion_words = ['amazing', 'incredible', 'shocking', 'surprising', 'exciting',
                        'fascinating', 'unbelievable', 'mind-blowing', 'revolutionary']
        emotion_count = sum(1 for word in emotion_words if word in script.lower())
        score += min(0.05, (emotion_count / 10) * 0.05)
        
        return min(score, 1.0)
    
    def _score_voice(self, voice_params: Dict[str, Any]) -> float:
        """Score voice quality parameters with detailed analysis"""
        if not voice_params:
            return 0.3  # Minimal score if no params
        
        score = 0.0
        
        # Voice clarity (25% weight)
        clarity = voice_params.get('clarity', 0.5)
        if clarity > 0.9:
            score += 0.25
        elif clarity > 0.8:
            score += 0.20
        elif clarity > 0.7:
            score += 0.15
        elif clarity > 0.6:
            score += 0.10
        else:
            score += 0.05
        
        # Speaking pace (20% weight)
        pace = voice_params.get('pace', 1.0)
        if 0.9 <= pace <= 1.1:  # Optimal pace
            score += 0.20
        elif 0.8 <= pace < 0.9 or 1.1 < pace <= 1.2:
            score += 0.15
        elif 0.7 <= pace < 0.8 or 1.2 < pace <= 1.3:
            score += 0.10
        else:
            score += 0.05
        
        # Emotion variance (20% weight)
        emotion_variance = voice_params.get('emotion_variance', 0)
        if emotion_variance > 0.4:
            score += 0.20
        elif emotion_variance > 0.3:
            score += 0.15
        elif emotion_variance > 0.2:
            score += 0.10
        else:
            score += 0.05
        
        # Volume consistency (15% weight)
        volume_consistency = voice_params.get('volume_consistency', 0.5)
        if volume_consistency > 0.8:
            score += 0.15
        elif volume_consistency > 0.6:
            score += 0.10
        else:
            score += 0.05
        
        # Pronunciation accuracy (10% weight)
        pronunciation = voice_params.get('pronunciation_score', 0.5)
        if pronunciation > 0.9:
            score += 0.10
        elif pronunciation > 0.7:
            score += 0.07
        else:
            score += 0.03
        
        # Background noise level (10% weight)
        noise_level = voice_params.get('background_noise', 1.0)
        if noise_level < 0.1:  # Very quiet background
            score += 0.10
        elif noise_level < 0.3:
            score += 0.07
        elif noise_level < 0.5:
            score += 0.04
        
        return min(score, 1.0)
    
    def _score_visuals(self, visuals: List[Dict[str, Any]]) -> float:
        """Score visual quality and relevance"""
        if not visuals:
            return 0.0
        
        score = 0.3  # Base score for having visuals
        
        # Visual diversity
        if len(set(v.get('type') for v in visuals)) >= 3:
            score += 0.2
        
        # Quality check
        high_quality = sum(1 for v in visuals if v.get('quality', 0) > 0.8)
        score += min(high_quality / len(visuals) * 0.3, 0.3)
        
        # Transition quality
        if all(v.get('transition') for v in visuals[1:]):
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_thumbnail(self, thumbnail: Dict[str, Any]) -> float:
        """Score thumbnail effectiveness"""
        score = 0.0
        
        if thumbnail.get('has_text', False):
            score += 0.25
        
        if thumbnail.get('contrast_ratio', 0) > 4.5:
            score += 0.25
        
        if thumbnail.get('face_detected', False):
            score += 0.2
        
        if thumbnail.get('emotion_intensity', 0) > 0.7:
            score += 0.15
        
        if thumbnail.get('brand_consistent', True):
            score += 0.15
        
        return min(score, 1.0)
    
    def _score_seo(self, content: Dict[str, Any]) -> float:
        """Score SEO optimization"""
        score = 0.0
        
        # Title optimization
        title = content.get('title', '')
        if 50 <= len(title) <= 60:
            score += 0.2
        if any(char in title for char in ['|', '-', ':']):
            score += 0.1
        
        # Description optimization
        description = content.get('description', '')
        if 150 <= len(description) <= 160:
            score += 0.2
        
        # Tags optimization
        tags = content.get('tags', [])
        if 5 <= len(tags) <= 15:
            score += 0.2
        
        # Keyword density
        if content.get('keyword_density', 0) > 0.02:
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_engagement_potential(self, content: Dict[str, Any]) -> float:
        """Score potential for audience engagement with comprehensive metrics"""
        score = 0.0
        
        # Trending topic analysis (20% weight)
        if content.get('is_trending', False):
            trend_score = content.get('trend_score', 0.5)
            score += min(0.20, trend_score * 0.20)
        else:
            # Check for evergreen content value
            if content.get('is_evergreen', False):
                score += 0.10
        
        # Emotional appeal (15% weight)
        emotion_score = content.get('emotion_score', 0)
        if emotion_score > 0.8:
            score += 0.15
        elif emotion_score > 0.6:
            score += 0.12
        elif emotion_score > 0.4:
            score += 0.08
        else:
            score += 0.04
        
        # Interactive elements (20% weight)
        interaction_score = 0
        if content.get('has_poll', False):
            interaction_score += 0.07
        if content.get('has_quiz', False):
            interaction_score += 0.07
        if content.get('has_challenge', False):
            interaction_score += 0.06
        if content.get('has_call_to_action', False):
            interaction_score += 0.05
        score += min(0.20, interaction_score)
        
        # Optimal duration (15% weight)
        duration = content.get('duration_seconds', 0)
        if 480 <= duration <= 720:  # 8-12 minutes
            score += 0.15
        elif 300 <= duration < 480 or 720 < duration <= 900:  # 5-8 or 12-15 minutes
            score += 0.12
        elif 180 <= duration < 300 or 900 < duration <= 1200:  # 3-5 or 15-20 minutes
            score += 0.08
        else:
            score += 0.04
        
        # Release timing (10% weight)
        release_time = content.get('release_time_score', 0.5)
        score += release_time * 0.10
        
        # Target audience match (10% weight)
        audience_match = content.get('audience_match_score', 0.5)
        score += audience_match * 0.10
        
        # Shareability factors (10% weight)
        shareability = 0
        if content.get('has_surprising_fact', False):
            shareability += 0.04
        if content.get('has_useful_tips', False):
            shareability += 0.03
        if content.get('has_entertainment_value', False):
            shareability += 0.03
        score += shareability
        
        return min(score, 1.0)
    
    def _generate_recommendations(self, *scores) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        script_score, voice_score, visual_score, thumbnail_score, seo_score, engagement_score = scores
        
        if script_score < 0.7:
            recommendations.append("Improve script structure with clear intro, body, and conclusion")
        
        if voice_score < 0.7:
            recommendations.append("Adjust voice pacing and add more emotional variation")
        
        if visual_score < 0.7:
            recommendations.append("Add more diverse and high-quality visuals")
        
        if thumbnail_score < 0.7:
            recommendations.append("Create more compelling thumbnail with text overlay and high contrast")
        
        if seo_score < 0.7:
            recommendations.append("Optimize title, description, and tags for better SEO")
        
        if engagement_score < 0.7:
            recommendations.append("Consider trending topics and add interactive elements")
        
        return recommendations