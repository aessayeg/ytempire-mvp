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
        """Score script quality"""
        if not script:
            return 0.0
        
        score = 0.0
        
        # Length check (optimal 1500-2500 words)
        word_count = len(script.split())
        if 1500 <= word_count <= 2500:
            score += 0.3
        elif 1000 <= word_count < 1500 or 2500 < word_count <= 3000:
            score += 0.2
        else:
            score += 0.1
        
        # Hook quality (first 15 seconds)
        hook = ' '.join(script.split()[:30])
        if any(word in hook.lower() for word in ['discover', 'learn', 'secret', 'amazing', 'how to']):
            score += 0.2
        
        # Call-to-action presence
        if any(phrase in script.lower() for phrase in ['subscribe', 'like', 'comment', 'share']):
            score += 0.2
        
        # Structure (intro, body, conclusion)
        if len(script.split('\n\n')) >= 3:
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_voice(self, voice_params: Dict[str, Any]) -> float:
        """Score voice quality parameters"""
        score = 0.5  # Base score
        
        if voice_params.get('clarity', 0) > 0.8:
            score += 0.2
        
        if 0.8 <= voice_params.get('pace', 1.0) <= 1.2:
            score += 0.15
        
        if voice_params.get('emotion_variance', 0) > 0.3:
            score += 0.15
        
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
        """Score potential for audience engagement"""
        score = 0.0
        
        # Trending topic
        if content.get('is_trending', False):
            score += 0.3
        
        # Emotional appeal
        if content.get('emotion_score', 0) > 0.6:
            score += 0.2
        
        # Interactive elements
        if content.get('has_poll', False) or content.get('has_quiz', False):
            score += 0.2
        
        # Optimal duration (8-12 minutes)
        duration = content.get('duration_seconds', 0)
        if 480 <= duration <= 720:
            score += 0.3
        
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