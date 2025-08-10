"""
Content Generation Tasks
Owner: Data Pipeline Engineer #1

AI-powered content generation for video scripts, titles, and descriptions.
Integrates with OpenAI GPT-4 and other AI services.
"""

from celery import current_task
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
import openai
from textblob import TextBlob

from app.core.celery_app import celery_app
from app.core.config import settings
from app.services.ai_service import AIService, ContentGenerationRequest
from app.models.video import Video
from app.core.database import get_db
from app.utils.cost_calculator import CostCalculator

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def generate_content_task(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate video content including script, title, description, and visual prompts.
    
    Args:
        pipeline_data: Pipeline data from previous task
        
    Returns:
        Content generation results with costs
    """
    try:
        video_id = pipeline_data['id']
        topic = pipeline_data['topic']
        channel_id = pipeline_data['channel_id']
        
        logger.info(f"Generating content for video: {video_id}, topic: {topic}")
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'content_generation', 'progress': 10, 'video_id': video_id}
        )
        
        # Get channel information for context
        db = next(get_db())
        from app.models.channel import Channel
        channel = db.query(Channel).filter(Channel.id == channel_id).first()
        
        channel_context = {
            'name': channel.name if channel else 'Unknown Channel',
            'category': channel.category if channel else 'general',
            'target_audience': channel.target_audience if channel else 'general audience',
            'tone': channel.tone if channel else 'informative'
        }
        
        # Initialize AI service
        ai_service = AIService()
        cost_calculator = CostCalculator()
        
        # Generate video script
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'script_generation', 'progress': 30, 'video_id': video_id}
        )
        
        script_request = ContentGenerationRequest(
            type='script',
            topic=topic,
            channel_context=channel_context,
            target_duration=pipeline_data.get('target_duration', 480),  # 8 minutes default
            max_cost=settings.MAX_SCRIPT_COST
        )
        
        script_result = ai_service.generate_content(script_request)
        script_cost = cost_calculator.calculate_openai_cost(
            script_result['tokens_used'],
            model=script_result['model']
        )
        
        # Generate title and description
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'metadata_generation', 'progress': 50, 'video_id': video_id}
        )
        
        metadata_request = ContentGenerationRequest(
            type='metadata',
            topic=topic,
            script=script_result['content'],
            channel_context=channel_context,
            max_cost=0.20  # Lower cost for metadata
        )
        
        metadata_result = ai_service.generate_content(metadata_request)
        metadata_cost = cost_calculator.calculate_openai_cost(
            metadata_result['tokens_used'],
            model=metadata_result['model']
        )
        
        # Generate visual prompts
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'visual_prompts', 'progress': 70, 'video_id': video_id}
        )
        
        visual_prompts_request = ContentGenerationRequest(
            type='visual_prompts',
            script=script_result['content'],
            topic=topic,
            max_cost=0.15
        )
        
        visual_prompts_result = ai_service.generate_content(visual_prompts_request)
        visual_prompts_cost = cost_calculator.calculate_openai_cost(
            visual_prompts_result['tokens_used'],
            model=visual_prompts_result['model']
        )
        
        # Analyze content quality
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'quality_analysis', 'progress': 85, 'video_id': video_id}
        )
        
        quality_score = analyze_content_quality(
            script_result['content'],
            metadata_result['content']['title'],
            topic
        )
        
        # Calculate total content generation cost
        total_content_cost = script_cost + metadata_cost + visual_prompts_cost
        
        # Validate cost doesn't exceed budget
        if total_content_cost > settings.MAX_SCRIPT_COST:
            logger.warning(f"Content generation exceeded budget: ${total_content_cost:.4f}")
            # Don't fail, but log warning
        
        # Prepare result
        result = {
            **pipeline_data,
            'script': script_result['content'],
            'title': metadata_result['content']['title'],
            'description': metadata_result['content']['description'],
            'tags': metadata_result['content'].get('tags', []),
            'visual_prompts': visual_prompts_result['content']['prompts'],
            'content_quality_score': quality_score,
            'content_cost': total_content_cost,
            'content_generation_completed': True,
            'content_metadata': {
                'script_tokens': script_result['tokens_used'],
                'script_model': script_result['model'],
                'generated_at': datetime.utcnow().isoformat(),
                'quality_metrics': quality_score
            }
        }
        
        # Update video record
        video = db.query(Video).filter(Video.id == video_id).first()
        if video:
            video.title = result['title']
            video.description = result['description']
            video.script = result['script']
            video.content_cost = total_content_cost
            video.quality_score = quality_score['overall_score']
            db.commit()
        
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'content_complete', 'progress': 100, 'video_id': video_id}
        )
        
        logger.info(f"Content generation completed for video: {video_id}, cost: ${total_content_cost:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Content generation failed for video {pipeline_data.get('id')}: {str(e)}")
        raise


def analyze_content_quality(script: str, title: str, topic: str) -> Dict[str, Any]:
    """
    Analyze the quality of generated content using various metrics.
    
    Args:
        script: Generated video script
        title: Generated video title
        topic: Original topic
        
    Returns:
        Quality analysis results
    """
    try:
        # Text analysis using TextBlob
        script_blob = TextBlob(script)
        title_blob = TextBlob(title)
        
        # Calculate metrics
        metrics = {
            'script_length': len(script.split()),
            'script_sentences': len(script_blob.sentences),
            'script_sentiment': {
                'polarity': script_blob.sentiment.polarity,
                'subjectivity': script_blob.sentiment.subjectivity
            },
            'title_sentiment': {
                'polarity': title_blob.sentiment.polarity,
                'subjectivity': title_blob.sentiment.subjectivity
            },
            'readability': calculate_readability(script),
            'topic_relevance': calculate_topic_relevance(script, topic),
            'engagement_potential': calculate_engagement_potential(title, script)
        }
        
        # Calculate overall quality score (0-100)
        overall_score = calculate_overall_quality_score(metrics)
        
        return {
            'overall_score': overall_score,
            'metrics': metrics,
            'recommendations': generate_quality_recommendations(metrics)
        }
        
    except Exception as e:
        logger.error(f"Quality analysis failed: {str(e)}")
        return {
            'overall_score': 50,  # Default neutral score
            'metrics': {},
            'recommendations': ['Quality analysis unavailable']
        }


def calculate_readability(text: str) -> float:
    """Calculate text readability score (simplified Flesch Reading Ease)."""
    try:
        words = len(text.split())
        sentences = text.count('.') + text.count('!') + text.count('?')
        syllables = sum([count_syllables(word) for word in text.split()])
        
        if sentences == 0 or words == 0:
            return 50.0  # Neutral score
        
        # Simplified Flesch Reading Ease
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, score))
        
    except:
        return 50.0


def count_syllables(word: str) -> int:
    """Count syllables in a word (simplified)."""
    word = word.lower()
    vowels = 'aeiouy'
    syllable_count = 0
    previous_was_vowel = False
    
    for char in word:
        if char in vowels:
            if not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = True
        else:
            previous_was_vowel = False
    
    if word.endswith('e'):
        syllable_count -= 1
    
    return max(1, syllable_count)


def calculate_topic_relevance(script: str, topic: str) -> float:
    """Calculate how relevant the script is to the topic."""
    try:
        script_words = set(script.lower().split())
        topic_words = set(topic.lower().split())
        
        # Simple word overlap calculation
        overlap = len(script_words.intersection(topic_words))
        total_topic_words = len(topic_words)
        
        if total_topic_words == 0:
            return 50.0
        
        relevance_score = min(100, (overlap / total_topic_words) * 100)
        return relevance_score
        
    except:
        return 50.0


def calculate_engagement_potential(title: str, script: str) -> float:
    """Calculate potential engagement based on title and content characteristics."""
    try:
        score = 50.0  # Base score
        
        # Title analysis
        title_lower = title.lower()
        engagement_words = [
            'amazing', 'incredible', 'shocking', 'secret', 'revealed',
            'best', 'worst', 'top', 'ultimate', 'perfect', 'easy',
            'quick', 'simple', 'advanced', 'complete', 'step-by-step'
        ]
        
        title_engagement = sum(1 for word in engagement_words if word in title_lower)
        score += min(20, title_engagement * 3)
        
        # Title length (optimal 50-60 characters)
        title_length_score = 100 - abs(len(title) - 55)
        score += min(10, title_length_score * 0.2)
        
        # Script structure analysis
        script_sentences = script.count('.') + script.count('!') + script.count('?')
        script_words = len(script.split())
        
        if script_sentences > 0:
            avg_sentence_length = script_words / script_sentences
            # Optimal sentence length is 15-20 words
            sentence_score = 100 - abs(avg_sentence_length - 17.5) * 2
            score += min(15, max(0, sentence_score) * 0.15)
        
        return min(100, max(0, score))
        
    except:
        return 50.0


def calculate_overall_quality_score(metrics: Dict[str, Any]) -> float:
    """Calculate overall quality score from individual metrics."""
    try:
        weights = {
            'readability': 0.25,
            'topic_relevance': 0.30,
            'engagement_potential': 0.25,
            'sentiment_balance': 0.20
        }
        
        # Normalize sentiment to 0-100 scale
        script_sentiment = metrics.get('script_sentiment', {}).get('polarity', 0)
        sentiment_score = (script_sentiment + 1) * 50  # Convert -1,1 to 0,100
        
        scores = {
            'readability': metrics.get('readability', 50),
            'topic_relevance': metrics.get('topic_relevance', 50),
            'engagement_potential': metrics.get('engagement_potential', 50),
            'sentiment_balance': sentiment_score
        }
        
        overall = sum(scores[metric] * weights[metric] for metric in weights.keys())
        return round(overall, 2)
        
    except:
        return 50.0


def generate_quality_recommendations(metrics: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on quality metrics."""
    recommendations = []
    
    try:
        if metrics.get('readability', 50) < 40:
            recommendations.append("Consider simplifying sentence structure for better readability")
        
        if metrics.get('topic_relevance', 50) < 30:
            recommendations.append("Increase focus on the main topic throughout the script")
        
        if metrics.get('engagement_potential', 50) < 40:
            recommendations.append("Add more engaging elements to the title and content")
        
        script_sentiment = metrics.get('script_sentiment', {}).get('polarity', 0)
        if script_sentiment < -0.3:
            recommendations.append("Consider adding more positive elements to balance tone")
        elif script_sentiment > 0.7:
            recommendations.append("Consider adding some critical analysis for balance")
        
        if metrics.get('script_length', 0) < 100:
            recommendations.append("Script may be too short - consider expanding content")
        elif metrics.get('script_length', 0) > 1000:
            recommendations.append("Script may be too long - consider condensing key points")
        
        if not recommendations:
            recommendations.append("Content quality looks good!")
            
    except:
        recommendations = ["Unable to generate specific recommendations"]
    
    return recommendations