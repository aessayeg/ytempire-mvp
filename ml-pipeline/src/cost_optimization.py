"""
Cost Optimization Strategy for YTEmpire
Target: <$3 per video
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ServiceCost:
    """Cost structure for AI service"""
    service: str
    model: str
    input_cost: float  # per 1K tokens/chars
    output_cost: float  # per 1K tokens/chars
    monthly_free_tier: float
    

@dataclass 
class VideoCostBreakdown:
    """Detailed cost breakdown for video"""
    script_generation: float
    voice_synthesis: float
    thumbnail_generation: float
    video_processing: float
    api_calls: float
    total: float
    

class CostOptimizer:
    """Optimize costs across AI services"""
    
    def __init__(self):
        # Service cost definitions
        self.service_costs = {
            'openai': {
                'gpt-4-turbo': ServiceCost('openai', 'gpt-4-turbo', 0.01, 0.03, 0),
                'gpt-3.5-turbo': ServiceCost('openai', 'gpt-3.5-turbo', 0.0005, 0.0015, 0),
                'dall-e-3': ServiceCost('openai', 'dall-e-3', 0.04, 0.08, 0),  # per image
            },
            'anthropic': {
                'claude-3-opus': ServiceCost('anthropic', 'claude-3-opus', 0.015, 0.075, 0),
                'claude-3-sonnet': ServiceCost('anthropic', 'claude-3-sonnet', 0.003, 0.015, 0),
            },
            'elevenlabs': {
                'standard': ServiceCost('elevenlabs', 'standard', 0.00018, 0, 10000),  # per char
                'premium': ServiceCost('elevenlabs', 'premium', 0.00030, 0, 0),
            },
            'google': {
                'tts-standard': ServiceCost('google', 'tts-standard', 0.000004, 0, 1000000),  # per char
                'tts-wavenet': ServiceCost('google', 'tts-wavenet', 0.000016, 0, 1000000),
            }
        }
        
        # Usage tracking
        self.daily_usage = {}
        self.monthly_usage = {}
        self.reset_usage_tracking()
        
        # Cost limits
        self.max_cost_per_video = 3.00
        self.daily_budget = 100.00
        self.monthly_budget = 10000.00
        
    def calculate_script_cost(self, 
                            prompt_tokens: int,
                            completion_tokens: int,
                            model: str = 'gpt-3.5-turbo') -> float:
        """Calculate cost for script generation"""
        service = self._get_service_from_model(model)
        cost_info = self.service_costs[service][model]
        
        input_cost = (prompt_tokens / 1000) * cost_info.input_cost
        output_cost = (completion_tokens / 1000) * cost_info.output_cost
        
        total_cost = input_cost + output_cost
        
        # Apply fallback strategy if too expensive
        if total_cost > 0.50:  # Script budget: $0.50
            logger.warning(f"Script cost ${total_cost:.2f} exceeds budget, using cheaper model")
            return self._get_fallback_script_cost(prompt_tokens, completion_tokens)
            
        return total_cost
        
    def calculate_voice_cost(self,
                           text_length: int,
                           voice_provider: str = 'elevenlabs',
                           quality: str = 'standard') -> float:
        """Calculate cost for voice synthesis"""
        cost_info = self.service_costs[voice_provider][quality]
        
        # Check free tier
        remaining_free = max(0, cost_info.monthly_free_tier - 
                            self.monthly_usage.get(f'{voice_provider}_chars', 0))
        
        billable_chars = max(0, text_length - remaining_free)
        cost = billable_chars * cost_info.input_cost
        
        # Fallback to cheaper option if needed
        if cost > 1.00:  # Voice budget: $1.00
            logger.warning(f"Voice cost ${cost:.2f} exceeds budget, using Google TTS")
            return self._calculate_google_tts_cost(text_length)
            
        # Update usage
        self.monthly_usage[f'{voice_provider}_chars'] = \
            self.monthly_usage.get(f'{voice_provider}_chars', 0) + text_length
            
        return cost
        
    def calculate_thumbnail_cost(self,
                                quality: str = 'hd',
                                size: str = '1024x1024') -> float:
        """Calculate cost for thumbnail generation"""
        # DALL-E 3 pricing
        price_map = {
            ('hd', '1024x1024'): 0.080,
            ('hd', '1792x1024'): 0.120,
            ('standard', '1024x1024'): 0.040,
        }
        
        cost = price_map.get((quality, size), 0.040)
        
        # Thumbnail budget: $0.12
        if cost > 0.12:
            logger.warning("Using standard quality for thumbnail to save costs")
            cost = 0.040
            
        return cost
        
    def calculate_video_processing_cost(self,
                                       duration_seconds: int,
                                       resolution: str = '1080p') -> float:
        """Calculate GPU/processing costs for video assembly"""
        # Estimated based on local GPU usage
        gpu_cost_per_hour = 0.50  # Local GPU electricity + amortization
        
        # Processing time estimation
        processing_time_hours = duration_seconds / 3600 * 2  # 2x realtime
        
        if resolution == '4k':
            processing_time_hours *= 2
        elif resolution == '720p':
            processing_time_hours *= 0.7
            
        return gpu_cost_per_hour * processing_time_hours
        
    def get_optimal_model_selection(self,
                                   task: str,
                                   quality_requirement: float) -> str:
        """Select optimal model based on cost and quality requirements"""
        model_options = {
            'script': [
                ('gpt-3.5-turbo', 0.7, 0.002),  # (model, quality, cost/1K)
                ('gpt-4-turbo', 0.95, 0.02),
                ('claude-3-sonnet', 0.85, 0.009),
            ],
            'voice': [
                ('google-standard', 0.7, 0.000004),
                ('google-wavenet', 0.85, 0.000016),
                ('elevenlabs-standard', 0.95, 0.00018),
            ],
        }
        
        options = model_options.get(task, [])
        
        # Filter by quality requirement
        valid_options = [(m, q, c) for m, q, c in options if q >= quality_requirement]
        
        if not valid_options:
            # Use best available if quality requirement too high
            valid_options = options
            
        # Sort by cost and return cheapest
        valid_options.sort(key=lambda x: x[2])
        return valid_options[0][0]
        
    def apply_cost_reduction_strategies(self,
                                       video_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply strategies to reduce video generation cost"""
        optimized_config = video_config.copy()
        
        # Strategy 1: Use cheaper models for non-critical content
        if video_config.get('content_type') == 'shorts':
            optimized_config['script_model'] = 'gpt-3.5-turbo'
            optimized_config['voice_quality'] = 'standard'
            
        # Strategy 2: Batch processing for better rates
        if self.daily_usage.get('videos', 0) > 10:
            optimized_config['batch_mode'] = True
            
        # Strategy 3: Cache and reuse common elements
        optimized_config['use_cache'] = True
        
        # Strategy 4: Time-based optimization (use cheaper services during off-peak)
        current_hour = datetime.now().hour
        if 2 <= current_hour <= 6:  # Off-peak hours
            optimized_config['quality_mode'] = 'economy'
            
        # Strategy 5: Progressive quality degradation if over budget
        daily_spent = self.get_daily_spent()
        if daily_spent > self.daily_budget * 0.8:
            logger.warning("Approaching daily budget, reducing quality")
            optimized_config['emergency_mode'] = True
            optimized_config['script_model'] = 'gpt-3.5-turbo'
            optimized_config['voice_provider'] = 'google'
            optimized_config['thumbnail_quality'] = 'standard'
            
        return optimized_config
        
    def estimate_video_cost(self, video_config: Dict[str, Any]) -> VideoCostBreakdown:
        """Estimate total cost for video generation"""
        # Script estimation (average 2000 tokens prompt, 1500 completion)
        script_cost = self.calculate_script_cost(
            2000, 1500, 
            video_config.get('script_model', 'gpt-3.5-turbo')
        )
        
        # Voice estimation (average 500 words = 2500 chars)
        voice_cost = self.calculate_voice_cost(
            2500,
            video_config.get('voice_provider', 'elevenlabs'),
            video_config.get('voice_quality', 'standard')
        )
        
        # Thumbnail
        thumbnail_cost = self.calculate_thumbnail_cost(
            video_config.get('thumbnail_quality', 'standard'),
            video_config.get('thumbnail_size', '1024x1024')
        )
        
        # Video processing
        processing_cost = self.calculate_video_processing_cost(
            video_config.get('duration', 600),
            video_config.get('resolution', '1080p')
        )
        
        # API calls (YouTube, analytics, etc.)
        api_cost = 0.01  # Estimated
        
        total = script_cost + voice_cost + thumbnail_cost + processing_cost + api_cost
        
        return VideoCostBreakdown(
            script_generation=script_cost,
            voice_synthesis=voice_cost,
            thumbnail_generation=thumbnail_cost,
            video_processing=processing_cost,
            api_calls=api_cost,
            total=total
        )
        
    def validate_cost_before_generation(self, 
                                       estimated_cost: float) -> Tuple[bool, str]:
        """Validate if video generation should proceed based on cost"""
        # Check per-video limit
        if estimated_cost > self.max_cost_per_video:
            return False, f"Exceeds per-video limit: ${estimated_cost:.2f} > ${self.max_cost_per_video}"
            
        # Check daily budget
        daily_spent = self.get_daily_spent()
        if daily_spent + estimated_cost > self.daily_budget:
            return False, f"Would exceed daily budget: ${daily_spent + estimated_cost:.2f} > ${self.daily_budget}"
            
        # Check monthly budget
        monthly_spent = self.get_monthly_spent()
        if monthly_spent + estimated_cost > self.monthly_budget:
            return False, f"Would exceed monthly budget: ${monthly_spent + estimated_cost:.2f} > ${self.monthly_budget}"
            
        return True, "Cost approved"
        
    def track_actual_cost(self, 
                         video_id: str,
                         cost_breakdown: VideoCostBreakdown):
        """Track actual costs after generation"""
        # Update daily tracking
        today = datetime.now().date().isoformat()
        if today not in self.daily_usage:
            self.daily_usage[today] = {'videos': 0, 'cost': 0.0}
            
        self.daily_usage[today]['videos'] += 1
        self.daily_usage[today]['cost'] += cost_breakdown.total
        
        # Update monthly tracking
        month = datetime.now().strftime('%Y-%m')
        if month not in self.monthly_usage:
            self.monthly_usage[month] = {'videos': 0, 'cost': 0.0}
            
        self.monthly_usage[month]['videos'] += 1
        self.monthly_usage[month]['cost'] += cost_breakdown.total
        
        # Log if approaching limits
        if cost_breakdown.total > self.max_cost_per_video * 0.9:
            logger.warning(f"Video {video_id} cost ${cost_breakdown.total:.2f} approaching limit")
            
    def get_cost_report(self) -> Dict[str, Any]:
        """Generate cost optimization report"""
        return {
            'daily': {
                'spent': self.get_daily_spent(),
                'budget': self.daily_budget,
                'remaining': self.daily_budget - self.get_daily_spent(),
                'videos_generated': self.daily_usage.get(
                    datetime.now().date().isoformat(), {}
                ).get('videos', 0)
            },
            'monthly': {
                'spent': self.get_monthly_spent(),
                'budget': self.monthly_budget,
                'remaining': self.monthly_budget - self.get_monthly_spent(),
                'videos_generated': self.monthly_usage.get(
                    datetime.now().strftime('%Y-%m'), {}
                ).get('videos', 0)
            },
            'average_cost_per_video': self._calculate_average_cost(),
            'cost_trend': self._calculate_cost_trend(),
            'optimization_suggestions': self._get_optimization_suggestions()
        }
        
    def _get_service_from_model(self, model: str) -> str:
        """Determine service provider from model name"""
        if 'gpt' in model or 'dall' in model:
            return 'openai'
        elif 'claude' in model:
            return 'anthropic'
        elif 'elevenlabs' in model:
            return 'elevenlabs'
        else:
            return 'google'
            
    def _get_fallback_script_cost(self, 
                                 prompt_tokens: int,
                                 completion_tokens: int) -> float:
        """Calculate cost using fallback model"""
        # Use GPT-3.5-turbo as fallback
        cost_info = self.service_costs['openai']['gpt-3.5-turbo']
        input_cost = (prompt_tokens / 1000) * cost_info.input_cost
        output_cost = (completion_tokens / 1000) * cost_info.output_cost
        return input_cost + output_cost
        
    def _calculate_google_tts_cost(self, text_length: int) -> float:
        """Calculate Google TTS cost as fallback"""
        cost_info = self.service_costs['google']['tts-standard']
        remaining_free = max(0, cost_info.monthly_free_tier - 
                            self.monthly_usage.get('google_chars', 0))
        billable_chars = max(0, text_length - remaining_free)
        
        self.monthly_usage['google_chars'] = \
            self.monthly_usage.get('google_chars', 0) + text_length
            
        return billable_chars * cost_info.input_cost
        
    def get_daily_spent(self) -> float:
        """Get total spent today"""
        today = datetime.now().date().isoformat()
        return self.daily_usage.get(today, {}).get('cost', 0.0)
        
    def get_monthly_spent(self) -> float:
        """Get total spent this month"""
        month = datetime.now().strftime('%Y-%m')
        return self.monthly_usage.get(month, {}).get('cost', 0.0)
        
    def _calculate_average_cost(self) -> float:
        """Calculate average cost per video"""
        month = datetime.now().strftime('%Y-%m')
        monthly_data = self.monthly_usage.get(month, {})
        
        videos = monthly_data.get('videos', 0)
        cost = monthly_data.get('cost', 0.0)
        
        return cost / videos if videos > 0 else 0.0
        
    def _calculate_cost_trend(self) -> str:
        """Calculate cost trend (increasing/decreasing/stable)"""
        # Compare last 7 days
        costs = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).date().isoformat()
            daily_cost = self.daily_usage.get(date, {}).get('cost', 0.0)
            costs.append(daily_cost)
            
        if len(costs) < 2:
            return 'insufficient_data'
            
        # Calculate trend
        recent_avg = np.mean(costs[:3])
        older_avg = np.mean(costs[3:])
        
        if recent_avg > older_avg * 1.1:
            return 'increasing'
        elif recent_avg < older_avg * 0.9:
            return 'decreasing'
        else:
            return 'stable'
            
    def _get_optimization_suggestions(self) -> List[str]:
        """Generate cost optimization suggestions"""
        suggestions = []
        
        avg_cost = self._calculate_average_cost()
        
        if avg_cost > 2.50:
            suggestions.append("Consider using GPT-3.5-turbo for script generation")
            suggestions.append("Switch to Google TTS for voice synthesis")
            
        if self.get_daily_spent() > self.daily_budget * 0.5:
            suggestions.append("Implement request batching to reduce API calls")
            suggestions.append("Enable aggressive caching for common content")
            
        monthly_spent = self.get_monthly_spent()
        if monthly_spent > self.monthly_budget * 0.7:
            suggestions.append("Consider reducing video resolution to 720p")
            suggestions.append("Use standard quality for thumbnails")
            suggestions.append("Implement content recycling for evergreen topics")
            
        return suggestions
        
    def reset_usage_tracking(self):
        """Reset usage tracking (for new billing period)"""
        # Check if new month
        current_month = datetime.now().strftime('%Y-%m')
        if current_month not in self.monthly_usage:
            self.monthly_usage[current_month] = {'videos': 0, 'cost': 0.0}
            
        # Clean old daily data (keep last 30 days)
        cutoff_date = (datetime.now() - timedelta(days=30)).date().isoformat()
        self.daily_usage = {
            date: data for date, data in self.daily_usage.items()
            if date >= cutoff_date
        }