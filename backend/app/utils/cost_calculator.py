"""
Cost Calculator Utility
Owner: VP of AI

Calculate costs for various AI services and operations.
Ensures videos stay under $3 budget limit.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceProvider(Enum):
    """Supported AI service providers."""
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    AZURE_TTS = "azure_tts"
    GOOGLE_TTS = "google_tts"
    STABILITY_AI = "stability_ai"
    REPLICATE = "replicate"
    ANTHROPIC = "anthropic"


class CostCalculator:
    """Calculate costs for AI services and track budgets."""
    
    def __init__(self):
        # Current pricing (as of 2025) - update as needed
        self.pricing = {
            # OpenAI pricing (per 1K tokens)
            ServiceProvider.OPENAI: {
                'gpt-4': {'input': 0.03, 'output': 0.06},
                'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
                'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
                'dalle-3-hd': 0.080,  # per image (1024x1024)
                'dalle-3-standard': 0.040,  # per image (1024x1024)
                'dalle-2': 0.020,  # per image (1024x1024)
                'whisper': 0.006,  # per minute
                'tts-1': 0.015,  # per 1K characters
                'tts-1-hd': 0.030,  # per 1K characters
            },
            
            # ElevenLabs pricing (per 1K characters)
            ServiceProvider.ELEVENLABS: {
                'turbo': 0.20,
                'multilingual': 0.30,
                'english': 0.22,
                'clone': 0.30
            },
            
            # Azure TTS pricing (per 1M characters)
            ServiceProvider.AZURE_TTS: {
                'neural': 16.0,   # per 1M characters
                'standard': 4.0,  # per 1M characters
            },
            
            # Google TTS pricing (per 1M characters)
            ServiceProvider.GOOGLE_TTS: {
                'wavenet': 16.0,    # per 1M characters
                'neural2': 16.0,    # per 1M characters
                'standard': 4.0,    # per 1M characters
            },
            
            # Stability AI pricing
            ServiceProvider.STABILITY_AI: {
                'stable-diffusion-xl': 0.040,  # per image
                'stable-diffusion-v1-6': 0.020,  # per image
            },
            
            # Replicate pricing (approximate)
            ServiceProvider.REPLICATE: {
                'llama-2-70b': 0.0013,  # per second
                'stable-diffusion': 0.0023,  # per second
                'whisper': 0.0001,  # per second
            }
        }
        
        # Budget limits
        self.MAX_VIDEO_COST = 3.00
        self.MAX_SCRIPT_COST = 0.50
        self.MAX_AUDIO_COST = 1.20
        self.MAX_IMAGE_COST = 0.80
        self.MAX_VIDEO_PROCESSING_COST = 0.50
    
    def calculate_openai_cost(
        self, 
        tokens_used: int, 
        model: str = "gpt-4", 
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None
    ) -> float:
        """Calculate cost for OpenAI API usage."""
        
        try:
            if model not in self.pricing[ServiceProvider.OPENAI]:
                logger.warning(f"Unknown OpenAI model: {model}, using gpt-4 pricing")
                model = "gpt-4"
            
            model_pricing = self.pricing[ServiceProvider.OPENAI][model]
            
            if isinstance(model_pricing, dict) and 'input' in model_pricing:
                # Token-based pricing with separate input/output costs
                if input_tokens is not None and output_tokens is not None:
                    cost = (
                        (input_tokens / 1000) * model_pricing['input'] +
                        (output_tokens / 1000) * model_pricing['output']
                    )
                else:
                    # Estimate 70% input, 30% output if not specified
                    estimated_input = int(tokens_used * 0.7)
                    estimated_output = int(tokens_used * 0.3)
                    cost = (
                        (estimated_input / 1000) * model_pricing['input'] +
                        (estimated_output / 1000) * model_pricing['output']
                    )
            else:
                # Simple per-token or per-unit pricing
                cost = (tokens_used / 1000) * model_pricing
            
            return round(cost, 6)
            
        except Exception as e:
            logger.error(f"Failed to calculate OpenAI cost: {str(e)}")
            return 0.05  # Default conservative estimate
    
    def calculate_elevenlabs_cost(
        self, 
        character_count: int, 
        voice_model: str = "english"
    ) -> float:
        """Calculate cost for ElevenLabs voice synthesis."""
        
        try:
            if voice_model not in self.pricing[ServiceProvider.ELEVENLABS]:
                logger.warning(f"Unknown ElevenLabs model: {voice_model}, using english pricing")
                voice_model = "english"
            
            price_per_1k = self.pricing[ServiceProvider.ELEVENLABS][voice_model]
            cost = (character_count / 1000) * price_per_1k
            
            return round(cost, 4)
            
        except Exception as e:
            logger.error(f"Failed to calculate ElevenLabs cost: {str(e)}")
            return character_count * 0.0003  # Conservative estimate
    
    def calculate_azure_tts_cost(
        self, 
        character_count: int, 
        voice_type: str = "neural"
    ) -> float:
        """Calculate cost for Azure Text-to-Speech."""
        
        try:
            if voice_type not in self.pricing[ServiceProvider.AZURE_TTS]:
                voice_type = "neural"
            
            price_per_1m = self.pricing[ServiceProvider.AZURE_TTS][voice_type]
            cost = (character_count / 1000000) * price_per_1m
            
            return round(cost, 6)
            
        except Exception as e:
            logger.error(f"Failed to calculate Azure TTS cost: {str(e)}")
            return character_count * 0.000016  # Conservative estimate
    
    def calculate_google_tts_cost(
        self, 
        character_count: int, 
        voice_type: str = "neural2"
    ) -> float:
        """Calculate cost for Google Text-to-Speech."""
        
        try:
            if voice_type not in self.pricing[ServiceProvider.GOOGLE_TTS]:
                voice_type = "neural2"
            
            price_per_1m = self.pricing[ServiceProvider.GOOGLE_TTS][voice_type]
            cost = (character_count / 1000000) * price_per_1m
            
            return round(cost, 6)
            
        except Exception as e:
            logger.error(f"Failed to calculate Google TTS cost: {str(e)}")
            return character_count * 0.000016  # Conservative estimate
    
    def calculate_image_generation_cost(
        self, 
        provider: ServiceProvider, 
        model: str, 
        image_count: int = 1,
        resolution: str = "1024x1024"
    ) -> float:
        """Calculate cost for image generation."""
        
        try:
            if provider == ServiceProvider.OPENAI:
                if model not in self.pricing[provider]:
                    model = "dalle-3-standard"
                
                base_cost = self.pricing[provider][model]
                
                # Adjust for resolution if needed
                if "512x512" in resolution and "dalle-3" in model:
                    base_cost *= 0.5  # DALL-E 3 is cheaper for smaller images
                
                return round(base_cost * image_count, 4)
                
            elif provider == ServiceProvider.STABILITY_AI:
                if model not in self.pricing[provider]:
                    model = "stable-diffusion-xl"
                
                base_cost = self.pricing[provider][model]
                return round(base_cost * image_count, 4)
                
            else:
                logger.warning(f"Image generation pricing not available for {provider}")
                return 0.04 * image_count  # Conservative estimate
                
        except Exception as e:
            logger.error(f"Failed to calculate image generation cost: {str(e)}")
            return 0.04 * image_count  # Conservative estimate
    
    def calculate_video_processing_cost(
        self, 
        duration_seconds: int, 
        resolution: str = "1080p",
        complexity: str = "medium"
    ) -> float:
        """Calculate cost for video processing and compilation."""
        
        try:
            # Base cost per minute of video
            base_cost_per_minute = {
                "720p": 0.02,
                "1080p": 0.03,
                "4k": 0.08
            }
            
            # Complexity multipliers
            complexity_multipliers = {
                "simple": 0.8,
                "medium": 1.0,
                "complex": 1.5
            }
            
            duration_minutes = duration_seconds / 60
            base_cost = base_cost_per_minute.get(resolution, 0.03)
            complexity_mult = complexity_multipliers.get(complexity, 1.0)
            
            total_cost = duration_minutes * base_cost * complexity_mult
            
            return round(total_cost, 4)
            
        except Exception as e:
            logger.error(f"Failed to calculate video processing cost: {str(e)}")
            return (duration_seconds / 60) * 0.03  # Conservative estimate
    
    def estimate_total_video_cost(
        self, 
        script_length: int,
        audio_length_seconds: int,
        image_count: int,
        video_duration_seconds: int,
        services_config: Dict[str, str]
    ) -> Dict[str, float]:
        """Estimate total cost for video generation before processing."""
        
        try:
            costs = {}
            
            # Script generation cost
            if 'script_model' in services_config:
                model = services_config['script_model']
                estimated_tokens = min(script_length * 2, 4000)  # Rough estimate
                costs['script_generation'] = self.calculate_openai_cost(
                    estimated_tokens, model
                )
            else:
                costs['script_generation'] = 0.25  # Conservative estimate
            
            # Audio synthesis cost
            if 'audio_service' in services_config:
                service = services_config['audio_service']
                if service == 'elevenlabs':
                    costs['audio_synthesis'] = self.calculate_elevenlabs_cost(
                        script_length, services_config.get('voice_model', 'english')
                    )
                elif service == 'azure':
                    costs['audio_synthesis'] = self.calculate_azure_tts_cost(
                        script_length, services_config.get('voice_type', 'neural')
                    )
                elif service == 'google':
                    costs['audio_synthesis'] = self.calculate_google_tts_cost(
                        script_length, services_config.get('voice_type', 'neural2')
                    )
                else:
                    costs['audio_synthesis'] = script_length * 0.0003
            else:
                costs['audio_synthesis'] = 0.60  # Conservative estimate
            
            # Image generation cost
            if 'image_service' in services_config:
                provider_str = services_config['image_service']
                try:
                    provider = ServiceProvider(provider_str)
                    model = services_config.get('image_model', 'dalle-3-standard')
                    costs['image_generation'] = self.calculate_image_generation_cost(
                        provider, model, image_count
                    )
                except ValueError:
                    costs['image_generation'] = 0.04 * image_count
            else:
                costs['image_generation'] = 0.04 * image_count
            
            # Video processing cost
            costs['video_processing'] = self.calculate_video_processing_cost(
                video_duration_seconds,
                services_config.get('resolution', '1080p'),
                services_config.get('complexity', 'medium')
            )
            
            # Additional overhead (API calls, storage, etc.)
            costs['overhead'] = 0.05
            
            # Calculate total
            costs['total_estimated'] = sum(costs.values())
            
            # Add budget analysis
            costs['within_budget'] = costs['total_estimated'] <= self.MAX_VIDEO_COST
            costs['budget_remaining'] = self.MAX_VIDEO_COST - costs['total_estimated']
            
            return costs
            
        except Exception as e:
            logger.error(f"Failed to estimate total video cost: {str(e)}")
            return {
                'total_estimated': 2.50,  # Conservative estimate
                'within_budget': True,
                'budget_remaining': 0.50,
                'error': str(e)
            }
    
    def validate_cost_against_budget(
        self, 
        current_costs: Dict[str, float], 
        stage: str = "total"
    ) -> Dict[str, Any]:
        """Validate current costs against budget limits."""
        
        try:
            validation_result = {
                'valid': True,
                'warnings': [],
                'errors': [],
                'budget_utilization': {}
            }
            
            # Check stage-specific limits
            if stage == "script" or "script" in current_costs:
                script_cost = current_costs.get('script_generation', 0)
                if script_cost > self.MAX_SCRIPT_COST:
                    validation_result['errors'].append(
                        f"Script cost ${script_cost:.4f} exceeds limit ${self.MAX_SCRIPT_COST}"
                    )
                    validation_result['valid'] = False
                elif script_cost > self.MAX_SCRIPT_COST * 0.8:
                    validation_result['warnings'].append(
                        f"Script cost ${script_cost:.4f} is near limit ${self.MAX_SCRIPT_COST}"
                    )
                
                validation_result['budget_utilization']['script'] = script_cost / self.MAX_SCRIPT_COST
            
            if stage == "audio" or "audio" in current_costs:
                audio_cost = current_costs.get('audio_synthesis', 0)
                if audio_cost > self.MAX_AUDIO_COST:
                    validation_result['errors'].append(
                        f"Audio cost ${audio_cost:.4f} exceeds limit ${self.MAX_AUDIO_COST}"
                    )
                    validation_result['valid'] = False
                elif audio_cost > self.MAX_AUDIO_COST * 0.8:
                    validation_result['warnings'].append(
                        f"Audio cost ${audio_cost:.4f} is near limit ${self.MAX_AUDIO_COST}"
                    )
                
                validation_result['budget_utilization']['audio'] = audio_cost / self.MAX_AUDIO_COST
            
            if stage == "images" or "images" in current_costs:
                image_cost = current_costs.get('image_generation', 0)
                if image_cost > self.MAX_IMAGE_COST:
                    validation_result['errors'].append(
                        f"Image cost ${image_cost:.4f} exceeds limit ${self.MAX_IMAGE_COST}"
                    )
                    validation_result['valid'] = False
                elif image_cost > self.MAX_IMAGE_COST * 0.8:
                    validation_result['warnings'].append(
                        f"Image cost ${image_cost:.4f} is near limit ${self.MAX_IMAGE_COST}"
                    )
                
                validation_result['budget_utilization']['images'] = image_cost / self.MAX_IMAGE_COST
            
            # Check total cost
            total_cost = sum(current_costs.values())
            if total_cost > self.MAX_VIDEO_COST:
                validation_result['errors'].append(
                    f"Total cost ${total_cost:.4f} exceeds video budget ${self.MAX_VIDEO_COST}"
                )
                validation_result['valid'] = False
            elif total_cost > self.MAX_VIDEO_COST * 0.85:
                validation_result['warnings'].append(
                    f"Total cost ${total_cost:.4f} is near video budget ${self.MAX_VIDEO_COST}"
                )
            
            validation_result['budget_utilization']['total'] = total_cost / self.MAX_VIDEO_COST
            validation_result['total_cost'] = total_cost
            validation_result['remaining_budget'] = self.MAX_VIDEO_COST - total_cost
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to validate cost against budget: {str(e)}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'budget_utilization': {}
            }
    
    def get_cost_optimization_suggestions(
        self, 
        current_costs: Dict[str, float],
        services_config: Dict[str, str]
    ) -> List[str]:
        """Get suggestions for optimizing costs."""
        
        suggestions = []
        
        try:
            total_cost = sum(current_costs.values())
            
            # Script optimization
            script_cost = current_costs.get('script_generation', 0)
            if script_cost > self.MAX_SCRIPT_COST * 0.7:
                if services_config.get('script_model') == 'gpt-4':
                    suggestions.append("Consider using GPT-4 Turbo or GPT-3.5 Turbo for script generation to reduce costs")
                suggestions.append("Optimize script prompts to reduce token usage")
            
            # Audio optimization
            audio_cost = current_costs.get('audio_synthesis', 0)
            if audio_cost > self.MAX_AUDIO_COST * 0.7:
                if services_config.get('audio_service') == 'elevenlabs':
                    suggestions.append("Consider Azure TTS or Google TTS for lower audio synthesis costs")
                suggestions.append("Split long scripts into shorter segments to optimize processing")
            
            # Image optimization
            image_cost = current_costs.get('image_generation', 0)
            if image_cost > self.MAX_IMAGE_COST * 0.7:
                suggestions.append("Reduce number of generated images or use lower resolution")
                if 'dalle-3-hd' in services_config.get('image_model', ''):
                    suggestions.append("Use DALL-E 3 standard instead of HD for cost savings")
            
            # Overall optimization
            if total_cost > self.MAX_VIDEO_COST * 0.85:
                suggestions.append("Consider reducing video length to lower overall costs")
                suggestions.append("Use more cost-effective AI models where quality allows")
                suggestions.append("Optimize content complexity to reduce processing costs")
            
            if not suggestions:
                suggestions.append("Costs are well within budget limits")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate cost optimization suggestions: {str(e)}")
            return ["Unable to generate cost optimization suggestions"]


# Global instance for easy import
cost_calculator = CostCalculator()