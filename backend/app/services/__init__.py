"""
Service aliases for backward compatibility after consolidation
"""

# Video generation aliases
from .video_generation_pipeline import VideoGenerationPipeline
from .video_generation_pipeline import VideoGenerationPipeline as VideoOrchestrator
from .video_generation_pipeline import VideoGenerationPipeline as EnhancedOrchestrator
from .video_generation_pipeline import VideoGenerationPipeline as VideoProcessor

# Analytics aliases
from .analytics_service import AnalyticsService
from .analytics_service import AnalyticsService as AnalyticsConnector
from .analytics_service import AnalyticsService as MetricsAggregator
from .analytics_service import AnalyticsService as ReportGenerator

# Cost tracking aliases
from .cost_tracking import CostTracker
from .cost_tracking import CostTracker as CostAggregator
from .cost_tracking import CostTracker as CostVerifier
from .cost_tracking import CostTracker as RevenueTracker

# Export main services
__all__ = [
    "VideoGenerationPipeline",
    "AnalyticsService",
    "CostTracker",
]
