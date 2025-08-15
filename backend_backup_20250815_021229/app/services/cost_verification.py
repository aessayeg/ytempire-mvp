#!/usr/bin/env python3
"""
Cost Verification Framework for YTEmpire
Tracks and verifies AI service costs to ensure <$3 per video target
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import Enum
import psycopg2
import redis
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceType(Enum):
    """AI Service types"""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    ELEVENLABS = "elevenlabs"
    GOOGLE_TTS = "google_tts"
    DALLE3 = "dalle3"
    STABLE_DIFFUSION = "stable_diffusion"

@dataclass
class ServicePricing:
    """Pricing for AI services"""
    service: ServiceType
    input_cost_per_1k: float  # Cost per 1000 tokens/characters
    output_cost_per_1k: float
    unit: str  # tokens, characters, requests, etc.
    
    # Fallback options
    fallback_service: Optional[ServiceType] = None
    fallback_threshold: float = 0.5  # Cost threshold to trigger fallback

# Current pricing (as of 2024)
PRICING_TABLE = {
    ServiceType.OPENAI_GPT4: ServicePricing(
        service=ServiceType.OPENAI_GPT4,
        input_cost_per_1k=0.03,
        output_cost_per_1k=0.06,
        unit="tokens",
        fallback_service=ServiceType.OPENAI_GPT35,
        fallback_threshold=0.50
    ),
    ServiceType.OPENAI_GPT35: ServicePricing(
        service=ServiceType.OPENAI_GPT35,
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.002,
        unit="tokens",
        fallback_service=ServiceType.ANTHROPIC_CLAUDE,
        fallback_threshold=0.10
    ),
    ServiceType.ANTHROPIC_CLAUDE: ServicePricing(
        service=ServiceType.ANTHROPIC_CLAUDE,
        input_cost_per_1k=0.008,
        output_cost_per_1k=0.024,
        unit="tokens",
        fallback_service=None,
        fallback_threshold=0.20
    ),
    ServiceType.ELEVENLABS: ServicePricing(
        service=ServiceType.ELEVENLABS,
        input_cost_per_1k=0.18,  # Per 1000 characters
        output_cost_per_1k=0,
        unit="characters",
        fallback_service=ServiceType.GOOGLE_TTS,
        fallback_threshold=0.50
    ),
    ServiceType.GOOGLE_TTS: ServicePricing(
        service=ServiceType.GOOGLE_TTS,
        input_cost_per_1k=0.004,  # Per 1000 characters
        output_cost_per_1k=0,
        unit="characters",
        fallback_service=None,
        fallback_threshold=0.10
    ),
    ServiceType.DALLE3: ServicePricing(
        service=ServiceType.DALLE3,
        input_cost_per_1k=0.04,  # Per image (1024x1024)
        output_cost_per_1k=0,
        unit="requests",
        fallback_service=ServiceType.STABLE_DIFFUSION,
        fallback_threshold=0.04
    ),
    ServiceType.STABLE_DIFFUSION: ServicePricing(
        service=ServiceType.STABLE_DIFFUSION,
        input_cost_per_1k=0.002,  # Per image
        output_cost_per_1k=0,
        unit="requests",
        fallback_service=None,
        fallback_threshold=0.01
    )
}

@dataclass
class CostRecord:
    """Individual cost record"""
    timestamp: datetime
    video_id: str
    service: ServiceType
    operation: str  # script_generation, voice_synthesis, thumbnail, etc.
    input_units: int
    output_units: int
    cost: float
    cached: bool = False
    fallback_used: bool = False
    metadata: Dict = field(default_factory=dict)

@dataclass
class VideoCostBreakdown:
    """Complete cost breakdown for a video"""
    video_id: str
    total_cost: float
    script_cost: float
    voice_cost: float
    thumbnail_cost: float
    processing_cost: float
    
    # Service usage
    services_used: Dict[str, float]
    
    # Optimization metrics
    cache_savings: float
    fallback_savings: float
    
    # Time metrics
    generation_time: float
    timestamp: datetime

class CostVerificationFramework:
    """Framework for tracking and verifying costs"""
    
    def __init__(self, 
                 db_url: str = "postgresql://ytempire:admin@localhost:5432/ytempire_db",
                 redis_url: str = "redis://localhost:6379/0"):
        self.db_url = db_url
        self.redis_url = redis_url
        self.redis_client = redis.from_url(redis_url)
        self.cost_records: List[CostRecord] = []
        
        # Cost limits
        self.max_cost_per_video = 3.0
        self.target_cost_per_video = 0.50
        self.daily_budget_limit = 50.0
        
        # Service-specific daily limits
        self.service_limits = {
            ServiceType.OPENAI_GPT4: 50.0,
            ServiceType.ELEVENLABS: 20.0,
            ServiceType.DALLE3: 10.0
        }
    
    def calculate_cost(self, 
                      service: ServiceType, 
                      input_units: int, 
                      output_units: int = 0,
                      cached: bool = False) -> float:
        """Calculate cost for a service usage"""
        if cached:
            return 0.0
        
        pricing = PRICING_TABLE[service]
        
        if pricing.unit == "requests":
            # Flat rate per request
            cost = pricing.input_cost_per_1k * input_units
        else:
            # Token/character based pricing
            input_cost = (input_units / 1000) * pricing.input_cost_per_1k
            output_cost = (output_units / 1000) * pricing.output_cost_per_1k
            cost = input_cost + output_cost
        
        return round(cost, 4)
    
    def should_use_fallback(self, service: ServiceType, current_cost: float) -> Tuple[bool, Optional[ServiceType]]:
        """Determine if fallback service should be used"""
        pricing = PRICING_TABLE[service]
        
        if not pricing.fallback_service:
            return False, None
        
        if current_cost >= pricing.fallback_threshold:
            return True, pricing.fallback_service
        
        # Check daily budget
        daily_cost = self.get_daily_cost(service)
        if service in self.service_limits and daily_cost >= self.service_limits[service]:
            return True, pricing.fallback_service
        
        return False, None
    
    def record_cost(self, record: CostRecord):
        """Record a cost entry"""
        self.cost_records.append(record)
        
        # Store in database
        try:
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO costs (
                    video_id, service, operation, input_units, output_units,
                    cost, cached, fallback_used, metadata, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                record.video_id,
                record.service.value,
                record.operation,
                record.input_units,
                record.output_units,
                record.cost,
                record.cached,
                record.fallback_used,
                json.dumps(record.metadata),
                record.timestamp
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to record cost in database: {e}")
        
        # Update Redis cache
        cache_key = f"cost:daily:{record.service.value}:{datetime.now().date()}"
        self.redis_client.incrbyfloat(cache_key, record.cost)
        self.redis_client.expire(cache_key, 86400 * 2)  # 2 days expiry
    
    def get_daily_cost(self, service: ServiceType) -> float:
        """Get total daily cost for a service"""
        cache_key = f"cost:daily:{service.value}:{datetime.now().date()}"
        cost = self.redis_client.get(cache_key)
        return float(cost) if cost else 0.0
    
    def get_video_cost(self, video_id: str) -> VideoCostBreakdown:
        """Get complete cost breakdown for a video"""
        try:
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    service, operation, SUM(cost) as total_cost,
                    SUM(CASE WHEN cached THEN cost ELSE 0 END) as cached_cost,
                    SUM(CASE WHEN fallback_used THEN cost ELSE 0 END) as fallback_cost,
                    MIN(created_at) as start_time,
                    MAX(created_at) as end_time
                FROM costs
                WHERE video_id = %s
                GROUP BY service, operation
            """, (video_id,))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not results:
                return None
            
            # Process results
            total_cost = 0
            script_cost = 0
            voice_cost = 0
            thumbnail_cost = 0
            processing_cost = 0
            services_used = {}
            cache_savings = 0
            fallback_savings = 0
            
            start_time = None
            end_time = None
            
            for row in results:
                service, operation, cost, cached_cost, fallback_cost, start, end = row
                
                total_cost += cost
                services_used[service] = services_used.get(service, 0) + cost
                cache_savings += cached_cost
                fallback_savings += fallback_cost
                
                if not start_time or start < start_time:
                    start_time = start
                if not end_time or end > end_time:
                    end_time = end
                
                # Categorize costs
                if "script" in operation.lower():
                    script_cost += cost
                elif "voice" in operation.lower() or "audio" in operation.lower():
                    voice_cost += cost
                elif "thumbnail" in operation.lower() or "image" in operation.lower():
                    thumbnail_cost += cost
                else:
                    processing_cost += cost
            
            generation_time = (end_time - start_time).total_seconds() if start_time and end_time else 0
            
            return VideoCostBreakdown(
                video_id=video_id,
                total_cost=round(total_cost, 4),
                script_cost=round(script_cost, 4),
                voice_cost=round(voice_cost, 4),
                thumbnail_cost=round(thumbnail_cost, 4),
                processing_cost=round(processing_cost, 4),
                services_used=services_used,
                cache_savings=round(cache_savings, 4),
                fallback_savings=round(fallback_savings, 4),
                generation_time=generation_time,
                timestamp=end_time or datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to get video cost: {e}")
            return None
    
    def verify_cost_compliance(self, video_id: str) -> Dict[str, any]:
        """Verify if video generation met cost targets"""
        breakdown = self.get_video_cost(video_id)
        
        if not breakdown:
            return {
                "compliant": False,
                "error": "No cost data found"
            }
        
        compliance = {
            "compliant": True,
            "total_cost": breakdown.total_cost,
            "under_limit": breakdown.total_cost <= self.max_cost_per_video,
            "near_target": breakdown.total_cost <= self.target_cost_per_video * 1.5,
            "breakdown": asdict(breakdown),
            "recommendations": []
        }
        
        # Check compliance
        if breakdown.total_cost > self.max_cost_per_video:
            compliance["compliant"] = False
            compliance["recommendations"].append(
                f"Cost exceeded limit: ${breakdown.total_cost:.2f} > ${self.max_cost_per_video}"
            )
        
        # Provide optimization recommendations
        if breakdown.script_cost > 0.5:
            compliance["recommendations"].append(
                "High script generation cost - consider using GPT-3.5 or caching common scripts"
            )
        
        if breakdown.voice_cost > 1.0:
            compliance["recommendations"].append(
                "High voice synthesis cost - consider using Google TTS for non-premium content"
            )
        
        if breakdown.thumbnail_cost > 0.5:
            compliance["recommendations"].append(
                "High thumbnail cost - consider using Stable Diffusion or template-based generation"
            )
        
        if breakdown.cache_savings < breakdown.total_cost * 0.1:
            compliance["recommendations"].append(
                "Low cache utilization - implement more aggressive caching strategies"
            )
        
        return compliance
    
    def optimize_service_selection(self, 
                                  operation: str,
                                  quality_required: float = 0.8) -> ServiceType:
        """Select optimal service based on cost and quality requirements"""
        service_options = []
        
        if "script" in operation.lower():
            if quality_required >= 0.9:
                service_options = [ServiceType.OPENAI_GPT4]
            elif quality_required >= 0.7:
                service_options = [ServiceType.OPENAI_GPT35, ServiceType.ANTHROPIC_CLAUDE]
            else:
                service_options = [ServiceType.OPENAI_GPT35]
        
        elif "voice" in operation.lower():
            if quality_required >= 0.8:
                service_options = [ServiceType.ELEVENLABS]
            else:
                service_options = [ServiceType.GOOGLE_TTS]
        
        elif "thumbnail" in operation.lower():
            if quality_required >= 0.9:
                service_options = [ServiceType.DALLE3]
            else:
                service_options = [ServiceType.STABLE_DIFFUSION]
        
        # Select based on current daily usage
        for service in service_options:
            daily_cost = self.get_daily_cost(service)
            if service not in self.service_limits or daily_cost < self.service_limits.get(service, float('inf')):
                return service
        
        # Return cheapest option if all are over limit
        return service_options[-1] if service_options else ServiceType.OPENAI_GPT35
    
    def generate_cost_report(self, 
                            start_date: datetime = None,
                            end_date: datetime = None) -> Dict:
        """Generate comprehensive cost report"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
        
        try:
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            # Overall statistics
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT video_id) as total_videos,
                    SUM(cost) as total_cost,
                    AVG(cost) as avg_cost_per_operation,
                    MIN(cost) as min_cost,
                    MAX(cost) as max_cost
                FROM costs
                WHERE created_at BETWEEN %s AND %s
            """, (start_date, end_date))
            
            overall_stats = cursor.fetchone()
            
            # Per-video statistics
            cursor.execute("""
                SELECT 
                    video_id,
                    SUM(cost) as total_cost,
                    COUNT(*) as operation_count
                FROM costs
                WHERE created_at BETWEEN %s AND %s
                GROUP BY video_id
                ORDER BY total_cost DESC
            """, (start_date, end_date))
            
            video_costs = cursor.fetchall()
            
            # Service breakdown
            cursor.execute("""
                SELECT 
                    service,
                    COUNT(*) as usage_count,
                    SUM(cost) as total_cost,
                    AVG(cost) as avg_cost
                FROM costs
                WHERE created_at BETWEEN %s AND %s
                GROUP BY service
                ORDER BY total_cost DESC
            """, (start_date, end_date))
            
            service_breakdown = cursor.fetchall()
            
            # Optimization metrics
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN cached THEN cost ELSE 0 END) as cache_savings,
                    SUM(CASE WHEN fallback_used THEN cost ELSE 0 END) as fallback_savings,
                    COUNT(CASE WHEN cached THEN 1 END) as cached_operations,
                    COUNT(CASE WHEN fallback_used THEN 1 END) as fallback_operations
                FROM costs
                WHERE created_at BETWEEN %s AND %s
            """, (start_date, end_date))
            
            optimization_stats = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            # Calculate per-video average
            if video_costs:
                video_cost_values = [cost for _, cost, _ in video_costs]
                avg_cost_per_video = sum(video_cost_values) / len(video_cost_values)
                videos_under_target = len([c for c in video_cost_values if c <= self.target_cost_per_video])
                videos_under_limit = len([c for c in video_cost_values if c <= self.max_cost_per_video])
            else:
                avg_cost_per_video = 0
                videos_under_target = 0
                videos_under_limit = 0
            
            report = {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "overall": {
                    "total_videos": overall_stats[0] or 0,
                    "total_cost": float(overall_stats[1] or 0),
                    "avg_cost_per_video": avg_cost_per_video,
                    "min_cost_per_operation": float(overall_stats[3] or 0),
                    "max_cost_per_operation": float(overall_stats[4] or 0)
                },
                "compliance": {
                    "videos_under_target": videos_under_target,
                    "videos_under_limit": videos_under_limit,
                    "target_compliance_rate": (videos_under_target / len(video_costs) * 100) if video_costs else 0,
                    "limit_compliance_rate": (videos_under_limit / len(video_costs) * 100) if video_costs else 0
                },
                "services": [
                    {
                        "service": service,
                        "usage_count": usage_count,
                        "total_cost": float(total_cost),
                        "avg_cost": float(avg_cost)
                    }
                    for service, usage_count, total_cost, avg_cost in service_breakdown
                ],
                "optimization": {
                    "cache_savings": float(optimization_stats[0] or 0),
                    "fallback_savings": float(optimization_stats[1] or 0),
                    "cached_operations": optimization_stats[2] or 0,
                    "fallback_operations": optimization_stats[3] or 0
                },
                "top_expensive_videos": [
                    {
                        "video_id": video_id,
                        "total_cost": float(total_cost),
                        "operations": operation_count
                    }
                    for video_id, total_cost, operation_count in video_costs[:10]
                ]
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate cost report: {e}")
            return None

def simulate_video_generation():
    """Simulate video generation with cost tracking"""
    framework = CostVerificationFramework()
    video_id = f"test_video_{int(time.time())}"
    
    logger.info(f"Simulating video generation: {video_id}")
    
    # 1. Script generation
    service = framework.optimize_service_selection("script_generation", quality_required=0.8)
    script_cost = framework.calculate_cost(service, input_units=1500, output_units=2000)
    
    framework.record_cost(CostRecord(
        timestamp=datetime.now(),
        video_id=video_id,
        service=service,
        operation="script_generation",
        input_units=1500,
        output_units=2000,
        cost=script_cost,
        cached=False,
        fallback_used=False
    ))
    
    logger.info(f"Script generation: {service.value} - ${script_cost:.4f}")
    
    # 2. Voice synthesis
    service = framework.optimize_service_selection("voice_synthesis", quality_required=0.7)
    voice_cost = framework.calculate_cost(service, input_units=5000)  # 5000 characters
    
    framework.record_cost(CostRecord(
        timestamp=datetime.now(),
        video_id=video_id,
        service=service,
        operation="voice_synthesis",
        input_units=5000,
        output_units=0,
        cost=voice_cost,
        cached=False,
        fallback_used=False
    ))
    
    logger.info(f"Voice synthesis: {service.value} - ${voice_cost:.4f}")
    
    # 3. Thumbnail generation
    service = framework.optimize_service_selection("thumbnail_generation", quality_required=0.9)
    thumbnail_cost = framework.calculate_cost(service, input_units=1)  # 1 image
    
    framework.record_cost(CostRecord(
        timestamp=datetime.now(),
        video_id=video_id,
        service=service,
        operation="thumbnail_generation",
        input_units=1,
        output_units=0,
        cost=thumbnail_cost,
        cached=False,
        fallback_used=False
    ))
    
    logger.info(f"Thumbnail generation: {service.value} - ${thumbnail_cost:.4f}")
    
    # Get cost breakdown
    breakdown = framework.get_video_cost(video_id)
    if breakdown:
        logger.info(f"\nVideo Cost Breakdown:")
        logger.info(f"  Total: ${breakdown.total_cost:.2f}")
        logger.info(f"  Script: ${breakdown.script_cost:.2f}")
        logger.info(f"  Voice: ${breakdown.voice_cost:.2f}")
        logger.info(f"  Thumbnail: ${breakdown.thumbnail_cost:.2f}")
    
    # Verify compliance
    compliance = framework.verify_cost_compliance(video_id)
    logger.info(f"\nCost Compliance:")
    logger.info(f"  Compliant: {compliance['compliant']}")
    logger.info(f"  Under limit ($3): {compliance['under_limit']}")
    logger.info(f"  Near target ($0.50): {compliance['near_target']}")
    
    if compliance['recommendations']:
        logger.info(f"\nRecommendations:")
        for rec in compliance['recommendations']:
            logger.info(f"  - {rec}")
    
    return video_id

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cost Verification Framework")
    parser.add_argument("--simulate", action="store_true", help="Run simulation")
    parser.add_argument("--report", action="store_true", help="Generate cost report")
    parser.add_argument("--video-id", help="Get cost for specific video")
    args = parser.parse_args()
    
    framework = CostVerificationFramework()
    
    if args.simulate:
        simulate_video_generation()
    
    if args.report:
        report = framework.generate_cost_report()
        if report:
            print(json.dumps(report, indent=2))
    
    if args.video_id:
        breakdown = framework.get_video_cost(args.video_id)
        if breakdown:
            print(json.dumps(asdict(breakdown), indent=2, default=str))
        
        compliance = framework.verify_cost_compliance(args.video_id)
        print(json.dumps(compliance, indent=2))

if __name__ == "__main__":
    main()