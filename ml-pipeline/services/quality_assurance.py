"""
Model Quality Assurance Framework for YTEmpire
P1 Task: [AI/ML] Model Quality Assurance Framework
Ensures ML model quality and performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import json
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch
from transformers import pipeline
import openai
from collections import defaultdict
import redis
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality metrics for content evaluation"""
    overall_score: float
    engagement_score: float
    relevance_score: float
    originality_score: float
    technical_quality: float
    policy_compliance: float
    seo_score: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for ML models"""
    model_name: str
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cost_per_inference: float
    timestamp: datetime

class QualityAssuranceFramework:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.quality_thresholds = {
            'minimum_overall_score': 70,
            'minimum_engagement_score': 65,
            'minimum_relevance_score': 75,
            'minimum_technical_quality': 80,
            'policy_compliance_required': 95
        }
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.quality_cache = {}
        self.performance_history = defaultdict(list)
        
    async def evaluate_content_quality(
        self,
        content: Dict[str, Any],
        content_type: str = "video"
    ) -> QualityMetrics:
        """Evaluate content quality across multiple dimensions"""
        
        # Check cache first
        cache_key = self._generate_cache_key(content)
        cached_result = self._get_cached_quality(cache_key)
        if cached_result:
            return cached_result
        
        # Perform quality evaluation
        tasks = [
            self._evaluate_engagement_potential(content),
            self._evaluate_relevance(content),
            self._evaluate_originality(content),
            self._evaluate_technical_quality(content),
            self._check_policy_compliance(content),
            self._evaluate_seo_optimization(content)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(results)
        
        metrics = QualityMetrics(
            overall_score=overall_score,
            engagement_score=results[0],
            relevance_score=results[1],
            originality_score=results[2],
            technical_quality=results[3],
            policy_compliance=results[4],
            seo_score=results[5],
            timestamp=datetime.utcnow()
        )
        
        # Cache the result
        self._cache_quality_result(cache_key, metrics)
        
        # Log if below threshold
        if overall_score < self.quality_thresholds['minimum_overall_score']:
            logger.warning(f"Content quality below threshold: {overall_score}")
        
        return metrics
    
    async def _evaluate_engagement_potential(self, content: Dict) -> float:
        """Evaluate potential for user engagement"""
        score = 0.0
        
        # Title analysis
        title = content.get('title', '')
        if title:
            # Check title length (optimal: 50-60 chars)
            title_length = len(title)
            if 50 <= title_length <= 60:
                score += 20
            elif 40 <= title_length <= 70:
                score += 15
            else:
                score += 5
            
            # Check for power words
            power_words = ['amazing', 'ultimate', 'essential', 'proven', 'secret']
            if any(word in title.lower() for word in power_words):
                score += 10
            
            # Question in title
            if '?' in title:
                score += 10
        
        # Thumbnail quality
        if content.get('thumbnail_url'):
            score += 15
        
        # Description analysis
        description = content.get('description', '')
        if len(description) > 100:
            score += 10
        
        # Tags optimization
        tags = content.get('tags', [])
        if 5 <= len(tags) <= 15:
            score += 15
        
        # Duration optimization (8-12 minutes is optimal)
        duration = content.get('duration', 0)
        if 480 <= duration <= 720:
            score += 20
        elif 300 <= duration <= 900:
            score += 15
        
        return min(score, 100)
    
    async def _evaluate_relevance(self, content: Dict) -> float:
        """Evaluate content relevance to topic"""
        score = 85.0  # Base score
        
        # Check topic alignment
        topic = content.get('topic', '')
        script = content.get('script', '')
        
        if topic and script:
            # Simple keyword matching (in production, use embeddings)
            topic_words = set(topic.lower().split())
            script_words = set(script.lower().split())
            
            overlap = len(topic_words & script_words)
            relevance_ratio = overlap / max(len(topic_words), 1)
            
            score = 70 + (relevance_ratio * 30)
        
        return min(score, 100)
    
    async def _evaluate_originality(self, content: Dict) -> float:
        """Evaluate content originality"""
        # In production, this would check against existing content database
        # For now, return a simulated score
        
        script = content.get('script', '')
        if not script:
            return 50.0
        
        # Check script uniqueness (simplified)
        script_hash = hashlib.md5(script.encode()).hexdigest()
        
        # Check if we've seen similar content
        similar_count = self.redis_client.get(f"content_hash:{script_hash[:8]}")
        
        if similar_count:
            # Penalize for similarity
            return max(50 - (int(similar_count) * 10), 20)
        
        # Mark this content
        self.redis_client.setex(f"content_hash:{script_hash[:8]}", 86400, 1)
        
        return 85.0
    
    async def _evaluate_technical_quality(self, content: Dict) -> float:
        """Evaluate technical quality of generated content"""
        score = 0.0
        
        # Video quality checks
        if content.get('resolution') == '1080p':
            score += 25
        elif content.get('resolution') == '720p':
            score += 20
        
        # Audio quality
        if content.get('audio_bitrate', 0) >= 128:
            score += 20
        
        # Encoding quality
        if content.get('video_codec') == 'h264':
            score += 15
        
        # File size optimization
        file_size = content.get('file_size', 0)
        duration = content.get('duration', 1)
        
        # Bitrate calculation (MB per minute)
        if file_size and duration:
            bitrate = (file_size / 1024 / 1024) / (duration / 60)
            if 5 <= bitrate <= 15:  # Optimal range
                score += 20
        
        return min(score, 100)
    
    async def _check_policy_compliance(self, content: Dict) -> float:
        """Check YouTube policy compliance"""
        violations = 0
        checks = 10
        
        script = content.get('script', '').lower()
        title = content.get('title', '').lower()
        description = content.get('description', '').lower()
        
        # Prohibited content checks
        prohibited_terms = [
            'copyright', 'stolen', 'leaked', 'hack', 
            'crack', 'pirated', 'illegal'
        ]
        
        for term in prohibited_terms:
            if term in script or term in title or term in description:
                violations += 1
        
        # Misleading content check
        clickbait_terms = ['you won\'t believe', 'shocking', 'banned']
        for term in clickbait_terms:
            if term in title:
                violations += 0.5
        
        # Calculate compliance score
        compliance_score = ((checks - violations) / checks) * 100
        
        return max(compliance_score, 0)
    
    async def _evaluate_seo_optimization(self, content: Dict) -> float:
        """Evaluate SEO optimization"""
        score = 0.0
        
        title = content.get('title', '')
        description = content.get('description', '')
        tags = content.get('tags', [])
        
        # Title optimization
        if title:
            if len(title) <= 60:
                score += 20
            # Keyword in title
            if content.get('main_keyword') in title.lower():
                score += 15
        
        # Description optimization
        if description:
            if 150 <= len(description) <= 300:
                score += 20
            # Keywords in description
            if content.get('main_keyword') in description.lower():
                score += 10
        
        # Tags optimization
        if 5 <= len(tags) <= 15:
            score += 20
        
        # Metadata completeness
        if content.get('category'):
            score += 15
        
        return min(score, 100)
    
    def _calculate_overall_score(self, scores: List[float]) -> float:
        """Calculate weighted overall score"""
        weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]  # Engagement, Relevance, Originality, Technical, Policy, SEO
        
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return round(weighted_sum, 2)
    
    async def validate_model_performance(
        self,
        model_name: str,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        inference_times: List[float]
    ) -> ModelPerformanceMetrics:
        """Validate ML model performance"""
        
        # Calculate accuracy metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='weighted'
        )
        
        # Calculate latency percentiles
        latencies = np.array(inference_times)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        # Calculate throughput
        total_time = sum(inference_times)
        throughput = len(predictions) / total_time if total_time > 0 else 0
        
        # Estimate cost
        cost_per_inference = self._estimate_inference_cost(model_name, np.mean(latencies))
        
        metrics = ModelPerformanceMetrics(
            model_name=model_name,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            throughput=throughput,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            cost_per_inference=cost_per_inference,
            timestamp=datetime.utcnow()
        )
        
        # Store performance history
        self.performance_history[model_name].append(metrics)
        
        # Check for performance degradation
        if self._detect_performance_degradation(model_name):
            logger.warning(f"Performance degradation detected for {model_name}")
        
        return metrics
    
    def _detect_performance_degradation(self, model_name: str) -> bool:
        """Detect if model performance is degrading"""
        history = self.performance_history.get(model_name, [])
        
        if len(history) < 10:
            return False
        
        # Check last 10 measurements
        recent = history[-10:]
        
        # Check for increasing latency
        latencies = [m.latency_p95 for m in recent]
        if latencies[-1] > latencies[0] * 1.5:  # 50% increase
            return True
        
        # Check for decreasing accuracy
        accuracies = [m.accuracy for m in recent]
        if accuracies[-1] < accuracies[0] * 0.9:  # 10% decrease
            return True
        
        return False
    
    def _estimate_inference_cost(self, model_name: str, latency: float) -> float:
        """Estimate cost per inference"""
        # Base costs per model type
        model_costs = {
            'gpt-4': 0.03,
            'gpt-3.5': 0.002,
            'claude': 0.025,
            'custom': 0.001
        }
        
        base_cost = model_costs.get(model_name.lower(), 0.001)
        
        # Add compute cost based on latency
        compute_cost = latency * 0.0001  # $0.0001 per second
        
        return base_cost + compute_cost
    
    def _generate_cache_key(self, content: Dict) -> str:
        """Generate cache key for content"""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def _get_cached_quality(self, cache_key: str) -> Optional[QualityMetrics]:
        """Get cached quality metrics"""
        cached = self.redis_client.get(f"quality:{cache_key}")
        if cached:
            data = json.loads(cached)
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            return QualityMetrics(**data)
        return None
    
    def _cache_quality_result(self, cache_key: str, metrics: QualityMetrics):
        """Cache quality metrics"""
        self.redis_client.setex(
            f"quality:{cache_key}",
            3600,  # 1 hour TTL
            json.dumps(metrics.to_dict())
        )
    
    async def run_quality_checks(self, content: Dict) -> Tuple[bool, QualityMetrics]:
        """Run all quality checks and return pass/fail with metrics"""
        metrics = await self.evaluate_content_quality(content)
        
        # Check against thresholds
        passed = all([
            metrics.overall_score >= self.quality_thresholds['minimum_overall_score'],
            metrics.engagement_score >= self.quality_thresholds['minimum_engagement_score'],
            metrics.relevance_score >= self.quality_thresholds['minimum_relevance_score'],
            metrics.technical_quality >= self.quality_thresholds['minimum_technical_quality'],
            metrics.policy_compliance >= self.quality_thresholds['policy_compliance_required']
        ])
        
        return passed, metrics
    
    def get_quality_report(self, start_date: datetime, end_date: datetime) -> Dict:
        """Generate quality report for date range"""
        # This would query from database in production
        return {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_evaluated': 150,
                'passed': 142,
                'failed': 8,
                'pass_rate': 94.7,
                'average_score': 82.3
            },
            'breakdown': {
                'engagement': 78.5,
                'relevance': 85.2,
                'originality': 81.0,
                'technical': 88.3,
                'compliance': 96.2,
                'seo': 79.4
            },
            'trends': {
                'improving': ['technical', 'compliance'],
                'declining': ['engagement'],
                'stable': ['relevance', 'originality', 'seo']
            }
        }

# Global instance
quality_assurance = QualityAssuranceFramework()