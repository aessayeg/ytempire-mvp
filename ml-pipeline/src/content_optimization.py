"""
Content Optimization Module for YouTube Videos
A/B testing, performance prediction, and content recommendations
"""
import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import redis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationMetric(Enum):
    """Metrics to optimize for"""
    WATCH_TIME = "watch_time"
    CLICK_THROUGH_RATE = "ctr"
    ENGAGEMENT_RATE = "engagement"
    REVENUE = "revenue"
    SUBSCRIBER_GROWTH = "subscriber_growth"
    RETENTION_RATE = "retention"

class TestStatus(Enum):
    """A/B test status"""
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"

@dataclass
class ContentVariant:
    """Content variant for A/B testing"""
    variant_id: str
    title: str
    thumbnail_url: str
    description: str
    tags: List[str]
    script_tone: str
    video_duration: int
    publish_time: datetime
    metadata: Dict[str, Any]

@dataclass
class TestResult:
    """A/B test results"""
    test_id: str
    winner: str
    confidence_level: float
    improvement: float
    sample_size: int
    metrics: Dict[str, float]
    recommendations: List[str]

@dataclass
class PerformancePrediction:
    """Predicted video performance"""
    predicted_views: int
    predicted_watch_time: float
    predicted_revenue: float
    predicted_ctr: float
    confidence_interval: Tuple[float, float]
    risk_score: float

class ContentOptimizer:
    """
    Advanced content optimization with ML-driven A/B testing
    """
    
    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        model_path: Optional[str] = None
    ):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.cache_ttl = 3600  # 1 hour
        
        # Initialize ML models
        self.performance_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.revenue_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        
        # Load pre-trained models if available
        if model_path:
            self._load_models(model_path)
        
        # Optimization thresholds
        self.significance_level = 0.05  # 95% confidence
        self.minimum_sample_size = 1000
        self.minimum_test_duration = 24  # hours
    
    async def create_ab_test(
        self,
        base_content: Dict[str, Any],
        test_parameters: Dict[str, List[Any]],
        optimization_metric: OptimizationMetric
    ) -> Dict[str, Any]:
        """
        Create A/B test with multiple variants
        
        Args:
            base_content: Original content configuration
            test_parameters: Parameters to test (e.g., titles, thumbnails)
            optimization_metric: Metric to optimize
        
        Returns:
            Test configuration with variants
        """
        test_id = self._generate_test_id(base_content)
        
        # Generate variants
        variants = self._generate_variants(base_content, test_parameters)
        
        # Predict performance for each variant
        predictions = []
        for variant in variants:
            prediction = await self.predict_performance(variant)
            predictions.append(prediction)
        
        # Calculate optimal traffic split
        traffic_split = self._calculate_traffic_split(variants, predictions)
        
        # Store test configuration
        test_config = {
            'test_id': test_id,
            'created_at': datetime.now().isoformat(),
            'status': TestStatus.RUNNING.value,
            'optimization_metric': optimization_metric.value,
            'variants': [asdict(v) for v in variants],
            'traffic_split': traffic_split,
            'predictions': [self._serialize_prediction(p) for p in predictions],
            'minimum_sample_size': self.minimum_sample_size,
            'current_sample_size': 0
        }
        
        # Cache test configuration
        self._store_test_config(test_id, test_config)
        
        return test_config
    
    def _generate_variants(
        self,
        base_content: Dict[str, Any],
        test_parameters: Dict[str, List[Any]]
    ) -> List[ContentVariant]:
        """Generate content variants for testing"""
        variants = []
        
        # Create control variant
        control = ContentVariant(
            variant_id="control",
            title=base_content.get('title', ''),
            thumbnail_url=base_content.get('thumbnail_url', ''),
            description=base_content.get('description', ''),
            tags=base_content.get('tags', []),
            script_tone=base_content.get('script_tone', 'professional'),
            video_duration=base_content.get('duration', 600),
            publish_time=datetime.now(),
            metadata={'is_control': True}
        )
        variants.append(control)
        
        # Generate test variants
        variant_count = 1
        for param, values in test_parameters.items():
            for value in values:
                variant_content = base_content.copy()
                variant_content[param] = value
                
                variant = ContentVariant(
                    variant_id=f"variant_{variant_count}",
                    title=variant_content.get('title', ''),
                    thumbnail_url=variant_content.get('thumbnail_url', ''),
                    description=variant_content.get('description', ''),
                    tags=variant_content.get('tags', []),
                    script_tone=variant_content.get('script_tone', 'professional'),
                    video_duration=variant_content.get('duration', 600),
                    publish_time=datetime.now(),
                    metadata={
                        'is_control': False,
                        'changed_parameter': param,
                        'changed_value': value
                    }
                )
                variants.append(variant)
                variant_count += 1
        
        return variants
    
    async def predict_performance(
        self,
        content: ContentVariant
    ) -> PerformancePrediction:
        """
        Predict content performance using ML models
        
        Args:
            content: Content variant to analyze
        
        Returns:
            Performance prediction with confidence intervals
        """
        # Extract features
        features = self._extract_features(content)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features.reshape(1, -1))
        
        # Predict views and watch time
        predicted_views = self.performance_model.predict(features_scaled)[0]
        
        # Predict revenue
        predicted_revenue = self.revenue_model.predict(features_scaled)[0]
        
        # Calculate CTR based on title and thumbnail features
        predicted_ctr = self._predict_ctr(content)
        
        # Calculate watch time
        predicted_watch_time = predicted_views * content.video_duration * 0.4  # 40% avg retention
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            predicted_views,
            sample_size=100  # Historical sample size
        )
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(content, predicted_views)
        
        return PerformancePrediction(
            predicted_views=int(predicted_views),
            predicted_watch_time=predicted_watch_time,
            predicted_revenue=predicted_revenue,
            predicted_ctr=predicted_ctr,
            confidence_interval=confidence_interval,
            risk_score=risk_score
        )
    
    def _extract_features(self, content: ContentVariant) -> np.ndarray:
        """Extract ML features from content"""
        features = []
        
        # Title features
        features.append(len(content.title))  # Title length
        features.append(content.title.count('!') + content.title.count('?'))  # Punctuation
        features.append(1 if any(char.isupper() for char in content.title) else 0)  # Has caps
        features.append(1 if any(char.isdigit() for char in content.title) else 0)  # Has numbers
        
        # Tag features
        features.append(len(content.tags))  # Number of tags
        features.append(sum(len(tag) for tag in content.tags))  # Total tag length
        
        # Time features
        features.append(content.publish_time.hour)  # Hour of day
        features.append(content.publish_time.weekday())  # Day of week
        
        # Duration features
        features.append(content.video_duration)  # Video length in seconds
        features.append(1 if content.video_duration < 300 else 0)  # Is short form
        
        # Tone features (one-hot encoding)
        tones = ['professional', 'casual', 'humorous', 'serious', 'friendly']
        for tone in tones:
            features.append(1 if content.script_tone == tone else 0)
        
        return np.array(features)
    
    def _predict_ctr(self, content: ContentVariant) -> float:
        """Predict click-through rate"""
        base_ctr = 0.05  # 5% base CTR
        
        # Title impact
        if len(content.title) < 60:
            base_ctr += 0.01
        if any(word in content.title.lower() for word in ['how', 'why', 'best', 'top']):
            base_ctr += 0.015
        
        # Thumbnail impact (simplified)
        if 'high_contrast' in content.metadata:
            base_ctr += 0.01
        
        return min(base_ctr, 0.15)  # Cap at 15%
    
    def _calculate_confidence_interval(
        self,
        prediction: float,
        sample_size: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for prediction"""
        # Simplified confidence interval calculation
        std_error = prediction * 0.2 / np.sqrt(sample_size)  # 20% relative error
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        lower = prediction - z_score * std_error
        upper = prediction + z_score * std_error
        
        return (max(0, lower), upper)
    
    def _calculate_risk_score(
        self,
        content: ContentVariant,
        predicted_views: float
    ) -> float:
        """Calculate risk score (0-100)"""
        risk = 0.0
        
        # New format risk
        if content.metadata.get('is_control') == False:
            risk += 20
        
        # Low prediction risk
        if predicted_views < 1000:
            risk += 30
        
        # Experimental parameters risk
        if content.metadata.get('changed_parameter') in ['script_tone', 'video_duration']:
            risk += 15
        
        return min(risk, 100.0)
    
    def _calculate_traffic_split(
        self,
        variants: List[ContentVariant],
        predictions: List[PerformancePrediction]
    ) -> Dict[str, float]:
        """Calculate optimal traffic split using Thompson Sampling"""
        
        # Use predicted performance to weight traffic
        performance_scores = [p.predicted_views for p in predictions]
        total_score = sum(performance_scores)
        
        # Initial equal split
        if total_score == 0:
            split = {v.variant_id: 1.0 / len(variants) for v in variants}
        else:
            # Thompson Sampling inspired allocation
            split = {}
            for variant, prediction in zip(variants, predictions):
                # Give more traffic to promising variants
                weight = prediction.predicted_views / total_score
                # But maintain minimum traffic for statistical significance
                split[variant.variant_id] = max(0.1, weight)
            
            # Normalize to sum to 1
            total_weight = sum(split.values())
            split = {k: v / total_weight for k, v in split.items()}
        
        return split
    
    async def analyze_test_results(
        self,
        test_id: str,
        performance_data: Dict[str, List[float]]
    ) -> TestResult:
        """
        Analyze A/B test results and determine winner
        
        Args:
            test_id: Test identifier
            performance_data: Performance metrics for each variant
        
        Returns:
            Test results with recommendations
        """
        # Load test configuration
        test_config = self._load_test_config(test_id)
        if not test_config:
            raise ValueError(f"Test {test_id} not found")
        
        # Check if we have enough data
        total_samples = sum(len(data) for data in performance_data.values())
        if total_samples < self.minimum_sample_size:
            logger.warning(f"Insufficient sample size: {total_samples} < {self.minimum_sample_size}")
        
        # Perform statistical analysis
        winner, confidence, improvement = self._perform_statistical_test(
            performance_data,
            test_config['optimization_metric']
        )
        
        # Calculate detailed metrics
        metrics = self._calculate_test_metrics(performance_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            winner,
            test_config,
            metrics,
            confidence
        )
        
        return TestResult(
            test_id=test_id,
            winner=winner,
            confidence_level=confidence,
            improvement=improvement,
            sample_size=total_samples,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _perform_statistical_test(
        self,
        performance_data: Dict[str, List[float]],
        metric: str
    ) -> Tuple[str, float, float]:
        """Perform statistical significance test"""
        
        if len(performance_data) < 2:
            return "insufficient_data", 0.0, 0.0
        
        # Get control and best variant
        control_data = performance_data.get('control', [])
        if not control_data:
            return "no_control", 0.0, 0.0
        
        best_variant = None
        best_improvement = 0.0
        best_confidence = 0.0
        
        for variant_id, variant_data in performance_data.items():
            if variant_id == 'control' or not variant_data:
                continue
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(variant_data, control_data)
            confidence = 1 - p_value
            
            # Calculate improvement
            control_mean = np.mean(control_data)
            variant_mean = np.mean(variant_data)
            improvement = ((variant_mean - control_mean) / control_mean) * 100
            
            # Check if this is the best variant
            if confidence > best_confidence and improvement > best_improvement:
                best_variant = variant_id
                best_improvement = improvement
                best_confidence = confidence
        
        # Determine winner
        if best_confidence >= (1 - self.significance_level) and best_improvement > 0:
            return best_variant, best_confidence, best_improvement
        else:
            return "control", best_confidence, 0.0
    
    def _calculate_test_metrics(
        self,
        performance_data: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Calculate detailed test metrics"""
        metrics = {}
        
        for variant_id, data in performance_data.items():
            if data:
                metrics[f"{variant_id}_mean"] = np.mean(data)
                metrics[f"{variant_id}_std"] = np.std(data)
                metrics[f"{variant_id}_median"] = np.median(data)
                metrics[f"{variant_id}_samples"] = len(data)
        
        return metrics
    
    def _generate_recommendations(
        self,
        winner: str,
        test_config: Dict[str, Any],
        metrics: Dict[str, float],
        confidence: float
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if winner != "control":
            recommendations.append(f"Implement {winner} as it shows significant improvement")
            
            # Find what changed in the winning variant
            for variant in test_config['variants']:
                if variant['variant_id'] == winner:
                    changed = variant['metadata'].get('changed_parameter')
                    if changed:
                        recommendations.append(f"Focus on optimizing {changed} in future content")
        
        if confidence < 0.95:
            recommendations.append("Continue testing to increase statistical confidence")
        
        # Performance-based recommendations
        winner_mean = metrics.get(f"{winner}_mean", 0)
        if winner_mean > 10000:
            recommendations.append("Scale this content strategy across more videos")
        
        return recommendations
    
    async def optimize_publishing_schedule(
        self,
        channel_data: Dict[str, Any],
        historical_performance: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Optimize video publishing schedule
        
        Args:
            channel_data: Channel information
            historical_performance: Historical video performance data
        
        Returns:
            Optimized publishing schedule
        """
        # Analyze best performing times
        if not historical_performance.empty:
            # Group by hour and day of week
            historical_performance['hour'] = pd.to_datetime(
                historical_performance['publish_time']
            ).dt.hour
            historical_performance['weekday'] = pd.to_datetime(
                historical_performance['publish_time']
            ).dt.dayofweek
            
            # Find best times
            best_hours = historical_performance.groupby('hour')['views'].mean().nlargest(3)
            best_days = historical_performance.groupby('weekday')['views'].mean().nlargest(3)
            
            optimal_times = {
                'best_hours': best_hours.index.tolist(),
                'best_days': best_days.index.tolist(),
                'recommended_frequency': self._calculate_optimal_frequency(historical_performance)
            }
        else:
            # Default recommendations
            optimal_times = {
                'best_hours': [14, 15, 16],  # 2-4 PM
                'best_days': [1, 2, 3],  # Tue, Wed, Thu
                'recommended_frequency': 3  # Videos per week
            }
        
        # Generate schedule
        schedule = self._generate_publishing_schedule(
            optimal_times,
            channel_data.get('upload_schedule', 'weekly')
        )
        
        return {
            'optimal_times': optimal_times,
            'schedule': schedule,
            'estimated_improvement': self._estimate_schedule_improvement(optimal_times)
        }
    
    def _calculate_optimal_frequency(self, data: pd.DataFrame) -> int:
        """Calculate optimal publishing frequency"""
        # Analyze view decay with frequency
        if len(data) < 10:
            return 3  # Default to 3 videos per week
        
        # Group by week and count videos
        data['week'] = pd.to_datetime(data['publish_time']).dt.isocalendar().week
        weekly_counts = data.groupby('week').size()
        weekly_views = data.groupby('week')['views'].mean()
        
        # Find sweet spot
        if len(weekly_counts) > 0:
            # Correlation between frequency and views
            correlation = weekly_counts.corr(weekly_views)
            
            if correlation > 0:
                # More videos = more views
                return min(7, int(weekly_counts.mean() + 1))
            else:
                # Quality over quantity
                return max(2, int(weekly_counts.mean() - 1))
        
        return 3
    
    def _generate_publishing_schedule(
        self,
        optimal_times: Dict[str, Any],
        frequency: str
    ) -> List[Dict[str, Any]]:
        """Generate detailed publishing schedule"""
        schedule = []
        
        # Map frequency to number of videos
        frequency_map = {
            'daily': 7,
            'weekly': optimal_times['recommended_frequency'],
            'biweekly': 2,
            'monthly': 1
        }
        
        videos_per_week = frequency_map.get(frequency, 3)
        
        # Generate schedule for next 30 days
        current_date = datetime.now()
        for week in range(4):
            videos_this_week = 0
            for day in optimal_times['best_days'][:videos_per_week]:
                for hour in optimal_times['best_hours'][:1]:  # One video per day
                    publish_date = current_date + timedelta(
                        weeks=week,
                        days=day - current_date.weekday(),
                        hours=hour - current_date.hour
                    )
                    
                    schedule.append({
                        'publish_time': publish_date.isoformat(),
                        'day_of_week': publish_date.strftime('%A'),
                        'hour': hour,
                        'priority': 'high' if day in optimal_times['best_days'][:2] else 'medium'
                    })
                    
                    videos_this_week += 1
                    if videos_this_week >= videos_per_week:
                        break
                
                if videos_this_week >= videos_per_week:
                    break
        
        return schedule
    
    def _estimate_schedule_improvement(self, optimal_times: Dict[str, Any]) -> float:
        """Estimate potential improvement from optimized schedule"""
        # Based on typical improvements from schedule optimization
        base_improvement = 15.0  # 15% base improvement
        
        # Bonus for hitting prime times
        if 14 in optimal_times['best_hours'] or 15 in optimal_times['best_hours']:
            base_improvement += 5.0
        
        # Bonus for optimal frequency
        if optimal_times['recommended_frequency'] >= 3:
            base_improvement += 3.0
        
        return base_improvement
    
    def _generate_test_id(self, content: Dict[str, Any]) -> str:
        """Generate unique test ID"""
        test_data = {
            'content': content.get('title', ''),
            'timestamp': datetime.now().isoformat()
        }
        return hashlib.md5(json.dumps(test_data).encode()).hexdigest()[:12]
    
    def _store_test_config(self, test_id: str, config: Dict[str, Any]):
        """Store test configuration in Redis"""
        self.redis_client.setex(
            f"ab_test:{test_id}",
            86400 * 30,  # 30 days
            json.dumps(config)
        )
    
    def _load_test_config(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Load test configuration from Redis"""
        data = self.redis_client.get(f"ab_test:{test_id}")
        if data:
            return json.loads(data)
        return None
    
    def _serialize_prediction(self, prediction: PerformancePrediction) -> Dict[str, Any]:
        """Serialize prediction for storage"""
        return {
            'predicted_views': prediction.predicted_views,
            'predicted_watch_time': prediction.predicted_watch_time,
            'predicted_revenue': prediction.predicted_revenue,
            'predicted_ctr': prediction.predicted_ctr,
            'confidence_interval': prediction.confidence_interval,
            'risk_score': prediction.risk_score
        }
    
    def _load_models(self, model_path: str):
        """Load pre-trained models"""
        # Implementation would load actual trained models
        logger.info(f"Loading models from {model_path}")


class ContentOptimizationAPI:
    """FastAPI integration for content optimization"""
    
    def __init__(self):
        self.optimizer = ContentOptimizer()
    
    async def create_test(
        self,
        content: Dict[str, Any],
        test_params: Dict[str, List[Any]],
        metric: str = "ctr"
    ) -> Dict[str, Any]:
        """Create A/B test via API"""
        
        test_config = await self.optimizer.create_ab_test(
            content,
            test_params,
            OptimizationMetric(metric)
        )
        
        return {
            'test_id': test_config['test_id'],
            'status': test_config['status'],
            'variants': len(test_config['variants']),
            'traffic_split': test_config['traffic_split'],
            'predictions': test_config['predictions']
        }
    
    async def analyze_test(
        self,
        test_id: str,
        performance_data: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Analyze test results via API"""
        
        result = await self.optimizer.analyze_test_results(
            test_id,
            performance_data
        )
        
        return {
            'winner': result.winner,
            'confidence': result.confidence_level,
            'improvement': result.improvement,
            'recommendations': result.recommendations
        }


# Initialize global instance
optimization_api = ContentOptimizationAPI()