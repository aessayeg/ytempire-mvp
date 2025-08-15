"""
A/B Testing Service
Comprehensive experiment management and statistical analysis
"""
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from enum import Enum
import numpy as np
from scipy import stats
import hashlib
import json
import logging
from dataclasses import dataclass

from app.models.user import User
from app.core.cache import cache_service

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class VariantAssignmentMethod(Enum):
    RANDOM = "random"
    DETERMINISTIC = "deterministic"
    WEIGHTED = "weighted"
    SEQUENTIAL = "sequential"


@dataclass
class ExperimentVariant:
    """Experiment variant configuration"""
    name: str
    allocation: float  # Percentage allocation (0-100)
    config: Dict[str, Any]  # Variant-specific configuration
    is_control: bool = False


@dataclass
class ExperimentMetrics:
    """Metrics for experiment analysis"""
    variant: str
    sample_size: int
    conversions: int
    conversion_rate: float
    revenue: float
    avg_revenue_per_user: float
    confidence_interval: Tuple[float, float]
    p_value: Optional[float] = None
    is_significant: bool = False
    lift: Optional[float] = None


class ABTestingService:
    """Service for A/B testing and experimentation"""
    
    def __init__(self):
        self.cache_ttl = 300  # 5 minutes cache
        self.minimum_sample_size = 100  # Minimum users per variant
        self.significance_level = 0.05  # 95% confidence
        
    async def create_experiment(
        self,
        db: AsyncSession,
        name: str,
        description: str,
        hypothesis: str,
        variants: List[ExperimentVariant],
        target_metric: str,
        target_audience: Optional[Dict[str, Any]] = None,
        duration_days: Optional[int] = None,
        min_sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a new A/B test experiment"""
        from app.models.experiment import Experiment
        
        # Validate variants
        if len(variants) < 2:
            raise ValueError("Experiment must have at least 2 variants")
            
        total_allocation = sum(v.allocation for v in variants)
        if abs(total_allocation - 100) > 0.01:
            raise ValueError(f"Variant allocations must sum to 100%, got {total_allocation}%")
            
        # Ensure one control variant
        control_variants = [v for v in variants if v.is_control]
        if len(control_variants) != 1:
            raise ValueError("Experiment must have exactly one control variant")
            
        # Create experiment
        experiment = Experiment(
            name=name,
            description=description,
            hypothesis=hypothesis,
            status=ExperimentStatus.DRAFT.value,
            variants=[
                {
                    'name': v.name,
                    'allocation': v.allocation,
                    'config': v.config,
                    'is_control': v.is_control
                }
                for v in variants
            ],
            target_metric=target_metric,
            target_audience=target_audience or {},
            start_date=None,
            end_date=None,
            duration_days=duration_days,
            min_sample_size=min_sample_size or self.minimum_sample_size,
            created_at=datetime.utcnow()
        )
        
        db.add(experiment)
        await db.commit()
        await db.refresh(experiment)
        
        return {
            'experiment_id': experiment.id,
            'name': experiment.name,
            'status': experiment.status,
            'variants': experiment.variants
        }
        
    async def start_experiment(
        self,
        db: AsyncSession,
        experiment_id: int
    ) -> Dict[str, Any]:
        """Start an experiment"""
        from app.models.experiment import Experiment
        
        # Get experiment
        exp_query = select(Experiment).where(Experiment.id == experiment_id)
        exp_result = await db.execute(exp_query)
        experiment = exp_result.scalar_one_or_none()
        
        if not experiment:
            raise ValueError("Experiment not found")
            
        if experiment.status != ExperimentStatus.DRAFT.value:
            raise ValueError(f"Cannot start experiment in {experiment.status} status")
            
        # Update experiment status
        experiment.status = ExperimentStatus.RUNNING.value
        experiment.start_date = datetime.utcnow()
        
        if experiment.duration_days:
            experiment.end_date = experiment.start_date + timedelta(days=experiment.duration_days)
            
        await db.commit()
        
        # Clear assignment cache
        await self._clear_experiment_cache(experiment_id)
        
        return {
            'experiment_id': experiment_id,
            'status': experiment.status,
            'start_date': experiment.start_date.isoformat()
        }
        
    async def get_variant_assignment(
        self,
        db: AsyncSession,
        experiment_id: int,
        user_id: int
    ) -> Dict[str, Any]:
        """Get or assign variant for a user"""
        # Check cache first
        cache_key = f"experiment:{experiment_id}:user:{user_id}"
        cached = await cache_service.get(cache_key)
        if cached:
            return cached
            
        from app.models.experiment import Experiment, ExperimentAssignment
        
        # Get experiment
        exp_query = select(Experiment).where(
            and_(
                Experiment.id == experiment_id,
                Experiment.status == ExperimentStatus.RUNNING.value
            )
        )
        exp_result = await db.execute(exp_query)
        experiment = exp_result.scalar_one_or_none()
        
        if not experiment:
            return {'variant': None, 'experiment_active': False}
            
        # Check if user meets target audience criteria
        if not await self._user_meets_criteria(db, user_id, experiment.target_audience):
            return {'variant': None, 'experiment_active': True, 'excluded': True}
            
        # Check existing assignment
        assign_query = select(ExperimentAssignment).where(
            and_(
                ExperimentAssignment.experiment_id == experiment_id,
                ExperimentAssignment.user_id == user_id
            )
        )
        assign_result = await db.execute(assign_query)
        assignment = assign_result.scalar_one_or_none()
        
        if assignment:
            result = {
                'variant': assignment.variant_name,
                'experiment_active': True,
                'config': self._get_variant_config(experiment.variants, assignment.variant_name)
            }
        else:
            # Assign variant
            variant = self._assign_variant(user_id, experiment.variants)
            
            # Store assignment
            assignment = ExperimentAssignment(
                experiment_id=experiment_id,
                user_id=user_id,
                variant_name=variant['name'],
                assigned_at=datetime.utcnow()
            )
            db.add(assignment)
            await db.commit()
            
            result = {
                'variant': variant['name'],
                'experiment_active': True,
                'config': variant.get('config', {})
            }
            
        # Cache assignment
        await cache_service.set(cache_key, result, self.cache_ttl)
        return result
        
    async def track_conversion(
        self,
        db: AsyncSession,
        experiment_id: int,
        user_id: int,
        metric_name: str,
        metric_value: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Track a conversion event for an experiment"""
        from app.models.experiment import ExperimentAssignment, ExperimentEvent
        
        # Get user's assignment
        assign_query = select(ExperimentAssignment).where(
            and_(
                ExperimentAssignment.experiment_id == experiment_id,
                ExperimentAssignment.user_id == user_id
            )
        )
        assign_result = await db.execute(assign_query)
        assignment = assign_result.scalar_one_or_none()
        
        if not assignment:
            return False  # User not in experiment
            
        # Record conversion event
        event = ExperimentEvent(
            experiment_id=experiment_id,
            user_id=user_id,
            variant_name=assignment.variant_name,
            event_type='conversion',
            metric_name=metric_name,
            metric_value=metric_value,
            metadata=metadata or {},
            timestamp=datetime.utcnow()
        )
        
        db.add(event)
        
        # Update assignment conversion status
        if not assignment.converted:
            assignment.converted = True
            assignment.conversion_time = datetime.utcnow()
            assignment.conversion_value = metric_value
            
        await db.commit()
        
        # Clear results cache
        await self._clear_results_cache(experiment_id)
        
        return True
        
    async def get_experiment_results(
        self,
        db: AsyncSession,
        experiment_id: int
    ) -> Dict[str, Any]:
        """Get comprehensive experiment results with statistical analysis"""
        # Check cache
        cache_key = f"experiment:results:{experiment_id}"
        cached = await cache_service.get(cache_key)
        if cached:
            return cached
            
        from app.models.experiment import Experiment, ExperimentAssignment, ExperimentEvent
        
        # Get experiment
        exp_query = select(Experiment).where(Experiment.id == experiment_id)
        exp_result = await db.execute(exp_query)
        experiment = exp_result.scalar_one_or_none()
        
        if not experiment:
            raise ValueError("Experiment not found")
            
        # Get metrics for each variant
        variant_metrics = []
        control_metrics = None
        
        for variant in experiment.variants:
            # Get assignments for variant
            assign_query = select(ExperimentAssignment).where(
                and_(
                    ExperimentAssignment.experiment_id == experiment_id,
                    ExperimentAssignment.variant_name == variant['name']
                )
            )
            assign_result = await db.execute(assign_query)
            assignments = assign_result.scalars().all()
            
            sample_size = len(assignments)
            conversions = sum(1 for a in assignments if a.converted)
            revenue = sum(a.conversion_value or 0 for a in assignments)
            
            # Calculate metrics
            conversion_rate = conversions / sample_size if sample_size > 0 else 0
            avg_revenue = revenue / sample_size if sample_size > 0 else 0
            
            # Confidence interval for conversion rate
            ci_lower, ci_upper = self._calculate_confidence_interval(
                conversions, sample_size
            )
            
            metrics = ExperimentMetrics(
                variant=variant['name'],
                sample_size=sample_size,
                conversions=conversions,
                conversion_rate=conversion_rate,
                revenue=revenue,
                avg_revenue_per_user=avg_revenue,
                confidence_interval=(ci_lower, ci_upper)
            )
            
            variant_metrics.append(metrics)
            
            if variant.get('is_control'):
                control_metrics = metrics
                
        # Statistical significance testing
        if control_metrics and len(variant_metrics) > 1:
            for metrics in variant_metrics:
                if metrics.variant != control_metrics.variant:
                    # Calculate p-value
                    p_value = self._calculate_p_value(
                        control_metrics.conversions,
                        control_metrics.sample_size,
                        metrics.conversions,
                        metrics.sample_size
                    )
                    
                    metrics.p_value = p_value
                    metrics.is_significant = p_value < self.significance_level
                    
                    # Calculate lift
                    if control_metrics.conversion_rate > 0:
                        metrics.lift = (
                            (metrics.conversion_rate - control_metrics.conversion_rate) 
                            / control_metrics.conversion_rate * 100
                        )
                        
        # Determine winner
        winner = self._determine_winner(variant_metrics)
        
        # Calculate required sample size for significance
        required_sample_size = self._calculate_required_sample_size(
            control_metrics.conversion_rate if control_metrics else 0.1
        )
        
        results = {
            'experiment_id': experiment_id,
            'name': experiment.name,
            'status': experiment.status,
            'variants': [
                {
                    'name': m.variant,
                    'sample_size': m.sample_size,
                    'conversions': m.conversions,
                    'conversion_rate': m.conversion_rate,
                    'confidence_interval': m.confidence_interval,
                    'revenue': m.revenue,
                    'avg_revenue_per_user': m.avg_revenue_per_user,
                    'p_value': m.p_value,
                    'is_significant': m.is_significant,
                    'lift': m.lift
                }
                for m in variant_metrics
            ],
            'winner': winner,
            'required_sample_size': required_sample_size,
            'can_conclude': all(
                m.sample_size >= required_sample_size 
                for m in variant_metrics
            )
        }
        
        # Cache results
        await cache_service.set(cache_key, results, self.cache_ttl)
        
        return results
        
    async def conclude_experiment(
        self,
        db: AsyncSession,
        experiment_id: int,
        winner_variant: Optional[str] = None
    ) -> Dict[str, Any]:
        """Conclude an experiment and optionally declare a winner"""
        from app.models.experiment import Experiment
        
        # Get experiment
        exp_query = select(Experiment).where(Experiment.id == experiment_id)
        exp_result = await db.execute(exp_query)
        experiment = exp_result.scalar_one_or_none()
        
        if not experiment:
            raise ValueError("Experiment not found")
            
        if experiment.status != ExperimentStatus.RUNNING.value:
            raise ValueError(f"Cannot conclude experiment in {experiment.status} status")
            
        # Get final results
        results = await self.get_experiment_results(db, experiment_id)
        
        # Update experiment
        experiment.status = ExperimentStatus.COMPLETED.value
        experiment.end_date = datetime.utcnow()
        experiment.winner_variant = winner_variant or results.get('winner')
        experiment.final_results = results
        
        await db.commit()
        
        # Clear all caches
        await self._clear_experiment_cache(experiment_id)
        await self._clear_results_cache(experiment_id)
        
        return {
            'experiment_id': experiment_id,
            'status': experiment.status,
            'winner': experiment.winner_variant,
            'results': results
        }
        
    async def get_active_experiments(
        self,
        db: AsyncSession,
        user_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get list of active experiments"""
        from app.models.experiment import Experiment
        
        query = select(Experiment).where(
            Experiment.status == ExperimentStatus.RUNNING.value
        )
        
        result = await db.execute(query)
        experiments = result.scalars().all()
        
        active_experiments = []
        for exp in experiments:
            exp_data = {
                'id': exp.id,
                'name': exp.name,
                'description': exp.description,
                'target_metric': exp.target_metric,
                'variants': exp.variants
            }
            
            # If user_id provided, include their assignment
            if user_id:
                assignment = await self.get_variant_assignment(db, exp.id, user_id)
                exp_data['user_variant'] = assignment.get('variant')
                
            active_experiments.append(exp_data)
            
        return active_experiments
        
    # Private helper methods
    async def _user_meets_criteria(
        self,
        db: AsyncSession,
        user_id: int,
        criteria: Dict[str, Any]
    ) -> bool:
        """Check if user meets experiment targeting criteria"""
        if not criteria:
            return True
            
        from app.models.user import User
        
        user_query = select(User).where(User.id == user_id)
        user_result = await db.execute(user_query)
        user = user_result.scalar_one_or_none()
        
        if not user:
            return False
            
        # Check various criteria
        if 'min_account_age_days' in criteria:
            account_age = (datetime.utcnow() - user.created_at).days
            if account_age < criteria['min_account_age_days']:
                return False
                
        if 'user_type' in criteria:
            if user.user_type not in criteria['user_type']:
                return False
                
        if 'has_subscription' in criteria:
            if bool(user.subscription_id) != criteria['has_subscription']:
                return False
                
        return True
        
    def _assign_variant(
        self,
        user_id: int,
        variants: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assign a variant to a user deterministically"""
        # Use deterministic hash for consistent assignment
        hash_input = f"{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Map hash to variant based on allocation
        position = hash_value % 100
        cumulative = 0
        
        for variant in variants:
            cumulative += variant['allocation']
            if position < cumulative:
                return variant
                
        # Fallback to last variant
        return variants[-1]
        
    def _get_variant_config(
        self,
        variants: List[Dict[str, Any]],
        variant_name: str
    ) -> Dict[str, Any]:
        """Get configuration for a specific variant"""
        for variant in variants:
            if variant['name'] == variant_name:
                return variant.get('config', {})
        return {}
        
    def _calculate_confidence_interval(
        self,
        successes: int,
        trials: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for proportion"""
        if trials == 0:
            return (0, 0)
            
        # Wilson score interval
        z = stats.norm.ppf((1 + confidence) / 2)
        p_hat = successes / trials
        
        denominator = 1 + z**2 / trials
        center = (p_hat + z**2 / (2 * trials)) / denominator
        spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denominator
        
        return (max(0, center - spread), min(1, center + spread))
        
    def _calculate_p_value(
        self,
        control_successes: int,
        control_trials: int,
        variant_successes: int,
        variant_trials: int
    ) -> float:
        """Calculate p-value using chi-squared test"""
        if control_trials == 0 or variant_trials == 0:
            return 1.0
            
        # Create contingency table
        table = [
            [control_successes, control_trials - control_successes],
            [variant_successes, variant_trials - variant_successes]
        ]
        
        # Perform chi-squared test
        try:
            _, p_value, _, _ = stats.chi2_contingency(table)
            return p_value
        except:
            return 1.0
            
    def _calculate_required_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float = 0.1,
        power: float = 0.8,
        significance: float = 0.05
    ) -> int:
        """Calculate required sample size for statistical significance"""
        if baseline_rate <= 0 or baseline_rate >= 1:
            return self.minimum_sample_size
            
        # Use power analysis formula
        z_alpha = stats.norm.ppf(1 - significance / 2)
        z_beta = stats.norm.ppf(power)
        
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        p_bar = (p1 + p2) / 2
        
        n = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + 
             z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2 / (p1 - p2)**2
             
        return max(self.minimum_sample_size, int(np.ceil(n)))
        
    def _determine_winner(
        self,
        metrics: List[ExperimentMetrics]
    ) -> Optional[str]:
        """Determine winning variant based on statistical significance"""
        if not metrics:
            return None
            
        # Find control
        control = next((m for m in metrics if m.variant == 'control'), None)
        if not control:
            control = metrics[0]
            
        # Find best performing variant with significance
        best_variant = control.variant
        best_rate = control.conversion_rate
        
        for m in metrics:
            if m.variant != control.variant:
                if m.is_significant and m.conversion_rate > best_rate:
                    best_variant = m.variant
                    best_rate = m.conversion_rate
                    
        # Only declare winner if significantly better than control
        if best_variant != control.variant:
            return best_variant
            
        return None
        
    async def _clear_experiment_cache(self, experiment_id: int):
        """Clear experiment assignment cache"""
        pattern = f"experiment:{experiment_id}:*"
        await cache_service.delete_pattern(pattern)
        
    async def _clear_results_cache(self, experiment_id: int):
        """Clear experiment results cache"""
        cache_key = f"experiment:results:{experiment_id}"
        await cache_service.delete(cache_key)


# Create singleton instance
ab_testing_service = ABTestingService()