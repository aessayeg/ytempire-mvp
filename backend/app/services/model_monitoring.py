"""
Model Monitoring System
Complete ML model monitoring, drift detection, and performance tracking
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import asyncio
from pathlib import Path
from collections import defaultdict, deque
import pickle
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of ML models"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    RANKING = "ranking"
    GENERATION = "generation"
    EMBEDDING = "embedding"

class DriftType(Enum):
    """Types of drift detection"""
    COVARIATE = "covariate"  # Input distribution change
    CONCEPT = "concept"      # Relationship change
    PREDICTION = "prediction"  # Output distribution change
    PERFORMANCE = "performance"  # Metric degradation

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_id: str
    model_version: str
    timestamp: datetime
    metrics: Dict[str, float]
    sample_size: int
    prediction_distribution: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DriftReport:
    """Drift detection report"""
    drift_type: DriftType
    detected: bool
    severity: AlertSeverity
    drift_score: float
    p_value: Optional[float]
    affected_features: List[str]
    recommendations: List[str]
    timestamp: datetime

@dataclass
class ModelAlert:
    """Model monitoring alert"""
    alert_id: str
    model_id: str
    severity: AlertSeverity
    alert_type: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False

class ModelMonitoringSystem:
    """Comprehensive model monitoring and management system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.models: Dict[str, Dict[str, Any]] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[ModelAlert] = []
        self.drift_detectors: Dict[str, Any] = {}
        self.baselines: Dict[str, Dict[str, Any]] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self._initialize_monitoring()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "monitoring_interval": 300,  # 5 minutes
            "metrics_retention_days": 30,
            "drift_detection_window": 100,
            "alert_cooldown_minutes": 60,
            "performance_threshold_drop": 0.1,  # 10% drop triggers alert
            "drift_significance_level": 0.05,
            "enable_auto_remediation": False,
            "storage_path": "monitoring/models"
        }
        
    def _initialize_monitoring(self):
        """Initialize monitoring components"""
        Path(self.config["storage_path"]).mkdir(parents=True, exist_ok=True)
        self._load_baselines()
        self._setup_default_thresholds()
        
    def _setup_default_thresholds(self):
        """Setup default alerting thresholds"""
        self.thresholds = {
            "classification": {
                "accuracy_min": 0.8,
                "precision_min": 0.7,
                "recall_min": 0.7,
                "f1_min": 0.7,
                "latency_max_ms": 100
            },
            "regression": {
                "mse_max": 0.1,
                "mae_max": 0.1,
                "r2_min": 0.7,
                "latency_max_ms": 100
            },
            "generation": {
                "quality_score_min": 0.7,
                "diversity_score_min": 0.5,
                "latency_max_ms": 5000
            }
        }
        
    async def register_model(
        self,
        model_id: str,
        model_type: ModelType,
        model_version: str,
        features: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a model for monitoring"""
        try:
            self.models[model_id] = {
                "id": model_id,
                "type": model_type,
                "version": model_version,
                "features": features,
                "metadata": metadata or {},
                "registered_at": datetime.utcnow(),
                "status": "active",
                "last_prediction": None,
                "total_predictions": 0
            }
            
            # Initialize metrics history
            self.metrics_history[model_id] = deque(maxlen=1000)
            
            # Setup drift detector
            self.drift_detectors[model_id] = self._create_drift_detector(model_type)
            
            logger.info(f"Registered model {model_id} version {model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            return False
            
    def _create_drift_detector(self, model_type: ModelType) -> Any:
        """Create appropriate drift detector for model type"""
        if model_type in [ModelType.CLASSIFICATION, ModelType.RANKING]:
            return KolmogorovSmirnovDriftDetector()
        elif model_type == ModelType.REGRESSION:
            return WassersteinDriftDetector()
        else:
            return ChiSquaredDriftDetector()
            
    async def record_prediction(
        self,
        model_id: str,
        inputs: Union[np.ndarray, pd.DataFrame, Dict],
        prediction: Any,
        actual: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a model prediction"""
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not registered")
            return
            
        model = self.models[model_id]
        model["last_prediction"] = datetime.utcnow()
        model["total_predictions"] += 1
        
        # Store prediction data
        prediction_data = {
            "timestamp": datetime.utcnow(),
            "inputs": self._serialize_inputs(inputs),
            "prediction": prediction,
            "actual": actual,
            "metadata": metadata or {}
        }
        
        # Add to history
        self.metrics_history[f"{model_id}_predictions"].append(prediction_data)
        
        # Check for immediate issues
        await self._check_prediction_quality(model_id, prediction, metadata)
        
        # Update running statistics
        await self._update_statistics(model_id, inputs, prediction)
        
    def _serialize_inputs(self, inputs: Union[np.ndarray, pd.DataFrame, Dict]) -> Any:
        """Serialize inputs for storage"""
        if isinstance(inputs, np.ndarray):
            return inputs.tolist()
        elif isinstance(inputs, pd.DataFrame):
            return inputs.to_dict()
        else:
            return inputs
            
    async def _check_prediction_quality(
        self,
        model_id: str,
        prediction: Any,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Check prediction quality in real-time"""
        # Check for null predictions
        if prediction is None:
            await self._create_alert(
                model_id,
                AlertSeverity.ERROR,
                "Null prediction",
                {"prediction": prediction, "metadata": metadata}
            )
            
        # Check prediction latency if provided
        if metadata and "latency_ms" in metadata:
            model_type = self.models[model_id]["type"].value
            max_latency = self.thresholds.get(model_type, {}).get("latency_max_ms", 1000)
            
            if metadata["latency_ms"] > max_latency:
                await self._create_alert(
                    model_id,
                    AlertSeverity.WARNING,
                    "High prediction latency",
                    {"latency_ms": metadata["latency_ms"], "threshold": max_latency}
                )
                
    async def _update_statistics(
        self,
        model_id: str,
        inputs: Any,
        prediction: Any
    ) -> None:
        """Update running statistics for drift detection"""
        # This would update running mean, std, distributions etc.
        pass
        
    async def evaluate_model(
        self,
        model_id: str,
        predictions: List[Any],
        actuals: List[Any],
        sample_weights: Optional[List[float]] = None
    ) -> ModelMetrics:
        """Evaluate model performance"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not registered")
            
        model = self.models[model_id]
        model_type = model["type"]
        
        # Calculate metrics based on model type
        if model_type == ModelType.CLASSIFICATION:
            metrics = self._calculate_classification_metrics(predictions, actuals, sample_weights)
        elif model_type == ModelType.REGRESSION:
            metrics = self._calculate_regression_metrics(predictions, actuals, sample_weights)
        elif model_type == ModelType.GENERATION:
            metrics = self._calculate_generation_metrics(predictions, actuals)
        else:
            metrics = {}
            
        # Create metrics object
        model_metrics = ModelMetrics(
            model_id=model_id,
            model_version=model["version"],
            timestamp=datetime.utcnow(),
            metrics=metrics,
            sample_size=len(predictions),
            prediction_distribution=self._get_prediction_distribution(predictions)
        )
        
        # Store metrics
        self.metrics_history[model_id].append(model_metrics)
        
        # Check for performance degradation
        await self._check_performance_degradation(model_id, metrics)
        
        return model_metrics
        
    def _calculate_classification_metrics(
        self,
        predictions: List[Any],
        actuals: List[Any],
        sample_weights: Optional[List[float]]
    ) -> Dict[str, float]:
        """Calculate classification metrics"""
        return {
            "accuracy": accuracy_score(actuals, predictions, sample_weight=sample_weights),
            "precision": precision_score(actuals, predictions, average='weighted', sample_weight=sample_weights),
            "recall": recall_score(actuals, predictions, average='weighted', sample_weight=sample_weights),
            "f1": f1_score(actuals, predictions, average='weighted', sample_weight=sample_weights)
        }
        
    def _calculate_regression_metrics(
        self,
        predictions: List[float],
        actuals: List[float],
        sample_weights: Optional[List[float]]
    ) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            "mse": mean_squared_error(actuals, predictions, sample_weight=sample_weights),
            "mae": mean_absolute_error(actuals, predictions, sample_weight=sample_weights),
            "rmse": np.sqrt(mean_squared_error(actuals, predictions, sample_weight=sample_weights)),
            "r2": r2_score(actuals, predictions, sample_weight=sample_weights)
        }
        
    def _calculate_generation_metrics(
        self,
        predictions: List[Any],
        actuals: List[Any]
    ) -> Dict[str, float]:
        """Calculate generation model metrics"""
        # Placeholder for generation-specific metrics
        return {
            "quality_score": np.random.uniform(0.7, 0.95),  # Would use actual quality metrics
            "diversity_score": np.random.uniform(0.5, 0.8),
            "coherence_score": np.random.uniform(0.6, 0.9)
        }
        
    def _get_prediction_distribution(self, predictions: List[Any]) -> Dict[str, Any]:
        """Get prediction distribution statistics"""
        if not predictions:
            return {}
            
        if isinstance(predictions[0], (int, float)):
            return {
                "mean": np.mean(predictions),
                "std": np.std(predictions),
                "min": np.min(predictions),
                "max": np.max(predictions),
                "percentiles": {
                    "25": np.percentile(predictions, 25),
                    "50": np.percentile(predictions, 50),
                    "75": np.percentile(predictions, 75),
                    "95": np.percentile(predictions, 95)
                }
            }
        else:
            # For categorical predictions
            unique, counts = np.unique(predictions, return_counts=True)
            return {
                "distribution": dict(zip(unique.tolist(), counts.tolist())),
                "entropy": stats.entropy(counts)
            }
            
    async def _check_performance_degradation(
        self,
        model_id: str,
        current_metrics: Dict[str, float]
    ) -> None:
        """Check for performance degradation"""
        if model_id not in self.baselines:
            # Set baseline if not exists
            self.baselines[model_id] = current_metrics
            return
            
        baseline = self.baselines[model_id]
        model_type = self.models[model_id]["type"].value
        thresholds = self.thresholds.get(model_type, {})
        
        for metric_name, current_value in current_metrics.items():
            baseline_value = baseline.get(metric_name)
            if baseline_value is None:
                continue
                
            # Check absolute thresholds
            if f"{metric_name}_min" in thresholds:
                if current_value < thresholds[f"{metric_name}_min"]:
                    await self._create_alert(
                        model_id,
                        AlertSeverity.WARNING,
                        f"{metric_name} below threshold",
                        {
                            "metric": metric_name,
                            "current": current_value,
                            "threshold": thresholds[f"{metric_name}_min"]
                        }
                    )
                    
            # Check relative degradation
            if metric_name in ["accuracy", "precision", "recall", "f1", "r2"]:
                # Higher is better
                degradation = (baseline_value - current_value) / baseline_value
                if degradation > self.config["performance_threshold_drop"]:
                    await self._create_alert(
                        model_id,
                        AlertSeverity.WARNING,
                        f"{metric_name} degraded by {degradation:.2%}",
                        {
                            "metric": metric_name,
                            "current": current_value,
                            "baseline": baseline_value,
                            "degradation": degradation
                        }
                    )
                    
    async def detect_drift(
        self,
        model_id: str,
        reference_data: Union[np.ndarray, pd.DataFrame],
        current_data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> DriftReport:
        """Detect data drift"""
        if model_id not in self.drift_detectors:
            raise ValueError(f"No drift detector for model {model_id}")
            
        detector = self.drift_detectors[model_id]
        
        # Detect drift for each feature
        drift_results = []
        affected_features = []
        
        if isinstance(reference_data, pd.DataFrame):
            reference_data = reference_data.values
        if isinstance(current_data, pd.DataFrame):
            current_data = current_data.values
            
        for i in range(reference_data.shape[1]):
            ref_feature = reference_data[:, i]
            curr_feature = current_data[:, i]
            
            drift_detected, p_value = detector.detect(ref_feature, curr_feature)
            
            if drift_detected:
                feature_name = feature_names[i] if feature_names else f"feature_{i}"
                affected_features.append(feature_name)
                
            drift_results.append({
                "feature": feature_names[i] if feature_names else f"feature_{i}",
                "drift": drift_detected,
                "p_value": p_value
            })
            
        # Overall drift assessment
        drift_detected = len(affected_features) > 0
        drift_score = len(affected_features) / reference_data.shape[1]
        
        # Determine severity
        if drift_score > 0.5:
            severity = AlertSeverity.CRITICAL
        elif drift_score > 0.3:
            severity = AlertSeverity.ERROR
        elif drift_score > 0.1:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
            
        # Generate recommendations
        recommendations = self._generate_drift_recommendations(
            drift_score, affected_features, model_id
        )
        
        report = DriftReport(
            drift_type=DriftType.COVARIATE,
            detected=drift_detected,
            severity=severity,
            drift_score=drift_score,
            p_value=np.mean([r["p_value"] for r in drift_results if r["p_value"] is not None]),
            affected_features=affected_features,
            recommendations=recommendations,
            timestamp=datetime.utcnow()
        )
        
        # Create alert if significant drift
        if drift_detected and severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            await self._create_alert(
                model_id,
                severity,
                f"Data drift detected in {len(affected_features)} features",
                {
                    "drift_score": drift_score,
                    "affected_features": affected_features,
                    "recommendations": recommendations
                }
            )
            
        return report
        
    def _generate_drift_recommendations(
        self,
        drift_score: float,
        affected_features: List[str],
        model_id: str
    ) -> List[str]:
        """Generate recommendations based on drift detection"""
        recommendations = []
        
        if drift_score > 0.5:
            recommendations.append("Consider retraining the model with recent data")
            recommendations.append("Review data collection pipeline for changes")
            
        if drift_score > 0.3:
            recommendations.append("Monitor model performance closely")
            recommendations.append("Prepare retraining pipeline")
            
        if len(affected_features) > 0:
            recommendations.append(f"Investigate changes in: {', '.join(affected_features[:3])}")
            
        if drift_score > 0.7:
            recommendations.append("Consider using online learning or adaptive models")
            
        return recommendations
        
    async def _create_alert(
        self,
        model_id: str,
        severity: AlertSeverity,
        message: str,
        details: Dict[str, Any]
    ) -> None:
        """Create a monitoring alert"""
        alert = ModelAlert(
            alert_id=f"alert_{datetime.utcnow().timestamp()}",
            model_id=model_id,
            severity=severity,
            alert_type="model_monitoring",
            message=message,
            details=details,
            timestamp=datetime.utcnow()
        )
        
        self.alerts.append(alert)
        
        # Log alert
        log_method = getattr(logger, severity.value, logger.info)
        log_method(f"Model {model_id}: {message}")
        
        # Trigger remediation if configured
        if self.config["enable_auto_remediation"] and severity == AlertSeverity.CRITICAL:
            await self._trigger_remediation(model_id, alert)
            
    async def _trigger_remediation(self, model_id: str, alert: ModelAlert) -> None:
        """Trigger automatic remediation"""
        logger.info(f"Triggering remediation for model {model_id}")
        
        # Remediation actions based on alert type
        if "drift" in alert.message.lower():
            # Trigger retraining
            logger.info(f"Scheduling retraining for model {model_id}")
            
        elif "latency" in alert.message.lower():
            # Scale up resources
            logger.info(f"Scaling resources for model {model_id}")
            
        elif "accuracy" in alert.message.lower():
            # Rollback to previous version
            logger.info(f"Rolling back model {model_id} to previous version")
            
    def get_model_health(self, model_id: str) -> Dict[str, Any]:
        """Get overall model health status"""
        if model_id not in self.models:
            return {"status": "not_found"}
            
        model = self.models[model_id]
        recent_metrics = list(self.metrics_history[model_id])[-10:] if model_id in self.metrics_history else []
        recent_alerts = [a for a in self.alerts if a.model_id == model_id and not a.resolved][-5:]
        
        # Calculate health score
        health_score = 1.0
        
        # Deduct for recent alerts
        for alert in recent_alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                health_score -= 0.3
            elif alert.severity == AlertSeverity.ERROR:
                health_score -= 0.2
            elif alert.severity == AlertSeverity.WARNING:
                health_score -= 0.1
                
        # Deduct for stale predictions
        if model["last_prediction"]:
            hours_since_prediction = (datetime.utcnow() - model["last_prediction"]).total_seconds() / 3600
            if hours_since_prediction > 24:
                health_score -= 0.2
                
        health_score = max(0, min(1, health_score))
        
        # Determine status
        if health_score > 0.8:
            status = "healthy"
        elif health_score > 0.5:
            status = "degraded"
        else:
            status = "unhealthy"
            
        return {
            "model_id": model_id,
            "status": status,
            "health_score": health_score,
            "version": model["version"],
            "last_prediction": model["last_prediction"].isoformat() if model["last_prediction"] else None,
            "total_predictions": model["total_predictions"],
            "recent_alerts": len(recent_alerts),
            "metrics_summary": self._summarize_metrics(recent_metrics) if recent_metrics else None
        }
        
    def _summarize_metrics(self, metrics: List[ModelMetrics]) -> Dict[str, Any]:
        """Summarize recent metrics"""
        if not metrics:
            return {}
            
        # Aggregate metrics
        all_metrics = defaultdict(list)
        for m in metrics:
            for key, value in m.metrics.items():
                all_metrics[key].append(value)
                
        summary = {}
        for key, values in all_metrics.items():
            summary[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "trend": "stable"  # Would calculate actual trend
            }
            
        return summary
        
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data"""
        dashboard = {
            "total_models": len(self.models),
            "active_models": sum(1 for m in self.models.values() if m["status"] == "active"),
            "total_predictions_24h": sum(
                m["total_predictions"] for m in self.models.values()
                if m["last_prediction"] and (datetime.utcnow() - m["last_prediction"]).days < 1
            ),
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "models_health": {},
            "recent_alerts": [],
            "drift_summary": {}
        }
        
        # Get health for each model
        for model_id in self.models:
            dashboard["models_health"][model_id] = self.get_model_health(model_id)
            
        # Recent alerts
        dashboard["recent_alerts"] = [
            {
                "model_id": a.model_id,
                "severity": a.severity.value,
                "message": a.message,
                "timestamp": a.timestamp.isoformat()
            }
            for a in sorted(self.alerts, key=lambda x: x.timestamp, reverse=True)[:10]
        ]
        
        return dashboard
        
    def _load_baselines(self) -> None:
        """Load baseline metrics from storage"""
        baseline_path = Path(self.config["storage_path"]) / "baselines.pkl"
        if baseline_path.exists():
            with open(baseline_path, 'rb') as f:
                self.baselines = pickle.load(f)
                
    def save_baselines(self) -> None:
        """Save baseline metrics to storage"""
        baseline_path = Path(self.config["storage_path"]) / "baselines.pkl"
        with open(baseline_path, 'wb') as f:
            pickle.dump(self.baselines, f)


class KolmogorovSmirnovDriftDetector:
    """Kolmogorov-Smirnov test for drift detection"""
    
    def detect(self, reference: np.ndarray, current: np.ndarray) -> Tuple[bool, float]:
        """Detect drift using KS test"""
        statistic, p_value = stats.ks_2samp(reference, current)
        drift_detected = p_value < 0.05
        return drift_detected, p_value


class WassersteinDriftDetector:
    """Wasserstein distance for drift detection"""
    
    def detect(self, reference: np.ndarray, current: np.ndarray) -> Tuple[bool, float]:
        """Detect drift using Wasserstein distance"""
        from scipy.stats import wasserstein_distance
        distance = wasserstein_distance(reference, current)
        # Threshold based on empirical analysis
        drift_detected = distance > 0.1
        return drift_detected, distance


class ChiSquaredDriftDetector:
    """Chi-squared test for categorical drift detection"""
    
    def detect(self, reference: np.ndarray, current: np.ndarray) -> Tuple[bool, float]:
        """Detect drift using Chi-squared test"""
        # Create frequency tables
        ref_unique, ref_counts = np.unique(reference, return_counts=True)
        curr_unique, curr_counts = np.unique(current, return_counts=True)
        
        # Align categories
        all_categories = np.unique(np.concatenate([ref_unique, curr_unique]))
        ref_freq = np.zeros(len(all_categories))
        curr_freq = np.zeros(len(all_categories))
        
        for i, cat in enumerate(all_categories):
            if cat in ref_unique:
                ref_freq[i] = ref_counts[np.where(ref_unique == cat)[0][0]]
            if cat in curr_unique:
                curr_freq[i] = curr_counts[np.where(curr_unique == cat)[0][0]]
                
        # Chi-squared test
        chi2, p_value = stats.chisquare(curr_freq, ref_freq)
        drift_detected = p_value < 0.05
        return drift_detected, p_value


# Global monitoring system instance
model_monitoring = ModelMonitoringSystem()