"""
Model Evaluation Framework
Owner: AI/ML Team Lead

Comprehensive model evaluation, A/B testing, and performance tracking system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    model_name: str
    model_version: str
    metrics: Dict[str, float]
    timestamp: datetime
    data_size: int
    metadata: Dict[str, Any] = None


@dataclass
class ABTestResult:
    """Container for A/B test results."""
    test_id: str
    model_a: str
    model_b: str
    metric_name: str
    model_a_performance: float
    model_b_performance: float
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    statistical_significance: bool
    winner: Optional[str] = None


class ModelEvaluator:
    """Comprehensive model evaluation and comparison system."""
    
    def __init__(self, metrics_storage_path: str = None):
        self.metrics_storage_path = metrics_storage_path or f"{settings.MODEL_STORAGE_PATH}/metrics"
        self.evaluation_history = []
        
    def evaluate_regression_model(self, 
                                  y_true: np.ndarray, 
                                  y_pred: np.ndarray,
                                  model_name: str,
                                  model_version: str = "1.0") -> ModelMetrics:
        """Evaluate regression model performance."""
        try:
            # Calculate standard regression metrics
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # Calculate Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100
            
            # Calculate custom metrics
            residuals = y_true - y_pred
            
            # Residual statistics
            residual_mean = np.mean(residuals)
            residual_std = np.std(residuals)
            
            # Percentage of predictions within acceptable range (e.g., 20%)
            acceptable_range = 0.2
            within_range = np.mean(np.abs(residuals / np.clip(y_true, 1e-8, None)) <= acceptable_range)
            
            # Median Absolute Error (more robust to outliers)
            median_ae = np.median(np.abs(residuals))
            
            metrics = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2,
                'mape': mape,
                'residual_mean': residual_mean,
                'residual_std': residual_std,
                'predictions_within_20pct': within_range,
                'median_absolute_error': median_ae,
                'max_error': np.max(np.abs(residuals)),
                'min_error': np.min(np.abs(residuals))
            }
            
            model_metrics = ModelMetrics(
                model_name=model_name,
                model_version=model_version,
                metrics=metrics,
                timestamp=datetime.now(),
                data_size=len(y_true)
            )
            
            self.evaluation_history.append(model_metrics)
            
            logger.info(f"Evaluated {model_name} v{model_version}: MAE={mae:.4f}, R²={r2:.4f}")
            return model_metrics
            
        except Exception as e:
            logger.error(f"Regression evaluation failed: {str(e)}")
            raise
    
    def evaluate_classification_model(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      y_pred_proba: Optional[np.ndarray] = None,
                                      model_name: str = "classifier",
                                      model_version: str = "1.0") -> ModelMetrics:
        """Evaluate classification model performance."""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.metrics import confusion_matrix, classification_report
            from sklearn.metrics import roc_auc_score, log_loss
            
            # Basic classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            # Additional metrics if probabilities are provided
            if y_pred_proba is not None:
                if y_pred_proba.shape[1] == 2:  # Binary classification
                    auc_roc = roc_auc_score(y_true, y_pred_proba[:, 1])
                    metrics['auc_roc'] = auc_roc
                
                # Log loss
                logloss = log_loss(y_true, y_pred_proba)
                metrics['log_loss'] = logloss
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            model_metrics = ModelMetrics(
                model_name=model_name,
                model_version=model_version,
                metrics=metrics,
                timestamp=datetime.now(),
                data_size=len(y_true)
            )
            
            self.evaluation_history.append(model_metrics)
            
            logger.info(f"Evaluated {model_name} v{model_version}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
            return model_metrics
            
        except Exception as e:
            logger.error(f"Classification evaluation failed: {str(e)}")
            raise
    
    def compare_models(self, 
                       metrics_list: List[ModelMetrics], 
                       primary_metric: str = 'mae') -> Dict[str, Any]:
        """Compare multiple models and rank them."""
        try:
            if not metrics_list:
                return {}
            
            # Extract comparison data
            comparison_data = []
            for metrics in metrics_list:
                if primary_metric in metrics.metrics:
                    comparison_data.append({
                        'model_name': metrics.model_name,
                        'model_version': metrics.model_version,
                        'primary_metric': metrics.metrics[primary_metric],
                        'timestamp': metrics.timestamp,
                        'data_size': metrics.data_size,
                        'all_metrics': metrics.metrics
                    })
            
            if not comparison_data:
                return {}
            
            # Sort by primary metric (lower is better for error metrics)
            is_error_metric = primary_metric in ['mae', 'mse', 'rmse', 'mape', 'log_loss']
            comparison_data.sort(
                key=lambda x: x['primary_metric'], 
                reverse=not is_error_metric
            )
            
            # Calculate rankings and improvements
            best_score = comparison_data[0]['primary_metric']
            for i, model_data in enumerate(comparison_data):
                model_data['rank'] = i + 1
                
                if is_error_metric:
                    # For error metrics, calculate percentage increase from best
                    improvement = (model_data['primary_metric'] - best_score) / best_score * 100
                else:
                    # For performance metrics, calculate percentage improvement
                    improvement = (model_data['primary_metric'] - best_score) / best_score * 100
                
                model_data['improvement_pct'] = improvement
            
            return {
                'primary_metric': primary_metric,
                'rankings': comparison_data,
                'best_model': comparison_data[0],
                'total_models': len(comparison_data),
                'comparison_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model comparison failed: {str(e)}")
            return {}
    
    def run_ab_test(self,
                    model_a_predictions: np.ndarray,
                    model_b_predictions: np.ndarray,
                    ground_truth: np.ndarray,
                    test_id: str,
                    model_a_name: str = "Model A",
                    model_b_name: str = "Model B",
                    metric_name: str = "mae",
                    alpha: float = 0.05) -> ABTestResult:
        """Run A/B test between two models."""
        try:
            # Calculate metric for both models
            if metric_name == "mae":
                model_a_metric = mean_absolute_error(ground_truth, model_a_predictions)
                model_b_metric = mean_absolute_error(ground_truth, model_b_predictions)
                is_lower_better = True
            elif metric_name == "rmse":
                model_a_metric = np.sqrt(mean_squared_error(ground_truth, model_a_predictions))
                model_b_metric = np.sqrt(mean_squared_error(ground_truth, model_b_predictions))
                is_lower_better = True
            elif metric_name == "r2":
                model_a_metric = r2_score(ground_truth, model_a_predictions)
                model_b_metric = r2_score(ground_truth, model_b_predictions)
                is_lower_better = False
            else:
                raise ValueError(f"Unsupported metric: {metric_name}")
            
            # Calculate residuals for statistical test
            model_a_residuals = np.abs(ground_truth - model_a_predictions)
            model_b_residuals = np.abs(ground_truth - model_b_predictions)
            
            # Perform statistical test (paired t-test)
            t_stat, p_value = stats.ttest_rel(model_a_residuals, model_b_residuals)
            
            # Calculate confidence interval for the difference
            diff = model_a_residuals - model_b_residuals
            diff_mean = np.mean(diff)
            diff_std = np.std(diff, ddof=1)
            n = len(diff)
            
            # 95% confidence interval
            ci_margin = stats.t.ppf(1 - alpha/2, n-1) * (diff_std / np.sqrt(n))
            confidence_interval = (diff_mean - ci_margin, diff_mean + ci_margin)
            
            # Determine statistical significance and winner
            is_significant = p_value < alpha
            
            winner = None
            if is_significant:
                if is_lower_better:
                    winner = model_a_name if model_a_metric < model_b_metric else model_b_name
                else:
                    winner = model_a_name if model_a_metric > model_b_metric else model_b_name
            
            result = ABTestResult(
                test_id=test_id,
                model_a=model_a_name,
                model_b=model_b_name,
                metric_name=metric_name,
                model_a_performance=model_a_metric,
                model_b_performance=model_b_metric,
                p_value=p_value,
                confidence_interval=confidence_interval,
                sample_size=n,
                statistical_significance=is_significant,
                winner=winner
            )
            
            logger.info(f"A/B test {test_id}: {model_a_name}={model_a_metric:.4f}, {model_b_name}={model_b_metric:.4f}, p={p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"A/B test failed: {str(e)}")
            raise
    
    def track_model_drift(self,
                          reference_predictions: np.ndarray,
                          current_predictions: np.ndarray,
                          ground_truth: np.ndarray,
                          drift_threshold: float = 0.1) -> Dict[str, Any]:
        """Detect model drift by comparing current vs reference performance."""
        try:
            # Calculate performance metrics for both
            ref_mae = mean_absolute_error(ground_truth, reference_predictions)
            curr_mae = mean_absolute_error(ground_truth, current_predictions)
            
            ref_rmse = np.sqrt(mean_squared_error(ground_truth, reference_predictions))
            curr_rmse = np.sqrt(mean_squared_error(ground_truth, current_predictions))
            
            ref_r2 = r2_score(ground_truth, reference_predictions)
            curr_r2 = r2_score(ground_truth, current_predictions)
            
            # Calculate percentage changes
            mae_change = (curr_mae - ref_mae) / ref_mae * 100
            rmse_change = (curr_rmse - ref_rmse) / ref_rmse * 100
            r2_change = (curr_r2 - ref_r2) / abs(ref_r2) * 100 if ref_r2 != 0 else 0
            
            # Detect drift
            mae_drift = abs(mae_change) > (drift_threshold * 100)
            rmse_drift = abs(rmse_change) > (drift_threshold * 100)
            r2_drift = abs(r2_change) > (drift_threshold * 100)
            
            drift_detected = mae_drift or rmse_drift or r2_drift
            
            # Statistical test for drift
            ref_residuals = np.abs(ground_truth - reference_predictions)
            curr_residuals = np.abs(ground_truth - current_predictions)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(ref_residuals, curr_residuals, alternative='two-sided')
            
            drift_result = {
                'drift_detected': drift_detected,
                'drift_threshold': drift_threshold,
                'metrics_comparison': {
                    'mae': {'reference': ref_mae, 'current': curr_mae, 'change_pct': mae_change, 'drift': mae_drift},
                    'rmse': {'reference': ref_rmse, 'current': curr_rmse, 'change_pct': rmse_change, 'drift': rmse_drift},
                    'r2': {'reference': ref_r2, 'current': curr_r2, 'change_pct': r2_change, 'drift': r2_drift}
                },
                'statistical_test': {
                    'test_type': 'Mann-Whitney U',
                    'u_statistic': u_stat,
                    'p_value': u_p_value,
                    'significant_drift': u_p_value < 0.05
                },
                'recommendation': self._get_drift_recommendation(drift_detected, mae_change, rmse_change),
                'timestamp': datetime.now().isoformat()
            }
            
            if drift_detected:
                logger.warning(f"Model drift detected: MAE change {mae_change:.2f}%, RMSE change {rmse_change:.2f}%")
            else:
                logger.info("No significant model drift detected")
            
            return drift_result
            
        except Exception as e:
            logger.error(f"Drift detection failed: {str(e)}")
            raise
    
    def _get_drift_recommendation(self, drift_detected: bool, mae_change: float, rmse_change: float) -> str:
        """Get recommendation based on drift analysis."""
        if not drift_detected:
            return "No action needed. Model performance is stable."
        
        if mae_change > 10 or rmse_change > 10:
            return "High drift detected. Consider retraining the model immediately."
        elif mae_change > 5 or rmse_change > 5:
            return "Moderate drift detected. Schedule model retraining soon."
        else:
            return "Minor drift detected. Monitor closely and retrain if trend continues."
    
    def generate_model_report(self, model_metrics: ModelMetrics) -> Dict[str, Any]:
        """Generate comprehensive model evaluation report."""
        try:
            report = {
                'model_info': {
                    'name': model_metrics.model_name,
                    'version': model_metrics.model_version,
                    'evaluation_date': model_metrics.timestamp.isoformat(),
                    'data_size': model_metrics.data_size
                },
                'performance_metrics': model_metrics.metrics,
                'performance_summary': self._get_performance_summary(model_metrics.metrics),
                'recommendations': self._get_model_recommendations(model_metrics.metrics),
                'quality_grade': self._calculate_quality_grade(model_metrics.metrics),
                'production_readiness': self._assess_production_readiness(model_metrics.metrics)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return {}
    
    def _get_performance_summary(self, metrics: Dict[str, float]) -> str:
        """Generate human-readable performance summary."""
        if 'mae' in metrics:
            mae = metrics['mae']
            r2 = metrics.get('r2_score', 0)
            
            if mae < 0.1 and r2 > 0.9:
                return "Excellent performance with high accuracy and strong predictive power."
            elif mae < 0.2 and r2 > 0.8:
                return "Good performance with acceptable accuracy for most use cases."
            elif mae < 0.5 and r2 > 0.6:
                return "Fair performance with moderate accuracy. Consider improvements."
            else:
                return "Poor performance with low accuracy. Requires significant improvements."
        
        return "Performance summary not available."
    
    def _get_model_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        
        if 'mae' in metrics:
            mae = metrics['mae']
            if mae > 0.3:
                recommendations.append("High prediction error. Consider feature engineering or model complexity increase.")
        
        if 'r2_score' in metrics:
            r2 = metrics['r2_score']
            if r2 < 0.7:
                recommendations.append("Low R² score indicates poor fit. Try different algorithms or more data.")
        
        if 'predictions_within_20pct' in metrics:
            within_range = metrics['predictions_within_20pct']
            if within_range < 0.8:
                recommendations.append("Less than 80% predictions within acceptable range. Improve model robustness.")
        
        if not recommendations:
            recommendations.append("Model performance is satisfactory. Monitor for drift over time.")
        
        return recommendations
    
    def _calculate_quality_grade(self, metrics: Dict[str, float]) -> str:
        """Calculate overall quality grade A-F."""
        if 'mae' in metrics and 'r2_score' in metrics:
            mae = metrics['mae']
            r2 = metrics['r2_score']
            
            # Simple grading system
            if mae < 0.1 and r2 > 0.95:
                return 'A'
            elif mae < 0.2 and r2 > 0.85:
                return 'B'
            elif mae < 0.3 and r2 > 0.70:
                return 'C'
            elif mae < 0.5 and r2 > 0.50:
                return 'D'
            else:
                return 'F'
        
        return 'N/A'
    
    def _assess_production_readiness(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess if model is ready for production deployment."""
        readiness_checks = {
            'accuracy_acceptable': False,
            'stability_acceptable': False,
            'performance_acceptable': False,
            'overall_ready': False
        }
        
        if 'mae' in metrics:
            readiness_checks['accuracy_acceptable'] = metrics['mae'] < 0.25
        
        if 'predictions_within_20pct' in metrics:
            readiness_checks['stability_acceptable'] = metrics['predictions_within_20pct'] > 0.75
        
        if 'r2_score' in metrics:
            readiness_checks['performance_acceptable'] = metrics['r2_score'] > 0.7
        
        readiness_checks['overall_ready'] = all([
            readiness_checks['accuracy_acceptable'],
            readiness_checks['stability_acceptable'],
            readiness_checks['performance_acceptable']
        ])
        
        return readiness_checks
    
    def save_metrics(self, model_metrics: ModelMetrics, filepath: str = None):
        """Save model metrics to disk."""
        try:
            import os
            import pickle
            
            if filepath is None:
                os.makedirs(self.metrics_storage_path, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"{self.metrics_storage_path}/{model_metrics.model_name}_{timestamp}.pkl"
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_metrics, f)
            
            logger.info(f"Model metrics saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    y_true = np.random.normal(100, 20, 1000)
    y_pred_a = y_true + np.random.normal(0, 5, 1000)  # Better model
    y_pred_b = y_true + np.random.normal(0, 10, 1000)  # Worse model
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate models
    metrics_a = evaluator.evaluate_regression_model(y_true, y_pred_a, "TrendPredictor", "1.0")
    metrics_b = evaluator.evaluate_regression_model(y_true, y_pred_b, "TrendPredictor", "0.9")
    
    # Compare models
    comparison = evaluator.compare_models([metrics_a, metrics_b], 'mae')
    print("Model Comparison:", json.dumps(comparison, indent=2, default=str))
    
    # Run A/B test
    ab_result = evaluator.run_ab_test(y_pred_a, y_pred_b, y_true, "test_001")
    print(f"A/B Test Winner: {ab_result.winner}")
    
    # Generate report
    report = evaluator.generate_model_report(metrics_a)
    print("Model Report:", json.dumps(report, indent=2, default=str))