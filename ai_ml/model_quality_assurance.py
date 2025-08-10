"""
Model Quality Assurance System
Comprehensive testing and validation for ML models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, roc_auc_score, log_loss
)
from sklearn.model_selection import cross_val_score, KFold
import tensorflow as tf
import torch
import mlflow
import json
import logging
from datetime import datetime
import asyncio
import aiohttp
from prometheus_client import Histogram, Counter, Gauge

# Metrics
model_evaluation_time = Histogram('model_evaluation_duration', 'Time to evaluate model', ['model_name', 'metric'])
model_performance = Gauge('model_performance_score', 'Model performance score', ['model_name', 'metric'])
drift_detected = Counter('model_drift_detected', 'Number of drift detections', ['model_name', 'drift_type'])

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Container for model metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    perplexity: Optional[float] = None
    bleu_score: Optional[float] = None
    rouge_score: Optional[Dict] = None
    custom_metrics: Dict[str, float] = None

@dataclass
class QualityReport:
    """Quality assurance report"""
    model_name: str
    version: str
    timestamp: datetime
    metrics: ModelMetrics
    data_quality: Dict
    drift_analysis: Dict
    bias_analysis: Dict
    robustness_tests: Dict
    recommendation: str
    passed: bool

class ModelQualityAssurance:
    """Comprehensive model quality assurance system"""
    
    def __init__(self, mlflow_uri: str = None):
        self.mlflow_uri = mlflow_uri
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        
        self.quality_thresholds = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.78,
            'auc_roc': 0.85,
            'r2': 0.70,
            'perplexity': 100,  # Lower is better
            'bleu_score': 0.30,
            'data_drift': 0.15,
            'concept_drift': 0.10
        }
    
    async def evaluate_model(self, 
                            model: Any,
                            test_data: Tuple[np.ndarray, np.ndarray],
                            model_type: str = 'classification',
                            model_name: str = 'model',
                            version: str = '1.0') -> QualityReport:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model (sklearn, tensorflow, pytorch, etc.)
            test_data: Tuple of (X_test, y_test)
            model_type: Type of model ('classification', 'regression', 'nlp')
            model_name: Name of the model
            version: Model version
        """
        X_test, y_test = test_data
        
        # Evaluate model performance
        metrics = await self._evaluate_performance(model, X_test, y_test, model_type)
        
        # Check data quality
        data_quality = await self._check_data_quality(X_test, y_test)
        
        # Detect drift
        drift_analysis = await self._detect_drift(model, X_test, y_test)
        
        # Check for bias
        bias_analysis = await self._analyze_bias(model, X_test, y_test)
        
        # Test robustness
        robustness_tests = await self._test_robustness(model, X_test, y_test)
        
        # Generate recommendation
        passed, recommendation = self._generate_recommendation(
            metrics, data_quality, drift_analysis, bias_analysis, robustness_tests
        )
        
        # Create report
        report = QualityReport(
            model_name=model_name,
            version=version,
            timestamp=datetime.utcnow(),
            metrics=metrics,
            data_quality=data_quality,
            drift_analysis=drift_analysis,
            bias_analysis=bias_analysis,
            robustness_tests=robustness_tests,
            recommendation=recommendation,
            passed=passed
        )
        
        # Log to MLflow
        if self.mlflow_uri:
            self._log_to_mlflow(report)
        
        # Update Prometheus metrics
        self._update_metrics(report)
        
        return report
    
    async def _evaluate_performance(self, model: Any, X_test: np.ndarray, 
                                   y_test: np.ndarray, model_type: str) -> ModelMetrics:
        """Evaluate model performance metrics"""
        metrics = ModelMetrics()
        
        # Get predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        elif isinstance(model, tf.keras.Model):
            y_pred = model.predict(X_test)
            y_pred = np.argmax(y_pred, axis=1) if model_type == 'classification' else y_pred
        elif isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test)
                y_pred = model(X_tensor).numpy()
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        
        if model_type == 'classification':
            # Classification metrics
            metrics.accuracy = accuracy_score(y_test, y_pred)
            metrics.precision, metrics.recall, metrics.f1_score, _ = \
                precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            # AUC-ROC for binary classification
            if len(np.unique(y_test)) == 2:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    metrics.auc_roc = roc_auc_score(y_test, y_proba)
        
        elif model_type == 'regression':
            # Regression metrics
            metrics.mse = mean_squared_error(y_test, y_pred)
            metrics.mae = mean_absolute_error(y_test, y_pred)
            metrics.r2 = r2_score(y_test, y_pred)
        
        elif model_type == 'nlp':
            # NLP-specific metrics would be calculated here
            # Placeholder for perplexity, BLEU, ROUGE scores
            metrics.perplexity = await self._calculate_perplexity(model, X_test)
            metrics.bleu_score = await self._calculate_bleu(y_test, y_pred)
            metrics.rouge_score = await self._calculate_rouge(y_test, y_pred)
        
        return metrics
    
    async def _check_data_quality(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Check test data quality"""
        quality_checks = {
            'missing_values': np.isnan(X_test).sum(),
            'duplicates': len(X_test) - len(np.unique(X_test, axis=0)),
            'outliers': self._detect_outliers(X_test),
            'class_imbalance': self._check_class_imbalance(y_test),
            'feature_correlation': self._check_feature_correlation(X_test),
            'data_size': len(X_test),
            'feature_count': X_test.shape[1] if len(X_test.shape) > 1 else 1
        }
        
        quality_checks['quality_score'] = self._calculate_data_quality_score(quality_checks)
        
        return quality_checks
    
    async def _detect_drift(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Detect data and concept drift"""
        drift_results = {}
        
        # Data drift detection using KS test
        if hasattr(self, 'reference_data'):
            from scipy.stats import ks_2samp
            
            drift_scores = []
            for i in range(X_test.shape[1]):
                statistic, p_value = ks_2samp(self.reference_data[:, i], X_test[:, i])
                drift_scores.append(p_value)
            
            drift_results['data_drift'] = {
                'detected': any(p < 0.05 for p in drift_scores),
                'max_drift': 1 - min(drift_scores),
                'drifted_features': [i for i, p in enumerate(drift_scores) if p < 0.05]
            }
        
        # Concept drift detection using model performance degradation
        if hasattr(self, 'baseline_performance'):
            current_performance = model.score(X_test, y_test) if hasattr(model, 'score') else 0
            performance_drop = self.baseline_performance - current_performance
            
            drift_results['concept_drift'] = {
                'detected': performance_drop > self.quality_thresholds['concept_drift'],
                'performance_drop': performance_drop
            }
        
        return drift_results
    
    async def _analyze_bias(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Analyze model bias"""
        bias_analysis = {}
        
        # Prediction bias
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
            bias_analysis['prediction_bias'] = np.mean(y_pred) - np.mean(y_test)
        
        # Feature importance bias (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            bias_analysis['feature_importance_variance'] = np.var(importances)
            bias_analysis['dominant_features'] = np.where(importances > np.mean(importances) + 2*np.std(importances))[0].tolist()
        
        # Demographic parity (if demographic features available)
        # This would require demographic feature identification
        
        return bias_analysis
    
    async def _test_robustness(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Test model robustness"""
        robustness_results = {}
        
        # Test with noise injection
        noise_levels = [0.01, 0.05, 0.1]
        noise_performance = []
        
        for noise_level in noise_levels:
            X_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
            if hasattr(model, 'score'):
                score = model.score(X_noisy, y_test)
            else:
                y_pred = model.predict(X_noisy)
                score = accuracy_score(y_test, y_pred)
            noise_performance.append(score)
        
        robustness_results['noise_robustness'] = {
            'scores': noise_performance,
            'degradation': max(0, noise_performance[0] - noise_performance[-1])
        }
        
        # Test with adversarial examples (simplified)
        if hasattr(model, 'predict'):
            # Simple FGSM-like perturbation
            epsilon = 0.1
            X_adv = X_test + epsilon * np.sign(np.random.randn(*X_test.shape))
            y_pred_adv = model.predict(X_adv)
            
            robustness_results['adversarial_robustness'] = {
                'accuracy': accuracy_score(y_test, y_pred_adv) if hasattr(y_pred_adv, '__len__') else 0
            }
        
        # Cross-validation stability
        if hasattr(model, 'fit'):
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
            robustness_results['cv_stability'] = {
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'min_score': np.min(cv_scores),
                'max_score': np.max(cv_scores)
            }
        
        return robustness_results
    
    def _detect_outliers(self, X: np.ndarray) -> int:
        """Detect outliers using IQR method"""
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        outlier_count = 0
        for i in range(X.shape[1]):
            Q1 = np.percentile(X[:, i], 25)
            Q3 = np.percentile(X[:, i], 75)
            IQR = Q3 - Q1
            outliers = (X[:, i] < Q1 - 1.5 * IQR) | (X[:, i] > Q3 + 1.5 * IQR)
            outlier_count += np.sum(outliers)
        
        return outlier_count
    
    def _check_class_imbalance(self, y: np.ndarray) -> float:
        """Check for class imbalance"""
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            return 0.0
        
        imbalance_ratio = min(counts) / max(counts)
        return imbalance_ratio
    
    def _check_feature_correlation(self, X: np.ndarray) -> float:
        """Check feature correlation"""
        if len(X.shape) == 1 or X.shape[1] < 2:
            return 0.0
        
        correlation_matrix = np.corrcoef(X.T)
        # Get upper triangle without diagonal
        upper_triangle = np.triu(correlation_matrix, k=1)
        high_correlation = np.sum(np.abs(upper_triangle) > 0.9)
        
        return high_correlation
    
    def _calculate_data_quality_score(self, quality_checks: Dict) -> float:
        """Calculate overall data quality score"""
        score = 100.0
        
        # Deduct points for quality issues
        if quality_checks['missing_values'] > 0:
            score -= min(20, quality_checks['missing_values'] * 0.1)
        
        if quality_checks['duplicates'] > 0:
            score -= min(10, quality_checks['duplicates'] * 0.05)
        
        if quality_checks['outliers'] > quality_checks['data_size'] * 0.1:
            score -= 15
        
        if quality_checks['class_imbalance'] < 0.1:
            score -= 20
        
        if quality_checks['feature_correlation'] > 0:
            score -= min(10, quality_checks['feature_correlation'] * 2)
        
        return max(0, score)
    
    async def _calculate_perplexity(self, model: Any, X_test: np.ndarray) -> float:
        """Calculate perplexity for language models"""
        # Placeholder - actual implementation would depend on model type
        return 50.0
    
    async def _calculate_bleu(self, reference: np.ndarray, hypothesis: np.ndarray) -> float:
        """Calculate BLEU score for NLP models"""
        # Placeholder - would use actual BLEU implementation
        return 0.35
    
    async def _calculate_rouge(self, reference: np.ndarray, hypothesis: np.ndarray) -> Dict:
        """Calculate ROUGE scores for NLP models"""
        # Placeholder - would use actual ROUGE implementation
        return {
            'rouge-1': 0.40,
            'rouge-2': 0.25,
            'rouge-l': 0.35
        }
    
    def _generate_recommendation(self, metrics: ModelMetrics, data_quality: Dict,
                                drift_analysis: Dict, bias_analysis: Dict,
                                robustness_tests: Dict) -> Tuple[bool, str]:
        """Generate recommendation based on all analyses"""
        issues = []
        passed = True
        
        # Check metrics against thresholds
        if metrics.accuracy and metrics.accuracy < self.quality_thresholds['accuracy']:
            issues.append(f"Accuracy below threshold ({metrics.accuracy:.2f} < {self.quality_thresholds['accuracy']})")
            passed = False
        
        if metrics.precision and metrics.precision < self.quality_thresholds['precision']:
            issues.append(f"Precision below threshold ({metrics.precision:.2f} < {self.quality_thresholds['precision']})")
            passed = False
        
        # Check data quality
        if data_quality['quality_score'] < 70:
            issues.append(f"Poor data quality (score: {data_quality['quality_score']:.1f})")
            passed = False
        
        # Check drift
        if drift_analysis.get('data_drift', {}).get('detected'):
            issues.append("Data drift detected")
            passed = False
        
        if drift_analysis.get('concept_drift', {}).get('detected'):
            issues.append("Concept drift detected")
            passed = False
        
        # Check robustness
        if robustness_tests.get('noise_robustness', {}).get('degradation', 0) > 0.2:
            issues.append("Poor noise robustness")
            passed = False
        
        # Generate recommendation
        if passed:
            recommendation = "Model passed all quality checks and is ready for deployment."
        else:
            recommendation = f"Model failed quality checks. Issues: {'; '.join(issues)}"
        
        return passed, recommendation
    
    def _log_to_mlflow(self, report: QualityReport):
        """Log quality report to MLflow"""
        with mlflow.start_run(run_name=f"{report.model_name}_qa_{report.version}"):
            # Log metrics
            if report.metrics.accuracy:
                mlflow.log_metric("accuracy", report.metrics.accuracy)
            if report.metrics.precision:
                mlflow.log_metric("precision", report.metrics.precision)
            if report.metrics.recall:
                mlflow.log_metric("recall", report.metrics.recall)
            if report.metrics.f1_score:
                mlflow.log_metric("f1_score", report.metrics.f1_score)
            
            # Log data quality
            mlflow.log_metric("data_quality_score", report.data_quality['quality_score'])
            
            # Log drift status
            mlflow.log_param("data_drift_detected", 
                            report.drift_analysis.get('data_drift', {}).get('detected', False))
            
            # Log pass/fail
            mlflow.log_param("qa_passed", report.passed)
            mlflow.log_param("recommendation", report.recommendation)
            
            # Log full report as artifact
            report_dict = {
                'model_name': report.model_name,
                'version': report.version,
                'timestamp': report.timestamp.isoformat(),
                'metrics': report.metrics.__dict__,
                'data_quality': report.data_quality,
                'drift_analysis': report.drift_analysis,
                'bias_analysis': report.bias_analysis,
                'robustness_tests': report.robustness_tests,
                'recommendation': report.recommendation,
                'passed': report.passed
            }
            
            with open('/tmp/qa_report.json', 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            mlflow.log_artifact('/tmp/qa_report.json')
    
    def _update_metrics(self, report: QualityReport):
        """Update Prometheus metrics"""
        if report.metrics.accuracy:
            model_performance.labels(model_name=report.model_name, metric='accuracy').set(report.metrics.accuracy)
        if report.metrics.f1_score:
            model_performance.labels(model_name=report.model_name, metric='f1_score').set(report.metrics.f1_score)
        
        if report.drift_analysis.get('data_drift', {}).get('detected'):
            drift_detected.labels(model_name=report.model_name, drift_type='data').inc()
        if report.drift_analysis.get('concept_drift', {}).get('detected'):
            drift_detected.labels(model_name=report.model_name, drift_type='concept').inc()

class AutomatedQATesting:
    """Automated testing pipeline for continuous model quality assurance"""
    
    def __init__(self, qa_system: ModelQualityAssurance):
        self.qa_system = qa_system
        self.test_suite = []
        
    def add_test(self, test_name: str, test_func: callable, **kwargs):
        """Add a test to the suite"""
        self.test_suite.append({
            'name': test_name,
            'func': test_func,
            'kwargs': kwargs
        })
    
    async def run_test_suite(self, model: Any, test_data: Tuple) -> List[Dict]:
        """Run all tests in the suite"""
        results = []
        
        for test in self.test_suite:
            try:
                result = await test['func'](model, test_data, **test['kwargs'])
                results.append({
                    'test_name': test['name'],
                    'status': 'passed' if result else 'failed',
                    'result': result
                })
            except Exception as e:
                results.append({
                    'test_name': test['name'],
                    'status': 'error',
                    'error': str(e)
                })
                logger.error(f"Test {test['name']} failed: {e}")
        
        return results
    
    async def performance_regression_test(self, model: Any, test_data: Tuple,
                                         baseline_score: float, tolerance: float = 0.05) -> bool:
        """Test for performance regression"""
        X_test, y_test = test_data
        
        if hasattr(model, 'score'):
            current_score = model.score(X_test, y_test)
        else:
            y_pred = model.predict(X_test)
            current_score = accuracy_score(y_test, y_pred)
        
        return current_score >= baseline_score - tolerance
    
    async def inference_time_test(self, model: Any, test_data: Tuple,
                                 max_time_ms: float = 100) -> bool:
        """Test inference time"""
        import time
        
        X_test = test_data[0]
        sample = X_test[:100]  # Test with 100 samples
        
        start_time = time.time()
        _ = model.predict(sample)
        inference_time = (time.time() - start_time) * 1000 / len(sample)
        
        return inference_time <= max_time_ms
    
    async def memory_usage_test(self, model: Any, max_memory_mb: float = 500) -> bool:
        """Test model memory usage"""
        import sys
        
        model_size = sys.getsizeof(model) / (1024 * 1024)  # Convert to MB
        return model_size <= max_memory_mb

# Example usage
async def main():
    # Initialize QA system
    qa_system = ModelQualityAssurance(mlflow_uri="http://localhost:5000")
    
    # Create automated testing pipeline
    auto_tester = AutomatedQATesting(qa_system)
    
    # Add tests to suite
    auto_tester.add_test("regression_test", auto_tester.performance_regression_test, 
                        baseline_score=0.85, tolerance=0.03)
    auto_tester.add_test("inference_time", auto_tester.inference_time_test,
                        max_time_ms=50)
    auto_tester.add_test("memory_usage", auto_tester.memory_usage_test,
                        max_memory_mb=300)
    
    # Example: Test a model
    # model = load_model()
    # test_data = load_test_data()
    # report = await qa_system.evaluate_model(model, test_data, model_type='classification')
    # test_results = await auto_tester.run_test_suite(model, test_data)
    
    print("Model Quality Assurance system initialized")

if __name__ == "__main__":
    asyncio.run(main())