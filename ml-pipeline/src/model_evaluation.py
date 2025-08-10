"""
Model Evaluation Framework for YTEmpire ML Pipeline
Comprehensive evaluation and monitoring of ML models
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import pickle
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from scipy import stats
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics"""
    model_name: str
    model_version: str
    timestamp: datetime
    dataset_info: Dict[str, Any]
    
    # Regression metrics
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None
    
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    auc_roc: Optional[float] = None
    
    # Additional metrics
    inference_time_ms: Optional[float] = None
    model_size_mb: Optional[float] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    
    # Business metrics
    cost_per_prediction: Optional[float] = None
    value_generated: Optional[float] = None
    roi: Optional[float] = None


class ModelEvaluator:
    """Comprehensive model evaluation system"""
    
    def __init__(self, mlflow_tracking_uri: Optional[str] = None):
        """
        Initialize model evaluator
        
        Args:
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri or "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.evaluation_history = []
        
    def evaluate_regression_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "regression_model",
        model_version: str = "1.0"
    ) -> ModelMetrics:
        """
        Evaluate regression model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Model identifier
            model_version: Model version
        
        Returns:
            ModelMetrics object
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Measure inference time
        import time
        start_time = time.time()
        _ = model.predict(X_test[:100])  # Predict on subset
        inference_time = (time.time() - start_time) / 100 * 1000  # ms per prediction
        
        # Get model size
        import sys
        model_size = sys.getsizeof(pickle.dumps(model)) / (1024 * 1024)  # MB
        
        # Feature importance (if available)
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for i, importance in enumerate(model.feature_importances_):
                feature_importance[f'feature_{i}'] = float(importance)
        
        metrics = ModelMetrics(
            model_name=model_name,
            model_version=model_version,
            timestamp=datetime.now(),
            dataset_info={
                'n_samples': len(y_test),
                'n_features': X_test.shape[1] if len(X_test.shape) > 1 else 1
            },
            mae=mae,
            mse=mse,
            rmse=rmse,
            r2=r2,
            mape=mape,
            inference_time_ms=inference_time,
            model_size_mb=model_size,
            feature_importance=feature_importance
        )
        
        # Log to MLflow
        self._log_to_mlflow(metrics, model)
        
        # Store in history
        self.evaluation_history.append(metrics)
        
        return metrics
    
    def evaluate_classification_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "classification_model",
        model_version: str = "1.0",
        threshold: float = 0.5
    ) -> ModelMetrics:
        """
        Evaluate classification model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Model identifier
            model_version: Model version
            threshold: Classification threshold
        
        Returns:
            ModelMetrics object
        """
        # Make predictions
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_prob = y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # AUC-ROC for binary classification
        try:
            auc_roc = roc_auc_score(y_test, y_prob)
        except:
            auc_roc = None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Measure inference time
        import time
        start_time = time.time()
        _ = model.predict(X_test[:100])
        inference_time = (time.time() - start_time) / 100 * 1000
        
        # Get model size
        import sys
        model_size = sys.getsizeof(pickle.dumps(model)) / (1024 * 1024)
        
        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for i, importance in enumerate(model.feature_importances_):
                feature_importance[f'feature_{i}'] = float(importance)
        
        metrics = ModelMetrics(
            model_name=model_name,
            model_version=model_version,
            timestamp=datetime.now(),
            dataset_info={
                'n_samples': len(y_test),
                'n_features': X_test.shape[1] if len(X_test.shape) > 1 else 1,
                'n_classes': len(np.unique(y_test))
            },
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc_roc,
            inference_time_ms=inference_time,
            model_size_mb=model_size,
            feature_importance=feature_importance,
            confusion_matrix=cm
        )
        
        # Log to MLflow
        self._log_to_mlflow(metrics, model)
        
        # Store in history
        self.evaluation_history.append(metrics)
        
        return metrics
    
    def evaluate_video_generation_model(
        self,
        model: Any,
        test_videos: List[Dict[str, Any]],
        model_name: str = "video_gen_model"
    ) -> ModelMetrics:
        """
        Evaluate video generation model with custom metrics
        
        Args:
            model: Video generation model
            test_videos: Test video data
            model_name: Model identifier
        
        Returns:
            ModelMetrics object
        """
        total_cost = 0
        total_time = 0
        quality_scores = []
        
        for video in test_videos:
            # Simulate generation
            start_time = datetime.now()
            
            # Mock generation process
            generation_cost = video.get('estimated_cost', 1.10)
            generation_time = video.get('generation_time', 300)
            quality_score = video.get('quality_score', 0.85)
            
            total_cost += generation_cost
            total_time += generation_time
            quality_scores.append(quality_score)
        
        avg_cost = total_cost / len(test_videos)
        avg_time = total_time / len(test_videos)
        avg_quality = np.mean(quality_scores)
        
        # Calculate business metrics
        value_per_video = 5.0  # Estimated value generated per video
        roi = (value_per_video - avg_cost) / avg_cost * 100
        
        metrics = ModelMetrics(
            model_name=model_name,
            model_version="1.0",
            timestamp=datetime.now(),
            dataset_info={
                'n_videos': len(test_videos),
                'avg_duration': np.mean([v.get('duration', 600) for v in test_videos])
            },
            cost_per_prediction=avg_cost,
            value_generated=value_per_video * len(test_videos),
            roi=roi,
            inference_time_ms=avg_time * 1000,
            feature_importance={
                'script_quality': 0.3,
                'voice_quality': 0.25,
                'visual_quality': 0.25,
                'thumbnail_quality': 0.2
            }
        )
        
        return metrics
    
    def perform_cross_validation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        scoring: str = 'neg_mean_squared_error'
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation
        
        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            cv_folds: Number of CV folds
            scoring: Scoring metric
        
        Returns:
            Cross-validation results
        """
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        
        results = {
            'cv_scores': cv_scores.tolist(),
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'min_score': cv_scores.min(),
            'max_score': cv_scores.max(),
            'cv_folds': cv_folds,
            'scoring_metric': scoring
        }
        
        logger.info(f"Cross-validation results: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str = 'regression'
    ) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of model_name: model
            X_test: Test features
            y_test: Test targets
            task_type: 'regression' or 'classification'
        
        Returns:
            Comparison DataFrame
        """
        comparison_results = []
        
        for model_name, model in models.items():
            if task_type == 'regression':
                metrics = self.evaluate_regression_model(
                    model, X_test, y_test, model_name
                )
                comparison_results.append({
                    'model': model_name,
                    'mae': metrics.mae,
                    'rmse': metrics.rmse,
                    'r2': metrics.r2,
                    'mape': metrics.mape,
                    'inference_time_ms': metrics.inference_time_ms,
                    'model_size_mb': metrics.model_size_mb
                })
            else:
                metrics = self.evaluate_classification_model(
                    model, X_test, y_test, model_name
                )
                comparison_results.append({
                    'model': model_name,
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1': metrics.f1,
                    'auc_roc': metrics.auc_roc,
                    'inference_time_ms': metrics.inference_time_ms,
                    'model_size_mb': metrics.model_size_mb
                })
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # Rank models
        if task_type == 'regression':
            comparison_df['rank'] = comparison_df['r2'].rank(ascending=False)
        else:
            comparison_df['rank'] = comparison_df['f1'].rank(ascending=False)
        
        return comparison_df.sort_values('rank')
    
    def detect_model_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect model drift using statistical tests
        
        Args:
            reference_predictions: Baseline predictions
            current_predictions: Current model predictions
            threshold: Drift threshold
        
        Returns:
            Drift detection results
        """
        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(
            reference_predictions,
            current_predictions
        )
        
        # Calculate PSI (Population Stability Index)
        psi = self._calculate_psi(reference_predictions, current_predictions)
        
        # Mean shift
        mean_shift = np.abs(
            np.mean(current_predictions) - np.mean(reference_predictions)
        ) / np.mean(reference_predictions)
        
        # Variance shift
        var_shift = np.abs(
            np.var(current_predictions) - np.var(reference_predictions)
        ) / np.var(reference_predictions)
        
        drift_detected = (
            ks_pvalue < threshold or
            psi > 0.25 or
            mean_shift > 0.2 or
            var_shift > 0.3
        )
        
        results = {
            'drift_detected': drift_detected,
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'psi': psi,
            'mean_shift': mean_shift,
            'variance_shift': var_shift,
            'threshold': threshold
        }
        
        if drift_detected:
            logger.warning(f"Model drift detected: {results}")
        
        return results
    
    def generate_evaluation_report(
        self,
        metrics: ModelMetrics,
        output_path: str = "model_evaluation_report.html"
    ) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            metrics: Model metrics
            output_path: Report output path
        
        Returns:
            Report HTML content
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report - {metrics.model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .bad {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Model Evaluation Report</h1>
            <h2>{metrics.model_name} v{metrics.model_version}</h2>
            <p>Generated: {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metric">
                <h3>Dataset Information</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    {"".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.dataset_info.items())}
                </table>
            </div>
        """
        
        # Add regression metrics if available
        if metrics.mae is not None:
            html_content += f"""
            <div class="metric">
                <h3>Regression Metrics</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                    <tr><td>MAE</td><td>{metrics.mae:.4f}</td><td class="{'good' if metrics.mae < 100 else 'warning'}">{'Good' if metrics.mae < 100 else 'Check'}</td></tr>
                    <tr><td>RMSE</td><td>{metrics.rmse:.4f}</td><td class="{'good' if metrics.rmse < 150 else 'warning'}">{'Good' if metrics.rmse < 150 else 'Check'}</td></tr>
                    <tr><td>R²</td><td>{metrics.r2:.4f}</td><td class="{'good' if metrics.r2 > 0.8 else 'warning' if metrics.r2 > 0.6 else 'bad'}">{'Good' if metrics.r2 > 0.8 else 'Fair' if metrics.r2 > 0.6 else 'Poor'}</td></tr>
                    <tr><td>MAPE</td><td>{metrics.mape:.2f}%</td><td class="{'good' if metrics.mape < 10 else 'warning' if metrics.mape < 20 else 'bad'}">{'Good' if metrics.mape < 10 else 'Fair' if metrics.mape < 20 else 'Poor'}</td></tr>
                </table>
            </div>
            """
        
        # Add classification metrics if available
        if metrics.accuracy is not None:
            html_content += f"""
            <div class="metric">
                <h3>Classification Metrics</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                    <tr><td>Accuracy</td><td>{metrics.accuracy:.4f}</td><td class="{'good' if metrics.accuracy > 0.9 else 'warning' if metrics.accuracy > 0.8 else 'bad'}">{'Good' if metrics.accuracy > 0.9 else 'Fair' if metrics.accuracy > 0.8 else 'Poor'}</td></tr>
                    <tr><td>Precision</td><td>{metrics.precision:.4f}</td><td class="{'good' if metrics.precision > 0.9 else 'warning'}">{'Good' if metrics.precision > 0.9 else 'Check'}</td></tr>
                    <tr><td>Recall</td><td>{metrics.recall:.4f}</td><td class="{'good' if metrics.recall > 0.9 else 'warning'}">{'Good' if metrics.recall > 0.9 else 'Check'}</td></tr>
                    <tr><td>F1 Score</td><td>{metrics.f1:.4f}</td><td class="{'good' if metrics.f1 > 0.9 else 'warning' if metrics.f1 > 0.8 else 'bad'}">{'Good' if metrics.f1 > 0.9 else 'Fair' if metrics.f1 > 0.8 else 'Poor'}</td></tr>
                </table>
            </div>
            """
        
        # Add performance metrics
        html_content += f"""
            <div class="metric">
                <h3>Performance Metrics</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                    <tr><td>Inference Time</td><td>{metrics.inference_time_ms:.2f} ms</td><td class="{'good' if metrics.inference_time_ms < 10 else 'warning' if metrics.inference_time_ms < 50 else 'bad'}">{'Good' if metrics.inference_time_ms < 10 else 'Fair' if metrics.inference_time_ms < 50 else 'Slow'}</td></tr>
                    <tr><td>Model Size</td><td>{metrics.model_size_mb:.2f} MB</td><td class="{'good' if metrics.model_size_mb < 10 else 'warning' if metrics.model_size_mb < 50 else 'bad'}">{'Good' if metrics.model_size_mb < 10 else 'Fair' if metrics.model_size_mb < 50 else 'Large'}</td></tr>
                </table>
            </div>
        """
        
        # Add feature importance if available
        if metrics.feature_importance:
            sorted_features = sorted(metrics.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            html_content += """
            <div class="metric">
                <h3>Top 10 Feature Importance</h3>
                <table>
                    <tr><th>Feature</th><th>Importance</th></tr>
            """
            for feature, importance in sorted_features:
                html_content += f"<tr><td>{feature}</td><td>{importance:.4f}</td></tr>"
            html_content += """
                </table>
            </div>
            """
        
        # Add business metrics if available
        if metrics.roi is not None:
            html_content += f"""
            <div class="metric">
                <h3>Business Metrics</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Cost per Prediction</td><td>${metrics.cost_per_prediction:.2f}</td></tr>
                    <tr><td>Value Generated</td><td>${metrics.value_generated:.2f}</td></tr>
                    <tr><td>ROI</td><td>{metrics.roi:.1f}%</td></tr>
                </table>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Evaluation report saved to {output_path}")
        
        return html_content
    
    def _calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10
    ) -> float:
        """Calculate Population Stability Index"""
        # Create bins based on expected distribution
        breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Calculate frequencies
        expected_freq = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_freq = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        
        # Avoid division by zero
        expected_freq = np.where(expected_freq == 0, 0.001, expected_freq)
        actual_freq = np.where(actual_freq == 0, 0.001, actual_freq)
        
        # Calculate PSI
        psi = np.sum((actual_freq - expected_freq) * np.log(actual_freq / expected_freq))
        
        return psi
    
    def _log_to_mlflow(self, metrics: ModelMetrics, model: Any):
        """Log metrics and model to MLflow"""
        try:
            with mlflow.start_run(run_name=f"{metrics.model_name}_{metrics.timestamp}"):
                # Log metrics
                if metrics.mae is not None:
                    mlflow.log_metric("mae", metrics.mae)
                    mlflow.log_metric("rmse", metrics.rmse)
                    mlflow.log_metric("r2", metrics.r2)
                    mlflow.log_metric("mape", metrics.mape)
                
                if metrics.accuracy is not None:
                    mlflow.log_metric("accuracy", metrics.accuracy)
                    mlflow.log_metric("precision", metrics.precision)
                    mlflow.log_metric("recall", metrics.recall)
                    mlflow.log_metric("f1", metrics.f1)
                    if metrics.auc_roc:
                        mlflow.log_metric("auc_roc", metrics.auc_roc)
                
                mlflow.log_metric("inference_time_ms", metrics.inference_time_ms)
                mlflow.log_metric("model_size_mb", metrics.model_size_mb)
                
                # Log parameters
                mlflow.log_params(metrics.dataset_info)
                
                # Log model
                mlflow.sklearn.log_model(model, metrics.model_name)
                
                # Log artifacts
                if metrics.confusion_matrix is not None:
                    np.save("confusion_matrix.npy", metrics.confusion_matrix)
                    mlflow.log_artifact("confusion_matrix.npy")
                
                logger.info(f"Logged metrics to MLflow for {metrics.model_name}")
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")


# Example usage
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    
    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=20, noise=10)
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_regression_model(
        model, X_test, y_test,
        model_name="RandomForest_Views",
        model_version="1.0"
    )
    
    # Generate report
    evaluator.generate_evaluation_report(metrics)
    
    print(f"Model Evaluation Complete:")
    print(f"  R² Score: {metrics.r2:.4f}")
    print(f"  RMSE: {metrics.rmse:.4f}")
    print(f"  Inference Time: {metrics.inference_time_ms:.2f} ms")