"""
Data Transformation Pipeline for YTEmpire
Handles data preprocessing and transformation for ML models
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, 
    chi2, f_classif, mutual_info_classif,
    f_regression, mutual_info_regression
)
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)


class DataValidator(BaseEstimator, TransformerMixin):
    """Validate and clean input data"""
    
    def __init__(self, 
                 drop_duplicates: bool = True,
                 drop_high_null: float = 0.95,
                 drop_single_value: bool = True):
        self.drop_duplicates = drop_duplicates
        self.drop_high_null = drop_high_null
        self.drop_single_value = drop_single_value
        self.columns_to_drop = []
        
    def fit(self, X, y=None):
        """Identify columns to drop"""
        self.columns_to_drop = []
        
        # Check for high null percentage
        if self.drop_high_null:
            null_pct = X.isnull().sum() / len(X)
            high_null_cols = null_pct[null_pct > self.drop_high_null].index.tolist()
            self.columns_to_drop.extend(high_null_cols)
            
        # Check for single value columns
        if self.drop_single_value:
            single_value_cols = [col for col in X.columns if X[col].nunique() <= 1]
            self.columns_to_drop.extend(single_value_cols)
            
        self.columns_to_drop = list(set(self.columns_to_drop))
        return self
    
    def transform(self, X):
        """Clean and validate data"""
        X_clean = X.copy()
        
        # Drop duplicate rows
        if self.drop_duplicates:
            X_clean = X_clean.drop_duplicates()
            
        # Drop identified columns
        X_clean = X_clean.drop(columns=self.columns_to_drop, errors='ignore')
        
        return X_clean


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Handle outliers in numerical features"""
    
    def __init__(self, 
                 method: str = 'iqr',
                 threshold: float = 1.5,
                 strategy: str = 'clip'):
        self.method = method  # 'iqr', 'zscore', 'isolation'
        self.threshold = threshold
        self.strategy = strategy  # 'clip', 'remove', 'transform'
        self.bounds = {}
        
    def fit(self, X, y=None):
        """Calculate outlier bounds"""
        self.bounds = {}
        
        for col in X.select_dtypes(include=[np.number]).columns:
            if self.method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.threshold * IQR
                upper = Q3 + self.threshold * IQR
            elif self.method == 'zscore':
                mean = X[col].mean()
                std = X[col].std()
                lower = mean - self.threshold * std
                upper = mean + self.threshold * std
            else:
                lower = X[col].quantile(0.01)
                upper = X[col].quantile(0.99)
                
            self.bounds[col] = {'lower': lower, 'upper': upper}
            
        return self
    
    def transform(self, X):
        """Handle outliers based on strategy"""
        X_transformed = X.copy()
        
        for col, bounds in self.bounds.items():
            if col in X_transformed.columns:
                if self.strategy == 'clip':
                    X_transformed[col] = X_transformed[col].clip(
                        lower=bounds['lower'],
                        upper=bounds['upper']
                    )
                elif self.strategy == 'remove':
                    mask = (X_transformed[col] >= bounds['lower']) & \
                           (X_transformed[col] <= bounds['upper'])
                    X_transformed = X_transformed[mask]
                elif self.strategy == 'transform':
                    # Log transform for positive skewed data
                    if X_transformed[col].min() > 0:
                        X_transformed[col] = np.log1p(X_transformed[col])
                        
        return X_transformed


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Create engineered features"""
    
    def __init__(self, 
                 create_polynomials: bool = True,
                 create_interactions: bool = True,
                 create_ratios: bool = True):
        self.create_polynomials = create_polynomials
        self.create_interactions = create_interactions
        self.create_ratios = create_ratios
        self.feature_names = []
        
    def fit(self, X, y=None):
        """Identify features to create"""
        self.feature_names = X.columns.tolist()
        return self
    
    def transform(self, X):
        """Create engineered features"""
        X_eng = X.copy()
        numeric_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()
        
        # Polynomial features for key metrics
        if self.create_polynomials and len(numeric_cols) > 0:
            for col in numeric_cols[:5]:  # Limit to top 5 to avoid explosion
                if col in X_eng.columns:
                    X_eng[f'{col}_squared'] = X_eng[col] ** 2
                    X_eng[f'{col}_cubed'] = X_eng[col] ** 3
                    X_eng[f'{col}_sqrt'] = np.sqrt(np.abs(X_eng[col]))
                    X_eng[f'{col}_log'] = np.log1p(np.abs(X_eng[col]))
                    
        # Interaction features
        if self.create_interactions and len(numeric_cols) > 1:
            # Key interactions
            interactions = [
                ('view_count', 'like_count'),
                ('view_count', 'comment_count'),
                ('subscriber_count', 'view_count'),
                ('cost', 'revenue'),
                ('duration', 'view_count')
            ]
            
            for col1, col2 in interactions:
                if col1 in X_eng.columns and col2 in X_eng.columns:
                    X_eng[f'{col1}_x_{col2}'] = X_eng[col1] * X_eng[col2]
                    
        # Ratio features
        if self.create_ratios and len(numeric_cols) > 1:
            ratios = [
                ('like_count', 'view_count', 'like_rate'),
                ('comment_count', 'view_count', 'comment_rate'),
                ('revenue', 'cost', 'roi'),
                ('view_count', 'subscriber_count', 'view_per_sub'),
                ('like_count', 'dislike_count', 'like_ratio')
            ]
            
            for num, denom, name in ratios:
                if num in X_eng.columns and denom in X_eng.columns:
                    X_eng[name] = X_eng[num] / (X_eng[denom] + 1e-10)
                    
        return X_eng


class DataTransformationPipeline:
    """Complete data transformation pipeline"""
    
    def __init__(self,
                 scaling_method: str = 'standard',
                 encoding_method: str = 'onehot',
                 imputation_strategy: str = 'mean',
                 feature_selection_method: Optional[str] = None,
                 dimensionality_reduction: Optional[str] = None,
                 n_components: int = 50):
        
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        self.imputation_strategy = imputation_strategy
        self.feature_selection_method = feature_selection_method
        self.dimensionality_reduction = dimensionality_reduction
        self.n_components = n_components
        
        self.numeric_pipeline = None
        self.categorical_pipeline = None
        self.full_pipeline = None
        self.feature_names = []
        self.numeric_features = []
        self.categorical_features = []
        
    def create_numeric_pipeline(self) -> Pipeline:
        """Create pipeline for numeric features"""
        steps = []
        
        # Imputation
        if self.imputation_strategy == 'knn':
            steps.append(('imputer', KNNImputer(n_neighbors=5)))
        else:
            steps.append(('imputer', SimpleImputer(strategy=self.imputation_strategy)))
            
        # Outlier handling
        steps.append(('outlier_handler', OutlierHandler()))
        
        # Scaling
        if self.scaling_method == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif self.scaling_method == 'minmax':
            steps.append(('scaler', MinMaxScaler()))
        elif self.scaling_method == 'robust':
            steps.append(('scaler', RobustScaler()))
            
        return Pipeline(steps)
    
    def create_categorical_pipeline(self) -> Pipeline:
        """Create pipeline for categorical features"""
        steps = []
        
        # Imputation
        steps.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))
        
        # Encoding
        if self.encoding_method == 'onehot':
            steps.append(('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')))
        elif self.encoding_method == 'label':
            steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
            
        return Pipeline(steps)
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit transformation pipeline"""
        # Data validation
        validator = DataValidator()
        X_clean = validator.fit_transform(X)
        
        # Identify feature types
        self.numeric_features = X_clean.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create pipelines
        transformers = []
        
        if self.numeric_features:
            self.numeric_pipeline = self.create_numeric_pipeline()
            transformers.append(('numeric', self.numeric_pipeline, self.numeric_features))
            
        if self.categorical_features:
            self.categorical_pipeline = self.create_categorical_pipeline()
            transformers.append(('categorical', self.categorical_pipeline, self.categorical_features))
            
        # Combine pipelines
        from sklearn.compose import ColumnTransformer
        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
        
        # Add feature engineering
        steps = [
            ('validator', validator),
            ('preprocessor', preprocessor),
            ('feature_engineering', FeatureEngineering())
        ]
        
        # Add feature selection
        if self.feature_selection_method:
            if self.feature_selection_method == 'kbest':
                selector = SelectKBest(k=min(self.n_components, X_clean.shape[1]))
            elif self.feature_selection_method == 'percentile':
                selector = SelectPercentile(percentile=50)
            else:
                selector = None
                
            if selector:
                steps.append(('feature_selection', selector))
                
        # Add dimensionality reduction
        if self.dimensionality_reduction:
            if self.dimensionality_reduction == 'pca':
                reducer = PCA(n_components=min(self.n_components, X_clean.shape[1]))
            elif self.dimensionality_reduction == 'svd':
                reducer = TruncatedSVD(n_components=min(self.n_components, X_clean.shape[1] - 1))
            elif self.dimensionality_reduction == 'nmf':
                reducer = NMF(n_components=min(self.n_components, X_clean.shape[1]))
            else:
                reducer = None
                
            if reducer:
                steps.append(('dim_reduction', reducer))
                
        self.full_pipeline = Pipeline(steps)
        self.full_pipeline.fit(X, y)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data"""
        if self.full_pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        return self.full_pipeline.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform data"""
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get transformed feature names"""
        # This is complex with sklearn pipelines
        # Return original features for now
        return self.numeric_features + self.categorical_features
    
    def save(self, filepath: str):
        """Save pipeline to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, filepath: str) -> 'DataTransformationPipeline':
        """Load pipeline from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline"""
        info = {
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'total_features': len(self.numeric_features) + len(self.categorical_features),
            'scaling_method': self.scaling_method,
            'encoding_method': self.encoding_method,
            'imputation_strategy': self.imputation_strategy,
            'feature_selection': self.feature_selection_method,
            'dimensionality_reduction': self.dimensionality_reduction,
            'n_components': self.n_components
        }
        
        if self.full_pipeline:
            info['pipeline_steps'] = [step[0] for step in self.full_pipeline.steps]
            
        return info


# Example usage
if __name__ == "__main__":
    # Create sample data
    import pandas as pd
    
    data = pd.DataFrame({
        'view_count': [1000, 2000, 1500, 3000, 2500],
        'like_count': [50, 120, 80, 200, 150],
        'comment_count': [10, 25, 15, 40, 30],
        'subscriber_count': [1000, 1500, 1200, 2000, 1800],
        'cost': [2.5, 3.0, 2.8, 3.5, 3.2],
        'revenue': [10, 15, 12, 20, 18],
        'category': ['tech', 'gaming', 'tech', 'gaming', 'food'],
        'duration': [600, 900, 750, 1200, 1000]
    })
    
    # Create and fit pipeline
    pipeline = DataTransformationPipeline(
        scaling_method='standard',
        encoding_method='onehot',
        imputation_strategy='mean',
        feature_selection_method=None,
        dimensionality_reduction='pca',
        n_components=3
    )
    
    transformed_data = pipeline.fit_transform(data)
    print(f"Original shape: {data.shape}")
    print(f"Transformed shape: {transformed_data.shape}")
    print(f"Pipeline info: {pipeline.get_pipeline_info()}")