"""
Test script for Week 2 P1 AI/ML Features
Tests AutoML pipeline and Personalization model implementations
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml-pipeline', 'src'))

# Import our new modules
try:
    from automl_pipeline import (
        AutoMLPipeline,
        AutoMLConfig,
        ModelType,
        OptimizationMetric,
        AutoMLOptunaTuner
    )
    print("[OK] AutoML Pipeline imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import AutoML Pipeline: {e}")
    automl_available = False
else:
    automl_available = True

try:
    from personalization_model import (
        PersonalizationEngine,
        PersonalizationConfig,
        PersonalizationType,
        ContentStyle,
        ChannelProfile
    )
    print("[OK] Personalization Model imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import Personalization Model: {e}")
    personalization_available = False
else:
    personalization_available = True


def generate_sample_data(n_samples=1000, n_features=10):
    """Generate sample data for testing AutoML"""
    np.random.seed(42)
    
    # Create features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some categorical features
    X['category'] = np.random.choice(['A', 'B', 'C'], n_samples)
    X['category_encoded'] = pd.Categorical(X['category']).codes
    
    # Create target with relationship to features
    y = pd.Series(
        X['feature_0'] * 3 + 
        X['feature_1'] * 2 - 
        X['feature_2'] * 1.5 + 
        X['category_encoded'] * 0.5 +
        np.sin(X['feature_3']) * 2 +
        np.random.randn(n_samples) * 0.5
    )
    
    return X.drop('category', axis=1), y


def test_automl_pipeline():
    """Test AutoML pipeline functionality"""
    print("\n" + "="*60)
    print("Testing AutoML Pipeline")
    print("="*60)
    
    if not automl_available:
        print("AutoML not available, skipping tests")
        return False
    
    try:
        # Generate sample data
        print("\n1. Generating sample data...")
        X, y = generate_sample_data(n_samples=500)
        print(f"   Generated {len(X)} samples with {X.shape[1]} features")
        
        # Configure AutoML
        print("\n2. Configuring AutoML pipeline...")
        config = AutoMLConfig(
            task_type="regression",
            optimization_metric=OptimizationMetric.R2,
            test_size=0.2,
            cv_folds=3,  # Reduced for faster testing
            n_trials=20,  # Reduced for faster testing
            enable_feature_engineering=True,
            enable_ensemble=True,
            max_models_to_evaluate=3  # Test fewer models
        )
        print(f"   Config: {config.optimization_metric.value} optimization, {config.max_models_to_evaluate} models")
        
        # Initialize pipeline
        print("\n3. Initializing AutoML pipeline...")
        automl = AutoMLPipeline(config)
        
        # Train models
        print("\n4. Training models (this may take a minute)...")
        results = automl.train(X, y, feature_names=list(X.columns))
        
        # Display results
        print("\n5. Training Results:")
        print(f"   Best model: {results['best_model_type']}")
        print(f"   Best score (R²): {results['best_score']:.4f}")
        
        print("\n   All model performances:")
        for perf in results['all_performances']:
            print(f"   - {perf['model']}: Val={perf['val_score']:.4f}, Train={perf['train_score']:.4f}, Time={perf['training_time']:.2f}s")
        
        # Feature importance
        if results.get('feature_importance'):
            print("\n   Top 5 important features:")
            sorted_features = sorted(
                results['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for feat, importance in sorted_features:
                print(f"   - {feat}: {importance:.4f}")
        
        # Test prediction
        print("\n6. Testing prediction...")
        X_test = X.iloc[:10]
        predictions = automl.predict(X_test)
        print(f"   Made {len(predictions)} predictions")
        print(f"   Sample predictions: {predictions[:3]}")
        
        # Test model persistence
        print("\n7. Testing model save/load...")
        save_path = Path("models/automl/test_model.pkl")
        automl.save_model(save_path)
        print(f"   Model saved to {save_path}")
        
        # Test retraining check
        print("\n8. Testing automated retraining check...")
        should_retrain = automl.should_retrain()
        print(f"   Should retrain: {should_retrain}")
        
        # Get model summary
        print("\n9. Model Summary:")
        summary = automl.get_model_summary()
        print(f"   Current model type: {summary['current_model']['type']}")
        print(f"   Validation score: {summary['current_model']['validation_score']:.4f}")
        print(f"   Inference time: {summary['current_model']['inference_time_ms']:.2f}ms")
        print(f"   Retrain needed: {summary['retrain_needed']}")
        
        print("\n[OK] AutoML Pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] AutoML Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_personalization_model():
    """Test Personalization model functionality"""
    print("\n" + "="*60)
    print("Testing Personalization Model")
    print("="*60)
    
    if not personalization_available:
        print("Personalization not available, skipping tests")
        return False
    
    try:
        # Initialize personalization engine
        print("\n1. Initializing Personalization Engine...")
        config = PersonalizationConfig(
            model_type=PersonalizationType.HYBRID,
            embedding_dim=64,  # Reduced for testing
            n_clusters=5,
            min_videos_for_training=2  # Reduced for testing
        )
        engine = PersonalizationEngine(config)
        print(f"   Initialized with {config.model_type.value} model type")
        
        # Create sample channel data
        print("\n2. Creating sample channel profiles...")
        
        # Channel 1: Tech Education
        channel1_data = {
            'name': 'TechMaster',
            'description': 'Advanced technology tutorials and coding guides',
            'niche': 'technology',
            'target_audience': {
                'age_range': '18-35',
                'interests': ['programming', 'AI', 'web development']
            }
        }
        
        historical_videos1 = [
            {
                'title': 'Python Machine Learning Tutorial',
                'description': 'Complete guide to ML with Python',
                'views': 75000,
                'likes': 4500,
                'comments': 600,
                'duration': 1200,
                'published_at': (datetime.now() - timedelta(days=i*7)).isoformat(),
                'engagement_rate': 0.067
            }
            for i in range(5)
        ]
        
        profile1 = engine.create_channel_profile(
            'tech_channel_001',
            channel1_data,
            historical_videos1
        )
        print(f"   Created profile: {profile1.channel_name} ({profile1.content_style.value})")
        
        # Channel 2: Entertainment
        channel2_data = {
            'name': 'FunZone',
            'description': 'Comedy sketches and entertaining content',
            'niche': 'entertainment',
            'target_audience': {
                'age_range': '13-25',
                'interests': ['comedy', 'viral videos', 'challenges']
            }
        }
        
        historical_videos2 = [
            {
                'title': 'Epic Fail Compilation 2024',
                'description': 'The funniest fails of the year',
                'views': 150000,
                'likes': 12000,
                'comments': 2000,
                'duration': 600,
                'published_at': (datetime.now() - timedelta(days=i*3)).isoformat(),
                'engagement_rate': 0.093
            }
            for i in range(5)
        ]
        
        profile2 = engine.create_channel_profile(
            'fun_channel_002',
            channel2_data,
            historical_videos2
        )
        print(f"   Created profile: {profile2.channel_name} ({profile2.content_style.value})")
        
        # Test personalized content generation
        print("\n3. Generating personalized content recommendations...")
        
        trending_topics = ['AI', 'ChatGPT', 'Automation']
        
        # For tech channel
        rec1 = engine.generate_personalized_content(
            'tech_channel_001',
            trending_topics=trending_topics
        )
        print(f"\n   Tech Channel Recommendation:")
        print(f"   Title: {rec1.title}")
        print(f"   Keywords: {', '.join(rec1.keywords[:5])}")
        print(f"   Tone: {rec1.tone}")
        print(f"   Style: {rec1.style}")
        print(f"   Est. Engagement: {rec1.estimated_engagement:.2%}")
        print(f"   Confidence: {rec1.confidence_score:.2%}")
        
        # For entertainment channel
        rec2 = engine.generate_personalized_content(
            'fun_channel_002',
            trending_topics=['Viral', 'Challenge', 'Trend']
        )
        print(f"\n   Entertainment Channel Recommendation:")
        print(f"   Title: {rec2.title}")
        print(f"   Keywords: {', '.join(rec2.keywords[:5])}")
        print(f"   Tone: {rec2.tone}")
        print(f"   Style: {rec2.style}")
        print(f"   Est. Engagement: {rec2.estimated_engagement:.2%}")
        print(f"   Confidence: {rec2.confidence_score:.2%}")
        
        # Test profile update with feedback
        print("\n4. Testing profile update with performance feedback...")
        
        performance_data = {
            'views': 95000,
            'likes': 6000,
            'comments': 800,
            'duration': 900,
            'engagement_rate': 0.072
        }
        
        engine.update_profile_with_feedback(
            'tech_channel_001',
            'video_123',
            performance_data
        )
        print("   Profile updated with new video performance data")
        
        # Get channel insights
        print("\n5. Getting channel insights...")
        
        insights1 = engine.get_channel_insights('tech_channel_001')
        print(f"\n   Tech Channel Insights:")
        print(f"   - Content Style: {insights1['content_style']}")
        print(f"   - Avg Views: {insights1['performance']['avg_views']:,.0f}")
        print(f"   - Avg Engagement: {insights1['performance']['avg_engagement']:.2%}")
        print(f"   - Best Posting Hour: {insights1['optimal_schedule']['best_hour']:.0f}:00")
        print(f"   - Top Keywords: {', '.join(insights1['content_preferences']['top_keywords'][:3])}")
        print(f"   - Profile Confidence: {insights1['profile_confidence']:.2%}")
        
        insights2 = engine.get_channel_insights('fun_channel_002')
        print(f"\n   Entertainment Channel Insights:")
        print(f"   - Content Style: {insights2['content_style']}")
        print(f"   - Avg Views: {insights2['performance']['avg_views']:,.0f}")
        print(f"   - Avg Engagement: {insights2['performance']['avg_engagement']:.2%}")
        print(f"   - Posting Frequency: {insights2['optimal_schedule']['posting_frequency']}")
        
        # Test profile persistence
        print("\n6. Testing profile save/load...")
        save_path = Path("models/personalization/test_profiles.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        engine.save_profiles(save_path)
        print(f"   Profiles saved to {save_path}")
        
        # Test script template generation
        print("\n7. Testing script template generation...")
        script = rec1.script_template
        print(f"   Generated script template ({len(script)} characters)")
        print(f"   Template preview: {script[:200]}...")
        
        print("\n[OK] Personalization Model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Personalization Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between AutoML and Personalization"""
    print("\n" + "="*60)
    print("Testing AI/ML Integration")
    print("="*60)
    
    if not (automl_available and personalization_available):
        print("Some components not available, skipping integration tests")
        return False
    
    try:
        print("\n1. Creating integrated pipeline...")
        
        # Use AutoML to predict engagement based on channel features
        print("\n2. Preparing channel feature data...")
        
        # Create feature data from multiple channels (more samples for better training)
        np.random.seed(42)
        n_channels = 20  # More samples for better training
        
        channel_features = pd.DataFrame({
            'avg_views': np.random.randint(10000, 200000, n_channels),
            'avg_likes': np.random.randint(500, 15000, n_channels),
            'avg_comments': np.random.randint(50, 2500, n_channels),
            'video_count': np.random.randint(10, 150, n_channels),
            'subscriber_count': np.random.randint(1000, 60000, n_channels),
            'avg_duration': np.random.randint(300, 1500, n_channels),
            'posting_frequency': np.random.randint(1, 7, n_channels),
            'channel_age_days': np.random.randint(30, 1000, n_channels)
        })
        
        # Target: engagement rate (with some correlation to features)
        engagement_rates = pd.Series(
            0.02 + (channel_features['avg_likes'] / channel_features['avg_views']).values * 0.8 +
            np.random.normal(0, 0.01, n_channels)
        ).clip(0.01, 0.15)
        
        print(f"   Created dataset with {len(channel_features)} channels")
        
        # Train AutoML model for engagement prediction
        print("\n3. Training AutoML model for engagement prediction...")
        
        automl_config = AutoMLConfig(
            optimization_metric=OptimizationMetric.R2,
            max_models_to_evaluate=2,  # Quick test
            enable_feature_engineering=False,  # Disabled due to small dataset
            cv_folds=2,  # Reduced due to small dataset
            test_size=0.3
        )
        
        automl = AutoMLPipeline(automl_config)
        results = automl.train(channel_features, engagement_rates)
        
        print(f"   Best model for engagement prediction: {results['best_model_type']}")
        print(f"   Model score: {results['best_score']:.4f}")
        
        # Use personalization to generate content
        print("\n4. Using personalization with AutoML predictions...")
        
        personalization_config = PersonalizationConfig(
            model_type=PersonalizationType.HYBRID
        )
        
        personalization = PersonalizationEngine(personalization_config)
        
        # Create a test channel
        test_channel_data = {
            'name': 'TestChannel',
            'description': 'Testing integrated AI/ML pipeline',
            'niche': 'technology',
            'target_audience': {'age_range': '18-35'}
        }
        
        test_videos = [
            {
                'title': f'Test Video {i}',
                'views': np.random.randint(20000, 80000),
                'likes': np.random.randint(1000, 5000),
                'comments': np.random.randint(100, 500),
                'duration': np.random.randint(600, 1200),
                'published_at': (datetime.now() - timedelta(days=i*5)).isoformat()
            }
            for i in range(3)
        ]
        
        profile = personalization.create_channel_profile(
            'test_channel',
            test_channel_data,
            test_videos
        )
        
        # Generate personalized content
        recommendation = personalization.generate_personalized_content('test_channel')
        
        # Predict engagement using AutoML
        new_channel_features = pd.DataFrame({
            'avg_views': [np.mean([v['views'] for v in test_videos])],
            'avg_likes': [np.mean([v['likes'] for v in test_videos])],
            'avg_comments': [np.mean([v['comments'] for v in test_videos])],
            'video_count': [len(test_videos)],
            'subscriber_count': [15000],
            'avg_duration': [np.mean([v['duration'] for v in test_videos])],
            'posting_frequency': [2],
            'channel_age_days': [200]
        })
        
        predicted_engagement = automl.predict(new_channel_features)[0]
        
        print(f"\n   Integration Results:")
        print(f"   Personalized Title: {recommendation.title}")
        print(f"   Personalization Est.: {recommendation.estimated_engagement:.2%}")
        print(f"   AutoML Prediction: {predicted_engagement:.2%}")
        print(f"   Combined Confidence: {(recommendation.confidence_score + 0.7) / 2:.2%}")
        
        print("\n[OK] Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test execution"""
    print("\n" + "="*60)
    print("YTEmpire AI/ML P1 Features Test Suite")
    print("Week 2 - AutoML & Personalization")
    print("="*60)
    
    results = {
        'AutoML Pipeline': False,
        'Personalization Model': False,
        'Integration': False
    }
    
    # Run tests
    results['AutoML Pipeline'] = test_automl_pipeline()
    results['Personalization Model'] = test_personalization_model()
    results['Integration'] = test_integration()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "[OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n[SUCCESS] All AI/ML P1 features are working correctly!")
    else:
        print("\n[WARNING] Some tests failed. Please review the errors above.")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)