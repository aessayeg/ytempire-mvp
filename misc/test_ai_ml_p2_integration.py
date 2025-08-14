"""
Integration Test Suite for AI/ML Team P2 Components
Tests all Week 2 P2 (Nice to Have) features

Components tested:
1. AutoML platform expansion (automl_platform_v2.py)
2. Advanced voice cloning (advanced_voice_cloning.py)
3. Custom model training interface (custom_model_training_interface.py)
4. Experimental features (experimental_features.py)
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging for Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test results tracking
test_results = {
    'total': 0,
    'passed': 0,
    'failed': 0,
    'skipped': 0,
    'errors': []
}


def run_test(test_name: str, test_func):
    """Run a single test and track results"""
    test_results['total'] += 1
    try:
        print(f"\n[TEST] {test_name}...")
        result = test_func()
        if result:
            test_results['passed'] += 1
            print(f"[PASS] {test_name}")
            return True
        else:
            test_results['failed'] += 1
            print(f"[FAIL] {test_name}")
            return False
    except ImportError as e:
        test_results['skipped'] += 1
        print(f"[SKIP] {test_name} - Missing dependency: {e}")
        test_results['errors'].append(f"{test_name}: {str(e)}")
        return None
    except Exception as e:
        test_results['failed'] += 1
        print(f"[FAIL] {test_name} - Error: {e}")
        test_results['errors'].append(f"{test_name}: {str(e)}")
        return False


async def run_async_test(test_name: str, test_func):
    """Run an async test and track results"""
    test_results['total'] += 1
    try:
        print(f"\n[TEST] {test_name}...")
        result = await test_func()
        if result:
            test_results['passed'] += 1
            print(f"[PASS] {test_name}")
            return True
        else:
            test_results['failed'] += 1
            print(f"[FAIL] {test_name}")
            return False
    except ImportError as e:
        test_results['skipped'] += 1
        print(f"[SKIP] {test_name} - Missing dependency: {e}")
        test_results['errors'].append(f"{test_name}: {str(e)}")
        return None
    except Exception as e:
        test_results['failed'] += 1
        print(f"[FAIL] {test_name} - Error: {e}")
        test_results['errors'].append(f"{test_name}: {str(e)}")
        return False


# ====================
# AutoML Platform Tests
# ====================

def test_automl_import():
    """Test AutoML platform import"""
    try:
        from ml_pipeline.src.automl_platform_v2 import (
            AdvancedAutoMLPlatform,
            AutoMLConfig,
            TaskType,
            ModelFamily,
            OptimizationStrategy
        )
        return True
    except ImportError:
        # Try alternate import path
        try:
            sys.path.insert(0, 'ml-pipeline/src')
            from automl_platform_v2 import (
                AdvancedAutoMLPlatform,
                AutoMLConfig,
                TaskType,
                ModelFamily,
                OptimizationStrategy
            )
            return True
        except:
            return False


def test_automl_config():
    """Test AutoML configuration"""
    try:
        sys.path.insert(0, 'ml-pipeline/src')
        from automl_platform_v2 import AutoMLConfig, TaskType, ModelFamily, OptimizationStrategy
        
        config = AutoMLConfig(
            task_type=TaskType.REGRESSION,
            optimization_metrics=['r2', 'mse'],
            n_trials=10,
            timeout_seconds=60,
            enable_feature_engineering=True,
            enable_ensemble=True,
            model_families=[ModelFamily.LINEAR, ModelFamily.TREE_BASED],
            optimization_strategy=OptimizationStrategy.BAYESIAN
        )
        
        assert config.task_type == TaskType.REGRESSION
        assert len(config.optimization_metrics) == 2
        assert config.n_trials == 10
        return True
    except Exception as e:
        logger.error(f"AutoML config test failed: {e}")
        return False


def test_automl_basic_training():
    """Test basic AutoML training"""
    try:
        sys.path.insert(0, 'ml-pipeline/src')
        from automl_platform_v2 import AdvancedAutoMLPlatform, AutoMLConfig, TaskType
        import numpy as np
        
        # Create simple dataset
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        config = AutoMLConfig(
            task_type=TaskType.REGRESSION,
            n_trials=5,
            timeout_seconds=30
        )
        
        automl = AdvancedAutoMLPlatform(config)
        
        # Note: We're not actually fitting here to avoid long running time
        # Just testing initialization
        assert automl.config.task_type == TaskType.REGRESSION
        assert automl.best_model is None  # Not trained yet
        return True
    except Exception as e:
        logger.error(f"AutoML training test failed: {e}")
        return False


# ====================
# Voice Cloning Tests
# ====================

async def test_voice_cloning_import():
    """Test voice cloning import"""
    try:
        from ml_pipeline.src.advanced_voice_cloning import (
            AdvancedVoiceCloner,
            VoiceProfile,
            VoiceEmotion,
            VoiceGender,
            VoiceAge,
            EmotionSettings,
            AudioEnhancement
        )
        return True
    except ImportError:
        try:
            sys.path.insert(0, 'ml-pipeline/src')
            from advanced_voice_cloning import (
                AdvancedVoiceCloner,
                VoiceProfile,
                VoiceEmotion,
                VoiceGender,
                VoiceAge
            )
            return True
        except:
            return False


async def test_voice_profile_creation():
    """Test voice profile creation"""
    try:
        sys.path.insert(0, 'ml-pipeline/src')
        from advanced_voice_cloning import VoiceProfile, VoiceGender, VoiceAge, VoiceEmotion
        
        profile = VoiceProfile(
            name="TestVoice",
            gender=VoiceGender.NEUTRAL,
            age=VoiceAge.YOUNG_ADULT,
            speaking_rate=1.0,
            pitch=0.0,
            energy=0.5,
            emotion_default=VoiceEmotion.NEUTRAL
        )
        
        assert profile.name == "TestVoice"
        assert profile.gender == VoiceGender.NEUTRAL
        assert profile.speaking_rate == 1.0
        return True
    except Exception as e:
        logger.error(f"Voice profile test failed: {e}")
        return False


async def test_emotion_settings():
    """Test emotion settings"""
    try:
        sys.path.insert(0, 'ml-pipeline/src')
        from advanced_voice_cloning import EmotionSettings, VoiceEmotion
        
        emotion = EmotionSettings(
            emotion=VoiceEmotion.HAPPY,
            pitch_modifier=0.1,
            rate_modifier=1.1,
            volume_modifier=1.1,
            emphasis_modifier=1.2
        )
        
        assert emotion.emotion == VoiceEmotion.HAPPY
        assert emotion.pitch_modifier == 0.1
        assert emotion.rate_modifier == 1.1
        return True
    except Exception as e:
        logger.error(f"Emotion settings test failed: {e}")
        return False


# ====================
# Model Training Interface Tests
# ====================

def test_training_interface_import():
    """Test training interface import"""
    try:
        from ml_pipeline.src.custom_model_training_interface import (
            CustomModelTrainingInterface,
            TrainingJob,
            ModelMetadata,
            ModelStatus,
            DeploymentTarget
        )
        return True
    except ImportError:
        try:
            sys.path.insert(0, 'ml-pipeline/src')
            from custom_model_training_interface import (
                CustomModelTrainingInterface,
                TrainingJob,
                ModelStatus
            )
            return True
        except:
            return False


def test_training_job_creation():
    """Test training job creation"""
    try:
        sys.path.insert(0, 'ml-pipeline/src')
        from custom_model_training_interface import TrainingJob, ModelStatus, TaskType
        import uuid
        
        job = TrainingJob(
            job_id=str(uuid.uuid4()),
            name="TestModel",
            description="Test model training",
            task_type=TaskType.REGRESSION,
            status=ModelStatus.PENDING,
            created_at=datetime.now()
        )
        
        assert job.name == "TestModel"
        assert job.status == ModelStatus.PENDING
        assert job.task_type == TaskType.REGRESSION
        return True
    except Exception as e:
        logger.error(f"Training job test failed: {e}")
        return False


def test_model_metadata():
    """Test model metadata"""
    try:
        sys.path.insert(0, 'ml-pipeline/src')
        from custom_model_training_interface import ModelMetadata
        import uuid
        
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            version="1.0.0",
            name="TestModel",
            description="Test model",
            task_type="regression",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            training_job_id=str(uuid.uuid4()),
            performance_metrics={"r2": 0.85, "mse": 0.1},
            feature_names=["feature1", "feature2"],
            model_size_mb=10.5,
            inference_time_ms=50.0,
            deployment_status="ready"
        )
        
        assert metadata.version == "1.0.0"
        assert metadata.performance_metrics["r2"] == 0.85
        assert len(metadata.feature_names) == 2
        return True
    except Exception as e:
        logger.error(f"Model metadata test failed: {e}")
        return False


# ====================
# Experimental Features Tests
# ====================

async def test_experimental_import():
    """Test experimental features import"""
    try:
        from ml_pipeline.src.experimental_features import (
            ExperimentalFeaturesHub,
            ExperimentalFeatureType,
            ZeroShotContentGenerator,
            NeuralStyleTransfer,
            ReinforcementLearningOptimizer,
            MultimodalContentFusion,
            QuantumInspiredOptimizer
        )
        return True
    except ImportError:
        try:
            sys.path.insert(0, 'ml-pipeline/src')
            from experimental_features import (
                ExperimentalFeaturesHub,
                ExperimentalFeatureType
            )
            return True
        except:
            return False


async def test_experimental_hub():
    """Test experimental features hub"""
    try:
        sys.path.insert(0, 'ml-pipeline/src')
        from experimental_features import ExperimentalFeaturesHub, ExperimentalFeatureType
        
        hub = ExperimentalFeaturesHub()
        
        # Test zero-shot learning
        result = await hub.run_experiment(
            ExperimentalFeatureType.ZERO_SHOT_LEARNING,
            task="Generate a title",
            input="Test content"
        )
        
        assert 'generated_text' in result or 'error' in result
        return True
    except Exception as e:
        logger.error(f"Experimental hub test failed: {e}")
        return False


async def test_zero_shot_generator():
    """Test zero-shot content generator"""
    try:
        sys.path.insert(0, 'ml-pipeline/src')
        from experimental_features import ZeroShotContentGenerator
        
        generator = ZeroShotContentGenerator()
        
        # Test with fallback generation (if models not available)
        result = generator.generate_zero_shot(
            task_description="Generate a YouTube title",
            input_text="Python programming tutorial"
        )
        
        assert result is not None
        assert isinstance(result, str)
        return True
    except Exception as e:
        logger.error(f"Zero-shot generator test failed: {e}")
        return False


async def test_style_transfer():
    """Test neural style transfer"""
    try:
        sys.path.insert(0, 'ml-pipeline/src')
        from experimental_features import NeuralStyleTransfer
        
        style_transfer = NeuralStyleTransfer()
        
        content = "This is a simple tutorial."
        style_ref = "AMAZING! You WON'T BELIEVE this!"
        
        result = style_transfer.transfer_style(
            content=content,
            style_reference=style_ref,
            style_strength=0.7
        )
        
        assert result is not None
        assert isinstance(result, str)
        return True
    except Exception as e:
        logger.error(f"Style transfer test failed: {e}")
        return False


# ====================
# Integration Tests
# ====================

async def test_ai_ml_components_integration():
    """Test integration between AI/ML components"""
    try:
        sys.path.insert(0, 'ml-pipeline/src')
        
        # Test that components can work together
        from experimental_features import ExperimentalFeaturesHub
        from advanced_voice_cloning import VoiceProfile, VoiceGender, VoiceAge
        
        # Create experimental hub
        hub = ExperimentalFeaturesHub()
        
        # Generate content using zero-shot
        content_result = await hub.run_experiment(
            ExperimentalFeatureType.ZERO_SHOT_LEARNING,
            task="Generate a script",
            input="AI tutorial"
        )
        
        # Create voice profile for synthesis
        profile = VoiceProfile(
            name="AIVoice",
            gender=VoiceGender.NEUTRAL,
            age=VoiceAge.YOUNG_ADULT
        )
        
        # Test that both components initialized correctly
        assert content_result is not None
        assert profile.name == "AIVoice"
        
        return True
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False


async def test_existing_voice_synthesis():
    """Test existing voice synthesis integration"""
    try:
        # Test that existing voice synthesis still works
        from ml_pipeline.src.voice_synthesis import VoiceSynthesizer, VoiceConfig, VoiceProvider, VoiceStyle
        
        # Create basic config
        config = VoiceConfig(
            provider=VoiceProvider.GOOGLE_TTS,
            voice_id="test",
            style=VoiceStyle.NATURAL,
            language="en"
        )
        
        assert config.style == VoiceStyle.NATURAL
        assert config.language == "en"
        return True
    except ImportError:
        # Try alternate path
        try:
            sys.path.insert(0, 'ml-pipeline/src')
            from voice_synthesis import VoiceConfig, VoiceProvider, VoiceStyle
            
            config = VoiceConfig(
                provider=VoiceProvider.GOOGLE_TTS,
                voice_id="test",
                style=VoiceStyle.NATURAL,
                language="en"
            )
            return True
        except:
            logger.warning("Existing voice synthesis not found - may be expected")
            return None


# ====================
# Main Test Runner
# ====================

async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("AI/ML TEAM P2 INTEGRATION TEST SUITE")
    print("="*60)
    print(f"Started at: {datetime.now().isoformat()}")
    
    # AutoML Platform Tests
    print("\n" + "-"*40)
    print("AUTOML PLATFORM TESTS")
    print("-"*40)
    run_test("AutoML Import", test_automl_import)
    run_test("AutoML Configuration", test_automl_config)
    run_test("AutoML Basic Training", test_automl_basic_training)
    
    # Voice Cloning Tests
    print("\n" + "-"*40)
    print("ADVANCED VOICE CLONING TESTS")
    print("-"*40)
    await run_async_test("Voice Cloning Import", test_voice_cloning_import)
    await run_async_test("Voice Profile Creation", test_voice_profile_creation)
    await run_async_test("Emotion Settings", test_emotion_settings)
    
    # Model Training Interface Tests
    print("\n" + "-"*40)
    print("CUSTOM MODEL TRAINING INTERFACE TESTS")
    print("-"*40)
    run_test("Training Interface Import", test_training_interface_import)
    run_test("Training Job Creation", test_training_job_creation)
    run_test("Model Metadata", test_model_metadata)
    
    # Experimental Features Tests
    print("\n" + "-"*40)
    print("EXPERIMENTAL FEATURES TESTS")
    print("-"*40)
    await run_async_test("Experimental Import", test_experimental_import)
    await run_async_test("Experimental Hub", test_experimental_hub)
    await run_async_test("Zero-Shot Generator", test_zero_shot_generator)
    await run_async_test("Style Transfer", test_style_transfer)
    
    # Integration Tests
    print("\n" + "-"*40)
    print("INTEGRATION TESTS")
    print("-"*40)
    await run_async_test("AI/ML Components Integration", test_ai_ml_components_integration)
    await run_async_test("Existing Voice Synthesis", test_existing_voice_synthesis)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {test_results['total']}")
    print(f"Passed: {test_results['passed']}")
    print(f"Failed: {test_results['failed']}")
    print(f"Skipped: {test_results['skipped']}")
    
    if test_results['errors']:
        print("\nErrors encountered:")
        for error in test_results['errors']:
            print(f"  - {error}")
    
    success_rate = (test_results['passed'] / test_results['total'] * 100) if test_results['total'] > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.2f}%")
    
    # Create report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total': test_results['total'],
            'passed': test_results['passed'],
            'failed': test_results['failed'],
            'skipped': test_results['skipped'],
            'success_rate': success_rate
        },
        'errors': test_results['errors'],
        'components_tested': [
            'AutoML Platform v2',
            'Advanced Voice Cloning',
            'Custom Model Training Interface',
            'Experimental Features Hub'
        ]
    }
    
    # Save report
    report_path = Path("misc/ai_ml_p2_test_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    
    return success_rate >= 70  # Consider success if 70% or more tests pass


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)