"""
Fix AI/ML P2 components to handle missing dependencies gracefully
This will update the components to work without all optional dependencies
"""

import os
import re
from pathlib import Path


def fix_experimental_features():
    """Fix experimental_features.py to handle gym import properly"""
    file_path = Path("ml-pipeline/src/experimental_features.py")
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the ContentOptimizationEnv class to handle missing gym
    old_class = """class ContentOptimizationEnv(gym.Env):
    \"\"\"Custom Gym environment for content optimization\"\"\"
    
    def __init__(self):
        super().__init__()
        
        # Action space: content parameters
        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(5,),  # 5 content parameters
            dtype=np.float32
        )
        
        # Observation space: current metrics
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(10,),  # 10 metric values
            dtype=np.float32
        )"""
    
    new_class = """class ContentOptimizationEnv:
    \"\"\"Custom environment for content optimization (gym-compatible if available)\"\"\"
    
    def __init__(self):
        # Simplified environment without gym dependency
        if RL_AVAILABLE:
            super().__init__()
            
            # Action space: content parameters
            self.action_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(5,),  # 5 content parameters
                dtype=np.float32
            )
            
            # Observation space: current metrics
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(10,),  # 10 metric values
                dtype=np.float32
            )
        else:
            # Fallback without gym
            self.action_space = None
            self.observation_space = None"""
    
    content = content.replace(old_class, new_class)
    
    # Add conditional inheritance
    content = content.replace(
        "class ContentOptimizationEnv:",
        "class ContentOptimizationEnv(gym.Env if RL_AVAILABLE else object):"
    )
    
    # Fix the _initialize_rl method
    old_init = """def _initialize_rl(self):
        \"\"\"Initialize RL environment and model\"\"\"
        if RL_AVAILABLE:
            try:
                # Create custom environment for content optimization
                self.env = ContentOptimizationEnv()
                self.model = PPO("MlpPolicy", self.env, verbose=0)
                logger.info("RL optimizer initialized")
            except:
                logger.warning("Could not initialize RL optimizer")"""
    
    new_init = """def _initialize_rl(self):
        \"\"\"Initialize RL environment and model\"\"\"
        if RL_AVAILABLE:
            try:
                # Create custom environment for content optimization
                self.env = ContentOptimizationEnv()
                self.model = PPO("MlpPolicy", self.env, verbose=0)
                logger.info("RL optimizer initialized")
            except Exception as e:
                logger.warning(f"Could not initialize RL optimizer: {e}")
                self.env = None
                self.model = None
        else:
            self.env = None
            self.model = None"""
    
    content = content.replace(old_init, new_init)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed: {file_path}")
    return True


def fix_advanced_voice_cloning():
    """Fix advanced_voice_cloning.py to handle missing librosa"""
    file_path = Path("ml-pipeline/src/advanced_voice_cloning.py")
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add try-except for librosa import
    old_import = """import librosa
import soundfile as sf"""
    
    new_import = """try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    logging.warning("Audio processing libraries not available. Install with: pip install librosa soundfile")"""
    
    content = content.replace(old_import, new_import)
    
    # Fix analyze_voice_sample to handle missing librosa
    old_analyze = """def analyze_voice_sample(self, audio_path: str) -> Dict[str, Any]:
        \"\"\"Analyze voice characteristics from audio sample\"\"\"
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)"""
    
    new_analyze = """def analyze_voice_sample(self, audio_path: str) -> Dict[str, Any]:
        \"\"\"Analyze voice characteristics from audio sample\"\"\"
        if not AUDIO_LIBS_AVAILABLE:
            logger.warning("Audio libraries not available, returning default analysis")
            return {
                'duration': 10.0,
                'pitch': {'mean': 150, 'std': 20, 'min': 100, 'max': 200},
                'energy': {'mean': 0.5, 'std': 0.1, 'min': 0.2, 'max': 0.8},
                'speaking_rate': 150,
                'gender': 'neutral',
                'age_group': 'young_adult'
            }
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)"""
    
    content = content.replace(old_analyze, new_analyze)
    
    # Fix all librosa references to check availability first
    content = re.sub(
        r'(\s+)(librosa\.\w+)',
        r'\1if AUDIO_LIBS_AVAILABLE:\n\1    \2\n\1else:\n\1    pass  # Fallback',
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed: {file_path}")
    return True


def main():
    """Fix all AI/ML P2 components"""
    print("Fixing AI/ML P2 components for missing dependencies...")
    
    fixes = [
        fix_experimental_features(),
        fix_advanced_voice_cloning()
    ]
    
    if all(fixes):
        print("\nAll fixes applied successfully!")
        return True
    else:
        print("\nSome fixes failed. Check the output above.")
        return False


if __name__ == "__main__":
    success = main()