"""
GPU Environment Configuration
Owner: ML Engineer
"""

import torch
import tensorflow as tf
import logging
import subprocess
import platform
from typing import Dict, Any, List
import GPUtil
import numpy as np

logger = logging.getLogger(__name__)


class GPUEnvironment:
    """
    GPU Environment setup and validation
    For Ryzen 9 9950X3D with RTX 5090
    """
    
    def __init__(self):
        self.gpu_available = False
        self.cuda_version = None
        self.gpu_info = {}
        self.benchmark_results = {}
        
    def check_cuda_installation(self) -> bool:
        """Check if CUDA is properly installed"""
        try:
            # Check CUDA compiler
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, 
                                  text=True)
            if result.returncode == 0:
                output = result.stdout
                # Extract CUDA version
                for line in output.split('\n'):
                    if 'release' in line:
                        self.cuda_version = line.split('release')[-1].strip().split(',')[0]
                        logger.info(f"CUDA version detected: {self.cuda_version}")
                        return True
        except FileNotFoundError:
            logger.error("CUDA compiler (nvcc) not found")
        except Exception as e:
            logger.error(f"Error checking CUDA: {e}")
        
        return False
    
    def setup_pytorch_gpu(self) -> bool:
        """Configure PyTorch for GPU usage"""
        try:
            if torch.cuda.is_available():
                self.gpu_available = True
                device_count = torch.cuda.device_count()
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    self.gpu_info[f"gpu_{i}"] = {
                        "name": props.name,
                        "memory_total": props.total_memory / 1024**3,  # GB
                        "memory_allocated": torch.cuda.memory_allocated(i) / 1024**3,
                        "memory_reserved": torch.cuda.memory_reserved(i) / 1024**3,
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multiprocessor_count": props.multi_processor_count
                    }
                
                # Set default device
                torch.cuda.set_device(0)
                logger.info(f"PyTorch GPU setup complete. {device_count} GPU(s) available")
                
                # Print GPU info
                for gpu_id, info in self.gpu_info.items():
                    logger.info(f"{gpu_id}: {info['name']} - {info['memory_total']:.2f}GB")
                
                return True
            else:
                logger.warning("PyTorch: No CUDA GPUs available")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up PyTorch GPU: {e}")
            return False
    
    def setup_tensorflow_gpu(self) -> bool:
        """Configure TensorFlow for GPU usage"""
        try:
            # Check GPU availability
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                # Enable memory growth to prevent TF from allocating all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Log GPU info
                logger.info(f"TensorFlow GPU setup complete. {len(gpus)} GPU(s) available")
                for gpu in gpus:
                    logger.info(f"TensorFlow GPU: {gpu.name}")
                
                return True
            else:
                logger.warning("TensorFlow: No GPUs available")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up TensorFlow GPU: {e}")
            return False
    
    def get_gpu_utilization(self) -> List[Dict[str, Any]]:
        """Get current GPU utilization"""
        gpu_stats = []
        
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_stats.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load * 100,  # Percentage
                    "memory_used": gpu.memoryUsed,  # MB
                    "memory_total": gpu.memoryTotal,  # MB
                    "memory_util": gpu.memoryUtil * 100,  # Percentage
                    "temperature": gpu.temperature  # Celsius
                })
        except Exception as e:
            logger.error(f"Error getting GPU utilization: {e}")
        
        return gpu_stats
    
    def benchmark_pytorch(self, size: int = 10000) -> Dict[str, float]:
        """Benchmark PyTorch operations on GPU"""
        if not self.gpu_available:
            logger.warning("GPU not available for PyTorch benchmark")
            return {}
        
        results = {}
        device = torch.device("cuda")
        
        # Matrix multiplication benchmark
        logger.info("Running PyTorch matrix multiplication benchmark...")
        a = torch.randn(size, size).to(device)
        b = torch.randn(size, size).to(device)
        
        # Warm up
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        import time
        start = time.time()
        for _ in range(100):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
        
        results['matmul_time'] = (end - start) / 100
        results['matmul_tflops'] = (2 * size**3) / (results['matmul_time'] * 1e12)
        
        # CNN benchmark
        logger.info("Running PyTorch CNN benchmark...")
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3),
            torch.nn.ReLU()
        ).to(device)
        
        input_tensor = torch.randn(32, 3, 224, 224).to(device)
        
        # Warm up
        for _ in range(10):
            output = model(input_tensor)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            output = model(input_tensor)
        torch.cuda.synchronize()
        end = time.time()
        
        results['cnn_inference_time'] = (end - start) / 100
        results['cnn_fps'] = 32 / results['cnn_inference_time']
        
        self.benchmark_results['pytorch'] = results
        logger.info(f"PyTorch benchmark complete: {results}")
        
        return results
    
    def benchmark_tensorflow(self, size: int = 10000) -> Dict[str, float]:
        """Benchmark TensorFlow operations on GPU"""
        results = {}
        
        try:
            # Matrix multiplication benchmark
            logger.info("Running TensorFlow matrix multiplication benchmark...")
            
            with tf.device('/GPU:0'):
                a = tf.random.normal([size, size])
                b = tf.random.normal([size, size])
                
                # Warm up
                for _ in range(10):
                    c = tf.matmul(a, b)
                
                # Benchmark
                import time
                start = time.time()
                for _ in range(100):
                    c = tf.matmul(a, b)
                end = time.time()
                
                results['matmul_time'] = (end - start) / 100
                results['matmul_tflops'] = (2 * size**3) / (results['matmul_time'] * 1e12)
            
            self.benchmark_results['tensorflow'] = results
            logger.info(f"TensorFlow benchmark complete: {results}")
            
        except Exception as e:
            logger.error(f"Error in TensorFlow benchmark: {e}")
        
        return results
    
    def optimize_for_inference(self) -> Dict[str, Any]:
        """Optimize GPU settings for inference workloads"""
        optimizations = {}
        
        if self.gpu_available:
            # PyTorch optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            optimizations['pytorch_cudnn_benchmark'] = True
            optimizations['pytorch_tf32'] = True
            
            # Set to inference mode
            torch.set_grad_enabled(False)
            optimizations['pytorch_grad_disabled'] = True
            
            # TensorFlow optimizations
            tf.config.optimizer.set_jit(True)
            optimizations['tensorflow_xla'] = True
            
            logger.info("GPU optimized for inference workloads")
        
        return optimizations
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get detailed GPU memory information"""
        memory_info = {}
        
        if self.gpu_available:
            # PyTorch memory info
            memory_info['pytorch'] = {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'reserved': torch.cuda.memory_reserved() / 1024**3,
                'free': (torch.cuda.get_device_properties(0).total_memory - 
                        torch.cuda.memory_allocated()) / 1024**3
            }
            
            # System GPU info
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # RTX 5090
                memory_info['system'] = {
                    'total': gpu.memoryTotal / 1024,  # GB
                    'used': gpu.memoryUsed / 1024,
                    'free': gpu.memoryFree / 1024,
                    'utilization': gpu.memoryUtil * 100
                }
        
        return memory_info
    
    def setup_mixed_precision(self) -> bool:
        """Setup mixed precision training for faster performance"""
        try:
            # PyTorch mixed precision
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            logger.info("PyTorch mixed precision (AMP) enabled")
            
            # TensorFlow mixed precision
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("TensorFlow mixed precision enabled")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up mixed precision: {e}")
            return False
    
    def run_full_setup(self) -> Dict[str, Any]:
        """Run complete GPU environment setup"""
        logger.info("Starting full GPU environment setup...")
        
        setup_results = {
            'cuda_installed': self.check_cuda_installation(),
            'pytorch_gpu': self.setup_pytorch_gpu(),
            'tensorflow_gpu': self.setup_tensorflow_gpu(),
            'mixed_precision': self.setup_mixed_precision(),
            'optimizations': self.optimize_for_inference(),
            'gpu_info': self.gpu_info,
            'memory_info': self.get_memory_info()
        }
        
        # Run benchmarks if GPU available
        if self.gpu_available:
            logger.info("Running GPU benchmarks...")
            self.benchmark_pytorch(size=5000)
            self.benchmark_tensorflow(size=5000)
            setup_results['benchmarks'] = self.benchmark_results
        
        logger.info("GPU environment setup complete")
        return setup_results


# Singleton instance
gpu_env = GPUEnvironment()

def get_gpu_environment() -> GPUEnvironment:
    """Get GPU environment instance"""
    return gpu_env

def is_gpu_available() -> bool:
    """Quick check if GPU is available"""
    return torch.cuda.is_available()

def get_device() -> torch.device:
    """Get appropriate device (GPU if available, else CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")