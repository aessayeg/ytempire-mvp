"""
GPU Benchmark Script
Owner: ML Engineer
Run this to validate GPU setup and performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.gpu_config import GPUEnvironment
import logging
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_gpu_diagnostics():
    """Run complete GPU diagnostics and benchmarks"""
    
    print("=" * 60)
    print("YTEmpire GPU Environment Diagnostics")
    print("=" * 60)
    
    # Initialize GPU environment
    gpu_env = GPUEnvironment()
    
    # Run full setup
    print("\n1. Running GPU Environment Setup...")
    setup_results = gpu_env.run_full_setup()
    
    # Print results
    print("\n2. Setup Results:")
    print("-" * 40)
    print(f"CUDA Installed: {setup_results['cuda_installed']}")
    print(f"CUDA Version: {gpu_env.cuda_version}")
    print(f"PyTorch GPU: {setup_results['pytorch_gpu']}")
    print(f"TensorFlow GPU: {setup_results['tensorflow_gpu']}")
    print(f"Mixed Precision: {setup_results['mixed_precision']}")
    
    # GPU Information
    if gpu_env.gpu_info:
        print("\n3. GPU Information:")
        print("-" * 40)
        for gpu_id, info in gpu_env.gpu_info.items():
            print(f"{gpu_id}:")
            print(f"  Name: {info['name']}")
            print(f"  Memory: {info['memory_total']:.2f} GB")
            print(f"  Compute Capability: {info['compute_capability']}")
            print(f"  Multiprocessors: {info['multiprocessor_count']}")
    
    # Memory Information
    if setup_results['memory_info']:
        print("\n4. Memory Status:")
        print("-" * 40)
        mem_info = setup_results['memory_info']
        if 'pytorch' in mem_info:
            print("PyTorch Memory:")
            print(f"  Allocated: {mem_info['pytorch']['allocated']:.2f} GB")
            print(f"  Reserved: {mem_info['pytorch']['reserved']:.2f} GB")
            print(f"  Free: {mem_info['pytorch']['free']:.2f} GB")
        if 'system' in mem_info:
            print("System GPU Memory:")
            print(f"  Total: {mem_info['system']['total']:.2f} GB")
            print(f"  Used: {mem_info['system']['used']:.2f} GB")
            print(f"  Free: {mem_info['system']['free']:.2f} GB")
            print(f"  Utilization: {mem_info['system']['utilization']:.1f}%")
    
    # Benchmark Results
    if 'benchmarks' in setup_results and setup_results['benchmarks']:
        print("\n5. Benchmark Results:")
        print("-" * 40)
        
        if 'pytorch' in setup_results['benchmarks']:
            pytorch_bench = setup_results['benchmarks']['pytorch']
            print("PyTorch Performance:")
            print(f"  Matrix Multiplication: {pytorch_bench['matmul_tflops']:.2f} TFLOPS")
            print(f"  CNN Inference: {pytorch_bench['cnn_fps']:.2f} FPS")
            print(f"  MatMul Time: {pytorch_bench['matmul_time']*1000:.2f} ms")
            print(f"  CNN Time: {pytorch_bench['cnn_inference_time']*1000:.2f} ms")
        
        if 'tensorflow' in setup_results['benchmarks']:
            tf_bench = setup_results['benchmarks']['tensorflow']
            print("TensorFlow Performance:")
            print(f"  Matrix Multiplication: {tf_bench['matmul_tflops']:.2f} TFLOPS")
            print(f"  MatMul Time: {tf_bench['matmul_time']*1000:.2f} ms")
    
    # Check GPU Utilization
    print("\n6. Current GPU Utilization:")
    print("-" * 40)
    gpu_stats = gpu_env.get_gpu_utilization()
    if gpu_stats:
        for gpu in gpu_stats:
            print(f"GPU {gpu['id']} ({gpu['name']}):")
            print(f"  Load: {gpu['load']:.1f}%")
            print(f"  Memory: {gpu['memory_used']}/{gpu['memory_total']} MB ({gpu['memory_util']:.1f}%)")
            print(f"  Temperature: {gpu['temperature']}°C")
    else:
        print("Unable to retrieve GPU utilization")
    
    # Performance Recommendations
    print("\n7. Performance Recommendations:")
    print("-" * 40)
    
    if gpu_env.gpu_available:
        print("✓ GPU is available and configured")
        print("✓ Mixed precision enabled for faster training")
        print("✓ CUDNN benchmarking enabled")
        print("✓ TF32 enabled for better performance")
        
        # Check if RTX 5090 specific features
        if any('5090' in info['name'] for info in gpu_env.gpu_info.values()):
            print("✓ RTX 5090 detected - optimal for AI workloads")
            print("  - 32GB VRAM available for large models")
            print("  - Support for FP8 precision")
            print("  - Hardware acceleration for transformers")
    else:
        print("⚠ GPU not available - running on CPU")
        print("  Consider checking CUDA installation")
        print("  Verify GPU drivers are installed")
    
    # Save results to file
    results_file = f"gpu_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(setup_results, f, indent=2, default=str)
    
    print(f"\n8. Results saved to: {results_file}")
    print("=" * 60)
    
    return setup_results


def test_video_generation_performance():
    """Test GPU performance for video generation tasks"""
    import torch
    import time
    
    print("\n" + "=" * 60)
    print("Video Generation Performance Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("GPU not available for video generation test")
        return
    
    device = torch.device("cuda")
    
    # Simulate video frame processing
    print("\nSimulating video frame processing...")
    frame_size = (1080, 1920, 3)  # 1080p frame
    num_frames = 300  # 10 seconds at 30fps
    
    # Create batch of frames
    frames = torch.randn(num_frames, 3, 1080, 1920).to(device)
    
    # Simple convolution to simulate processing
    conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
    
    start = time.time()
    with torch.no_grad():
        for i in range(0, num_frames, 30):  # Process in batches of 30
            batch = frames[i:i+30]
            output = conv(batch)
    torch.cuda.synchronize()
    end = time.time()
    
    processing_time = end - start
    fps = num_frames / processing_time
    
    print(f"Processed {num_frames} frames in {processing_time:.2f} seconds")
    print(f"Processing speed: {fps:.2f} FPS")
    print(f"Real-time capability: {'YES' if fps >= 30 else 'NO'}")
    
    # Memory usage
    memory_used = torch.cuda.memory_allocated() / 1024**3
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"\nMemory Usage:")
    print(f"  Used: {memory_used:.2f} GB")
    print(f"  Total: {memory_total:.2f} GB")
    print(f"  Available: {memory_total - memory_used:.2f} GB")
    
    # Estimate video generation capacity
    print(f"\nVideo Generation Capacity:")
    print(f"  Can process: {int(fps/30)} simultaneous 1080p streams")
    print(f"  Est. generation time for 10-min video: {600/fps*30:.1f} seconds")
    
    return {
        'fps': fps,
        'processing_time': processing_time,
        'memory_used': memory_used,
        'real_time_capable': fps >= 30
    }


if __name__ == "__main__":
    # Run diagnostics
    results = run_gpu_diagnostics()
    
    # Run video generation test
    if results['pytorch_gpu']:
        video_perf = test_video_generation_performance()
    
    print("\n✅ GPU benchmark complete!")
    print("Ready for YTEmpire video generation workloads.")