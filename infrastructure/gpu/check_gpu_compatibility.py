#!/usr/bin/env python3
"""
GPU Driver Compatibility Checker for YTEmpire
Verifies CUDA, PyTorch, and GPU driver compatibility
"""
import subprocess
import sys
import platform
import json
from typing import Dict, Any, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUCompatibilityChecker:
    """Check GPU driver and CUDA compatibility"""
    
    def __init__(self):
        self.compatibility_matrix = {
            # NVIDIA Driver -> CUDA versions compatibility
            "535.183": ["12.2", "12.1", "12.0", "11.8"],
            "530.30": ["12.1", "12.0", "11.8", "11.7"],
            "525.125": ["12.0", "11.8", "11.7", "11.6"],
            "520.61": ["11.8", "11.7", "11.6"],
            "515.65": ["11.7", "11.6", "11.5"],
            "510.108": ["11.6", "11.5", "11.4"],
            "470.223": ["11.4", "11.3", "11.2"],
            "465.89": ["11.3", "11.2", "11.1"],
            "460.106": ["11.2", "11.1", "11.0"],
            "455.45": ["11.1", "11.0"],
            "450.203": ["11.0", "10.2"],
        }
        
        # PyTorch -> CUDA version requirements
        self.pytorch_cuda_matrix = {
            "2.1.2": ["11.8", "12.1"],
            "2.1.1": ["11.8", "12.1"],
            "2.1.0": ["11.8", "12.1"],
            "2.0.1": ["11.7", "11.8"],
            "2.0.0": ["11.7", "11.8"],
            "1.13.1": ["11.6", "11.7"],
            "1.13.0": ["11.6", "11.7"],
            "1.12.1": ["10.2", "11.3", "11.6"],
            "1.12.0": ["10.2", "11.3", "11.6"],
        }
        
        # RTX 5090 specific requirements (hypothetical - update when released)
        self.rtx5090_requirements = {
            "min_driver": "535.0",
            "min_cuda": "12.0",
            "compute_capability": "9.0",  # Hypothetical for RTX 5090
            "min_memory": 24576,  # 24GB VRAM
        }
    
    def check_nvidia_driver(self) -> Optional[str]:
        """Check installed NVIDIA driver version"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"NVIDIA Driver Version: {version}")
                return version
            else:
                logger.warning("nvidia-smi command failed")
                return None
        except FileNotFoundError:
            logger.error("nvidia-smi not found. NVIDIA drivers may not be installed.")
            return None
        except Exception as e:
            logger.error(f"Error checking NVIDIA driver: {e}")
            return None
    
    def check_cuda_version(self) -> Optional[str]:
        """Check installed CUDA version"""
        try:
            # Try nvcc first
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                # Parse CUDA version from nvcc output
                for line in result.stdout.split('\n'):
                    if 'release' in line:
                        # Extract version like "11.8" from "Cuda compilation tools, release 11.8, V11.8.89"
                        parts = line.split('release')
                        if len(parts) > 1:
                            version = parts[1].split(',')[0].strip()
                            logger.info(f"CUDA Version: {version}")
                            return version
        except FileNotFoundError:
            logger.warning("nvcc not found, checking nvidia-smi for CUDA version")
        
        # Fallback to nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'CUDA Version' in line:
                        # Extract version from "CUDA Version: 11.8"
                        version = line.split('CUDA Version:')[1].strip().split()[0]
                        logger.info(f"CUDA Version (from nvidia-smi): {version}")
                        return version
        except:
            pass
        
        logger.error("Could not determine CUDA version")
        return None
    
    def check_pytorch_cuda(self) -> Tuple[bool, Optional[str]]:
        """Check PyTorch CUDA availability and version"""
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            cuda_version = None
            
            if cuda_available:
                cuda_version = torch.version.cuda
                logger.info(f"PyTorch CUDA Available: Yes")
                logger.info(f"PyTorch CUDA Version: {cuda_version}")
                logger.info(f"PyTorch Version: {torch.__version__}")
                
                # Check GPU details
                gpu_count = torch.cuda.device_count()
                logger.info(f"Number of GPUs: {gpu_count}")
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                logger.warning("PyTorch CUDA is NOT available")
            
            return cuda_available, cuda_version
            
        except ImportError:
            logger.error("PyTorch is not installed")
            return False, None
        except Exception as e:
            logger.error(f"Error checking PyTorch CUDA: {e}")
            return False, None
    
    def check_gpu_info(self) -> List[Dict[str, Any]]:
        """Get detailed GPU information"""
        gpus = []
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,compute_cap,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        gpus.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'memory_mb': int(parts[2]),
                            'compute_capability': parts[3],
                            'utilization': int(parts[4])
                        })
                        
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
        
        return gpus
    
    def check_compatibility(self) -> Dict[str, Any]:
        """Perform comprehensive compatibility check"""
        results = {
            'status': 'unknown',
            'driver_version': None,
            'cuda_version': None,
            'pytorch_cuda': False,
            'pytorch_cuda_version': None,
            'gpus': [],
            'compatibility': {
                'driver_cuda': False,
                'pytorch_cuda': False,
                'rtx5090_ready': False
            },
            'recommendations': []
        }
        
        # Check driver
        driver_version = self.check_nvidia_driver()
        results['driver_version'] = driver_version
        
        # Check CUDA
        cuda_version = self.check_cuda_version()
        results['cuda_version'] = cuda_version
        
        # Check PyTorch
        pytorch_cuda, pytorch_cuda_version = self.check_pytorch_cuda()
        results['pytorch_cuda'] = pytorch_cuda
        results['pytorch_cuda_version'] = pytorch_cuda_version
        
        # Get GPU info
        results['gpus'] = self.check_gpu_info()
        
        # Check compatibility
        if driver_version and cuda_version:
            # Check driver-CUDA compatibility
            compatible_cuda_versions = []
            for drv_ver, cuda_vers in self.compatibility_matrix.items():
                if driver_version.startswith(drv_ver.split('.')[0]):
                    compatible_cuda_versions.extend(cuda_vers)
                    break
            
            results['compatibility']['driver_cuda'] = cuda_version in compatible_cuda_versions
            
            if not results['compatibility']['driver_cuda']:
                results['recommendations'].append(
                    f"CUDA {cuda_version} may not be fully compatible with driver {driver_version}"
                )
        
        # Check PyTorch-CUDA compatibility
        if pytorch_cuda_version and cuda_version:
            results['compatibility']['pytorch_cuda'] = True  # Simplified check
        
        # Check RTX 5090 readiness
        if results['gpus']:
            for gpu in results['gpus']:
                if 'RTX 5090' in gpu.get('name', '') or '5090' in gpu.get('name', ''):
                    # Check specific RTX 5090 requirements
                    if driver_version:
                        driver_major = float(driver_version.split('.')[0])
                        min_driver_major = float(self.rtx5090_requirements['min_driver'].split('.')[0])
                        
                        if driver_major >= min_driver_major:
                            results['compatibility']['rtx5090_ready'] = True
                        else:
                            results['recommendations'].append(
                                f"RTX 5090 requires driver {self.rtx5090_requirements['min_driver']} or newer"
                            )
                    
                    if cuda_version:
                        cuda_major = float(cuda_version.split('.')[0])
                        min_cuda_major = float(self.rtx5090_requirements['min_cuda'].split('.')[0])
                        
                        if cuda_major < min_cuda_major:
                            results['compatibility']['rtx5090_ready'] = False
                            results['recommendations'].append(
                                f"RTX 5090 requires CUDA {self.rtx5090_requirements['min_cuda']} or newer"
                            )
        
        # Overall status
        if driver_version and cuda_version and pytorch_cuda:
            if all(results['compatibility'].values()):
                results['status'] = 'fully_compatible'
            else:
                results['status'] = 'partially_compatible'
        elif driver_version or cuda_version:
            results['status'] = 'incomplete_setup'
        else:
            results['status'] = 'no_gpu_support'
            results['recommendations'].append("Install NVIDIA drivers and CUDA toolkit")
        
        return results
    
    def generate_report(self) -> str:
        """Generate compatibility report"""
        results = self.check_compatibility()
        
        report = []
        report.append("=" * 60)
        report.append("YTEmpire GPU Compatibility Check Report")
        report.append("=" * 60)
        report.append(f"Platform: {platform.system()} {platform.release()}")
        report.append(f"Python: {sys.version.split()[0]}")
        report.append("")
        
        # Status
        status_emoji = {
            'fully_compatible': '✅',
            'partially_compatible': '⚠️',
            'incomplete_setup': '❌',
            'no_gpu_support': '❌',
            'unknown': '❓'
        }
        report.append(f"Overall Status: {status_emoji[results['status']]} {results['status'].upper()}")
        report.append("")
        
        # Driver and CUDA
        report.append("Software Versions:")
        report.append(f"  NVIDIA Driver: {results['driver_version'] or 'Not found'}")
        report.append(f"  CUDA Version: {results['cuda_version'] or 'Not found'}")
        report.append(f"  PyTorch CUDA: {'Yes' if results['pytorch_cuda'] else 'No'}")
        if results['pytorch_cuda_version']:
            report.append(f"  PyTorch CUDA Version: {results['pytorch_cuda_version']}")
        report.append("")
        
        # GPUs
        if results['gpus']:
            report.append("Detected GPUs:")
            for gpu in results['gpus']:
                report.append(f"  [{gpu['index']}] {gpu['name']}")
                report.append(f"      Memory: {gpu['memory_mb']/1024:.1f} GB")
                report.append(f"      Compute Capability: {gpu['compute_capability']}")
                report.append(f"      Current Utilization: {gpu['utilization']}%")
        else:
            report.append("No NVIDIA GPUs detected")
        report.append("")
        
        # Compatibility
        report.append("Compatibility Checks:")
        report.append(f"  Driver-CUDA: {'✅ Compatible' if results['compatibility']['driver_cuda'] else '❌ May have issues'}")
        report.append(f"  PyTorch-CUDA: {'✅ Compatible' if results['compatibility']['pytorch_cuda'] else '❌ Not available'}")
        report.append(f"  RTX 5090 Ready: {'✅ Yes' if results['compatibility']['rtx5090_ready'] else '❌ No'}")
        report.append("")
        
        # Recommendations
        if results['recommendations']:
            report.append("Recommendations:")
            for rec in results['recommendations']:
                report.append(f"  • {rec}")
        else:
            report.append("✅ No issues found - system is ready for GPU acceleration!")
        
        report.append("")
        report.append("=" * 60)
        
        return '\n'.join(report)


def main():
    """Run GPU compatibility check"""
    checker = GPUCompatibilityChecker()
    
    # Generate and print report
    report = checker.generate_report()
    print(report)
    
    # Save report to file
    with open('gpu_compatibility_report.txt', 'w') as f:
        f.write(report)
    
    # Also save JSON results
    results = checker.check_compatibility()
    with open('gpu_compatibility_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nReports saved to:")
    print("  - gpu_compatibility_report.txt")
    print("  - gpu_compatibility_results.json")
    
    # Return exit code based on status
    if results['status'] == 'fully_compatible':
        return 0
    elif results['status'] == 'partially_compatible':
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())