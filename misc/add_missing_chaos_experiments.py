"""
Add missing experiment classes to chaos_engineering_suite.py
"""

import sys
from pathlib import Path

# Missing experiment classes to add
missing_experiments = '''

class APIServiceFailureExperiment(ChaosExperiment):
    """Simulate API service failures"""
    
    def __init__(self):
        super().__init__(
            name="API Service Failure",
            description="Simulates API service crashes and failures",
            duration=300
        )
    
    def setup(self) -> bool:
        """Setup API failure experiment"""
        logger.info("Setting up API service failure experiment")
        self.baseline_metrics = self.collect_metrics('baseline')
        return True
    
    def execute(self) -> Dict[str, Any]:
        """Execute API failure"""
        logger.info("Simulating API service failure")
        
        results = {
            'failure_type': 'api_crash',
            'start_time': datetime.now().isoformat()
        }
        
        try:
            # Simulate API failure by stopping the service
            if DOCKER_AVAILABLE and self.docker_client:
                try:
                    api_container = self.docker_client.containers.get('ytempire_backend')
                    api_container.stop()
                    results['api_stopped'] = True
                    time.sleep(30)  # Keep it down for 30 seconds
                    api_container.start()
                    results['api_restarted'] = True
                except Exception as e:
                    results['docker_error'] = str(e)
            else:
                # Fallback simulation
                results['simulated'] = True
                time.sleep(30)
            
            results['end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            results['error'] = str(e)
        
        self.experiment_metrics = self.collect_metrics('experiment')
        results['metrics'] = self.experiment_metrics
        
        return results
    
    def cleanup(self) -> bool:
        """Cleanup after experiment"""
        logger.info("Cleaning up API service failure experiment")
        
        # Ensure API is running
        if DOCKER_AVAILABLE and self.docker_client:
            try:
                api_container = self.docker_client.containers.get('ytempire_backend')
                if api_container.status != 'running':
                    api_container.start()
            except:
                pass
        
        self.recovery_metrics = self.collect_metrics('recovery')
        return True


class HighMemoryPressureExperiment(ChaosExperiment):
    """Simulate high memory pressure"""
    
    def __init__(self):
        super().__init__(
            name="High Memory Pressure",
            description="Simulates high memory usage scenarios",
            duration=300
        )
    
    def setup(self) -> bool:
        """Setup memory pressure experiment"""
        logger.info("Setting up high memory pressure experiment")
        self.baseline_metrics = self.collect_metrics('baseline')
        return True
    
    def execute(self) -> Dict[str, Any]:
        """Execute memory pressure simulation"""
        logger.info("Simulating high memory pressure")
        
        results = {
            'pressure_type': 'memory_exhaustion',
            'start_time': datetime.now().isoformat()
        }
        
        try:
            # Simulate memory pressure
            import gc
            memory_hog = []
            
            # Allocate large chunks of memory
            for i in range(10):
                try:
                    # Allocate 100MB chunks
                    chunk = bytearray(100 * 1024 * 1024)
                    memory_hog.append(chunk)
                    time.sleep(2)
                except MemoryError:
                    results['memory_exhausted'] = True
                    break
            
            results['memory_allocated_mb'] = len(memory_hog) * 100
            
            # Hold for a bit
            time.sleep(10)
            
            # Release memory
            del memory_hog
            gc.collect()
            
            results['memory_released'] = True
            results['end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            results['error'] = str(e)
        
        self.experiment_metrics = self.collect_metrics('experiment')
        results['metrics'] = self.experiment_metrics
        
        return results
    
    def cleanup(self) -> bool:
        """Cleanup after experiment"""
        logger.info("Cleaning up memory pressure experiment")
        
        import gc
        gc.collect()
        
        self.recovery_metrics = self.collect_metrics('recovery')
        return True


class DiskFullExperiment(ChaosExperiment):
    """Simulate disk full scenarios"""
    
    def __init__(self):
        super().__init__(
            name="Disk Full",
            description="Simulates disk space exhaustion",
            duration=300
        )
        self.temp_file = None
    
    def setup(self) -> bool:
        """Setup disk full experiment"""
        logger.info("Setting up disk full experiment")
        self.baseline_metrics = self.collect_metrics('baseline')
        return True
    
    def execute(self) -> Dict[str, Any]:
        """Execute disk full simulation"""
        logger.info("Simulating disk full scenario")
        
        results = {
            'failure_type': 'disk_exhaustion',
            'start_time': datetime.now().isoformat()
        }
        
        try:
            # Create a large temporary file
            import tempfile
            self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.chaos')
            
            # Write large amounts of data (but not too much to avoid real issues)
            chunk_size = 10 * 1024 * 1024  # 10MB chunks
            max_size = 500 * 1024 * 1024  # Max 500MB
            
            written = 0
            while written < max_size:
                self.temp_file.write(b'0' * chunk_size)
                written += chunk_size
                time.sleep(1)
            
            results['file_size_mb'] = written / (1024 * 1024)
            self.temp_file.close()
            
            # Hold for observation
            time.sleep(10)
            
            results['end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            results['error'] = str(e)
        
        self.experiment_metrics = self.collect_metrics('experiment')
        results['metrics'] = self.experiment_metrics
        
        return results
    
    def cleanup(self) -> bool:
        """Cleanup after experiment"""
        logger.info("Cleaning up disk full experiment")
        
        # Remove temporary file
        if self.temp_file:
            try:
                import os
                os.unlink(self.temp_file.name)
            except:
                pass
        
        self.recovery_metrics = self.collect_metrics('recovery')
        return True


class LatencyInjectionExperiment(ChaosExperiment):
    """Inject network latency"""
    
    def __init__(self):
        super().__init__(
            name="Latency Injection",
            description="Simulates network latency and slow responses",
            duration=300
        )
    
    def setup(self) -> bool:
        """Setup latency injection experiment"""
        logger.info("Setting up latency injection experiment")
        self.baseline_metrics = self.collect_metrics('baseline')
        return True
    
    def execute(self) -> Dict[str, Any]:
        """Execute latency injection"""
        logger.info("Injecting network latency")
        
        results = {
            'injection_type': 'network_latency',
            'start_time': datetime.now().isoformat(),
            'latency_ms': 500
        }
        
        try:
            # On Windows, we simulate latency
            # In production on Linux, you would use tc (traffic control)
            
            if sys.platform == 'win32':
                # Windows simulation
                results['platform'] = 'windows'
                results['simulation'] = True
                
                # Simulate by adding delays to a mock service
                for i in range(10):
                    time.sleep(results['latency_ms'] / 1000)
                    
            else:
                # Linux - use tc command
                import subprocess
                
                # Add latency to localhost
                cmd = f"tc qdisc add dev lo root netem delay {results['latency_ms']}ms"
                subprocess.run(cmd, shell=True, capture_output=True)
                results['tc_rule_added'] = True
                
                # Keep latency for duration
                time.sleep(30)
                
                # Remove latency
                cmd = "tc qdisc del dev lo root"
                subprocess.run(cmd, shell=True, capture_output=True)
                results['tc_rule_removed'] = True
            
            results['end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            results['error'] = str(e)
        
        self.experiment_metrics = self.collect_metrics('experiment')
        results['metrics'] = self.experiment_metrics
        
        return results
    
    def cleanup(self) -> bool:
        """Cleanup after experiment"""
        logger.info("Cleaning up latency injection experiment")
        
        # Ensure tc rules are removed on Linux
        if sys.platform != 'win32':
            try:
                import subprocess
                subprocess.run("tc qdisc del dev lo root", shell=True, capture_output=True)
            except:
                pass
        
        self.recovery_metrics = self.collect_metrics('recovery')
        return True


class RandomServiceRestartExperiment(ChaosExperiment):
    """Randomly restart services"""
    
    def __init__(self):
        super().__init__(
            name="Random Service Restart",
            description="Randomly restarts services to test recovery",
            duration=300
        )
    
    def setup(self) -> bool:
        """Setup random restart experiment"""
        logger.info("Setting up random service restart experiment")
        self.baseline_metrics = self.collect_metrics('baseline')
        return True
    
    def execute(self) -> Dict[str, Any]:
        """Execute random service restarts"""
        logger.info("Executing random service restarts")
        
        results = {
            'restart_type': 'random_services',
            'start_time': datetime.now().isoformat(),
            'services_restarted': []
        }
        
        try:
            services = ['redis', 'celery_worker', 'celery_beat']
            
            if DOCKER_AVAILABLE and self.docker_client:
                for service in services:
                    try:
                        container_name = f'ytempire_{service}'
                        container = self.docker_client.containers.get(container_name)
                        
                        # Restart the service
                        container.restart()
                        results['services_restarted'].append(service)
                        
                        # Wait a bit between restarts
                        time.sleep(10)
                        
                    except Exception as e:
                        results[f'{service}_error'] = str(e)
            else:
                # Simulation mode
                results['simulated'] = True
                results['services_restarted'] = services
                time.sleep(30)
            
            results['end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            results['error'] = str(e)
        
        self.experiment_metrics = self.collect_metrics('experiment')
        results['metrics'] = self.experiment_metrics
        
        return results
    
    def cleanup(self) -> bool:
        """Cleanup after experiment"""
        logger.info("Cleaning up random restart experiment")
        
        # Ensure all services are running
        if DOCKER_AVAILABLE and self.docker_client:
            services = ['redis', 'celery_worker', 'celery_beat', 'backend']
            for service in services:
                try:
                    container = self.docker_client.containers.get(f'ytempire_{service}')
                    if container.status != 'running':
                        container.start()
                except:
                    pass
        
        self.recovery_metrics = self.collect_metrics('recovery')
        return True


# Insert these classes before the main() function
'''

def add_missing_experiments():
    """Add missing experiment classes to chaos_engineering_suite.py"""
    
    file_path = Path("infrastructure/testing/chaos_engineering_suite.py")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return False
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find where to insert (before the main() function)
    insert_pos = content.find('def main():')
    
    if insert_pos == -1:
        # If no main function, append at the end
        content += missing_experiments
    else:
        # Insert before main()
        content = content[:insert_pos] + missing_experiments + '\n\n' + content[insert_pos:]
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Successfully added missing experiment classes to {file_path}")
    return True

if __name__ == "__main__":
    add_missing_experiments()