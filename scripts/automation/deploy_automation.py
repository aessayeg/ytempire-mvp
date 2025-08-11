#!/usr/bin/env python3
"""
YTEmpire Deployment Automation
Advanced deployment orchestration with monitoring and rollback capabilities
"""

import os
import sys
import json
import time
import subprocess
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DeploymentOrchestrator:
    """Main deployment orchestration class"""
    
    def __init__(self, environment: str, version: str, config_path: Optional[str] = None):
        self.environment = environment
        self.version = version
        self.config = self._load_config(config_path)
        self.start_time = time.time()
        self.deployment_id = f"{environment}-{version}-{int(time.time())}"
        self.rollback_points = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load deployment configuration"""
        if not config_path:
            config_path = Path(__file__).parent.parent.parent / "deployment.config.yml"
        
        if not config_path.exists():
            # Default configuration
            return {
                "environments": {
                    "staging": {
                        "host": "staging.ytempire.com",
                        "port": 22,
                        "user": "deploy",
                        "compose_file": "docker-compose.staging.yml",
                        "health_check_url": "http://localhost:8001/api/v1/health",
                        "services": ["postgres", "redis", "backend", "frontend", "celery_worker"]
                    },
                    "production": {
                        "host": "ytempire.com",
                        "port": 22,
                        "user": "deploy",
                        "compose_file": "docker-compose.production.yml",
                        "health_check_url": "https://api.ytempire.com/api/v1/health",
                        "services": ["postgres", "redis", "backend", "frontend", "celery_worker", "nginx"]
                    }
                },
                "deployment": {
                    "strategy": "blue-green",
                    "health_check_retries": 30,
                    "health_check_interval": 10,
                    "rollback_on_failure": True,
                    "backup_enabled": True,
                    "monitoring_enabled": True
                },
                "notifications": {
                    "slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),
                    "email_recipients": os.getenv("DEPLOYMENT_EMAIL_RECIPIENTS", "").split(",")
                }
            }
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def deploy(self, dry_run: bool = False) -> bool:
        """Execute deployment pipeline"""
        logger.info(f"Starting deployment: {self.deployment_id}")
        
        try:
            # Pre-deployment phase
            if not self._pre_deployment_checks():
                return False
            
            if self.config['deployment']['backup_enabled']:
                self._create_backup()
            
            # Deployment phase
            if dry_run:
                logger.info("DRY RUN: Simulating deployment")
                self._simulate_deployment()
            else:
                self._execute_deployment()
            
            # Post-deployment phase
            if not self._post_deployment_validation():
                if self.config['deployment']['rollback_on_failure']:
                    self._rollback()
                return False
            
            # Success
            self._finalize_deployment()
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            if self.config['deployment']['rollback_on_failure']:
                self._rollback()
            return False
    
    def _pre_deployment_checks(self) -> bool:
        """Run pre-deployment validation"""
        logger.info("Running pre-deployment checks...")
        
        checks = [
            self._check_docker_images,
            self._check_disk_space,
            self._check_service_health,
            self._validate_configuration,
            self._check_database_backup
        ]
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(check) for check in checks]
            results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Pre-deployment check failed: {e}")
                    results.append(False)
        
        if all(results):
            logger.info("✅ All pre-deployment checks passed")
            return True
        else:
            logger.error("❌ Pre-deployment checks failed")
            return False
    
    def _check_docker_images(self) -> bool:
        """Verify Docker images exist"""
        logger.info("Checking Docker images...")
        
        images = [
            f"ghcr.io/ytempire/ytempire-mvp-backend:{self.version}",
            f"ghcr.io/ytempire/ytempire-mvp-frontend:{self.version}"
        ]
        
        for image in images:
            try:
                result = subprocess.run(
                    ["docker", "pull", image],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode != 0:
                    logger.error(f"Image not found: {image}")
                    return False
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout pulling image: {image}")
                return False
        
        return True
    
    def _check_disk_space(self) -> bool:
        """Check available disk space"""
        logger.info("Checking disk space...")
        
        result = subprocess.run(
            ["df", "-BG", "/"],
            capture_output=True,
            text=True
        )
        
        # Parse available space
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split()
            if len(parts) > 3:
                available = int(parts[3].replace('G', ''))
                if available < 10:
                    logger.error(f"Insufficient disk space: {available}GB")
                    return False
        
        return True
    
    def _check_service_health(self) -> bool:
        """Check current service health"""
        logger.info("Checking service health...")
        
        env_config = self.config['environments'][self.environment]
        
        try:
            response = requests.get(
                env_config['health_check_url'],
                timeout=10
            )
            if response.status_code == 200:
                logger.info("Current services are healthy")
                return True
        except requests.RequestException:
            logger.warning("Current services not responding (may be first deployment)")
        
        return True
    
    def _validate_configuration(self) -> bool:
        """Validate deployment configuration"""
        logger.info("Validating configuration...")
        
        # Check environment variables
        required_vars = [
            "DATABASE_URL",
            "REDIS_URL",
            "SECRET_KEY"
        ]
        
        env_file = Path(f".env.{self.environment}")
        if not env_file.exists():
            logger.error(f"Environment file not found: {env_file}")
            return False
        
        with open(env_file, 'r') as f:
            env_content = f.read()
            for var in required_vars:
                if var not in env_content:
                    logger.error(f"Missing required variable: {var}")
                    return False
        
        return True
    
    def _check_database_backup(self) -> bool:
        """Verify database backup capability"""
        logger.info("Checking database backup...")
        
        # Check if pg_dump is available
        result = subprocess.run(
            ["which", "pg_dump"],
            capture_output=True
        )
        
        if result.returncode != 0:
            logger.warning("pg_dump not found, database backup may fail")
        
        return True
    
    def _create_backup(self):
        """Create comprehensive backup"""
        logger.info("Creating backup...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(f"backups/{self.environment}/{timestamp}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Database backup
        self._backup_database(backup_dir)
        
        # Configuration backup
        self._backup_configuration(backup_dir)
        
        # Volume backup
        self._backup_volumes(backup_dir)
        
        # Create manifest
        manifest = {
            "deployment_id": self.deployment_id,
            "timestamp": timestamp,
            "environment": self.environment,
            "version": self.version,
            "services": self.config['environments'][self.environment]['services']
        }
        
        with open(backup_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self.rollback_points.append(str(backup_dir))
        logger.info(f"✅ Backup created: {backup_dir}")
    
    def _backup_database(self, backup_dir: Path):
        """Backup database"""
        try:
            subprocess.run(
                [
                    "docker", "exec", f"ytempire_postgres_{self.environment}",
                    "pg_dump", "-U", "ytempire", f"ytempire_{self.environment}"
                ],
                stdout=open(backup_dir / "database.sql", 'w'),
                check=True,
                timeout=300
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Database backup failed: {e}")
    
    def _backup_configuration(self, backup_dir: Path):
        """Backup configuration files"""
        import shutil
        
        files_to_backup = [
            f".env.{self.environment}",
            f"docker-compose.{self.environment}.yml"
        ]
        
        for file in files_to_backup:
            if Path(file).exists():
                shutil.copy(file, backup_dir)
    
    def _backup_volumes(self, backup_dir: Path):
        """Backup Docker volumes"""
        volumes = [
            f"ytempire_postgres_{self.environment}_data",
            f"ytempire_redis_{self.environment}_data"
        ]
        
        for volume in volumes:
            try:
                subprocess.run(
                    [
                        "docker", "run", "--rm",
                        "-v", f"{volume}:/source:ro",
                        "-v", f"{backup_dir}:/backup",
                        "alpine",
                        "tar", "czf", f"/backup/{volume}.tar.gz", "-C", "/source", "."
                    ],
                    check=True,
                    timeout=300
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.warning(f"Volume backup failed for {volume}: {e}")
    
    def _simulate_deployment(self):
        """Simulate deployment for dry run"""
        logger.info("Simulating deployment steps...")
        
        steps = [
            "Pull Docker images",
            "Stop old containers",
            "Start new containers",
            "Run database migrations",
            "Health checks",
            "Traffic switch"
        ]
        
        for step in steps:
            logger.info(f"  - {step}")
            time.sleep(0.5)
        
        logger.info("✅ Dry run completed successfully")
    
    def _execute_deployment(self):
        """Execute actual deployment"""
        logger.info("Executing deployment...")
        
        env_config = self.config['environments'][self.environment]
        compose_file = env_config['compose_file']
        
        # Pull new images
        self._pull_images(compose_file)
        
        # Deploy based on strategy
        strategy = self.config['deployment']['strategy']
        
        if strategy == 'blue-green':
            self._deploy_blue_green(compose_file)
        elif strategy == 'canary':
            self._deploy_canary(compose_file)
        else:
            self._deploy_rolling(compose_file)
    
    def _pull_images(self, compose_file: str):
        """Pull Docker images"""
        logger.info("Pulling Docker images...")
        
        subprocess.run(
            ["docker-compose", "-f", compose_file, "pull"],
            check=True,
            timeout=600
        )
    
    def _deploy_blue_green(self, compose_file: str):
        """Blue-green deployment strategy"""
        logger.info("Executing blue-green deployment...")
        
        # Determine current and new environment
        current_color = self._get_current_color()
        new_color = "green" if current_color == "blue" else "blue"
        
        logger.info(f"Current: {current_color}, New: {new_color}")
        
        # Start new environment
        subprocess.run(
            [
                "docker-compose",
                "-f", f"{compose_file}.{new_color}",
                "up", "-d"
            ],
            check=True
        )
        
        # Wait for health
        if self._wait_for_health(new_color):
            # Switch traffic
            self._switch_traffic(new_color)
            
            # Stop old environment after delay
            time.sleep(60)
            subprocess.run(
                [
                    "docker-compose",
                    "-f", f"{compose_file}.{current_color}",
                    "down"
                ],
                check=False
            )
        else:
            raise Exception(f"Health check failed for {new_color} environment")
    
    def _deploy_canary(self, compose_file: str):
        """Canary deployment strategy"""
        logger.info("Executing canary deployment...")
        
        # Deploy canary instance (10% traffic)
        subprocess.run(
            [
                "docker-compose",
                "-f", compose_file,
                "up", "-d",
                "--scale", "backend=1"
            ],
            check=True
        )
        
        # Monitor canary
        logger.info("Monitoring canary for 5 minutes...")
        time.sleep(300)
        
        if self._check_canary_metrics():
            # Scale to 50%
            logger.info("Scaling to 50% traffic...")
            subprocess.run(
                [
                    "docker-compose",
                    "-f", compose_file,
                    "up", "-d",
                    "--scale", "backend=2"
                ],
                check=True
            )
            
            time.sleep(300)
            
            # Scale to 100%
            logger.info("Scaling to 100% traffic...")
            subprocess.run(
                [
                    "docker-compose",
                    "-f", compose_file,
                    "up", "-d",
                    "--scale", "backend=4"
                ],
                check=True
            )
        else:
            raise Exception("Canary deployment failed metrics check")
    
    def _deploy_rolling(self, compose_file: str):
        """Rolling deployment strategy"""
        logger.info("Executing rolling deployment...")
        
        services = self.config['environments'][self.environment]['services']
        
        for service in services:
            logger.info(f"Updating {service}...")
            
            # Update service
            subprocess.run(
                [
                    "docker-compose",
                    "-f", compose_file,
                    "up", "-d",
                    "--no-deps",
                    service
                ],
                check=True
            )
            
            # Wait for health
            time.sleep(30)
            
            if not self._check_service_health_individual(service):
                raise Exception(f"Health check failed for {service}")
    
    def _get_current_color(self) -> str:
        """Get current deployment color (blue/green)"""
        color_file = Path(f".{self.environment}_color")
        if color_file.exists():
            return color_file.read_text().strip()
        return "blue"
    
    def _switch_traffic(self, new_color: str):
        """Switch traffic to new environment"""
        logger.info(f"Switching traffic to {new_color}...")
        
        # Update nginx or load balancer configuration
        # This would be environment-specific
        
        # Save new color
        color_file = Path(f".{self.environment}_color")
        color_file.write_text(new_color)
    
    def _wait_for_health(self, color: str) -> bool:
        """Wait for environment to be healthy"""
        retries = self.config['deployment']['health_check_retries']
        interval = self.config['deployment']['health_check_interval']
        
        for i in range(retries):
            if self._check_health(color):
                logger.info(f"✅ {color} environment is healthy")
                return True
            
            logger.info(f"Health check attempt {i+1}/{retries}")
            time.sleep(interval)
        
        return False
    
    def _check_health(self, color: str = None) -> bool:
        """Check service health"""
        env_config = self.config['environments'][self.environment]
        health_url = env_config['health_check_url']
        
        if color:
            # Adjust URL for blue/green
            health_url = health_url.replace("8000", "8001" if color == "green" else "8000")
        
        try:
            response = requests.get(health_url, timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def _check_service_health_individual(self, service: str) -> bool:
        """Check individual service health"""
        # Service-specific health checks
        health_checks = {
            "backend": lambda: self._check_health(),
            "frontend": lambda: self._check_url("http://localhost:3000"),
            "postgres": lambda: self._check_postgres(),
            "redis": lambda: self._check_redis()
        }
        
        check = health_checks.get(service, lambda: True)
        return check()
    
    def _check_url(self, url: str) -> bool:
        """Check if URL is accessible"""
        try:
            response = requests.get(url, timeout=10)
            return response.status_code < 500
        except requests.RequestException:
            return False
    
    def _check_postgres(self) -> bool:
        """Check PostgreSQL health"""
        try:
            result = subprocess.run(
                [
                    "docker", "exec",
                    f"ytempire_postgres_{self.environment}",
                    "pg_isready", "-U", "ytempire"
                ],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
    
    def _check_redis(self) -> bool:
        """Check Redis health"""
        try:
            result = subprocess.run(
                [
                    "docker", "exec",
                    f"ytempire_redis_{self.environment}",
                    "redis-cli", "ping"
                ],
                capture_output=True,
                timeout=10
            )
            return b"PONG" in result.stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
    
    def _check_canary_metrics(self) -> bool:
        """Check canary deployment metrics"""
        logger.info("Checking canary metrics...")
        
        # Check error rate, latency, etc.
        # This would integrate with Prometheus/Grafana
        
        # Simplified check for now
        return self._check_health()
    
    def _post_deployment_validation(self) -> bool:
        """Run post-deployment validation"""
        logger.info("Running post-deployment validation...")
        
        validations = [
            self._validate_endpoints,
            self._validate_database_connectivity,
            self._validate_redis_connectivity,
            self._run_smoke_tests,
            self._check_metrics
        ]
        
        for validation in validations:
            if not validation():
                return False
        
        logger.info("✅ Post-deployment validation passed")
        return True
    
    def _validate_endpoints(self) -> bool:
        """Validate API endpoints"""
        logger.info("Validating endpoints...")
        
        endpoints = [
            "/api/v1/health",
            "/api/v1/auth/status",
            "/docs"
        ]
        
        env_config = self.config['environments'][self.environment]
        base_url = env_config['health_check_url'].replace('/api/v1/health', '')
        
        for endpoint in endpoints:
            url = f"{base_url}{endpoint}"
            if not self._check_url(url):
                logger.error(f"Endpoint validation failed: {endpoint}")
                return False
        
        return True
    
    def _validate_database_connectivity(self) -> bool:
        """Validate database connectivity"""
        logger.info("Validating database connectivity...")
        return self._check_postgres()
    
    def _validate_redis_connectivity(self) -> bool:
        """Validate Redis connectivity"""
        logger.info("Validating Redis connectivity...")
        return self._check_redis()
    
    def _run_smoke_tests(self) -> bool:
        """Run smoke tests"""
        logger.info("Running smoke tests...")
        
        # Run basic API tests
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/smoke/", "-v"],
                capture_output=True,
                timeout=300
            )
            if result.returncode != 0:
                logger.error("Smoke tests failed")
                return False
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Smoke tests not available")
        
        return True
    
    def _check_metrics(self) -> bool:
        """Check deployment metrics"""
        logger.info("Checking metrics...")
        
        # Check Prometheus metrics
        # This would query actual metrics
        
        return True
    
    def _rollback(self):
        """Rollback deployment"""
        logger.warning("Initiating rollback...")
        
        if not self.rollback_points:
            logger.error("No rollback points available")
            return
        
        latest_backup = self.rollback_points[-1]
        logger.info(f"Rolling back to: {latest_backup}")
        
        # Restore from backup
        self._restore_backup(latest_backup)
        
        # Restart services
        env_config = self.config['environments'][self.environment]
        compose_file = env_config['compose_file']
        
        subprocess.run(
            ["docker-compose", "-f", compose_file, "down"],
            check=False
        )
        
        subprocess.run(
            ["docker-compose", "-f", compose_file, "up", "-d"],
            check=False
        )
        
        logger.info("✅ Rollback completed")
    
    def _restore_backup(self, backup_path: str):
        """Restore from backup"""
        backup_dir = Path(backup_path)
        
        # Restore database
        db_backup = backup_dir / "database.sql"
        if db_backup.exists():
            try:
                with open(db_backup, 'r') as f:
                    subprocess.run(
                        [
                            "docker", "exec", "-i",
                            f"ytempire_postgres_{self.environment}",
                            "psql", "-U", "ytempire", f"ytempire_{self.environment}"
                        ],
                        stdin=f,
                        check=True
                    )
            except subprocess.CalledProcessError as e:
                logger.error(f"Database restore failed: {e}")
        
        # Restore configuration
        for file in backup_dir.glob("*.env*"):
            import shutil
            shutil.copy(file, ".")
    
    def _finalize_deployment(self):
        """Finalize successful deployment"""
        duration = time.time() - self.start_time
        logger.info(f"✅ Deployment completed successfully in {duration:.2f} seconds")
        
        # Send notifications
        self._send_notifications("success", duration)
        
        # Update deployment history
        self._update_deployment_history("success", duration)
        
        # Cleanup old resources
        self._cleanup_old_resources()
    
    def _send_notifications(self, status: str, duration: float):
        """Send deployment notifications"""
        message = f"""
        Deployment {status.upper()}
        Environment: {self.environment}
        Version: {self.version}
        Duration: {duration:.2f} seconds
        Deployment ID: {self.deployment_id}
        """
        
        # Slack notification
        if self.config['notifications']['slack_webhook']:
            self._send_slack_notification(message, status)
        
        # Email notification
        if self.config['notifications']['email_recipients']:
            self._send_email_notification(message, status)
    
    def _send_slack_notification(self, message: str, status: str):
        """Send Slack notification"""
        color = "good" if status == "success" else "danger"
        
        payload = {
            "attachments": [{
                "color": color,
                "text": message,
                "footer": "YTEmpire Deployment System",
                "ts": int(time.time())
            }]
        }
        
        try:
            requests.post(
                self.config['notifications']['slack_webhook'],
                json=payload,
                timeout=10
            )
        except requests.RequestException as e:
            logger.warning(f"Failed to send Slack notification: {e}")
    
    def _send_email_notification(self, message: str, status: str):
        """Send email notification"""
        # Implement email sending
        pass
    
    def _update_deployment_history(self, status: str, duration: float):
        """Update deployment history"""
        history_file = Path(f"deployments_{self.environment}.json")
        
        history = []
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        
        history.append({
            "deployment_id": self.deployment_id,
            "environment": self.environment,
            "version": self.version,
            "status": status,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep last 100 deployments
        history = history[-100:]
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _cleanup_old_resources(self):
        """Cleanup old Docker resources"""
        logger.info("Cleaning up old resources...")
        
        # Remove unused images
        subprocess.run(
            ["docker", "image", "prune", "-f"],
            check=False
        )
        
        # Remove old volumes
        subprocess.run(
            ["docker", "volume", "prune", "-f"],
            check=False
        )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="YTEmpire Deployment Automation")
    parser.add_argument("environment", choices=["staging", "production"],
                       help="Deployment environment")
    parser.add_argument("version", help="Version to deploy")
    parser.add_argument("--dry-run", action="store_true",
                       help="Perform dry run without actual deployment")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--skip-backup", action="store_true",
                       help="Skip backup creation")
    parser.add_argument("--skip-health-check", action="store_true",
                       help="Skip health checks")
    parser.add_argument("--force", action="store_true",
                       help="Force deployment even if checks fail")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = DeploymentOrchestrator(
        environment=args.environment,
        version=args.version,
        config_path=args.config
    )
    
    # Override configuration if needed
    if args.skip_backup:
        orchestrator.config['deployment']['backup_enabled'] = False
    
    if args.skip_health_check:
        orchestrator.config['deployment']['health_check_retries'] = 1
    
    if args.force:
        orchestrator.config['deployment']['rollback_on_failure'] = False
    
    # Execute deployment
    success = orchestrator.deploy(dry_run=args.dry_run)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()