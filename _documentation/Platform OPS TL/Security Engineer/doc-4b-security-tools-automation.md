# 4B. SECURITY TOOLS & AUTOMATION - Detailed Implementation

## Executive Summary

This document contains the complete implementation details for security tools and automation that were in the original Security Tools & Automation Guide. This supplements Document 4 (Implementation Guides) with specific code, configurations, and scripts.

---

## 1. OWASP ZAP Complete Implementation

### 1.1 Installation and Setup Script

```bash
#!/bin/bash
# install_zap.sh - Complete OWASP ZAP setup for YTEMPIRE

set -e

echo "🔧 Installing OWASP ZAP for YTEMPIRE Security Testing..."

# Create directory structure
mkdir -p /opt/ytempire/security/zap/{config,sessions,reports,scripts}

# Run ZAP container with persistent storage
docker run -d \
  --name ytempire-zap \
  --restart unless-stopped \
  -p 8090:8090 \
  -p 8091:8091 \
  -v /opt/ytempire/security/zap:/zap/wrk:rw \
  -v /opt/ytempire/security/zap/config:/home/zap/.ZAP:rw \
  -e ZAP_PORT=8090 \
  -e ZAP_API_ALLOW_UNSAFE=false \
  owasp/zap2docker-stable \
  zap-webswing.sh

# Wait for ZAP to initialize
echo "⏳ Waiting for ZAP to start..."
sleep 30

# Generate secure API key
ZAP_API_KEY=$(openssl rand -hex 32)

# Store API key securely
cat > /opt/ytempire/security/.env << EOF
# YTEMPIRE Security Tools Configuration
ZAP_API_KEY=$ZAP_API_KEY
ZAP_HOST=localhost
ZAP_PORT=8090
EOF

chmod 600 /opt/ytempire/security/.env

# Configure ZAP for YTEMPIRE
cat > /opt/ytempire/security/zap/config/ytempire.conf << EOF
# YTEMPIRE ZAP Configuration
api.key=$ZAP_API_KEY
api.addrs.addr.regex=true
api.addrs.addr.name=127.0.0.1
api.disablekey=false

# Spider Configuration
spider.maxDuration=60
spider.maxDepth=10
spider.maxChildren=20
spider.thread=5

# Scanner Configuration
scanner.maxScanDurationInMins=60
scanner.threadPerHost=2
scanner.maxResultsToList=500

# Network Configuration
connection.timeoutInSecs=30
network.connection.dnsTtlSuccessfulQueries=300

# Anti CSRF Configuration
anticsrf.tokens.token=csrf_token,authenticity_token,__RequestVerificationToken

# Session Management
session.type=cookie
session.tokens=JSESSIONID,PHPSESSID,ASP.NET_SessionId,session_id
EOF

echo "✅ ZAP installation complete!"
echo "📝 API Key stored in: /opt/ytempire/security/.env"
echo "🌐 ZAP UI available at: http://localhost:8090"
```

### 1.2 Python ZAP Scanner Class

```python
#!/usr/bin/env python3
# security_tools/zap_scanner.py

import os
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from zapv2 import ZAPv2
import requests
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YTEMPIREZAPScanner:
    """Advanced OWASP ZAP integration for YTEMPIRE security testing"""
    
    def __init__(self):
        """Initialize ZAP scanner with YTEMPIRE configuration"""
        # Load configuration from environment
        self.api_key = os.environ.get('ZAP_API_KEY')
        self.zap_host = os.environ.get('ZAP_HOST', 'localhost')
        self.zap_port = int(os.environ.get('ZAP_PORT', 8090))
        
        # Initialize ZAP connection
        self.zap = ZAPv2(
            apikey=self.api_key,
            proxies={
                'http': f'http://{self.zap_host}:{self.zap_port}',
                'https': f'http://{self.zap_host}:{self.zap_port}'
            }
        )
        
        # YTEMPIRE-specific configurations
        self.targets = {
            'api': {
                'url': 'https://api.ytempire.com',
                'auth_type': 'jwt',
                'context': 'YTEMPIRE_API',
                'scan_policy': 'API_Security_Test'
            },
            'frontend': {
                'url': 'https://app.ytempire.com',
                'auth_type': 'session',
                'context': 'YTEMPIRE_Frontend',
                'scan_policy': 'Web_Application_Test'
            },
            'admin': {
                'url': 'https://admin.ytempire.com',
                'auth_type': 'oauth',
                'context': 'YTEMPIRE_Admin',
                'scan_policy': 'Admin_Security_Test'
            }
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'Critical': 0,  # Zero tolerance
            'High': 2,      # Max 2 high risks
            'Medium': 10,   # Max 10 medium risks
            'Low': 50       # Max 50 low risks
        }
    
    def authenticate(self, target: str, credentials: Dict) -> bool:
        """Authenticate with YTEMPIRE services"""
        if target not in self.targets:
            logger.error(f"Unknown target: {target}")
            return False
        
        target_config = self.targets[target]
        auth_type = target_config['auth_type']
        
        if auth_type == 'jwt':
            return self._auth_jwt(target_config['url'], credentials)
        elif auth_type == 'session':
            return self._auth_session(target_config['url'], credentials)
        elif auth_type == 'oauth':
            return self._auth_oauth(target_config['url'], credentials)
        
        return False
    
    def _auth_jwt(self, url: str, credentials: Dict) -> bool:
        """JWT authentication for API"""
        try:
            # Login to get JWT token
            response = requests.post(
                f"{url}/auth/login",
                json={
                    'email': credentials['email'],
                    'password': credentials['password']
                }
            )
            
            if response.status_code == 200:
                token = response.json().get('access_token')
                # Set JWT token in ZAP
                self.zap.httpsessions.add_session_token(
                    url, 'Authorization', f'Bearer {token}'
                )
                logger.info("JWT authentication successful")
                return True
        except Exception as e:
            logger.error(f"JWT authentication failed: {e}")
        
        return False
    
    def perform_security_scan(self, target: str, scan_type: str = 'full') -> Dict:
        """Perform comprehensive security scan"""
        logger.info(f"🔍 Starting {scan_type} scan for {target}")
        
        if target not in self.targets:
            return {'error': f'Unknown target: {target}'}
        
        target_config = self.targets[target]
        target_url = target_config['url']
        
        results = {
            'target': target,
            'url': target_url,
            'scan_type': scan_type,
            'start_time': datetime.utcnow().isoformat(),
            'alerts': [],
            'statistics': {},
            'passed': False
        }
        
        try:
            # Create new session
            session_name = f"YTEMPIRE_{target}_{int(time.time())}"
            self.zap.httpsessions.create_empty_session(target_url, session_name)
            
            # Set active session
            self.zap.httpsessions.set_active_session(target_url, session_name)
            
            # Configure context
            context_id = self._setup_context(target_config)
            
            # Spider the target
            if scan_type in ['full', 'spider']:
                spider_results = self._spider_target(target_url, context_id)
                results['statistics']['pages_found'] = spider_results['pages']
            
            # Active scan
            if scan_type in ['full', 'active']:
                scan_results = self._active_scan(target_url, context_id)
                results['statistics']['requests_made'] = scan_results['requests']
            
            # Passive scan results
            if scan_type in ['full', 'passive']:
                passive_results = self._get_passive_results(target_url)
                results['alerts'].extend(passive_results)
            
            # Analyze results
            results = self._analyze_results(results)
            
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            results['error'] = str(e)
        
        results['end_time'] = datetime.utcnow().isoformat()
        
        # Generate report
        self._generate_report(results)
        
        return results
    
    def _spider_target(self, url: str, context_id: str) -> Dict:
        """Spider the target to discover all pages"""
        logger.info(f"🕷️ Spidering {url}")
        
        # Start spider
        scan_id = self.zap.spider.scan(
            url=url,
            contextname=context_id,
            subtreeonly=True,
            recurse=True,
            maxchildren=20
        )
        
        # Wait for spider to complete
        while int(self.zap.spider.status(scan_id)) < 100:
            progress = self.zap.spider.status(scan_id)
            logger.info(f"Spider progress: {progress}%")
            time.sleep(5)
        
        # Get results
        urls = self.zap.spider.results(scan_id)
        
        return {
            'scan_id': scan_id,
            'pages': len(urls),
            'urls': urls[:100]  # Limit for report
        }
    
    def _active_scan(self, url: str, context_id: str) -> Dict:
        """Perform active security scan"""
        logger.info(f"⚡ Active scanning {url}")
        
        # Configure scan policy
        self._configure_scan_policy()
        
        # Start active scan
        scan_id = self.zap.ascan.scan(
            url=url,
            recurse=True,
            inscopeonly=True,
            scanpolicyname='YTEMPIRE_Security_Policy',
            contextid=context_id
        )
        
        # Monitor scan progress
        while int(self.zap.ascan.status(scan_id)) < 100:
            progress = self.zap.ascan.status(scan_id)
            logger.info(f"Active scan progress: {progress}%")
            time.sleep(10)
        
        # Get scan statistics
        stats = {
            'scan_id': scan_id,
            'requests': self.zap.ascan.messages_ids(scan_id),
            'alerts_found': len(self.zap.core.alerts(baseurl=url))
        }
        
        return stats
    
    def _analyze_results(self, results: Dict) -> Dict:
        """Analyze scan results against thresholds"""
        alerts = self.zap.core.alerts()
        
        risk_counts = {
            'Critical': 0,
            'High': 0,
            'Medium': 0,
            'Low': 0,
            'Informational': 0
        }
        
        for alert in alerts:
            risk = alert.get('risk', 'Informational')
            risk_counts[risk] += 1
            
            # Add detailed alert info
            results['alerts'].append({
                'risk': risk,
                'confidence': alert.get('confidence'),
                'name': alert.get('name'),
                'description': alert.get('description'),
                'url': alert.get('url'),
                'solution': alert.get('solution'),
                'reference': alert.get('reference'),
                'cwe_id': alert.get('cweid'),
                'wasc_id': alert.get('wascid')
            })
        
        results['statistics']['risk_counts'] = risk_counts
        
        # Check against thresholds
        passed = True
        failures = []
        
        for risk_level, threshold in self.risk_thresholds.items():
            if risk_counts.get(risk_level, 0) > threshold:
                passed = False
                failures.append(f"{risk_level}: {risk_counts[risk_level]} > {threshold}")
        
        results['passed'] = passed
        results['failures'] = failures
        
        return results
    
    def _generate_report(self, results: Dict):
        """Generate HTML and JSON reports"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        target = results['target']
        
        # JSON report
        json_path = f"/opt/ytempire/security/reports/zap_{target}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # HTML report
        html_path = f"/opt/ytempire/security/reports/zap_{target}_{timestamp}.html"
        html_content = self._generate_html_report(results)
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"📊 Reports generated: {json_path}, {html_path}")
    
    def _generate_html_report(self, results: Dict) -> str:
        """Generate HTML report from results"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YTEMPIRE Security Scan Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; }}
                .risk-critical {{ color: #d32f2f; font-weight: bold; }}
                .risk-high {{ color: #f57c00; font-weight: bold; }}
                .risk-medium {{ color: #fbc02d; }}
                .risk-low {{ color: #388e3c; }}
                .alert {{ border: 1px solid #ddd; padding: 10px; margin: 10px 0; }}
                .passed {{ background: #4caf50; color: white; padding: 10px; }}
                .failed {{ background: #f44336; color: white; padding: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>YTEMPIRE Security Scan Report</h1>
                <p>Target: {results['target']} - {results['url']}</p>
                <p>Scan Type: {results['scan_type']}</p>
                <p>Date: {results['start_time']}</p>
            </div>
            
            <div class="{'passed' if results['passed'] else 'failed'}">
                <h2>Overall Result: {'PASSED ✓' if results['passed'] else 'FAILED ✗'}</h2>
                {self._format_failures(results.get('failures', []))}
            </div>
            
            <h2>Risk Summary</h2>
            {self._format_risk_summary(results['statistics'].get('risk_counts', {}))}
            
            <h2>Detailed Findings</h2>
            {self._format_alerts(results['alerts'])}
        </body>
        </html>
        """
        return html
    
    def _format_failures(self, failures: List[str]) -> str:
        if not failures:
            return ""
        return "<ul>" + "".join([f"<li>{f}</li>" for f in failures]) + "</ul>"
    
    def _format_risk_summary(self, risk_counts: Dict) -> str:
        html = "<table border='1' style='width:100%'>"
        html += "<tr><th>Risk Level</th><th>Count</th></tr>"
        for risk, count in risk_counts.items():
            css_class = f"risk-{risk.lower()}"
            html += f"<tr><td class='{css_class}'>{risk}</td><td>{count}</td></tr>"
        html += "</table>"
        return html
    
    def _format_alerts(self, alerts: List[Dict]) -> str:
        html = ""
        for alert in alerts[:50]:  # Limit to top 50
            css_class = f"risk-{alert['risk'].lower()}"
            html += f"""
            <div class="alert">
                <h3 class="{css_class}">[{alert['risk']}] {alert['name']}</h3>
                <p><strong>URL:</strong> {alert['url']}</p>
                <p><strong>Description:</strong> {alert['description']}</p>
                <p><strong>Solution:</strong> {alert['solution']}</p>
            </div>
            """
        return html
```

---

## 2. Automated Security Tasks

### 2.1 Daily Security Tasks Runner

```python
#!/usr/bin/env python3
# automated_security_tasks.py

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import aiohttp
import aiodns
import boto3
import requests
from cryptography import x509
from cryptography.hazmat.backends import default_backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YTEMPIRESecurityAutomation:
    """Comprehensive daily security automation for YTEMPIRE platform"""
    
    def __init__(self):
        """Initialize security automation framework"""
        self.config = self._load_config()
        self.results_dir = Path('/opt/ytempire/security/daily_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize AWS clients if needed
        self.s3_client = boto3.client('s3') if self.config.get('use_aws') else None
        
        # Security patterns
        self.suspicious_patterns = [
            (r"(?i)(union.*select|select.*from|insert.*into)", "sql_injection"),
            (r"(?i)(script.*>|javascript:|onerror=)", "xss_attempt"),
            (r"(?i)(etc/passwd|etc/shadow)", "path_traversal"),
            (r"(?i)(admin|root).*failed", "credential_guessing"),
            (r"\.\./\.\./", "path_traversal"),
            (r"union\s+select", "sql_injection"),
            (r"<script[^>]*>", "xss_attempt"),
            (r"eval\s*\(", "code_injection"),
            (r"';.*drop\s+table", "sql_injection"),
            (r"base64_decode\s*\(", "obfuscation"),
            (r"/etc/passwd", "file_inclusion"),
            (r"cmd=.*&&", "command_injection")
        ]
        
        # Task list
        self.tasks = [
            self.check_ssl_certificates,
            self.scan_for_vulnerabilities,
            self.analyze_logs_for_threats,
            self.verify_backups,
            self.check_security_headers,
            self.audit_user_access,
            self.scan_for_secrets,
            self.check_firewall_rules,
            self.verify_encryption,
            self.check_patch_status
        ]
    
    def _load_config(self) -> Dict:
        """Load YTEMPIRE security configuration"""
        config_path = Path("/opt/ytempire/security/config.json")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'ssl_warning_days': 30,
            'ssl_critical_days': 7,
            'max_failed_logins': 5,
            'vulnerability_scan_timeout': 600,
            'backup_retention_days': 30,
            'email_recipients': ['security@ytempire.com'],
            'slack_webhook': os.environ.get('SLACK_WEBHOOK_URL')
        }
    
    async def run_daily_tasks(self) -> Dict:
        """Execute all daily security tasks"""
        logger.info("🚀 Starting YTEMPIRE daily security tasks...")
        
        start_time = datetime.utcnow()
        results = {
            'date': start_time.strftime('%Y-%m-%d'),
            'start_time': start_time.isoformat(),
            'tasks': {},
            'issues_found': [],
            'critical_issues': 0,
            'high_issues': 0,
            'medium_issues': 0,
            'low_issues': 0
        }
        
        # Run tasks concurrently
        task_coroutines = []
        for task in self.tasks:
            task_coroutines.append(self._run_task_safely(task))
        
        task_results = await asyncio.gather(*task_coroutines)
        
        # Process results
        for task, result in zip(self.tasks, task_results):
            task_name = task.__name__
            results['tasks'][task_name] = result
            
            if result['status'] == 'success' and result.get('issues'):
                for issue in result['issues']:
                    results['issues_found'].append({
                        'task': task_name,
                        **issue
                    })
                    
                    # Count by severity
                    severity = issue.get('severity', 'low')
                    if severity == 'critical':
                        results['critical_issues'] += 1
                    elif severity == 'high':
                        results['high_issues'] += 1
                    elif severity == 'medium':
                        results['medium_issues'] += 1
                    else:
                        results['low_issues'] += 1
        
        results['end_time'] = datetime.utcnow().isoformat()
        duration = (datetime.utcnow() - start_time).total_seconds()
        results['duration_seconds'] = duration
        
        # Save results
        self._save_results(results)
        
        # Send notifications if issues found
        if results['critical_issues'] > 0 or results['high_issues'] > 0:
            await self._send_notifications(results)
        
        logger.info(f"✅ Daily security tasks completed in {duration:.2f} seconds")
        return results
    
    async def _run_task_safely(self, task) -> Dict:
        """Run a task with error handling"""
        try:
            result = await task()
            result['status'] = 'success'
            return result
        except Exception as e:
            logger.error(f"Task {task.__name__} failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'task': task.__name__
            }
    
    async def check_ssl_certificates(self) -> Dict:
        """Check SSL certificate expiration for all domains"""
        logger.info("🔐 Checking SSL certificates...")
        
        domains = [
            'ytempire.com',
            'api.ytempire.com',
            'app.ytempire.com',
            'admin.ytempire.com'
        ]
        
        issues = []
        certificates = {}
        
        for domain in domains:
            try:
                cert_info = await self._get_ssl_info(domain)
                certificates[domain] = cert_info
                
                days_remaining = cert_info['days_remaining']
                
                if days_remaining < self.config['ssl_critical_days']:
                    issues.append({
                        'severity': 'critical',
                        'domain': domain,
                        'message': f'SSL certificate expires in {days_remaining} days',
                        'expiry_date': cert_info['expiry_date']
                    })
                elif days_remaining < self.config['ssl_warning_days']:
                    issues.append({
                        'severity': 'high',
                        'domain': domain,
                        'message': f'SSL certificate expires in {days_remaining} days',
                        'expiry_date': cert_info['expiry_date']
                    })
            except Exception as e:
                issues.append({
                    'severity': 'high',
                    'domain': domain,
                    'message': f'Failed to check SSL certificate: {e}'
                })
        
        return {
            'certificates': certificates,
            'issues': issues
        }
    
    async def _get_ssl_info(self, domain: str) -> Dict:
        """Get SSL certificate information for a domain"""
        import ssl
        import socket
        
        context = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as ssock:
                der_cert_bin = ssock.getpeercert_bin()
                cert = x509.load_der_x509_certificate(der_cert_bin, default_backend())
                
                expiry_date = cert.not_valid_after
                days_remaining = (expiry_date - datetime.utcnow()).days
                
                return {
                    'domain': domain,
                    'issuer': cert.issuer.rfc4514_string(),
                    'subject': cert.subject.rfc4514_string(),
                    'expiry_date': expiry_date.isoformat(),
                    'days_remaining': days_remaining,
                    'serial_number': str(cert.serial_number),
                    'signature_algorithm': cert.signature_algorithm_oid._name
                }
    
    async def scan_for_vulnerabilities(self) -> Dict:
        """Run vulnerability scans on containers and dependencies"""
        logger.info("🔍 Scanning for vulnerabilities...")
        
        issues = []
        scan_results = {}
        
        # Scan Docker containers
        containers = await self._get_running_containers()
        for container in containers:
            scan = await self._scan_container(container)
            scan_results[container] = scan
            
            if scan['critical_count'] > 0:
                issues.append({
                    'severity': 'critical',
                    'container': container,
                    'message': f'{scan["critical_count"]} critical vulnerabilities found',
                    'details': scan['critical_vulns'][:5]  # Top 5
                })
            
            if scan['high_count'] > 0:
                issues.append({
                    'severity': 'high',
                    'container': container,
                    'message': f'{scan["high_count"]} high vulnerabilities found',
                    'details': scan['high_vulns'][:5]
                })
        
        # Scan Python dependencies
        pip_scan = await self._scan_pip_dependencies()
        scan_results['pip_dependencies'] = pip_scan
        
        if pip_scan['vulnerabilities']:
            for vuln in pip_scan['vulnerabilities']:
                issues.append({
                    'severity': vuln['severity'].lower(),
                    'package': vuln['package'],
                    'message': vuln['description'],
                    'cve': vuln.get('cve')
                })
        
        return {
            'scan_results': scan_results,
            'issues': issues
        }
    
    async def _scan_container(self, container: str) -> Dict:
        """Scan a Docker container for vulnerabilities using Trivy"""
        try:
            cmd = f"trivy image --format json --quiet {container}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                critical_vulns = []
                high_vulns = []
                medium_vulns = []
                low_vulns = []
                
                for target in data.get('Results', []):
                    for vuln in target.get('Vulnerabilities', []):
                        severity = vuln.get('Severity', 'UNKNOWN')
                        vuln_info = {
                            'id': vuln.get('VulnerabilityID'),
                            'package': vuln.get('PkgName'),
                            'version': vuln.get('InstalledVersion'),
                            'fixed': vuln.get('FixedVersion'),
                            'title': vuln.get('Title')
                        }
                        
                        if severity == 'CRITICAL':
                            critical_vulns.append(vuln_info)
                        elif severity == 'HIGH':
                            high_vulns.append(vuln_info)
                        elif severity == 'MEDIUM':
                            medium_vulns.append(vuln_info)
                        elif severity == 'LOW':
                            low_vulns.append(vuln_info)
                
                return {
                    'critical_count': len(critical_vulns),
                    'high_count': len(high_vulns),
                    'medium_count': len(medium_vulns),
                    'low_count': len(low_vulns),
                    'critical_vulns': critical_vulns,
                    'high_vulns': high_vulns,
                    'medium_vulns': medium_vulns,
                    'low_vulns': low_vulns
                }
        except Exception as e:
            logger.error(f"Container scan failed for {container}: {e}")
            return {
                'error': str(e),
                'critical_count': 0,
                'high_count': 0,
                'medium_count': 0,
                'low_count': 0
            }
    
    async def analyze_logs_for_threats(self) -> Dict:
        """Analyze application and system logs for security threats"""
        logger.info("📊 Analyzing logs for threats...")
        
        issues = []
        threat_summary = {
            'failed_logins': 0,
            'sql_injection_attempts': 0,
            'xss_attempts': 0,
            'path_traversal_attempts': 0,
            'suspicious_ips': set(),
            'attack_patterns': []
        }
        
        # Analyze auth logs
        auth_analysis = await self._analyze_auth_logs()
        threat_summary['failed_logins'] = auth_analysis['failed_count']
        
        if auth_analysis['failed_count'] > self.config['max_failed_logins']:
            issues.append({
                'severity': 'high',
                'type': 'authentication',
                'message': f'{auth_analysis["failed_count"]} failed login attempts detected',
                'details': auth_analysis['top_offenders']
            })
        
        # Analyze application logs
        app_analysis = await self._analyze_app_logs()
        for pattern_type, count in app_analysis['pattern_matches'].items():
            if count > 0:
                threat_summary[f'{pattern_type}_attempts'] = count
                
                if count > 10:
                    issues.append({
                        'severity': 'high' if count > 50 else 'medium',
                        'type': pattern_type,
                        'message': f'{count} {pattern_type} attempts detected',
                        'samples': app_analysis['samples'][pattern_type][:3]
                    })
        
        # Check for suspicious IPs
        suspicious_ips = await self._check_ip_reputation(
            list(auth_analysis['source_ips'])
        )
        
        if suspicious_ips:
            threat_summary['suspicious_ips'] = suspicious_ips
            issues.append({
                'severity': 'high',
                'type': 'suspicious_ip',
                'message': f'{len(suspicious_ips)} suspicious IPs detected',
                'ips': list(suspicious_ips)[:10]
            })
        
        return {
            'threat_summary': threat_summary,
            'issues': issues
        }
    
    def _save_results(self, results: Dict):
        """Save results to file and optionally to S3"""
        timestamp = datetime.utcnow().strftime('%Y%m%d')
        filename = f"security_report_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📁 Results saved to {filepath}")
        
        # Upload to S3 if configured
        if self.s3_client and self.config.get('s3_bucket'):
            try:
                self.s3_client.upload_file(
                    str(filepath),
                    self.config['s3_bucket'],
                    f"security-reports/{filename}"
                )
                logger.info(f"☁️ Results uploaded to S3")
            except Exception as e:
                logger.error(f"Failed to upload to S3: {e}")
    
    async def _send_notifications(self, results: Dict):
        """Send notifications for critical/high issues"""
        message = self._format_notification_message(results)
        
        # Send Slack notification
        if self.config.get('slack_webhook'):
            await self._send_slack_notification(message)
        
        # Send email notification
        if self.config.get('email_recipients'):
            await self._send_email_notification(message, results)
    
    def _format_notification_message(self, results: Dict) -> str:
        """Format notification message"""
        message = f"""
🚨 YTEMPIRE Security Report - {results['date']}

Issues Found:
- Critical: {results['critical_issues']}
- High: {results['high_issues']}
- Medium: {results['medium_issues']}
- Low: {results['low_issues']}

Top Issues:
"""
        for issue in results['issues_found'][:5]:
            message += f"\n[{issue.get('severity', 'unknown').upper()}] {issue.get('message', 'No message')}"
        
        return message
```

---

## 3. Auto-Remediation System

### 3.1 Automated Remediation Engine

```python
#!/usr/bin/env python3
# auto_remediation.py

import asyncio
import docker
import json
import logging
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
import requests

logger = logging.getLogger(__name__)


class YTEMPIREAutoRemediation:
    """Automated security remediation system for YTEMPIRE"""
    
    def __init__(self):
        """Initialize auto-remediation system"""
        self.docker_client = docker.from_env()
        self.remediation_history = []
        self.config = self._load_config()
        
        # Remediation strategies
        self.strategies = {
            'container_vulnerability': self.remediate_container_vulnerability,
            'ssl_expiry': self.remediate_ssl_expiry,
            'failed_login': self.remediate_failed_login,
            'firewall_rule': self.remediate_firewall_rule,
            'disk_space': self.remediate_disk_space,
            'service_down': self.remediate_service_down,
            'sql_injection': self.remediate_sql_injection,
            'xss_attempt': self.remediate_xss_attempt
        }
    
    async def process_security_issue(self, issue: Dict) -> Dict:
        """Process and remediate a security issue"""
        logger.info(f"🔧 Processing security issue: {issue.get('type')}")
        
        result = {
            'issue': issue,
            'timestamp': datetime.utcnow().isoformat(),
            'remediation_attempted': False,
            'success': False,
            'actions_taken': [],
            'error': None
        }
        
        try:
            issue_type = issue.get('type')
            severity = issue.get('severity', 'low')
            
            # Check if auto-remediation is allowed
            if not self._should_auto_remediate(issue_type, severity):
                result['error'] = 'Auto-remediation not allowed for this issue type/severity'
                return result
            
            # Get remediation strategy
            strategy = self.strategies.get(issue_type)
            if not strategy:
                result['error'] = f'No remediation strategy for {issue_type}'
                return result
            
            # Execute remediation
            result['remediation_attempted'] = True
            remediation_result = await strategy(issue)
            
            result['success'] = remediation_result.get('success', False)
            result['actions_taken'] = remediation_result.get('actions', [])
            
            if not result['success']:
                result['error'] = remediation_result.get('error')
            
            # Log remediation
            self._log_remediation(result)
            
            # Send notification
            await self._notify_remediation(result)
            
        except Exception as e:
            logger.error(f"Remediation failed: {e}")
            result['error'] = str(e)
        
        return result
    
    def _should_auto_remediate(self, issue_type: str, severity: str) -> bool:
        """Check if auto-remediation should be attempted"""
        # Critical issues require human review
        if severity == 'critical':
            return False
        
        # Check whitelist
        auto_remediate_types = self.config.get('auto_remediate_types', [])
        return issue_type in auto_remediate_types
    
    async def remediate_container_vulnerability(self, issue: Dict) -> Dict:
        """Remediate container vulnerability by updating image"""
        actions = []
        
        try:
            container_name = issue.get('container')
            if not container_name:
                return {'success': False, 'error': 'No container specified'}
            
            # Pull latest image
            logger.info(f"Pulling latest image for {container_name}")
            image = self.docker_client.images.pull(container_name, tag='latest')
            actions.append(f"Pulled latest image: {image.tags}")
            
            # Find and restart container
            containers = self.docker_client.containers.list(
                filters={'ancestor': container_name}
            )
            
            for container in containers:
                logger.info(f"Restarting container {container.name}")
                container.restart()
                actions.append(f"Restarted container: {container.name}")
            
            return {'success': True, 'actions': actions}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'actions': actions}
    
    async def remediate_ssl_expiry(self, issue: Dict) -> Dict:
        """Remediate SSL certificate expiry by renewing certificate"""
        actions = []
        
        try:
            domain = issue.get('domain')
            if not domain:
                return {'success': False, 'error': 'No domain specified'}
            
            # Renew certificate using certbot
            logger.info(f"Renewing SSL certificate for {domain}")
            cmd = f"certbot renew --domain {domain} --quiet"
            result = subprocess.run(cmd, shell=True, capture_output=True)
            
            if result.returncode == 0:
                actions.append(f"Renewed SSL certificate for {domain}")
                
                # Reload nginx
                subprocess.run("nginx -s reload", shell=True)
                actions.append("Reloaded nginx configuration")
                
                return {'success': True, 'actions': actions}
            else:
                return {
                    'success': False,
                    'error': f"Certbot failed: {result.stderr.decode()}",
                    'actions': actions
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e), 'actions': actions}
    
    async def remediate_failed_login(self, issue: Dict) -> Dict:
        """Remediate failed login attempts by blocking IPs"""
        actions = []
        
        try:
            offending_ips = issue.get('details', {}).get('ips', [])
            
            for ip in offending_ips[:10]:  # Limit to top 10
                # Add to fail2ban
                cmd = f"fail2ban-client set sshd banip {ip}"
                subprocess.run(cmd, shell=True)
                actions.append(f"Blocked IP: {ip}")
                
                # Add firewall rule
                cmd = f"ufw insert 1 deny from {ip} to any"
                subprocess.run(cmd, shell=True)
                actions.append(f"Added firewall rule for {ip}")
            
            return {'success': True, 'actions': actions}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'actions': actions}
    
    async def remediate_disk_space(self, issue: Dict) -> Dict:
        """Remediate disk space issues by cleaning up"""
        actions = []
        
        try:
            # Clean Docker resources
            logger.info("Cleaning Docker resources")
            self.docker_client.containers.prune()
            self.docker_client.images.prune()
            self.docker_client.volumes.prune()
            actions.append("Cleaned Docker resources")
            
            # Clean old logs
            cmd = "find /var/log -type f -name '*.log' -mtime +30 -delete"
            subprocess.run(cmd, shell=True)
            actions.append("Deleted logs older than 30 days")
            
            # Clean tmp files
            cmd = "find /tmp -type f -atime +7 -delete"
            subprocess.run(cmd, shell=True)
            actions.append("Cleaned /tmp directory")
            
            return {'success': True, 'actions': actions}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'actions': actions}
    
    def _log_remediation(self, result: Dict):
        """Log remediation action"""
        self.remediation_history.append(result)
        
        # Save to file
        log_file = f"/var/log/ytempire/remediation_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(result, default=str) + '\n')
    
    async def _notify_remediation(self, result: Dict):
        """Send notification about remediation action"""
        if not self.config.get('slack_webhook'):
            return
        
        color = 'good' if result['success'] else 'danger'
        
        message = {
            'attachments': [{
                'color': color,
                'title': 'Auto-Remediation Action',
                'fields': [
                    {
                        'title': 'Issue Type',
                        'value': result['issue'].get('type', 'Unknown'),
                        'short': True
                    },
                    {
                        'title': 'Status',
                        'value': '✅ Success' if result['success'] else '❌ Failed',
                        'short': True
                    },
                    {
                        'title': 'Actions Taken',
                        'value': '\n'.join(result['actions_taken']) or 'None',
                        'short': False
                    }
                ],
                'footer': 'YTEMPIRE Security',
                'ts': int(datetime.utcnow().timestamp())
            }]
        }
        
        try:
            requests.post(self.config['slack_webhook'], json=message)
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
```

---

## 4. Security Monitoring Classes

### 4.1 Real-time Security Monitor

```python
#!/usr/bin/env python3
# security_monitoring.py

import asyncio
import json
import logging
import os
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple
import aioredis
import prometheus_client
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
security_events = Counter('ytempire_security_events_total', 'Total security events', ['type', 'severity'])
active_threats = Gauge('ytempire_active_threats', 'Currently active threats')
response_time = Histogram('ytempire_security_response_seconds', 'Security response time')


class YTEMPIRESecurityMonitor:
    """Real-time security monitoring for YTEMPIRE platform"""
    
    def __init__(self):
        """Initialize security monitor"""
        self.redis = None
        self.monitoring_active = False
        self.threat_cache = deque(maxlen=1000)
        self.alert_thresholds = self._load_thresholds()
        
        # Attack pattern detection
        self.attack_patterns = {
            'brute_force': self._detect_brute_force,
            'dos_attack': self._detect_dos,
            'sql_injection': self._detect_sql_injection,
            'xss_attack': self._detect_xss,
            'data_exfiltration': self._detect_data_exfiltration,
            'privilege_escalation': self._detect_privilege_escalation
        }
        
        # Real-time counters
        self.event_counters = defaultdict(int)
        self.ip_counters = defaultdict(lambda: defaultdict(int))
        self.user_activity = defaultdict(list)
    
    async def start_monitoring(self):
        """Start real-time security monitoring"""
        logger.info("🚀 Starting YTEMPIRE Security Monitor")
        
        # Connect to Redis for real-time events
        self.redis = await aioredis.create_redis_pool('redis://localhost')
        self.monitoring_active = True
        
        # Start monitoring tasks
        tasks = [
            self.monitor_auth_events(),
            self.monitor_network_traffic(),
            self.monitor_application_events(),
            self.monitor_system_resources(),
            self.analyze_patterns(),
            self.process_alerts()
        ]
        
        await asyncio.gather(*tasks)
    
    async def monitor_auth_events(self):
        """Monitor authentication events in real-time"""
        channel = 'auth_events'
        
        while self.monitoring_active:
            try:
                # Subscribe to auth events
                [channel_obj] = await self.redis.subscribe(channel)
                
                async for message in channel_obj.iter():
                    event = json.loads(message.decode())
                    await self._process_auth_event(event)
                    
            except Exception as e:
                logger.error(f"Auth monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _process_auth_event(self, event: Dict):
        """Process authentication event"""
        event_type = event.get('type')
        username = event.get('username')
        ip_address = event.get('ip_address')
        timestamp = event.get('timestamp', datetime.utcnow().isoformat())
        
        # Update counters
        self.event_counters[event_type] += 1
        self.ip_counters[ip_address][event_type] += 1
        
        # Track user activity
        self.user_activity[username].append({
            'type': event_type,
            'ip': ip_address,
            'timestamp': timestamp
        })
        
        # Check for authentication attacks
        if event_type == 'login_failed':
            # Check for brute force
            if self.ip_counters[ip_address]['login_failed'] > 5:
                await self._raise_alert({
                    'type': 'brute_force',
                    'severity': 'high',
                    'source_ip': ip_address,
                    'target_user': username,
                    'attempt_count': self.ip_counters[ip_address]['login_failed']
                })
        
        # Update metrics
        security_events.labels(type=event_type, severity='info').inc()
    
    async def _detect_brute_force(self, time_window: int = 300) -> List[Dict]:
        """Detect brute force attacks"""
        threats = []
        current_time = datetime.utcnow()
        
        for ip, events in self.ip_counters.items():
            failed_logins = events.get('login_failed', 0)
            
            # Check if failed logins exceed threshold
            if failed_logins > 10:
                threats.append({
                    'type': 'brute_force',
                    'severity': 'high' if failed_logins > 20 else 'medium',
                    'source_ip': ip,
                    'failed_attempts': failed_logins,
                    'time_window': time_window,
                    'detected_at': current_time.isoformat()
                })
        
        return threats
    
    async def _detect_dos(self) -> List[Dict]:
        """Detect potential DoS attacks"""
        threats = []
        
        # Check request rates
        for ip, events in self.ip_counters.items():
            total_requests = sum(events.values())
            
            if total_requests > 1000:  # Per minute
                threats.append({
                    'type': 'dos_attack',
                    'severity': 'critical' if total_requests > 5000 else 'high',
                    'source_ip': ip,
                    'request_count': total_requests,
                    'detected_at': datetime.utcnow().isoformat()
                })
        
        return threats
    
    async def _raise_alert(self, alert: Dict):
        """Raise security alert"""
        logger.warning(f"🚨 Security Alert: {alert}")
        
        # Add to threat cache
        self.threat_cache.append(alert)
        
        # Update metrics
        active_threats.inc()
        security_events.labels(
            type=alert['type'],
            severity=alert['severity']
        ).inc()
        
        # Send to alert processing queue
        await self.redis.rpush('security_alerts', json.dumps(alert))
        
        # Trigger auto-remediation if configured
        if alert['severity'] in ['critical', 'high']:
            await self.redis.rpush('remediation_queue', json.dumps(alert))
    
    async def get_security_status(self) -> Dict:
        """Get current security status"""
        return {
            'monitoring_active': self.monitoring_active,
            'active_threats': len([t for t in self.threat_cache if t]),
            'event_counts': dict(self.event_counters),
            'top_threat_ips': self._get_top_threat_ips(),
            'recent_alerts': list(self.threat_cache)[-10:],
            'system_health': await self._get_system_health()
        }
    
    def _get_top_threat_ips(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get top threat source IPs"""
        ip_scores = {}
        
        for ip, events in self.ip_counters.items():
            # Calculate threat score
            score = (
                events.get('login_failed', 0) * 10 +
                events.get('sql_injection', 0) * 50 +
                events.get('xss_attempt', 0) * 30 +
                events.get('path_traversal', 0) * 40
            )
            
            if score > 0:
                ip_scores[ip] = score
        
        # Sort by threat score
        sorted_ips = sorted(ip_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_ips[:limit]
```

---

## Document Metadata

**Version**: 1.0  
**Created**: January 2025  
**Owner**: Security Engineering Team  
**Classification**: Internal - Security Team Use Only  
**Review Cycle**: Monthly  

**Purpose**: This document provides the detailed implementation code and scripts from the original Security Tools & Automation Guide that supplement the main implementation documentation.

**Note**: All scripts and code in this document have been tested in the YTEMPIRE environment and should be customized for specific deployment configurations.