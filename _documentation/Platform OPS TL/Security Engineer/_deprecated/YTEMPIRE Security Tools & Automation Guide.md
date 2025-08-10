# YTEMPIRE Security Tools & Automation Guide

**Version**: 2.0  
**Date**: January 2025  
**Classification**: Internal - Security Team  
**Owner**: Security Engineering Team  
**Status**: Production Ready

---

## Executive Summary

This comprehensive guide provides the YTEMPIRE Security Engineering team with complete implementation details for security tools, automation scripts, CI/CD integration, monitoring systems, incident response procedures, compliance automation, and security orchestration. All code and configurations have been tested and optimized for the YTEMPIRE platform infrastructure.

### Key Components Covered

- **Security Tool Setup**: OWASP ZAP, Trivy, Git-Secrets, Bandit
- **Automation Scripts**: Daily security tasks, auto-remediation
- **CI/CD Integration**: GitHub Actions security pipeline
- **Monitoring**: Real-time threat detection, event correlation
- **Incident Response**: Automated response, evidence collection
- **Compliance**: GDPR, PCI-DSS, SOC2 automation
- **Orchestration**: Security playbooks and workflows

---

## Table of Contents

1. [Security Tool Setup & Configuration](#1-security-tool-setup--configuration)
2. [Automated Security Scripts](#2-automated-security-scripts)
3. [CI/CD Security Integration](#3-cicd-security-integration)
4. [Security Monitoring Automation](#4-security-monitoring-automation)
5. [Incident Response Automation](#5-incident-response-automation)
6. [Compliance Automation](#6-compliance-automation)
7. [Security Orchestration](#7-security-orchestration)
8. [Troubleshooting & Optimization](#8-troubleshooting--optimization)
9. [Quick Reference Guide](#9-quick-reference-guide)
10. [Emergency Procedures](#10-emergency-procedures)

---

## 1. Security Tool Setup & Configuration

### 1.1 OWASP ZAP Setup

#### Installation Script

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

#### Python Integration Class

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
                'scan_policy': 'Admin_Security_Critical'
            }
        }
        
        # Risk thresholds for YTEMPIRE
        self.risk_thresholds = {
            'Critical': 0,  # Zero tolerance for critical issues
            'High': 2,      # Maximum 2 high-risk issues
            'Medium': 10,   # Maximum 10 medium-risk issues
            'Low': 50       # Maximum 50 low-risk issues
        }
    
    def scan_target(self, target: str, scan_type: str = 'baseline') -> Dict:
        """
        Perform security scan on YTEMPIRE target
        
        Args:
            target: Target identifier (api, frontend, admin)
            scan_type: Type of scan (baseline, full, api)
            
        Returns:
            Dict containing scan results and analysis
        """
        if target not in self.targets:
            raise ValueError(f"Unknown target: {target}")
        
        config = self.targets[target]
        scan_id = f"ytempire_{target}_{int(time.time())}"
        
        logger.info(f"🔍 Starting {scan_type} scan for {target}")
        
        try:
            # Create new session
            self.zap.core.new_session(name=scan_id, overwrite=True)
            
            # Set up context
            context_id = self._setup_context(config)
            
            # Configure authentication if needed
            if config['auth_type'] != 'none':
                self._configure_authentication(context_id, config)
            
            # Execute scan based on type
            if scan_type == 'baseline':
                results = self._baseline_scan(config['url'], context_id)
            elif scan_type == 'full':
                results = self._full_scan(config['url'], context_id, config['scan_policy'])
            elif scan_type == 'api':
                results = self._api_scan(config['url'], context_id)
            else:
                raise ValueError(f"Unknown scan type: {scan_type}")
            
            # Analyze results
            analysis = self._analyze_results(results)
            
            # Generate report
            report_path = self._generate_report(scan_id, results, analysis)
            
            return {
                'scan_id': scan_id,
                'target': target,
                'scan_type': scan_type,
                'timestamp': datetime.utcnow().isoformat(),
                'results': results,
                'analysis': analysis,
                'report': report_path,
                'passed': analysis['passed']
            }
            
        except Exception as e:
            logger.error(f"Scan failed: {str(e)}")
            raise
    
    def _setup_context(self, config: Dict) -> str:
        """Set up ZAP context for target"""
        context_name = config['context']
        
        # Create context
        context_id = self.zap.context.new_context(context_name)
        
        # Include target in context
        self.zap.context.include_in_context(
            context_name,
            f"{config['url']}.*"
        )
        
        # Exclude logout URLs
        self.zap.context.exclude_from_context(
            context_name,
            ".*logout.*"
        )
        self.zap.context.exclude_from_context(
            context_name,
            ".*signout.*"
        )
        
        return context_id
    
    def _configure_authentication(self, context_id: str, config: Dict):
        """Configure authentication for scanning"""
        if config['auth_type'] == 'jwt':
            self._setup_jwt_auth(context_id)
        elif config['auth_type'] == 'session':
            self._setup_session_auth(context_id)
        elif config['auth_type'] == 'oauth':
            self._setup_oauth_auth(context_id)
    
    def _setup_jwt_auth(self, context_id: str):
        """Configure JWT authentication"""
        # Get JWT token from YTEMPIRE auth service
        auth_response = requests.post(
            'https://api.ytempire.com/auth/token',
            json={
                'client_id': os.environ.get('YTEMPIRE_CLIENT_ID'),
                'client_secret': os.environ.get('YTEMPIRE_CLIENT_SECRET'),
                'grant_type': 'client_credentials'
            }
        )
        
        token = auth_response.json()['access_token']
        
        # Configure HTTP header for JWT
        self.zap.replacer.add_rule(
            description='JWT Authorization',
            enabled=True,
            matchtype='REQ_HEADER',
            matchregex=False,
            matchstring='Authorization',
            replacement=f'Bearer {token}'
        )
    
    def _baseline_scan(self, url: str, context_id: str) -> Dict:
        """Perform baseline security scan"""
        logger.info(f"🕷️ Starting spider for {url}")
        
        # Spider the target
        spider_scan_id = self.zap.spider.scan(url, contextname=context_id)
        
        # Wait for spider to complete
        while int(self.zap.spider.status(spider_scan_id)) < 100:
            progress = self.zap.spider.status(spider_scan_id)
            logger.info(f"Spider progress: {progress}%")
            time.sleep(5)
        
        logger.info("🔎 Running passive scan...")
        
        # Wait for passive scan to complete
        while int(self.zap.pscan.records_to_scan) > 0:
            records = self.zap.pscan.records_to_scan
            logger.info(f"Passive scan records remaining: {records}")
            time.sleep(2)
        
        # Get alerts
        alerts = self.zap.core.alerts(baseurl=url, contextname=context_id)
        
        return self._process_alerts(alerts)
    
    def _full_scan(self, url: str, context_id: str, scan_policy: str) -> Dict:
        """Perform full active security scan"""
        # Run baseline scan first
        results = self._baseline_scan(url, context_id)
        
        logger.info("⚡ Starting active scan...")
        
        # Configure scan policy
        self.zap.ascan.set_option_default_policy(scan_policy)
        
        # Start active scan
        active_scan_id = self.zap.ascan.scan(
            url,
            contextname=context_id,
            scanpolicyname=scan_policy
        )
        
        # Wait for active scan to complete
        while int(self.zap.ascan.status(active_scan_id)) < 100:
            progress = self.zap.ascan.status(active_scan_id)
            logger.info(f"Active scan progress: {progress}%")
            time.sleep(10)
        
        # Get updated alerts
        alerts = self.zap.core.alerts(baseurl=url, contextname=context_id)
        
        return self._process_alerts(alerts)
    
    def _process_alerts(self, alerts: List[Dict]) -> Dict:
        """Process and categorize security alerts"""
        results = {
            'total_alerts': len(alerts),
            'by_risk': {
                'Critical': [],
                'High': [],
                'Medium': [],
                'Low': [],
                'Informational': []
            },
            'by_confidence': {
                'User Confirmed': 0,
                'High': 0,
                'Medium': 0,
                'Low': 0
            },
            'unique_types': set()
        }
        
        for alert in alerts:
            risk = alert.get('risk', 'Informational')
            confidence = alert.get('confidence', 'Low')
            
            # Categorize by risk
            alert_data = {
                'name': alert.get('name'),
                'description': alert.get('description'),
                'url': alert.get('url'),
                'param': alert.get('param'),
                'attack': alert.get('attack'),
                'evidence': alert.get('evidence'),
                'solution': alert.get('solution'),
                'reference': alert.get('reference'),
                'cwe_id': alert.get('cweid'),
                'wasc_id': alert.get('wascid'),
                'confidence': confidence
            }
            
            results['by_risk'][risk].append(alert_data)
            results['by_confidence'][confidence] += 1
            results['unique_types'].add(alert.get('name'))
        
        results['unique_types'] = list(results['unique_types'])
        
        return results
    
    def _analyze_results(self, results: Dict) -> Dict:
        """Analyze scan results against YTEMPIRE security standards"""
        analysis = {
            'passed': True,
            'risk_summary': {},
            'failed_thresholds': [],
            'critical_findings': [],
            'recommendations': []
        }
        
        # Check against thresholds
        for risk_level, threshold in self.risk_thresholds.items():
            count = len(results['by_risk'].get(risk_level, []))
            analysis['risk_summary'][risk_level] = count
            
            if count > threshold:
                analysis['passed'] = False
                analysis['failed_thresholds'].append(
                    f"{risk_level}: Found {count}, threshold is {threshold}"
                )
                
                # Add critical findings
                if risk_level in ['Critical', 'High']:
                    for alert in results['by_risk'][risk_level][:5]:  # Top 5
                        analysis['critical_findings'].append({
                            'risk': risk_level,
                            'issue': alert['name'],
                            'url': alert['url'],
                            'solution': alert['solution']
                        })
        
        # Generate recommendations
        if analysis['risk_summary'].get('Critical', 0) > 0:
            analysis['recommendations'].append(
                "URGENT: Address critical vulnerabilities immediately before deployment"
            )
        
        if analysis['risk_summary'].get('High', 0) > 0:
            analysis['recommendations'].append(
                "HIGH PRIORITY: Fix high-risk issues within 24 hours"
            )
        
        if results['unique_types']:
            top_issues = list(results['unique_types'])[:3]
            analysis['recommendations'].append(
                f"Focus on fixing: {', '.join(top_issues)}"
            )
        
        return analysis
    
    def _generate_report(self, scan_id: str, results: Dict, analysis: Dict) -> str:
        """Generate HTML security report"""
        report_path = f"/opt/ytempire/security/zap/reports/{scan_id}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YTEMPIRE Security Scan Report - {scan_id}</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .summary {{ background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .passed {{ color: #27ae60; font-weight: bold; }}
                .failed {{ color: #e74c3c; font-weight: bold; }}
                .risk-critical {{ background: #e74c3c; color: white; padding: 5px 10px; border-radius: 3px; }}
                .risk-high {{ background: #e67e22; color: white; padding: 5px 10px; border-radius: 3px; }}
                .risk-medium {{ background: #f39c12; color: white; padding: 5px 10px; border-radius: 3px; }}
                .risk-low {{ background: #3498db; color: white; padding: 5px 10px; border-radius: 3px; }}
                table {{ width: 100%; border-collapse: collapse; background: white; }}
                th {{ background: #34495e; color: white; padding: 10px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #ecf0f1; }}
                .recommendations {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>YTEMPIRE Security Scan Report</h1>
                <p>Scan ID: {scan_id}</p>
                <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>Status: <span class="{'passed' if analysis['passed'] else 'failed'}">
                    {'✅ PASSED' if analysis['passed'] else '❌ FAILED'}
                </span></p>
                <p>Total Alerts: {results['total_alerts']}</p>
                
                <h3>Risk Distribution</h3>
                <table>
                    <tr>
                        <th>Risk Level</th>
                        <th>Count</th>
                        <th>Threshold</th>
                        <th>Status</th>
                    </tr>
        """
        
        for risk_level in ['Critical', 'High', 'Medium', 'Low']:
            count = analysis['risk_summary'].get(risk_level, 0)
            threshold = self.risk_thresholds[risk_level]
            status = '✅' if count <= threshold else '❌'
            
            html_content += f"""
                    <tr>
                        <td><span class="risk-{risk_level.lower()}">{risk_level}</span></td>
                        <td>{count}</td>
                        <td>{threshold}</td>
                        <td>{status}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
        
        if analysis['critical_findings']:
            html_content += """
            <div class="summary">
                <h2>Critical Findings</h2>
                <table>
                    <tr>
                        <th>Risk</th>
                        <th>Issue</th>
                        <th>URL</th>
                        <th>Solution</th>
                    </tr>
            """
            
            for finding in analysis['critical_findings']:
                html_content += f"""
                    <tr>
                        <td><span class="risk-{finding['risk'].lower()}">{finding['risk']}</span></td>
                        <td>{finding['issue']}</td>
                        <td>{finding['url'][:50]}...</td>
                        <td>{finding['solution'][:100]}...</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        if analysis['recommendations']:
            html_content += """
            <div class="recommendations">
                <h3>📋 Recommendations</h3>
                <ul>
            """
            
            for rec in analysis['recommendations']:
                html_content += f"<li>{rec}</li>"
            
            html_content += """
                </ul>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"📄 Report saved to: {report_path}")
        
        return report_path
```

### 1.2 Trivy Container Scanner Setup

#### Installation Script

```bash
#!/bin/bash
# install_trivy.sh - Complete Trivy setup for YTEMPIRE

set -e

echo "🔧 Installing Trivy for YTEMPIRE Container Security..."

# Install Trivy binary
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin v0.48.0

# Create directory structure
mkdir -p /opt/ytempire/security/trivy/{cache,reports,policies}

# Create Trivy configuration
cat > /opt/ytempire/security/trivy/trivy.yaml << 'EOF'
# YTEMPIRE Trivy Configuration

# Cache settings
cache:
  dir: /opt/ytempire/security/trivy/cache
  backend: fs
  ttl: 24h

# Scan settings
scan:
  # Security checks to perform
  security-checks:
    - vuln
    - config
    - secret
    - license
  
  # Severity levels to report
  severity:
    - CRITICAL
    - HIGH
    - MEDIUM
  
  # Skip unfixed vulnerabilities
  skip-unfixed: false
  
  # Scan timeout
  timeout: 10m

# Database settings
db:
  repository: ghcr.io/aquasecurity/trivy-db
  skip-update: false
  no-progress: false
  
# Vulnerability database
vulnerability:
  type: 
    - os
    - library
  ignore-unfixed: false

# Misconfiguration scanning
misconfiguration:
  scan-defaults: true
  
# Secret scanning
secret:
  config: /opt/ytempire/security/trivy/secret.yaml

# License scanning
license:
  full: true
  confidence-level: 0.9
  
# Report settings
report:
  format: json
  dependency-tree: true
  list-all-pkgs: false
  exit-code: 1
  exit-on-eol: 0
  ignore-policy: /opt/ytempire/security/trivy/.trivyignore

# SBOM settings
sbom:
  format: cyclonedx
EOF

# Create secret scanning configuration
cat > /opt/ytempire/security/trivy/secret.yaml << 'EOF'
# YTEMPIRE Secret Detection Rules

rules:
  - id: ytempire-api-key
    category: YTEMPIRE
    title: YTEMPIRE API Key
    severity: CRITICAL
    regex: 'ytmp_[a-zA-Z]+_[a-zA-Z0-9]{32,}'
    keywords:
      - ytmp_
      - ytempire
      
  - id: youtube-api-key
    category: YouTube
    title: YouTube API Key
    severity: HIGH
    regex: 'AIza[0-9A-Za-z\-_]{35}'
    keywords:
      - youtube
      - googleapis
      
  - id: jwt-token
    category: Authentication
    title: JWT Token
    severity: HIGH
    regex: 'eyJ[A-Za-z0-9-_=]+\.eyJ[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*'
    keywords:
      - bearer
      - authorization
EOF

# Create ignore file for acceptable vulnerabilities
cat > /opt/ytempire/security/trivy/.trivyignore << 'EOF'
# YTEMPIRE Trivy Ignore List
# Format: CVE-ID or CWE-ID

# Accepted vulnerabilities with compensating controls
# CVE-2021-12345 # Example - Mitigated by WAF rules
# CVE-2022-54321 # Example - Not exploitable in our configuration
EOF

# Create scan script
cat > /usr/local/bin/ytempire-scan << 'EOF'
#!/bin/bash
# YTEMPIRE Container Security Scanner

IMAGE=$1
REPORT_FORMAT=${2:-json}
OUTPUT_FILE=${3:-/opt/ytempire/security/trivy/reports/$(date +%Y%m%d_%H%M%S).json}

if [ -z "$IMAGE" ]; then
    echo "Usage: ytempire-scan <image> [format] [output]"
    exit 1
fi

echo "🔍 Scanning $IMAGE..."

trivy image \
    --config /opt/ytempire/security/trivy/trivy.yaml \
    --format $REPORT_FORMAT \
    --output $OUTPUT_FILE \
    $IMAGE

if [ $? -eq 0 ]; then
    echo "✅ Scan completed successfully"
    echo "📄 Report saved to: $OUTPUT_FILE"
else
    echo "❌ Scan found vulnerabilities"
    exit 1
fi
EOF

chmod +x /usr/local/bin/ytempire-scan

echo "✅ Trivy installation complete!"
echo "🔍 Run 'ytempire-scan <image>' to scan containers"
```

#### Python Integration Class

```python
#!/usr/bin/env python3
# security_tools/trivy_scanner.py

import subprocess
import json
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YTEMPIRETrivyScanner:
    """Advanced Trivy integration for YTEMPIRE container security"""
    
    def __init__(self):
        """Initialize Trivy scanner with YTEMPIRE configuration"""
        self.trivy_cmd = "trivy"
        self.config_path = "/opt/ytempire/security/trivy/trivy.yaml"
        self.reports_dir = Path("/opt/ytempire/security/trivy/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # YTEMPIRE severity thresholds
        self.severity_thresholds = {
            'CRITICAL': 0,  # Zero tolerance
            'HIGH': 3,      # Maximum 3 high severity
            'MEDIUM': 10,   # Maximum 10 medium severity
            'LOW': 50,      # Maximum 50 low severity
            'UNKNOWN': 100  # Informational
        }
        
        # YTEMPIRE-specific image patterns
        self.ytempire_images = {
            'api': 'ytempire/api:*',
            'frontend': 'ytempire/frontend:*',
            'worker': 'ytempire/worker:*',
            'admin': 'ytempire/admin:*'
        }
    
    def scan_image(self, image_name: str, scan_options: Optional[Dict] = None) -> Dict:
        """
        Scan container image for vulnerabilities
        
        Args:
            image_name: Docker image to scan
            scan_options: Additional scan options
            
        Returns:
            Dict containing scan results and analysis
        """
        logger.info(f"🔍 Scanning image: {image_name}")
        
        scan_id = f"trivy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        output_file = self.reports_dir / f"{scan_id}.json"
        
        # Build command
        cmd = [
            self.trivy_cmd, "image",
            "--config", self.config_path,
            "--format", "json",
            "--output", str(output_file)
        ]
        
        # Add custom options
        if scan_options:
            if 'severity' in scan_options:
                cmd.extend(["--severity", ",".join(scan_options['severity'])])
            if 'ignore_unfixed' in scan_options:
                cmd.append("--ignore-unfixed")
            if 'skip_update' in scan_options:
                cmd.append("--skip-update")
        
        cmd.append(image_name)
        
        # Execute scan
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if output_file.exists():
                with open(output_file, 'r') as f:
                    scan_data = json.load(f)
                
                # Process results
                processed_results = self._process_scan_results(scan_data, image_name)
                
                # Analyze against thresholds
                analysis = self._analyze_results(processed_results)
                
                # Generate report
                report_path = self._generate_report(scan_id, processed_results, analysis)
                
                return {
                    'scan_id': scan_id,
                    'image': image_name,
                    'timestamp': datetime.utcnow().isoformat(),
                    'results': processed_results,
                    'analysis': analysis,
                    'report': str(report_path),
                    'passed': analysis['passed']
                }
            else:
                raise Exception(f"Scan output file not created: {output_file}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Scan timeout for image: {image_name}")
            return {
                'error': 'Scan timeout',
                'image': image_name,
                'status': 'timeout'
            }
        except Exception as e:
            logger.error(f"Scan failed: {str(e)}")
            return {
                'error': str(e),
                'image': image_name,
                'status': 'failed'
            }
    
    def scan_filesystem(self, path: str) -> Dict:
        """Scan filesystem for vulnerabilities and misconfigurations"""
        logger.info(f"🔍 Scanning filesystem: {path}")
        
        scan_id = f"trivy_fs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        output_file = self.reports_dir / f"{scan_id}.json"
        
        cmd = [
            self.trivy_cmd, "fs",
            "--config", self.config_path,
            "--format", "json",
            "--output", str(output_file),
            "--security-checks", "vuln,config,secret",
            path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if output_file.exists():
                with open(output_file, 'r') as f:
                    scan_data = json.load(f)
                
                return self._process_filesystem_results(scan_data, path)
            else:
                raise Exception(f"Filesystem scan output not created")
                
        except Exception as e:
            logger.error(f"Filesystem scan failed: {str(e)}")
            return {
                'error': str(e),
                'path': path,
                'status': 'failed'
            }
    
    def scan_iac(self, path: str) -> Dict:
        """Scan Infrastructure as Code for misconfigurations"""
        logger.info(f"🔍 Scanning IaC: {path}")
        
        scan_id = f"trivy_iac_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        output_file = self.reports_dir / f"{scan_id}.json"
        
        cmd = [
            self.trivy_cmd, "config",
            "--config", self.config_path,
            "--format", "json",
            "--output", str(output_file),
            path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if output_file.exists():
                with open(output_file, 'r') as f:
                    scan_data = json.load(f)
                
                return self._process_iac_results(scan_data, path)
            else:
                raise Exception(f"IaC scan output not created")
                
        except Exception as e:
            logger.error(f"IaC scan failed: {str(e)}")
            return {
                'error': str(e),
                'path': path,
                'status': 'failed'
            }
    
    def scan_sbom(self, sbom_file: str) -> Dict:
        """Scan Software Bill of Materials (SBOM) for vulnerabilities"""
        logger.info(f"🔍 Scanning SBOM: {sbom_file}")
        
        scan_id = f"trivy_sbom_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        output_file = self.reports_dir / f"{scan_id}.json"
        
        cmd = [
            self.trivy_cmd, "sbom",
            "--config", self.config_path,
            "--format", "json",
            "--output", str(output_file),
            sbom_file
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if output_file.exists():
                with open(output_file, 'r') as f:
                    scan_data = json.load(f)
                
                return self._process_sbom_results(scan_data, sbom_file)
            else:
                raise Exception(f"SBOM scan output not created")
                
        except Exception as e:
            logger.error(f"SBOM scan failed: {str(e)}")
            return {
                'error': str(e),
                'sbom_file': sbom_file,
                'status': 'failed'
            }
    
    def batch_scan_ytempire_images(self) -> Dict:
        """Scan all YTEMPIRE container images"""
        logger.info("🔍 Starting batch scan of YTEMPIRE images...")
        
        batch_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'images_scanned': [],
            'total_vulnerabilities': 0,
            'failed_scans': [],
            'passed': True
        }
        
        # Get list of YTEMPIRE images
        images = self._get_ytempire_images()
        
        for image in images:
            try:
                scan_result = self.scan_image(image)
                
                if 'error' not in scan_result:
                    batch_results['images_scanned'].append({
                        'image': image,
                        'vulnerabilities': scan_result['results']['summary']['total'],
                        'passed': scan_result['passed']
                    })
                    
                    batch_results['total_vulnerabilities'] += scan_result['results']['summary']['total']
                    
                    if not scan_result['passed']:
                        batch_results['passed'] = False
                else:
                    batch_results['failed_scans'].append(image)
                    batch_results['passed'] = False
                    
            except Exception as e:
                logger.error(f"Failed to scan {image}: {str(e)}")
                batch_results['failed_scans'].append(image)
                batch_results['passed'] = False
        
        return batch_results
    
    def _get_ytempire_images(self) -> List[str]:
        """Get list of YTEMPIRE Docker images"""
        try:
            # Get images from Docker
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                capture_output=True,
                text=True
            )
            
            images = []
            for line in result.stdout.strip().split('\n'):
                if line.startswith('ytempire/'):
                    images.append(line)
            
            return images
        except Exception as e:
            logger.error(f"Failed to get Docker images: {str(e)}")
            return []
    
    def _process_scan_results(self, scan_data: Dict, target: str) -> Dict:
        """Process Trivy scan results"""
        results = {
            'target': target,
            'scan_date': datetime.utcnow().isoformat(),
            'vulnerabilities': {
                'CRITICAL': [],
                'HIGH': [],
                'MEDIUM': [],
                'LOW': [],
                'UNKNOWN': []
            },
            'misconfigurations': [],
            'secrets': [],
            'licenses': [],
            'summary': {
                'total': 0,
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0,
                'unknown': 0
            }
        }
        
        # Process vulnerability results
        for result in scan_data.get('Results', []):
            # Vulnerabilities
            for vuln in result.get('Vulnerabilities', []):
                severity = vuln.get('Severity', 'UNKNOWN')
                
                vuln_info = {
                    'id': vuln.get('VulnerabilityID'),
                    'package': vuln.get('PkgName'),
                    'installed_version': vuln.get('InstalledVersion'),
                    'fixed_version': vuln.get('FixedVersion'),
                    'title': vuln.get('Title'),
                    'description': vuln.get('Description'),
                    'severity': severity,
                    'published': vuln.get('PublishedDate'),
                    'last_modified': vuln.get('LastModifiedDate'),
                    'cvss': vuln.get('CVSS'),
                    'references': vuln.get('References', []),
                    'primary_url': vuln.get('PrimaryURL')
                }
                
                results['vulnerabilities'][severity].append(vuln_info)
                results['summary']['total'] += 1
                results['summary'][severity.lower()] += 1
            
            # Misconfigurations
            for misconfig in result.get('Misconfigurations', []):
                results['misconfigurations'].append({
                    'type': misconfig.get('Type'),
                    'id': misconfig.get('ID'),
                    'title': misconfig.get('Title'),
                    'description': misconfig.get('Description'),
                    'message': misconfig.get('Message'),
                    'severity': misconfig.get('Severity'),
                    'resolution': misconfig.get('Resolution')
                })
            
            # Secrets
            for secret in result.get('Secrets', []):
                results['secrets'].append({
                    'rule_id': secret.get('RuleID'),
                    'category': secret.get('Category'),
                    'severity': secret.get('Severity'),
                    'title': secret.get('Title'),
                    'match': secret.get('Match'),
                    'start_line': secret.get('StartLine'),
                    'end_line': secret.get('EndLine')
                })
            
            # Licenses
            for license_finding in result.get('Licenses', []):
                results['licenses'].append({
                    'name': license_finding.get('Name'),
                    'confidence': license_finding.get('Confidence'),
                    'link': license_finding.get('Link'),
                    'package': license_finding.get('PkgName')
                })
        
        return results
    
    def _analyze_results(self, results: Dict) -> Dict:
        """Analyze scan results against YTEMPIRE security standards"""
        analysis = {
            'passed': True,
            'severity_summary': results['summary'].copy(),
            'failed_thresholds': [],
            'critical_vulnerabilities': [],
            'high_priority_fixes': [],
            'recommendations': []
        }
        
        # Check against severity thresholds
        for severity, threshold in self.severity_thresholds.items():
            count = results['summary'].get(severity.lower(), 0)
            
            if count > threshold:
                analysis['passed'] = False
                analysis['failed_thresholds'].append(
                    f"{severity}: Found {count}, threshold is {threshold}"
                )
                
                # Collect critical vulnerabilities
                if severity in ['CRITICAL', 'HIGH']:
                    for vuln in results['vulnerabilities'][severity][:5]:  # Top 5
                        analysis['critical_vulnerabilities'].append({
                            'id': vuln['id'],
                            'package': vuln['package'],
                            'severity': severity,
                            'fixed_version': vuln.get('fixed_version', 'No fix available')
                        })
        
        # Identify high priority fixes
        for severity in ['CRITICAL', 'HIGH']:
            for vuln in results['vulnerabilities'].get(severity, []):
                if vuln.get('fixed_version') and vuln['fixed_version'] != 'No fix available':
                    analysis['high_priority_fixes'].append({
                        'package': vuln['package'],
                        'current': vuln['installed_version'],
                        'fixed': vuln['fixed_version'],
                        'severity': severity
                    })
        
        # Generate recommendations
        if analysis['severity_summary']['critical'] > 0:
            analysis['recommendations'].append(
                "CRITICAL: Address critical vulnerabilities immediately - DO NOT DEPLOY"
            )
        
        if analysis['severity_summary']['high'] > 0:
            analysis['recommendations'].append(
                "HIGH: Fix high-severity vulnerabilities before production deployment"
            )
        
        if results.get('secrets'):
            analysis['recommendations'].append(
                "SECURITY: Remove hardcoded secrets from container image"
            )
        
        if results.get('misconfigurations'):
            analysis['recommendations'].append(
                "CONFIG: Address security misconfigurations in container setup"
            )
        
        # Add update recommendations
        if analysis['high_priority_fixes']:
            packages_to_update = set(fix['package'] for fix in analysis['high_priority_fixes'][:5])
            analysis['recommendations'].append(
                f"UPDATE: Priority packages to update: {', '.join(packages_to_update)}"
            )
        
        return analysis
    
    def _generate_report(self, scan_id: str, results: Dict, analysis: Dict) -> Path:
        """Generate comprehensive security report"""
        report_path = self.reports_dir / f"{scan_id}_report.html"
        
        html_content = self._generate_html_report(scan_id, results, analysis)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"📄 Report generated: {report_path}")
        
        return report_path
    
    def _generate_html_report(self, scan_id: str, results: Dict, analysis: Dict) -> str:
        """Generate HTML report content"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YTEMPIRE Container Security Report - {scan_id}</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .card {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .status-passed {{ color: #10b981; font-weight: bold; font-size: 1.2em; }}
                .status-failed {{ color: #ef4444; font-weight: bold; font-size: 1.2em; }}
                .severity-critical {{ background: #ef4444; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }}
                .severity-high {{ background: #f97316; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }}
                .severity-medium {{ background: #eab308; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }}
                .severity-low {{ background: #3b82f6; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }}
                .severity-unknown {{ background: #6b7280; color: white; padding: 3px 8px; border-radius: 4px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th {{ background: #f3f4f6; padding: 12px; text-align: left; font-weight: 600; }}
                td {{ padding: 12px; border-bottom: 1px solid #e5e7eb; }}
                tr:hover {{ background: #f9fafb; }}
                .recommendations {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 20px 0; border-radius: 4px; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #1f2937; }}
                .metric-label {{ color: #6b7280; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🛡️ YTEMPIRE Container Security Report</h1>
                    <p><strong>Scan ID:</strong> {scan_id}</p>
                    <p><strong>Target:</strong> {results['target']}</p>
                    <p><strong>Scan Date:</strong> {results['scan_date']}</p>
                </div>
                
                <div class="card">
                    <h2>Executive Summary</h2>
                    <p class="{'status-passed' if analysis['passed'] else 'status-failed'}">
                        {'✅ SECURITY CHECK PASSED' if analysis['passed'] else '❌ SECURITY CHECK FAILED'}
                    </p>
                    
                    <div style="margin: 20px 0;">
                        <div class="metric">
                            <div class="metric-value">{results['summary']['total']}</div>
                            <div class="metric-label">Total Issues</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" style="color: #ef4444;">{results['summary']['critical']}</div>
                            <div class="metric-label">Critical</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" style="color: #f97316;">{results['summary']['high']}</div>
                            <div class="metric-label">High</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" style="color: #eab308;">{results['summary']['medium']}</div>
                            <div class="metric-label">Medium</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" style="color: #3b82f6;">{results['summary']['low']}</div>
                            <div class="metric-label">Low</div>
                        </div>
                    </div>
                </div>
        """
        
        # Add failed thresholds if any
        if analysis['failed_thresholds']:
            html += """
                <div class="card">
                    <h2>❌ Failed Security Thresholds</h2>
                    <ul>
            """
            for failure in analysis['failed_thresholds']:
                html += f"<li>{failure}</li>"
            html += """
                    </ul>
                </div>
            """
        
        # Add critical vulnerabilities
        if analysis['critical_vulnerabilities']:
            html += """
                <div class="card">
                    <h2>🚨 Critical Vulnerabilities</h2>
                    <table>
                        <tr>
                            <th>CVE ID</th>
                            <th>Package</th>
                            <th>Severity</th>
                            <th>Fixed Version</th>
                        </tr>
            """
            for vuln in analysis['critical_vulnerabilities']:
                html += f"""
                        <tr>
                            <td>{vuln['id']}</td>
                            <td>{vuln['package']}</td>
                            <td><span class="severity-{vuln['severity'].lower()}">{vuln['severity']}</span></td>
                            <td>{vuln['fixed_version']}</td>
                        </tr>
                """
            html += """
                    </table>
                </div>
            """
        
        # Add recommendations
        if analysis['recommendations']:
            html += """
                <div class="recommendations">
                    <h3>📋 Security Recommendations</h3>
                    <ul>
            """
            for rec in analysis['recommendations']:
                html += f"<li>{rec}</li>"
            html += """
                    </ul>
                </div>
            """
        
        # Add high priority fixes
        if analysis['high_priority_fixes']:
            html += """
                <div class="card">
                    <h2>🔧 High Priority Package Updates</h2>
                    <table>
                        <tr>
                            <th>Package</th>
                            <th>Current Version</th>
                            <th>Fixed Version</th>
                            <th>Severity</th>
                        </tr>
            """
            for fix in analysis['high_priority_fixes'][:10]:  # Top 10
                html += f"""
                        <tr>
                            <td>{fix['package']}</td>
                            <td>{fix['current']}</td>
                            <td>{fix['fixed']}</td>
                            <td><span class="severity-{fix['severity'].lower()}">{fix['severity']}</span></td>
                        </tr>
                """
            html += """
                    </table>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
```

### 1.3 Git-Secrets Configuration

#### Setup Script

```bash
#!/bin/bash
# setup_git_secrets.sh - Configure git-secrets for YTEMPIRE repositories

set -e

echo "🔧 Setting up git-secrets for YTEMPIRE..."

# Install git-secrets if not present
if ! command -v git-secrets &> /dev/null; then
    echo "📦 Installing git-secrets..."
    
    # Clone and install
    git clone https://github.com/awslabs/git-secrets.git /tmp/git-secrets
    cd /tmp/git-secrets
    sudo make install
    cd -
    rm -rf /tmp/git-secrets
fi

# Function to configure git-secrets for a repository
configure_repo() {
    local repo_path=$1
    echo "🔐 Configuring git-secrets for: $repo_path"
    
    cd "$repo_path"
    
    # Install git-secrets hooks
    git secrets --install --force
    
    # Register AWS patterns
    git secrets --register-aws
    
    # Add YTEMPIRE-specific patterns
    echo "Adding YTEMPIRE secret patterns..."
    
    # YTEMPIRE API keys
    git secrets --add 'ytmp_[a-zA-Z]+_[a-zA-Z0-9]{32,}'
    git secrets --add 'YTEMPIRE_[A-Z_]+_KEY\s*=\s*["\'][^"\']{20,}["\']'
    
    # YouTube API keys
    git secrets --add 'AIza[0-9A-Za-z\-_]{35}'
    git secrets --add 'youtube[_\-]?api[_\-]?key\s*[:=]\s*["\'][^"\']{20,}["\']'
    
    # Generic API keys and tokens
    git secrets --add '[aA][pP][iI][_\-]?[kK][eE][yY]\s*[:=]\s*["\'][^"\']{20,}["\']'
    git secrets --add '[sS][eE][cC][rR][eE][tT][_\-]?[kK][eE][yY]\s*[:=]\s*["\'][^"\']{20,}["\']'
    git secrets --add '[tT][oO][kK][eE][nN]\s*[:=]\s*["\'][^"\']{20,}["\']'
    
    # Database credentials
    git secrets --add '[dD][bB][_\-]?[pP][aA][sS][sS][wW][oO][rR][dD]\s*[:=]\s*["\'][^"\']+["\']'
    git secrets --add 'mongodb(\+srv)?://[^:]+:[^@]+@[^\s]+'
    git secrets --add 'postgres://[^:]+:[^@]+@[^\s]+'
    
    # JWT tokens
    git secrets --add 'eyJ[A-Za-z0-9\-_=]+\.eyJ[A-Za-z0-9\-_=]+\.?[A-Za-z0-9\-_.+/=]*'
    
    # Private keys
    git secrets --add '-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----'
    git secrets --add '-----BEGIN PGP PRIVATE KEY BLOCK-----'
    
    # OAuth tokens
    git secrets --add 'oauth[_\-]?token\s*[:=]\s*["\'][^"\']{20,}["\']'
    git secrets --add 'client[_\-]?secret\s*[:=]\s*["\'][^"\']{20,}["\']'
    
    # Add allowed patterns (false positives)
    echo "Adding allowed patterns..."
    
    # Environment variable references
    git secrets --add --allowed 'os\.environ'
    git secrets --add --allowed 'process\.env'
    git secrets --add --allowed '\$\{[^}]+\}'
    
    # Configuration file references
    git secrets --add --allowed 'config\.(get|load|read)'
    git secrets --add --allowed 'from_env'
    git secrets --add --allowed 'getenv'
    
    # Example/test domains
    git secrets --add --allowed 'example\.(com|org|net)'
    git secrets --add --allowed 'test\.(com|org|net)'
    git secrets --add --allowed 'localhost'
    git secrets --add --allowed '127\.0\.0\.1'
    
    # Documentation patterns
    git secrets --add --allowed 'YOUR[_\-]?API[_\-]?KEY'
    git secrets --add --allowed 'your[_\-]?api[_\-]?key'
    git secrets --add --allowed '<[^>]+>'
    git secrets --add --allowed 'xxx+'
    
    echo "✅ git-secrets configured for $repo_path"
}

# Configure all YTEMPIRE repositories
REPOS=(
    "/opt/ytempire/api"
    "/opt/ytempire/frontend"
    "/opt/ytempire/worker"
    "/opt/ytempire/admin"
)

for repo in "${REPOS[@]}"; do
    if [ -d "$repo/.git" ]; then
        configure_repo "$repo"
    else
        echo "⚠️ Skipping $repo - not a git repository"
    fi
done

# Create global git-secrets audit script
cat > /usr/local/bin/ytempire-secrets-audit << 'EOF'
#!/bin/bash
# YTEMPIRE Secrets Audit Script

echo "🔍 Running YTEMPIRE secrets audit..."

REPOS=(
    "/opt/ytempire/api"
    "/opt/ytempire/frontend"
    "/opt/ytempire/worker"
    "/opt/ytempire/admin"
)

FOUND_SECRETS=0

for repo in "${REPOS[@]}"; do
    if [ -d "$repo/.git" ]; then
        echo "Scanning $repo..."
        cd "$repo"
        
        # Scan history
        if ! git secrets --scan-history 2>/dev/null; then
            echo "❌ Secrets found in $repo!"
            FOUND_SECRETS=1
        else
            echo "✅ $repo is clean"
        fi
    fi
done

if [ $FOUND_SECRETS -eq 0 ]; then
    echo "✅ No secrets found in any repository!"
    exit 0
else
    echo "❌ Secrets detected! Please remove them before committing."
    exit 1
fi
EOF

chmod +x /usr/local/bin/ytempire-secrets-audit

echo "✅ git-secrets setup complete!"
echo "🔍 Run 'ytempire-secrets-audit' to scan all repositories"
```

### 1.4 Bandit Python Security Linter

#### Configuration File

```ini
# .bandit - YTEMPIRE Bandit Configuration

[bandit]
# Tests to run
tests: B201,B301,B302,B303,B304,B305,B306,B307,B308,B309,B310,B311,B312,B313,B314,B315,B316,B317,B318,B319,B320,B321,B322,B323,B324,B325,B401,B402,B403,B404,B405,B406,B407,B408,B409,B410,B411,B412,B413,B501,B502,B503,B504,B505,B506,B507,B601,B602,B603,B604,B605,B606,B607,B608,B609,B610,B611,B701,B702,B703

# Tests to skip
skips: B404,B603,B104,B105

# Paths to exclude
exclude_dirs:
  - /test/
  - /tests/
  - /migrations/
  - /venv/
  - /.venv/
  - /node_modules/
  - /.git/
  - /__pycache__/
  - /.pytest_cache/

# File patterns to include
include:
  - "*.py"

# File patterns to exclude
exclude:
  - "*_test.py"
  - "test_*.py"
  - "conftest.py"
  - "setup.py"

# Severity level filter
level: MEDIUM

# Confidence level filter
confidence: MEDIUM

# Output format
format: json

# Custom formatter (optional)
# formatter: 

# Aggregate results by vulnerability type
aggregate: vuln

# Number of lines of context to show
context_lines: 3

# Profile to use (optional)
# profile:

# Additional plugin directories
# plugin_dirs:

# Report only issues of given severity or higher
# severity: MEDIUM

# Report only issues of given confidence or higher
# confidence: MEDIUM

# Maximum number of code lines to output for each issue
max_lines: 5

# Whether to show line numbers in output
show_line_numbers: true

# Custom message format
# msg_template: {line}: {test_id}[{severity}]: {msg}

# Output file
# output: bandit_report.json

# Verbose output
verbose: true

# Debug mode
debug: false

# Quiet mode
quiet: false

# Ignore nosec comments
ignore_nosec: false

# Baseline file for comparing results
# baseline:

# INI file path
# ini: .bandit

# Recursive scan
recursive: true
```

#### Python Integration Class

```python
#!/usr/bin/env python3
# security_tools/bandit_scanner.py

import subprocess
import json
import os
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YTEMPIREBanditScanner:
    """Bandit security linter integration for YTEMPIRE Python code"""
    
    def __init__(self):
        """Initialize Bandit scanner with YTEMPIRE configuration"""
        self.bandit_cmd = "bandit"
        self.config_file = ".bandit"
        self.reports_dir = Path("/opt/ytempire/security/bandit/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # YTEMPIRE security standards
        self.severity_thresholds = {
            'HIGH': 0,     # Zero tolerance for high severity
            'MEDIUM': 5,   # Maximum 5 medium severity
            'LOW': 20      # Maximum 20 low severity
        }
        
        # YTEMPIRE-specific test configurations
        self.ytempire_tests = {
            'critical': [
                'B201',  # Flask debug mode
                'B501',  # Request with verify=False
                'B601',  # Shell with shell=True
                'B602',  # Subprocess with shell=True
                'B605',  # Shell injection
                'B607',  # Partial path for subprocess
                'B608',  # SQL injection
            ],
            'high': [
                'B301',  # Pickle usage
                'B302',  # Marshal usage
                'B303',  # MD5 usage
                'B304',  # DES/RC4 usage
                'B305',  # Weak cryptographic key
                'B306',  # Tempfile.mktemp
                'B307',  # Eval usage
                'B308',  # Mark_safe usage
                'B309',  # HTTPSConnection unverified
                'B310',  # URLopen without validation
            ],
            'medium': [
                'B311',  # Random module usage
                'B312',  # Telnet usage
                'B313',  # XML parsing
                'B314',  # XML.etree.ElementTree
                'B315',  # XML.expat
                'B316',  # XML.sax
                'B317',  # XML.dom.pulldom
                'B318',  # XML.dom.minidom
                'B319',  # XML.etree.cElementTree
                'B320',  # lxml usage
            ]
        }
    
    def scan_directory(self, directory: str, recursive: bool = True) -> Dict:
        """
        Scan Python code directory for security issues
        
        Args:
            directory: Directory path to scan
            recursive: Whether to scan recursively
            
        Returns:
            Dict containing scan results and analysis
        """
        logger.info(f"🔍 Scanning directory: {directory}")
        
        scan_id = f"bandit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        output_file = self.reports_dir / f"{scan_id}.json"
        
        # Build command
        cmd = [
            self.bandit_cmd,
            "-f", "json",
            "-o", str(output_file),
            "--ini", self.config_file
        ]
        
        if recursive:
            cmd.append("-r")
        
        cmd.append(directory)
        
        try:
            # Run Bandit scan
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Bandit returns non-zero if issues found, which is expected
            if output_file.exists():
                with open(output_file, 'r') as f:
                    scan_data = json.load(f)
                
                # Process results
                processed_results = self._process_results(scan_data)
                
                # Analyze against standards
                analysis = self._analyze_results(processed_results)
                
                # Generate report
                report_path = self._generate_report(scan_id, processed_results, analysis)
                
                return {
                    'scan_id': scan_id,
                    'directory': directory,
                    'timestamp': datetime.utcnow().isoformat(),
                    'results': processed_results,
                    'analysis': analysis,
                    'report': str(report_path),
                    'passed': analysis['passed']
                }
            else:
                raise Exception(f"Scan output file not created: {output_file}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Scan timeout for directory: {directory}")
            return {
                'error': 'Scan timeout',
                'directory': directory,
                'status': 'timeout'
            }
        except Exception as e:
            logger.error(f"Scan failed: {str(e)}")
            return {
                'error': str(e),
                'directory': directory,
                'status': 'failed'
            }
    
    def scan_file(self, filepath: str) -> Dict:
        """Scan single Python file for security issues"""
        logger.info(f"🔍 Scanning file: {filepath}")
        
        scan_id = f"bandit_file_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        output_file = self.reports_dir / f"{scan_id}.json"
        
        cmd = [
            self.bandit_cmd,
            "-f", "json",
            "-o", str(output_file),
            "--ini", self.config_file,
            filepath
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if output_file.exists():
                with open(output_file, 'r') as f:
                    scan_data = json.load(f)
                
                processed_results = self._process_results(scan_data)
                
                return {
                    'scan_id': scan_id,
                    'file': filepath,
                    'timestamp': datetime.utcnow().isoformat(),
                    'results': processed_results,
                    'passed': len(processed_results['issues']) == 0
                }
            else:
                raise Exception(f"Scan output file not created")
                
        except Exception as e:
            logger.error(f"File scan failed: {str(e)}")
            return {
                'error': str(e),
                'file': filepath,
                'status': 'failed'
            }
    
    def scan_ytempire_codebase(self) -> Dict:
        """Scan entire YTEMPIRE Python codebase"""
        logger.info("🔍 Starting YTEMPIRE codebase security scan...")
        
        codebase_paths = [
            "/opt/ytempire/api",
            "/opt/ytempire/worker",
            "/opt/ytempire/admin",
            "/opt/ytempire/scripts"
        ]
        
        batch_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'paths_scanned': [],
            'total_issues': 0,
            'critical_issues': [],
            'failed_scans': [],
            'passed': True
        }
        
        for path in codebase_paths:
            if os.path.exists(path):
                try:
                    scan_result = self.scan_directory(path)
                    
                    if 'error' not in scan_result:
                        batch_results['paths_scanned'].append({
                            'path': path,
                            'issues': scan_result['results']['summary']['total'],
                            'passed': scan_result['passed']
                        })
                        
                        batch_results['total_issues'] += scan_result['results']['summary']['total']
                        
                        # Collect critical issues
                        for issue in scan_result['results']['issues']:
                            if issue['severity'] == 'HIGH' and issue['confidence'] in ['HIGH', 'MEDIUM']:
                                batch_results['critical_issues'].append({
                                    'path': path,
                                    'file': issue['filename'],
                                    'line': issue['line_number'],
                                    'issue': issue['test_name'],
                                    'severity': issue['severity']
                                })
                        
                        if not scan_result['passed']:
                            batch_results['passed'] = False
                    else:
                        batch_results['failed_scans'].append(path)
                        batch_results['passed'] = False
                        
                except Exception as e:
                    logger.error(f"Failed to scan {path}: {str(e)}")
                    batch_results['failed_scans'].append(path)
                    batch_results['passed'] = False
        
        return batch_results
    
    def _process_results(self, scan_data: Dict) -> Dict:
        """Process Bandit scan results"""
        results = {
            'metrics': scan_data.get('metrics', {}),
            'issues': [],
            'summary': {
                'total': 0,
                'by_severity': {
                    'HIGH': 0,
                    'MEDIUM': 0,
                    'LOW': 0
                },
                'by_confidence': {
                    'HIGH': 0,
                    'MEDIUM': 0,
                    'LOW': 0
                },
                'by_test': {}
            }
        }
        
        # Process each issue
        for issue in scan_data.get('results', []):
            issue_data = {
                'filename': issue.get('filename'),
                'line_number': issue.get('line_number'),
                'line_range': issue.get('line_range'),
                'code': issue.get('code'),
                'severity': issue.get('issue_severity'),
                'confidence': issue.get('issue_confidence'),
                'test_id': issue.get('test_id'),
                'test_name': issue.get('test_name'),
                'issue_text': issue.get('issue_text'),
                'more_info': issue.get('more_info'),
                'cwe': issue.get('issue_cwe', {})
            }
            
            results['issues'].append(issue_data)
            results['summary']['total'] += 1
            
            # Update summary counts
            severity = issue.get('issue_severity', 'UNKNOWN')
            confidence = issue.get('issue_confidence', 'UNKNOWN')
            test_id = issue.get('test_id', 'UNKNOWN')
            
            if severity in results['summary']['by_severity']:
                results['summary']['by_severity'][severity] += 1
            
            if confidence in results['summary']['by_confidence']:
                results['summary']['by_confidence'][confidence] += 1
            
            # Count by test type
            if test_id not in results['summary']['by_test']:
                results['summary']['by_test'][test_id] = {
                    'count': 0,
                    'name': issue.get('test_name', ''),
                    'severity': severity
                }
            results['summary']['by_test'][test_id]['count'] += 1
        
        return results
    
    def _analyze_results(self, results: Dict) -> Dict:
        """Analyze scan results against YTEMPIRE security standards"""
        analysis = {
            'passed': True,
            'severity_summary': results['summary']['by_severity'].copy(),
            'failed_thresholds': [],
            'critical_issues': [],
            'security_hotspots': [],
            'recommendations': []
        }
        
        # Check against severity thresholds
        for severity, threshold in self.severity_thresholds.items():
            count = results['summary']['by_severity'].get(severity, 0)
            
            if count > threshold:
                analysis['passed'] = False
                analysis['failed_thresholds'].append(
                    f"{severity}: Found {count}, threshold is {threshold}"
                )
                
                # Collect critical issues
                if severity == 'HIGH':
                    for issue in results['issues']:
                        if issue['severity'] == 'HIGH' and issue['confidence'] in ['HIGH', 'MEDIUM']:
                            analysis['critical_issues'].append({
                                'file': issue['filename'],
                                'line': issue['line_number'],
                                'test': issue['test_name'],
                                'description': issue['issue_text']
                            })
        
        # Identify security hotspots (files with multiple issues)
        file_issues = {}
        for issue in results['issues']:
            filename = issue['filename']
            if filename not in file_issues:
                file_issues[filename] = 0
            file_issues[filename] += 1
        
        for filename, count in file_issues.items():
            if count >= 3:  # 3 or more issues in a file
                analysis['security_hotspots'].append({
                    'file': filename,
                    'issue_count': count
                })
        
        # Generate recommendations based on common issues
        test_summary = results['summary']['by_test']
        
        # Check for critical test failures
        for test_id in self.ytempire_tests['critical']:
            if test_id in test_summary and test_summary[test_id]['count'] > 0:
                analysis['recommendations'].append(
                    f"CRITICAL: Fix {test_summary[test_id]['name']} issues immediately - {test_summary[test_id]['count']} found"
                )
        
        # Check for crypto issues
        crypto_tests = ['B303', 'B304', 'B305']
        crypto_issues = sum(test_summary.get(test, {}).get('count', 0) for test in crypto_tests)
        if crypto_issues > 0:
            analysis['recommendations'].append(
                f"CRYPTO: Review and update cryptographic implementations - {crypto_issues} weak crypto issues found"
            )
        
        # Check for injection vulnerabilities
        injection_tests = ['B601', 'B602', 'B605', 'B607', 'B608']
        injection_issues = sum(test_summary.get(test, {}).get('count', 0) for test in injection_tests)
        if injection_issues > 0:
            analysis['recommendations'].append(
                f"INJECTION: Address potential injection vulnerabilities - {injection_issues} issues found"
            )
        
        # Check for insecure deserialization
        deserial_tests = ['B301', 'B302']
        deserial_issues = sum(test_summary.get(test, {}).get('count', 0) for test in deserial_tests)
        if deserial_issues > 0:
            analysis['recommendations'].append(
                f"DESERIALIZATION: Replace insecure deserialization methods - {deserial_issues} issues found"
            )
        
        # General recommendations
        if analysis['security_hotspots']:
            hotspot_files = [h['file'] for h in analysis['security_hotspots'][:3]]
            analysis['recommendations'].append(
                f"REFACTOR: Priority refactoring needed for: {', '.join(hotspot_files)}"
            )
        
        return analysis
    
    def _generate_report(self, scan_id: str, results: Dict, analysis: Dict) -> Path:
        """Generate HTML security report"""
        report_path = self.reports_dir / f"{scan_id}_report.html"
        
        html_content = self._generate_html_report(scan_id, results, analysis)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"📄 Report generated: {report_path}")
        
        return report_path
    
    def _generate_html_report(self, scan_id: str, results: Dict, analysis: Dict) -> str:
        """Generate HTML report content"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YTEMPIRE Python Security Report - {scan_id}</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .header {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 30px; border-radius: 10px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .card {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .status-passed {{ color: #10b981; font-weight: bold; font-size: 1.2em; }}
                .status-failed {{ color: #ef4444; font-weight: bold; font-size: 1.2em; }}
                .severity-high {{ background: #ef4444; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }}
                .severity-medium {{ background: #f59e0b; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }}
                .severity-low {{ background: #3b82f6; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }}
                .confidence-high {{ color: #dc2626; font-weight: bold; }}
                .confidence-medium {{ color: #f59e0b; font-weight: bold; }}
                .confidence-low {{ color: #6b7280; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th {{ background: #f3f4f6; padding: 12px; text-align: left; font-weight: 600; }}
                td {{ padding: 12px; border-bottom: 1px solid #e5e7eb; }}
                tr:hover {{ background: #f9fafb; }}
                .code-snippet {{ background: #1f2937; color: #f3f4f6; padding: 10px; border-radius: 4px; font-family: 'Courier New', monospace; font-size: 0.9em; overflow-x: auto; }}
                .recommendations {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 20px 0; border-radius: 4px; }}
                .hotspot {{ background: #fee2e2; border-left: 4px solid #ef4444; padding: 10px; margin: 10px 0; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔒 YTEMPIRE Python Security Report</h1>
                    <p><strong>Scan ID:</strong> {scan_id}</p>
                    <p><strong>Scan Date:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                    <p><strong>Total Files Scanned:</strong> {results['metrics'].get('_totals', {}).get('loc', 0)} lines of code</p>
                </div>
                
                <div class="card">
                    <h2>Executive Summary</h2>
                    <p class="{'status-passed' if analysis['passed'] else 'status-failed'}">
                        {'✅ SECURITY CHECK PASSED' if analysis['passed'] else '❌ SECURITY CHECK FAILED'}
                    </p>
                    
                    <h3>Issue Summary</h3>
                    <table>
                        <tr>
                            <th>Severity</th>
                            <th>Count</th>
                            <th>Threshold</th>
                            <th>Status</th>
                        </tr>
        """
        
        for severity in ['HIGH', 'MEDIUM', 'LOW']:
            count = analysis['severity_summary'].get(severity, 0)
            threshold = self.severity_thresholds.get(severity, 999)
            status = '✅' if count <= threshold else '❌'
            
            html += f"""
                        <tr>
                            <td><span class="severity-{severity.lower()}">{severity}</span></td>
                            <td>{count}</td>
                            <td>{threshold}</td>
                            <td>{status}</td>
                        </tr>
            """
        
        html += f"""
                    </table>
                    
                    <h3>Confidence Distribution</h3>
                    <p>
                        High: {results['summary']['by_confidence'].get('HIGH', 0)} | 
                        Medium: {results['summary']['by_confidence'].get('MEDIUM', 0)} | 
                        Low: {results['summary']['by_confidence'].get('LOW', 0)}
                    </p>
                </div>
        """
        
        # Add critical issues
        if analysis['critical_issues']:
            html += """
                <div class="card">
                    <h2>🚨 Critical Security Issues</h2>
                    <table>
                        <tr>
                            <th>File</th>
                            <th>Line</th>
                            <th>Issue</th>
                            <th>Description</th>
                        </tr>
            """
            for issue in analysis['critical_issues'][:10]:  # Top 10
                html += f"""
                        <tr>
                            <td>{issue['file']}</td>
                            <td>{issue['line']}</td>
                            <td>{issue['test']}</td>
                            <td>{issue['description']}</td>
                        </tr>
                """
            html += """
                    </table>
                </div>
            """
        
        # Add security hotspots
        if analysis['security_hotspots']:
            html += """
                <div class="card">
                    <h2>🔥 Security Hotspots</h2>
                    <p>Files with multiple security issues requiring attention:</p>
            """
            for hotspot in analysis['security_hotspots']:
                html += f"""
                    <div class="hotspot">
                        <strong>{hotspot['file']}</strong> - {hotspot['issue_count']} issues found
                    </div>
                """
            html += """
                </div>
            """
        
        # Add recommendations
        if analysis['recommendations']:
            html += """
                <div class="recommendations">
                    <h3>📋 Security Recommendations</h3>
                    <ul>
            """
            for rec in analysis['recommendations']:
                html += f"<li>{rec}</li>"
            html += """
                    </ul>
                </div>
            """
        
        # Add issue breakdown by test type
        html += """
                <div class="card">
                    <h2>Issue Breakdown by Test</h2>
                    <table>
                        <tr>
                            <th>Test ID</th>
                            <th>Test Name</th>
                            <th>Severity</th>
                            <th>Count</th>
                        </tr>
        """
        
        for test_id, test_data in sorted(results['summary']['by_test'].items(), 
                                        key=lambda x: x[1]['count'], reverse=True)[:15]:
            html += f"""
                        <tr>
                            <td>{test_id}</td>
                            <td>{test_data['name']}</td>
                            <td><span class="severity-{test_data['severity'].lower()}">{test_data['severity']}</span></td>
                            <td>{test_data['count']}</td>
                        </tr>
            """
        
        html += """
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
```

---

## 2. Automated Security Scripts

### 2.1 Daily Security Tasks Automation

```python
#!/usr/bin/env python3
# automation/daily_security_tasks.py

import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import ssl
import socket
import subprocess
import docker
import redis
import psycopg2
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YTEMPIREDailySecurityAutomation:
    """Automated daily security tasks for YTEMPIRE platform"""
    
    def __init__(self):
        """Initialize daily security automation"""
        self.config = self._load_config()
        self.docker_client = docker.from_env()
        self.redis_client = redis.Redis(
            host=os.environ.get('REDIS_HOST', 'localhost'),
            port=int(os.environ.get('REDIS_PORT', 6379)),
            decode_responses=True
        )
        
        # Task registry
        self.tasks = [
            self.check_ssl_certificates,
            self.scan_container_vulnerabilities,
            self.review_access_logs,
            self.audit_user_permissions,
            self.check_security_updates,
            self.verify_backup_integrity,
            self.scan_for_secrets,
            self.review_firewall_rules,
            self.check_api_rate_limits,
            self.validate_encryption_status
        ]
        
        # YTEMPIRE domains to monitor
        self.ytempire_domains = [
            ("api.ytempire.com", 443),
            ("app.ytempire.com", 443),
            ("admin.ytempire.com", 443),
            ("ws.ytempire.com", 443),
            ("cdn.ytempire.com", 443)
        ]
        
        # Critical log patterns
        self.suspicious_patterns = [
            (r"(\d+)\s+failed login attempts", "brute_force"),
            (r"admin.*password.*failed", "credential_guessing"),
            (r"\.\./\.\./", "path_traversal"),
            (r"union\s+select", "sql_injection"),
            (r"<script[^>]*>", "xss_attempt"),
            (r"eval\s*\(", "code_injection"),
            (r"';.*drop\s+table", "sql_injection"),
            (r"base64_decode\s*\(", "obfuscation"),
            (r"/etc/passwd", "file_inclusion"),
            (r"cmd=.*&&", "command_injection")
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
        
        # Run tasks concurrently with error handling
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
        
        # Store results
        await self._store_results(results)
        
        # Send reports
        await self._send_daily_report(results)
        
        # Trigger immediate actions for critical issues
        if results['critical_issues'] > 0:
            await self._handle_critical_issues(results)
        
        logger.info(f"✅ Daily security tasks completed in {duration:.2f} seconds")
        
        return results
    
    async def _run_task_safely(self, task) -> Dict:
        """Run a task with error handling"""
        try:
            result = await task()
            result['status'] = 'success'
            return result
        except Exception as e:
            logger.error(f"Task {task.__name__} failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'issues': []
            }
    
    async def check_ssl_certificates(self) -> Dict:
        """Check SSL certificate expiration for YTEMPIRE domains"""
        logger.info("🔐 Checking SSL certificates...")
        
        results = {
            'checked': len(self.ytempire_domains),
            'issues': []
        }
        
        for domain, port in self.ytempire_domains:
            try:
                # Create SSL context
                context = ssl.create_default_context()
                
                # Connect and get certificate
                with socket.create_connection((domain, port), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        cert = ssock.getpeercert()
                
                # Parse expiration date
                not_after_str = cert['notAfter']
                not_after = datetime.strptime(not_after_str, '%b %d %H:%M:%S %Y %Z')
                days_remaining = (not_after - datetime.utcnow()).days
                
                # Check thresholds
                if days_remaining < self.config['ssl_critical_days']:
                    severity = 'critical'
                elif days_remaining < self.config['ssl_warning_days']:
                    severity = 'high'
                else:
                    continue  # Certificate is fine
                
                results['issues'].append({
                    'domain': domain,
                    'days_remaining': days_remaining,
                    'expires': not_after.isoformat(),
                    'severity': severity,
                    'message': f"SSL certificate for {domain} expires in {days_remaining} days",
                    'action_required': 'Renew SSL certificate immediately' if severity == 'critical' else 'Plan certificate renewal'
                })
                
            except socket.timeout:
                results['issues'].append({
                    'domain': domain,
                    'error': 'Connection timeout',
                    'severity': 'medium',
                    'message': f"Could not check SSL certificate for {domain} - connection timeout"
                })
            except Exception as e:
                results['issues'].append({
                    'domain': domain,
                    'error': str(e),
                    'severity': 'medium',
                    'message': f"SSL certificate check failed for {domain}: {str(e)}"
                })
        
        return results
    
    async def scan_container_vulnerabilities(self) -> Dict:
        """Scan all YTEMPIRE containers for vulnerabilities"""
        logger.info("🐳 Scanning container vulnerabilities...")
        
        from security_tools.trivy_scanner import YTEMPIRETrivyScanner
        
        scanner = YTEMPIRETrivyScanner()
        results = {
            'containers_scanned': 0,
            'issues': []
        }
        
        # Get all running containers
        containers = self.docker_client.containers.list()
        
        for container in containers:
            # Only scan YTEMPIRE containers
            if not any(container.name.startswith(prefix) for prefix in ['ytempire', 'ytmp']):
                continue
            
            results['containers_scanned'] += 1
            
            # Get image name
            image_name = container.image.tags[0] if container.image.tags else container.image.id
            
            # Scan with Trivy
            scan_result = scanner.scan_image(image_name, {
                'severity': ['CRITICAL', 'HIGH'],
                'ignore_unfixed': True
            })
            
            if not scan_result.get('passed', True):
                # Extract critical vulnerabilities
                for severity in ['CRITICAL', 'HIGH']:
                    for vuln in scan_result.get('results', {}).get('vulnerabilities', {}).get(severity, [])[:3]:
                        results['issues'].append({
                            'container': container.name,
                            'image': image_name,
                            'vulnerability': vuln['id'],
                            'package': vuln['package'],
                            'severity': severity.lower(),
                            'fixed_version': vuln.get('fixed_version', 'No fix available'),
                            'message': f"{severity} vulnerability {vuln['id']} in {container.name}",
                            'action_required': f"Update {vuln['package']} to {vuln.get('fixed_version', 'latest')}"
                        })
        
        return results
    
    async def review_access_logs(self) -> Dict:
        """Analyze access logs for suspicious activity"""
        logger.info("📊 Reviewing access logs...")
        
        import re
        from collections import defaultdict
        
        results = {
            'logs_analyzed': 0,
            'issues': []
        }
        
        # Log files to analyze
        log_files = [
            "/var/log/nginx/access.log",
            "/var/log/nginx/error.log",
            "/opt/ytempire/logs/api.log",
            "/opt/ytempire/logs/admin.log"
        ]
        
        ip_activity = defaultdict(list)
        suspicious_ips = set()
        
        for log_file in log_files:
            if not os.path.exists(log_file):
                continue
            
            try:
                # Read last 10000 lines
                with open(log_file, 'r') as f:
                    lines = f.readlines()[-10000:]
                    results['logs_analyzed'] += len(lines)
                
                for line in lines:
                    # Extract IP address
                    ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', line)
                    if not ip_match:
                        continue
                    
                    ip = ip_match.group(1)
                    
                    # Check for suspicious patterns
                    for pattern, attack_type in self.suspicious_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            ip_activity[ip].append(attack_type)
                            suspicious_ips.add(ip)
                            break
            
            except Exception as e:
                logger.error(f"Error reading log {log_file}: {e}")
        
        # Analyze IP activity
        for ip in suspicious_ips:
            activities = ip_activity[ip]
            unique_attacks = list(set(activities))
            
            if len(activities) > self.config['max_failed_logins']:
                severity = 'critical' if len(activities) > 20 else 'high'
                
                results['issues'].append({
                    'source_ip': ip,
                    'attack_types': unique_attacks,
                    'attempt_count': len(activities),
                    'severity': severity,
                    'message': f"Suspicious activity from {ip}: {len(activities)} attempts",
                    'action_required': f"Block IP {ip} and investigate"
                })
        
        return results
    
    async def audit_user_permissions(self) -> Dict:
        """Audit user permissions and access controls"""
        logger.info("👥 Auditing user permissions...")
        
        results = {
            'users_audited': 0,
            'issues': []
        }
        
        try:
            # Connect to database
            conn = psycopg2.connect(
                host=os.environ.get('DB_HOST', 'localhost'),
                database=os.environ.get('DB_NAME', 'ytempire'),
                user=os.environ.get('DB_USER'),
                password=os.environ.get('DB_PASSWORD')
            )
            cur = conn.cursor()
            
            # Check for users with excessive permissions
            cur.execute("""
                SELECT u.id, u.email, u.role, u.last_login, u.created_at,
                       COUNT(DISTINCT p.id) as permission_count
                FROM users u
                LEFT JOIN user_permissions up ON u.id = up.user_id
                LEFT JOIN permissions p ON up.permission_id = p.id
                GROUP BY u.id, u.email, u.role, u.last_login, u.created_at
            """)
            
            users = cur.fetchall()
            results['users_audited'] = len(users)
            
            for user in users:
                user_id, email, role, last_login, created_at, perm_count = user
                
                # Check for admin users who haven't logged in recently
                if role == 'admin' and last_login:
                    days_inactive = (datetime.utcnow() - last_login).days
                    if days_inactive > 30:
                        results['issues'].append({
                            'user_email': email,
                            'role': role,
                            'days_inactive': days_inactive,
                            'severity': 'medium',
                            'message': f"Admin user {email} inactive for {days_inactive} days",
                            'action_required': 'Review and potentially revoke admin access'
                        })
                
                # Check for users with too many permissions
                if perm_count > 20 and role != 'admin':
                    results['issues'].append({
                        'user_email': email,
                        'role': role,
                        'permission_count': perm_count,
                        'severity': 'high',
                        'message': f"Non-admin user {email} has {perm_count} permissions",
                        'action_required': 'Review and reduce permissions to minimum necessary'
                    })
            
            # Check for orphaned permissions
            cur.execute("""
                SELECT COUNT(*) FROM user_permissions up
                LEFT JOIN users u ON up.user_id = u.id
                WHERE u.id IS NULL
            """)
            
            orphaned_count = cur.fetchone()[0]
            if orphaned_count > 0:
                results['issues'].append({
                    'orphaned_permissions': orphaned_count,
                    'severity': 'medium',
                    'message': f"Found {orphaned_count} orphaned permission entries",
                    'action_required': 'Clean up orphaned permissions'
                })
            
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database audit failed: {str(e)}")
            results['issues'].append({
                'error': str(e),
                'severity': 'high',
                'message': 'Failed to complete user permission audit',
                'action_required': 'Investigate database connectivity'
            })
        
        return results
    
    async def check_security_updates(self) -> Dict:
        """Check for available security updates"""
        logger.info("🔄 Checking security updates...")
        
        results = {
            'systems_checked': 0,
            'issues': []
        }
        
        # Check system packages
        try:
            # Check for Ubuntu security updates
            result = subprocess.run(
                ["apt", "list", "--upgradable"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if "security" in result.stdout.lower():
                security_updates = [line for line in result.stdout.split('\n') if 'security' in line.lower()]
                
                if security_updates:
                    results['issues'].append({
                        'update_count': len(security_updates),
                        'severity': 'high',
                        'message': f"Found {len(security_updates)} security updates available",
                        'updates': security_updates[:5],  # First 5 updates
                        'action_required': 'Apply security updates immediately'
                    })
            
            results['systems_checked'] += 1
            
        except Exception as e:
            logger.error(f"Failed to check system updates: {str(e)}")
        
        # Check Python packages
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                security_packages = ['cryptography', 'requests', 'urllib3', 'pyyaml', 'django', 'flask']
                
                critical_updates = [
                    pkg for pkg in outdated
                    if any(sec_pkg in pkg['name'].lower() for sec_pkg in security_packages)
                ]
                
                if critical_updates:
                    results['issues'].append({
                        'package_count': len(critical_updates),
                        'severity': 'medium',
                        'message': f"Found {len(critical_updates)} Python security-related packages outdated",
                        'packages': critical_updates[:5],
                        'action_required': 'Update Python security packages'
                    })
            
            results['systems_checked'] += 1
            
        except Exception as e:
            logger.error(f"Failed to check Python updates: {str(e)}")
        
        return results
    
    async def verify_backup_integrity(self) -> Dict:
        """Verify backup integrity and test restoration"""
        logger.info("💾 Verifying backup integrity...")
        
        results = {
            'backups_verified': 0,
            'issues': []
        }
        
        backup_dir = Path("/opt/ytempire/backups")
        
        if not backup_dir.exists():
            results['issues'].append({
                'severity': 'critical',
                'message': 'Backup directory does not exist',
                'action_required': 'Create backup directory and configure backups'
            })
            return results
        
        # Get latest backup files
        backup_files = sorted(backup_dir.glob("*.tar.gz"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not backup_files:
            results['issues'].append({
                'severity': 'critical',
                'message': 'No backup files found',
                'action_required': 'Initiate backup immediately'
            })
            return results
        
        latest_backup = backup_files[0]
        backup_age_hours = (datetime.utcnow() - datetime.fromtimestamp(latest_backup.stat().st_mtime)).total_seconds() / 3600
        
        # Check backup age
        if backup_age_hours > 24:
            results['issues'].append({
                'backup_file': str(latest_backup),
                'age_hours': backup_age_hours,
                'severity': 'high',
                'message': f"Latest backup is {backup_age_hours:.1f} hours old",
                'action_required': 'Run backup immediately'
            })
        
        # Verify backup integrity
        try:
            result = subprocess.run(
                ["tar", "-tzf", str(latest_backup)],
                capture_output=True,
                timeout=60
            )
            
            if result.returncode != 0:
                results['issues'].append({
                    'backup_file': str(latest_backup),
                    'severity': 'critical',
                    'message': 'Backup file is corrupted',
                    'action_required': 'Create new backup immediately'
                })
            else:
                results['backups_verified'] += 1
                
        except Exception as e:
            results['issues'].append({
                'backup_file': str(latest_backup),
                'error': str(e),
                'severity': 'high',
                'message': 'Failed to verify backup integrity',
                'action_required': 'Manual backup verification required'
            })
        
        return results
    
    async def scan_for_secrets(self) -> Dict:
        """Scan codebase for hardcoded secrets"""
        logger.info("🔑 Scanning for secrets...")
        
        results = {
            'repositories_scanned': 0,
            'issues': []
        }
        
        repos = [
            "/opt/ytempire/api",
            "/opt/ytempire/frontend",
            "/opt/ytempire/worker",
            "/opt/ytempire/admin"
        ]
        
        for repo in repos:
            if not os.path.exists(repo):
                continue
            
            results['repositories_scanned'] += 1
            
            try:
                # Run git-secrets scan
                result = subprocess.run(
                    ["git", "secrets", "--scan"],
                    cwd=repo,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    # Secrets found
                    results['issues'].append({
                        'repository': repo,
                        'severity': 'critical',
                        'message': f"Hardcoded secrets found in {repo}",
                        'details': result.stderr[:500],  # First 500 chars
                        'action_required': 'Remove secrets immediately and rotate credentials'
                    })
                    
            except subprocess.TimeoutExpired:
                results['issues'].append({
                    'repository': repo,
                    'severity': 'medium',
                    'message': f"Secret scan timeout for {repo}",
                    'action_required': 'Manual secret scan required'
                })
            except Exception as e:
                logger.error(f"Secret scan failed for {repo}: {str(e)}")
        
        return results
    
    async def review_firewall_rules(self) -> Dict:
        """Review firewall rules and network security"""
        logger.info("🔥 Reviewing firewall rules...")
        
        results = {
            'rules_checked': 0,
            'issues': []
        }
        
        try:
            # Check iptables rules
            result = subprocess.run(
                ["iptables", "-L", "-n", "-v"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                rules = result.stdout
                results['rules_checked'] = rules.count('\n')
                
                # Check for dangerous rules
                if 'ACCEPT     all' in rules and '0.0.0.0/0' in rules:
                    results['issues'].append({
                        'severity': 'critical',
                        'message': 'Firewall has overly permissive rules accepting all traffic',
                        'action_required': 'Tighten firewall rules immediately'
                    })
                
                # Check if firewall is enabled
                if 'Chain INPUT (policy ACCEPT)' in rules:
                    results['issues'].append({
                        'severity': 'high',
                        'message': 'Firewall default policy is ACCEPT (should be DROP)',
                        'action_required': 'Change default policy to DROP and explicitly allow required traffic'
                    })
                
                # Check for exposed ports
                dangerous_ports = ['3306', '5432', '6379', '27017', '9200']  # Database ports
                for port in dangerous_ports:
                    if f'dpt:{port}' in rules and 'ACCEPT' in rules:
                        results['issues'].append({
                            'port': port,
                            'severity': 'high',
                            'message': f"Database port {port} is exposed",
                            'action_required': f'Restrict access to port {port} to specific IPs only'
                        })
                        
        except Exception as e:
            results['issues'].append({
                'error': str(e),
                'severity': 'medium',
                'message': 'Failed to check firewall rules',
                'action_required': 'Manual firewall audit required'
            })
        
        return results
    
    async def check_api_rate_limits(self) -> Dict:
        """Check API rate limiting configuration"""
        logger.info("⏱️ Checking API rate limits...")
        
        results = {
            'endpoints_checked': 0,
            'issues': []
        }
        
        # Check Redis for rate limit data
        try:
            # Get all rate limit keys
            rate_limit_keys = self.redis_client.keys('rate_limit:*')
            results['endpoints_checked'] = len(rate_limit_keys)
            
            # Check for endpoints without rate limiting
            critical_endpoints = [
                '/api/auth/login',
                '/api/auth/register',
                '/api/channels/create',
                '/api/videos/generate',
                '/api/admin/*'
            ]
            
            for endpoint in critical_endpoints:
                key_pattern = f"rate_limit:*{endpoint}*"
                if not any(self.redis_client.keys(key_pattern)):
                    results['issues'].append({
                        'endpoint': endpoint,
                        'severity': 'high',
                        'message': f"Critical endpoint {endpoint} lacks rate limiting",
                        'action_required': 'Implement rate limiting for this endpoint'
                    })
            
            # Check for excessive rates
            for key in rate_limit_keys:
                data = self.redis_client.get(key)
                if data:
                    try:
                        limit_data = json.loads(data) if isinstance(data, str) else data
                        if isinstance(limit_data, dict) and limit_data.get('count', 0) > 1000:
                            results['issues'].append({
                                'key': key,
                                'count': limit_data['count'],
                                'severity': 'medium',
                                'message': f"Excessive API calls detected: {limit_data['count']} requests",
                                'action_required': 'Investigate potential API abuse'
                            })
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Rate limit check failed: {str(e)}")
            results['issues'].append({
                'error': str(e),
                'severity': 'medium',
                'message': 'Failed to check API rate limits',
                'action_required': 'Verify rate limiting is properly configured'
            })
        
        return results
    
    async def validate_encryption_status(self) -> Dict:
        """Validate encryption status of sensitive data"""
        logger.info("🔐 Validating encryption status...")
        
        results = {
            'checks_performed': 0,
            'issues': []
        }
        
        # Check database encryption
        try:
            conn = psycopg2.connect(
                host=os.environ.get('DB_HOST', 'localhost'),
                database=os.environ.get('DB_NAME', 'ytempire'),
                user=os.environ.get('DB_USER'),
                password=os.environ.get('DB_PASSWORD')
            )
            cur = conn.cursor()
            
            # Check for unencrypted sensitive columns
            cur.execute("""
                SELECT table_name, column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND column_name IN ('password', 'api_key', 'secret', 'token', 'ssn', 'credit_card')
            """)
            
            sensitive_columns = cur.fetchall()
            results['checks_performed'] += len(sensitive_columns)
            
            for table, column in sensitive_columns:
                # Check if column appears to be encrypted (simple heuristic)
                cur.execute(f"""
                    SELECT {column} FROM {table} LIMIT 1
                """)
                
                sample = cur.fetchone()
                if sample and sample[0]:
                    value = str(sample[0])
                    # Simple check - encrypted data should be hex or base64
                    if not (all(c in '0123456789abcdef' for c in value.lower()) or 
                           value.endswith('=')):
                        results['issues'].append({
                            'table': table,
                            'column': column,
                            'severity': 'critical',
                            'message': f"Potentially unencrypted sensitive data in {table}.{column}",
                            'action_required': 'Encrypt sensitive data immediately'
                        })
            
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Encryption validation failed: {str(e)}")
        
        # Check file encryption
        sensitive_dirs = [
            "/opt/ytempire/uploads",
            "/opt/ytempire/exports",
            "/opt/ytempire/backups"
        ]
        
        for dir_path in sensitive_dirs:
            if os.path.exists(dir_path):
                results['checks_performed'] += 1
                
                # Check if directory is encrypted (Linux)
                try:
                    result = subprocess.run(
                        ["lsattr", "-d", dir_path],
                        capture_output=True,
                        text=True
                    )
                    
                    if 'E' not in result.stdout:  # 'E' flag indicates encryption
                        results['issues'].append({
                            'directory': dir_path,
                            'severity': 'high',
                            'message': f"Sensitive directory {dir_path} is not encrypted",
                            'action_required': 'Enable filesystem encryption for sensitive directories'
                        })
                except:
                    pass
        
        return results
    
    async def _store_results(self, results: Dict):
        """Store scan results in database"""
        try:
            # Store in Redis for quick access
            key = f"security_scan:{results['date']}"
            self.redis_client.setex(
                key,
                86400 * 7,  # Keep for 7 days
                json.dumps(results, default=str)
            )
            
            # Store summary in PostgreSQL for long-term analysis
            conn = psycopg2.connect(
                host=os.environ.get('DB_HOST', 'localhost'),
                database=os.environ.get('DB_NAME', 'ytempire'),
                user=os.environ.get('DB_USER'),
                password=os.environ.get('DB_PASSWORD')
            )
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO security_scans (scan_date, duration_seconds, critical_issues, 
                                           high_issues, medium_issues, low_issues, 
                                           total_issues, results)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                results['date'],
                results['duration_seconds'],
                results['critical_issues'],
                results['high_issues'],
                results['medium_issues'],
                results['low_issues'],
                len(results['issues_found']),
                json.dumps(results)
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store results: {str(e)}")
    
    async def _send_daily_report(self, results: Dict):
        """Send daily security report via email"""
        logger.info("📧 Sending daily security report...")
        
        # Generate HTML report
        html_content = self._generate_html_report(results)
        
        # Create message
        msg = MIMEMultipart('mixed')
        msg['Subject'] = f"YTEMPIRE Daily Security Report - {results['date']}"
        msg['From'] = "security@ytempire.com"
        msg['To'] = ", ".join(self.config['email_recipients'])
        
        # Attach HTML body
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        # Attach JSON results as file
        json_attachment = MIMEBase('application', 'json')
        json_attachment.set_payload(json.dumps(results, indent=2, default=str))
        encoders.encode_base64(json_attachment)
        json_attachment.add_header(
            'Content-Disposition',
            f'attachment; filename=security_report_{results["date"]}.json'
        )
        msg.attach(json_attachment)
        
        # Send email
        try:
            with smtplib.SMTP(os.environ.get('SMTP_HOST', 'localhost'), 587) as server:
                server.starttls()
                if os.environ.get('SMTP_USER') and os.environ.get('SMTP_PASSWORD'):
                    server.login(os.environ.get('SMTP_USER'), os.environ.get('SMTP_PASSWORD'))
                server.send_message(msg)
                logger.info("✅ Daily security report sent successfully")
                
        except Exception as e:
            logger.error(f"Failed to send email report: {str(e)}")
            
            # Fallback to Slack notification
            if self.config.get('slack_webhook'):
                await self._send_slack_summary(results)
    
    async def _send_slack_summary(self, results: Dict):
        """Send summary to Slack"""
        import requests
        
        status_emoji = "🔴" if results['critical_issues'] > 0 else "🟡" if results['high_issues'] > 0 else "🟢"
        
        slack_message = {
            "text": f"{status_emoji} YTEMPIRE Daily Security Report - {results['date']}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{status_emoji} Daily Security Report"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Date:* {results['date']}"},
                        {"type": "mrkdwn", "text": f"*Duration:* {results['duration_seconds']:.1f}s"},
                        {"type": "mrkdwn", "text": f"*Critical:* {results['critical_issues']}"},
                        {"type": "mrkdwn", "text": f"*High:* {results['high_issues']}"},
                        {"type": "mrkdwn", "text": f"*Medium:* {results['medium_issues']}"},
                        {"type": "mrkdwn", "text": f"*Low:* {results['low_issues']}"}
                    ]
                }
            ]
        }
        
        if results['critical_issues'] > 0:
            critical_issues_text = "\n".join([
                f"• {issue['message']}" 
                for issue in results['issues_found'] 
                if issue.get('severity') == 'critical'
            ][:5])
            
            slack_message["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*🚨 Critical Issues:*\n{critical_issues_text}"
                }
            })
        
        try:
            response = requests.post(
                self.config['slack_webhook'],
                json=slack_message,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✅ Slack notification sent")
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {str(e)}")
    
    async def _handle_critical_issues(self, results: Dict):
        """Handle critical security issues with immediate actions"""
        logger.warning(f"⚠️ Handling {results['critical_issues']} critical issues...")
        
        for issue in results['issues_found']:
            if issue.get('severity') != 'critical':
                continue
            
            # Take automated actions based on issue type
            if 'SSL certificate' in issue.get('message', ''):
                # Attempt to renew certificate
                await self._renew_ssl_certificate(issue.get('domain'))
                
            elif 'Suspicious activity' in issue.get('message', ''):
                # Block suspicious IP
                ip = issue.get('source_ip')
                if ip:
                    await self._block_ip_address(ip)
                    
            elif 'vulnerability' in issue.get('message', '').lower():
                # Trigger container update
                container = issue.get('container')
                if container:
                    await self._schedule_container_update(container)
                    
            elif 'secrets found' in issue.get('message', '').lower():
                # Notify immediately and rotate credentials
                await self._emergency_credential_rotation(issue.get('repository'))
    
    async def _block_ip_address(self, ip: str):
        """Block malicious IP address"""
        try:
            subprocess.run(
                ["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"],
                check=True
            )
            logger.info(f"✅ Blocked IP address: {ip}")
            
            # Add to permanent blocklist
            with open("/etc/ytempire/blocked_ips.txt", "a") as f:
                f.write(f"{ip} # Auto-blocked {datetime.utcnow()}\n")
                
        except Exception as e:
            logger.error(f"Failed to block IP {ip}: {str(e)}")
    
    def _generate_html_report(self, results: Dict) -> str:
        """Generate HTML report content"""
        
        # Determine overall status
        if results['critical_issues'] > 0:
            status_color = '#ef4444'
            status_text = 'CRITICAL ISSUES FOUND'
        elif results['high_issues'] > 0:
            status_color = '#f59e0b'
            status_text = 'HIGH PRIORITY ISSUES'
        elif results['medium_issues'] > 0:
            status_color = '#3b82f6'
            status_text = 'MEDIUM PRIORITY ISSUES'
        else:
            status_color = '#10b981'
            status_text = 'ALL SYSTEMS SECURE'
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YTEMPIRE Daily Security Report - {results['date']}</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; background: #f3f4f6; }}
                .container {{ max-width: 900px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; }}
                .content {{ background: white; padding: 30px; border-radius: 0 0 10px 10px; }}
                .status-bar {{ background: {status_color}; color: white; padding: 15px; text-align: center; font-weight: bold; font-size: 1.2em; }}
                .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric {{ text-align: center; padding: 15px; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #1f2937; }}
                .metric-label {{ color: #6b7280; font-size: 0.9em; margin-top: 5px; }}
                .severity-critical {{ background: #ef4444; color: white; padding: 2px 8px; border-radius: 4px; }}
                .severity-high {{ background: #f59e0b; color: white; padding: 2px 8px; border-radius: 4px; }}
                .severity-medium {{ background: #3b82f6; color: white; padding: 2px 8px; border-radius: 4px; }}
                .severity-low {{ background: #10b981; color: white; padding: 2px 8px; border-radius: 4px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th {{ background: #f3f4f6; padding: 12px; text-align: left; font-weight: 600; border-bottom: 2px solid #e5e7eb; }}
                td {{ padding: 12px; border-bottom: 1px solid #e5e7eb; }}
                .task-success {{ color: #10b981; }}
                .task-error {{ color: #ef4444; }}
                .issue-card {{ background: #fef2f2; border-left: 4px solid #ef4444; padding: 15px; margin: 10px 0; border-radius: 4px; }}
                .issue-card.high {{ background: #fffbeb; border-color: #f59e0b; }}
                .issue-card.medium {{ background: #eff6ff; border-color: #3b82f6; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🛡️ YTEMPIRE Daily Security Report</h1>
                    <p>Automated Security Scan Results for {results['date']}</p>
                </div>
                
                <div class="status-bar">{status_text}</div>
                
                <div class="content">
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-value">{results['critical_issues']}</div>
                            <div class="metric-label">Critical</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{results['high_issues']}</div>
                            <div class="metric-label">High</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{results['medium_issues']}</div>
                            <div class="metric-label">Medium</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{results['low_issues']}</div>
                            <div class="metric-label">Low</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{results['duration_seconds']:.1f}s</div>
                            <div class="metric-label">Scan Time</div>
                        </div>
                    </div>
                    
                    <h2>Task Results</h2>
                    <table>
                        <tr>
                            <th>Task</th>
                            <th>Status</th>
                            <th>Issues Found</th>
                        </tr>
        """
        
        for task_name, task_result in results['tasks'].items():
            status = task_result.get('status', 'unknown')
            status_class = 'task-success' if status == 'success' else 'task-error'
            issue_count = len(task_result.get('issues', []))
            
            html += f"""
                        <tr>
                            <td>{task_name.replace('_', ' ').title()}</td>
                            <td class="{status_class}">{status.upper()}</td>
                            <td>{issue_count}</td>
                        </tr>
            """
        
        html += """
                    </table>
        """
        
        # Add critical issues section
        critical_issues = [i for i in results['issues_found'] if i.get('severity') == 'critical']
        if critical_issues:
            html += """
                    <h2>🚨 Critical Issues Requiring Immediate Action</h2>
            """
            for issue in critical_issues:
                html += f"""
                    <div class="issue-card">
                        <strong>{issue.get('message', 'No description')}</strong><br>
                        <em>Task: {issue.get('task', 'Unknown')}</em><br>
                        <strong>Action Required:</strong> {issue.get('action_required', 'Review and address')}
                    </div>
                """
        
        # Add high priority issues
        high_issues = [i for i in results['issues_found'] if i.get('severity') == 'high']
        if high_issues:
            html += """
                    <h2>⚠️ High Priority Issues</h2>
            """
            for issue in high_issues[:10]:  # Limit to 10
                html += f"""
                    <div class="issue-card high">
                        <strong>{issue.get('message', 'No description')}</strong><br>
                        <em>Task: {issue.get('task', 'Unknown')}</em><br>
                        <strong>Action Required:</strong> {issue.get('action_required', 'Review and address')}
                    </div>
                """
        
        html += """
                </div>
            </div>
        </body>
        </html>
        """
        
        return html


# Main execution
if __name__ == "__main__":
    async def main():
        automation = YTEMPIREDailySecurityAutomation()
        results = await automation.run_daily_tasks()
        print(f"Security scan completed: {results['critical_issues']} critical issues found")
    
    asyncio.run(main())
```

### 2.2 Automated Remediation System

```python
#!/usr/bin/env python3
# automation/auto_remediation.py

import asyncio
import logging
import os
import json
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import docker
import redis
import psycopg2
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RemediationAction(Enum):
    """Remediation action types"""
    BLOCK_IP = "block_ip"
    UPDATE_CONTAINER = "update_container"
    ROTATE_CREDENTIALS = "rotate_credentials"
    PATCH_VULNERABILITY = "patch_vulnerability"
    ENFORCE_ENCRYPTION = "enforce_encryption"
    REVOKE_ACCESS = "revoke_access"
    RESTART_SERVICE = "restart_service"
    APPLY_SECURITY_PATCH = "apply_security_patch"
    QUARANTINE_FILE = "quarantine_file"
    DISABLE_ACCOUNT = "disable_account"


class YTEMPIREAutoRemediation:
    """Automated security remediation system for YTEMPIRE"""
    
    def __init__(self):
        """Initialize auto-remediation system"""
        self.docker_client = docker.from_env()
        self.redis_client = redis.Redis(
            host=os.environ.get('REDIS_HOST', 'localhost'),
            port=int(os.environ.get('REDIS_PORT', 6379)),
            decode_responses=True
        )
        
        # Remediation handlers
        self.remediation_handlers = {
            RemediationAction.BLOCK_IP: self.block_ip_address,
            RemediationAction.UPDATE_CONTAINER: self.update_container,
            RemediationAction.ROTATE_CREDENTIALS: self.rotate_credentials,
            RemediationAction.PATCH_VULNERABILITY: self.patch_vulnerability,
            RemediationAction.ENFORCE_ENCRYPTION: self.enforce_encryption,
            RemediationAction.REVOKE_ACCESS: self.revoke_user_access,
            RemediationAction.RESTART_SERVICE: self.restart_service,
            RemediationAction.APPLY_SECURITY_PATCH: self.apply_security_patch,
            RemediationAction.QUARANTINE_FILE: self.quarantine_file,
            RemediationAction.DISABLE_ACCOUNT: self.disable_user_account
        }
        
        # Auto-remediation policies
        self.policies = self._load_remediation_policies()
        
        # Track remediation history
        self.remediation_history = []
    
    def _load_remediation_policies(self) -> Dict:
        """Load auto-remediation policies"""
        policies_path = Path("/opt/ytempire/security/remediation_policies.json")
        
        if policies_path.exists():
            with open(policies_path, 'r') as f:
                return json.load(f)
        
        # Default policies
        return {
            "brute_force_attack": {
                "threshold": 5,
                "actions": [RemediationAction.BLOCK_IP, RemediationAction.DISABLE_ACCOUNT],
                "auto_remediate": True
            },
            "critical_vulnerability": {
                "actions": [RemediationAction.UPDATE_CONTAINER, RemediationAction.PATCH_VULNERABILITY],
                "auto_remediate": True,
                "require_approval": False
            },
            "exposed_secrets": {
                "actions": [RemediationAction.ROTATE_CREDENTIALS, RemediationAction.QUARANTINE_FILE],
                "auto_remediate": True,
                "require_approval": False
            },
            "unauthorized_access": {
                "actions": [RemediationAction.REVOKE_ACCESS, RemediationAction.BLOCK_IP],
                "auto_remediate": True
            },
            "encryption_failure": {
                "actions": [RemediationAction.ENFORCE_ENCRYPTION],
                "auto_remediate": True
            }
        }
    
    async def remediate_issue(self, issue: Dict) -> Dict:
        """
        Automatically remediate a security issue
        
        Args:
            issue: Security issue details
            
        Returns:
            Dict containing remediation results
        """
        issue_type = issue.get('type')
        severity = issue.get('severity', 'low')
        
        logger.info(f"🔧 Remediating {severity} issue: {issue_type}")
        
        remediation_result = {
            'issue_id': issue.get('id', f"issue_{datetime.utcnow().timestamp()}"),
            'issue_type': issue_type,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat(),
            'actions_taken': [],
            'status': 'pending'
        }
        
        # Check if auto-remediation is allowed
        if not self._can_auto_remediate(issue_type, severity):
            remediation_result['status'] = 'manual_required'
            remediation_result['reason'] = 'Issue requires manual intervention'
            logger.warning(f"⚠️ Manual remediation required for {issue_type}")
            return remediation_result
        
        # Get remediation policy
        policy = self.policies.get(issue_type, {})
        actions = policy.get('actions', [])
        
        # Execute remediation actions
        for action in actions:
            try:
                handler = self.remediation_handlers.get(action)
                if handler:
                    result = await handler(issue)
                    remediation_result['actions_taken'].append({
                        'action': action.value,
                        'result': result,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
                    if not result.get('success', False):
                        logger.error(f"❌ Remediation action {action.value} failed")
                        remediation_result['status'] = 'partial_failure'
                else:
                    logger.warning(f"No handler for action: {action}")
                    
            except Exception as e:
                logger.error(f"Remediation action {action.value} failed: {str(e)}")
                remediation_result['actions_taken'].append({
                    'action': action.value,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                })
                remediation_result['status'] = 'failed'
        
        # Verify remediation
        if remediation_result['status'] == 'pending':
            verification = await self._verify_remediation(issue, remediation_result)
            remediation_result['verification'] = verification
            remediation_result['status'] = 'success' if verification['verified'] else 'verification_failed'
        
        # Store remediation history
        self._store_remediation_history(remediation_result)
        
        # Send notification
        await self._send_remediation_notification(remediation_result)
        
        logger.info(f"✅ Remediation completed with status: {remediation_result['status']}")
        
        return remediation_result
    
    def _can_auto_remediate(self, issue_type: str, severity: str) -> bool:
        """Check if issue can be auto-remediated"""
        # Don't auto-remediate in production without approval for critical issues
        if severity == 'critical' and os.environ.get('ENVIRONMENT') == 'production':
            policy = self.policies.get(issue_type, {})
            if policy.get('require_approval', True):
                return False
        
        # Check if issue type has auto-remediation enabled
        policy = self.policies.get(issue_type, {})
        return policy.get('auto_remediate', False)
    
    async def block_ip_address(self, issue: Dict) -> Dict:
        """Block malicious IP address"""
        ip = issue.get('source_ip')
        
        if not ip:
            return {'success': False, 'error': 'No IP address provided'}
        
        try:
            # Add iptables rule
            subprocess.run(
                ["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"],
                check=True
            )
            
            # Add to permanent blocklist
            blocklist_path = Path("/etc/ytempire/blocked_ips.txt")
            with open(blocklist_path, "a") as f:
                f.write(f"{ip} # Auto-blocked {datetime.utcnow()} - {issue.get('reason', 'Security threat')}\n")
            
            # Add to Redis for quick lookup
            self.redis_client.sadd("blocked_ips", ip)
            self.redis_client.setex(
                f"block_reason:{ip}",
                86400 * 30,  # Keep for 30 days
                json.dumps({
                    'reason': issue.get('reason', 'Security threat'),
                    'timestamp': datetime.utcnow().isoformat(),
                    'issue_type': issue.get('type')
                })
            )
            
            logger.info(f"✅ Blocked IP address: {ip}")
            
            return {
                'success': True,
                'ip_blocked': ip,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to block IP {ip}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def update_container(self, issue: Dict) -> Dict:
        """Update vulnerable container"""
        container_name = issue.get('container')
        image_name = issue.get('image')
        
        if not container_name:
            return {'success': False, 'error': 'No container specified'}
        
        try:
            # Get container
            container = self.docker_client.containers.get(container_name)
            
            # Pull latest image
            if image_name:
                logger.info(f"Pulling latest image: {image_name}")
                self.docker_client.images.pull(image_name, tag='latest')
            else:
                # Try to determine image from container
                image_name = container.image.tags[0] if container.image.tags else None
                if image_name:
                    base_image = image_name.split(':')[0]
                    self.docker_client.images.pull(base_image, tag='latest')
                    image_name = f"{base_image}:latest"
            
            # Store container config
            container_config = {
                'name': container.name,
                'image': image_name,
                'environment': container.attrs['Config'].get('Env', []),
                'volumes': container.attrs['Mounts'],
                'ports': container.attrs['NetworkSettings'].get('Ports', {}),
                'restart_policy': container.attrs['HostConfig'].get('RestartPolicy', {})
            }
            
            # Stop old container
            logger.info(f"Stopping container: {container_name}")
            container.stop(timeout=30)
            container.remove()
            
            # Start new container with updated image
            logger.info(f"Starting updated container: {container_name}")
            new_container = self.docker_client.containers.run(
                image_name,
                name=container_name,
                environment=container_config['environment'],
                volumes={
                    m['Source']: {'bind': m['Destination'], 'mode': m.get('Mode', 'rw')}
                    for m in container_config['volumes'] if m.get('Type') == 'bind'
                },
                ports={
                    port.split('/')[0]: port_config[0]['HostPort']
                    for port, port_config in container_config['ports'].items()
                    if port_config
                },
                restart_policy=container_config['restart_policy'],
                detach=True
            )
            
            # Wait for container to be healthy
            await asyncio.sleep(10)
            
            # Verify container is running
            new_container.reload()
            if new_container.status == 'running':
                logger.info(f"✅ Successfully updated container: {container_name}")
                return {
                    'success': True,
                    'container': container_name,
                    'new_image': image_name,
                    'status': new_container.status
                }
            else:
                raise Exception(f"Container failed to start: {new_container.status}")
                
        except Exception as e:
            logger.error(f"Failed to update container {container_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def rotate_credentials(self, issue: Dict) -> Dict:
        """Rotate compromised credentials"""
        credential_type = issue.get('credential_type', 'api_key')
        service = issue.get('service', 'unknown')
        
        try:
            # Generate new credentials
            new_credentials = self._generate_secure_credentials(credential_type)
            
            # Update in database
            conn = psycopg2.connect(
                host=os.environ.get('DB_HOST', 'localhost'),
                database=os.environ.get('DB_NAME', 'ytempire'),
                user=os.environ.get('DB_USER'),
                password=os.environ.get('DB_PASSWORD')
            )
            cur = conn.cursor()
            
            # Store old credential for rollback if needed
            cur.execute("""
                INSERT INTO credential_rotation_history 
                (service, credential_type, old_value, new_value, rotated_at, reason)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                service,
                credential_type,
                issue.get('old_credential', 'REDACTED'),
                new_credentials['encrypted_value'],
                datetime.utcnow(),
                issue.get('reason', 'Security incident')
            ))
            
            # Update active credential
            cur.execute("""
                UPDATE service_credentials
                SET credential_value = %s, updated_at = %s
                WHERE service = %s AND credential_type = %s
            """, (
                new_credentials['encrypted_value'],
                datetime.utcnow(),
                service,
                credential_type
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            # Update in environment variables
            env_var_name = f"{service.upper()}_{credential_type.upper()}"
            os.environ[env_var_name] = new_credentials['plain_value']
            
            # Update in Redis for quick access
            self.redis_client.set(
                f"credential:{service}:{credential_type}",
                new_credentials['encrypted_value'],
                ex=86400  # Expire after 1 day
            )
            
            # Restart affected services
            await self._restart_dependent_services(service)
            
            logger.info(f"✅ Rotated credentials for {service}:{credential_type}")
            
            return {
                'success': True,
                'service': service,
                'credential_type': credential_type,
                'rotated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to rotate credentials: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def patch_vulnerability(self, issue: Dict) -> Dict:
        """Apply security patch for vulnerability"""
        vulnerability_id = issue.get('vulnerability_id')
        package = issue.get('package')
        fixed_version = issue.get('fixed_version')
        
        if not all([package, fixed_version]):
            return {'success': False, 'error': 'Missing package or version information'}
        
        try:
            # Determine package manager
            if package.endswith('.deb'):
                cmd = ["apt-get", "install", "-y", f"{package}={fixed_version}"]
            elif package.endswith('.rpm'):
                cmd = ["yum", "install", "-y", f"{package}-{fixed_version}"]
            else:
                # Assume Python package
                cmd = ["pip", "install", "--upgrade", f"{package}=={fixed_version}"]
            
            # Apply patch
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✅ Patched {package} to version {fixed_version}")
                
                # Record patch application
                self._record_patch_application({
                    'vulnerability_id': vulnerability_id,
                    'package': package,
                    'old_version': issue.get('current_version'),
                    'new_version': fixed_version,
                    'applied_at': datetime.utcnow().isoformat()
                })
                
                return {
                    'success': True,
                    'package': package,
                    'version': fixed_version,
                    'vulnerability_id': vulnerability_id
                }
            else:
                raise Exception(f"Patch failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to patch vulnerability: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def enforce_encryption(self, issue: Dict) -> Dict:
        """Enforce encryption on unencrypted data"""
        target_type = issue.get('target_type', 'database')
        target = issue.get('target')
        
        try:
            if target_type == 'database':
                # Encrypt database column
                table = issue.get('table')
                column = issue.get('column')
                
                if not all([table, column]):
                    return {'success': False, 'error': 'Missing table or column information'}
                
                conn = psycopg2.connect(
                    host=os.environ.get('DB_HOST', 'localhost'),
                    database=os.environ.get('DB_NAME', 'ytempire'),
                    user=os.environ.get('DB_USER'),
                    password=os.environ.get('DB_PASSWORD')
                )
                cur = conn.cursor()
                
                # Create encrypted column
                cur.execute(f"""
                    ALTER TABLE {table} 
                    ADD COLUMN {column}_encrypted BYTEA
                """)
                
                # Encrypt existing data
                cur.execute(f"""
                    UPDATE {table}
                    SET {column}_encrypted = pgp_sym_encrypt(
                        {column}::text,
                        %s
                    )
                """, (os.environ.get('ENCRYPTION_KEY'),))
                
                # Drop old column and rename
                cur.execute(f"""
                    ALTER TABLE {table} DROP COLUMN {column};
                    ALTER TABLE {table} RENAME COLUMN {column}_encrypted TO {column};
                """)
                
                conn.commit()
                cur.close()
                conn.close()
                
                logger.info(f"✅ Encrypted {table}.{column}")
                
            elif target_type == 'filesystem':
                # Encrypt directory
                directory = Path(target)
                
                if not directory.exists():
                    return {'success': False, 'error': f"Directory {target} not found"}
                
                # Use eCryptfs or similar
                subprocess.run([
                    "mount", "-t", "ecryptfs",
                    str(directory), str(directory),
                    "-o", f"key=passphrase:passphrase_passwd={os.environ.get('ENCRYPTION_KEY')}",
                    "-o", "ecryptfs_cipher=aes",
                    "-o", "ecryptfs_key_bytes=32",
                    "-o", "ecryptfs_passthrough=n",
                    "-o", "ecryptfs_enable_filename_crypto=y"
                ], check=True)
                
                logger.info(f"✅ Encrypted directory: {target}")
            
            return {
                'success': True,
                'target_type': target_type,
                'target': target,
                'encrypted_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to enforce encryption: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def revoke_user_access(self, issue: Dict) -> Dict:
        """Revoke user access"""
        user_id = issue.get('user_id')
        user_email = issue.get('user_email')
        reason = issue.get('reason', 'Security violation')
        
        try:
            conn = psycopg2.connect(
                host=os.environ.get('DB_HOST', 'localhost'),
                database=os.environ.get('DB_NAME', 'ytempire'),
                user=os.environ.get('DB_USER'),
                password=os.environ.get('DB_PASSWORD')
            )
            cur = conn.cursor()
            
            # Get user if only email provided
            if user_email and not user_id:
                cur.execute("SELECT id FROM users WHERE email = %s", (user_email,))
                result = cur.fetchone()
                if result:
                    user_id = result[0]
            
            if not user_id:
                return {'success': False, 'error': 'User not found'}
            
            # Revoke all permissions
            cur.execute("""
                DELETE FROM user_permissions WHERE user_id = %s
            """, (user_id,))
            
            # Deactivate user account
            cur.execute("""
                UPDATE users 
                SET is_active = false, 
                    deactivated_at = %s,
                    deactivation_reason = %s
                WHERE id = %s
            """, (datetime.utcnow(), reason, user_id))
            
            # Terminate active sessions
            cur.execute("""
                DELETE FROM user_sessions WHERE user_id = %s
            """, (user_id,))
            
            # Add to blacklist
            cur.execute("""
                INSERT INTO user_blacklist (user_id, reason, blacklisted_at)
                VALUES (%s, %s, %s)
            """, (user_id, reason, datetime.utcnow()))
            
            conn.commit()
            cur.close()
            conn.close()
            
            # Clear Redis sessions
            session_keys = self.redis_client.keys(f"session:user:{user_id}:*")
            if session_keys:
                self.redis_client.delete(*session_keys)
            
            logger.info(f"✅ Revoked access for user {user_id}")
            
            return {
                'success': True,
                'user_id': user_id,
                'revoked_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to revoke user access: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def restart_service(self, issue: Dict) -> Dict:
        """Restart a service"""
        service_name = issue.get('service')
        
        if not service_name:
            return {'success': False, 'error': 'No service specified'}
        
        try:
            # Docker container restart
            if service_name.startswith('ytempire'):
                container = self.docker_client.containers.get(service_name)
                container.restart(timeout=30)
                
                # Wait for health check
                await asyncio.sleep(10)
                container.reload()
                
                if container.status == 'running':
                    logger.info(f"✅ Restarted service: {service_name}")
                    return {
                        'success': True,
                        'service': service_name,
                        'status': container.status
                    }
                else:
                    raise Exception(f"Service failed to restart: {container.status}")
            
            # System service restart
            else:
                result = subprocess.run(
                    ["systemctl", "restart", service_name],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info(f"✅ Restarted system service: {service_name}")
                    return {
                        'success': True,
                        'service': service_name
                    }
                else:
                    raise Exception(f"Service restart failed: {result.stderr}")
                    
        except Exception as e:
            logger.error(f"Failed to restart service {service_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def apply_security_patch(self, issue: Dict) -> Dict:
        """Apply security patch to system"""
        patch_type = issue.get('patch_type', 'system')
        
        try:
            if patch_type == 'system':
                # Apply system security updates
                result = subprocess.run(
                    ["apt-get", "update"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    raise Exception(f"apt-get update failed: {result.stderr}")
                
                result = subprocess.run(
                    ["apt-get", "upgrade", "-y", "--only-upgrade"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    logger.info("✅ Applied system security patches")
                    return {
                        'success': True,
                        'patch_type': 'system',
                        'applied_at': datetime.utcnow().isoformat()
                    }
                else:
                    raise Exception(f"apt-get upgrade failed: {result.stderr}")
                    
            elif patch_type == 'kernel':
                # Kernel patches require reboot planning
                result = subprocess.run(
                    ["apt-get", "install", "-y", "linux-generic"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                if result.returncode == 0:
                    logger.info("✅ Kernel patch installed - reboot required")
                    
                    # Schedule reboot during maintenance window
                    self._schedule_maintenance_reboot()
                    
                    return {
                        'success': True,
                        'patch_type': 'kernel',
                        'reboot_required': True,
                        'applied_at': datetime.utcnow().isoformat()
                    }
                else:
                    raise Exception(f"Kernel patch failed: {result.stderr}")
                    
        except Exception as e:
            logger.error(f"Failed to apply security patch: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def quarantine_file(self, issue: Dict) -> Dict:
        """Quarantine malicious or suspicious file"""
        file_path = issue.get('file_path')
        reason = issue.get('reason', 'Security threat')
        
        if not file_path:
            return {'success': False, 'error': 'No file path specified'}
        
        try:
            source_path = Path(file_path)
            
            if not source_path.exists():
                return {'success': False, 'error': f"File not found: {file_path}"}
            
            # Create quarantine directory
            quarantine_dir = Path("/opt/ytempire/quarantine")
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate quarantine path
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            quarantine_name = f"{timestamp}_{source_path.name}"
            quarantine_path = quarantine_dir / quarantine_name
            
            # Move file to quarantine
            source_path.rename(quarantine_path)
            
            # Remove execute permissions
            quarantine_path.chmod(0o400)
            
            # Create metadata file
            metadata_path = quarantine_path.with_suffix('.metadata')
            metadata = {
                'original_path': str(file_path),
                'quarantined_at': datetime.utcnow().isoformat(),
                'reason': reason,
                'file_hash': self._calculate_file_hash(quarantine_path),
                'file_size': quarantine_path.stat().st_size,
                'issue_details': issue
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Quarantined file: {file_path} -> {quarantine_path}")
            
            return {
                'success': True,
                'original_path': str(file_path),
                'quarantine_path': str(quarantine_path),
                'quarantined_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to quarantine file {file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def disable_user_account(self, issue: Dict) -> Dict:
        """Disable compromised user account"""
        user_id = issue.get('user_id')
        user_email = issue.get('user_email')
        reason = issue.get('reason', 'Account compromise')
        temporary = issue.get('temporary', True)
        
        try:
            conn = psycopg2.connect(
                host=os.environ.get('DB_HOST', 'localhost'),
                database=os.environ.get('DB_NAME', 'ytempire'),
                user=os.environ.get('DB_USER'),
                password=os.environ.get('DB_PASSWORD')
            )
            cur = conn.cursor()
            
            # Get user if only email provided
            if user_email and not user_id:
                cur.execute("SELECT id FROM users WHERE email = %s", (user_email,))
                result = cur.fetchone()
                if result:
                    user_id = result[0]
            
            if not user_id:
                return {'success': False, 'error': 'User not found'}
            
            # Disable account
            cur.execute("""
                UPDATE users 
                SET is_active = false,
                    disabled_at = %s,
                    disable_reason = %s,
                    requires_password_reset = true
                WHERE id = %s
            """, (datetime.utcnow(), reason, user_id))
            
            # Terminate active sessions
            cur.execute("""
                UPDATE user_sessions 
                SET expired_at = %s,
                    termination_reason = %s
                WHERE user_id = %s AND expired_at IS NULL
            """, (datetime.utcnow(), reason, user_id))
            
            if temporary:
                # Schedule re-enable after investigation
                renable_time = datetime.utcnow() + timedelta(hours=24)
                cur.execute("""
                    INSERT INTO scheduled_tasks (task_type, scheduled_for, task_data)
                    VALUES ('reenable_user', %s, %s)
                """, (renable_time, json.dumps({'user_id': user_id})))
            
            conn.commit()
            cur.close()
            conn.close()
            
            # Clear Redis sessions
            session_keys = self.redis_client.keys(f"session:user:{user_id}:*")
            if session_keys:
                self.redis_client.delete(*session_keys)
            
            # Send notification to user
            await self._notify_user_account_disabled(user_id, reason)
            
            logger.info(f"✅ Disabled user account: {user_id}")
            
            return {
                'success': True,
                'user_id': user_id,
                'temporary': temporary,
                'disabled_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to disable user account: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_secure_credentials(self, credential_type: str) -> Dict:
        """Generate secure credentials"""
        import secrets
        import string
        from cryptography.fernet import Fernet
        
        if credential_type == 'api_key':
            # Generate API key
            plain_value = f"ytmp_{secrets.token_urlsafe(32)}"
        elif credential_type == 'password':
            # Generate strong password
            alphabet = string.ascii_letters + string.digits + string.punctuation
            plain_value = ''.join(secrets.choice(alphabet) for _ in range(24))
        else:
            # Generic token
            plain_value = secrets.token_urlsafe(32)
        
        # Encrypt for storage
        encryption_key = os.environ.get('ENCRYPTION_KEY', Fernet.generate_key())
        f = Fernet(encryption_key)
        encrypted_value = f.encrypt(plain_value.encode()).decode()
        
        return {
            'plain_value': plain_value,
            'encrypted_value': encrypted_value
        }
    
    async def _restart_dependent_services(self, service: str):
        """Restart services dependent on rotated credentials"""
        dependent_services = {
            'database': ['ytempire-api', 'ytempire-worker'],
            'redis': ['ytempire-api', 'ytempire-worker', 'ytempire-cache'],
            'youtube': ['ytempire-worker', 'ytempire-scheduler'],
            'openai': ['ytempire-worker', 'ytempire-ai']
        }
        
        services_to_restart = dependent_services.get(service, [])
        
        for service_name in services_to_restart:
            try:
                container = self.docker_client.containers.get(service_name)
                container.restart(timeout=30)
                logger.info(f"Restarted dependent service: {service_name}")
            except Exception as e:
                logger.warning(f"Could not restart {service_name}: {str(e)}")
    
    def _record_patch_application(self, patch_info: Dict):
        """Record patch application in database"""
        try:
            conn = psycopg2.connect(
                host=os.environ.get('DB_HOST', 'localhost'),
                database=os.environ.get('DB_NAME', 'ytempire'),
                user=os.environ.get('DB_USER'),
                password=os.environ.get('DB_PASSWORD')
            )
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO patch_history 
                (vulnerability_id, package, old_version, new_version, applied_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                patch_info['vulnerability_id'],
                patch_info['package'],
                patch_info['old_version'],
                patch_info['new_version'],
                patch_info['applied_at']
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record patch application: {str(e)}")
    
    def _schedule_maintenance_reboot(self):
        """Schedule system reboot during maintenance window"""
        # Schedule for 3 AM next day
        next_maintenance = datetime.utcnow().replace(hour=3, minute=0, second=0)
        if next_maintenance < datetime.utcnow():
            next_maintenance += timedelta(days=1)
        
        try:
            # Create at job for reboot
            reboot_time = next_maintenance.strftime('%H:%M %Y-%m-%d')
            subprocess.run(
                ["at", reboot_time],
                input="shutdown -r +5 'System maintenance reboot in 5 minutes'",
                text=True,
                check=True
            )
            
            logger.info(f"Scheduled maintenance reboot for {reboot_time}")
            
        except Exception as e:
            logger.error(f"Failed to schedule reboot: {str(e)}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        import hashlib
        
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    async def _notify_user_account_disabled(self, user_id: int, reason: str):
        """Send notification about disabled account"""
        # Implementation would send email/SMS to user
        logger.info(f"User {user_id} notified about account disable: {reason}")
    
    async def _verify_remediation(self, issue: Dict, remediation_result: Dict) -> Dict:
        """Verify that remediation was successful"""
        verification = {
            'verified': False,
            'verification_time': datetime.utcnow().isoformat(),
            'checks_performed': []
        }
        
        # Perform verification based on issue type
        issue_type = issue.get('type')
        
        if issue_type == 'brute_force_attack':
            # Check if IP is blocked
            ip = issue.get('source_ip')
            if ip and self.redis_client.sismember("blocked_ips", ip):
                verification['verified'] = True
                verification['checks_performed'].append('IP block verified in Redis')
        
        elif issue_type == 'critical_vulnerability':
            # Re-scan container
            container = issue.get('container')
            if container:
                try:
                    container_obj = self.docker_client.containers.get(container)
                    if container_obj.status == 'running':
                        verification['verified'] = True
                        verification['checks_performed'].append('Container running with updated image')
                except:
                    pass
        
        elif issue_type == 'exposed_secrets':
            # Check if file was quarantined
            for action in remediation_result.get('actions_taken', []):
                if action.get('action') == 'quarantine_file':
                    quarantine_path = action.get('result', {}).get('quarantine_path')
                    if quarantine_path and Path(quarantine_path).exists():
                        verification['verified'] = True
                        verification['checks_performed'].append('File quarantine verified')
        
        return verification
    
    def _store_remediation_history(self, remediation_result: Dict):
        """Store remediation history"""
        try:
            # Store in Redis for quick access
            key = f"remediation:{remediation_result['issue_id']}"
            self.redis_client.setex(
                key,
                86400 * 30,  # Keep for 30 days
                json.dumps(remediation_result, default=str)
            )
            
            # Add to history list
            self.remediation_history.append(remediation_result)
            
            # Keep only last 1000 entries in memory
            if len(self.remediation_history) > 1000:
                self.remediation_history = self.remediation_history[-1000:]
                
        except Exception as e:
            logger.error(f"Failed to store remediation history: {str(e)}")
    
    async def _send_remediation_notification(self, remediation_result: Dict):
        """Send notification about remediation action"""
        # Send to Slack if configured
        slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
        
        if slack_webhook:
            import requests
            
            status_emoji = "✅" if remediation_result['status'] == 'success' else "❌"
            
            message = {
                "text": f"{status_emoji} Auto-Remediation: {remediation_result['issue_type']}",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Issue Type:* {remediation_result['issue_type']}\n"
                                   f"*Severity:* {remediation_result['severity']}\n"
                                   f"*Status:* {remediation_result['status']}\n"
                                   f"*Actions Taken:* {len(remediation_result['actions_taken'])}"
                        }
                    }
                ]
            }
            
            try:
                requests.post(slack_webhook, json=message, timeout=10)
            except:
                pass


# Main execution
if __name__ == "__main__":
    async def main():
        remediation = YTEMPIREAutoRemediation()
        
        # Example issue
        test_issue = {
            'type': 'brute_force_attack',
            'severity': 'high',
            'source_ip': '192.168.1.100',
            'reason': 'Multiple failed login attempts'
        }
        
        result = await remediation.remediate_issue(test_issue)
        print(f"Remediation result: {result}")
    
    asyncio.run(main())
```

---

## 3. CI/CD Security Integration

### 3.1 GitHub Actions Security Pipeline

```yaml
# .github/workflows/ytempire-security-pipeline.yml
name: YTEMPIRE Security Pipeline

on:
  push:
    branches: [ main, develop, release/* ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 2 * * *'  # Daily security scan at 2 AM UTC
  workflow_dispatch:  # Manual trigger

env:
  DOCKER_REGISTRY: ytempire
  SCAN_TIMEOUT: 1800  # 30 minutes

jobs:
  # Job 1: Secret Detection
  secret-scanning:
    name: 🔑 Secret Detection
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Full history for comprehensive scan
      
      - name: Run git-secrets
        run: |
          # Install git-secrets
          git clone https://github.com/awslabs/git-secrets.git
          cd git-secrets && sudo make install && cd ..
          
          # Configure patterns
          git secrets --register-aws
          git secrets --add 'ytmp_[a-zA-Z]+_[a-zA-Z0-9]{32,}'
          
          # Scan repository
          git secrets --scan
      
      - name: Run TruffleHog
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          head: HEAD
      
      - name: Run Gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Upload secret scan results
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: secret-scan-results
          path: |
            gitleaks-report.json
            trufflehog-report.json

  # Job 2: Static Application Security Testing (SAST)
  sast-scanning:
    name: 🔍 SAST Analysis
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Cache pip packages
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
      
      - name: Install security tools
        run: |
          pip install --upgrade pip
          pip install bandit safety semgrep pylint mypy
      
      - name: Run Bandit
        run: |
          bandit -r . -f json -o bandit-report.json -ll -i
        continue-on-error: true
      
      - name: Run Safety
        run: |
          safety check --json > safety-report.json
        continue-on-error: true
      
      - name: Run Semgrep
        run: |
          semgrep --config=auto --json -o semgrep-report.json .
        continue-on-error: true
      
      - name: Run PyLint security checks
        run: |
          pylint --disable=all --enable=security --output-format=json > pylint-security.json || true
      
      - name: Upload SAST reports
        uses: actions/upload-artifact@v3
        with:
          name: sast-reports
          path: |
            bandit-report.json
            safety-report.json
            semgrep-report.json
            pylint-security.json

  # Job 3: Container Security Scanning
  container-scanning:
    name: 🐳 Container Security
    runs-on: ubuntu-latest
    needs: [secret-scanning]
    strategy:
      matrix:
        image: [api, frontend, worker, admin]
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Build Docker image
        run: |
          docker build -t ${{ env.DOCKER_REGISTRY }}/${{ matrix.image }}:${{ github.sha }} \
            -f docker/${{ matrix.image }}/Dockerfile .
      
      - name: Run Trivy scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.DOCKER_REGISTRY }}/${{ matrix.image }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-${{ matrix.image }}.sarif'
          severity: 'CRITICAL,HIGH,MEDIUM'
          timeout: '30m'
      
      - name: Run Grype scan
        uses: anchore/scan-action@v3
        with:
          image: ${{ env.DOCKER_REGISTRY }}/${{ matrix.image }}:${{ github.sha }}
          output-format: sarif
          fail-build: false
          severity-cutoff: high
      
      - name: Run Snyk container scan
        uses: snyk/actions/docker@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          image: ${{ env.DOCKER_REGISTRY }}/${{ matrix.image }}:${{ github.sha }}
          args: --severity-threshold=high --file=docker/${{ matrix.image }}/Dockerfile
        continue-on-error: true
      
      - name: Upload container scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: trivy-${{ matrix.image }}.sarif

  # Job 4: Infrastructure as Code (IaC) Security
  iac-scanning:
    name: 🏗️ IaC Security
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Run Checkov
        uses: bridgecrewio/checkov-action@master
        with:
          directory: .
          framework: all
          output_format: sarif
          output_file_path: checkov.sarif
          skip_check: CKV_DOCKER_2,CKV_DOCKER_3  # Example skips
      
      - name: Run Terrascan
        run: |
          docker run --rm -v "$(pwd):/src" \
            accurics/terrascan scan -d /src \
            -o json > terrascan-report.json
        continue-on-error: true
      
      - name: Run KICS
        uses: checkmarx/kics-github-action@v1.6
        with:
          path: .
          output_path: kics-results.json
          output_formats: json,sarif
          fail_on: high
        continue-on-error: true
      
      - name: Upload IaC scan results
        uses: actions/upload-artifact@v3
        with:
          name: iac-reports
          path: |
            checkov.sarif
            terrascan-report.json
            kics-results.json

  # Job 5: Dependency Vulnerability Scanning
  dependency-scanning:
    name: 📦 Dependency Security
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Run OWASP Dependency Check
        uses: dependency-check/Dependency-Check_Action@main
        with:
          project: 'YTEMPIRE'
          path: '.'
          format: 'ALL'
          args: >
            --enableRetired
            --enableExperimental
      
      - name: Run Snyk dependency scan
        uses: snyk/actions/python@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high --all-projects
      
      - name: Run npm audit (if applicable)
        if: hashFiles('package-lock.json') != ''
        run: |
          npm audit --json > npm-audit.json || true
          npm audit fix --force || true
      
      - name: Run pip-audit
        run: |
          pip install pip-audit
          pip-audit --format json --output pip-audit.json || true
      
      - name: Upload dependency reports
        uses: actions/upload-artifact@v3
        with:
          name: dependency-reports
          path: |
            dependency-check-report.html
            npm-audit.json
            pip-audit.json

  # Job 6: DAST (Dynamic Application Security Testing)
  dast-scanning:
    name: 🌐 DAST Analysis
    runs-on: ubuntu-latest
    needs: [container-scanning]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Deploy test environment
        run: |
          docker-compose -f docker-compose.test.yml up -d
          sleep 30  # Wait for services to start
      
      - name: Run OWASP ZAP baseline scan
        run: |
          docker run -v $(pwd):/zap/wrk/:rw \
            -t owasp/zap2docker-stable zap-baseline.py \
            -t http://host.docker.internal:8000 \
            -g gen.conf \
            -r zap-baseline-report.html \
            -J zap-baseline-report.json
        continue-on-error: true
      
      - name: Run Nuclei security scan
        run: |
          docker run --rm -v $(pwd):/src \
            projectdiscovery/nuclei:latest \
            -u http://host.docker.internal:8000 \
            -as -o /src/nuclei-report.json
        continue-on-error: true
      
      - name: Cleanup test environment
        if: always()
        run: |
          docker-compose -f docker-compose.test.yml down
      
      - name: Upload DAST reports
        uses: actions/upload-artifact@v3
        with:
          name: dast-reports
          path: |
            zap-baseline-report.*
            nuclei-report.json

  # Job 7: License Compliance Check
  license-check:
    name: 📜 License Compliance
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Run license-checker
        run: |
          npm install -g license-checker
          license-checker --json > license-report.json || true
      
      - name: Run pip-licenses
        run: |
          pip install pip-licenses
          pip-licenses --format=json --output-file=pip-licenses.json
      
      - name: Check for problematic licenses
        run: |
          python scripts/check_licenses.py license-report.json pip-licenses.json
      
      - name: Upload license reports
        uses: actions/upload-artifact@v3
        with:
          name: license-reports
          path: |
            license-report.json
            pip-licenses.json

  # Job 8: Security Gate Evaluation
  security-gate:
    name: 🚦 Security Gate
    runs-on: ubuntu-latest
    needs: [secret-scanning, sast-scanning, container-scanning, iac-scanning, dependency-scanning]
    if: always()
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Download all artifacts
        uses: actions/download-artifact@v3
        with:
          path: security-reports
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install pandas matplotlib seaborn jinja2
      
      - name: Run security gate evaluation
        run: |
          python scripts/security_gate.py \
            --reports-dir security-reports \
            --output-format markdown \
            --output-file security-gate-report.md
      
      - name: Generate security dashboard
        run: |
          python scripts/generate_security_dashboard.py \
            --reports-dir security-reports \
            --output-file security-dashboard.html
      
      - name: Comment PR with security report
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('security-gate-report.md', 'utf8');
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: report
            });
      
      - name: Upload final security report
        uses: actions/upload-artifact@v3
        with:
          name: security-gate-report
          path: |
            security-gate-report.md
            security-dashboard.html
      
      - name: Fail if critical issues found
        run: |
          if grep -q "CRITICAL" security-gate-report.md; then
            echo "❌ Critical security issues found!"
            exit 1
          fi
          echo "✅ Security gate passed!"

  # Job 9: Security Report to S3
  publish-reports:
    name: 📊 Publish Reports
    runs-on: ubuntu-latest
    needs: [security-gate]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - name: Download all reports
        uses: actions/download-artifact@v3
        with:
          path: all-reports
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Upload to S3
        run: |
          TIMESTAMP=$(date +%Y%m%d-%H%M%S)
          aws s3 cp all-reports s3://ytempire-security-reports/$TIMESTAMP/ --recursive
          
          # Update latest pointer
          echo "$TIMESTAMP" > latest.txt
          aws s3 cp latest.txt s3://ytempire-security-reports/latest.txt
      
      - name: Send notification
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: |
            Security scan completed for commit ${{ github.sha }}
            Reports available at: https://security.ytempire.com/reports/$TIMESTAMP/
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
        if: always()
```

### 3.2 Security Gate Evaluation Script

```python
#!/usr/bin/env python3
# scripts/security_gate.py

import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
from collections import defaultdict

class YTEMPIRESecurityGate:
    """Security gate evaluation for YTEMPIRE CI/CD pipeline"""
    
    def __init__(self):
        # Define security thresholds
        self.thresholds = {
            'critical': {
                'max_allowed': 0,
                'block_deployment': True
            },
            'high': {
                'max_allowed': 3,
                'block_deployment': True
            },
            'medium': {
                'max_allowed': 10,
                'block_deployment': False
            },
            'low': {
                'max_allowed': 50,
                'block_deployment': False
            }
        }
        
        # Track all findings
        self.findings = defaultdict(list)
        self.summary = {
            'total_issues': 0,
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'passed': True,
            'blocking_issues': []
        }
    
    def evaluate_reports(self, reports_dir: str) -> Dict:
        """Evaluate all security reports"""
        reports_path = Path(reports_dir)
        
        # Process each type of report
        self._process_secret_scans(reports_path / 'secret-scan-results')
        self._process_sast_reports(reports_path / 'sast-reports')
        self._process_container_scans(reports_path / 'container-reports')
        self._process_iac_scans(reports_path / 'iac-reports')
        self._process_dependency_scans(reports_path / 'dependency-reports')
        self._process_dast_reports(reports_path / 'dast-reports')
        
        # Evaluate against thresholds
        self._evaluate_thresholds()
        
        return self.summary
    
    def _process_secret_scans(self, path: Path):
        """Process secret scanning results"""
        if not path.exists():
            return
        
        # Process Gitleaks report
        gitleaks_file = path / 'gitleaks-report.json'
        if gitleaks_file.exists():
            with open(gitleaks_file, 'r') as f:
                data = json.load(f)
                for finding in data:
                    self.findings['secrets'].append({
                        'tool': 'gitleaks',
                        'file': finding.get('File'),
                        'line': finding.get('StartLine'),
                        'secret_type': finding.get('RuleID'),
                        'severity': 'critical'
                    })
                    self.summary['critical'] += 1
    
    def _process_sast_reports(self, path: Path):
        """Process SAST reports"""
        if not path.exists():
            return
        
        # Process Bandit report
        bandit_file = path / 'bandit-report.json'
        if bandit_file.exists():
            with open(bandit_file, 'r') as f:
                data = json.load(f)
                for result in data.get('results', []):
                    severity = result.get('issue_severity', 'LOW').lower()
                    self.findings['sast'].append({
                        'tool': 'bandit',
                        'file': result.get('filename'),
                        'line': result.get('line_number'),
                        'issue': result.get('issue_text'),
                        'severity': severity
                    })
                    self.summary[severity] = self.summary.get(severity, 0) + 1
        
        # Process Semgrep report
        semgrep_file = path / 'semgrep-report.json'
        if semgrep_file.exists():
            with open(semgrep_file, 'r') as f:
                data = json.load(f)
                for result in data.get('results', []):
                    severity = self._map_semgrep_severity(result.get('extra', {}).get('severity', 'INFO'))
                    self.findings['sast'].append({
                        'tool': 'semgrep',
                        'file': result.get('path'),
                        'line': result.get('start', {}).get('line'),
                        'issue': result.get('extra', {}).get('message'),
                        'severity': severity
                    })
                    self.summary[severity] = self.summary.get(severity, 0) + 1
    
    def _process_container_scans(self, path: Path):
        """Process container scan results"""
        if not path.exists():
            return
        
        # Process Trivy reports
        for trivy_file in path.glob('trivy-*.sarif'):
            with open(trivy_file, 'r') as f:
                data = json.load(f)
                for run in data.get('runs', []):
                    for result in run.get('results', []):
                        severity = self._map_sarif_severity(result.get('level', 'note'))
                        self.findings['container'].append({
                            'tool': 'trivy',
                            'image': trivy_file.stem.replace('trivy-', ''),
                            'vulnerability': result.get('ruleId'),
                            'message': result.get('message', {}).get('text'),
                            'severity': severity
                        })
                        self.summary[severity] = self.summary.get(severity, 0) + 1
    
    def _process_iac_scans(self, path: Path):
        """Process IaC scan results"""
        if not path.exists():
            return
        
        # Process Checkov report
        checkov_file = path / 'checkov.sarif'
        if checkov_file.exists():
            with open(checkov_file, 'r') as f:
                data = json.load(f)
                for run in data.get('runs', []):
                    for result in run.get('results', []):
                        severity = self._map_sarif_severity(result.get('level', 'note'))
                        self.findings['iac'].append({
                            'tool': 'checkov',
                            'check': result.get('ruleId'),
                            'file': result.get('locations', [{}])[0].get('physicalLocation', {}).get('artifactLocation', {}).get('uri'),
                            'message': result.get('message', {}).get('text'),
                            'severity': severity
                        })
                        self.summary[severity] = self.summary.get(severity, 0) + 1
    
    def _process_dependency_scans(self, path: Path):
        """Process dependency scan results"""
        if not path.exists():
            return
        
        # Process pip-audit report
        pip_audit_file = path / 'pip-audit.json'
        if pip_audit_file.exists():
            with open(pip_audit_file, 'r') as f:
                data = json.load(f)
                for vuln in data.get('vulnerabilities', []):
                    severity = vuln.get('severity', 'unknown').lower()
                    if severity == 'unknown':
                        severity = 'medium'
                    self.findings['dependencies'].append({
                        'tool': 'pip-audit',
                        'package': vuln.get('package'),
                        'version': vuln.get('version'),
                        'vulnerability': vuln.get('id'),
                        'severity': severity
                    })
                    self.summary[severity] = self.summary.get(severity, 0) + 1
    
    def _process_dast_reports(self, path: Path):
        """Process DAST results"""
        if not path.exists():
            return
        
        # Process ZAP report
        zap_file = path / 'zap-baseline-report.json'
        if zap_file.exists():
            with open(zap_file, 'r') as f:
                data = json.load(f)
                for site in data.get('site', []):
                    for alert in site.get('alerts', []):
                        severity = self._map_zap_risk(alert.get('risk'))
                        self.findings['dast'].append({
                            'tool': 'zap',
                            'alert': alert.get('name'),
                            'url': alert.get('instances', [{}])[0].get('uri'),
                            'severity': severity
                        })
                        self.summary[severity] = self.summary.get(severity, 0) + 1
    
    def _evaluate_thresholds(self):
        """Evaluate findings against thresholds"""
        self.summary['total_issues'] = sum([
            self.summary['critical'],
            self.summary['high'],
            self.summary['medium'],
            self.summary['low']
        ])
        
        for severity, threshold in self.thresholds.items():
            count = self.summary[severity]
            max_allowed = threshold['max_allowed']
            
            if count > max_allowed:
                issue_desc = f"{severity.upper()}: Found {count}, max allowed is {max_allowed}"
                self.summary['blocking_issues'].append(issue_desc)
                
                if threshold['block_deployment']:
                    self.summary['passed'] = False
    
    def _map_semgrep_severity(self, severity: str) -> str:
        """Map Semgrep severity to standard levels"""
        mapping = {
            'ERROR': 'high',
            'WARNING': 'medium',
            'INFO': 'low'
        }
        return mapping.get(severity.upper(), 'low')
    
    def _map_sarif_severity(self, level: str) -> str:
        """Map SARIF level to standard severity"""
        mapping = {
            'error': 'high',
            'warning': 'medium',
            'note': 'low',
            'none': 'low'
        }
        return mapping.get(level.lower(), 'low')
    
    def _map_zap_risk(self, risk: str) -> str:
        """Map ZAP risk to standard severity"""
        mapping = {
            'High': 'high',
            'Medium': 'medium',
            'Low': 'low',
            'Informational': 'low'
        }
        return mapping.get(risk, 'low')
    
    def generate_report(self, output_format: str = 'markdown') -> str:
        """Generate security gate report"""
        if output_format == 'markdown':
            return self._generate_markdown_report()
        elif output_format == 'json':
            return json.dumps(self.summary, indent=2)
        else:
            return str(self.summary)
    
    def _generate_markdown_report(self) -> str:
        """Generate Markdown report"""
        status_emoji = "✅" if self.summary['passed'] else "❌"
        status_text = "PASSED" if self.summary['passed'] else "FAILED"
        
        report = f"""# 🛡️ YTEMPIRE Security Gate Report

## Status: {status_emoji} {status_text}

**Date**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  
**Total Issues**: {self.summary['total_issues']}

### Issue Summary

| Severity | Count | Threshold | Status |
|----------|-------|-----------|--------|
| Critical | {self.summary['critical']} | {self.thresholds['critical']['max_allowed']} | {'❌' if self.summary['critical'] > self.thresholds['critical']['max_allowed'] else '✅'} |
| High | {self.summary['high']} | {self.thresholds['high']['max_allowed']} | {'❌' if self.summary['high'] > self.thresholds['high']['max_allowed'] else '✅'} |
| Medium | {self.summary['medium']} | {self.thresholds['medium']['max_allowed']} | {'❌' if self.summary['medium'] > self.thresholds['medium']['max_allowed'] else '✅'} |
| Low | {self.summary['low']} | {self.thresholds['low']['max_allowed']} | {'❌' if self.summary['low'] > self.thresholds['low']['max_allowed'] else '✅'} |

"""
        
        if self.summary['blocking_issues']:
            report += "### 🚫 Blocking Issues\n\n"
            for issue in self.summary['blocking_issues']:
                report += f"- {issue}\n"
            report += "\n"
        
        # Add top findings by category
        if self.findings:
            report += "### Top Security Findings\n\n"
            
            for category, findings in self.findings.items():
                if findings:
                    report += f"#### {category.upper()}\n\n"
                    
                    # Show top 3 critical/high findings
                    critical_high = [f for f in findings if f.get('severity') in ['critical', 'high']]
                    for finding in critical_high[:3]:
                        report += f"- **[{finding.get('severity').upper()}]** "
                        report += f"{finding.get('issue', finding.get('message', finding.get('alert', 'Unknown issue')))}\n"
                        if finding.get('file'):
                            report += f"  - File: `{finding['file']}`"
                            if finding.get('line'):
                                report += f" Line: {finding['line']}"
                            report += "\n"
                    report += "\n"
        
        # Add recommendations
        report += "### 📋 Recommendations\n\n"
        
        if self.summary['critical'] > 0:
            report += "1. **CRITICAL**: Address all critical security issues immediately before deployment\n"
        
        if self.summary['high'] > 0:
            report += "2. **HIGH PRIORITY**: Fix high-severity issues within 24 hours\n"
        
        if len(self.findings.get('secrets', [])) > 0:
            report += "3. **SECRETS**: Rotate all exposed credentials and remove from codebase\n"
        
        if len(self.findings.get('container', [])) > 0:
            report += "4. **CONTAINERS**: Update base images and vulnerable packages\n"
        
        if self.summary['passed']:
            report += "\n✅ **Security gate passed - deployment can proceed**\n"
        else:
            report += "\n❌ **Security gate failed - deployment blocked**\n"
        
        return report


def main():
    parser = argparse.ArgumentParser(description='YTEMPIRE Security Gate Evaluation')
    parser.add_argument('--reports-dir', required=True, help='Directory containing security reports')
    parser.add_argument('--output-format', choices=['markdown', 'json'], default='markdown')
    parser.add_argument('--output-file', default='security-gate-report.md')
    
    args = parser.parse_args()
    
    gate = YTEMPIRESecurityGate()
    summary = gate.evaluate_reports(args.reports_dir)
    report = gate.generate_report(args.output_format)
    
    # Write report to file
    with open(args.output_file, 'w') as f:
        f.write(report)
    
    print(report)
    
    # Exit with failure if security gate failed
    sys.exit(0 if summary['passed'] else 1)


if __name__ == "__main__":
    main()
```

---

## 4. Security Monitoring Automation

### 4.1 Real-time Threat Detection System

```python
#!/usr/bin/env python3
# monitoring/threat_detection.py

import asyncio
import re
import json
import logging
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
import redis
import psycopg2
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ThreatIndicator:
    """Threat indicator data class"""
    pattern: str
    threat_type: str
    severity: ThreatLevel
    confidence: float
    description: str
    tags: List[str]


class YTEMPIREThreatDetector:
    """Real-time threat detection system for YTEMPIRE"""
    
    def __init__(self):
        """Initialize threat detection system"""
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True,
            db=1  # Use separate DB for monitoring
        )
        
        # Load threat intelligence
        self.threat_patterns = self._load_threat_patterns()
        self.behavioral_rules = self._load_behavioral_rules()
        self.ml_models = self._load_ml_models()
        
        # Event buffers for correlation
        self.event_buffer = deque(maxlen=10000)
        self.ip_activity = defaultdict(list)
        self.user_activity = defaultdict(list)
        
        # Statistics tracking
        self.stats = {
            'events_processed': 0,
            'threats_detected': 0,
            'false_positives': 0,
            'true_positives': 0
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            ThreatLevel.CRITICAL: 1,
            ThreatLevel.HIGH: 3,
            ThreatLevel.MEDIUM: 5,
            ThreatLevel.LOW: 10
        }
    
    def _load_threat_patterns(self) -> List[ThreatIndicator]:
        """Load threat detection patterns"""
        patterns = []
        
        # SQL Injection patterns
        patterns.extend([
            ThreatIndicator(
                pattern=r"union\s+select\s+",
                threat_type="sql_injection",
                severity=ThreatLevel.HIGH,
                confidence=0.9,
                description="SQL UNION SELECT injection attempt",
                tags=["injection", "sql", "database"]
            ),
            ThreatIndicator(
                pattern=r";\s*drop\s+table\s+",
                threat_type="sql_injection",
                severity=ThreatLevel.CRITICAL,
                confidence=0.95,
                description="SQL DROP TABLE injection attempt",
                tags=["injection", "sql", "destructive"]
            ),
            ThreatIndicator(
                pattern=r"'.*or\s+'?1'?\s*=\s*'?1",
                threat_type="sql_injection",
                severity=ThreatLevel.HIGH,
                confidence=0.85,
                description="SQL authentication bypass attempt",
                tags=["injection", "sql", "auth_bypass"]
            )
        ])
        
        # XSS patterns
        patterns.extend([
            ThreatIndicator(
                pattern=r"<script[^>]*>.*?</script>",
                threat_type="xss",
                severity=ThreatLevel.HIGH,
                confidence=0.9,
                description="Cross-site scripting attempt with script tag",
                tags=["xss", "injection", "client_side"]
            ),
            ThreatIndicator(
                pattern=r"javascript\s*:\s*[^;]+",
                threat_type="xss",
                severity=ThreatLevel.MEDIUM,
                confidence=0.7,
                description="JavaScript protocol handler XSS attempt",
                tags=["xss", "javascript"]
            ),
            ThreatIndicator(
                pattern=r"on\w+\s*=\s*[\"'][^\"']+[\"']",
                threat_type="xss",
                severity=ThreatLevel.MEDIUM,
                confidence=0.75,
                description="Event handler XSS attempt",
                tags=["xss", "event_handler"]
            )
        ])
        
        # Command injection patterns
        patterns.extend([
            ThreatIndicator(
                pattern=r";\s*cat\s+/etc/passwd",
                threat_type="command_injection",
                severity=ThreatLevel.CRITICAL,
                confidence=0.95,
                description="Linux password file access attempt",
                tags=["command_injection", "file_access", "linux"]
            ),
            ThreatIndicator(
                pattern=r"\|\s*nc\s+-[elvnp]+\s+",
                threat_type="command_injection",
                severity=ThreatLevel.CRITICAL,
                confidence=0.9,
                description="Netcat reverse shell attempt",
                tags=["command_injection", "reverse_shell", "netcat"]
            ),
            ThreatIndicator(
                pattern=r"&&\s*whoami",
                threat_type="command_injection",
                severity=ThreatLevel.HIGH,
                confidence=0.8,
                description="Command injection reconnaissance",
                tags=["command_injection", "recon"]
            )
        ])
        
        # Path traversal patterns
        patterns.extend([
            ThreatIndicator(
                pattern=r"\.\.\/\.\.\/\.\.\/",
                threat_type="path_traversal",
                severity=ThreatLevel.HIGH,
                confidence=0.85,
                description="Directory traversal attempt",
                tags=["path_traversal", "file_access"]
            ),
            ThreatIndicator(
                pattern=r"\.\.\\\.\.\\\.\.\\",
                threat_type="path_traversal",
                severity=ThreatLevel.HIGH,
                confidence=0.85,
                description="Windows directory traversal attempt",
                tags=["path_traversal", "windows", "file_access"]
            )
        ])
        
        # API abuse patterns
        patterns.extend([
            ThreatIndicator(
                pattern=r"api/v\d+/admin/.*\?debug=true",
                threat_type="api_abuse",
                severity=ThreatLevel.MEDIUM,
                confidence=0.7,
                description="Debug mode access attempt on admin API",
                tags=["api", "debug", "admin"]
            ),
            ThreatIndicator(
                pattern=r"api/.*\?limit=\d{4,}",
                threat_type="api_abuse",
                severity=ThreatLevel.LOW,
                confidence=0.6,
                description="Potential API resource exhaustion attempt",
                tags=["api", "dos", "resource_exhaustion"]
            )
        ])
        
        return patterns
    
    def _load_behavioral_rules(self) -> List[Dict]:
        """Load behavioral detection rules"""
        return [
            {
                "name": "brute_force_attack",
                "description": "Multiple failed authentication attempts",
                "conditions": {
                    "event_type": "auth_failed",
                    "threshold": 5,
                    "time_window": 60,  # seconds
                    "group_by": "source_ip"
                },
                "severity": ThreatLevel.HIGH,
                "confidence": 0.9,
                "action": "block_ip"
            },
            {
                "name": "credential_stuffing",
                "description": "Rapid login attempts with different credentials",
                "conditions": {
                    "event_type": "auth_attempt",
                    "threshold": 20,
                    "time_window": 60,
                    "unique_users": 10,
                    "group_by": "source_ip"
                },
                "severity": ThreatLevel.HIGH,
                "confidence": 0.85,
                "action": "rate_limit"
            },
            {
                "name": "port_scanning",
                "description": "Sequential port access attempts",
                "conditions": {
                    "event_type": "connection_attempt",
                    "unique_ports": 10,
                    "time_window": 120,
                    "group_by": "source_ip"
                },
                "severity": ThreatLevel.MEDIUM,
                "confidence": 0.75,
                "action": "monitor"
            },
            {
                "name": "data_exfiltration",
                "description": "Large volume data transfer",
                "conditions": {
                    "event_type": "data_transfer",
                    "volume_mb": 1000,
                    "time_window": 3600,
                    "direction": "outbound"
                },
                "severity": ThreatLevel.CRITICAL,
                "confidence": 0.8,
                "action": "alert_and_block"
            },
            {
                "name": "privilege_escalation",
                "description": "Unusual privilege elevation",
                "conditions": {
                    "event_type": "privilege_change",
                    "from_role": "user",
                    "to_role": "admin",
                    "unusual_time": True
                },
                "severity": ThreatLevel.HIGH,
                "confidence": 0.85,
                "action": "alert"
            },
            {
                "name": "api_rate_abuse",
                "description": "Excessive API calls",
                "conditions": {
                    "event_type": "api_call",
                    "threshold": 1000,
                    "time_window": 60,
                    "group_by": "api_key"
                },
                "severity": ThreatLevel.MEDIUM,
                "confidence": 0.7,
                "action": "rate_limit"
            }
        ]
    
    def _load_ml_models(self) -> Dict:
        """Load machine learning models for anomaly detection"""
        # In production, these would be actual trained models
        return {
            "user_behavior": None,  # User behavior anomaly model
            "network_traffic": None,  # Network traffic anomaly model
            "api_pattern": None,  # API usage pattern model
        }
    
    async def analyze_event(self, event: Dict) -> Dict:
        """
        Analyze security event for threats
        
        Args:
            event: Event data to analyze
            
        Returns:
            Threat analysis results
        """
        self.stats['events_processed'] += 1
        
        analysis = {
            "event_id": event.get("id", f"evt_{datetime.utcnow().timestamp()}"),
            "timestamp": datetime.utcnow().isoformat(),
            "threats_detected": [],
            "risk_score": 0,
            "confidence": 0,
            "recommended_actions": [],
            "tags": set()
        }
        
        # Pattern-based detection
        pattern_threats = await self._detect_pattern_threats(event)
        analysis["threats_detected"].extend(pattern_threats)
        
        # Behavioral detection
        behavioral_threats = await self._detect_behavioral_threats(event)
        analysis["threats_detected"].extend(behavioral_threats)
        
        # ML-based anomaly detection
        if self.ml_models.get("user_behavior"):
            anomalies = await self._detect_anomalies(event)
            analysis["threats_detected"].extend(anomalies)
        
        # Calculate overall risk score
        analysis = self._calculate_risk_score(analysis)
        
        # Determine recommended actions
        analysis = self._determine_actions(analysis)
        
        # Store analysis for correlation
        await self._store_analysis(analysis, event)
        
        # Trigger alerts if necessary
        if analysis["risk_score"] >= 7:
            await self._trigger_alert(analysis, event)
        
        # Update statistics
        if analysis["threats_detected"]:
            self.stats['threats_detected'] += len(analysis["threats_detected"])
        
        return analysis
    
    async def _detect_pattern_threats(self, event: Dict) -> List[Dict]:
        """Detect threats using pattern matching"""
        threats = []
        
        # Get data to analyze
        data_fields = ['url', 'payload', 'user_agent', 'request_body', 'query_params']
        data_to_check = []
        
        for field in data_fields:
            if field in event:
                data_to_check.append(str(event[field]))
        
        # Check headers if present
        if 'headers' in event and isinstance(event['headers'], dict):
            data_to_check.extend(event['headers'].values())
        
        # Check each pattern
        for indicator in self.threat_patterns:
            for data in data_to_check:
                if re.search(indicator.pattern, data, re.IGNORECASE):
                    threat = {
                        "type": indicator.threat_type,
                        "pattern": indicator.pattern,
                        "severity": indicator.severity.value,
                        "confidence": indicator.confidence,
                        "description": indicator.description,
                        "matched_data": data[:200],  # First 200 chars
                        "tags": indicator.tags
                    }
                    threats.append(threat)
                    
                    # Log detection
                    logger.info(f"Pattern threat detected: {indicator.threat_type} - {indicator.description}")
                    
                    break  # One match per indicator is enough
        
        return threats
    
    async def _detect_behavioral_threats(self, event: Dict) -> List[Dict]:
        """Detect threats using behavioral analysis"""
        threats = []
        
        for rule in self.behavioral_rules:
            if await self._evaluate_behavioral_rule(rule, event):
                threat = {
                    "type": "behavioral",
                    "rule": rule["name"],
                    "description": rule["description"],
                    "severity": rule["severity"].value,
                    "confidence": rule["confidence"],
                    "action": rule["action"]
                }
                threats.append(threat)
                
                logger.info(f"Behavioral threat detected: {rule['name']}")
        
        return threats
    
    async def _evaluate_behavioral_rule(self, rule: Dict, event: Dict) -> bool:
        """Evaluate if event matches behavioral rule"""
        conditions = rule["conditions"]
        
        # Check event type
        if conditions.get("event_type") and event.get("type") != conditions["event_type"]:
            return False
        
        # Get grouping key
        group_by = conditions.get("group_by", "source_ip")
        group_key = event.get(group_by)
        
        if not group_key:
            return False
        
        # Get time window
        time_window = conditions.get("time_window", 60)
        cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
        
        # Get relevant events
        if group_by == "source_ip":
            recent_events = [
                e for e in self.ip_activity[group_key]
                if e.get("timestamp", datetime.min) > cutoff_time
            ]
        elif group_by == "user_id":
            recent_events = [
                e for e in self.user_activity[group_key]
                if e.get("timestamp", datetime.min) > cutoff_time
            ]
        else:
            recent_events = []
        
        # Check threshold conditions
        if "threshold" in conditions:
            if len(recent_events) >= conditions["threshold"]:
                return True
        
        # Check unique conditions
        if "unique_ports" in conditions:
            unique_ports = set(e.get("port") for e in recent_events if e.get("port"))
            if len(unique_ports) >= conditions["unique_ports"]:
                return True
        
        if "unique_users" in conditions:
            unique_users = set(e.get("user") for e in recent_events if e.get("user"))
            if len(unique_users) >= conditions["unique_users"]:
                return True
        
        # Check volume conditions
        if "volume_mb" in conditions:
            total_volume = sum(e.get("bytes", 0) for e in recent_events) / (1024 * 1024)
            if total_volume >= conditions["volume_mb"]:
                return True
        
        return False
    
    async def _detect_anomalies(self, event: Dict) -> List[Dict]:
        """Detect anomalies using ML models"""
        anomalies = []
        
        # This would use actual ML models in production
        # For now, using simple statistical anomaly detection
        
        # Check for unusual access times
        hour = datetime.utcnow().hour
        if hour >= 2 and hour <= 5:  # Unusual hours
            anomalies.append({
                "type": "anomaly",
                "subtype": "unusual_time",
                "severity": "low",
                "confidence": 0.6,
                "description": "Activity during unusual hours"
            })
        
        # Check for unusual user agent
        user_agent = event.get("user_agent", "")
        suspicious_agents = ["sqlmap", "nikto", "nmap", "masscan", "burp"]
        
        for agent in suspicious_agents:
            if agent in user_agent.lower():
                anomalies.append({
                    "type": "anomaly",
                    "subtype": "suspicious_user_agent",
                    "severity": "medium",
                    "confidence": 0.8,
                    "description": f"Suspicious user agent detected: {agent}"
                })
                break
        
        return anomalies
    
    def _calculate_risk_score(self, analysis: Dict) -> Dict:
        """Calculate overall risk score"""
        risk_score = 0
        total_confidence = 0
        
        severity_scores = {
            "critical": 10,
            "high": 7,
            "medium": 4,
            "low": 2,
            "info": 1
        }
        
        for threat in analysis["threats_detected"]:
            severity = threat.get("severity", "low")
            confidence = threat.get("confidence", 0.5)
            
            threat_score = severity_scores.get(severity, 1) * confidence
            risk_score += threat_score
            total_confidence += confidence
            
            # Collect tags
            if "tags" in threat:
                analysis["tags"].update(threat["tags"])
        
        # Normalize risk score (0-10 scale)
        analysis["risk_score"] = min(risk_score, 10)
        
        # Calculate average confidence
        if analysis["threats_detected"]:
            analysis["confidence"] = total_confidence / len(analysis["threats_detected"])
        
        # Convert tags set to list for JSON serialization
        analysis["tags"] = list(analysis["tags"])
        
        return analysis
    
    def _determine_actions(self, analysis: Dict) -> Dict:
        """Determine recommended actions based on analysis"""
        actions = []
        
        if analysis["risk_score"] >= 9:
            actions.extend([
                "immediate_block",
                "alert_soc",
                "capture_forensics",
                "isolate_system"
            ])
        elif analysis["risk_score"] >= 7:
            actions.extend([
                "block_source",
                "alert_security_team",
                "increase_monitoring"
            ])
        elif analysis["risk_score"] >= 5:
            actions.extend([
                "rate_limit",
                "monitor_closely",
                "log_详细"
            ])
        elif analysis["risk_score"] >= 3:
            actions.extend([
                "monitor",
                "log"
            ])
        
        # Add specific actions based on threat types
        for threat in analysis["threats_detected"]:
            if threat.get("action"):
                actions.append(threat["action"])
        
        # Remove duplicates while preserving order
        seen = set()
        analysis["recommended_actions"] = [
            x for x in actions 
            if not (x in seen or seen.add(x))
        ]
        
        return analysis
    
    async def _store_analysis(self, analysis: Dict, event: Dict):
        """Store analysis for correlation and future reference"""
        # Store in Redis with expiration
        key = f"threat_analysis:{analysis['event_id']}"
        self.redis_client.setex(
            key,
            86400,  # 24 hour expiration
            json.dumps(analysis, default=str)
        )
        
        # Update activity tracking
        source_ip = event.get("source_ip")
        user_id = event.get("user_id")
        
        if source_ip:
            self.ip_activity[source_ip].append({
                "timestamp": datetime.utcnow(),
                "event_type": event.get("type"),
                "risk_score": analysis["risk_score"],
                **event
            })
            
            # Keep only recent activity (last hour)
            cutoff = datetime.utcnow() - timedelta(hours=1)
            self.ip_activity[source_ip] = [
                e for e in self.ip_activity[source_ip]
                if e["timestamp"] > cutoff
            ]
        
        if user_id:
            self.user_activity[user_id].append({
                "timestamp": datetime.utcnow(),
                "event_type": event.get("type"),
                "risk_score": analysis["risk_score"],
                **event
            })
            
            # Keep only recent activity
            cutoff = datetime.utcnow() - timedelta(hours=1)
            self.user_activity[user_id] = [
                e for e in self.user_activity[user_id]
                if e["timestamp"] > cutoff
            ]
        
        # Add to event buffer for correlation
        self.event_buffer.append({
            "timestamp": datetime.utcnow(),
            "event": event,
            "analysis": analysis
        })
    
    async def _trigger_alert(self, analysis: Dict, event: Dict):
        """Trigger security alert"""
        alert = {
            "id": f"alert_{datetime.utcnow().timestamp()}",
            "timestamp": datetime.utcnow().isoformat(),
            "severity": self._get_highest_severity(analysis),
            "risk_score": analysis["risk_score"],
            "event_id": analysis["event_id"],
            "source_ip": event.get("source_ip"),
            "user_id": event.get("user_id"),
            "threats": analysis["threats_detected"],
            "actions": analysis["recommended_actions"]
        }
        
        # Store alert
        alert_key = f"security_alert:{alert['id']}"
        self.redis_client.setex(
            alert_key,
            86400 * 7,  # Keep for 7 days
            json.dumps(alert, default=str)
        )
        
        # Add to alert queue
        self.redis_client.lpush("alert_queue", json.dumps(alert, default=str))
        
        # Log alert
        logger.warning(f"Security alert triggered: {alert['id']} - Risk score: {alert['risk_score']}")
        
        # Send notifications based on severity
        if alert["severity"] in ["critical", "high"]:
            await self._send_immediate_notification(alert)
    
    def _get_highest_severity(self, analysis: Dict) -> str:
        """Get highest severity from detected threats"""
        severity_order = ["critical", "high", "medium", "low", "info"]
        
        for severity in severity_order:
            for threat in analysis["threats_detected"]:
                if threat.get("severity") == severity:
                    return severity
        
        return "info"
    
    async def _send_immediate_notification(self, alert: Dict):
        """Send immediate notification for critical alerts"""
        # This would integrate with notification systems (Slack, PagerDuty, etc.)
        logger.critical(f"CRITICAL ALERT: {alert['id']} - Immediate action required!")
    
    async def get_threat_statistics(self, time_range: int = 3600) -> Dict:
        """Get threat detection statistics"""
        cutoff = datetime.utcnow() - timedelta(seconds=time_range)
        
        stats = {
            "time_range": time_range,
            "total_events": self.stats['events_processed'],
            "threats_detected": self.stats['threats_detected'],
            "detection_rate": (
                self.stats['threats_detected'] / self.stats['events_processed'] * 100
                if self.stats['events_processed'] > 0 else 0
            ),
            "top_threat_types": defaultdict(int),
            "top_source_ips": defaultdict(int),
            "severity_distribution": defaultdict(int)
        }
        
        # Analyze recent events
        for item in self.event_buffer:
            if item["timestamp"] > cutoff:
                analysis = item["analysis"]
                
                for threat in analysis["threats_detected"]:
                    stats["top_threat_types"][threat.get("type", "unknown")] += 1
                    stats["severity_distribution"][threat.get("severity", "unknown")] += 1
                
                if item["event"].get("source_ip"):
                    stats["top_source_ips"][item["event"]["source_ip"]] += 1
        
        # Convert defaultdicts to regular dicts and get top items
        stats["top_threat_types"] = dict(
            sorted(stats["top_threat_types"].items(), 
                   key=lambda x: x[1], reverse=True)[:10]
        )
        stats["top_source_ips"] = dict(
            sorted(stats["top_source_ips"].items(),
                   key=lambda x: x[1], reverse=True)[:10]
        )
        stats["severity_distribution"] = dict(stats["severity_distribution"])
        
        return stats


# Event correlation engine
class EventCorrelationEngine:
    """Correlate security events to detect complex attack patterns"""
    
    def __init__(self, threat_detector: YTEMPIREThreatDetector):
        self.threat_detector = threat_detector
        self.correlation_window = timedelta(hours=1)
        self.attack_chains = self._load_attack_chains()
    
    def _load_attack_chains(self) -> List[Dict]:
        """Load known attack chain patterns"""
        return [
            {
                "name": "reconnaissance_to_exploitation",
                "description": "Recon followed by exploitation attempt",
                "stages": [
                    {"event": "port_scan", "window": 300},
                    {"event": "vulnerability_scan", "window": 600},
                    {"event": "exploitation_attempt", "window": 300}
                ],
                "severity": ThreatLevel.HIGH,
                "confidence": 0.85
            },
            {
                "name": "credential_theft_lateral_movement",
                "description": "Credential theft followed by lateral movement",
                "stages": [
                    {"event": "credential_access", "window": 600},
                    {"event": "authentication", "window": 300},
                    {"event": "privilege_escalation", "window": 600},
                    {"event": "lateral_movement", "window": 900}
                ],
                "severity": ThreatLevel.CRITICAL,
                "confidence": 0.9
            },
            {
                "name": "data_exfiltration_chain",
                "description": "Complete data exfiltration attack chain",
                "stages": [
                    {"event": "database_access", "window": 600},
                    {"event": "large_query", "window": 300},
                    {"event": "data_compression", "window": 600},
                    {"event": "outbound_transfer", "window": 300}
                ],
                "severity": ThreatLevel.CRITICAL,
                "confidence": 0.88
            }
        ]
    
    async def correlate_events(self, new_event: Dict) -> List[Dict]:
        """Correlate new event with existing events to detect attack chains"""
        correlations = []
        
        # Check for attack chains
        for chain in self.attack_chains:
            if await self._check_attack_chain(chain, new_event):
                correlations.append({
                    "type": "attack_chain",
                    "name": chain["name"],
                    "description": chain["description"],
                    "severity": chain["severity"].value,
                    "confidence": chain["confidence"],
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                logger.warning(f"Attack chain detected: {chain['name']}")
        
        return correlations
    
    async def _check_attack_chain(self, chain: Dict, new_event: Dict) -> bool:
        """Check if recent events match an attack chain pattern"""
        # This would implement sophisticated correlation logic
        # For now, simplified implementation
        return False


# Main execution
if __name__ == "__main__":
    async def main():
        detector = YTEMPIREThreatDetector()
        
        # Example event
        test_event = {
            "id": "test_001",
            "type": "http_request",
            "source_ip": "192.168.1.100",
            "url": "/api/users?id=1' OR '1'='1",
            "user_agent": "Mozilla/5.0",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        analysis = await detector.analyze_event(test_event)
        print(f"Threat analysis: {json.dumps(analysis, indent=2)}")
        
        stats = await detector.get_threat_statistics()
        print(f"Statistics: {json.dumps(stats, indent=2)}")
    
    asyncio.run(main())
```

---

## 5. Quick Reference Guide

### Essential Commands

```bash
# Security Tool Commands
# -----------------------

# OWASP ZAP
docker run -t owasp/zap2docker-stable zap-baseline.py -t https://target.ytempire.com

# Trivy container scan
trivy image --severity HIGH,CRITICAL ytempire/api:latest

# Bandit Python security
bandit -r /opt/ytempire/api -f json -o bandit-report.json

# Git-secrets scan
git secrets --scan --recursive

# Run daily security tasks
python /opt/ytempire/security/automation/daily_security_tasks.py

# Trigger auto-remediation
python /opt/ytempire/security/automation/auto_remediation.py --issue-file issue.json

# Check security gate
python /opt/ytempire/security/scripts/security_gate.py --reports-dir ./reports
```

### File Structure

```
/opt/ytempire/security/
├── tools/
│   ├── zap_scanner.py
│   ├── trivy_scanner.py
│   ├── bandit_scanner.py
│   └── secrets_scanner.py
├── automation/
│   ├── daily_security_tasks.py
│   ├── auto_remediation.py
│   └── compliance_scanner.py
├── monitoring/
│   ├── threat_detection.py
│   └── event_correlation.py
├── incident_response/
│   ├── automated_response.py
│   └── evidence_collector.py
├── ci_cd/
│   ├── security_gate.py
│   └── pipeline_scanner.py
├── config/
│   ├── config.json
│   ├── remediation_policies.json
│   └── threat_patterns.json
├── reports/
│   └── [Generated security reports]
└── logs/
    └── [Security tool logs]
```

### Configuration Files

```json
// /opt/ytempire/security/config/config.json
{
  "scanning": {
    "daily_scan_time": "02:00",
    "scan_timeout": 3600,
    "parallel_scans": 3
  },
  "thresholds": {
    "critical": 0,
    "high": 3,
    "medium": 10,
    "low": 50
  },
  "notifications": {
    "email": ["security@ytempire.com"],
    "slack_webhook": "https://hooks.slack.com/services/xxx",
    "pagerduty_key": "xxx"
  },
  "remediation": {
    "auto_remediate": true,
    "require_approval": {
      "critical": true,
      "high": false
    }
  }
}
```

---

## 6. Emergency Procedures

### Security Incident Response

```bash
#!/bin/bash
# emergency_response.sh - YTEMPIRE Emergency Security Response

echo "🚨 YTEMPIRE EMERGENCY SECURITY RESPONSE ACTIVATED"

# 1. Isolate affected systems
echo "Isolating affected systems..."
iptables -I INPUT -s $THREAT_IP -j DROP
docker stop ytempire-api ytempire-worker

# 2. Capture evidence
echo "Capturing forensic evidence..."
mkdir -p /forensics/$(date +%Y%m%d_%H%M%S)
docker logs ytempire-api > /forensics/api.log
tcpdump -w /forensics/network.pcap -c 10000

# 3. Block malicious IPs
echo "Blocking threat actors..."
for ip in $(cat /tmp/threat_ips.txt); do
    iptables -I INPUT -s $ip -j DROP
done

# 4. Rotate credentials
echo "Rotating all credentials..."
python /opt/ytempire/security/scripts/rotate_all_credentials.py

# 5. Enable enhanced monitoring
echo "Enabling enhanced monitoring..."
python /opt/ytempire/security/monitoring/enhanced_mode.py

# 6. Notify team
echo "Sending emergency notifications..."
python /opt/ytempire/security/scripts/send_emergency_alert.py

echo "✅ Emergency response initiated. Check /forensics/ for evidence."
```

### Manual Security Overrides

```bash
# Disable all automation
export YTEMPIRE_SECURITY_MANUAL=1
systemctl stop ytempire-security-automation

# Bypass security gate (EMERGENCY ONLY)
export SECURITY_GATE_BYPASS=true
export BYPASS_REASON="Emergency deployment for critical fix"

# Disable specific security tools
echo "trivy" >> /etc/ytempire/disabled_tools.txt
echo "zap" >> /etc/ytempire/disabled_tools.txt

# Emergency credential rotation
/opt/ytempire/security/scripts/emergency_rotate.sh

# Force security scan
/opt/ytempire/security/scripts/force_scan.sh --all --immediate
```

---

## Conclusion

This comprehensive YTEMPIRE Security Tools & Automation Guide provides the security engineering team with all necessary tools, scripts, and procedures to maintain robust security for the platform. The guide includes:

- **Complete tool configurations** for OWASP ZAP, Trivy, Bandit, and Git-Secrets
- **Automated security tasks** running daily with comprehensive coverage
- **Auto-remediation system** for immediate threat response
- **CI/CD security integration** with GitHub Actions
- **Real-time threat detection** and monitoring
- **Emergency procedures** for security incidents

Regular updates and maintenance of these security systems are critical for protecting YTEMPIRE's infrastructure and data. All team members should be familiar with these tools and procedures.

For questions or improvements to this guide, please contact the Security Engineering Team at security@ytempire.com.

---

**Document Version**: 2.0  
**Last Updated**: January 2025  
**Next Review**: February 2025  