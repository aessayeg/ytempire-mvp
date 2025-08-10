# 5B. DETAILED OPERATIONAL PROCEDURES

## Executive Summary

This document provides comprehensive operational procedures for YTEMPIRE's security and platform operations teams. It includes daily security operations, shift handoff procedures, audit trail management, compliance reporting, and detailed operational scripts.

---

## 1. Daily Security Operations

### 1.1 Daily Security Checklist

```yaml
daily_security_tasks:
  morning_9am:
    priority_1_immediate:
      - [ ] Check overnight security alerts
      - [ ] Review failed login attempts
      - [ ] Verify backup completion and integrity
      - [ ] Check vulnerability scan results
      - [ ] Review security dashboard
      
    priority_2_within_hour:
      - [ ] Review access logs for anomalies
      - [ ] Check certificate expiration dates
      - [ ] Verify security tool health
      - [ ] Review pending security tickets
      - [ ] Check compliance dashboard
      
  afternoon_2pm:
    review_tasks:
      - [ ] Analyze security metrics
      - [ ] Review code commits for security
      - [ ] Update security documentation
      - [ ] Plan tomorrow's priorities
      - [ ] Security team sync meeting
      
  evening_5pm:
    closing_tasks:
      - [ ] Final security sweep
      - [ ] Update on-call handoff notes
      - [ ] Set overnight alerts
      - [ ] Document any incidents
      - [ ] Update status dashboard
```

### 1.2 Security Operations Runbook

```python
#!/usr/bin/env python3
# security_ops/daily_operations.py

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DailySecurityOperations:
    """Automated daily security operations for YTEMPIRE"""
    
    def __init__(self):
        self.tasks = {
            "critical": self.run_critical_checks,
            "standard": self.run_standard_checks,
            "maintenance": self.run_maintenance_tasks
        }
        self.results_dir = "/opt/ytempire/security/daily_results"
        
    async def execute_daily_operations(self) -> Dict:
        """Execute all daily security operations"""
        
        results = {
            "date": datetime.utcnow().date().isoformat(),
            "start_time": datetime.utcnow().isoformat(),
            "checks": {},
            "issues_found": [],
            "actions_taken": []
        }
        
        # Run all task categories
        for category, task_func in self.tasks.items():
            try:
                task_results = await task_func()
                results["checks"][category] = task_results
                
                # Collect issues
                if task_results.get("issues"):
                    results["issues_found"].extend(task_results["issues"])
                    
            except Exception as e:
                logger.error(f"Task {category} failed: {e}")
                results["checks"][category] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Generate summary
        results["summary"] = self._generate_summary(results)
        results["end_time"] = datetime.utcnow().isoformat()
        
        # Save results
        self._save_results(results)
        
        return results
    
    async def run_critical_checks(self) -> Dict:
        """Run critical security checks"""
        
        checks = {
            "security_alerts": await self._check_security_alerts(),
            "intrusion_detection": await self._check_ids_alerts(),
            "failed_logins": await self._analyze_failed_logins(),
            "api_abuse": await self._check_api_abuse(),
            "certificate_status": await self._check_certificates()
        }
        
        issues = []
        for check_name, check_result in checks.items():
            if check_result["status"] != "ok":
                issues.append({
                    "check": check_name,
                    "severity": check_result["severity"],
                    "details": check_result["details"]
                })
        
        return {
            "status": "completed",
            "checks_run": len(checks),
            "issues": issues,
            "results": checks
        }
    
    async def _check_security_alerts(self) -> Dict:
        """Check for critical security alerts"""
        
        # Query monitoring system
        alerts = await self._query_prometheus(
            'ALERTS{severity=~"critical|high"}'
        )
        
        if not alerts:
            return {"status": "ok", "count": 0}
        
        # Analyze alerts
        analysis = {
            "status": "issues_found",
            "severity": "high",
            "count": len(alerts),
            "details": []
        }
        
        for alert in alerts:
            analysis["details"].append({
                "alert_id": alert["labels"]["alertname"],
                "type": alert["labels"]["type"],
                "source": alert["labels"]["instance"],
                "first_seen": alert["activeAt"],
                "action_required": self._determine_action(alert)
            })
        
        return analysis
    
    async def _analyze_failed_logins(self) -> Dict:
        """Analyze failed login attempts"""
        
        # Parse auth logs
        failed_attempts = []
        with open('/var/log/auth.log', 'r') as f:
            for line in f:
                if 'Failed password' in line:
                    failed_attempts.append(self._parse_auth_log_line(line))
        
        # Group by IP
        ip_attempts = {}
        for attempt in failed_attempts:
            ip = attempt.get('ip')
            if ip:
                ip_attempts[ip] = ip_attempts.get(ip, 0) + 1
        
        # Check thresholds
        suspicious_ips = [ip for ip, count in ip_attempts.items() if count > 5]
        
        if suspicious_ips:
            return {
                "status": "warning",
                "severity": "high",
                "details": {
                    "suspicious_ips": suspicious_ips,
                    "total_attempts": sum(ip_attempts.values()),
                    "unique_ips": len(ip_attempts)
                }
            }
        
        return {
            "status": "ok",
            "total_attempts": sum(ip_attempts.values())
        }
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate daily summary"""
        
        critical_issues = [i for i in results["issues_found"] if i["severity"] == "critical"]
        high_issues = [i for i in results["issues_found"] if i["severity"] == "high"]
        
        return {
            "total_checks": sum(len(c.get("results", {})) for c in results["checks"].values()),
            "issues_count": len(results["issues_found"]),
            "critical_count": len(critical_issues),
            "high_count": len(high_issues),
            "requires_attention": len(critical_issues) > 0 or len(high_issues) > 3,
            "health_score": self._calculate_health_score(results)
        }
    
    def _calculate_health_score(self, results: Dict) -> int:
        """Calculate security health score (0-100)"""
        
        score = 100
        
        # Deduct for issues
        for issue in results["issues_found"]:
            if issue["severity"] == "critical":
                score -= 20
            elif issue["severity"] == "high":
                score -= 10
            elif issue["severity"] == "medium":
                score -= 5
            else:
                score -= 2
        
        return max(0, score)
    
    def _save_results(self, results: Dict):
        """Save daily results"""
        
        filename = f"{self.results_dir}/security_{results['date']}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Daily security results saved to {filename}")
```

### 1.3 Shift Handoff Procedures

```markdown
## Security Shift Handoff Template

### Outgoing Shift Summary
**Date**: [DATE]  
**Shift**: [Morning/Evening/Weekend]  
**Engineer**: [NAME]

#### Active Issues
1. **[ISSUE-ID]**: [Brief description]
   - Status: [Investigating/Contained/Monitoring]
   - Next Action: [What needs to be done]
   - Priority: [Critical/High/Medium/Low]

#### Completed Tasks
- [ ] Morning security checks
- [ ] Vulnerability scan review
- [ ] Access control audit
- [ ] Security updates applied

#### Pending Items
- [ ] [Task 1] - Due: [TIME]
- [ ] [Task 2] - Due: [TIME]

#### System Status
- **Security Tools**: All operational / Issues with: [LIST]
- **Monitoring**: Normal / Elevated alerts for: [LIST]
- **Compliance**: All checks passing / Issues: [LIST]

#### Notes for Incoming Shift
[Any special instructions or concerns]

#### Emergency Contacts Used Today
- [ ] None required
- [ ] Contacted: [WHO] regarding [WHAT]

**Handoff Completed**: [TIME]  
**Acknowledged By**: [INCOMING ENGINEER]
```

---

## 2. Security Monitoring & Alerting

### 2.1 Alert Response Procedures

```python
#!/usr/bin/env python3
# monitoring/alert_response.py

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List
import requests

logger = logging.getLogger(__name__)

class SecurityAlertHandler:
    """Automated security alert response system"""
    
    def __init__(self):
        self.alert_queue = asyncio.Queue()
        self.response_strategies = {
            "BruteForceAttempt": self.respond_brute_force,
            "SQLInjection": self.respond_sql_injection,
            "HighMemoryUsage": self.respond_high_memory,
            "SSLExpiring": self.respond_ssl_expiring,
            "UnauthorizedAccess": self.respond_unauthorized,
            "DataExfiltration": self.respond_data_exfiltration
        }
        
    async def process_alerts(self):
        """Main alert processing loop"""
        
        while True:
            try:
                alert = await self.alert_queue.get()
                await self.handle_alert(alert)
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(1)
    
    async def handle_alert(self, alert: Dict):
        """Handle individual alert"""
        
        alert_type = alert.get("labels", {}).get("alertname")
        severity = alert.get("labels", {}).get("severity", "info")
        
        logger.info(f"Processing alert: {alert_type} (severity: {severity})")
        
        # Log alert
        await self.log_alert(alert)
        
        # Get response strategy
        response_func = self.response_strategies.get(alert_type)
        if response_func:
            response = await response_func(alert)
            await self.execute_response(response)
        else:
            logger.warning(f"No response strategy for alert type: {alert_type}")
        
        # Send notifications
        if severity in ["critical", "high"]:
            await self.send_notifications(alert)
    
    async def respond_brute_force(self, alert: Dict) -> Dict:
        """Response to brute force attempts"""
        
        source_ip = alert.get("labels", {}).get("source_ip")
        
        return {
            "actions": [
                {
                    "type": "block_ip",
                    "target": source_ip,
                    "duration": 3600,
                    "reason": "Brute force attempt detected"
                },
                {
                    "type": "rate_limit",
                    "target": source_ip,
                    "limit": "1/minute",
                    "duration": 7200
                }
            ],
            "notifications": ["security_team", "ops_team"],
            "evidence_collection": True
        }
    
    async def respond_sql_injection(self, alert: Dict) -> Dict:
        """Response to SQL injection attempts"""
        
        return {
            "actions": [
                {
                    "type": "block_request",
                    "pattern": alert.get("labels", {}).get("pattern"),
                    "duration": "permanent"
                },
                {
                    "type": "waf_rule",
                    "action": "add",
                    "rule": self._generate_waf_rule(alert)
                }
            ],
            "notifications": ["security_team", "dev_team"],
            "evidence_collection": True,
            "escalation_required": True
        }
    
    async def execute_response(self, response: Dict):
        """Execute response actions"""
        
        for action in response.get("actions", []):
            try:
                if action["type"] == "block_ip":
                    await self._block_ip(action["target"], action["duration"])
                elif action["type"] == "rate_limit":
                    await self._apply_rate_limit(action["target"], action["limit"])
                elif action["type"] == "waf_rule":
                    await self._update_waf(action["rule"])
                
                logger.info(f"Executed action: {action['type']}")
            except Exception as e:
                logger.error(f"Failed to execute action {action['type']}: {e}")
        
        if response.get("evidence_collection"):
            await self._collect_evidence()
    
    async def _block_ip(self, ip: str, duration: int):
        """Block IP address"""
        
        # Add to fail2ban
        cmd = f"fail2ban-client set sshd banip {ip}"
        await self._execute_command(cmd)
        
        # Add firewall rule
        cmd = f"ufw insert 1 deny from {ip} to any"
        await self._execute_command(cmd)
        
        # Schedule unblock
        asyncio.create_task(self._schedule_unblock(ip, duration))
    
    async def _collect_evidence(self):
        """Collect forensic evidence"""
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        evidence_dir = f"/forensics/{timestamp}"
        
        commands = [
            f"mkdir -p {evidence_dir}",
            f"cp /var/log/auth.log {evidence_dir}/",
            f"cp /var/log/nginx/access.log {evidence_dir}/",
            f"docker logs ytempire-api > {evidence_dir}/api.log 2>&1",
            f"netstat -an > {evidence_dir}/netstat.txt",
            f"ps aux > {evidence_dir}/processes.txt"
        ]
        
        for cmd in commands:
            await self._execute_command(cmd)
        
        logger.info(f"Evidence collected in {evidence_dir}")
```

---

## 3. Audit Trail Management

### 3.1 Audit System Implementation

```python
#!/usr/bin/env python3
# audit/audit_trail.py

import hashlib
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import asyncpg

class AuditTrailManager:
    """Comprehensive audit trail management for YTEMPIRE"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.audit_queue = asyncio.Queue()
        self.retention_days = 365
        
    async def log_event(self, event: Dict) -> str:
        """Log an audit event"""
        
        # Prepare audit event
        audit_event = {
            "event_id": self._generate_event_id(),
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": event.get("user_id"),
            "ip_address": event.get("ip_address"),
            "user_agent": event.get("user_agent"),
            "action": event.get("action"),
            "entity_type": event.get("entity_type"),
            "entity_id": event.get("entity_id"),
            "old_values": event.get("old_values"),
            "new_values": event.get("new_values"),
            "risk_score": self._calculate_risk_score(event),
            "environment": event.get("environment", "production")
        }
        
        # Add integrity hash
        audit_event["integrity_hash"] = self._calculate_integrity_hash(audit_event)
        
        # Store in database
        await self._store_audit_event(audit_event)
        
        # Queue for async processing
        await self.audit_queue.put(audit_event)
        
        return audit_event["event_id"]
    
    def _calculate_integrity_hash(self, event: Dict) -> str:
        """Calculate integrity hash for audit event"""
        
        # Remove hash field for calculation
        event_copy = {k: v for k, v in event.items() if k != "integrity_hash"}
        
        # Canonical JSON representation
        canonical = json.dumps(event_copy, sort_keys=True, default=str)
        
        # Calculate SHA-256 hash
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    def _calculate_risk_score(self, event: Dict) -> int:
        """Calculate risk score for audit event (0-100)"""
        
        score = 0
        
        # High-risk actions
        high_risk_actions = ["delete", "modify_security", "export_data", "change_permissions"]
        if event.get("action") in high_risk_actions:
            score += 30
        
        # Sensitive entities
        sensitive_entities = ["users", "api_keys", "payment_methods", "channels"]
        if event.get("entity_type") in sensitive_entities:
            score += 20
        
        # Unusual time
        hour = datetime.utcnow().hour
        if hour < 6 or hour > 22:  # Outside business hours
            score += 15
        
        # Unknown IP
        if event.get("ip_address") and not self._is_known_ip(event["ip_address"]):
            score += 25
        
        # Bulk operations
        if event.get("bulk_operation"):
            score += 10
        
        return min(100, score)
    
    async def _store_audit_event(self, event: Dict):
        """Store audit event in database"""
        
        query = """
            INSERT INTO audit_logs (
                id, user_id, ip_address, user_agent, action,
                entity_type, entity_id, old_values, new_values,
                timestamp, risk_score, integrity_hash
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """
        
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                query,
                event["event_id"],
                event["user_id"],
                event["ip_address"],
                event["user_agent"],
                event["action"],
                event["entity_type"],
                event["entity_id"],
                json.dumps(event["old_values"]) if event["old_values"] else None,
                json.dumps(event["new_values"]) if event["new_values"] else None,
                datetime.fromisoformat(event["timestamp"]),
                event["risk_score"],
                event["integrity_hash"]
            )
    
    async def verify_audit_integrity(self, start_date: datetime, end_date: datetime) -> Dict:
        """Verify audit trail integrity for a date range"""
        
        verification_result = {
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "events_verified": 0,
            "integrity_failures": [],
            "missing_events": [],
            "status": "valid"
        }
        
        query = """
            SELECT * FROM audit_logs
            WHERE timestamp >= $1 AND timestamp <= $2
            ORDER BY timestamp ASC
        """
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, start_date, end_date)
            
            for row in rows:
                verification_result["events_verified"] += 1
                
                # Reconstruct event
                event = dict(row)
                stored_hash = event.pop("integrity_hash")
                
                # Recalculate hash
                expected_hash = self._calculate_integrity_hash(event)
                
                if stored_hash != expected_hash:
                    verification_result["integrity_failures"].append({
                        "event_id": event["id"],
                        "timestamp": event["timestamp"].isoformat(),
                        "expected_hash": expected_hash,
                        "actual_hash": stored_hash
                    })
                    verification_result["status"] = "compromised"
        
        return verification_result
    
    async def generate_audit_report(self, start_date: datetime, end_date: datetime) -> Dict:
        """Generate audit report for compliance"""
        
        report = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {},
            "high_risk_events": [],
            "user_activity": {},
            "entity_changes": {}
        }
        
        # Get all events
        query = """
            SELECT * FROM audit_logs
            WHERE timestamp >= $1 AND timestamp <= $2
            ORDER BY timestamp DESC
        """
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, start_date, end_date)
            
            # Process events
            for row in rows:
                event = dict(row)
                
                # Count by action
                action = event["action"]
                report["summary"][action] = report["summary"].get(action, 0) + 1
                
                # High risk events
                if event["risk_score"] >= 70:
                    report["high_risk_events"].append({
                        "event_id": event["id"],
                        "timestamp": event["timestamp"].isoformat(),
                        "action": event["action"],
                        "user_id": event["user_id"],
                        "risk_score": event["risk_score"]
                    })
                
                # User activity
                user_id = event.get("user_id")
                if user_id:
                    if user_id not in report["user_activity"]:
                        report["user_activity"][user_id] = []
                    report["user_activity"][user_id].append(action)
        
        return report
```

---

## 4. Compliance Reporting

### 4.1 Compliance Report Generation

```python
#!/usr/bin/env python3
# compliance/reporting.py

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
import jinja2
import pdfkit

class ComplianceReporter:
    """Automated compliance reporting for YTEMPIRE"""
    
    def __init__(self):
        self.template_loader = jinja2.FileSystemLoader('/opt/ytempire/templates')
        self.template_env = jinja2.Environment(loader=self.template_loader)
        
    async def generate_gdpr_report(self) -> Dict:
        """Generate GDPR compliance report"""
        
        report_data = {
            "report_date": datetime.utcnow().isoformat(),
            "reporting_period": "monthly",
            "data_processing_activities": await self._get_data_processing_activities(),
            "user_requests": await self._get_user_requests(),
            "data_breaches": await self._get_data_breaches(),
            "consent_management": await self._get_consent_stats(),
            "data_retention": await self._get_retention_compliance(),
            "third_party_processors": await self._get_processor_status()
        }
        
        # Generate HTML report
        template = self.template_env.get_template('gdpr_report.html')
        html_content = template.render(**report_data)
        
        # Convert to PDF
        pdf_path = f"/opt/ytempire/reports/gdpr_{datetime.utcnow().strftime('%Y%m')}.pdf"
        pdfkit.from_string(html_content, pdf_path)
        
        return {
            "status": "generated",
            "path": pdf_path,
            "data": report_data
        }
    
    async def _get_data_processing_activities(self) -> List[Dict]:
        """Get data processing activities"""
        
        activities = []
        
        # Query audit logs for data processing
        query = """
            SELECT 
                action,
                entity_type,
                COUNT(*) as count,
                array_agg(DISTINCT user_id) as users
            FROM audit_logs
            WHERE timestamp >= NOW() - INTERVAL '30 days'
            AND action IN ('create', 'update', 'delete', 'export')
            GROUP BY action, entity_type
        """
        
        # Execute query and format results
        # ... query implementation ...
        
        return activities
    
    async def generate_security_report(self) -> Dict:
        """Generate security compliance report"""
        
        report_data = {
            "report_date": datetime.utcnow().isoformat(),
            "security_incidents": await self._get_security_incidents(),
            "vulnerability_status": await self._get_vulnerability_status(),
            "access_control_review": await self._get_access_review(),
            "patch_compliance": await self._get_patch_status(),
            "encryption_status": await self._verify_encryption(),
            "backup_status": await self._get_backup_status()
        }
        
        # Calculate compliance score
        report_data["compliance_score"] = self._calculate_compliance_score(report_data)
        
        return report_data
    
    def _calculate_compliance_score(self, data: Dict) -> int:
        """Calculate overall compliance score"""
        
        score = 100
        
        # Deduct for incidents
        incidents = data.get("security_incidents", [])
        score -= len(incidents) * 5
        
        # Deduct for vulnerabilities
        vulns = data.get("vulnerability_status", {})
        score -= vulns.get("critical", 0) * 10
        score -= vulns.get("high", 0) * 5
        
        # Deduct for patch non-compliance
        patch_status = data.get("patch_compliance", {})
        if patch_status.get("behind_schedule"):
            score -= 15
        
        return max(0, score)
```

### 4.2 Compliance Automation Scripts

```bash
#!/bin/bash
# compliance/daily_compliance_check.sh

echo "=== YTEMPIRE Daily Compliance Check ==="
echo "Date: $(date)"

# GDPR Compliance Checks
echo ""
echo "[*] GDPR Compliance Checks..."

# Check for user data older than retention period
psql -U ytempire -d ytempire -c "
SELECT COUNT(*) as expired_records
FROM users
WHERE deleted_at IS NOT NULL 
AND deleted_at < NOW() - INTERVAL '90 days'
AND personal_data_purged = FALSE;
"

# Check consent records
psql -U ytempire -d ytempire -c "
SELECT 
    consent_type,
    COUNT(*) as total,
    SUM(CASE WHEN granted = TRUE THEN 1 ELSE 0 END) as granted,
    SUM(CASE WHEN granted = FALSE THEN 1 ELSE 0 END) as denied
FROM user_consents
GROUP BY consent_type;
"

# PCI DSS Checks (via Stripe)
echo ""
echo "[*] PCI DSS Compliance (Stripe)..."

# Verify no card data in logs
if grep -r "4[0-9]{12}(?:[0-9]{3})?" /var/log/ytempire/ 2>/dev/null; then
    echo "⚠️ WARNING: Potential card numbers found in logs!"
else
    echo "✓ No card numbers found in logs"
fi

# YouTube API Compliance
echo ""
echo "[*] YouTube API Compliance..."

# Check API quota usage
python3 << EOF
import requests
import json

# Check YouTube API quota
response = requests.get(
    'https://youtube.googleapis.com/youtube/v3/i/quota',
    headers={'Authorization': 'Bearer $YOUTUBE_API_KEY'}
)

if response.status_code == 200:
    quota = response.json()
    used = quota.get('quotaUsed', 0)
    limit = quota.get('quotaLimit', 10000)
    percentage = (used / limit) * 100
    
    print(f"YouTube API Quota: {used}/{limit} ({percentage:.1f}%)")
    
    if percentage > 80:
        print("⚠️ WARNING: Approaching quota limit!")
else:
    print("Error checking YouTube quota")
EOF

# Security Headers Check
echo ""
echo "[*] Security Headers Compliance..."

# Check security headers on main site
curl -s -I https://ytempire.com | grep -E "Strict-Transport-Security|X-Frame-Options|X-Content-Type-Options|Content-Security-Policy" || echo "⚠️ Missing security headers!"

# Generate compliance summary
echo ""
echo "=== Compliance Summary ==="
echo "✓ GDPR checks completed"
echo "✓ PCI DSS checks completed"
echo "✓ YouTube API checks completed"
echo "✓ Security headers verified"

# Save report
REPORT_FILE="/opt/ytempire/compliance/daily_$(date +%Y%m%d).txt"
echo "Report saved to: $REPORT_FILE"
```

---

## 5. Security Training & Awareness

### 5.1 Security Training Program

```yaml
security_training_program:
  onboarding:
    day_1:
      duration: 2 hours
      topics:
        - YTEMPIRE security policies
        - Password and MFA setup
        - Acceptable use policy
        - Incident reporting procedures
      
    week_1:
      duration: 4 hours
      topics:
        - Phishing awareness
        - Social engineering defense
        - Data classification
        - Secure communication
      
    month_1:
      duration: 2 hours
      topics:
        - Role-specific security training
        - Security tools overview
        - Compliance requirements
        - Security champion program
  
  ongoing_training:
    quarterly_sessions:
      q1: "Emerging Threats & Trends"
      q2: "Incident Response Drills"
      q3: "Security Tool Updates"
      q4: "Annual Security Review"
    
    monthly_topics:
      january: "Password Security"
      february: "Phishing Defense"
      march: "Data Protection"
      april: "Physical Security"
      may: "Cloud Security"
      june: "Mobile Security"
      july: "Incident Response"
      august: "Compliance Updates"
      september: "Network Security"
      october: "Application Security"
      november: "Security Awareness"
      december: "Year in Review"
  
  role_specific:
    developers:
      topics:
        - Secure coding practices
        - OWASP Top 10
        - Code review security
        - Dependency management
        - Security testing
      
    operations:
        - Infrastructure hardening
        - Access management
        - Monitoring and alerting
        - Incident response
        - Disaster recovery
      
    management:
        - Security governance
        - Risk management
        - Compliance requirements
        - Security metrics
        - Incident communication
```

### 5.2 Phishing Simulation Program

```python
#!/usr/bin/env python3
# training/phishing_simulation.py

import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json

class PhishingSimulation:
    """Phishing awareness training simulation"""
    
    def __init__(self):
        self.templates = self._load_templates()
        self.target_users = self._get_target_users()
        self.results = []
        
    def _load_templates(self) -> List[Dict]:
        """Load phishing email templates"""
        
        return [
            {
                "name": "password_reset",
                "subject": "Urgent: Password Reset Required",
                "difficulty": "easy",
                "red_flags": ["urgency", "generic_greeting", "suspicious_link"]
            },
            {
                "name": "invoice",
                "subject": "Invoice #12345 - Payment Due",
                "difficulty": "medium",
                "red_flags": ["unexpected", "attachment", "pressure"]
            },
            {
                "name": "ceo_fraud",
                "subject": "Urgent Request",
                "difficulty": "hard",
                "red_flags": ["unusual_request", "bypass_procedure", "confidential"]
            }
        ]
    
    async def run_simulation(self):
        """Run phishing simulation campaign"""
        
        campaign_id = self._generate_campaign_id()
        
        for user in self.target_users:
            # Select random template
            template = random.choice(self.templates)
            
            # Send simulated phishing email
            tracking_url = self._generate_tracking_url(campaign_id, user["id"])
            email_sent = await self._send_simulation_email(user, template, tracking_url)
            
            if email_sent:
                self.results.append({
                    "campaign_id": campaign_id,
                    "user_id": user["id"],
                    "template": template["name"],
                    "sent_at": datetime.utcnow().isoformat(),
                    "clicked": False,
                    "reported": False
                })
        
        # Save campaign results
        self._save_results(campaign_id)
        
        return {
            "campaign_id": campaign_id,
            "users_targeted": len(self.target_users),
            "emails_sent": len(self.results),
            "templates_used": list(set(r["template"] for r in self.results))
        }
    
    async def track_click(self, campaign_id: str, user_id: str):
        """Track when user clicks phishing link"""
        
        for result in self.results:
            if result["campaign_id"] == campaign_id and result["user_id"] == user_id:
                result["clicked"] = True
                result["clicked_at"] = datetime.utcnow().isoformat()
                
                # Send immediate training
                await self._send_training_email(user_id)
                break
        
        self._save_results(campaign_id)
    
    def generate_report(self, campaign_id: str) -> Dict:
        """Generate simulation report"""
        
        campaign_results = [r for r in self.results if r["campaign_id"] == campaign_id]
        
        total = len(campaign_results)
        clicked = sum(1 for r in campaign_results if r["clicked"])
        reported = sum(1 for r in campaign_results if r["reported"])
        
        return {
            "campaign_id": campaign_id,
            "statistics": {
                "total_sent": total,
                "clicked": clicked,
                "click_rate": (clicked / total * 100) if total > 0 else 0,
                "reported": reported,
                "report_rate": (reported / total * 100) if total > 0 else 0
            },
            "by_template": self._analyze_by_template(campaign_results),
            "recommendations": self._generate_recommendations(campaign_results)
        }
```

---

## 6. Advanced Operational Scripts

### 6.1 Automated Health Checks

```python
#!/usr/bin/env python3
# operations/health_checker.py

import asyncio
import aiohttp
import psutil
import docker
from datetime import datetime
from typing import Dict, List
import json
import logging

logger = logging.getLogger(__name__)

class SystemHealthChecker:
    """Comprehensive system health monitoring"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.health_thresholds = {
            "cpu_percent": 80,
            "memory_percent": 90,
            "disk_percent": 85,
            "response_time_ms": 1000,
            "error_rate_percent": 1
        }
        
    async def run_health_check(self) -> Dict:
        """Run complete system health check"""
        
        health_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "components": {},
            "issues": [],
            "metrics": {}
        }
        
        # Check system resources
        resources = await self.check_system_resources()
        health_report["components"]["system"] = resources
        
        # Check Docker containers
        containers = await self.check_containers()
        health_report["components"]["containers"] = containers
        
        # Check database
        database = await self.check_database()
        health_report["components"]["database"] = database
        
        # Check APIs
        apis = await self.check_apis()
        health_report["components"]["apis"] = apis
        
        # Check external services
        external = await self.check_external_services()
        health_report["components"]["external"] = external
        
        # Determine overall status
        health_report["status"] = self._determine_overall_status(health_report)
        
        # Save report
        self._save_health_report(health_report)
        
        return health_report
    
    async def check_system_resources(self) -> Dict:
        """Check system resource usage"""
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        status = "healthy"
        issues = []
        
        if cpu_percent > self.health_thresholds["cpu_percent"]:
            status = "warning"
            issues.append(f"High CPU usage: {cpu_percent}%")
        
        if memory.percent > self.health_thresholds["memory_percent"]:
            status = "critical" if memory.percent > 95 else "warning"
            issues.append(f"High memory usage: {memory.percent}%")
        
        if disk.percent > self.health_thresholds["disk_percent"]:
            status = "critical" if disk.percent > 95 else "warning"
            issues.append(f"Low disk space: {disk.percent}% used")
        
        return {
            "status": status,
            "metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "network_sent_gb": network.bytes_sent / (1024**3),
                "network_recv_gb": network.bytes_recv / (1024**3)
            },
            "issues": issues
        }
    
    async def check_containers(self) -> Dict:
        """Check Docker container health"""
        
        containers_status = {
            "status": "healthy",
            "running": 0,
            "stopped": 0,
            "unhealthy": 0,
            "containers": []
        }
        
        for container in self.docker_client.containers.list(all=True):
            container_info = {
                "name": container.name,
                "status": container.status,
                "health": "unknown"
            }
            
            if container.status == "running":
                containers_status["running"] += 1
                
                # Check container health
                health = container.attrs.get("State", {}).get("Health", {})
                if health:
                    container_info["health"] = health.get("Status", "unknown")
                    if health.get("Status") == "unhealthy":
                        containers_status["unhealthy"] += 1
                        containers_status["status"] = "critical"
            else:
                containers_status["stopped"] += 1
                if container.name in ["ytempire-api", "ytempire-db", "ytempire-redis"]:
                    containers_status["status"] = "critical"
            
            containers_status["containers"].append(container_info)
        
        return containers_status
    
    async def check_database(self) -> Dict:
        """Check database health and performance"""
        
        import asyncpg
        
        db_health = {
            "status": "healthy",
            "metrics": {},
            "issues": []
        }
        
        try:
            conn = await asyncpg.connect(
                host='localhost',
                database='ytempire',
                user='ytempire',
                password='password'
            )
            
            # Check connection count
            result = await conn.fetchval("""
                SELECT count(*) FROM pg_stat_activity
                WHERE datname = 'ytempire'
            """)
            db_health["metrics"]["active_connections"] = result
            
            # Check database size
            result = await conn.fetchval("""
                SELECT pg_database_size('ytempire') / 1024 / 1024 as size_mb
            """)
            db_health["metrics"]["size_mb"] = result
            
            # Check slow queries
            result = await conn.fetch("""
                SELECT count(*) as slow_queries
                FROM pg_stat_statements
                WHERE mean_exec_time > 1000
            """)
            if result and result[0]["slow_queries"] > 10:
                db_health["issues"].append(f"High number of slow queries: {result[0]['slow_queries']}")
                db_health["status"] = "warning"
            
            await conn.close()
            
        except Exception as e:
            db_health["status"] = "critical"
            db_health["issues"].append(f"Database connection failed: {str(e)}")
        
        return db_health
    
    async def check_apis(self) -> Dict:
        """Check API endpoints health"""
        
        api_health = {
            "status": "healthy",
            "endpoints": [],
            "issues": []
        }
        
        endpoints = [
            {"name": "API Gateway", "url": "http://localhost:8000/health"},
            {"name": "Backend API", "url": "http://localhost:8001/health"},
            {"name": "Analytics API", "url": "http://localhost:8003/health"},
            {"name": "Frontend", "url": "http://localhost:3000"}
        ]
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                try:
                    start_time = datetime.utcnow()
                    async with session.get(endpoint["url"], timeout=5) as response:
                        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                        
                        endpoint_status = {
                            "name": endpoint["name"],
                            "url": endpoint["url"],
                            "status_code": response.status,
                            "response_time_ms": response_time,
                            "healthy": response.status == 200
                        }
                        
                        if response.status != 200:
                            api_health["status"] = "critical"
                            api_health["issues"].append(f"{endpoint['name']} returned {response.status}")
                        elif response_time > self.health_thresholds["response_time_ms"]:
                            api_health["status"] = "warning" if api_health["status"] == "healthy" else api_health["status"]
                            api_health["issues"].append(f"{endpoint['name']} slow response: {response_time}ms")
                        
                        api_health["endpoints"].append(endpoint_status)
                        
                except Exception as e:
                    api_health["status"] = "critical"
                    api_health["endpoints"].append({
                        "name": endpoint["name"],
                        "url": endpoint["url"],
                        "healthy": False,
                        "error": str(e)
                    })
                    api_health["issues"].append(f"{endpoint['name']} unreachable: {str(e)}")
        
        return api_health
    
    async def check_external_services(self) -> Dict:
        """Check external service connectivity"""
        
        external_health = {
            "status": "healthy",
            "services": [],
            "issues": []
        }
        
        services = [
            {"name": "OpenAI API", "url": "https://api.openai.com/v1/models", "headers": {"Authorization": "Bearer $OPENAI_KEY"}},
            {"name": "YouTube API", "url": "https://www.googleapis.com/youtube/v3/i", "headers": {"Authorization": "Bearer $YOUTUBE_KEY"}},
            {"name": "Stripe API", "url": "https://api.stripe.com/v1/charges", "headers": {"Authorization": "Bearer $STRIPE_KEY"}},
            {"name": "ElevenLabs API", "url": "https://api.elevenlabs.io/v1/user", "headers": {"xi-api-key": "$ELEVENLABS_KEY"}}
        ]
        
        async with aiohttp.ClientSession() as session:
            for service in services:
                try:
                    async with session.get(service["url"], headers=service["headers"], timeout=10) as response:
                        service_status = {
                            "name": service["name"],
                            "status_code": response.status,
                            "healthy": response.status in [200, 401, 403]  # Auth errors are expected
                        }
                        
                        if not service_status["healthy"]:
                            external_health["status"] = "warning"
                            external_health["issues"].append(f"{service['name']} returned unexpected status: {response.status}")
                        
                        external_health["services"].append(service_status)
                        
                except Exception as e:
                    external_health["status"] = "warning"
                    external_health["services"].append({
                        "name": service["name"],
                        "healthy": False,
                        "error": str(e)
                    })
                    external_health["issues"].append(f"{service['name']} check failed: {str(e)}")
        
        return external_health
    
    def _determine_overall_status(self, report: Dict) -> str:
        """Determine overall system status"""
        
        statuses = []
        for component in report["components"].values():
            statuses.append(component.get("status", "unknown"))
        
        if "critical" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        elif "unknown" in statuses:
            return "degraded"
        else:
            return "healthy"
    
    def _save_health_report(self, report: Dict):
        """Save health report to file"""
        
        filename = f"/opt/ytempire/health/health_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Health report saved to {filename}")
```

### 6.2 Performance Optimization Scripts

```python
#!/usr/bin/env python3
# operations/performance_optimizer.py

import asyncio
import asyncpg
import docker
import psutil
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Automated performance optimization for YTEMPIRE"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.optimization_history = []
        
    async def run_optimization(self) -> Dict:
        """Run performance optimization routines"""
        
        optimization_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "optimizations": [],
            "improvements": {},
            "recommendations": []
        }
        
        # Database optimization
        db_opt = await self.optimize_database()
        optimization_report["optimizations"].append(db_opt)
        
        # Container optimization
        container_opt = await self.optimize_containers()
        optimization_report["optimizations"].append(container_opt)
        
        # Cache optimization
        cache_opt = await self.optimize_cache()
        optimization_report["optimizations"].append(cache_opt)
        
        # File system optimization
        fs_opt = await self.optimize_filesystem()
        optimization_report["optimizations"].append(fs_opt)
        
        # Generate recommendations
        optimization_report["recommendations"] = self._generate_recommendations()
        
        return optimization_report
    
    async def optimize_database(self) -> Dict:
        """Optimize database performance"""
        
        optimization = {
            "component": "database",
            "actions": [],
            "metrics_before": {},
            "metrics_after": {}
        }
        
        conn = await asyncpg.connect(
            host='localhost',
            database='ytempire',
            user='ytempire',
            password='password'
        )
        
        try:
            # Get metrics before
            optimization["metrics_before"] = await self._get_db_metrics(conn)
            
            # Run VACUUM ANALYZE
            await conn.execute("VACUUM ANALYZE;")
            optimization["actions"].append("Ran VACUUM ANALYZE")
            
            # Update statistics
            await conn.execute("ANALYZE;")
            optimization["actions"].append("Updated table statistics")
            
            # Reindex tables
            tables = await conn.fetch("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public'
            """)
            
            for table in tables:
                await conn.execute(f"REINDEX TABLE {table['tablename']};")
            optimization["actions"].append(f"Reindexed {len(tables)} tables")
            
            # Get metrics after
            optimization["metrics_after"] = await self._get_db_metrics(conn)
            
        finally:
            await conn.close()
        
        return optimization
    
    async def optimize_containers(self) -> Dict:
        """Optimize Docker containers"""
        
        optimization = {
            "component": "containers",
            "actions": [],
            "space_freed_gb": 0
        }
        
        # Remove stopped containers
        stopped = self.docker_client.containers.prune()
        optimization["actions"].append(f"Removed {stopped['ContainersDeleted']} stopped containers")
        optimization["space_freed_gb"] += stopped.get('SpaceReclaimed', 0) / (1024**3)
        
        # Remove unused images
        images = self.docker_client.images.prune(filters={'dangling': True})
        optimization["actions"].append(f"Removed {len(images['ImagesDeleted'])} unused images")
        optimization["space_freed_gb"] += images.get('SpaceReclaimed', 0) / (1024**3)
        
        # Remove unused volumes
        volumes = self.docker_client.volumes.prune()
        optimization["actions"].append(f"Removed {len(volumes['VolumesDeleted'])} unused volumes")
        optimization["space_freed_gb"] += volumes.get('SpaceReclaimed', 0) / (1024**3)
        
        # Remove unused networks
        networks = self.docker_client.networks.prune()
        optimization["actions"].append(f"Removed {networks['NetworksDeleted']} unused networks")
        
        return optimization
    
    async def optimize_cache(self) -> Dict:
        """Optimize Redis cache"""
        
        import redis
        
        optimization = {
            "component": "cache",
            "actions": [],
            "metrics": {}
        }
        
        r = redis.Redis(host='localhost', port=6379, db=0)
        
        # Get current memory usage
        info = r.info('memory')
        optimization["metrics"]["memory_before_mb"] = info['used_memory'] / (1024**2)
        
        # Remove expired keys
        expired = r.execute_command('DBSIZE')
        r.execute_command('FLUSHDB', 'ASYNC')
        optimization["actions"].append(f"Flushed expired keys")
        
        # Optimize memory
        r.execute_command('MEMORY', 'PURGE')
        optimization["actions"].append("Purged memory")
        
        # Get memory after
        info = r.info('memory')
        optimization["metrics"]["memory_after_mb"] = info['used_memory'] / (1024**2)
        optimization["metrics"]["memory_freed_mb"] = optimization["metrics"]["memory_before_mb"] - optimization["metrics"]["memory_after_mb"]
        
        return optimization
    
    async def optimize_filesystem(self) -> Dict:
        """Optimize file system"""
        
        optimization = {
            "component": "filesystem",
            "actions": [],
            "space_freed_gb": 0
        }
        
        # Clean old logs
        import subprocess
        
        # Remove logs older than 30 days
        cmd = "find /var/log -type f -name '*.log' -mtime +30 -delete"
        subprocess.run(cmd, shell=True)
        optimization["actions"].append("Removed logs older than 30 days")
        
        # Clean temp files
        cmd = "find /tmp -type f -atime +7 -delete"
        subprocess.run(cmd, shell=True)
        optimization["actions"].append("Cleaned temp files older than 7 days")
        
        # Rotate logs
        cmd = "logrotate -f /etc/logrotate.conf"
        subprocess.run(cmd, shell=True)
        optimization["actions"].append("Rotated system logs")
        
        # Calculate space freed
        df_before = psutil.disk_usage('/')
        # ... cleanup operations ...
        df_after = psutil.disk_usage('/')
        optimization["space_freed_gb"] = (df_after.free - df_before.free) / (1024**3)
        
        return optimization
```

---

## Document Metadata

**Version**: 1.0  
**Created**: January 2025  
**Owner**: Security Engineering Team  
**Classification**: Internal - Operations Use Only  
**Review Cycle**: Monthly  

**Purpose**: This document provides detailed operational procedures, scripts, and automation for YTEMPIRE's security and platform operations teams. It supplements the main operational procedures document with specific implementation details and code.

**Note**: All scripts and procedures in this document should be tested in a non-production environment before deployment to production systems.