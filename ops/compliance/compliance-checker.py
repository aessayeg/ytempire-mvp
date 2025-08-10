#!/usr/bin/env python3
"""
YTEmpire Compliance Checker
P2 Enhancement - Automated compliance verification for GDPR, SOC2, YouTube ToS
"""

import os
import sys
import json
import yaml
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import requests
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('compliance_checker')

@dataclass
class ComplianceViolation:
    """Represents a compliance violation"""
    rule_id: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    category: str  # 'gdpr', 'soc2', 'youtube_tos'
    description: str
    file_path: str
    line_number: Optional[int] = None
    recommendation: str = ""
    compliant: bool = False

@dataclass
class ComplianceReport:
    """Complete compliance assessment report"""
    timestamp: datetime
    overall_score: float  # 0-100%
    gdpr_score: float
    soc2_score: float
    youtube_tos_score: float
    violations: List[ComplianceViolation]
    summary: Dict[str, int]
    recommendations: List[str]
    action_items: List[str]

class GDPRChecker:
    """GDPR compliance checker"""
    
    def __init__(self):
        self.personal_data_patterns = [
            r'email[_\s]*address',
            r'phone[_\s]*number',
            r'social[_\s]*security',
            r'credit[_\s]*card',
            r'bank[_\s]*account',
            r'passport[_\s]*number',
            r'driver[_\s]*license',
            r'ip[_\s]*address',
            r'user[_\s]*agent',
            r'biometric[_\s]*data',
            r'geolocation',
            r'cookie[_\s]*data'
        ]
        
        self.consent_patterns = [
            r'consent',
            r'opt[_\s]*in',
            r'opt[_\s]*out',
            r'agree[_\s]*to[_\s]*terms',
            r'privacy[_\s]*policy',
            r'data[_\s]*processing[_\s]*agreement'
        ]
        
        self.encryption_patterns = [
            r'encrypt',
            r'hash',
            r'bcrypt',
            r'scrypt',
            r'pbkdf2',
            r'aes',
            r'rsa'
        ]
        
        self.retention_patterns = [
            r'delete',
            r'expire',
            r'retention[_\s]*period',
            r'purge[_\s]*data',
            r'data[_\s]*lifecycle',
            r'cleanup[_\s]*job'
        ]
    
    def check_personal_data_handling(self, file_path: str, content: str) -> List[ComplianceViolation]:
        """Check for proper personal data handling"""
        violations = []
        
        for line_num, line in enumerate(content.split('\n'), 1):
            line_lower = line.lower()
            
            # Check if personal data is mentioned without encryption context
            for pattern in self.personal_data_patterns:
                if re.search(pattern, line_lower):
                    # Check if encryption/security is mentioned in same context
                    has_security = any(re.search(enc_pattern, line_lower) for enc_pattern in self.encryption_patterns)
                    
                    if not has_security:
                        violations.append(ComplianceViolation(
                            rule_id="GDPR-001",
                            severity="high",
                            category="gdpr",
                            description=f"Personal data '{pattern}' found without encryption context",
                            file_path=file_path,
                            line_number=line_num,
                            recommendation="Ensure personal data is encrypted at rest and in transit"
                        ))
        
        return violations
    
    def check_consent_management(self, file_path: str, content: str) -> List[ComplianceViolation]:
        """Check for consent management implementation"""
        violations = []
        
        has_consent_mechanism = any(re.search(pattern, content.lower()) for pattern in self.consent_patterns)
        
        # Check for user-facing files (HTML, JSX, TSX) that should have consent mechanisms
        if file_path.endswith(('.html', '.jsx', '.tsx', '.vue')) and not has_consent_mechanism:
            violations.append(ComplianceViolation(
                rule_id="GDPR-002",
                severity="medium",
                category="gdpr",
                description="User interface files should implement consent mechanisms",
                file_path=file_path,
                recommendation="Implement cookie consent banners and data processing agreements"
            ))
        
        return violations
    
    def check_data_retention(self, file_path: str, content: str) -> List[ComplianceViolation]:
        """Check for data retention policies"""
        violations = []
        
        has_retention_policy = any(re.search(pattern, content.lower()) for pattern in self.retention_patterns)
        
        # Check database-related files for retention policies
        if ('model' in file_path.lower() or 'schema' in file_path.lower() or 
            'database' in file_path.lower()) and not has_retention_policy:
            violations.append(ComplianceViolation(
                rule_id="GDPR-003",
                severity="medium", 
                category="gdpr",
                description="Database models should implement data retention policies",
                file_path=file_path,
                recommendation="Implement automated data deletion and retention policies"
            ))
        
        return violations
    
    def check_data_portability(self, file_path: str, content: str) -> List[ComplianceViolation]:
        """Check for data portability features"""
        violations = []
        
        export_patterns = [r'export', r'download[_\s]*data', r'data[_\s]*export']
        has_export = any(re.search(pattern, content.lower()) for pattern in export_patterns)
        
        # Check API files for data export capabilities
        if ('api' in file_path.lower() or 'endpoint' in file_path.lower()) and 'user' in content.lower():
            if not has_export:
                violations.append(ComplianceViolation(
                    rule_id="GDPR-004",
                    severity="low",
                    category="gdpr", 
                    description="User APIs should provide data export functionality",
                    file_path=file_path,
                    recommendation="Implement data export endpoints for user data portability"
                ))
        
        return violations

class SOC2Checker:
    """SOC2 compliance checker"""
    
    def __init__(self):
        self.security_patterns = [
            r'authentication',
            r'authorization',
            r'access[_\s]*control',
            r'role[_\s]*based',
            r'permission',
            r'security[_\s]*check'
        ]
        
        self.logging_patterns = [
            r'log',
            r'audit',
            r'track',
            r'monitor',
            r'event[_\s]*recording'
        ]
        
        self.encryption_patterns = [
            r'encrypt',
            r'tls',
            r'ssl',
            r'https',
            r'certificate'
        ]
        
        self.backup_patterns = [
            r'backup',
            r'restore',
            r'recovery',
            r'disaster[_\s]*recovery',
            r'redundancy'
        ]
    
    def check_access_controls(self, file_path: str, content: str) -> List[ComplianceViolation]:
        """Check for proper access controls"""
        violations = []
        
        # Check authentication/authorization files
        if any(keyword in file_path.lower() for keyword in ['auth', 'security', 'middleware']):
            has_security = any(re.search(pattern, content.lower()) for pattern in self.security_patterns)
            
            if not has_security:
                violations.append(ComplianceViolation(
                    rule_id="SOC2-001",
                    severity="high",
                    category="soc2",
                    description="Authentication files should implement proper access controls",
                    file_path=file_path,
                    recommendation="Implement role-based access control with proper authentication"
                ))
        
        return violations
    
    def check_logging_monitoring(self, file_path: str, content: str) -> List[ComplianceViolation]:
        """Check for logging and monitoring"""
        violations = []
        
        has_logging = any(re.search(pattern, content.lower()) for pattern in self.logging_patterns)
        
        # Check critical business logic files for logging
        if any(keyword in file_path.lower() for keyword in ['service', 'controller', 'api', 'endpoint']):
            if not has_logging:
                violations.append(ComplianceViolation(
                    rule_id="SOC2-002",
                    severity="medium",
                    category="soc2", 
                    description="Business logic should implement audit logging",
                    file_path=file_path,
                    recommendation="Add comprehensive logging and monitoring to track all critical operations"
                ))
        
        return violations
    
    def check_data_encryption(self, file_path: str, content: str) -> List[ComplianceViolation]:
        """Check for data encryption"""
        violations = []
        
        has_encryption = any(re.search(pattern, content.lower()) for pattern in self.encryption_patterns)
        
        # Check configuration files for encryption settings
        if file_path.endswith(('.config', '.yaml', '.yml', '.env', '.json')):
            if 'password' in content.lower() or 'secret' in content.lower():
                if not has_encryption:
                    violations.append(ComplianceViolation(
                        rule_id="SOC2-003",
                        severity="critical",
                        category="soc2",
                        description="Configuration files with secrets should use encryption",
                        file_path=file_path,
                        recommendation="Encrypt sensitive configuration data and use secure secret management"
                    ))
        
        return violations
    
    def check_backup_procedures(self, file_path: str, content: str) -> List[ComplianceViolation]:
        """Check for backup and recovery procedures"""
        violations = []
        
        has_backup = any(re.search(pattern, content.lower()) for pattern in self.backup_patterns)
        
        # Check database or infrastructure files for backup procedures
        if any(keyword in file_path.lower() for keyword in ['database', 'db', 'infrastructure', 'deploy']):
            if not has_backup:
                violations.append(ComplianceViolation(
                    rule_id="SOC2-004",
                    severity="medium",
                    category="soc2",
                    description="Infrastructure files should implement backup procedures",
                    file_path=file_path,
                    recommendation="Implement automated backup and disaster recovery procedures"
                ))
        
        return violations

class YouTubeToSChecker:
    """YouTube Terms of Service compliance checker"""
    
    def __init__(self):
        self.content_policy_patterns = [
            r'content[_\s]*policy',
            r'community[_\s]*guidelines',
            r'copyright[_\s]*check',
            r'content[_\s]*moderation',
            r'policy[_\s]*violation'
        ]
        
        self.api_compliance_patterns = [
            r'youtube[_\s]*api',
            r'quota[_\s]*limit',
            r'rate[_\s]*limit',
            r'api[_\s]*key[_\s]*rotation',
            r'usage[_\s]*monitoring'
        ]
        
        self.automation_disclosure_patterns = [
            r'automated[_\s]*content',
            r'ai[_\s]*generated',
            r'bot[_\s]*generated',
            r'artificial[_\s]*intelligence',
            r'automation[_\s]*disclosure'
        ]
        
        self.monetization_patterns = [
            r'monetization',
            r'revenue[_\s]*sharing',
            r'advertisement',
            r'sponsored[_\s]*content',
            r'brand[_\s]*partnership'
        ]
    
    def check_content_policy_compliance(self, file_path: str, content: str) -> List[ComplianceViolation]:
        """Check for content policy compliance measures"""
        violations = []
        
        has_content_policy = any(re.search(pattern, content.lower()) for pattern in self.content_policy_patterns)
        
        # Check content generation files for policy compliance
        if any(keyword in file_path.lower() for keyword in ['content', 'video', 'script', 'generation']):
            if not has_content_policy:
                violations.append(ComplianceViolation(
                    rule_id="YOUTUBE-001",
                    severity="high",
                    category="youtube_tos",
                    description="Content generation should implement policy compliance checks",
                    file_path=file_path,
                    recommendation="Implement content policy validation before video generation"
                ))
        
        return violations
    
    def check_api_usage_compliance(self, file_path: str, content: str) -> List[ComplianceViolation]:
        """Check for YouTube API usage compliance"""
        violations = []
        
        has_api_compliance = any(re.search(pattern, content.lower()) for pattern in self.api_compliance_patterns)
        
        # Check YouTube integration files
        if 'youtube' in file_path.lower() and 'api' in content.lower():
            if not has_api_compliance:
                violations.append(ComplianceViolation(
                    rule_id="YOUTUBE-002",
                    severity="medium",
                    category="youtube_tos",
                    description="YouTube API usage should implement quota and rate limiting",
                    file_path=file_path,
                    recommendation="Implement proper quota management and rate limiting for YouTube API"
                ))
        
        return violations
    
    def check_automation_disclosure(self, file_path: str, content: str) -> List[ComplianceViolation]:
        """Check for automation disclosure"""
        violations = []
        
        has_disclosure = any(re.search(pattern, content.lower()) for pattern in self.automation_disclosure_patterns)
        
        # Check if file deals with automated content generation
        if any(keyword in file_path.lower() for keyword in ['ai', 'ml', 'generation', 'automation']):
            if not has_disclosure:
                violations.append(ComplianceViolation(
                    rule_id="YOUTUBE-003",
                    severity="medium",
                    category="youtube_tos",
                    description="Automated content generation should include disclosure",
                    file_path=file_path,
                    recommendation="Add clear disclosure that content is AI/automated generated"
                ))
        
        return violations
    
    def check_monetization_compliance(self, file_path: str, content: str) -> List[ComplianceViolation]:
        """Check for monetization compliance"""
        violations = []
        
        has_monetization = any(re.search(pattern, content.lower()) for pattern in self.monetization_patterns)
        
        # Check revenue-related files
        if any(keyword in file_path.lower() for keyword in ['revenue', 'monetiz', 'payment', 'earning']):
            if has_monetization and 'disclosure' not in content.lower():
                violations.append(ComplianceViolation(
                    rule_id="YOUTUBE-004",
                    severity="low",
                    category="youtube_tos",
                    description="Monetization features should include proper disclosure",
                    file_path=file_path,
                    recommendation="Add monetization and sponsorship disclosure mechanisms"
                ))
        
        return violations

class ComplianceChecker:
    """Main compliance checker orchestrator"""
    
    def __init__(self, config_path: str = "compliance_config.yaml"):
        self.config = self._load_config(config_path)
        self.gdpr_checker = GDPRChecker()
        self.soc2_checker = SOC2Checker()
        self.youtube_checker = YouTubeToSChecker()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            'scan_paths': ['./src', './app', './frontend/src', './backend/app'],
            'exclude_patterns': ['test', 'spec', '.git', 'node_modules', '__pycache__'],
            'file_extensions': ['.py', '.js', '.ts', '.jsx', '.tsx', '.vue', '.html', '.yaml', '.yml', '.json'],
            'severity_weights': {
                'critical': 4,
                'high': 3,
                'medium': 2,
                'low': 1
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def scan_files(self) -> List[Tuple[str, str]]:
        """Scan files in configured paths"""
        files = []
        
        for scan_path in self.config['scan_paths']:
            if not os.path.exists(scan_path):
                logger.warning(f"Scan path does not exist: {scan_path}")
                continue
            
            for root, dirs, filenames in os.walk(scan_path):
                # Exclude certain directories
                dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.config['exclude_patterns'])]
                
                for filename in filenames:
                    # Check file extension
                    if any(filename.endswith(ext) for ext in self.config['file_extensions']):
                        file_path = os.path.join(root, filename)
                        
                        # Skip excluded patterns
                        if any(pattern in file_path for pattern in self.config['exclude_patterns']):
                            continue
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            files.append((file_path, content))
                        except (UnicodeDecodeError, IOError) as e:
                            logger.warning(f"Could not read file {file_path}: {e}")
        
        return files
    
    def check_gdpr_compliance(self, files: List[Tuple[str, str]]) -> List[ComplianceViolation]:
        """Run GDPR compliance checks"""
        violations = []
        
        for file_path, content in files:
            violations.extend(self.gdpr_checker.check_personal_data_handling(file_path, content))
            violations.extend(self.gdpr_checker.check_consent_management(file_path, content))
            violations.extend(self.gdpr_checker.check_data_retention(file_path, content))
            violations.extend(self.gdpr_checker.check_data_portability(file_path, content))
        
        return violations
    
    def check_soc2_compliance(self, files: List[Tuple[str, str]]) -> List[ComplianceViolation]:
        """Run SOC2 compliance checks"""
        violations = []
        
        for file_path, content in files:
            violations.extend(self.soc2_checker.check_access_controls(file_path, content))
            violations.extend(self.soc2_checker.check_logging_monitoring(file_path, content))
            violations.extend(self.soc2_checker.check_data_encryption(file_path, content))
            violations.extend(self.soc2_checker.check_backup_procedures(file_path, content))
        
        return violations
    
    def check_youtube_compliance(self, files: List[Tuple[str, str]]) -> List[ComplianceViolation]:
        """Run YouTube ToS compliance checks"""
        violations = []
        
        for file_path, content in files:
            violations.extend(self.youtube_checker.check_content_policy_compliance(file_path, content))
            violations.extend(self.youtube_checker.check_api_usage_compliance(file_path, content))
            violations.extend(self.youtube_checker.check_automation_disclosure(file_path, content))
            violations.extend(self.youtube_checker.check_monetization_compliance(file_path, content))
        
        return violations
    
    def calculate_compliance_score(self, violations: List[ComplianceViolation], category: str = None) -> float:
        """Calculate compliance score based on violations"""
        if category:
            category_violations = [v for v in violations if v.category == category]
        else:
            category_violations = violations
        
        if not category_violations:
            return 100.0
        
        # Weight violations by severity
        total_weight = sum(self.config['severity_weights'][v.severity] for v in category_violations)
        max_possible_weight = len(category_violations) * self.config['severity_weights']['critical']
        
        if max_possible_weight == 0:
            return 100.0
        
        score = max(0, 100 - (total_weight / max_possible_weight * 100))
        return round(score, 2)
    
    def generate_recommendations(self, violations: List[ComplianceViolation]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Group violations by severity and category
        critical_violations = [v for v in violations if v.severity == 'critical']
        high_violations = [v for v in violations if v.severity == 'high']
        
        if critical_violations:
            recommendations.append("🔴 CRITICAL: Address all critical security violations immediately")
            for violation in critical_violations[:3]:  # Top 3 critical
                recommendations.append(f"   • {violation.recommendation}")
        
        if high_violations:
            recommendations.append("🟡 HIGH PRIORITY: Resolve high-severity compliance issues")
            for violation in high_violations[:3]:  # Top 3 high
                recommendations.append(f"   • {violation.recommendation}")
        
        # Category-specific recommendations
        gdpr_violations = [v for v in violations if v.category == 'gdpr']
        if len(gdpr_violations) > 5:
            recommendations.append("🛡️ GDPR: Implement comprehensive data privacy framework")
        
        soc2_violations = [v for v in violations if v.category == 'soc2']
        if len(soc2_violations) > 5:
            recommendations.append("🔒 SOC2: Enhance security controls and monitoring")
        
        youtube_violations = [v for v in violations if v.category == 'youtube_tos']
        if len(youtube_violations) > 3:
            recommendations.append("📺 YOUTUBE: Review and implement ToS compliance measures")
        
        return recommendations
    
    def generate_action_items(self, violations: List[ComplianceViolation]) -> List[str]:
        """Generate prioritized action items"""
        action_items = []
        
        # Critical actions
        critical_violations = [v for v in violations if v.severity == 'critical']
        for violation in critical_violations:
            action_items.append(f"🔴 FIX: {violation.description} in {violation.file_path}")
        
        # High priority actions
        high_violations = [v for v in violations if v.severity == 'high']
        for violation in high_violations[:5]:  # Top 5 high priority
            action_items.append(f"🟡 ADDRESS: {violation.description}")
        
        # Implementation actions
        if any(v.category == 'gdpr' for v in violations):
            action_items.append("📝 IMPLEMENT: GDPR consent management system")
        
        if any(v.category == 'soc2' for v in violations):
            action_items.append("🔒 IMPLEMENT: Comprehensive audit logging")
        
        if any(v.category == 'youtube_tos' for v in violations):
            action_items.append("📺 IMPLEMENT: Content policy validation pipeline")
        
        return action_items[:10]  # Limit to top 10
    
    def generate_report(self) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        logger.info("Starting compliance scan...")
        
        # Scan files
        files = self.scan_files()
        logger.info(f"Scanning {len(files)} files for compliance violations")
        
        # Run compliance checks
        gdpr_violations = self.check_gdpr_compliance(files)
        soc2_violations = self.check_soc2_compliance(files)
        youtube_violations = self.check_youtube_compliance(files)
        
        all_violations = gdpr_violations + soc2_violations + youtube_violations
        
        logger.info(f"Found {len(all_violations)} compliance violations")
        
        # Calculate scores
        overall_score = self.calculate_compliance_score(all_violations)
        gdpr_score = self.calculate_compliance_score(gdpr_violations, 'gdpr')
        soc2_score = self.calculate_compliance_score(soc2_violations, 'soc2')
        youtube_score = self.calculate_compliance_score(youtube_violations, 'youtube_tos')
        
        # Generate summary
        summary = {
            'total_violations': len(all_violations),
            'critical': len([v for v in all_violations if v.severity == 'critical']),
            'high': len([v for v in all_violations if v.severity == 'high']),
            'medium': len([v for v in all_violations if v.severity == 'medium']),
            'low': len([v for v in all_violations if v.severity == 'low'])
        }
        
        # Generate recommendations and actions
        recommendations = self.generate_recommendations(all_violations)
        action_items = self.generate_action_items(all_violations)
        
        return ComplianceReport(
            timestamp=datetime.now(),
            overall_score=overall_score,
            gdpr_score=gdpr_score,
            soc2_score=soc2_score,
            youtube_tos_score=youtube_score,
            violations=all_violations,
            summary=summary,
            recommendations=recommendations,
            action_items=action_items
        )
    
    def save_report(self, report: ComplianceReport, output_path: str = "compliance_report.json"):
        """Save report to file"""
        report_dict = asdict(report)
        
        # Convert datetime to string for JSON serialization
        report_dict['timestamp'] = report.timestamp.isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Compliance report saved to {output_path}")
    
    def print_report(self, report: ComplianceReport):
        """Print formatted report to console"""
        print("\n" + "="*80)
        print("🛡️  YTEMPIRE COMPLIANCE ASSESSMENT REPORT")
        print("="*80)
        print(f"📅 Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()
        
        print("📊 COMPLIANCE SCORES")
        print("-" * 50)
        print(f"🎯 Overall Score:    {report.overall_score:6.1f}%")
        print(f"🛡️  GDPR Score:       {report.gdpr_score:6.1f}%")
        print(f"🔒 SOC2 Score:       {report.soc2_score:6.1f}%")
        print(f"📺 YouTube ToS:      {report.youtube_tos_score:6.1f}%")
        print()
        
        print("📈 VIOLATION SUMMARY")
        print("-" * 50)
        print(f"🔴 Critical:  {report.summary['critical']:4d}")
        print(f"🟠 High:      {report.summary['high']:4d}")
        print(f"🟡 Medium:    {report.summary['medium']:4d}")
        print(f"🟢 Low:       {report.summary['low']:4d}")
        print(f"📊 Total:     {report.summary['total_violations']:4d}")
        print()
        
        if report.violations:
            print("⚠️  TOP VIOLATIONS")
            print("-" * 50)
            # Show top 10 most severe violations
            sorted_violations = sorted(report.violations, 
                                     key=lambda x: self.config['severity_weights'][x.severity], 
                                     reverse=True)
            
            for violation in sorted_violations[:10]:
                severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}[violation.severity]
                print(f"{severity_icon} {violation.rule_id}: {violation.description}")
                print(f"   📁 {violation.file_path}")
                if violation.line_number:
                    print(f"   📍 Line {violation.line_number}")
                print()
        
        print("💡 RECOMMENDATIONS")
        print("-" * 50)
        for recommendation in report.recommendations[:8]:
            print(f"  {recommendation}")
        print()
        
        print("✅ ACTION ITEMS")
        print("-" * 50)
        for i, action in enumerate(report.action_items[:8], 1):
            print(f"{i:2d}. {action}")
        print()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="YTEmpire Compliance Checker")
    parser.add_argument('--config', default='compliance_config.yaml', help='Configuration file path')
    parser.add_argument('--output', default='compliance_report.json', help='Output report file')
    parser.add_argument('--format', choices=['json', 'console', 'both'], default='both', help='Output format')
    parser.add_argument('--category', choices=['gdpr', 'soc2', 'youtube_tos'], help='Check specific category only')
    
    args = parser.parse_args()
    
    try:
        checker = ComplianceChecker(args.config)
        report = checker.generate_report()
        
        if args.format in ['json', 'both']:
            checker.save_report(report, args.output)
        
        if args.format in ['console', 'both']:
            checker.print_report(report)
            
        # Exit with error code if critical violations found
        if report.summary['critical'] > 0:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()