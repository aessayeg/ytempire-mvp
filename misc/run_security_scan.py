"""
Security Scanning Script for YTEmpire MVP
Performs comprehensive security assessment
"""
import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

class SecurityScanner:
    def __init__(self):
        self.project_root = Path("C:/Users/Hp/projects/ytempire-mvp")
        self.results = {
            "scan_timestamp": datetime.now().isoformat(),
            "vulnerabilities": [],
            "dependencies": [],
            "secrets": [],
            "docker": [],
            "summary": {}
        }
        
    def print_header(self, title: str):
        """Print formatted header"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
        
    def run_command(self, command: List[str], cwd: str = None) -> tuple:
        """Run a command and return output"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=cwd or self.project_root,
                timeout=60
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def scan_python_dependencies(self):
        """Scan Python dependencies for vulnerabilities"""
        self.print_header("Python Dependency Scan")
        
        backend_path = self.project_root / "backend"
        requirements_file = backend_path / "requirements.txt"
        
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return
            
        # Run pip-audit for vulnerability scanning
        print("Running pip-audit...")
        success, stdout, stderr = self.run_command(
            ["pip-audit", "--requirement", str(requirements_file), "--format", "json"],
            cwd=str(backend_path)
        )
        
        if success and stdout:
            try:
                vulnerabilities = json.loads(stdout)
                if vulnerabilities:
                    print(f"⚠️ Found {len(vulnerabilities)} vulnerabilities")
                    for vuln in vulnerabilities[:5]:  # Show first 5
                        self.results["dependencies"].append({
                            "type": "python",
                            "package": vuln.get("name", "unknown"),
                            "severity": vuln.get("vulns", [{}])[0].get("severity", "unknown"),
                            "description": vuln.get("vulns", [{}])[0].get("description", "")
                        })
                        print(f"  - {vuln.get('name')}: {vuln.get('vulns', [{}])[0].get('id', 'unknown')}")
                else:
                    print("✅ No vulnerabilities found in Python dependencies")
            except json.JSONDecodeError:
                print("⚠️ Could not parse pip-audit output")
        else:
            # Fallback to safety check
            print("Running safety check...")
            success, stdout, stderr = self.run_command(
                ["safety", "check", "--json"],
                cwd=str(backend_path)
            )
            if success:
                print("✅ Safety check completed")
            else:
                print("⚠️ Safety check failed:", stderr[:100])
    
    def scan_npm_dependencies(self):
        """Scan NPM dependencies for vulnerabilities"""
        self.print_header("NPM Dependency Scan")
        
        frontend_path = self.project_root / "frontend"
        package_json = frontend_path / "package.json"
        
        if not package_json.exists():
            print("❌ package.json not found")
            return
            
        print("Running npm audit...")
        success, stdout, stderr = self.run_command(
            ["npm", "audit", "--json"],
            cwd=str(frontend_path)
        )
        
        if stdout:
            try:
                audit_data = json.loads(stdout)
                vulnerabilities = audit_data.get("vulnerabilities", {})
                
                if vulnerabilities:
                    critical = sum(1 for v in vulnerabilities.values() if v.get("severity") == "critical")
                    high = sum(1 for v in vulnerabilities.values() if v.get("severity") == "high")
                    medium = sum(1 for v in vulnerabilities.values() if v.get("severity") == "medium")
                    low = sum(1 for v in vulnerabilities.values() if v.get("severity") == "low")
                    
                    print(f"⚠️ Found vulnerabilities:")
                    print(f"  Critical: {critical}")
                    print(f"  High: {high}")
                    print(f"  Medium: {medium}")
                    print(f"  Low: {low}")
                    
                    for name, vuln in list(vulnerabilities.items())[:5]:
                        self.results["dependencies"].append({
                            "type": "npm",
                            "package": name,
                            "severity": vuln.get("severity"),
                            "via": vuln.get("via", [{}])[0] if isinstance(vuln.get("via"), list) else vuln.get("via")
                        })
                else:
                    print("✅ No vulnerabilities found in NPM dependencies")
            except json.JSONDecodeError:
                print("⚠️ Could not parse npm audit output")
    
    def scan_for_secrets(self):
        """Scan for exposed secrets and API keys"""
        self.print_header("Secret Detection Scan")
        
        secret_patterns = [
            (r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']([^"\']+)["\']', "API Key"),
            (r'(?i)(secret[_-]?key|secret)\s*[:=]\s*["\']([^"\']+)["\']', "Secret Key"),
            (r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']([^"\']+)["\']', "Password"),
            (r'sk-[a-zA-Z0-9]{48}', "OpenAI API Key"),
            (r'ghp_[a-zA-Z0-9]{36}', "GitHub Personal Token"),
            (r'(?i)aws[_-]?access[_-]?key[_-]?id\s*[:=]\s*["\']([^"\']+)["\']', "AWS Access Key"),
            (r'(?i)stripe[_-]?api[_-]?key\s*[:=]\s*["\']([^"\']+)["\']', "Stripe API Key"),
        ]
        
        files_to_scan = [
            "backend/app/core/config.py",
            "docker-compose.yml",
            "docker-compose.production.yml",
            ".env.example"
        ]
        
        secrets_found = []
        for file_path in files_to_scan:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"Scanning {file_path}...")
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    for pattern, secret_type in secret_patterns:
                        import re
                        matches = re.finditer(pattern, content)
                        for match in matches:
                            # Check if it's a placeholder
                            value = match.group(0) if len(match.groups()) == 0 else match.group(1) if len(match.groups()) == 1 else match.group(2)
                            if not any(placeholder in value.lower() for placeholder in ['your-', 'change', 'example', 'placeholder', 'xxx']):
                                secrets_found.append({
                                    "file": file_path,
                                    "type": secret_type,
                                    "line": content[:match.start()].count('\n') + 1
                                })
        
        if secrets_found:
            print(f"⚠️ Found {len(secrets_found)} potential secrets:")
            for secret in secrets_found[:5]:
                print(f"  - {secret['type']} in {secret['file']} at line {secret['line']}")
                self.results["secrets"].append(secret)
        else:
            print("✅ No exposed secrets found")
    
    def scan_docker_security(self):
        """Scan Docker configurations for security issues"""
        self.print_header("Docker Security Scan")
        
        docker_files = [
            "backend/Dockerfile",
            "frontend/Dockerfile",
            "ml-pipeline/Dockerfile"
        ]
        
        security_issues = []
        for docker_file in docker_files:
            full_path = self.project_root / docker_file
            if full_path.exists():
                print(f"Scanning {docker_file}...")
                with open(full_path, 'r') as f:
                    content = f.read()
                    
                    # Check for security best practices
                    if "USER" not in content:
                        security_issues.append({
                            "file": docker_file,
                            "issue": "No USER instruction - running as root",
                            "severity": "high"
                        })
                    
                    if "latest" in content:
                        security_issues.append({
                            "file": docker_file,
                            "issue": "Using 'latest' tag - not reproducible",
                            "severity": "medium"
                        })
                    
                    if "apt-get upgrade" in content:
                        security_issues.append({
                            "file": docker_file,
                            "issue": "Using apt-get upgrade - can break reproducibility",
                            "severity": "low"
                        })
        
        if security_issues:
            print(f"⚠️ Found {len(security_issues)} Docker security issues:")
            for issue in security_issues:
                print(f"  - {issue['file']}: {issue['issue']} ({issue['severity']})")
                self.results["docker"].append(issue)
        else:
            print("✅ Docker configurations look secure")
    
    def check_security_headers(self):
        """Check if security headers are configured"""
        self.print_header("Security Headers Check")
        
        main_py = self.project_root / "backend" / "app" / "main.py"
        if main_py.exists():
            with open(main_py, 'r') as f:
                content = f.read()
                
                headers_to_check = [
                    ("X-Content-Type-Options", "nosniff"),
                    ("X-Frame-Options", "DENY"),
                    ("X-XSS-Protection", "1; mode=block"),
                    ("Strict-Transport-Security", "max-age"),
                    ("Content-Security-Policy", "default-src")
                ]
                
                missing_headers = []
                for header, value in headers_to_check:
                    if header not in content and value not in content:
                        missing_headers.append(header)
                
                if missing_headers:
                    print(f"⚠️ Missing security headers:")
                    for header in missing_headers:
                        print(f"  - {header}")
                        self.results["vulnerabilities"].append({
                            "type": "missing_header",
                            "header": header,
                            "severity": "medium"
                        })
                else:
                    print("✅ All security headers configured")
    
    def generate_summary(self):
        """Generate scan summary"""
        self.print_header("Security Scan Summary")
        
        total_issues = (
            len(self.results["vulnerabilities"]) +
            len(self.results["dependencies"]) +
            len(self.results["secrets"]) +
            len(self.results["docker"])
        )
        
        critical_count = sum(1 for item in self.results["dependencies"] if item.get("severity") == "critical")
        high_count = sum(1 for item in self.results["dependencies"] if item.get("severity") == "high")
        
        self.results["summary"] = {
            "total_issues": total_issues,
            "critical_issues": critical_count,
            "high_issues": high_count,
            "scan_completed": datetime.now().isoformat()
        }
        
        print(f"Total Issues Found: {total_issues}")
        print(f"  Critical: {critical_count}")
        print(f"  High: {high_count}")
        print(f"  Secrets: {len(self.results['secrets'])}")
        print(f"  Docker: {len(self.results['docker'])}")
        
        # Save results to file
        output_file = self.project_root / "misc" / "security_scan_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Full report saved to: {output_file}")
        
        return total_issues == 0
    
    def run_full_scan(self):
        """Run complete security scan"""
        print("\n" + "="*60)
        print("  YTEmpire MVP Security Scanner")
        print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("="*60)
        
        # Run all scans
        self.scan_python_dependencies()
        self.scan_npm_dependencies()
        self.scan_for_secrets()
        self.scan_docker_security()
        self.check_security_headers()
        
        # Generate summary
        all_clear = self.generate_summary()
        
        if all_clear:
            print("\n✅ SECURITY SCAN PASSED - No critical issues found!")
        else:
            print("\n⚠️ SECURITY SCAN COMPLETED - Review and address issues")
        
        return all_clear

if __name__ == "__main__":
    scanner = SecurityScanner()
    scanner.run_full_scan()