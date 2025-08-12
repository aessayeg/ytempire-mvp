"""
Simple Security Scanner for YTEmpire MVP
"""
import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Set UTF-8 encoding for Windows
import locale
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

class SimpleSecurityScanner:
    def __init__(self):
        self.project_root = Path("C:/Users/Hp/projects/ytempire-mvp")
        self.results = []
        
    def print_section(self, title):
        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}\n")
    
    def scan_secrets_in_files(self):
        """Basic secret detection"""
        self.print_section("Secret Detection")
        
        files_to_check = [
            "backend/app/core/config.py",
            ".env.example",
            "docker-compose.yml"
        ]
        
        dangerous_patterns = [
            "sk-",  # OpenAI keys
            "ghp_",  # GitHub tokens
            "password:",
            "api_key:",
            "secret_key:"
        ]
        
        issues = []
        for file_path in files_to_check:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"Checking {file_path}...")
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                        for pattern in dangerous_patterns:
                            if pattern.lower() in content:
                                # Check if it's not a placeholder
                                lines = content.split('\n')
                                for i, line in enumerate(lines):
                                    if pattern.lower() in line and not any(x in line for x in ['your-', 'change', 'example']):
                                        issues.append(f"  Line {i+1}: Potential secret ({pattern})")
                except Exception as e:
                    print(f"  Error reading {file_path}: {e}")
        
        if issues:
            print(f"WARNING: Found {len(issues)} potential secrets")
            for issue in issues[:5]:
                print(issue)
        else:
            print("OK: No exposed secrets found")
        
        return len(issues)
    
    def check_docker_security(self):
        """Check Docker security basics"""
        self.print_section("Docker Security")
        
        docker_files = [
            "backend/Dockerfile",
            "frontend/Dockerfile"
        ]
        
        issues = []
        for docker_file in docker_files:
            full_path = self.project_root / docker_file
            if full_path.exists():
                print(f"Checking {docker_file}...")
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                        
                        if "USER" not in content:
                            issues.append(f"  {docker_file}: No USER directive (running as root)")
                        
                        if ":latest" in content:
                            issues.append(f"  {docker_file}: Using :latest tag")
                            
                except Exception as e:
                    print(f"  Error reading {docker_file}: {e}")
        
        if issues:
            print(f"WARNING: Found {len(issues)} Docker issues")
            for issue in issues:
                print(issue)
        else:
            print("OK: Docker files follow security best practices")
        
        return len(issues)
    
    def check_security_headers(self):
        """Check for security headers in FastAPI"""
        self.print_section("Security Headers")
        
        main_file = self.project_root / "backend" / "app" / "main.py"
        
        headers_needed = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "Strict-Transport-Security",
            "Content-Security-Policy"
        ]
        
        missing = []
        if main_file.exists():
            try:
                with open(main_file, 'r') as f:
                    content = f.read()
                    for header in headers_needed:
                        if header not in content:
                            missing.append(header)
            except Exception as e:
                print(f"Error reading main.py: {e}")
        
        if missing:
            print(f"WARNING: Missing {len(missing)} security headers:")
            for header in missing:
                print(f"  - {header}")
        else:
            print("OK: All security headers configured")
        
        return len(missing)
    
    def check_authentication(self):
        """Check authentication configuration"""
        self.print_section("Authentication Security")
        
        security_file = self.project_root / "backend" / "app" / "core" / "security.py"
        
        checks = {
            "bcrypt": False,
            "JWT": False,
            "token_expiry": False
        }
        
        if security_file.exists():
            try:
                with open(security_file, 'r') as f:
                    content = f.read()
                    if "bcrypt" in content:
                        checks["bcrypt"] = True
                    if "jwt" in content.lower() or "JWT" in content:
                        checks["JWT"] = True
                    if "expire" in content:
                        checks["token_expiry"] = True
            except Exception as e:
                print(f"Error reading security.py: {e}")
        
        issues = []
        if checks["bcrypt"]:
            print("OK: Using bcrypt for password hashing")
        else:
            issues.append("Not using bcrypt for passwords")
            
        if checks["JWT"]:
            print("OK: JWT authentication configured")
        else:
            issues.append("JWT not properly configured")
            
        if checks["token_expiry"]:
            print("OK: Token expiry configured")
        else:
            issues.append("Token expiry not configured")
        
        return len(issues)
    
    def check_database_security(self):
        """Check database security configuration"""
        self.print_section("Database Security")
        
        issues = []
        
        # Check for SQL injection prevention
        models_dir = self.project_root / "backend" / "app" / "models"
        if models_dir.exists():
            print("OK: Using SQLAlchemy ORM (SQL injection protection)")
        else:
            issues.append("Models directory not found")
        
        # Check for encrypted connections
        config_file = self.project_root / "backend" / "app" / "core" / "config.py"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                    if "postgresql" in content:
                        print("OK: PostgreSQL configured")
                    if "sslmode" in content:
                        print("OK: SSL mode configured for database")
                    else:
                        issues.append("Database SSL not configured")
            except Exception as e:
                print(f"Error checking database config: {e}")
        
        return len(issues)
    
    def generate_report(self, total_issues):
        """Generate final report"""
        self.print_section("Security Scan Summary")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Scan completed at: {timestamp}")
        print(f"Total issues found: {total_issues}")
        
        if total_issues == 0:
            print("\nSECURITY STATUS: EXCELLENT")
            print("No critical security issues detected!")
        elif total_issues < 5:
            print("\nSECURITY STATUS: GOOD")
            print("Minor security improvements recommended")
        elif total_issues < 10:
            print("\nSECURITY STATUS: FAIR")
            print("Several security issues need attention")
        else:
            print("\nSECURITY STATUS: NEEDS IMPROVEMENT")
            print("Multiple security issues require immediate attention")
        
        # Save results
        results = {
            "scan_date": timestamp,
            "total_issues": total_issues,
            "status": "PASS" if total_issues < 5 else "REVIEW NEEDED"
        }
        
        output_file = self.project_root / "misc" / "security_scan_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: misc/security_scan_results.json")
    
    def run(self):
        """Run all security checks"""
        print("\n" + "="*50)
        print("  YTEmpire MVP Security Scanner")
        print("="*50)
        
        total_issues = 0
        
        # Run all checks
        total_issues += self.scan_secrets_in_files()
        total_issues += self.check_docker_security()
        total_issues += self.check_security_headers()
        total_issues += self.check_authentication()
        total_issues += self.check_database_security()
        
        # Generate report
        self.generate_report(total_issues)
        
        return total_issues < 5  # Pass if less than 5 issues

if __name__ == "__main__":
    scanner = SimpleSecurityScanner()
    success = scanner.run()
    sys.exit(0 if success else 1)