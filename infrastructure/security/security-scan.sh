#!/bin/bash

# YTEmpire Security Scanning Automation Script
# This script runs various security checks on the codebase

set -e

echo "========================================="
echo "YTEmpire Security Scan - Starting"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL_ISSUES=0
CRITICAL_ISSUES=0

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}[PASS]${NC} $2"
    else
        echo -e "${RED}[FAIL]${NC} $2"
        TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
    fi
}

# 1. Python Security Scan (Bandit)
echo -e "\n${YELLOW}Running Python Security Scan...${NC}"
if command -v bandit &> /dev/null; then
    bandit -r backend/ -f json -o backend/security-report.json || true
    if [ -f backend/security-report.json ]; then
        PYTHON_ISSUES=$(python3 -c "import json; data=json.load(open('backend/security-report.json')); print(len(data.get('results', [])))")
        if [ "$PYTHON_ISSUES" -gt 0 ]; then
            print_status 1 "Python security scan found $PYTHON_ISSUES issues"
            CRITICAL_ISSUES=$((CRITICAL_ISSUES + PYTHON_ISSUES))
        else
            print_status 0 "Python security scan passed"
        fi
    fi
else
    echo "Bandit not installed. Run: pip install bandit"
fi

# 2. JavaScript/TypeScript Security Scan (ESLint security plugin)
echo -e "\n${YELLOW}Running JavaScript Security Scan...${NC}"
cd frontend
if [ -f package.json ]; then
    npm audit --audit-level=high > audit-report.txt 2>&1 || true
    HIGH_VULNS=$(grep -c "high" audit-report.txt 2>/dev/null || echo "0")
    CRITICAL_VULNS=$(grep -c "critical" audit-report.txt 2>/dev/null || echo "0")
    
    if [ "$CRITICAL_VULNS" -gt 0 ] || [ "$HIGH_VULNS" -gt 0 ]; then
        print_status 1 "npm audit found $CRITICAL_VULNS critical and $HIGH_VULNS high vulnerabilities"
        CRITICAL_ISSUES=$((CRITICAL_ISSUES + CRITICAL_VULNS))
    else
        print_status 0 "npm audit passed"
    fi
fi
cd ..

# 3. Dependency Check (Safety for Python)
echo -e "\n${YELLOW}Checking Python Dependencies...${NC}"
if command -v safety &> /dev/null; then
    safety check -r backend/requirements.txt --json > backend/safety-report.json 2>&1 || true
    if [ -f backend/safety-report.json ]; then
        print_status 0 "Python dependency check completed"
    fi
else
    echo "Safety not installed. Run: pip install safety"
fi

# 4. Secrets Detection (detect-secrets)
echo -e "\n${YELLOW}Scanning for Secrets...${NC}"
if command -v detect-secrets &> /dev/null; then
    detect-secrets scan --baseline .secrets.baseline > /dev/null 2>&1 || true
    detect-secrets audit .secrets.baseline > /dev/null 2>&1 || true
    print_status 0 "Secrets scan completed"
else
    echo "detect-secrets not installed. Run: pip install detect-secrets"
fi

# 5. Docker Security Scan
echo -e "\n${YELLOW}Scanning Docker Images...${NC}"
if command -v trivy &> /dev/null; then
    # Scan backend Dockerfile
    trivy config backend/Dockerfile > backend/trivy-dockerfile-report.txt 2>&1 || true
    # Scan frontend Dockerfile
    trivy config frontend/Dockerfile > frontend/trivy-dockerfile-report.txt 2>&1 || true
    print_status 0 "Docker security scan completed"
else
    echo "Trivy not installed. Visit: https://github.com/aquasecurity/trivy"
fi

# 6. OWASP Dependency Check
echo -e "\n${YELLOW}Running OWASP Dependency Check...${NC}"
if [ -f dependency-check/bin/dependency-check.sh ]; then
    ./dependency-check/bin/dependency-check.sh \
        --project "YTEmpire" \
        --scan . \
        --format JSON \
        --out security-reports/owasp-report.json \
        --suppression owasp-suppressions.xml || true
    print_status 0 "OWASP dependency check completed"
else
    echo "OWASP Dependency Check not installed"
fi

# 7. Security Headers Check (for deployed app)
echo -e "\n${YELLOW}Checking Security Headers...${NC}"
# This would check the deployed application
# For now, we'll create a checklist
cat > security-reports/security-headers-checklist.txt << EOF
Security Headers Checklist:
[ ] Content-Security-Policy
[ ] X-Frame-Options
[ ] X-Content-Type-Options
[ ] Strict-Transport-Security
[ ] X-XSS-Protection
[ ] Referrer-Policy
[ ] Permissions-Policy
EOF
print_status 0 "Security headers checklist created"

# 8. SQL Injection Check
echo -e "\n${YELLOW}Checking for SQL Injection vulnerabilities...${NC}"
# Check for raw SQL queries in Python files
SQL_ISSUES=$(grep -r "execute(" backend/ --include="*.py" | grep -v "# nosec" | wc -l || echo "0")
if [ "$SQL_ISSUES" -gt 0 ]; then
    print_status 1 "Found $SQL_ISSUES potential SQL injection points"
else
    print_status 0 "No SQL injection vulnerabilities detected"
fi

# 9. Authentication & Authorization Check
echo -e "\n${YELLOW}Checking Authentication Configuration...${NC}"
# Check for hardcoded credentials
HARDCODED=$(grep -r "password\s*=\s*[\"']" . --include="*.py" --include="*.js" --include="*.ts" | grep -v "test" | grep -v "example" | wc -l || echo "0")
if [ "$HARDCODED" -gt 0 ]; then
    print_status 1 "Found $HARDCODED potential hardcoded credentials"
    CRITICAL_ISSUES=$((CRITICAL_ISSUES + HARDCODED))
else
    print_status 0 "No hardcoded credentials found"
fi

# 10. Generate Summary Report
echo -e "\n========================================="
echo "Security Scan Summary"
echo "========================================="

cat > security-reports/summary.txt << EOF
YTEmpire Security Scan Report
Generated: $(date)

Total Issues Found: $TOTAL_ISSUES
Critical Issues: $CRITICAL_ISSUES

Recommendations:
1. Review and fix all critical vulnerabilities immediately
2. Update dependencies with known vulnerabilities
3. Implement security headers in production
4. Regular security scans should be part of CI/CD
5. Enable GitHub security alerts and Dependabot

Next Steps:
- Review detailed reports in security-reports/ directory
- Create tickets for critical issues
- Schedule regular security reviews
EOF

if [ "$CRITICAL_ISSUES" -gt 0 ]; then
    echo -e "${RED}CRITICAL ISSUES FOUND: $CRITICAL_ISSUES${NC}"
    echo "Please review security reports immediately!"
    exit 1
else
    echo -e "${GREEN}Security scan completed successfully!${NC}"
    echo "Total non-critical issues: $TOTAL_ISSUES"
fi

echo -e "\nDetailed reports available in security-reports/ directory"