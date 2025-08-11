# YTEmpire Security Baseline Configuration for Windows
# PowerShell script for Windows development environment

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "YTEmpire Security Configuration (Windows)" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator"))
{
    Write-Host "This script requires Administrator privileges. Please run as Administrator." -ForegroundColor Red
    Exit 1
}

# Function to check if a feature is installed
function Test-WindowsFeature {
    param($FeatureName)
    $feature = Get-WindowsOptionalFeature -Online -FeatureName $FeatureName -ErrorAction SilentlyContinue
    return $feature.State -eq 'Enabled'
}

Write-Host "[1/8] Configuring Windows Firewall..." -ForegroundColor Yellow

# Enable Windows Firewall
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True

# Remove existing YTEmpire rules
Get-NetFirewallRule -DisplayName "YTEmpire*" -ErrorAction SilentlyContinue | Remove-NetFirewallRule

# Add firewall rules for YTEmpire services
$rules = @(
    @{DisplayName="YTEmpire Backend API"; LocalPort=8000; Protocol="TCP"; Direction="Inbound"},
    @{DisplayName="YTEmpire Frontend"; LocalPort=3000; Protocol="TCP"; Direction="Inbound"},
    @{DisplayName="YTEmpire PostgreSQL"; LocalPort=5432; Protocol="TCP"; Direction="Inbound"; RemoteAddress="LocalSubnet"},
    @{DisplayName="YTEmpire Redis"; LocalPort=6379; Protocol="TCP"; Direction="Inbound"; RemoteAddress="LocalSubnet"},
    @{DisplayName="YTEmpire Flower"; LocalPort=5555; Protocol="TCP"; Direction="Inbound"},
    @{DisplayName="YTEmpire Grafana"; LocalPort=3001; Protocol="TCP"; Direction="Inbound"},
    @{DisplayName="YTEmpire Prometheus"; LocalPort=9090; Protocol="TCP"; Direction="Inbound"},
    @{DisplayName="YTEmpire N8N"; LocalPort=5678; Protocol="TCP"; Direction="Inbound"}
)

foreach ($rule in $rules) {
    New-NetFirewallRule @rule -Action Allow -Enabled True
    Write-Host "  Added firewall rule: $($rule.DisplayName)" -ForegroundColor Green
}

Write-Host "[2/8] Configuring Windows Defender..." -ForegroundColor Yellow

# Enable Windows Defender features
Set-MpPreference -DisableRealtimeMonitoring $false
Set-MpPreference -DisableBehaviorMonitoring $false
Set-MpPreference -DisableIOAVProtection $false
Set-MpPreference -DisablePrivacyMode $false
Set-MpPreference -SignatureDisableUpdateOnStartupWithoutEngine $false
Set-MpPreference -DisableArchiveScanning $false
Set-MpPreference -DisableIntrusionPreventionSystem $false

# Add exclusions for development folders to prevent performance issues
Add-MpPreference -ExclusionPath "C:\Users\$env:USERNAME\projects\ytempire-mvp\node_modules"
Add-MpPreference -ExclusionPath "C:\Users\$env:USERNAME\projects\ytempire-mvp\venv"
Add-MpPreference -ExclusionPath "C:\Users\$env:USERNAME\projects\ytempire-mvp\.git"
Add-MpPreference -ExclusionProcess "node.exe"
Add-MpPreference -ExclusionProcess "python.exe"

Write-Host "  Windows Defender configured" -ForegroundColor Green

Write-Host "[3/8] Setting up audit policies..." -ForegroundColor Yellow

# Enable audit policies
auditpol /set /category:"Logon/Logoff" /success:enable /failure:enable
auditpol /set /category:"Account Logon" /success:enable /failure:enable
auditpol /set /category:"Account Management" /success:enable /failure:enable
auditpol /set /category:"Object Access" /success:enable /failure:enable

Write-Host "  Audit policies configured" -ForegroundColor Green

Write-Host "[4/8] Creating security directories..." -ForegroundColor Yellow

# Create directories for logs and backups
$dirs = @(
    "C:\YTEmpire\logs",
    "C:\YTEmpire\backups",
    "C:\YTEmpire\security",
    "C:\YTEmpire\certificates"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created directory: $dir" -ForegroundColor Green
    }
}

Write-Host "[5/8] Setting up environment variables security..." -ForegroundColor Yellow

# Create a template for secure environment variables
$envTemplate = @"
# YTEmpire Secure Environment Variables Template
# Copy this to .env and fill in your actual values
# NEVER commit the actual .env file to version control

# Database
DATABASE_URL=postgresql+asyncpg://ytempire:CHANGE_THIS_PASSWORD@localhost:5432/ytempire_db

# Redis
REDIS_URL=redis://localhost:6379/0

# Secret Keys (Generate with: openssl rand -hex 32)
SECRET_KEY=GENERATE_A_RANDOM_SECRET_KEY_HERE
JWT_SECRET_KEY=GENERATE_ANOTHER_RANDOM_SECRET_KEY_HERE

# API Keys (Keep these secure!)
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...
PEXELS_API_KEY=...
PIXABAY_API_KEY=...

# YouTube OAuth (15 accounts)
YOUTUBE_CLIENT_ID_0=...
YOUTUBE_CLIENT_SECRET_0=...
# ... repeat for accounts 1-14

# Stripe
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Security Settings
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
SESSION_COOKIE_SECURE=true
CSRF_COOKIE_SECURE=true

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# Monitoring
SENTRY_DSN=
ENABLE_METRICS=true
"@

$envTemplate | Out-File -FilePath "C:\Users\$env:USERNAME\projects\ytempire-mvp\.env.template" -Encoding UTF8
Write-Host "  Created .env.template file" -ForegroundColor Green

Write-Host "[6/8] Creating security monitoring scripts..." -ForegroundColor Yellow

# Create PowerShell security audit script
$auditScript = @'
# YTEmpire Security Audit Script for Windows

Write-Host "YTEmpire Security Audit Report" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host "Date: $(Get-Date)" -ForegroundColor White
Write-Host ""

Write-Host "1. System Information:" -ForegroundColor Yellow
Get-ComputerInfo | Select-Object CsName, WindowsVersion, OsArchitecture, CsTotalPhysicalMemory

Write-Host "`n2. Failed Login Attempts (last 24h):" -ForegroundColor Yellow
Get-EventLog -LogName Security -After (Get-Date).AddDays(-1) | 
    Where-Object {$_.EventID -eq 4625} | 
    Select-Object TimeGenerated, Message -First 10

Write-Host "`n3. Firewall Status:" -ForegroundColor Yellow
Get-NetFirewallProfile | Select-Object Name, Enabled

Write-Host "`n4. YTEmpire Firewall Rules:" -ForegroundColor Yellow
Get-NetFirewallRule -DisplayName "YTEmpire*" | 
    Select-Object DisplayName, Enabled, Direction, Action

Write-Host "`n5. Open Ports:" -ForegroundColor Yellow
Get-NetTCPConnection -State Listen | 
    Select-Object LocalAddress, LocalPort, State | 
    Sort-Object LocalPort -Unique

Write-Host "`n6. Running Services:" -ForegroundColor Yellow
Get-Service | Where-Object {$_.Status -eq 'Running'} | 
    Select-Object Name, DisplayName -First 20

Write-Host "`n7. Recent Windows Updates:" -ForegroundColor Yellow
Get-HotFix | Sort-Object InstalledOn -Descending | 
    Select-Object HotFixID, InstalledOn, Description -First 10

Write-Host "`n8. Disk Usage:" -ForegroundColor Yellow
Get-PSDrive -PSProvider FileSystem | 
    Select-Object Name, @{n='Used(GB)';e={[math]::Round($_.Used/1GB,2)}}, 
                  @{n='Free(GB)';e={[math]::Round($_.Free/1GB,2)}}

Write-Host "`n9. Docker Containers (if Docker is running):" -ForegroundColor Yellow
try {
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
} catch {
    Write-Host "Docker is not running or not installed" -ForegroundColor Gray
}

Write-Host "`nAudit complete." -ForegroundColor Green
'@

$auditScript | Out-File -FilePath "C:\YTEmpire\security\security-audit.ps1" -Encoding UTF8
Write-Host "  Created security audit script" -ForegroundColor Green

Write-Host "[7/8] Setting up scheduled security tasks..." -ForegroundColor Yellow

# Create scheduled task for daily security audit
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File C:\YTEmpire\security\security-audit.ps1"

$trigger = New-ScheduledTaskTrigger -Daily -At 3am

$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" `
    -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName "YTEmpire Security Audit" `
    -Action $action -Trigger $trigger -Principal $principal `
    -Description "Daily security audit for YTEmpire" -Force | Out-Null

Write-Host "  Created scheduled security audit task" -ForegroundColor Green

Write-Host "[8/8] Creating development security checklist..." -ForegroundColor Yellow

$checklist = @"
# YTEmpire Development Security Checklist

## Before Starting Development
- [ ] Ensure .env file is not tracked in Git
- [ ] Verify all API keys are stored in environment variables
- [ ] Check that debug mode is disabled in production configs
- [ ] Confirm HTTPS is enforced for production URLs

## Code Security
- [ ] No hardcoded credentials or API keys
- [ ] Input validation on all user inputs
- [ ] SQL injection prevention (use parameterized queries)
- [ ] XSS prevention (sanitize outputs)
- [ ] CSRF tokens implemented
- [ ] Rate limiting configured
- [ ] Authentication required for sensitive endpoints
- [ ] Authorization checks for user actions

## API Security
- [ ] JWT tokens with expiration
- [ ] API key rotation schedule
- [ ] Request signing for webhooks
- [ ] CORS properly configured
- [ ] API versioning implemented

## Data Security
- [ ] Encryption at rest for sensitive data
- [ ] Encryption in transit (TLS/SSL)
- [ ] PII data handling compliance
- [ ] Regular backups configured
- [ ] Backup encryption enabled

## Infrastructure Security
- [ ] Firewall rules reviewed
- [ ] Unnecessary ports closed
- [ ] Security updates applied
- [ ] Monitoring and alerting configured
- [ ] Log aggregation setup
- [ ] Incident response plan documented

## Third-Party Services
- [ ] API key permissions minimized
- [ ] Service quotas configured
- [ ] Webhook validation implemented
- [ ] OAuth scopes minimized

## Before Deployment
- [ ] Security audit completed
- [ ] Penetration testing performed
- [ ] Dependencies vulnerability scan
- [ ] Docker images scanned
- [ ] Secrets management configured
- [ ] SSL certificates valid

## Regular Maintenance
- [ ] Weekly security updates check
- [ ] Monthly API key rotation
- [ ] Quarterly security audit
- [ ] Annual penetration testing
"@

$checklist | Out-File -FilePath "C:\YTEmpire\security\security-checklist.md" -Encoding UTF8
Write-Host "  Created security checklist" -ForegroundColor Green

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Security Configuration Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Summary of changes:" -ForegroundColor Yellow
Write-Host "1. Windows Firewall configured with YTEmpire rules"
Write-Host "2. Windows Defender configured with exclusions"
Write-Host "3. Audit policies enabled"
Write-Host "4. Security directories created"
Write-Host "5. Environment variable template created"
Write-Host "6. Security audit script created"
Write-Host "7. Scheduled security tasks configured"
Write-Host "8. Development security checklist created"
Write-Host ""
Write-Host "Important next steps:" -ForegroundColor Yellow
Write-Host "1. Review and customize .env.template"
Write-Host "2. Run security audit: C:\YTEmpire\security\security-audit.ps1"
Write-Host "3. Review security checklist: C:\YTEmpire\security\security-checklist.md"
Write-Host "4. Configure SSL certificates for production"
Write-Host "5. Set up backup strategy"
Write-Host ""
Write-Host "Security files location: C:\YTEmpire\security\" -ForegroundColor Cyan
Write-Host ""