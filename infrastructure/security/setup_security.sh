#!/bin/bash

# YTEmpire Security Baseline Configuration Script
# This script sets up essential security measures for the MVP

set -e

echo "========================================="
echo "YTEmpire Security Baseline Configuration"
echo "========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Update system
echo "[1/10] Updating system packages..."
apt-get update
apt-get upgrade -y

# Install security tools
echo "[2/10] Installing security tools..."
apt-get install -y ufw fail2ban unattended-upgrades aide rkhunter clamav clamav-daemon

# Configure UFW Firewall
echo "[3/10] Configuring UFW firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing

# Allow SSH (change port if needed)
ufw allow 22/tcp comment 'SSH'

# Allow HTTP and HTTPS
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'

# Allow FastAPI backend
ufw allow 8000/tcp comment 'FastAPI Backend'

# Allow Frontend dev server
ufw allow 3000/tcp comment 'React Frontend'

# Allow PostgreSQL (only from localhost)
ufw allow from 127.0.0.1 to any port 5432 comment 'PostgreSQL local'

# Allow Redis (only from localhost)
ufw allow from 127.0.0.1 to any port 6379 comment 'Redis local'

# Allow Flower monitoring
ufw allow 5555/tcp comment 'Flower Celery Monitor'

# Allow Grafana
ufw allow 3001/tcp comment 'Grafana Dashboard'

# Allow Prometheus
ufw allow 9090/tcp comment 'Prometheus Metrics'

# Allow N8N
ufw allow 5678/tcp comment 'N8N Workflows'

# Enable UFW
ufw --force enable
ufw status verbose

# Configure Fail2ban
echo "[4/10] Configuring Fail2ban..."
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
destemail = admin@ytempire.com
sender = fail2ban@ytempire.com
action = %(action_mwl)s

[sshd]
enabled = true
port = 22
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log

[nginx-noscript]
enabled = true
port = http,https
filter = nginx-noscript
logpath = /var/log/nginx/access.log
maxretry = 6

[nginx-badbots]
enabled = true
port = http,https
filter = nginx-badbots
logpath = /var/log/nginx/access.log
maxretry = 2

[nginx-noproxy]
enabled = true
port = http,https
filter = nginx-noproxy
logpath = /var/log/nginx/error.log
maxretry = 2
EOF

systemctl restart fail2ban
systemctl enable fail2ban

# SSH Hardening
echo "[5/10] Hardening SSH configuration..."
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

cat > /etc/ssh/sshd_config.d/99-ytempire-hardening.conf << 'EOF'
# YTEmpire SSH Hardening Configuration
Protocol 2
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
PermitEmptyPasswords no
MaxAuthTries 3
MaxSessions 10
ClientAliveInterval 300
ClientAliveCountMax 2
LoginGraceTime 60
X11Forwarding no
AllowUsers ytempire
DenyUsers root
EOF

# Create ytempire user if not exists
if ! id -u ytempire > /dev/null 2>&1; then
    echo "Creating ytempire user..."
    useradd -m -s /bin/bash ytempire
    usermod -aG sudo ytempire
    echo "Please set password for ytempire user:"
    passwd ytempire
fi

systemctl restart sshd

# Configure automatic security updates
echo "[6/10] Configuring automatic security updates..."
cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESM:${distro_codename}";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Kernel-Packages "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
Unattended-Upgrade::Automatic-Reboot-Time "03:00";
EOF

cat > /etc/apt/apt.conf.d/20auto-upgrades << 'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Download-Upgradeable-Packages "1";
APT::Periodic::AutocleanInterval "7";
APT::Periodic::Unattended-Upgrade "1";
EOF

# Configure sysctl for security
echo "[7/10] Configuring kernel security parameters..."
cat > /etc/sysctl.d/99-ytempire-security.conf << 'EOF'
# Network security
net.ipv4.tcp_syncookies = 1
net.ipv4.ip_forward = 0
net.ipv6.conf.all.forwarding = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0
net.ipv4.conf.default.secure_redirects = 0
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
net.ipv4.tcp_timestamps = 0

# Kernel hardening
kernel.randomize_va_space = 2
kernel.yama.ptrace_scope = 1
kernel.core_uses_pid = 1
kernel.kptr_restrict = 2
kernel.dmesg_restrict = 1
kernel.sysrq = 0
fs.suid_dumpable = 0
EOF

sysctl -p /etc/sysctl.d/99-ytempire-security.conf

# Setup AIDE (File Integrity Monitoring)
echo "[8/10] Setting up AIDE file integrity monitoring..."
aideinit
mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db

# Create AIDE check script
cat > /etc/cron.daily/aide-check << 'EOF'
#!/bin/bash
/usr/bin/aide --check | mail -s "AIDE Daily Check" admin@ytempire.com
EOF
chmod +x /etc/cron.daily/aide-check

# Setup ClamAV
echo "[9/10] Configuring ClamAV antivirus..."
systemctl stop clamav-freshclam
freshclam
systemctl start clamav-freshclam
systemctl enable clamav-freshclam

# Create daily scan script
cat > /etc/cron.daily/clamav-scan << 'EOF'
#!/bin/bash
LOGFILE="/var/log/clamav/daily-scan.log"
clamscan -r --infected --remove /home /var/www /app >> $LOGFILE
echo "ClamAV scan completed: $(date)" >> $LOGFILE
EOF
chmod +x /etc/cron.daily/clamav-scan

# Create security audit script
echo "[10/10] Creating security audit script..."
cat > /usr/local/bin/ytempire-security-audit << 'EOF'
#!/bin/bash

echo "YTEmpire Security Audit Report"
echo "=============================="
echo "Date: $(date)"
echo ""

echo "1. System Information:"
uname -a
echo ""

echo "2. Failed Login Attempts (last 24h):"
grep "Failed password" /var/log/auth.log | tail -20
echo ""

echo "3. Firewall Status:"
ufw status numbered
echo ""

echo "4. Fail2ban Status:"
fail2ban-client status
echo ""

echo "5. Open Ports:"
ss -tulpn | grep LISTEN
echo ""

echo "6. Active Users:"
w
echo ""

echo "7. Recent System Logins:"
last -10
echo ""

echo "8. Disk Usage:"
df -h
echo ""

echo "9. Running Services:"
systemctl list-units --state=running --type=service
echo ""

echo "10. Security Updates Available:"
apt list --upgradable 2>/dev/null | grep -i security
echo ""

echo "Audit complete."
EOF
chmod +x /usr/local/bin/ytempire-security-audit

# Create logrotate configuration
cat > /etc/logrotate.d/ytempire << 'EOF'
/var/log/ytempire/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 640 ytempire ytempire
    sharedscripts
    postrotate
        systemctl reload nginx > /dev/null 2>&1 || true
    endscript
}
EOF

# Final summary
echo ""
echo "========================================="
echo "Security Configuration Complete!"
echo "========================================="
echo ""
echo "Summary of changes:"
echo "1. UFW firewall configured and enabled"
echo "2. Fail2ban configured for intrusion prevention"
echo "3. SSH hardened with key-only authentication"
echo "4. Automatic security updates enabled"
echo "5. Kernel security parameters configured"
echo "6. AIDE file integrity monitoring setup"
echo "7. ClamAV antivirus configured"
echo "8. Security audit script created"
echo ""
echo "Important next steps:"
echo "1. Generate SSH keys for the ytempire user"
echo "2. Configure SSL certificates with Let's Encrypt"
echo "3. Review and adjust firewall rules as needed"
echo "4. Set up regular backups"
echo "5. Configure log monitoring and alerting"
echo ""
echo "Run security audit: ytempire-security-audit"
echo ""