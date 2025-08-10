#!/bin/bash

# Security Baseline Configuration Script
# Owner: Security Engineer #1
# Created: Day 1
# Purpose: Configure security baseline for YTEmpire infrastructure

set -e

echo "==========================================="
echo "YTEmpire Security Baseline Configuration"
echo "==========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
   echo "Please run as root (use sudo)"
   exit 1
fi

# 1. System Updates
echo "[1/10] Updating system packages..."
apt-get update && apt-get upgrade -y

# 2. Configure UFW Firewall
echo "[2/10] Configuring UFW firewall..."

# Reset UFW to defaults
ufw --force reset

# Default policies
ufw default deny incoming
ufw default allow outgoing
ufw default deny routed

# Allow SSH (rate limited)
ufw limit 22/tcp comment 'SSH rate limited'

# Allow HTTP and HTTPS
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'

# Allow application ports
ufw allow 3000/tcp comment 'Frontend Dev Server'
ufw allow 8000/tcp comment 'Backend API'
ufw allow 5432/tcp comment 'PostgreSQL'
ufw allow 6379/tcp comment 'Redis'
ufw allow 5555/tcp comment 'Flower'
ufw allow 5678/tcp comment 'N8N'
ufw allow 9090/tcp comment 'Prometheus'
ufw allow 3001/tcp comment 'Grafana'

# Enable UFW
ufw --force enable
ufw status verbose

# 3. Install and Configure Fail2ban
echo "[3/10] Installing and configuring Fail2ban..."

apt-get install -y fail2ban

# Create jail.local configuration
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
destemail = security@ytempire.com
sender = fail2ban@ytempire.com
action = %(action_mwl)s

[sshd]
enabled = true
port = 22
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[sshd-ddos]
enabled = true
port = 22
filter = sshd-ddos
logpath = /var/log/auth.log
maxretry = 10

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
port = http,https
logpath = /var/log/nginx/error.log

[nginx-badbots]
enabled = true
port = http,https
filter = nginx-badbots
logpath = /var/log/nginx/access.log
maxretry = 2

[postgresql]
enabled = true
port = 5432
filter = postgresql
logpath = /var/log/postgresql/*.log
maxretry = 3
EOF

# Create filter for PostgreSQL
cat > /etc/fail2ban/filter.d/postgresql.conf << 'EOF'
[Definition]
failregex = ^.*FATAL:  password authentication failed for user ".*"$
            ^.*FATAL:  no pg_hba.conf entry for host ".*"$
ignoreregex =
EOF

# Create filter for SSH DDoS
cat > /etc/fail2ban/filter.d/sshd-ddos.conf << 'EOF'
[Definition]
failregex = ^.*sshd.*: Connection from <HOST> port .*$
ignoreregex =
EOF

# Restart fail2ban
systemctl restart fail2ban
systemctl enable fail2ban

# 4. SSH Hardening
echo "[4/10] Hardening SSH configuration..."

# Backup original SSH config
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

# Apply hardened SSH configuration
cat > /etc/ssh/sshd_config.d/99-ytempire-hardening.conf << 'EOF'
# YTEmpire SSH Hardening Configuration

# Protocol and Port
Port 22
Protocol 2

# Authentication
PermitRootLogin no
PubkeyAuthentication yes
PasswordAuthentication no
PermitEmptyPasswords no
ChallengeResponseAuthentication no
MaxAuthTries 3
MaxSessions 10

# Security Features
StrictModes yes
IgnoreRhosts yes
HostbasedAuthentication no
X11Forwarding no
AllowUsers ytempire deploy

# Timeouts
ClientAliveInterval 300
ClientAliveCountMax 2
LoginGraceTime 60

# Logging
LogLevel VERBOSE
SyslogFacility AUTH

# Crypto Settings
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com,aes128-gcm@openssh.com
MACs hmac-sha2-512-etm@openssh.com,hmac-sha2-256-etm@openssh.com
KexAlgorithms curve25519-sha256,curve25519-sha256@libssh.org

# Banner
Banner /etc/ssh/banner.txt
EOF

# Create SSH banner
cat > /etc/ssh/banner.txt << 'EOF'
##############################################################
#                                                            #
#  YTEmpire Production System                               #
#  Unauthorized access is strictly prohibited               #
#  All activities are monitored and logged                  #
#                                                            #
##############################################################
EOF

# Restart SSH service
systemctl restart sshd

# 5. Install Security Tools
echo "[5/10] Installing security tools..."

apt-get install -y \
    auditd \
    aide \
    rkhunter \
    clamav \
    clamav-daemon \
    lynis \
    chkrootkit

# 6. Configure Auditd
echo "[6/10] Configuring audit daemon..."

cat > /etc/audit/rules.d/ytempire.rules << 'EOF'
# YTEmpire Audit Rules

# Monitor authentication events
-w /var/log/auth.log -p wa -k authentication
-w /etc/passwd -p wa -k passwd_changes
-w /etc/shadow -p wa -k shadow_changes
-w /etc/group -p wa -k group_changes

# Monitor sudo commands
-w /etc/sudoers -p wa -k sudoers_changes
-w /usr/bin/sudo -p x -k sudo_usage

# Monitor SSH
-w /etc/ssh/sshd_config -p wa -k sshd_config

# Monitor system calls
-a exit,always -F arch=b64 -S execve -k command_execution
-a exit,always -F arch=b64 -S socket -S bind -k network_binding

# Monitor file deletion
-a exit,always -F arch=b64 -S unlink -S rmdir -k file_deletion

# Monitor kernel modules
-w /sbin/insmod -p x -k kernel_modules
-w /sbin/rmmod -p x -k kernel_modules
-w /sbin/modprobe -p x -k kernel_modules
EOF

# Load audit rules
augenrules --load
systemctl restart auditd
systemctl enable auditd

# 7. Configure AIDE (File Integrity)
echo "[7/10] Initializing AIDE file integrity monitoring..."

# Initialize AIDE database
aideinit
mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db

# Create AIDE configuration
cat > /etc/aide/aide.conf.d/99_ytempire << 'EOF'
# YTEmpire AIDE Configuration

# Monitor application directories
/opt/ytempire R
/var/www R
/etc/ytempire R

# Monitor configuration files
/etc/nginx R
/etc/postgresql R
/etc/redis R

# Exclude log files
!/var/log/
!/tmp/
!/var/tmp/
EOF

# 8. System Hardening
echo "[8/10] Applying system hardening..."

# Disable unnecessary services
systemctl disable bluetooth.service 2>/dev/null || true
systemctl disable cups.service 2>/dev/null || true
systemctl disable avahi-daemon.service 2>/dev/null || true

# Kernel hardening via sysctl
cat > /etc/sysctl.d/99-ytempire-hardening.conf << 'EOF'
# YTEmpire Kernel Hardening

# Network Security
net.ipv4.tcp_syncookies = 1
net.ipv4.ip_forward = 0
net.ipv6.conf.all.forwarding = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
net.ipv4.tcp_timestamps = 0

# File System Security
fs.protected_hardlinks = 1
fs.protected_symlinks = 1
fs.suid_dumpable = 0

# Process Security
kernel.randomize_va_space = 2
kernel.yama.ptrace_scope = 1
kernel.core_uses_pid = 1

# Memory Security
kernel.kptr_restrict = 2
kernel.dmesg_restrict = 1
kernel.printk = 3 3 3 3
kernel.unprivileged_bpf_disabled = 1
kernel.unprivileged_userns_clone = 0

# Panic Settings
kernel.panic = 60
kernel.panic_on_oops = 1
EOF

# Apply sysctl settings
sysctl -p /etc/sysctl.d/99-ytempire-hardening.conf

# 9. Set up Log Rotation
echo "[9/10] Configuring log rotation..."

cat > /etc/logrotate.d/ytempire << 'EOF'
/var/log/ytempire/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 ytempire ytempire
    sharedscripts
    postrotate
        systemctl reload ytempire-backend || true
    endscript
}

/var/log/nginx/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 www-data adm
    sharedscripts
    postrotate
        nginx -s reload || true
    endscript
}
EOF

# 10. Create Security Check Script
echo "[10/10] Creating security check script..."

cat > /usr/local/bin/ytempire-security-check.sh << 'EOF'
#!/bin/bash

# YTEmpire Security Check Script
# Run daily via cron

echo "YTEmpire Security Check - $(date)"
echo "======================================="

# Check for failed login attempts
echo "Failed Login Attempts:"
grep "Failed password" /var/log/auth.log | tail -5

# Check for sudo usage
echo -e "\nRecent Sudo Usage:"
grep "sudo" /var/log/auth.log | tail -5

# Check open ports
echo -e "\nOpen Ports:"
ss -tuln | grep LISTEN

# Check for rootkits
echo -e "\nRootkit Check:"
rkhunter --check --skip-keypress --report-warnings-only

# Check system integrity
echo -e "\nFile Integrity Check:"
aide --check

# Check for security updates
echo -e "\nSecurity Updates:"
apt list --upgradable 2>/dev/null | grep -i security

# Disk usage
echo -e "\nDisk Usage:"
df -h | grep -E '^/dev/'

echo -e "\nSecurity check complete."
EOF

chmod +x /usr/local/bin/ytempire-security-check.sh

# Add to crontab
(crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/ytempire-security-check.sh >> /var/log/ytempire-security.log 2>&1") | crontab -

echo "==========================================="
echo "Security baseline configuration complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Create non-root user: useradd -m -s /bin/bash ytempire"
echo "2. Set up SSH keys for the new user"
echo "3. Configure application-specific security"
echo "4. Review and test firewall rules"
echo "5. Schedule regular security audits"
echo ""
echo "Security tools installed:"
echo "- UFW (firewall)"
echo "- Fail2ban (intrusion prevention)"
echo "- AIDE (file integrity)"
echo "- Auditd (system auditing)"
echo "- RKHunter (rootkit detection)"
echo "- ClamAV (antivirus)"
echo "- Lynis (security auditing)"
echo ""
echo "Run 'lynis audit system' for a comprehensive security audit"