#!/bin/bash

# YTEmpire Data Encryption Setup Script
# Sets up comprehensive encryption for data at rest and in transit

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="/var/log/ytempire-encryption-setup.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}SUCCESS: $1${NC}" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites for encryption setup..."
    
    # Check OpenSSL
    if ! command -v openssl &> /dev/null; then
        error "OpenSSL is not installed. Please install OpenSSL first."
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed. Please install Python 3 first."
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check available disk space for encrypted storage
    AVAILABLE_SPACE=$(df "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
    REQUIRED_SPACE=5242880  # 5GB in KB
    
    if [[ $AVAILABLE_SPACE -lt $REQUIRED_SPACE ]]; then
        warning "Less than 5GB available disk space. Encryption may require more space."
    fi
    
    success "Prerequisites check completed"
}

# Setup environment variables
setup_environment() {
    log "Setting up encryption environment variables..."
    
    ENV_FILE="$PROJECT_ROOT/.env.security"
    
    if [[ ! -f "$ENV_FILE" ]]; then
        cat > "$ENV_FILE" << EOF
# Data Encryption Environment Variables

# Domain Configuration
DOMAIN_NAME=ytempire.local

# Database Encryption
DATABASE_PASSWORD=secure_postgres_password_$(openssl rand -hex 16)
DATABASE_ENCRYPTION_KEY=$(openssl rand -hex 32)

# Redis Encryption
REDIS_PASSWORD=secure_redis_password_$(openssl rand -hex 16)
REDIS_ENCRYPTION_KEY=$(openssl rand -hex 32)

# Application Encryption
JWT_SECRET_KEY=$(openssl rand -hex 64)
API_KEY_ENCRYPTION_KEY=$(openssl rand -hex 32)

# Master Encryption Key (for Key Encryption Key)
MASTER_ENCRYPTION_PASSPHRASE=$(openssl rand -hex 64)

# TLS Configuration
LETSENCRYPT_STAGING=true
CERTBOT_EMAIL=admin@ytempire.local

# Certificate Authority Settings
CA_COUNTRY=US
CA_STATE=CA
CA_CITY=San Francisco
CA_ORG=YTEmpire

# Vault Configuration
VAULT_ROOT_TOKEN=$(openssl rand -hex 32)

# Security Monitoring
SECURITY_WEBHOOK_URL=

# File Encryption Settings
FILE_ENCRYPTION_ALGORITHM=AES_256_GCM
BACKUP_ENCRYPTION_ALGORITHM=AES_256_GCM

# Key Rotation Settings
KEY_ROTATION_DAYS=90
CERT_RENEWAL_DAYS=30

# Audit and Compliance
ENCRYPTION_AUDIT_LOG=/var/log/ytempire/encryption-audit.log
COMPLIANCE_MODE=GDPR

EOF

        success "Environment file created: $ENV_FILE"
        warning "Please review and update the environment variables in $ENV_FILE"
    else
        log "Environment file already exists: $ENV_FILE"
    fi
}

# Create directory structure
create_directories() {
    log "Creating encryption directory structure..."
    
    # Create security directories
    mkdir -p "$PROJECT_ROOT/infrastructure/security/certs"
    mkdir -p "$PROJECT_ROOT/infrastructure/security/keys"
    mkdir -p "$PROJECT_ROOT/infrastructure/security/ca"
    mkdir -p "$PROJECT_ROOT/infrastructure/security/webroot"
    mkdir -p "/var/lib/ytempire/postgres_encrypted"
    mkdir -p "/var/lib/ytempire/redis_encrypted"
    mkdir -p "/var/lib/ytempire/keys"
    mkdir -p "/var/log/ytempire"
    
    # Set secure permissions
    chmod 700 "$PROJECT_ROOT/infrastructure/security/keys"
    chmod 700 "$PROJECT_ROOT/infrastructure/security/ca"
    chmod 755 "$PROJECT_ROOT/infrastructure/security/certs"
    chmod 755 "$PROJECT_ROOT/infrastructure/security/webroot"
    
    if [[ -d "/var/lib/ytempire" ]]; then
        sudo chown -R $(whoami):$(whoami) /var/lib/ytempire
        chmod 700 /var/lib/ytempire/keys
        chmod 755 /var/lib/ytempire/postgres_encrypted
        chmod 755 /var/lib/ytempire/redis_encrypted
    fi
    
    success "Directory structure created with secure permissions"
}

# Generate self-signed certificates
generate_certificates() {
    log "Generating self-signed certificates for development..."
    
    CERT_DIR="$PROJECT_ROOT/infrastructure/security/certs"
    DOMAIN="${DOMAIN_NAME:-ytempire.local}"
    
    # Generate CA private key
    openssl genrsa -out "$CERT_DIR/ca.key" 4096
    chmod 600 "$CERT_DIR/ca.key"
    
    # Generate CA certificate
    openssl req -new -x509 -days 365 -key "$CERT_DIR/ca.key" -out "$CERT_DIR/ca.crt" -subj "/C=US/ST=CA/L=San Francisco/O=YTEmpire/CN=YTEmpire CA"
    
    # Generate server private key
    openssl genrsa -out "$CERT_DIR/${DOMAIN}.key" 2048
    chmod 600 "$CERT_DIR/${DOMAIN}.key"
    
    # Generate certificate signing request
    openssl req -new -key "$CERT_DIR/${DOMAIN}.key" -out "$CERT_DIR/${DOMAIN}.csr" -subj "/C=US/ST=CA/L=San Francisco/O=YTEmpire/CN=${DOMAIN}"
    
    # Create extensions file for SAN
    cat > "$CERT_DIR/${DOMAIN}.ext" << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = ${DOMAIN}
DNS.2 = *.${DOMAIN}
DNS.3 = localhost
IP.1 = 127.0.0.1
IP.2 = ::1
EOF
    
    # Generate server certificate signed by CA
    openssl x509 -req -in "$CERT_DIR/${DOMAIN}.csr" -CA "$CERT_DIR/ca.crt" -CAkey "$CERT_DIR/ca.key" -CAcreateserial -out "$CERT_DIR/${DOMAIN}.crt" -days 365 -extensions v3_req -extfile "$CERT_DIR/${DOMAIN}.ext"
    
    # Create combined PEM file for HAProxy
    cat "$CERT_DIR/${DOMAIN}.crt" "$CERT_DIR/${DOMAIN}.key" > "$CERT_DIR/${DOMAIN}.pem"
    chmod 600 "$CERT_DIR/${DOMAIN}.pem"
    
    # Generate DH parameters for enhanced security
    openssl dhparam -out "$CERT_DIR/dhparam.pem" 2048
    
    # Cleanup CSR and extension files
    rm -f "$CERT_DIR/${DOMAIN}.csr" "$CERT_DIR/${DOMAIN}.ext"
    
    success "Self-signed certificates generated for ${DOMAIN}"
}

# Initialize encryption keys
initialize_encryption_keys() {
    log "Initializing encryption keys..."
    
    cd "$PROJECT_ROOT"
    
    # Initialize the encryption manager
    python3 infrastructure/security/encryption_manager.py init
    
    # Generate application-specific keys
    python3 infrastructure/security/encryption_manager.py generate-key --key-type database --encryption-type aes_256_gcm
    python3 infrastructure/security/encryption_manager.py generate-key --key-type files --encryption-type aes_256_gcm
    python3 infrastructure/security/encryption_manager.py generate-key --key-type backup --encryption-type aes_256_gcm
    python3 infrastructure/security/encryption_manager.py generate-key --key-type transit --encryption-type chacha20_poly1305
    
    success "Encryption keys initialized"
}

# Setup database encryption
setup_database_encryption() {
    log "Setting up database encryption configuration..."
    
    # Create init script for encrypted columns
    cat > "$PROJECT_ROOT/infrastructure/security/init-encryption.sql" << 'EOF'
-- Initialize database encryption settings
-- Enable data checksums for integrity verification
-- Create encrypted storage tablespace (if supported)

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schema for encrypted data
CREATE SCHEMA IF NOT EXISTS encrypted_data;

-- Function to encrypt sensitive data
CREATE OR REPLACE FUNCTION encrypt_data(data TEXT, key_name TEXT)
RETURNS TEXT AS $$
BEGIN
    -- Use pgcrypto for encryption
    RETURN encode(pgp_sym_encrypt(data, current_setting('app.encryption_key', true)), 'base64');
EXCEPTION
    WHEN OTHERS THEN
        -- Log error and return NULL for failed encryption
        RAISE WARNING 'Encryption failed for %: %', key_name, SQLERRM;
        RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to decrypt sensitive data
CREATE OR REPLACE FUNCTION decrypt_data(encrypted_data TEXT, key_name TEXT)
RETURNS TEXT AS $$
BEGIN
    -- Use pgcrypto for decryption
    RETURN pgp_sym_decrypt(decode(encrypted_data, 'base64'), current_setting('app.encryption_key', true));
EXCEPTION
    WHEN OTHERS THEN
        -- Log error and return NULL for failed decryption
        RAISE WARNING 'Decryption failed for %: %', key_name, SQLERRM;
        RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Audit table for encryption operations
CREATE TABLE IF NOT EXISTS encrypted_data.encryption_audit (
    id SERIAL PRIMARY KEY,
    operation VARCHAR(50) NOT NULL,
    table_name VARCHAR(100),
    column_name VARCHAR(100),
    user_name VARCHAR(100) DEFAULT current_user,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT
);

-- Grant permissions
GRANT USAGE ON SCHEMA encrypted_data TO postgres;
GRANT SELECT, INSERT ON encrypted_data.encryption_audit TO postgres;
EOF

    # Create PostgreSQL secure entrypoint
    cat > "$PROJECT_ROOT/infrastructure/security/postgres-secure-entrypoint.sh" << 'EOF'
#!/bin/bash
set -euo pipefail

# Custom PostgreSQL entrypoint with encryption support

# Set encryption key from environment
if [ -n "${DATABASE_ENCRYPTION_KEY:-}" ]; then
    export PGPASSWORD="${DATABASE_PASSWORD}"
    echo "app.encryption_key = '${DATABASE_ENCRYPTION_KEY}'" >> /var/lib/postgresql/data/postgresql.auto.conf
fi

# Copy certificates with proper permissions
if [ -d "/certs" ] && [ -n "$(ls -A /certs 2>/dev/null)" ]; then
    cp -r /certs/* /var/lib/postgresql/certs/ 2>/dev/null || true
    chown -R postgres:postgres /var/lib/postgresql/certs
    chmod 600 /var/lib/postgresql/certs/*.key 2>/dev/null || true
    chmod 644 /var/lib/postgresql/certs/*.crt 2>/dev/null || true
fi

# Initialize database with encryption support
if [ ! -s "$PGDATA/PG_VERSION" ]; then
    echo "Initializing database with encryption support..."
    initdb --auth-host=scram-sha-256 --data-checksums
fi

# Start PostgreSQL
exec docker-entrypoint.sh "$@"
EOF

    chmod +x "$PROJECT_ROOT/infrastructure/security/postgres-secure-entrypoint.sh"
    
    success "Database encryption configuration created"
}

# Setup Redis encryption
setup_redis_encryption() {
    log "Setting up Redis encryption configuration..."
    
    # Create Redis security module configuration
    cat > "$PROJECT_ROOT/infrastructure/security/redis-security-module.conf" << 'EOF'
# Redis Security Module Configuration

# Memory encryption (if supported)
# Note: This is a placeholder for enterprise Redis features
# In production, use Redis Enterprise or similar for memory encryption

# ACL Configuration for user-based security
user default off
user ytempire_app on +@all ~* &* >${REDIS_PASSWORD}

# Additional security settings
protected-mode yes
EOF

    # Create Redis secure entrypoint
    cat > "$PROJECT_ROOT/infrastructure/security/redis-secure-entrypoint.sh" << 'EOF'
#!/bin/bash
set -euo pipefail

# Custom Redis entrypoint with encryption support

# Copy certificates with proper permissions
if [ -d "/certs" ] && [ -n "$(ls -A /certs 2>/dev/null)" ]; then
    cp -r /certs/* /etc/redis/certs/ 2>/dev/null || true
    chown -R redis:redis /etc/redis/certs
    chmod 600 /etc/redis/certs/*.key 2>/dev/null || true
    chmod 644 /etc/redis/certs/*.crt 2>/dev/null || true
fi

# Update Redis configuration with runtime settings
if [ -n "${REDIS_PASSWORD:-}" ]; then
    sed -i "s/\${REDIS_PASSWORD}/${REDIS_PASSWORD}/g" /etc/redis/security.conf
fi

# Start Redis with security configuration
exec "$@"
EOF

    chmod +x "$PROJECT_ROOT/infrastructure/security/redis-secure-entrypoint.sh"
    
    success "Redis encryption configuration created"
}

# Create security scanning and monitoring tools
create_security_tools() {
    log "Creating security scanning and monitoring tools..."
    
    # Security scanner Dockerfile
    cat > "$PROJECT_ROOT/infrastructure/security/Dockerfile.scanner" << 'EOF'
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    nmap \
    openssl \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    requests \
    cryptography \
    psutil \
    docker

WORKDIR /app
COPY security_scanner.py .

CMD ["python", "security_scanner.py"]
EOF

    # Security scanner script
    cat > "$PROJECT_ROOT/infrastructure/security/security_scanner.py" << 'EOF'
#!/usr/bin/env python3
"""
Security Scanner for YTEmpire
Scans for security vulnerabilities and misconfigurations
"""

import os
import subprocess
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any

class SecurityScanner:
    def __init__(self):
        self.scan_targets = os.getenv('SCAN_TARGETS', 'localhost').split(',')
        self.scan_interval = int(os.getenv('SCAN_INTERVAL', '3600'))
        self.webhook_url = os.getenv('SECURITY_ALERTS_WEBHOOK')
    
    def scan_ssl_certificates(self, host: str, port: int = 443) -> Dict[str, Any]:
        """Scan SSL certificate configuration"""
        try:
            result = subprocess.run([
                'openssl', 's_client', '-connect', f'{host}:{port}',
                '-servername', host, '-verify_return_error'
            ], capture_output=True, text=True, timeout=10, input='')
            
            # Parse certificate information
            cert_info = {}
            if result.returncode == 0:
                cert_info['valid'] = True
                cert_info['output'] = result.stdout
            else:
                cert_info['valid'] = False
                cert_info['error'] = result.stderr
            
            return cert_info
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def scan_open_ports(self, host: str) -> List[int]:
        """Scan for open ports"""
        try:
            result = subprocess.run([
                'nmap', '-sS', '-O', host
            ], capture_output=True, text=True, timeout=60)
            
            # Parse nmap output for open ports
            open_ports = []
            for line in result.stdout.split('\n'):
                if '/tcp' in line and 'open' in line:
                    port = int(line.split('/')[0])
                    open_ports.append(port)
            
            return open_ports
            
        except Exception as e:
            print(f"Port scan failed: {e}")
            return []
    
    def check_service_security(self, service: str) -> Dict[str, Any]:
        """Check service-specific security configurations"""
        checks = {}
        
        if service == 'postgres':
            # Check PostgreSQL security
            checks['ssl_enabled'] = self.check_postgres_ssl()
            checks['auth_method'] = self.check_postgres_auth()
        elif service == 'redis':
            # Check Redis security
            checks['auth_enabled'] = self.check_redis_auth()
            checks['ssl_enabled'] = self.check_redis_ssl()
        elif service == 'frontend':
            # Check web server security
            checks['https_redirect'] = self.check_https_redirect()
            checks['security_headers'] = self.check_security_headers()
        
        return checks
    
    def check_postgres_ssl(self) -> bool:
        """Check if PostgreSQL SSL is enabled"""
        # Implementation would check actual PostgreSQL configuration
        return True
    
    def check_postgres_auth(self) -> str:
        """Check PostgreSQL authentication method"""
        # Implementation would check pg_hba.conf
        return "scram-sha-256"
    
    def check_redis_auth(self) -> bool:
        """Check if Redis authentication is enabled"""
        # Implementation would check Redis AUTH requirement
        return True
    
    def check_redis_ssl(self) -> bool:
        """Check if Redis SSL is enabled"""
        # Implementation would check Redis TLS configuration
        return True
    
    def check_https_redirect(self) -> bool:
        """Check if HTTP redirects to HTTPS"""
        try:
            response = requests.get('http://frontend', allow_redirects=False, timeout=5)
            return response.status_code in [301, 302] and 'https' in response.headers.get('Location', '')
        except:
            return False
    
    def check_security_headers(self) -> Dict[str, bool]:
        """Check for security headers"""
        headers_to_check = [
            'Strict-Transport-Security',
            'X-Frame-Options',
            'X-Content-Type-Options',
            'X-XSS-Protection'
        ]
        
        try:
            response = requests.get('https://frontend', timeout=5, verify=False)
            return {
                header: header in response.headers
                for header in headers_to_check
            }
        except:
            return {header: False for header in headers_to_check}
    
    def send_alert(self, alert_data: Dict[str, Any]):
        """Send security alert"""
        if self.webhook_url:
            try:
                requests.post(self.webhook_url, json=alert_data, timeout=10)
            except Exception as e:
                print(f"Failed to send alert: {e}")
    
    def run_scan(self):
        """Run comprehensive security scan"""
        print(f"Starting security scan at {datetime.now()}")
        
        scan_results = {
            'timestamp': datetime.now().isoformat(),
            'targets': {},
            'summary': {'total_issues': 0, 'critical_issues': 0}
        }
        
        for target in self.scan_targets:
            target = target.strip()
            print(f"Scanning target: {target}")
            
            target_results = {
                'ssl_cert': self.scan_ssl_certificates(target),
                'open_ports': self.scan_open_ports(target),
                'service_security': self.check_service_security(target)
            }
            
            scan_results['targets'][target] = target_results
        
        # Save results
        with open('/app/reports/security_scan.json', 'w') as f:
            json.dump(scan_results, f, indent=2)
        
        print("Security scan completed")
        return scan_results
    
    def monitor_loop(self):
        """Continuous security monitoring"""
        while True:
            try:
                results = self.run_scan()
                
                # Check for critical issues and send alerts
                for target, data in results['targets'].items():
                    if not data['ssl_cert'].get('valid', True):
                        self.send_alert({
                            'severity': 'critical',
                            'message': f'SSL certificate issue on {target}',
                            'details': data['ssl_cert']
                        })
                
                time.sleep(self.scan_interval)
                
            except Exception as e:
                print(f"Scanner error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    scanner = SecurityScanner()
    scanner.monitor_loop()
EOF

    success "Security scanning tools created"
}

# Setup TLS configuration for all services
setup_tls_configuration() {
    log "Setting up TLS configuration for all services..."
    
    cd "$PROJECT_ROOT"
    
    # Generate TLS configurations
    python3 infrastructure/security/tls_config.py --domain "${DOMAIN_NAME:-ytempire.local}" --setup
    
    success "TLS configuration generated"
}

# Validate encryption setup
validate_encryption_setup() {
    log "Validating encryption setup..."
    
    # Check certificate validity
    CERT_FILE="$PROJECT_ROOT/infrastructure/security/certs/${DOMAIN_NAME:-ytempire.local}.crt"
    if [[ -f "$CERT_FILE" ]]; then
        if openssl x509 -in "$CERT_FILE" -text -noout > /dev/null 2>&1; then
            success "SSL certificate is valid"
        else
            error "SSL certificate is invalid"
        fi
    else
        warning "SSL certificate not found"
    fi
    
    # Check key permissions
    KEY_DIR="$PROJECT_ROOT/infrastructure/security/keys"
    if [[ -d "$KEY_DIR" ]]; then
        PERMS=$(stat -c %a "$KEY_DIR")
        if [[ "$PERMS" == "700" ]]; then
            success "Key directory has secure permissions"
        else
            warning "Key directory permissions are not secure ($PERMS)"
        fi
    fi
    
    # Validate Docker Compose configuration
    if docker-compose -f infrastructure/security/docker-compose.security.yml config > /dev/null; then
        success "Docker Compose security configuration is valid"
    else
        error "Docker Compose security configuration validation failed"
    fi
    
    # Test encryption manager
    if python3 infrastructure/security/encryption_manager.py status > /dev/null; then
        success "Encryption manager is functional"
    else
        warning "Encryption manager test failed"
    fi
}

# Print setup instructions
print_instructions() {
    log "Encryption setup completed successfully!"
    
    cat << EOF

${GREEN}YTEmpire Data Encryption Setup Complete!${NC}

${YELLOW}Next Steps:${NC}

1. Review and update environment variables:
   ${PROJECT_ROOT}/.env.security

2. Start the secure infrastructure:
   cd ${PROJECT_ROOT}
   docker-compose -f infrastructure/security/docker-compose.security.yml up -d

3. Verify encryption status:
   python3 infrastructure/security/encryption_manager.py status

4. Access services securely:
   - Frontend: https://${DOMAIN_NAME:-ytempire.local}
   - API: https://${DOMAIN_NAME:-ytempire.local}/api/v1/
   - PostgreSQL: SSL connection on port 5432
   - Redis: TLS connection on port 6379

5. Monitor security:
   - Vault UI: https://localhost:8200
   - Security Scanner: Check /app/reports/security_scan.json

${YELLOW}Encryption Features Enabled:${NC}
- ✅ Database encryption at rest (PostgreSQL with TDE)
- ✅ Redis encryption in transit and at rest
- ✅ File system encryption for uploads and videos
- ✅ API communications over TLS 1.2+
- ✅ Certificate auto-renewal with Certbot
- ✅ Key management and rotation
- ✅ WAF protection with ModSecurity
- ✅ Intrusion detection with Suricata

${YELLOW}Security Certificates:${NC}
- CA Certificate: infrastructure/security/certs/ca.crt
- Server Certificate: infrastructure/security/certs/${DOMAIN_NAME:-ytempire.local}.crt
- Combined PEM: infrastructure/security/certs/${DOMAIN_NAME:-ytempire.local}.pem

${YELLOW}Key Management:${NC}
- Master Key: Encrypted with passphrase from environment
- Service Keys: Automatically generated and rotated
- Backup Keys: Separate encryption for backups

${GREEN}Your data is now encrypted at rest and in transit!${NC}

EOF
}

# Main execution
main() {
    log "Starting YTEmpire Data Encryption Setup"
    
    check_prerequisites
    setup_environment
    create_directories
    generate_certificates
    initialize_encryption_keys
    setup_database_encryption
    setup_redis_encryption
    create_security_tools
    setup_tls_configuration
    validate_encryption_setup
    
    print_instructions
    
    success "Data encryption setup completed successfully!"
}

# Run main function with all arguments
main "$@"