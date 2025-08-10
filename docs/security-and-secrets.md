# Security and Secrets Management

## Overview

YTEmpire implements a comprehensive security framework with advanced secrets management, automatic rotation, audit logging, and encryption capabilities using HashiCorp Vault and custom security services.

## Architecture

### Core Components

1. **HashiCorp Vault Integration**
   - Centralized secrets storage
   - Authentication and authorization
   - Secrets versioning and audit trails

2. **Enhanced Secrets Manager**
   - Local encryption layer
   - Automatic rotation scheduling
   - Metadata management and audit logging

3. **Celery-based Automation**
   - Scheduled rotation tasks
   - Security monitoring and reporting
   - Integrity validation

4. **API Security Layer**
   - Admin-only access controls
   - Comprehensive audit logging
   - Task status monitoring

## Features

### 🔐 Secrets Management
- **Multiple Secret Types**: API keys, database passwords, encryption keys, JWT secrets, webhooks, OAuth tokens, service accounts, TLS certificates
- **Metadata Tracking**: Creation time, expiration, rotation history, tags
- **Local Encryption**: Optional additional encryption layer before Vault storage
- **Bulk Operations**: Create, rotate, and manage secrets in bulk

### 🔄 Automatic Rotation
- **Scheduled Rotation**: Configurable rotation intervals
- **Auto-generation**: Secure random value generation based on secret type
- **Rotation Status Tracking**: Pending, in-progress, completed, failed states
- **Retry Logic**: Automatic retry with exponential backoff

### 📊 Security Monitoring
- **Audit Logging**: Comprehensive access and modification logs
- **Integrity Validation**: Regular integrity checks for stored secrets
- **Security Reports**: Daily security status reports
- **Metrics Integration**: Prometheus metrics for monitoring

### 🚨 Compliance & Auditing
- **Access Tracking**: Who accessed what and when
- **Rotation History**: Complete rotation timeline
- **Security Alerts**: Notifications for failed rotations or suspicious activity
- **Compliance Reports**: Automated compliance status reports

## Secret Types

| Type | Description | Auto-Generation | Default Length |
|------|-------------|-----------------|----------------|
| `API_KEY` | Third-party API keys | ✅ | 32 chars |
| `DATABASE_PASSWORD` | Database credentials | ✅ | 16 chars |
| `ENCRYPTION_KEY` | Encryption keys | ✅ | 32 bytes (base64) |
| `JWT_SECRET` | JWT signing secrets | ✅ | 32 chars |
| `WEBHOOK_SECRET` | Webhook signatures | ✅ | 32 chars |
| `OAUTH_TOKEN` | OAuth access tokens | ❌ | N/A |
| `SERVICE_ACCOUNT_KEY` | Service account keys | ❌ | N/A |
| `TLS_CERTIFICATE` | TLS certificates | ❌ | N/A |

## API Endpoints

### Secrets Management

```
POST   /api/v1/secrets/                    # Create secret
GET    /api/v1/secrets/                    # List secrets
GET    /api/v1/secrets/{id}                # Get secret details
POST   /api/v1/secrets/{id}/rotate         # Rotate secret
DELETE /api/v1/secrets/{id}                # Delete secret
POST   /api/v1/secrets/rotate-bulk         # Bulk rotation
```

### Security & Auditing

```
GET    /api/v1/secrets/audit/report        # Security report
POST   /api/v1/secrets/audit/integrity-check # Integrity validation
GET    /api/v1/secrets/stats               # Statistics
GET    /api/v1/secrets/types               # Available types
GET    /api/v1/secrets/task/{id}/status    # Task status
```

## Configuration

### Environment Variables

```env
# HashiCorp Vault
VAULT_URL=http://localhost:8200
VAULT_TOKEN=your-vault-token
USE_VAULT=true

# Local Encryption (for additional security layer)
LOCAL_ENCRYPTION_KEY=base64-encoded-key

# Security Settings
SECRET_ROTATION_ENABLED=true
DEFAULT_ROTATION_INTERVAL_DAYS=90
SECURITY_AUDIT_ENABLED=true
```

### Vault Configuration

```hcl
# vault-config.hcl
storage "consul" {
  address = "127.0.0.1:8500"
  path    = "vault/"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 1
}

api_addr = "http://127.0.0.1:8200"
cluster_addr = "https://127.0.0.1:8201"
ui = true
```

## Usage Examples

### Create a Secret

```bash
curl -X POST "http://localhost:8000/api/v1/secrets/" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "name": "openai_api_key",
    "value": "sk-1234567890abcdef",
    "secret_type": "API_KEY",
    "rotation_interval_days": 30,
    "tags": {
      "service": "ai",
      "environment": "production"
    },
    "encrypt_locally": true
  }'
```

### List Secrets

```bash
curl -X GET "http://localhost:8000/api/v1/secrets/?secret_type=API_KEY&expired_only=false" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Rotate a Secret

```bash
curl -X POST "http://localhost:8000/api/v1/secrets/abc123def456/rotate" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Get Security Report

```bash
curl -X GET "http://localhost:8000/api/v1/secrets/audit/report?generate_new=true" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Automated Tasks

### Scheduled Tasks

| Task | Schedule | Description |
|------|----------|-------------|
| `check_expired_secrets` | Daily | Check for expired secrets and trigger rotation |
| `audit_secrets_access` | Every 12 hours | Audit access patterns for security |
| `validate_secret_integrity` | Weekly | Validate integrity of stored secrets |
| `cleanup_old_secret_versions` | Weekly | Clean up old secret versions |
| `generate_security_report` | Daily | Generate comprehensive security report |

### Manual Tasks

```python
from app.tasks.secret_rotation import rotate_secret_task, bulk_rotate_secrets

# Rotate single secret
result = rotate_secret_task.delay("secret_id_123")

# Bulk rotate all API keys
result = bulk_rotate_secrets.delay("API_KEY")
```

## Security Best Practices

### 1. Access Control
- **Admin Only**: Secrets management requires admin privileges
- **Principle of Least Privilege**: Grant minimum necessary access
- **Regular Access Reviews**: Audit who has secrets access

### 2. Encryption
- **Transit Encryption**: HTTPS for all API communications
- **At-Rest Encryption**: Vault encryption + optional local encryption
- **Key Management**: Regular rotation of encryption keys

### 3. Rotation Strategy
- **Regular Rotation**: Set appropriate intervals based on secret sensitivity
- **Automated Rotation**: Use automatic rotation for supported secret types
- **Emergency Rotation**: Capability for immediate rotation when compromised

### 4. Monitoring & Alerting
- **Access Monitoring**: Log all secret access attempts
- **Failed Rotation Alerts**: Monitor for rotation failures
- **Security Metrics**: Track security posture over time

### 5. Backup & Recovery
- **Vault Snapshots**: Regular Vault data backups
- **Disaster Recovery**: Documented recovery procedures
- **Testing**: Regular restore testing

## Development Setup

### 1. Start HashiCorp Vault

```bash
# Development mode (not for production)
vault server -dev -dev-root-token-id="dev-token"

# Initialize Vault (production setup)
vault operator init
vault operator unseal <unseal-key>
```

### 2. Configure Vault

```bash
# Enable KV v2 secrets engine
vault secrets enable -version=2 kv

# Create policies
vault policy write ytempire-secrets - <<EOF
path "kv/data/ytempire/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
EOF

# Create token
vault token create -policy="ytempire-secrets"
```

### 3. Start Application

```bash
# Set environment variables
export VAULT_URL="http://localhost:8200"
export VAULT_TOKEN="your-vault-token"
export USE_VAULT="true"

# Start application
uvicorn app.main:app --reload
```

### 4. Test Secrets Management

```bash
# Create test secret
curl -X POST "http://localhost:8000/api/v1/secrets/" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "name": "test_api_key",
    "value": "test-value-123",
    "secret_type": "API_KEY",
    "rotation_interval_days": 1
  }'

# Wait for rotation (or trigger manually)
# Check rotation status
curl -X GET "http://localhost:8000/api/v1/secrets/stats" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Production Deployment

### 1. Vault High Availability

```yaml
# docker-compose.vault.yml
version: '3.8'
services:
  vault:
    image: vault:1.15
    cap_add:
      - IPC_LOCK
    environment:
      VAULT_DEV_ROOT_TOKEN_ID: ""  # Don't use in production
      VAULT_DEV_LISTEN_ADDRESS: "0.0.0.0:8200"
    volumes:
      - vault-data:/vault/data
      - ./vault-config.hcl:/vault/config/vault-config.hcl
    ports:
      - "8200:8200"
    command: vault server -config=/vault/config/vault-config.hcl
```

### 2. Security Hardening

1. **Use TLS**: Enable TLS for all Vault communications
2. **Authentication**: Use proper authentication methods (not dev tokens)
3. **Network Security**: Restrict network access to Vault
4. **Audit Logging**: Enable Vault audit logging
5. **Monitoring**: Set up monitoring and alerting

### 3. Backup Strategy

```bash
# Vault snapshots
vault operator raft snapshot save backup.snap

# Database backups
pg_dump ytempire_db > backup.sql

# Application secrets backup
# Implement custom backup procedures for critical secrets
```

## Monitoring & Metrics

### Prometheus Metrics

```
# Secrets management metrics
secrets_created_total{type="API_KEY"}
secrets_accessed_total
secrets_rotated_successfully_total
secrets_rotation_errors_total{operation="rotate"}
secret_integrity_checks_completed_total
expired_secrets_count
```

### Grafana Dashboard

Key metrics to monitor:
- Number of secrets by type
- Rotation success/failure rates
- Access patterns and anomalies
- Time since last rotation
- Security audit scores

### Alerts

```yaml
# alerts.yml
groups:
- name: secrets-security
  rules:
  - alert: ExpiredSecretsHigh
    expr: expired_secrets_count > 5
    for: 1h
    labels:
      severity: warning
    annotations:
      summary: "High number of expired secrets"

  - alert: RotationFailure
    expr: increase(secrets_rotation_errors_total[1h]) > 3
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Multiple secret rotation failures"
```

## Compliance & Auditing

### Audit Requirements

The system provides comprehensive audit trails for:
- Secret creation, access, modification, deletion
- Rotation events and status changes
- Failed access attempts
- Administrative actions

### Compliance Standards

Supports compliance with:
- **SOC 2**: Security controls and audit trails
- **PCI DSS**: Secure handling of sensitive data
- **GDPR**: Data protection and privacy
- **HIPAA**: Healthcare data security (if applicable)

### Audit Reports

Automated reports include:
- Daily security status
- Monthly compliance summary
- Quarterly access reviews
- Annual security assessment

## Troubleshooting

### Common Issues

1. **Vault Connection Failed**
   ```bash
   # Check Vault status
   curl http://localhost:8200/v1/sys/health
   
   # Verify token
   vault token lookup
   ```

2. **Secret Not Found**
   ```bash
   # List secrets in Vault
   vault kv list kv/ytempire/secrets/
   
   # Check application logs
   docker logs ytempire-backend
   ```

3. **Rotation Failures**
   ```bash
   # Check Celery worker status
   celery -A app.core.celery_app inspect active
   
   # Review task logs
   celery -A app.core.celery_app events
   ```

4. **Permission Denied**
   ```bash
   # Check Vault policies
   vault token capabilities kv/data/ytempire/secrets/
   
   # Verify user permissions
   # (Check application admin roles)
   ```

### Debugging Commands

```bash
# Test Vault connectivity
vault status

# List secrets
vault kv list kv/ytempire/secrets/

# Get secret metadata
vault kv metadata kv/ytempire/secrets/secret_id

# Check audit logs
vault audit list
```

## Security Considerations

### Threat Model

**Threats Addressed:**
- Credential theft and misuse
- Insider threats
- Compliance violations
- Data breaches

**Attack Vectors Mitigated:**
- Hardcoded secrets in code
- Long-lived credentials
- Unencrypted secret storage
- Lack of access audit trails

### Security Controls

1. **Authentication & Authorization**
   - Multi-factor authentication for admin access
   - Role-based access control
   - API token validation

2. **Data Protection**
   - Encryption at rest and in transit
   - Secure key management
   - Data classification and handling

3. **Monitoring & Detection**
   - Real-time security monitoring
   - Anomaly detection
   - Incident response procedures

4. **Compliance & Governance**
   - Regular security assessments
   - Compliance reporting
   - Policy enforcement

---

**Owner**: Security Engineer #1  
**Last Updated**: 2025-08-10  
**Version**: 1.0.0