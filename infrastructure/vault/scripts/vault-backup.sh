#!/bin/bash
# Vault Backup Script
# Owner: Security Engineer #1

set -e

BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="vault_backup_${TIMESTAMP}.json"

echo "📦 Creating Vault backup..."

# Create backup directory
mkdir -p $BACKUP_DIR

# Check if Vault is sealed
VAULT_STATUS=$(curl -s http://localhost:8200/v1/sys/health | jq -r '.sealed')

if [ "$VAULT_STATUS" = "true" ]; then
    echo "❌ Vault is sealed. Cannot create backup."
    exit 1
fi

# Read Vault token from environment or file
if [ -z "$VAULT_TOKEN" ]; then
    if [ -f "vault-credentials.json" ]; then
        VAULT_TOKEN=$(jq -r '.root_token' vault-credentials.json)
    else
        echo "❌ No Vault token found. Set VAULT_TOKEN environment variable or ensure vault-credentials.json exists."
        exit 1
    fi
fi

echo "🔍 Backing up secrets..."

# Create backup structure
cat > "$BACKUP_DIR/$BACKUP_FILE" << EOF
{
    "backup_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "vault_version": "$(curl -s http://localhost:8200/v1/sys/version | jq -r '.version')",
    "secrets": {},
    "policies": {},
    "auth_methods": {},
    "mounts": {}
}
EOF

# Backup secrets
echo "🗃️  Backing up application secrets..."
SECRETS=$(curl -s \
    --header "X-Vault-Token: $VAULT_TOKEN" \
    http://localhost:8200/v1/secret/data/ytempire/app/config)

if [ "$?" -eq 0 ] && [ "$(echo $SECRETS | jq -r '.data')" != "null" ]; then
    echo $SECRETS | jq '.data.data' > "$BACKUP_DIR/app_config_${TIMESTAMP}.json"
fi

# Backup policies
echo "📋 Backing up policies..."
curl -s \
    --header "X-Vault-Token: $VAULT_TOKEN" \
    http://localhost:8200/v1/sys/policies/acl/ytempire-app \
    | jq > "$BACKUP_DIR/ytempire_app_policy_${TIMESTAMP}.json"

echo "✅ Backup completed: $BACKUP_DIR/$BACKUP_FILE"
echo "📁 Backup location: $(pwd)/$BACKUP_DIR"

# List backup files
echo ""
echo "📋 Available backups:"
ls -la $BACKUP_DIR/vault_backup_*.json | tail -5