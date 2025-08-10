#!/bin/bash
# Vault Restore Script
# Owner: Security Engineer #1

set -e

if [ $# -eq 0 ]; then
    echo "❌ Usage: $0 <backup_file>"
    echo "📋 Available backups:"
    ls -la ./backups/vault_backup_*.json 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_FILE=$1
BACKUP_DIR="./backups"

if [ ! -f "$BACKUP_DIR/$BACKUP_FILE" ]; then
    echo "❌ Backup file not found: $BACKUP_DIR/$BACKUP_FILE"
    exit 1
fi

echo "🔄 Restoring Vault from backup: $BACKUP_FILE"

# Read Vault token
if [ -z "$VAULT_TOKEN" ]; then
    if [ -f "vault-credentials.json" ]; then
        VAULT_TOKEN=$(jq -r '.root_token' vault-credentials.json)
    else
        echo "❌ No Vault token found. Set VAULT_TOKEN environment variable."
        exit 1
    fi
fi

# Check if Vault is accessible
curl -s --header "X-Vault-Token: $VAULT_TOKEN" http://localhost:8200/v1/sys/health > /dev/null
if [ $? -ne 0 ]; then
    echo "❌ Cannot access Vault. Ensure Vault is running and unsealed."
    exit 1
fi

echo "⚠️  This will overwrite existing Vault data. Continue? (y/N)"
read -r CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "❌ Restore cancelled."
    exit 1
fi

echo "📁 Restoring secrets..."

# Extract timestamp from backup file
TIMESTAMP=$(echo $BACKUP_FILE | grep -o '[0-9]\{8\}_[0-9]\{6\}')

# Restore app config
if [ -f "$BACKUP_DIR/app_config_${TIMESTAMP}.json" ]; then
    echo "🔧 Restoring application configuration..."
    APP_CONFIG=$(cat "$BACKUP_DIR/app_config_${TIMESTAMP}.json")
    curl -s \
        --header "X-Vault-Token: $VAULT_TOKEN" \
        --request POST \
        --data "{\"data\": $APP_CONFIG}" \
        http://localhost:8200/v1/secret/data/ytempire/app/config
    
    if [ $? -eq 0 ]; then
        echo "✅ Application configuration restored"
    else
        echo "❌ Failed to restore application configuration"
    fi
fi

# Restore policies
if [ -f "$BACKUP_DIR/ytempire_app_policy_${TIMESTAMP}.json" ]; then
    echo "📋 Restoring policies..."
    curl -s \
        --header "X-Vault-Token: $VAULT_TOKEN" \
        --request PUT \
        --data @"$BACKUP_DIR/ytempire_app_policy_${TIMESTAMP}.json" \
        http://localhost:8200/v1/sys/policies/acl/ytempire-app
    
    if [ $? -eq 0 ]; then
        echo "✅ Policies restored"
    else
        echo "❌ Failed to restore policies"
    fi
fi

echo "✅ Vault restore completed from backup: $BACKUP_FILE"
echo "🧪 Test restored configuration:"
echo "   curl -s --header \"X-Vault-Token: \$VAULT_TOKEN\" http://localhost:8200/v1/secret/data/ytempire/app/config"