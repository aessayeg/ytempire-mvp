#!/bin/bash
# HashiCorp Vault Initialization Script
# Owner: Security Engineer #1

set -e

echo "🔧 Initializing HashiCorp Vault for YTEmpire..."

# Wait for Vault to be ready
echo "⏳ Waiting for Vault to be ready..."
until curl -s http://localhost:8200/v1/sys/health > /dev/null; do
    echo "Waiting for Vault to start..."
    sleep 2
done

# Check if Vault is already initialized
VAULT_STATUS=$(curl -s http://localhost:8200/v1/sys/init | jq -r '.initialized')

if [ "$VAULT_STATUS" = "true" ]; then
    echo "✅ Vault is already initialized"
    exit 0
fi

echo "🚀 Initializing Vault..."

# Initialize Vault
INIT_RESPONSE=$(curl -s \
    --request POST \
    --data '{"secret_shares": 5, "secret_threshold": 3}' \
    http://localhost:8200/v1/sys/init)

# Extract keys and root token
UNSEAL_KEYS=$(echo $INIT_RESPONSE | jq -r '.keys[]')
ROOT_TOKEN=$(echo $INIT_RESPONSE | jq -r '.root_token')

echo "💾 Saving Vault credentials (STORE THESE SECURELY!)..."
cat > vault-credentials.json << EOF
{
    "unseal_keys": $(echo $INIT_RESPONSE | jq '.keys'),
    "root_token": "$ROOT_TOKEN",
    "initialized_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "🔓 Unsealing Vault..."
# Unseal with first 3 keys
KEYS_ARRAY=($UNSEAL_KEYS)
for i in {0..2}; do
    curl -s \
        --request POST \
        --data "{\"key\": \"${KEYS_ARRAY[$i]}\"}" \
        http://localhost:8200/v1/sys/unseal > /dev/null
done

echo "⚙️  Configuring Vault..."

# Enable secret engines
echo "📁 Enabling secret engines..."
curl -s \
    --header "X-Vault-Token: $ROOT_TOKEN" \
    --request POST \
    --data '{"type": "kv-v2"}' \
    http://localhost:8200/v1/sys/mounts/secret

curl -s \
    --header "X-Vault-Token: $ROOT_TOKEN" \
    --request POST \
    --data '{"type": "database"}' \
    http://localhost:8200/v1/sys/mounts/database

curl -s \
    --header "X-Vault-Token: $ROOT_TOKEN" \
    --request POST \
    --data '{"type": "transit"}' \
    http://localhost:8200/v1/sys/mounts/transit

curl -s \
    --header "X-Vault-Token: $ROOT_TOKEN" \
    --request POST \
    --data '{"type": "pki"}' \
    http://localhost:8200/v1/sys/mounts/pki

echo "🔑 Creating transit encryption key..."
curl -s \
    --header "X-Vault-Token: $ROOT_TOKEN" \
    --request POST \
    --data '{"type": "aes256-gcm96"}' \
    http://localhost:8200/v1/transit/keys/ytempire

echo "📋 Creating policies..."
# Upload policies
curl -s \
    --header "X-Vault-Token: $ROOT_TOKEN" \
    --request PUT \
    --data @../policies/ytempire-app-policy.hcl \
    http://localhost:8200/v1/sys/policies/acl/ytempire-app

echo "👤 Setting up authentication..."
# Enable userpass auth
curl -s \
    --header "X-Vault-Token: $ROOT_TOKEN" \
    --request POST \
    --data '{"type": "userpass"}' \
    http://localhost:8200/v1/sys/auth/userpass

# Create application user
curl -s \
    --header "X-Vault-Token: $ROOT_TOKEN" \
    --request POST \
    --data '{"password": "ytempire-app-secret", "policies": "ytempire-app"}' \
    http://localhost:8200/v1/auth/userpass/users/ytempire-app

echo "🗃️  Storing initial secrets..."

# Store application secrets
curl -s \
    --header "X-Vault-Token: $ROOT_TOKEN" \
    --request POST \
    --data '{
        "data": {
            "database_url": "postgresql://ytempire:ytempire_pass@localhost/ytempire_db",
            "secret_key": "ytempire-jwt-secret-key-change-in-production",
            "redis_url": "redis://localhost:6379/0"
        }
    }' \
    http://localhost:8200/v1/secret/data/ytempire/app/config

# Store API keys (placeholders)
curl -s \
    --header "X-Vault-Token: $ROOT_TOKEN" \
    --request POST \
    --data '{
        "data": {
            "openai_api_key": "sk-your-openai-api-key-here",
            "elevenlabs_api_key": "your-elevenlabs-api-key-here",
            "youtube_client_id": "your-youtube-client-id.googleusercontent.com",
            "youtube_client_secret": "your-youtube-client-secret"
        }
    }' \
    http://localhost:8200/v1/secret/data/ytempire/api-keys/external

# Configure database connection
curl -s \
    --header "X-Vault-Token: $ROOT_TOKEN" \
    --request POST \
    --data '{
        "plugin_name": "postgresql-database-plugin",
        "connection_url": "postgresql://{{username}}:{{password}}@localhost:5432/ytempire_db",
        "allowed_roles": "ytempire-app-role",
        "username": "ytempire",
        "password": "ytempire_pass"
    }' \
    http://localhost:8200/v1/database/config/ytempire-db

# Create database role
curl -s \
    --header "X-Vault-Token: $ROOT_TOKEN" \
    --request POST \
    --data '{
        "db_name": "ytempire-db",
        "creation_statements": ["CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD \"{{password}}\" VALID UNTIL \"{{expiration}}\"; GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO \"{{name}}\";"],
        "default_ttl": "1h",
        "max_ttl": "24h"
    }' \
    http://localhost:8200/v1/database/roles/ytempire-app-role

echo "✅ Vault initialization completed successfully!"
echo ""
echo "⚠️  IMPORTANT: Save vault-credentials.json securely and remove it from this location!"
echo "   Root Token: $ROOT_TOKEN"
echo ""
echo "🔗 Vault UI: http://localhost:8200"
echo "👤 App User: ytempire-app"
echo "🔐 App Password: ytempire-app-secret"
echo ""
echo "🧪 Test Vault access:"
echo "   curl -s --header \"X-Vault-Token: \$ROOT_TOKEN\" http://localhost:8200/v1/secret/data/ytempire/app/config"