# YTEmpire Application Vault Policy
# Owner: Security Engineer #1

# Application secrets access
path "secret/data/ytempire/app/*" {
  capabilities = ["read"]
}

path "secret/data/ytempire/api-keys/*" {
  capabilities = ["read"]
}

path "secret/data/ytempire/database/*" {
  capabilities = ["read"]
}

# Database dynamic secrets
path "database/creds/ytempire-app-role" {
  capabilities = ["read"]
}

# PKI for SSL certificates
path "pki/issue/ytempire-role" {
  capabilities = ["create", "update"]
}

# AWS dynamic secrets (if using AWS)
path "aws/creds/ytempire-s3-role" {
  capabilities = ["read"]
}

# Transit encryption
path "transit/encrypt/ytempire" {
  capabilities = ["create", "update"]
}

path "transit/decrypt/ytempire" {
  capabilities = ["create", "update"]
}

# Metadata access
path "secret/metadata/ytempire/*" {
  capabilities = ["list", "read"]
}

# Deny access to other applications' secrets
path "secret/data/other-apps/*" {
  capabilities = ["deny"]
}