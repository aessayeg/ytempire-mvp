# HashiCorp Vault Configuration
# Owner: Security Engineer #1

# Vault server configuration
ui = true
api_addr = "http://127.0.0.1:8200"
cluster_addr = "https://127.0.0.1:8201"

# Storage backend - using file storage for development/testing
# In production, use Consul, etcd, or cloud storage
storage "file" {
  path = "/opt/vault/data"
}

# Alternative storage for production
# storage "consul" {
#   address = "127.0.0.1:8500"
#   path    = "vault/"
# }

# Listener configuration
listener "tcp" {
  address       = "0.0.0.0:8200"
  tls_disable   = 1  # Disable for development - enable in production
  # tls_cert_file = "/opt/vault/tls/vault.crt"
  # tls_key_file  = "/opt/vault/tls/vault.key"
}

# Telemetry
telemetry {
  prometheus_retention_time = "30s"
  disable_hostname = true
}

# Logging
log_level = "INFO"
log_format = "json"

# Disable mlock for development - enable in production
disable_mlock = true

# Cluster configuration
cluster_name = "ytempire-vault-cluster"

# Plugin directory
plugin_directory = "/opt/vault/plugins"

# Default lease configuration
default_lease_ttl = "768h"   # 32 days
max_lease_ttl = "8760h"      # 1 year

# API configuration
default_max_request_duration = "90s"