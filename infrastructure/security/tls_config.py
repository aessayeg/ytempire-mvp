#!/usr/bin/env python3
"""
TLS Configuration Management for YTEmpire
Comprehensive TLS setup for all services
"""

import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path

class TLSConfigurator:
    """Manages TLS configuration for all services"""
    
    def __init__(self, base_domain: str = "ytempire.local"):
        self.base_domain = base_domain
        self.cert_dir = Path("/etc/ytempire/certs")
        self.cert_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
        
    def generate_nginx_tls_config(self) -> str:
        """Generate Nginx TLS configuration"""
        return f"""
# TLS Configuration for Nginx (Frontend)
server {{
    listen 80;
    listen [::]:80;
    server_name {self.base_domain} *.{self.base_domain};
    
    # Redirect HTTP to HTTPS
    location / {{
        return 301 https://$server_name$request_uri;
    }}
    
    # Let's Encrypt challenge
    location /.well-known/acme-challenge/ {{
        root /var/www/certbot;
    }}
}}

server {{
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name {self.base_domain} *.{self.base_domain};
    
    # SSL Certificate Configuration
    ssl_certificate /etc/nginx/certs/{self.base_domain}.crt;
    ssl_certificate_key /etc/nginx/certs/{self.base_domain}.key;
    ssl_trusted_certificate /etc/nginx/certs/{self.base_domain}.crt;
    
    # SSL Protocol Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384:ECDHE-RSA-AES128-SHA:ECDHE-RSA-AES256-SHA:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES128-SHA256:DHE-RSA-AES256-SHA256:DHE-RSA-AES128-SHA:DHE-RSA-AES256-SHA:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!3DES:!MD5:!PSK;
    ssl_prefer_server_ciphers on;
    ssl_ecdh_curve secp384r1;
    
    # SSL Session Configuration
    ssl_session_timeout 10m;
    ssl_session_cache shared:SSL:10m;
    ssl_session_tickets off;
    
    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    resolver 8.8.8.8 8.8.4.4 valid=300s;
    resolver_timeout 5s;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # Frontend Application
    location / {{
        try_files $uri $uri/ /index.html;
        root /usr/share/nginx/html;
        index index.html;
        
        # Cache static assets
        location ~* \\.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {{
            expires 1y;
            add_header Cache-Control "public, immutable";
        }}
    }}
    
    # API Proxy
    location /api/ {{
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port $server_port;
        proxy_cache_bypass $http_upgrade;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }}
    
    # WebSocket Proxy
    location /ws/ {{
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }}
}}
"""

    def generate_haproxy_tls_config(self) -> str:
        """Generate HAProxy TLS configuration"""
        return f"""
# HAProxy TLS Configuration
global
    daemon
    maxconn 4096
    log stdout local0 info
    
    # SSL Configuration
    ssl-default-bind-ciphers ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!SHA1:!AESCCM
    ssl-default-bind-options no-sslv3 no-tlsv10 no-tlsv11 ssl-min-ver TLSv1.2
    ssl-default-server-ciphers ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!SHA1:!AESCCM
    ssl-default-server-options no-sslv3 no-tlsv10 no-tlsv11 ssl-min-ver TLSv1.2
    
    # Certificate bundle
    tune.ssl.default-dh-param 2048

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    option dontlognull
    option http-server-close
    option forwardfor
    option redispatch
    retries 3
    
    # Health checks
    option httpchk GET /health
    http-check expect status 200

# Frontend for HTTPS traffic
frontend ytempire_https
    bind *:443 ssl crt /etc/ssl/certs/{self.base_domain}.pem
    
    # Security headers
    http-response set-header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
    http-response set-header X-Frame-Options "DENY"
    http-response set-header X-Content-Type-Options "nosniff"
    http-response set-header X-XSS-Protection "1; mode=block"
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s),http_err_rate(10s)
    http-request track-sc0 src
    http-request deny if {{ sc_http_req_rate(0) gt 100 }}
    
    # Route based on path
    acl is_api path_beg /api/
    acl is_websocket path_beg /ws/
    
    use_backend api_backend if is_api
    use_backend websocket_backend if is_websocket
    default_backend frontend_backend

# Frontend for HTTP traffic (redirect to HTTPS)
frontend ytempire_http
    bind *:80
    redirect scheme https code 301

# Backend configurations remain the same
backend api_backend
    balance roundrobin
    option httpchk GET /health
    
    # Enable SSL for backend communication (optional)
    # server api1 backend:8000 check ssl verify none
    server-template api- 1-10 backend:8000 check inter 2s

backend websocket_backend
    balance source
    option httpchk GET /health
    
    server-template ws- 1-5 backend:8000 check inter 2s

backend frontend_backend
    balance roundrobin
    option httpchk GET /
    
    server-template frontend- 1-5 frontend:80 check inter 5s
"""

    def generate_postgres_tls_config(self) -> str:
        """Generate PostgreSQL TLS configuration"""
        return f"""
# PostgreSQL TLS Configuration
# Add to postgresql.conf

# Connection Settings
listen_addresses = '*'
port = 5432

# SSL Configuration
ssl = on
ssl_cert_file = '/var/lib/postgresql/certs/{self.base_domain}.crt'
ssl_key_file = '/var/lib/postgresql/certs/{self.base_domain}.key'
ssl_ca_file = '/var/lib/postgresql/certs/ca.crt'

# SSL Protocol Settings
ssl_ciphers = 'HIGH:MEDIUM:+3DES:!aNULL'
ssl_prefer_server_ciphers = on
ssl_min_protocol_version = 'TLSv1.2'
ssl_max_protocol_version = 'TLSv1.3'

# Client Authentication
ssl_crl_file = ''

# Logging
log_connections = on
log_disconnections = on
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,ssl=%c,client=%h '

# Connection Security
password_encryption = scram-sha-256

# Host-based Authentication (pg_hba.conf)
# Replace with actual configuration:
# hostssl all all 0.0.0.0/0 scram-sha-256
# hostssl replication replicator 0.0.0.0/0 scram-sha-256
"""

    def generate_redis_tls_config(self) -> str:
        """Generate Redis TLS configuration"""
        return f"""
# Redis TLS Configuration
# Add to redis.conf

# Basic Configuration
port 0
tls-port 6379

# TLS Configuration
tls-cert-file /etc/redis/certs/{self.base_domain}.crt
tls-key-file /etc/redis/certs/{self.base_domain}.key
tls-ca-cert-file /etc/redis/certs/ca.crt

# TLS Protocol Settings
tls-protocols "TLSv1.2 TLSv1.3"
tls-ciphersuites "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_128_GCM_SHA256"
tls-ciphers "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!SHA1:!AESCCM"
tls-prefer-server-ciphers yes

# TLS Session Configuration
tls-session-caching no
tls-session-cache-size 5000
tls-session-cache-timeout 60

# Client Authentication
tls-auth-clients yes

# DH Parameters
tls-dh-params-file /etc/redis/certs/dhparam.pem

# Security
requirepass your_redis_password
"""

    def generate_docker_compose_tls_config(self) -> str:
        """Generate Docker Compose with TLS configuration"""
        return f"""
version: '3.8'

services:
  # Frontend with TLS termination
  frontend:
    image: nginx:alpine
    volumes:
      - ./frontend/dist:/usr/share/nginx/html
      - ./infrastructure/security/nginx-tls.conf:/etc/nginx/nginx.conf
      - ./infrastructure/security/certs:/etc/nginx/certs:ro
    ports:
      - "80:80"
      - "443:443"
    networks:
      - ytempire_network
    restart: unless-stopped
    depends_on:
      - backend

  # Backend API server
  backend:
    build: ./backend
    environment:
      DATABASE_URL: postgresql://postgres:password@postgres:5432/ytempire?sslmode=require
      REDIS_URL: rediss://redis:6379/0  # Redis with TLS
      TLS_ENABLED: "true"
      TLS_CERT_FILE: /app/certs/{self.base_domain}.crt
      TLS_KEY_FILE: /app/certs/{self.base_domain}.key
    volumes:
      - ./infrastructure/security/certs:/app/certs:ro
    networks:
      - ytempire_network
    restart: unless-stopped
    depends_on:
      - postgres
      - redis

  # PostgreSQL with TLS
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ytempire
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_INITDB_ARGS: "--auth-host=scram-sha-256"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./infrastructure/security/postgres-tls.conf:/etc/postgresql/postgresql.conf
      - ./infrastructure/security/certs:/var/lib/postgresql/certs:ro
      - ./infrastructure/security/pg_hba.conf:/var/lib/postgresql/data/pg_hba.conf
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    ports:
      - "5432:5432"
    networks:
      - ytempire_network
    restart: unless-stopped

  # Redis with TLS
  redis:
    image: redis:7-alpine
    command: redis-server /etc/redis/redis.conf
    volumes:
      - redis_data:/data
      - ./infrastructure/security/redis-tls.conf:/etc/redis/redis.conf
      - ./infrastructure/security/certs:/etc/redis/certs:ro
    ports:
      - "6379:6379"
    networks:
      - ytempire_network
    restart: unless-stopped

  # Certificate management with Certbot
  certbot:
    image: certbot/certbot
    volumes:
      - ./infrastructure/security/certs:/etc/letsencrypt
      - ./infrastructure/security/webroot:/var/www/certbot
    command: certonly --webroot --webroot-path=/var/www/certbot --email admin@{self.base_domain} --agree-tos --no-eff-email --staging -d {self.base_domain} -d *.{self.base_domain}
    depends_on:
      - frontend

volumes:
  postgres_data:
  redis_data:

networks:
  ytempire_network:
    driver: bridge
"""

    def generate_pg_hba_config(self) -> str:
        """Generate PostgreSQL host-based authentication"""
        return """
# PostgreSQL Client Authentication Configuration File
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# Local connections
local   all             all                                     trust

# IPv4 local connections with SSL required
hostssl all             all             127.0.0.1/32            scram-sha-256
hostssl all             all             172.16.0.0/12           scram-sha-256
hostssl all             all             10.0.0.0/8              scram-sha-256
hostssl all             all             192.168.0.0/16          scram-sha-256

# IPv6 local connections with SSL required
hostssl all             all             ::1/128                 scram-sha-256

# Replication connections with SSL required
hostssl replication     replicator      127.0.0.1/32           scram-sha-256
hostssl replication     replicator      172.16.0.0/12          scram-sha-256
hostssl replication     replicator      10.0.0.0/8             scram-sha-256
hostssl replication     replicator      192.168.0.0/16         scram-sha-256

# Deny all non-SSL connections
host    all             all             0.0.0.0/0               reject
host    all             all             ::/0                    reject
"""

    def generate_application_tls_config(self) -> str:
        """Generate application-level TLS configuration"""
        return f"""
# FastAPI TLS Configuration
import ssl
import uvicorn
from fastapi import FastAPI

app = FastAPI()

# TLS Configuration for production
ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain(
    certfile="/app/certs/{self.base_domain}.crt",
    keyfile="/app/certs/{self.base_domain}.key"
)

# Security settings
ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!SHA1:!AESCCM')
ssl_context.options |= ssl.OP_NO_SSLv2
ssl_context.options |= ssl.OP_NO_SSLv3
ssl_context.options |= ssl.OP_NO_TLSv1
ssl_context.options |= ssl.OP_NO_TLSv1_1

# Database connection with TLS
DATABASE_URL = "postgresql://postgres:password@postgres:5432/ytempire?sslmode=require&sslcert=/app/certs/{self.base_domain}.crt&sslkey=/app/certs/{self.base_domain}.key&sslrootcert=/app/certs/ca.crt"

# Redis connection with TLS
REDIS_URL = "rediss://redis:6379/0?ssl_cert_reqs=required&ssl_ca_certs=/app/certs/ca.crt&ssl_certfile=/app/certs/{self.base_domain}.crt&ssl_keyfile=/app/certs/{self.base_domain}.key"

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="/app/certs/{self.base_domain}.key",
        ssl_certfile="/app/certs/{self.base_domain}.crt",
        ssl_version=ssl.PROTOCOL_TLSv1_2
    )
"""

    def generate_dhparam_file(self, output_path: str, key_size: int = 2048):
        """Generate DH parameters file"""
        subprocess.run([
            "openssl", "dhparam", "-out", output_path, str(key_size)
        ], check=True)
        os.chmod(output_path, 0o600)

    def setup_comprehensive_tls(self) -> Dict[str, str]:
        """Set up comprehensive TLS configuration for all services"""
        config_files = {}
        
        # Create configuration files
        configs = {
            'nginx-tls.conf': self.generate_nginx_tls_config(),
            'haproxy-tls.cfg': self.generate_haproxy_tls_config(),
            'postgres-tls.conf': self.generate_postgres_tls_config(),
            'redis-tls.conf': self.generate_redis_tls_config(),
            'docker-compose.tls.yml': self.generate_docker_compose_tls_config(),
            'pg_hba.conf': self.generate_pg_hba_config(),
            'app_tls_config.py': self.generate_application_tls_config()
        }
        
        security_dir = Path("infrastructure/security")
        security_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, content in configs.items():
            file_path = security_dir / filename
            with open(file_path, 'w') as f:
                f.write(content)
            config_files[filename] = str(file_path)
        
        # Generate DH parameters
        dhparam_path = security_dir / "certs" / "dhparam.pem"
        dhparam_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not dhparam_path.exists():
            print("Generating DH parameters (this may take a while)...")
            self.generate_dhparam_file(str(dhparam_path))
            config_files['dhparam.pem'] = str(dhparam_path)
        
        return config_files

def main():
    """CLI interface for TLS configuration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TLS Configuration Generator")
    parser.add_argument("--domain", default="ytempire.local", help="Base domain name")
    parser.add_argument("--setup", action="store_true", help="Set up comprehensive TLS configuration")
    
    args = parser.parse_args()
    
    configurator = TLSConfigurator(args.domain)
    
    if args.setup:
        print(f"Setting up TLS configuration for domain: {args.domain}")
        config_files = configurator.setup_comprehensive_tls()
        
        print("Generated TLS configuration files:")
        for filename, path in config_files.items():
            print(f"  {filename}: {path}")
        
        print(f"\nNext steps:")
        print(f"1. Generate certificates using: python encryption_manager.py generate-cert --domain {args.domain}")
        print(f"2. Update environment variables with certificate paths")
        print(f"3. Deploy using: docker-compose -f infrastructure/security/docker-compose.tls.yml up -d")
    else:
        print("TLS Configurator ready. Use --setup to generate configuration files.")

if __name__ == "__main__":
    main()