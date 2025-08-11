#!/bin/bash

# SSL/TLS Configuration Script for YTEmpire
# Sets up HTTPS with Let's Encrypt certificates

set -e

# Configuration
DOMAIN=${DOMAIN:-ytempire.com}
EMAIL=${EMAIL:-admin@ytempire.com}
NGINX_DIR="/etc/nginx"
CERT_DIR="/etc/letsencrypt"
APP_DIR="/app/ytempire"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================"
echo "YTEmpire SSL/TLS Setup"
echo "======================================${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root${NC}"
    exit 1
fi

# 1. Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
apt-get update
apt-get install -y certbot python3-certbot-nginx nginx openssl

# 2. Generate DH parameters for enhanced security
echo -e "${YELLOW}Generating DH parameters (this may take a while)...${NC}"
if [ ! -f "$NGINX_DIR/dhparam.pem" ]; then
    openssl dhparam -out "$NGINX_DIR/dhparam.pem" 2048
fi

# 3. Create SSL configuration snippet
echo -e "${YELLOW}Creating SSL configuration...${NC}"
cat > "$NGINX_DIR/snippets/ssl-params.conf" <<EOF
# SSL Configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_prefer_server_ciphers off;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;
ssl_session_timeout 1d;
ssl_session_cache shared:SSL:10m;
ssl_session_tickets off;
ssl_stapling on;
ssl_stapling_verify on;

# Security headers
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
add_header X-Frame-Options DENY always;
add_header X-Content-Type-Options nosniff always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self' https:; script-src 'self' 'unsafe-inline' 'unsafe-eval' https:; style-src 'self' 'unsafe-inline' https:; img-src 'self' data: https:; font-src 'self' data: https:; connect-src 'self' https: wss:; media-src 'self' https:; object-src 'none'; frame-src 'self' https:; base-uri 'self'; form-action 'self' https:; frame-ancestors 'none';" always;

# DH parameters
ssl_dhparam $NGINX_DIR/dhparam.pem;

# DNS resolver
resolver 8.8.8.8 8.8.4.4 valid=300s;
resolver_timeout 5s;
EOF

# 4. Create initial Nginx configuration
echo -e "${YELLOW}Creating Nginx configuration...${NC}"
cat > "$NGINX_DIR/sites-available/ytempire" <<EOF
# HTTP server - redirect to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN www.$DOMAIN;
    
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }
    
    location / {
        return 301 https://\$server_name\$request_uri;
    }
}

# HTTPS server
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name $DOMAIN www.$DOMAIN;
    
    # SSL certificates (will be updated by certbot)
    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;
    
    # Include SSL parameters
    include snippets/ssl-params.conf;
    
    # Logging
    access_log /var/log/nginx/ytempire_access.log;
    error_log /var/log/nginx/ytempire_error.log;
    
    # Client body size for file uploads
    client_max_body_size 100M;
    
    # Timeouts
    proxy_connect_timeout 600;
    proxy_send_timeout 600;
    proxy_read_timeout 600;
    send_timeout 600;
    
    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        proxy_buffering off;
    }
    
    # WebSocket support
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
    }
    
    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
    }
    
    # Static files
    location /static {
        alias $APP_DIR/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # Media files
    location /media {
        alias $APP_DIR/media;
        expires 7d;
        add_header Cache-Control "public";
    }
    
    # Monitoring endpoints
    location /metrics {
        proxy_pass http://localhost:8000/metrics;
        allow 127.0.0.1;
        deny all;
    }
    
    location /health {
        proxy_pass http://localhost:8000/health;
        access_log off;
    }
}
EOF

# 5. Enable the site
ln -sf "$NGINX_DIR/sites-available/ytempire" "$NGINX_DIR/sites-enabled/"

# 6. Test Nginx configuration
echo -e "${YELLOW}Testing Nginx configuration...${NC}"
nginx -t

# 7. Create certbot webroot directory
mkdir -p /var/www/certbot

# 8. Restart Nginx
systemctl restart nginx

# 9. Obtain SSL certificate
echo -e "${YELLOW}Obtaining SSL certificate from Let's Encrypt...${NC}"
certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    --force-renewal \
    -d $DOMAIN \
    -d www.$DOMAIN

# 10. Update Nginx configuration with correct certificate paths
sed -i "s|/etc/letsencrypt/live/$DOMAIN/|$CERT_DIR/live/$DOMAIN/|g" "$NGINX_DIR/sites-available/ytempire"

# 11. Reload Nginx with new certificates
systemctl reload nginx

# 12. Set up automatic renewal
echo -e "${YELLOW}Setting up automatic certificate renewal...${NC}"
cat > /etc/systemd/system/certbot-renewal.service <<EOF
[Unit]
Description=Certbot Renewal
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/bin/certbot renew --quiet --post-hook "systemctl reload nginx"
EOF

cat > /etc/systemd/system/certbot-renewal.timer <<EOF
[Unit]
Description=Run certbot renewal twice daily
After=network.target

[Timer]
OnCalendar=*-*-* 00,12:00:00
RandomizedDelaySec=3600
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Enable and start the timer
systemctl daemon-reload
systemctl enable certbot-renewal.timer
systemctl start certbot-renewal.timer

# 13. Create monitoring script
cat > /usr/local/bin/check-ssl-cert.sh <<'EOF'
#!/bin/bash
DOMAIN="$1"
DAYS_WARNING=30

expiry_date=$(echo | openssl s_client -servername $DOMAIN -connect $DOMAIN:443 2>/dev/null | openssl x509 -noout -dates | grep notAfter | cut -d= -f2)
expiry_epoch=$(date -d "$expiry_date" +%s)
current_epoch=$(date +%s)
days_left=$(( ($expiry_epoch - $current_epoch) / 86400 ))

if [ $days_left -lt $DAYS_WARNING ]; then
    echo "WARNING: SSL certificate for $DOMAIN expires in $days_left days"
    # Send alert (implement your alerting mechanism here)
else
    echo "SSL certificate for $DOMAIN is valid for $days_left more days"
fi
EOF

chmod +x /usr/local/bin/check-ssl-cert.sh

# 14. Add SSL monitoring to cron
echo "0 9 * * * /usr/local/bin/check-ssl-cert.sh $DOMAIN" | crontab -l | { cat; echo "0 9 * * * /usr/local/bin/check-ssl-cert.sh $DOMAIN"; } | crontab -

# 15. Security scan
echo -e "${YELLOW}Running SSL/TLS security scan...${NC}"
# Test SSL configuration
echo | openssl s_client -connect $DOMAIN:443 -servername $DOMAIN -tls1_2 2>/dev/null | grep "Cipher"
echo | openssl s_client -connect $DOMAIN:443 -servername $DOMAIN -tls1_3 2>/dev/null | grep "Cipher"

echo -e "${GREEN}======================================"
echo "SSL/TLS Setup Complete!"
echo "======================================"
echo "Domain: https://$DOMAIN"
echo "Certificate: $CERT_DIR/live/$DOMAIN/"
echo "Renewal: Automatic (twice daily)"
echo "Security Headers: Enabled"
echo "TLS Versions: 1.2, 1.3"
echo "======================================${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Test your SSL configuration at: https://www.ssllabs.com/ssltest/"
echo "2. Update your application configuration to use HTTPS"
echo "3. Monitor certificate expiry with: /usr/local/bin/check-ssl-cert.sh $DOMAIN"