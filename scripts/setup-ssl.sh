#!/bin/bash
# SSL Certificate Setup Script
# Owner: DevOps Engineer

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOMAIN=${1:-ytempire.local}
SSL_DIR="./nginx/ssl"
CERT_PATH="$SSL_DIR/$DOMAIN.crt"
KEY_PATH="$SSL_DIR/$DOMAIN.key"
CSR_PATH="$SSL_DIR/$DOMAIN.csr"

echo -e "${BLUE}YTEmpire SSL Certificate Setup${NC}"
echo "=============================="

# Create SSL directory
echo -e "${YELLOW}Creating SSL directory...${NC}"
mkdir -p "$SSL_DIR"

# Function to generate self-signed certificate
generate_self_signed() {
    echo -e "${YELLOW}Generating self-signed certificate for $DOMAIN...${NC}"
    
    # Generate private key
    openssl genrsa -out "$KEY_PATH" 2048
    
    # Generate certificate signing request
    openssl req -new -key "$KEY_PATH" -out "$CSR_PATH" -subj "/C=US/ST=CA/L=San Francisco/O=YTEmpire/OU=IT/CN=$DOMAIN"
    
    # Generate self-signed certificate
    openssl x509 -req -in "$CSR_PATH" -signkey "$KEY_PATH" -out "$CERT_PATH" -days 365
    
    # Set proper permissions
    chmod 600 "$KEY_PATH"
    chmod 644 "$CERT_PATH"
    
    # Clean up CSR
    rm "$CSR_PATH"
    
    echo -e "${GREEN}Self-signed certificate generated successfully!${NC}"
    echo "Certificate: $CERT_PATH"
    echo "Private Key: $KEY_PATH"
    echo -e "${YELLOW}Note: This is a self-signed certificate. Browsers will show security warnings.${NC}"
}

# Function to setup Let's Encrypt certificate
setup_letsencrypt() {
    echo -e "${YELLOW}Setting up Let's Encrypt certificate for $DOMAIN...${NC}"
    
    # Check if certbot is installed
    if ! command -v certbot &> /dev/null; then
        echo -e "${RED}Error: certbot is not installed. Please install certbot first.${NC}"
        echo "Ubuntu/Debian: sudo apt-get install certbot"
        echo "CentOS/RHEL: sudo yum install certbot"
        echo "macOS: brew install certbot"
        exit 1
    fi
    
    # Stop nginx if running
    if docker ps | grep -q ytempire_nginx; then
        echo -e "${YELLOW}Stopping nginx container...${NC}"
        docker stop ytempire_nginx || true
    fi
    
    # Generate certificate
    echo -e "${YELLOW}Requesting certificate from Let's Encrypt...${NC}"
    certbot certonly --standalone \
        --email admin@$DOMAIN \
        --agree-tos \
        --no-eff-email \
        -d $DOMAIN
    
    # Copy certificates to our SSL directory
    sudo cp "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" "$CERT_PATH"
    sudo cp "/etc/letsencrypt/live/$DOMAIN/privkey.pem" "$KEY_PATH"
    
    # Set proper ownership and permissions
    sudo chown $USER:$USER "$CERT_PATH" "$KEY_PATH"
    chmod 644 "$CERT_PATH"
    chmod 600 "$KEY_PATH"
    
    echo -e "${GREEN}Let's Encrypt certificate generated successfully!${NC}"
    echo "Certificate: $CERT_PATH"
    echo "Private Key: $KEY_PATH"
}

# Function to setup certificate renewal
setup_renewal() {
    echo -e "${YELLOW}Setting up automatic certificate renewal...${NC}"
    
    # Create renewal script
    cat > ./scripts/renew-ssl.sh << 'EOF'
#!/bin/bash
# SSL Certificate Renewal Script

DOMAIN=${1:-ytempire.local}
SSL_DIR="./nginx/ssl"

# Stop nginx
docker stop ytempire_nginx || true

# Renew certificate
certbot renew

# Copy renewed certificates
sudo cp "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" "$SSL_DIR/$DOMAIN.crt"
sudo cp "/etc/letsencrypt/live/$DOMAIN/privkey.pem" "$SSL_DIR/$DOMAIN.key"

# Set permissions
sudo chown $USER:$USER "$SSL_DIR/$DOMAIN.crt" "$SSL_DIR/$DOMAIN.key"
chmod 644 "$SSL_DIR/$DOMAIN.crt"
chmod 600 "$SSL_DIR/$DOMAIN.key"

# Restart nginx
docker start ytempire_nginx

echo "SSL certificate renewed successfully!"
EOF
    
    chmod +x ./scripts/renew-ssl.sh
    
    # Add to crontab (optional)
    echo -e "${BLUE}To setup automatic renewal, add this to your crontab (crontab -e):${NC}"
    echo "0 3 * * 1 /path/to/ytempire/scripts/renew-ssl.sh $DOMAIN"
}

# Main menu
echo -e "${BLUE}Choose SSL certificate type:${NC}"
echo "1) Self-signed certificate (for development)"
echo "2) Let's Encrypt certificate (for production)"
echo "3) Use existing certificate files"
echo ""

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        generate_self_signed
        ;;
    2)
        if [[ "$DOMAIN" == "ytempire.local" ]]; then
            echo -e "${RED}Error: Let's Encrypt requires a valid domain name.${NC}"
            echo "Please run: $0 yourdomain.com"
            exit 1
        fi
        setup_letsencrypt
        setup_renewal
        ;;
    3)
        echo -e "${YELLOW}Please place your certificate files in:${NC}"
        echo "Certificate: $CERT_PATH"
        echo "Private Key: $KEY_PATH"
        echo -e "${BLUE}Make sure the private key has 600 permissions and certificate has 644 permissions.${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

# Verify certificate files
if [[ -f "$CERT_PATH" && -f "$KEY_PATH" ]]; then
    echo -e "${GREEN}SSL setup completed successfully!${NC}"
    echo ""
    echo "Certificate information:"
    openssl x509 -in "$CERT_PATH" -text -noout | grep -E "Subject:|Issuer:|Not Before:|Not After:"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Update your DNS to point $DOMAIN to this server"
    echo "2. Start the application with: docker-compose -f docker-compose.prod.yml up -d"
    echo "3. Access your application at: https://$DOMAIN"
else
    echo -e "${RED}Error: Certificate files not found. SSL setup failed.${NC}"
    exit 1
fi