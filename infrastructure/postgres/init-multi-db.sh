#!/bin/bash
set -e

# Create multiple databases for YTEmpire services
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE n8n;
    GRANT ALL PRIVILEGES ON DATABASE n8n TO $POSTGRES_USER;
    
    CREATE DATABASE ytempire_test;
    GRANT ALL PRIVILEGES ON DATABASE ytempire_test TO $POSTGRES_USER;
    
    CREATE DATABASE ytempire_analytics;
    GRANT ALL PRIVILEGES ON DATABASE ytempire_analytics TO $POSTGRES_USER;
EOSQL

echo "Multiple databases created successfully"