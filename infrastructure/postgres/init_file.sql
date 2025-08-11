-- YTEmpire PostgreSQL initialization
-- Set password for ytempire user
ALTER USER ytempire WITH PASSWORD 'admin';

-- Create additional databases
CREATE DATABASE IF NOT EXISTS n8n;
CREATE DATABASE IF NOT EXISTS ytempire_test;
CREATE DATABASE IF NOT EXISTS ytempire_analytics;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE n8n TO ytempire;
GRANT ALL PRIVILEGES ON DATABASE ytempire_test TO ytempire;
GRANT ALL PRIVILEGES ON DATABASE ytempire_analytics TO ytempire;