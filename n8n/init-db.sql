-- N8N Database Initialization Script
-- Owner: Integration Specialist

-- Create N8N schema
CREATE SCHEMA IF NOT EXISTS n8n;

-- Set default schema
SET search_path TO n8n;

-- Grant permissions to n8n user
GRANT ALL PRIVILEGES ON SCHEMA n8n TO ytempire;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA n8n TO ytempire;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA n8n TO ytempire;

-- Create additional permissions for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA n8n GRANT ALL ON TABLES TO ytempire;
ALTER DEFAULT PRIVILEGES IN SCHEMA n8n GRANT ALL ON SEQUENCES TO ytempire;

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Log the initialization
INSERT INTO public.initialization_log (service, initialized_at) 
VALUES ('n8n', NOW()) 
ON CONFLICT (service) DO UPDATE SET initialized_at = NOW();