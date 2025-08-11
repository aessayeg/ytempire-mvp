-- Create replication user for PostgreSQL streaming replication
-- This script runs during database initialization

DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'replicator') THEN

      CREATE ROLE replicator WITH REPLICATION LOGIN PASSWORD 'repl_pass';
   END IF;
END
$do$;

-- Grant necessary permissions
GRANT CONNECT ON DATABASE ytempire TO replicator;

-- Create archive directory if it doesn't exist
\! mkdir -p /var/lib/postgresql/archive

-- Set up basic tables for backup metadata (if they don't exist)
CREATE TABLE IF NOT EXISTS backup_history (
    id SERIAL PRIMARY KEY,
    backup_id VARCHAR(255) UNIQUE NOT NULL,
    backup_type VARCHAR(50) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) NOT NULL,
    file_path TEXT,
    file_size BIGINT,
    checksum VARCHAR(64),
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_backup_history_backup_id ON backup_history(backup_id);
CREATE INDEX IF NOT EXISTS idx_backup_history_status ON backup_history(status);
CREATE INDEX IF NOT EXISTS idx_backup_history_start_time ON backup_history(start_time);

-- Grant permissions on backup tables
GRANT SELECT, INSERT, UPDATE ON backup_history TO postgres;
GRANT USAGE ON SEQUENCE backup_history_id_seq TO postgres;