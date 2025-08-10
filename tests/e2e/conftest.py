"""
Pytest configuration for E2E tests
"""
import pytest
import subprocess
import time
import requests
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

@pytest.fixture(scope="session")
def docker_services():
    """Start all required Docker services for E2E tests"""
    print("Starting Docker services...")
    
    # Start services
    subprocess.run(["docker-compose", "-f", "docker-compose.test.yml", "up", "-d"], check=True)
    
    # Wait for services to be ready
    time.sleep(10)
    
    # Check if services are healthy
    max_retries = 30
    for i in range(max_retries):
        try:
            # Check backend health
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                print("Backend is ready")
                break
        except:
            pass
        time.sleep(2)
    
    yield
    
    # Cleanup
    print("Stopping Docker services...")
    subprocess.run(["docker-compose", "-f", "docker-compose.test.yml", "down", "-v"], check=True)

@pytest.fixture(scope="session")
def test_database():
    """Create test database and apply migrations"""
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="ytempire",
        password="ytempire123",
        database="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Create test database
    cursor.execute("DROP DATABASE IF EXISTS ytempire_test")
    cursor.execute("CREATE DATABASE ytempire_test")
    
    cursor.close()
    conn.close()
    
    # Run migrations
    subprocess.run(
        ["alembic", "upgrade", "head"],
        env={"DATABASE_URL": "postgresql://ytempire:ytempire123@localhost:5432/ytempire_test"},
        cwd="backend",
        check=True
    )
    
    yield
    
    # Cleanup is handled by docker-compose down

@pytest.fixture(scope="function")
def test_user(test_database):
    """Create a test user for authentication"""
    import requests
    
    user_data = {
        "email": "test@ytempire.com",
        "username": "testuser",
        "password": "TestPassword123!",
        "full_name": "Test User"
    }
    
    # Register user
    response = requests.post(
        "http://localhost:8000/api/v1/auth/register",
        json=user_data
    )
    
    if response.status_code == 201:
        return user_data
    
    # User might already exist, just return credentials
    return user_data

@pytest.fixture
def auth_headers(test_user):
    """Get authentication headers for API requests"""
    import requests
    
    response = requests.post(
        "http://localhost:8000/api/v1/auth/login",
        json={
            "email": test_user["email"],
            "password": test_user["password"]
        }
    )
    
    if response.status_code == 200:
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    return {}

# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )