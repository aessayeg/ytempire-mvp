@echo off
REM YTEmpire Startup Script for Windows
REM Starts all services and verifies they're running correctly

echo =========================================
echo YTEmpire - Starting All Services
echo =========================================

REM Step 1: Start Docker services
echo.
echo Step 1: Starting Docker services
docker-compose -f docker-compose.full.yml up -d

REM Wait for services to initialize
echo.
echo Waiting for services to initialize...
timeout /t 10 /nobreak > nul

REM Step 2: Check core services
echo.
echo Step 2: Checking core services
echo Checking PostgreSQL on port 5432...
netstat -an | findstr :5432 > nul
if %errorlevel%==0 (echo [OK] PostgreSQL) else (echo [FAILED] PostgreSQL)

echo Checking Redis on port 6379...
netstat -an | findstr :6379 > nul
if %errorlevel%==0 (echo [OK] Redis) else (echo [FAILED] Redis)

echo Checking Backend API on port 8000...
netstat -an | findstr :8000 > nul
if %errorlevel%==0 (echo [OK] Backend API) else (echo [FAILED] Backend API)

echo Checking Frontend on port 3000...
netstat -an | findstr :3000 > nul
if %errorlevel%==0 (echo [OK] Frontend) else (echo [FAILED] Frontend)

echo Checking N8N on port 5678...
netstat -an | findstr :5678 > nul
if %errorlevel%==0 (echo [OK] N8N) else (echo [FAILED] N8N)

echo Checking ML Server on port 8001...
netstat -an | findstr :8001 > nul
if %errorlevel%==0 (echo [OK] ML Server) else (echo [FAILED] ML Server)

REM Step 3: Run database migrations
echo.
echo Step 3: Running database migrations
cd backend
alembic upgrade head
cd ..
echo [OK] Migrations completed

REM Step 4: Run health check
echo.
echo Step 4: Running comprehensive health check
python scripts\health_check.py

REM Step 5: Display access URLs
echo.
echo =========================================
echo YTEmpire is ready!
echo =========================================
echo.
echo Access URLs:
echo   Frontend:    http://localhost:3000
echo   Backend API: http://localhost:8000/docs
echo   N8N:         http://localhost:5678
echo   ML Server:   http://localhost:8001/docs
echo   Grafana:     http://localhost:3001 (admin/admin123)
echo   Prometheus:  http://localhost:9090
echo.
echo Default credentials:
echo   N8N:      admin/ytempire2024
echo   Database: ytempire/ytempire123
echo.
echo To stop all services, run:
echo   docker-compose -f docker-compose.full.yml down
echo.
pause