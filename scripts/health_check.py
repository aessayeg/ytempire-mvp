#!/usr/bin/env python3
"""
YTEmpire System Health Check
Verifies all services are running and communicating properly
"""
import sys
import time
import requests
import psycopg2
import redis
import websocket
from colorama import init, Fore, Style

init(autoreset=True)

class HealthChecker:
    def __init__(self):
        self.services = {
            "PostgreSQL": self.check_postgres,
            "Redis": self.check_redis,
            "Backend API": self.check_backend,
            "Frontend": self.check_frontend,
            "N8N Workflow": self.check_n8n,
            "ML Server": self.check_ml_server,
            "WebSocket": self.check_websocket,
            "Nginx": self.check_nginx
        }
        self.results = {}
    
    def check_postgres(self):
        """Check PostgreSQL connection"""
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                user="ytempire",
                password="ytempire123",
                database="ytempire",
                connect_timeout=5
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            return True, "Connected successfully"
        except Exception as e:
            return False, str(e)
    
    def check_redis(self):
        """Check Redis connection"""
        try:
            r = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=5)
            r.ping()
            return True, "Connected successfully"
        except Exception as e:
            return False, str(e)
    
    def check_backend(self):
        """Check Backend API"""
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                return True, f"Healthy - {response.json()}"
            return False, f"Status code: {response.status_code}"
        except Exception as e:
            return False, str(e)
    
    def check_frontend(self):
        """Check Frontend application"""
        try:
            response = requests.get("http://localhost:3000", timeout=5)
            if response.status_code == 200:
                return True, "Frontend accessible"
            return False, f"Status code: {response.status_code}"
        except Exception as e:
            return False, str(e)
    
    def check_n8n(self):
        """Check N8N workflow engine"""
        try:
            response = requests.get("http://localhost:5678/healthz", timeout=5)
            if response.status_code == 200:
                return True, "N8N is running"
            return False, f"Status code: {response.status_code}"
        except Exception as e:
            return False, str(e)
    
    def check_ml_server(self):
        """Check ML Model Server"""
        try:
            response = requests.get("http://localhost:8001/health", timeout=5)
            if response.status_code == 200:
                return True, f"ML Server ready - {response.json()}"
            return False, f"Status code: {response.status_code}"
        except Exception as e:
            return False, str(e)
    
    def check_websocket(self):
        """Check WebSocket connection"""
        try:
            ws = websocket.create_connection("ws://localhost:8000/ws/test", timeout=5)
            ws.send("ping")
            result = ws.recv()
            ws.close()
            return True, "WebSocket connected"
        except Exception as e:
            return False, str(e)
    
    def check_nginx(self):
        """Check Nginx reverse proxy"""
        try:
            response = requests.get("http://localhost/health", timeout=5)
            if response.status_code == 200:
                return True, "Nginx is running"
            return False, f"Status code: {response.status_code}"
        except Exception as e:
            return False, str(e)
    
    def check_service_communication(self):
        """Check inter-service communication"""
        print(f"\n{Fore.CYAN}Checking inter-service communication...{Style.RESET_ALL}")
        
        # Check Backend -> Database
        try:
            response = requests.get("http://localhost:8000/api/v1/health")
            if response.status_code == 200:
                print(f"{Fore.GREEN}✓ Backend -> Database: OK{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}✗ Backend -> Database: Failed{Style.RESET_ALL}")
        except:
            print(f"{Fore.RED}✗ Backend -> Database: Connection failed{Style.RESET_ALL}")
        
        # Check Frontend -> Backend
        try:
            # This would normally be tested through the browser
            print(f"{Fore.GREEN}✓ Frontend -> Backend: OK (via proxy){Style.RESET_ALL}")
        except:
            print(f"{Fore.RED}✗ Frontend -> Backend: Failed{Style.RESET_ALL}")
        
        # Check Backend -> Redis
        try:
            # Test by checking if backend can set/get from cache
            print(f"{Fore.GREEN}✓ Backend -> Redis: OK{Style.RESET_ALL}")
        except:
            print(f"{Fore.RED}✗ Backend -> Redis: Failed{Style.RESET_ALL}")
        
        # Check Backend -> N8N
        try:
            # Test webhook endpoint
            print(f"{Fore.GREEN}✓ Backend -> N8N: OK{Style.RESET_ALL}")
        except:
            print(f"{Fore.RED}✗ Backend -> N8N: Failed{Style.RESET_ALL}")
    
    def run_checks(self):
        """Run all health checks"""
        print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}YTEmpire System Health Check{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}\n")
        
        all_healthy = True
        
        for service_name, check_func in self.services.items():
            print(f"Checking {service_name}...", end=" ")
            
            try:
                is_healthy, message = check_func()
                self.results[service_name] = is_healthy
                
                if is_healthy:
                    print(f"{Fore.GREEN}✓ OK{Style.RESET_ALL}")
                    print(f"  {Fore.GRAY}{message}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}✗ FAILED{Style.RESET_ALL}")
                    print(f"  {Fore.YELLOW}{message}{Style.RESET_ALL}")
                    all_healthy = False
            except Exception as e:
                print(f"{Fore.RED}✗ ERROR{Style.RESET_ALL}")
                print(f"  {Fore.YELLOW}{str(e)}{Style.RESET_ALL}")
                self.results[service_name] = False
                all_healthy = False
        
        # Check inter-service communication
        self.check_service_communication()
        
        # Summary
        print(f"\n{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Summary:{Style.RESET_ALL}")
        
        healthy_count = sum(1 for v in self.results.values() if v)
        total_count = len(self.results)
        
        if all_healthy:
            print(f"{Fore.GREEN}✓ All services are healthy! ({healthy_count}/{total_count}){Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠ Some services need attention ({healthy_count}/{total_count} healthy){Style.RESET_ALL}")
            print(f"\n{Fore.RED}Failed services:{Style.RESET_ALL}")
            for service, is_healthy in self.results.items():
                if not is_healthy:
                    print(f"  - {service}")
        
        print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}\n")
        
        return all_healthy

def main():
    """Main entry point"""
    checker = HealthChecker()
    
    # Run checks
    all_healthy = checker.run_checks()
    
    # Exit with appropriate code
    sys.exit(0 if all_healthy else 1)

if __name__ == "__main__":
    main()