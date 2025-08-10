"""
End-to-End Authentication Flow Tests
"""
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time

class TestAuthenticationFlow:
    """Test complete authentication flow"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test browser"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(10)
        self.base_url = "http://localhost:5173"
        self.api_url = "http://localhost:8000"
        
        yield
        
        self.driver.quit()
    
    def test_user_registration(self):
        """Test new user registration flow"""
        # Navigate to registration page
        self.driver.get(f"{self.base_url}/register")
        
        # Fill registration form
        email_input = self.driver.find_element(By.NAME, "email")
        email_input.send_keys("testuser@example.com")
        
        username_input = self.driver.find_element(By.NAME, "username")
        username_input.send_keys("testuser")
        
        password_input = self.driver.find_element(By.NAME, "password")
        password_input.send_keys("TestPassword123!")
        
        confirm_password_input = self.driver.find_element(By.NAME, "confirmPassword")
        confirm_password_input.send_keys("TestPassword123!")
        
        # Submit form
        submit_button = self.driver.find_element(By.TYPE, "submit")
        submit_button.click()
        
        # Wait for redirect to login
        WebDriverWait(self.driver, 10).until(
            EC.url_contains("/login")
        )
        
        assert "/login" in self.driver.current_url
    
    def test_user_login(self):
        """Test user login flow"""
        # Navigate to login page
        self.driver.get(f"{self.base_url}/login")
        
        # Fill login form
        email_input = self.driver.find_element(By.NAME, "email")
        email_input.send_keys("testuser@example.com")
        
        password_input = self.driver.find_element(By.NAME, "password")
        password_input.send_keys("TestPassword123!")
        
        # Submit form
        submit_button = self.driver.find_element(By.TYPE, "submit")
        submit_button.click()
        
        # Wait for redirect to dashboard
        WebDriverWait(self.driver, 10).until(
            EC.url_contains("/dashboard")
        )
        
        # Verify dashboard loaded
        assert "/dashboard" in self.driver.current_url
        
        # Check for user menu
        user_menu = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "user-menu"))
        )
        assert user_menu is not None
    
    def test_protected_route_redirect(self):
        """Test that protected routes redirect to login when not authenticated"""
        # Try to access dashboard without login
        self.driver.get(f"{self.base_url}/dashboard")
        
        # Should redirect to login
        WebDriverWait(self.driver, 10).until(
            EC.url_contains("/login")
        )
        
        assert "/login" in self.driver.current_url
    
    def test_logout(self):
        """Test logout flow"""
        # First login
        self.test_user_login()
        
        # Find and click logout button
        user_menu = self.driver.find_element(By.CLASS_NAME, "user-menu")
        user_menu.click()
        
        logout_button = self.driver.find_element(By.TEXT, "Logout")
        logout_button.click()
        
        # Should redirect to login
        WebDriverWait(self.driver, 10).until(
            EC.url_contains("/login")
        )
        
        assert "/login" in self.driver.current_url
        
        # Try to access dashboard again
        self.driver.get(f"{self.base_url}/dashboard")
        
        # Should still be redirected to login
        assert "/login" in self.driver.current_url