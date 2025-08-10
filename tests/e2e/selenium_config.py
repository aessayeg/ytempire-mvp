"""
Selenium Configuration for E2E Testing
"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import os
import pytest
from typing import Optional


class SeleniumConfig:
    """Selenium WebDriver configuration"""
    
    DEFAULT_TIMEOUT = 10
    DEFAULT_POLL_FREQUENCY = 0.5
    
    @staticmethod
    def get_chrome_options() -> Options:
        """Get Chrome options for testing"""
        options = Options()
        
        # Headless mode for CI/CD
        if os.environ.get('CI'):
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
        
        # Common options
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--start-maximized')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-popup-blocking')
        options.add_argument('--disable-blink-features=AutomationControlled')
        
        # Performance options
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        options.add_experimental_option('useAutomationExtension', False)
        prefs = {
            'profile.default_content_setting_values.notifications': 2,
            'profile.default_content_settings.popups': 0
        }
        options.add_experimental_option('prefs', prefs)
        
        return options
    
    @staticmethod
    def get_firefox_options():
        """Get Firefox options for testing"""
        from selenium.webdriver.firefox.options import Options
        options = Options()
        
        if os.environ.get('CI'):
            options.add_argument('--headless')
        
        options.add_argument('--width=1920')
        options.add_argument('--height=1080')
        
        return options
    
    @staticmethod
    def create_driver(browser: str = 'chrome') -> webdriver:
        """Create WebDriver instance"""
        if browser.lower() == 'chrome':
            options = SeleniumConfig.get_chrome_options()
            driver = webdriver.Chrome(options=options)
        elif browser.lower() == 'firefox':
            options = SeleniumConfig.get_firefox_options()
            driver = webdriver.Firefox(options=options)
        else:
            raise ValueError(f"Unsupported browser: {browser}")
        
        driver.implicitly_wait(SeleniumConfig.DEFAULT_TIMEOUT)
        return driver


class BasePage:
    """Base page object for all pages"""
    
    def __init__(self, driver: webdriver):
        self.driver = driver
        self.wait = WebDriverWait(driver, SeleniumConfig.DEFAULT_TIMEOUT)
        self.base_url = os.environ.get('BASE_URL', 'http://localhost:3000')
    
    def go_to(self, path: str = ''):
        """Navigate to page"""
        url = f"{self.base_url}{path}"
        self.driver.get(url)
    
    def find_element(self, by: By, value: str):
        """Find element with wait"""
        return self.wait.until(
            EC.presence_of_element_located((by, value))
        )
    
    def find_elements(self, by: By, value: str):
        """Find multiple elements"""
        return self.wait.until(
            EC.presence_of_all_elements_located((by, value))
        )
    
    def click_element(self, by: By, value: str):
        """Click element"""
        element = self.wait.until(
            EC.element_to_be_clickable((by, value))
        )
        element.click()
    
    def input_text(self, by: By, value: str, text: str):
        """Input text into element"""
        element = self.find_element(by, value)
        element.clear()
        element.send_keys(text)
    
    def get_text(self, by: By, value: str) -> str:
        """Get element text"""
        element = self.find_element(by, value)
        return element.text
    
    def is_element_visible(self, by: By, value: str) -> bool:
        """Check if element is visible"""
        try:
            self.wait.until(
                EC.visibility_of_element_located((by, value))
            )
            return True
        except TimeoutException:
            return False
    
    def wait_for_url_contains(self, url_part: str):
        """Wait for URL to contain specific text"""
        self.wait.until(EC.url_contains(url_part))
    
    def take_screenshot(self, filename: str):
        """Take screenshot"""
        self.driver.save_screenshot(filename)


class LoginPage(BasePage):
    """Login page object"""
    
    # Locators
    EMAIL_INPUT = (By.ID, 'email')
    PASSWORD_INPUT = (By.ID, 'password')
    LOGIN_BUTTON = (By.XPATH, '//button[@type="submit"]')
    ERROR_MESSAGE = (By.CLASS_NAME, 'error-message')
    
    def login(self, email: str, password: str):
        """Perform login"""
        self.input_text(*self.EMAIL_INPUT, email)
        self.input_text(*self.PASSWORD_INPUT, password)
        self.click_element(*self.LOGIN_BUTTON)
    
    def get_error_message(self) -> Optional[str]:
        """Get error message if present"""
        if self.is_element_visible(*self.ERROR_MESSAGE):
            return self.get_text(*self.ERROR_MESSAGE)
        return None


class DashboardPage(BasePage):
    """Dashboard page object"""
    
    # Locators
    DASHBOARD_TITLE = (By.TAG_NAME, 'h1')
    VIDEO_COUNT = (By.XPATH, '//div[@data-testid="video-count"]')
    CHANNEL_COUNT = (By.XPATH, '//div[@data-testid="channel-count"]')
    CREATE_VIDEO_BUTTON = (By.XPATH, '//button[contains(text(), "Create Video")]')
    
    def get_video_count(self) -> int:
        """Get video count from dashboard"""
        text = self.get_text(*self.VIDEO_COUNT)
        return int(''.join(filter(str.isdigit, text)))
    
    def get_channel_count(self) -> int:
        """Get channel count from dashboard"""
        text = self.get_text(*self.CHANNEL_COUNT)
        return int(''.join(filter(str.isdigit, text)))
    
    def click_create_video(self):
        """Click create video button"""
        self.click_element(*self.CREATE_VIDEO_BUTTON)


class VideoQueuePage(BasePage):
    """Video queue page object"""
    
    # Locators
    VIDEO_LIST = (By.CLASS_NAME, 'video-list')
    VIDEO_ITEMS = (By.CLASS_NAME, 'video-item')
    FILTER_DROPDOWN = (By.ID, 'status-filter')
    PROCESS_ALL_BUTTON = (By.XPATH, '//button[contains(text(), "Process All")]')
    
    def get_video_count(self) -> int:
        """Get number of videos in queue"""
        videos = self.find_elements(*self.VIDEO_ITEMS)
        return len(videos)
    
    def filter_by_status(self, status: str):
        """Filter videos by status"""
        from selenium.webdriver.support.ui import Select
        dropdown = self.find_element(*self.FILTER_DROPDOWN)
        select = Select(dropdown)
        select.select_by_value(status)
    
    def click_process_all(self):
        """Click process all button"""
        self.click_element(*self.PROCESS_ALL_BUTTON)


# Pytest fixtures
@pytest.fixture(scope='session')
def browser_name():
    """Get browser name from environment or default"""
    return os.environ.get('BROWSER', 'chrome')


@pytest.fixture(scope='function')
def driver(browser_name):
    """Create WebDriver for test"""
    driver = SeleniumConfig.create_driver(browser_name)
    yield driver
    driver.quit()


@pytest.fixture
def login_page(driver):
    """Create login page object"""
    return LoginPage(driver)


@pytest.fixture
def dashboard_page(driver):
    """Create dashboard page object"""
    return DashboardPage(driver)


@pytest.fixture
def video_queue_page(driver):
    """Create video queue page object"""
    return VideoQueuePage(driver)