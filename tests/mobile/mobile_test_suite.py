"""
Mobile Testing Suite for YTEmpire
P1 Task: [OPS] Mobile Testing Implementation
Tests responsive design and mobile functionality
"""

import pytest
import asyncio
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from appium import webdriver as appium_driver
import time
import json
from typing import Dict, List, Any

# Mobile device configurations
MOBILE_DEVICES = {
    "iPhone_14_Pro": {
        "width": 393,
        "height": 852,
        "pixelRatio": 3,
        "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15"
    },
    "iPhone_SE": {
        "width": 375,
        "height": 667,
        "pixelRatio": 2,
        "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15"
    },
    "Samsung_Galaxy_S23": {
        "width": 360,
        "height": 780,
        "pixelRatio": 3,
        "userAgent": "Mozilla/5.0 (Linux; Android 13; SM-S918B) AppleWebKit/537.36"
    },
    "iPad_Pro": {
        "width": 1024,
        "height": 1366,
        "pixelRatio": 2,
        "userAgent": "Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) AppleWebKit/605.1.15"
    },
    "Pixel_7": {
        "width": 412,
        "height": 915,
        "pixelRatio": 2.625,
        "userAgent": "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36"
    }
}

class MobileTestBase:
    """Base class for mobile testing"""
    
    def setup_mobile_driver(self, device_name: str):
        """Setup Chrome driver with mobile emulation"""
        chrome_options = Options()
        device_config = MOBILE_DEVICES[device_name]
        
        mobile_emulation = {
            "deviceMetrics": {
                "width": device_config["width"],
                "height": device_config["height"],
                "pixelRatio": device_config["pixelRatio"]
            },
            "userAgent": device_config["userAgent"]
        }
        
        chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        
        return webdriver.Chrome(options=chrome_options)
    
    def check_responsive_breakpoint(self, driver, width: int):
        """Check if responsive breakpoint is working"""
        driver.set_window_size(width, 800)
        time.sleep(1)
        
        # Check if mobile menu is visible
        try:
            mobile_menu = driver.find_element(By.CLASS_NAME, "mobile-menu")
            return mobile_menu.is_displayed()
        except:
            return False
    
    def test_touch_interactions(self, driver):
        """Test touch-specific interactions"""
        actions = ActionChains(driver)
        
        # Test swipe gesture
        element = driver.find_element(By.CLASS_NAME, "swipeable-content")
        actions.click_and_hold(element).move_by_offset(-100, 0).release().perform()
        
        # Test tap
        button = driver.find_element(By.CLASS_NAME, "touch-button")
        actions.click(button).perform()
        
        # Test long press
        actions.click_and_hold(element).pause(2).release().perform()

class TestMobileResponsiveDesign(MobileTestBase):
    """Test responsive design across different devices"""
    
    @pytest.mark.parametrize("device", list(MOBILE_DEVICES.keys()))
    def test_001_responsive_layout(self, device):
        """Test layout responds correctly to different screen sizes"""
        driver = self.setup_mobile_driver(device)
        
        try:
            driver.get("http://localhost:3000")
            wait = WebDriverWait(driver, 10)
            
            # Check viewport meta tag
            viewport = driver.find_element(By.XPATH, "//meta[@name='viewport']")
            assert viewport.get_attribute("content") == "width=device-width, initial-scale=1"
            
            # Check mobile navigation
            if MOBILE_DEVICES[device]["width"] < 768:
                # Should show hamburger menu
                hamburger = wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "mobile-menu-toggle"))
                )
                assert hamburger.is_displayed()
                
                # Desktop nav should be hidden
                desktop_nav = driver.find_element(By.CLASS_NAME, "desktop-nav")
                assert not desktop_nav.is_displayed()
            else:
                # Should show desktop navigation
                desktop_nav = driver.find_element(By.CLASS_NAME, "desktop-nav")
                assert desktop_nav.is_displayed()
            
            # Check content scaling
            content = driver.find_element(By.CLASS_NAME, "main-content")
            content_width = content.size["width"]
            viewport_width = driver.execute_script("return window.innerWidth")
            
            # Content should not overflow viewport
            assert content_width <= viewport_width
            
        finally:
            driver.quit()
    
    def test_002_breakpoint_transitions(self):
        """Test smooth transitions between breakpoints"""
        driver = webdriver.Chrome()
        
        try:
            driver.get("http://localhost:3000")
            
            # Test different breakpoints
            breakpoints = [320, 480, 768, 1024, 1280, 1920]
            
            for width in breakpoints:
                driver.set_window_size(width, 800)
                time.sleep(0.5)
                
                # Check layout adjustments
                if width < 768:
                    assert self.check_responsive_breakpoint(driver, width)
                    
                    # Check font sizes scale appropriately
                    heading = driver.find_element(By.TAG_NAME, "h1")
                    font_size = heading.value_of_css_property("font-size")
                    assert font_size  # Should have appropriate mobile font size
                
                # Check grid layouts
                grid = driver.find_element(By.CLASS_NAME, "video-grid")
                columns = driver.execute_script(
                    "return window.getComputedStyle(arguments[0]).gridTemplateColumns",
                    grid
                )
                
                if width < 480:
                    assert "1fr" in columns  # Single column
                elif width < 768:
                    assert "repeat(2" in columns  # Two columns
                else:
                    assert "repeat(" in columns  # Multiple columns
                    
        finally:
            driver.quit()
    
    def test_003_touch_friendly_elements(self):
        """Test touch-friendly UI elements"""
        driver = self.setup_mobile_driver("iPhone_14_Pro")
        
        try:
            driver.get("http://localhost:3000")
            
            # Check minimum touch target sizes (44x44px for iOS, 48x48px for Android)
            buttons = driver.find_elements(By.TAG_NAME, "button")
            links = driver.find_elements(By.TAG_NAME, "a")
            
            min_size = 44  # iOS guideline
            
            for element in buttons + links:
                if element.is_displayed():
                    size = element.size
                    assert size["width"] >= min_size or size["height"] >= min_size
            
            # Check spacing between interactive elements
            interactive_elements = driver.find_elements(By.CSS_SELECTOR, "button, a, input")
            
            for i in range(len(interactive_elements) - 1):
                if interactive_elements[i].is_displayed():
                    rect1 = interactive_elements[i].rect
                    rect2 = interactive_elements[i + 1].rect
                    
                    # Calculate distance between elements
                    distance = min(
                        abs(rect1["x"] + rect1["width"] - rect2["x"]),
                        abs(rect1["y"] + rect1["height"] - rect2["y"])
                    )
                    
                    # Minimum 8px spacing for touch targets
                    if distance > 0:
                        assert distance >= 8
                        
        finally:
            driver.quit()

class TestMobilePerformance(MobileTestBase):
    """Test mobile performance metrics"""
    
    def test_004_page_load_performance(self):
        """Test page load performance on mobile"""
        driver = self.setup_mobile_driver("Samsung_Galaxy_S23")
        
        try:
            driver.get("http://localhost:3000")
            
            # Get performance metrics
            performance = driver.execute_script(
                "return window.performance.timing"
            )
            
            # Calculate metrics
            page_load_time = performance["loadEventEnd"] - performance["navigationStart"]
            dom_ready_time = performance["domContentLoadedEventEnd"] - performance["navigationStart"]
            
            # Mobile performance targets
            assert page_load_time < 3000  # 3 seconds
            assert dom_ready_time < 2000  # 2 seconds
            
            # Check First Contentful Paint
            fcp = driver.execute_script("""
                const perfData = window.performance.getEntriesByType('paint');
                return perfData.find(p => p.name === 'first-contentful-paint')?.startTime;
            """)
            
            assert fcp < 1500  # 1.5 seconds
            
            # Check Largest Contentful Paint
            lcp_observer = driver.execute_script("""
                return new Promise((resolve) => {
                    new PerformanceObserver((list) => {
                        const entries = list.getEntries();
                        const lastEntry = entries[entries.length - 1];
                        resolve(lastEntry.renderTime || lastEntry.loadTime);
                    }).observe({entryTypes: ['largest-contentful-paint']});
                });
            """)
            
        finally:
            driver.quit()
    
    def test_005_mobile_data_usage(self):
        """Test data usage optimization for mobile"""
        driver = self.setup_mobile_driver("Pixel_7")
        
        try:
            # Enable network throttling to simulate 3G
            driver.execute_cdp_cmd('Network.enable', {})
            driver.execute_cdp_cmd('Network.emulateNetworkConditions', {
                'offline': False,
                'downloadThroughput': 1.6 * 1024 * 1024 / 8,  # 1.6 Mbps
                'uploadThroughput': 768 * 1024 / 8,  # 768 Kbps
                'latency': 300  # 300ms
            })
            
            driver.get("http://localhost:3000")
            
            # Check if lazy loading is implemented
            images = driver.find_elements(By.TAG_NAME, "img")
            for img in images:
                loading_attr = img.get_attribute("loading")
                # Images below fold should have lazy loading
                if img.location["y"] > 800:
                    assert loading_attr == "lazy"
            
            # Check resource sizes
            resources = driver.execute_script("""
                return performance.getEntriesByType('resource').map(r => ({
                    name: r.name,
                    size: r.transferSize,
                    type: r.initiatorType
                }));
            """)
            
            # Check image optimization
            for resource in resources:
                if resource["type"] == "img":
                    # Images should be optimized (< 200KB for mobile)
                    assert resource["size"] < 200 * 1024
                    
        finally:
            driver.quit()

class TestMobileUserExperience(MobileTestBase):
    """Test mobile-specific user experience"""
    
    def test_006_mobile_navigation(self):
        """Test mobile navigation functionality"""
        driver = self.setup_mobile_driver("iPhone_SE")
        
        try:
            driver.get("http://localhost:3000")
            wait = WebDriverWait(driver, 10)
            
            # Test hamburger menu
            hamburger = wait.until(
                EC.element_to_be_clickable((By.CLASS_NAME, "mobile-menu-toggle"))
            )
            hamburger.click()
            
            # Check menu opens
            mobile_menu = wait.until(
                EC.visibility_of_element_located((By.CLASS_NAME, "mobile-menu"))
            )
            assert mobile_menu.is_displayed()
            
            # Test menu items
            menu_items = mobile_menu.find_elements(By.TAG_NAME, "a")
            assert len(menu_items) > 0
            
            # Test close menu
            close_btn = driver.find_element(By.CLASS_NAME, "mobile-menu-close")
            close_btn.click()
            
            # Menu should be hidden
            wait.until(EC.invisibility_of_element(mobile_menu))
            
        finally:
            driver.quit()
    
    def test_007_mobile_forms(self):
        """Test form usability on mobile"""
        driver = self.setup_mobile_driver("Samsung_Galaxy_S23")
        
        try:
            driver.get("http://localhost:3000/login")
            wait = WebDriverWait(driver, 10)
            
            # Check input field attributes
            email_input = wait.until(
                EC.presence_of_element_located((By.NAME, "email"))
            )
            
            # Check mobile-friendly input types
            assert email_input.get_attribute("type") == "email"
            assert email_input.get_attribute("autocomplete") == "email"
            
            password_input = driver.find_element(By.NAME, "password")
            assert password_input.get_attribute("type") == "password"
            
            # Check input sizes are touch-friendly
            input_height = email_input.size["height"]
            assert input_height >= 44  # Minimum touch target
            
            # Test virtual keyboard doesn't cover inputs
            email_input.click()
            time.sleep(0.5)
            
            # Check if input is still visible
            assert driver.execute_script(
                "return arguments[0].getBoundingClientRect().top > 0",
                email_input
            )
            
        finally:
            driver.quit()
    
    def test_008_mobile_video_player(self):
        """Test video player on mobile devices"""
        driver = self.setup_mobile_driver("iPad_Pro")
        
        try:
            driver.get("http://localhost:3000/videos/test-video")
            wait = WebDriverWait(driver, 10)
            
            # Find video player
            video_player = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "video-player"))
            )
            
            # Check if controls are mobile-optimized
            controls = video_player.find_element(By.CLASS_NAME, "video-controls")
            assert controls.is_displayed()
            
            # Check touch controls size
            play_button = controls.find_element(By.CLASS_NAME, "play-button")
            button_size = play_button.size
            assert button_size["width"] >= 44 and button_size["height"] >= 44
            
            # Test fullscreen on mobile
            fullscreen_btn = controls.find_element(By.CLASS_NAME, "fullscreen-button")
            fullscreen_btn.click()
            
            # Check if video goes fullscreen
            time.sleep(1)
            is_fullscreen = driver.execute_script(
                "return document.fullscreenElement !== null"
            )
            assert is_fullscreen
            
        finally:
            driver.quit()

class TestMobileAccessibility(MobileTestBase):
    """Test mobile accessibility features"""
    
    def test_009_mobile_accessibility(self):
        """Test accessibility on mobile devices"""
        driver = self.setup_mobile_driver("iPhone_14_Pro")
        
        try:
            driver.get("http://localhost:3000")
            
            # Check zoom is not disabled
            viewport = driver.find_element(By.XPATH, "//meta[@name='viewport']")
            content = viewport.get_attribute("content")
            assert "user-scalable=no" not in content
            assert "maximum-scale=1" not in content
            
            # Check focus indicators
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for button in buttons[:5]:  # Test first 5 buttons
                if button.is_displayed():
                    driver.execute_script("arguments[0].focus()", button)
                    outline = button.value_of_css_property("outline")
                    assert outline != "none"
            
            # Check ARIA labels for touch elements
            touch_elements = driver.find_elements(By.CSS_SELECTOR, "[role='button']")
            for element in touch_elements:
                aria_label = element.get_attribute("aria-label")
                assert aria_label is not None and len(aria_label) > 0
                
        finally:
            driver.quit()
    
    def test_010_orientation_handling(self):
        """Test orientation change handling"""
        driver = self.setup_mobile_driver("Samsung_Galaxy_S23")
        
        try:
            driver.get("http://localhost:3000")
            
            # Test portrait orientation
            driver.set_window_size(360, 780)
            time.sleep(0.5)
            
            portrait_layout = driver.find_element(By.CLASS_NAME, "main-content")
            portrait_width = portrait_layout.size["width"]
            
            # Test landscape orientation
            driver.set_window_size(780, 360)
            time.sleep(0.5)
            
            landscape_width = portrait_layout.size["width"]
            
            # Layout should adjust to orientation
            assert landscape_width > portrait_width
            
            # Check if UI elements reposition correctly
            header = driver.find_element(By.TAG_NAME, "header")
            header_height = header.size["height"]
            
            # Header should be more compact in landscape
            assert header_height < 100  # Reasonable height for landscape
            
        finally:
            driver.quit()

def run_mobile_tests():
    """Run all mobile tests"""
    pytest.main([
        __file__,
        "-v",
        "--html=mobile_test_report.html",
        "--self-contained-html",
        "-n", "4"  # Run tests in parallel
    ])

if __name__ == "__main__":
    print("Starting mobile test suite...")
    run_mobile_tests()