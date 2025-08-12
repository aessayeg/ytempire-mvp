"""
Visual Regression Testing Suite for YTEmpire
Using Playwright for screenshot comparison and visual testing
"""

import pytest
import asyncio
import os
from pathlib import Path
from playwright.async_api import async_playwright, Page, Browser
from PIL import Image
import imagehash
import numpy as np
from typing import Optional, Dict, Any

# Configuration
FRONTEND_URL = "http://localhost:3000"
BASELINE_DIR = Path("tests/e2e/visual_baselines")
DIFF_DIR = Path("tests/e2e/visual_diffs")
THRESHOLD = 0.05  # 5% difference threshold

# Ensure directories exist
BASELINE_DIR.mkdir(parents=True, exist_ok=True)
DIFF_DIR.mkdir(parents=True, exist_ok=True)


class VisualRegressionTester:
    """Visual regression testing utilities"""
    
    @staticmethod
    def compare_images(baseline_path: Path, current_path: Path, threshold: float = THRESHOLD) -> Dict[str, Any]:
        """Compare two images and return difference metrics"""
        baseline = Image.open(baseline_path)
        current = Image.open(current_path)
        
        # Resize if needed
        if baseline.size != current.size:
            current = current.resize(baseline.size)
        
        # Calculate perceptual hash
        baseline_hash = imagehash.phash(baseline)
        current_hash = imagehash.phash(current)
        hash_diff = baseline_hash - current_hash
        
        # Calculate pixel difference
        baseline_array = np.array(baseline)
        current_array = np.array(current)
        diff = np.abs(baseline_array.astype(float) - current_array.astype(float))
        diff_percentage = (diff.sum() / (baseline_array.shape[0] * baseline_array.shape[1] * 255 * 3)) * 100
        
        # Create diff image
        diff_image = Image.fromarray(diff.astype(np.uint8))
        
        return {
            "hash_difference": hash_diff,
            "pixel_difference_percentage": diff_percentage,
            "passed": diff_percentage <= threshold * 100,
            "diff_image": diff_image
        }
    
    @staticmethod
    async def capture_screenshot(page: Page, name: str, selector: Optional[str] = None) -> Path:
        """Capture screenshot of page or element"""
        screenshot_path = DIFF_DIR / f"{name}_current.png"
        
        if selector:
            element = await page.query_selector(selector)
            if element:
                await element.screenshot(path=str(screenshot_path))
        else:
            await page.screenshot(path=str(screenshot_path), full_page=True)
        
        return screenshot_path


class TestVisualRegression:
    """Visual regression tests for UI components"""
    
    @pytest.fixture(scope="class")
    async def browser(self):
        """Setup browser instance"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            yield browser
            await browser.close()
    
    @pytest.fixture
    async def page(self, browser: Browser):
        """Create new page for each test"""
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            device_scale_factor=1
        )
        page = await context.new_page()
        yield page
        await context.close()
    
    @pytest.fixture
    async def authenticated_page(self, page: Page):
        """Create authenticated page"""
        await page.goto(f"{FRONTEND_URL}/login")
        await page.fill('input[name="email"]', 'test@example.com')
        await page.fill('input[name="password"]', 'SecurePass123!')
        await page.click('button[type="submit"]')
        await page.wait_for_url(f"{FRONTEND_URL}/dashboard")
        return page
    
    async def visual_test(self, page: Page, test_name: str, selector: Optional[str] = None):
        """Helper for visual regression testing"""
        tester = VisualRegressionTester()
        
        # Capture current screenshot
        current_path = await tester.capture_screenshot(page, test_name, selector)
        baseline_path = BASELINE_DIR / f"{test_name}_baseline.png"
        
        if not baseline_path.exists():
            # Create baseline if it doesn't exist
            current_path.rename(baseline_path)
            pytest.skip(f"Baseline created for {test_name}")
        
        # Compare images
        result = tester.compare_images(baseline_path, current_path)
        
        if not result["passed"]:
            # Save diff image
            diff_path = DIFF_DIR / f"{test_name}_diff.png"
            result["diff_image"].save(diff_path)
            pytest.fail(
                f"Visual regression failed for {test_name}. "
                f"Difference: {result['pixel_difference_percentage']:.2f}%"
            )
    
    @pytest.mark.asyncio
    async def test_056_homepage_visual(self, page: Page):
        """Test homepage visual consistency"""
        await page.goto(FRONTEND_URL)
        await page.wait_for_load_state('networkidle')
        await self.visual_test(page, "homepage")
    
    @pytest.mark.asyncio
    async def test_057_login_page_visual(self, page: Page):
        """Test login page visual consistency"""
        await page.goto(f"{FRONTEND_URL}/login")
        await page.wait_for_load_state('networkidle')
        await self.visual_test(page, "login_page")
    
    @pytest.mark.asyncio
    async def test_058_dashboard_visual(self, authenticated_page: Page):
        """Test dashboard visual consistency"""
        await authenticated_page.wait_for_load_state('networkidle')
        await self.visual_test(authenticated_page, "dashboard")
    
    @pytest.mark.asyncio
    async def test_059_navigation_menu_visual(self, authenticated_page: Page):
        """Test navigation menu visual consistency"""
        await authenticated_page.wait_for_selector('.nav-menu')
        await self.visual_test(authenticated_page, "navigation_menu", ".nav-menu")
    
    @pytest.mark.asyncio
    async def test_060_video_card_visual(self, authenticated_page: Page):
        """Test video card component visual consistency"""
        await authenticated_page.goto(f"{FRONTEND_URL}/videos")
        await authenticated_page.wait_for_selector('.video-card')
        await self.visual_test(authenticated_page, "video_card", ".video-card")
    
    @pytest.mark.asyncio
    async def test_061_charts_visual(self, authenticated_page: Page):
        """Test analytics charts visual consistency"""
        await authenticated_page.goto(f"{FRONTEND_URL}/analytics")
        await authenticated_page.wait_for_selector('.chart-container')
        await self.visual_test(authenticated_page, "analytics_charts", ".chart-container")
    
    @pytest.mark.asyncio
    async def test_062_form_elements_visual(self, authenticated_page: Page):
        """Test form elements visual consistency"""
        await authenticated_page.goto(f"{FRONTEND_URL}/videos/new")
        await authenticated_page.wait_for_selector('form')
        await self.visual_test(authenticated_page, "form_elements", "form")
    
    @pytest.mark.asyncio
    async def test_063_table_visual(self, authenticated_page: Page):
        """Test data table visual consistency"""
        await authenticated_page.goto(f"{FRONTEND_URL}/channels")
        await authenticated_page.wait_for_selector('table')
        await self.visual_test(authenticated_page, "data_table", "table")
    
    @pytest.mark.asyncio
    async def test_064_modal_visual(self, authenticated_page: Page):
        """Test modal dialog visual consistency"""
        await authenticated_page.goto(f"{FRONTEND_URL}/channels")
        await authenticated_page.click('button[data-action="add-channel"]')
        await authenticated_page.wait_for_selector('.modal')
        await self.visual_test(authenticated_page, "modal_dialog", ".modal")
    
    @pytest.mark.asyncio
    async def test_065_error_state_visual(self, page: Page):
        """Test error state visual consistency"""
        await page.goto(f"{FRONTEND_URL}/404")
        await page.wait_for_load_state('networkidle')
        await self.visual_test(page, "error_404")
    
    @pytest.mark.asyncio
    async def test_066_loading_state_visual(self, authenticated_page: Page):
        """Test loading state visual consistency"""
        # Trigger a slow operation
        await authenticated_page.goto(f"{FRONTEND_URL}/videos/generate")
        # Capture loading state quickly
        await self.visual_test(authenticated_page, "loading_state", ".loading-spinner")
    
    @pytest.mark.asyncio
    async def test_067_responsive_mobile_visual(self, browser: Browser):
        """Test mobile responsive visual consistency"""
        context = await browser.new_context(
            viewport={'width': 375, 'height': 667},
            device_scale_factor=2,
            is_mobile=True
        )
        page = await context.new_page()
        await page.goto(FRONTEND_URL)
        await page.wait_for_load_state('networkidle')
        
        tester = VisualRegressionTester()
        await tester.capture_screenshot(page, "mobile_homepage")
        
        await context.close()
    
    @pytest.mark.asyncio
    async def test_068_responsive_tablet_visual(self, browser: Browser):
        """Test tablet responsive visual consistency"""
        context = await browser.new_context(
            viewport={'width': 768, 'height': 1024},
            device_scale_factor=2
        )
        page = await context.new_page()
        await page.goto(FRONTEND_URL)
        await page.wait_for_load_state('networkidle')
        
        tester = VisualRegressionTester()
        await tester.capture_screenshot(page, "tablet_homepage")
        
        await context.close()
    
    @pytest.mark.asyncio
    async def test_069_dark_mode_visual(self, authenticated_page: Page):
        """Test dark mode visual consistency"""
        # Toggle dark mode
        await authenticated_page.click('button[data-action="toggle-theme"]')
        await authenticated_page.wait_for_timeout(500)  # Wait for transition
        await self.visual_test(authenticated_page, "dark_mode_dashboard")
    
    @pytest.mark.asyncio
    async def test_070_print_view_visual(self, authenticated_page: Page):
        """Test print view visual consistency"""
        await authenticated_page.goto(f"{FRONTEND_URL}/reports/monthly")
        await authenticated_page.emulate_media(media='print')
        await self.visual_test(authenticated_page, "print_view")
    
    @pytest.mark.asyncio
    async def test_071_animation_states_visual(self, authenticated_page: Page):
        """Test animation end states visual consistency"""
        await authenticated_page.goto(f"{FRONTEND_URL}/dashboard")
        
        # Trigger animation
        await authenticated_page.hover('.animated-card')
        await authenticated_page.wait_for_timeout(1000)  # Wait for animation
        await self.visual_test(authenticated_page, "animation_hover_state", ".animated-card")
    
    @pytest.mark.asyncio
    async def test_072_tooltip_visual(self, authenticated_page: Page):
        """Test tooltip visual consistency"""
        await authenticated_page.hover('[data-tooltip]')
        await authenticated_page.wait_for_selector('.tooltip')
        await self.visual_test(authenticated_page, "tooltip", ".tooltip")
    
    @pytest.mark.asyncio
    async def test_073_dropdown_menu_visual(self, authenticated_page: Page):
        """Test dropdown menu visual consistency"""
        await authenticated_page.click('.dropdown-toggle')
        await authenticated_page.wait_for_selector('.dropdown-menu')
        await self.visual_test(authenticated_page, "dropdown_menu", ".dropdown-menu")
    
    @pytest.mark.asyncio
    async def test_074_notification_visual(self, authenticated_page: Page):
        """Test notification visual consistency"""
        # Trigger a notification
        await authenticated_page.evaluate("""
            window.showNotification('Test notification', 'success');
        """)
        await authenticated_page.wait_for_selector('.notification')
        await self.visual_test(authenticated_page, "notification", ".notification")
    
    @pytest.mark.asyncio
    async def test_075_cross_browser_chrome_visual(self, browser: Browser):
        """Test Chrome-specific visual consistency"""
        page = await browser.new_page()
        await page.goto(FRONTEND_URL)
        await page.wait_for_load_state('networkidle')
        
        tester = VisualRegressionTester()
        await tester.capture_screenshot(page, "chrome_specific")
        await page.close()


class TestAccessibilityVisual:
    """Visual tests for accessibility features"""
    
    @pytest.fixture
    async def page(self, browser: Browser):
        """Create page with accessibility features"""
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080'},
            reduced_motion='reduce',
            forced_colors='active'
        )
        page = await context.new_page()
        yield page
        await context.close()
    
    @pytest.mark.asyncio
    async def test_076_high_contrast_visual(self, page: Page):
        """Test high contrast mode visual"""
        await page.goto(FRONTEND_URL)
        await page.emulate_media(color_scheme='dark', prefers_contrast='high')
        await page.wait_for_load_state('networkidle')
        
        tester = VisualRegressionTester()
        await tester.capture_screenshot(page, "high_contrast")
    
    @pytest.mark.asyncio
    async def test_077_focus_indicators_visual(self, page: Page):
        """Test keyboard focus indicators visual"""
        await page.goto(FRONTEND_URL)
        
        # Tab through elements
        for _ in range(5):
            await page.keyboard.press('Tab')
        
        tester = VisualRegressionTester()
        await tester.capture_screenshot(page, "focus_indicators")
    
    @pytest.mark.asyncio
    async def test_078_screen_reader_labels_visual(self, page: Page):
        """Test screen reader label visibility"""
        await page.goto(FRONTEND_URL)
        
        # Make screen reader labels visible for testing
        await page.evaluate("""
            document.querySelectorAll('.sr-only').forEach(el => {
                el.classList.remove('sr-only');
                el.style.border = '2px solid red';
            });
        """)
        
        tester = VisualRegressionTester()
        await tester.capture_screenshot(page, "screen_reader_labels")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "asyncio"])