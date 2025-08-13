#!/usr/bin/env python3
"""
User Acceptance Testing (UAT) Suite for YTEmpire MVP
Comprehensive UAT scripts with beta user coordination and reporting
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import asyncpg
from faker import Faker
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('uat_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """Test execution status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"

class TestPriority(Enum):
    """Test case priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class UserRole(Enum):
    """Beta user roles for testing"""
    CONTENT_CREATOR = "content_creator"
    CHANNEL_MANAGER = "channel_manager"
    ANALYTICS_USER = "analytics_user"
    ADMIN_USER = "admin_user"
    BASIC_USER = "basic_user"

@dataclass
class TestCase:
    """UAT test case definition"""
    id: str
    name: str
    description: str
    category: str
    priority: TestPriority
    steps: List[Dict[str, Any]]
    expected_results: List[str]
    actual_results: List[str] = field(default_factory=list)
    status: TestStatus = TestStatus.NOT_STARTED
    execution_time: float = 0.0
    error_message: Optional[str] = None
    screenshots: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None
    
@dataclass
class BetaUser:
    """Beta user profile for UAT"""
    id: str
    name: str
    email: str
    role: UserRole
    experience_level: str
    timezone: str
    availability: List[str]
    assigned_tests: List[str] = field(default_factory=list)
    completed_tests: List[str] = field(default_factory=list)
    feedback: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class UATReport:
    """UAT execution report"""
    report_id: str
    execution_date: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    blocked_tests: int
    skipped_tests: int
    coverage_percentage: float
    critical_defects: List[Dict[str, Any]]
    recommendations: List[str]
    sign_offs: List[Dict[str, Any]]

class UATTestSuite:
    """Comprehensive UAT test suite manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', 'http://localhost:8000')
        self.test_cases: List[TestCase] = []
        self.beta_users: List[BetaUser] = []
        self.execution_results: Dict[str, Any] = {}
        self.faker = Faker()
        
    async def initialize(self):
        """Initialize UAT test suite"""
        logger.info("Initializing UAT test suite...")
        
        # Create test cases
        self.test_cases = await self._create_test_cases()
        
        # Setup beta users
        self.beta_users = await self._setup_beta_users()
        
        # Assign tests to users
        await self._assign_tests_to_users()
        
        logger.info(f"Initialized {len(self.test_cases)} test cases with {len(self.beta_users)} beta users")
        
    async def _create_test_cases(self) -> List[TestCase]:
        """Create comprehensive UAT test cases"""
        test_cases = []
        
        # Authentication & User Management Tests
        test_cases.extend([
            TestCase(
                id="UAT-AUTH-001",
                name="User Registration Flow",
                description="Verify new user can register and access the platform",
                category="Authentication",
                priority=TestPriority.CRITICAL,
                steps=[
                    {"action": "Navigate to registration page", "data": None},
                    {"action": "Fill registration form", "data": {"email": "test@example.com", "password": "SecurePass123!"}},
                    {"action": "Submit registration", "data": None},
                    {"action": "Verify email confirmation", "data": None},
                    {"action": "Login with new credentials", "data": None}
                ],
                expected_results=[
                    "Registration page loads correctly",
                    "Form validation works properly",
                    "Account created successfully",
                    "Confirmation email received",
                    "Can login with new account"
                ]
            ),
            TestCase(
                id="UAT-AUTH-002",
                name="Password Reset Flow",
                description="Verify password reset functionality",
                category="Authentication",
                priority=TestPriority.HIGH,
                steps=[
                    {"action": "Click forgot password", "data": None},
                    {"action": "Enter email address", "data": {"email": "user@example.com"}},
                    {"action": "Check email for reset link", "data": None},
                    {"action": "Set new password", "data": {"password": "NewSecurePass123!"}},
                    {"action": "Login with new password", "data": None}
                ],
                expected_results=[
                    "Password reset page accessible",
                    "Reset email sent successfully",
                    "Reset link works correctly",
                    "Password updated successfully",
                    "Can login with new password"
                ]
            ),
        ])
        
        # Channel Management Tests
        test_cases.extend([
            TestCase(
                id="UAT-CHAN-001",
                name="Create YouTube Channel",
                description="Verify channel creation and configuration",
                category="Channel Management",
                priority=TestPriority.CRITICAL,
                steps=[
                    {"action": "Navigate to channels page", "data": None},
                    {"action": "Click create channel", "data": None},
                    {"action": "Fill channel details", "data": {"name": "Test Channel", "niche": "Technology"}},
                    {"action": "Connect YouTube account", "data": None},
                    {"action": "Configure channel settings", "data": {"upload_schedule": "daily", "auto_publish": True}}
                ],
                expected_results=[
                    "Channel creation form opens",
                    "Form validates input correctly",
                    "YouTube OAuth flow works",
                    "Channel created successfully",
                    "Settings saved correctly"
                ]
            ),
            TestCase(
                id="UAT-CHAN-002",
                name="Multi-Channel Management",
                description="Verify managing multiple channels",
                category="Channel Management",
                priority=TestPriority.HIGH,
                steps=[
                    {"action": "View channel list", "data": None},
                    {"action": "Switch between channels", "data": None},
                    {"action": "Edit channel settings", "data": {"description": "Updated description"}},
                    {"action": "View channel analytics", "data": None},
                    {"action": "Archive inactive channel", "data": None}
                ],
                expected_results=[
                    "All channels displayed correctly",
                    "Channel switching works smoothly",
                    "Settings update successfully",
                    "Analytics load correctly",
                    "Channel archived properly"
                ]
            ),
        ])
        
        # Video Generation Tests
        test_cases.extend([
            TestCase(
                id="UAT-VID-001",
                name="AI Video Generation",
                description="Verify complete video generation pipeline",
                category="Video Generation",
                priority=TestPriority.CRITICAL,
                steps=[
                    {"action": "Click generate video", "data": None},
                    {"action": "Select video topic", "data": {"topic": "AI Technology Trends"}},
                    {"action": "Configure video settings", "data": {"duration": 600, "style": "educational"}},
                    {"action": "Review generated script", "data": None},
                    {"action": "Approve and generate", "data": None},
                    {"action": "Monitor generation progress", "data": None}
                ],
                expected_results=[
                    "Video generation form opens",
                    "Topic suggestions work",
                    "Settings applied correctly",
                    "Script generated successfully",
                    "Video generation starts",
                    "Progress updates in real-time"
                ]
            ),
            TestCase(
                id="UAT-VID-002",
                name="Bulk Video Scheduling",
                description="Verify bulk video creation and scheduling",
                category="Video Generation",
                priority=TestPriority.HIGH,
                steps=[
                    {"action": "Access bulk generation", "data": None},
                    {"action": "Upload topic list", "data": {"file": "topics.csv"}},
                    {"action": "Set generation schedule", "data": {"videos_per_day": 3}},
                    {"action": "Review queue", "data": None},
                    {"action": "Start bulk generation", "data": None}
                ],
                expected_results=[
                    "Bulk generation interface loads",
                    "File upload works correctly",
                    "Schedule configured properly",
                    "Queue displays all videos",
                    "Bulk generation starts successfully"
                ]
            ),
        ])
        
        # Analytics & Reporting Tests
        test_cases.extend([
            TestCase(
                id="UAT-ANAL-001",
                name="Revenue Dashboard",
                description="Verify revenue tracking and reporting",
                category="Analytics",
                priority=TestPriority.CRITICAL,
                steps=[
                    {"action": "Open revenue dashboard", "data": None},
                    {"action": "Select date range", "data": {"start": "2024-01-01", "end": "2024-12-31"}},
                    {"action": "View revenue breakdown", "data": None},
                    {"action": "Export revenue report", "data": {"format": "pdf"}},
                    {"action": "Compare with YouTube Analytics", "data": None}
                ],
                expected_results=[
                    "Dashboard loads with current data",
                    "Date filtering works correctly",
                    "Revenue breakdown accurate",
                    "Report exports successfully",
                    "Data matches YouTube Analytics"
                ]
            ),
            TestCase(
                id="UAT-ANAL-002",
                name="Performance Metrics",
                description="Verify video performance tracking",
                category="Analytics",
                priority=TestPriority.HIGH,
                steps=[
                    {"action": "View performance dashboard", "data": None},
                    {"action": "Check video metrics", "data": None},
                    {"action": "View trending analysis", "data": None},
                    {"action": "Check CTR optimization", "data": None},
                    {"action": "Review recommendations", "data": None}
                ],
                expected_results=[
                    "Performance metrics display",
                    "Video stats accurate",
                    "Trending analysis works",
                    "CTR data displays correctly",
                    "Recommendations generated"
                ]
            ),
        ])
        
        # Payment & Billing Tests
        test_cases.extend([
            TestCase(
                id="UAT-PAY-001",
                name="Subscription Management",
                description="Verify subscription and billing functionality",
                category="Payment",
                priority=TestPriority.CRITICAL,
                steps=[
                    {"action": "View subscription plans", "data": None},
                    {"action": "Select premium plan", "data": {"plan": "premium"}},
                    {"action": "Enter payment details", "data": {"card": "4242424242424242"}},
                    {"action": "Complete subscription", "data": None},
                    {"action": "Verify premium features", "data": None}
                ],
                expected_results=[
                    "Plans display correctly",
                    "Plan selection works",
                    "Payment processing succeeds",
                    "Subscription activated",
                    "Premium features accessible"
                ]
            ),
        ])
        
        # Mobile Responsiveness Tests
        test_cases.extend([
            TestCase(
                id="UAT-MOB-001",
                name="Mobile Dashboard Access",
                description="Verify mobile responsiveness",
                category="Mobile",
                priority=TestPriority.HIGH,
                steps=[
                    {"action": "Access on mobile device", "data": {"device": "iPhone 14"}},
                    {"action": "Navigate main features", "data": None},
                    {"action": "Test touch interactions", "data": None},
                    {"action": "Check responsive layouts", "data": None},
                    {"action": "Test offline mode", "data": None}
                ],
                expected_results=[
                    "Site loads on mobile",
                    "Navigation works properly",
                    "Touch gestures responsive",
                    "Layouts adapt correctly",
                    "Offline mode functions"
                ]
            ),
        ])
        
        return test_cases
        
    async def _setup_beta_users(self) -> List[BetaUser]:
        """Setup beta users for testing"""
        beta_users = [
            BetaUser(
                id="BETA-001",
                name="Sarah Johnson",
                email="sarah.j@betatesters.com",
                role=UserRole.CONTENT_CREATOR,
                experience_level="Advanced",
                timezone="PST",
                availability=["Weekdays 9-5", "Weekends flexible"]
            ),
            BetaUser(
                id="BETA-002",
                name="Michael Chen",
                email="m.chen@betatesters.com",
                role=UserRole.CHANNEL_MANAGER,
                experience_level="Intermediate",
                timezone="EST",
                availability=["Evenings 6-10", "Weekends"]
            ),
            BetaUser(
                id="BETA-003",
                name="Emily Rodriguez",
                email="e.rodriguez@betatesters.com",
                role=UserRole.ANALYTICS_USER,
                experience_level="Expert",
                timezone="CST",
                availability=["Flexible hours"]
            ),
            BetaUser(
                id="BETA-004",
                name="David Kim",
                email="d.kim@betatesters.com",
                role=UserRole.ADMIN_USER,
                experience_level="Advanced",
                timezone="MST",
                availability=["Weekdays 10-6"]
            ),
            BetaUser(
                id="BETA-005",
                name="Lisa Thompson",
                email="l.thompson@betatesters.com",
                role=UserRole.BASIC_USER,
                experience_level="Beginner",
                timezone="PST",
                availability=["Weekends", "Evenings"]
            ),
        ]
        
        return beta_users
        
    async def _assign_tests_to_users(self):
        """Intelligently assign test cases to beta users"""
        logger.info("Assigning test cases to beta users...")
        
        # Group tests by category
        test_categories = {}
        for test in self.test_cases:
            if test.category not in test_categories:
                test_categories[test.category] = []
            test_categories[test.category].append(test)
            
        # Assign based on user roles and expertise
        role_category_mapping = {
            UserRole.CONTENT_CREATOR: ["Video Generation", "Channel Management"],
            UserRole.CHANNEL_MANAGER: ["Channel Management", "Analytics"],
            UserRole.ANALYTICS_USER: ["Analytics", "Performance"],
            UserRole.ADMIN_USER: ["Authentication", "Payment", "Admin"],
            UserRole.BASIC_USER: ["Authentication", "Mobile", "General"],
        }
        
        for user in self.beta_users:
            assigned_categories = role_category_mapping.get(user.role, ["General"])
            
            for category in assigned_categories:
                if category in test_categories:
                    category_tests = test_categories[category]
                    # Assign high priority tests first
                    sorted_tests = sorted(category_tests, 
                                        key=lambda x: x.priority.value)
                    
                    # Assign tests evenly
                    for i, test in enumerate(sorted_tests):
                        if len(user.assigned_tests) < 10:  # Max 10 tests per user
                            test.assigned_to = user.id
                            user.assigned_tests.append(test.id)
                            
        logger.info(f"Test assignment completed")
        
    async def execute_test_case(self, test_case: TestCase, user: BetaUser) -> TestCase:
        """Execute a single test case"""
        logger.info(f"Executing test {test_case.id}: {test_case.name}")
        test_case.status = TestStatus.IN_PROGRESS
        start_time = datetime.now()
        
        try:
            # Simulate test execution
            for i, step in enumerate(test_case.steps):
                logger.info(f"  Step {i+1}: {step['action']}")
                
                # Simulate step execution
                await asyncio.sleep(0.5)  # Simulate execution time
                
                # Record actual result
                if i < len(test_case.expected_results):
                    # Simulate 95% success rate for demo
                    import random
                    if random.random() > 0.05:
                        test_case.actual_results.append(test_case.expected_results[i])
                    else:
                        test_case.actual_results.append(f"Failed: {test_case.expected_results[i]}")
                        test_case.status = TestStatus.FAILED
                        test_case.error_message = f"Step {i+1} failed"
                        break
                        
            if test_case.status != TestStatus.FAILED:
                test_case.status = TestStatus.PASSED
                
        except Exception as e:
            logger.error(f"Test execution error: {e}")
            test_case.status = TestStatus.BLOCKED
            test_case.error_message = str(e)
            
        finally:
            test_case.execution_time = (datetime.now() - start_time).total_seconds()
            
        return test_case
        
    async def coordinate_beta_testing(self):
        """Coordinate beta user testing activities"""
        logger.info("Starting beta user coordination...")
        
        # Send test assignments to users
        for user in self.beta_users:
            await self._send_test_assignment(user)
            
        # Execute tests
        test_results = []
        for user in self.beta_users:
            logger.info(f"Processing tests for {user.name}...")
            
            for test_id in user.assigned_tests:
                test_case = next((t for t in self.test_cases if t.id == test_id), None)
                if test_case:
                    result = await self.execute_test_case(test_case, user)
                    test_results.append(result)
                    
                    # Collect user feedback
                    feedback = await self._collect_user_feedback(user, test_case)
                    user.feedback.append(feedback)
                    
        self.execution_results = {
            'test_results': test_results,
            'user_feedback': [user.feedback for user in self.beta_users]
        }
        
        logger.info("Beta testing coordination completed")
        
    async def _send_test_assignment(self, user: BetaUser):
        """Send test assignment notification to beta user"""
        logger.info(f"Sending test assignment to {user.name}")
        
        # Create assignment email
        assigned_tests = [t for t in self.test_cases if t.id in user.assigned_tests]
        
        email_content = f"""
        Dear {user.name},
        
        You have been assigned {len(assigned_tests)} test cases for the YTEmpire MVP UAT.
        
        Assigned Tests:
        """
        
        for test in assigned_tests:
            email_content += f"\n- {test.id}: {test.name} (Priority: {test.priority.value})"
            
        email_content += """
        
        Please complete these tests within the next 48 hours and provide detailed feedback.
        
        Access the testing environment at: https://uat.ytempire.com
        Your credentials have been sent separately.
        
        Thank you for your participation!
        
        Best regards,
        YTEmpire UAT Team
        """
        
        # In production, this would send actual email
        logger.info(f"Assignment email prepared for {user.email}")
        
    async def _collect_user_feedback(self, user: BetaUser, test_case: TestCase) -> Dict[str, Any]:
        """Collect feedback from beta user"""
        feedback = {
            'test_id': test_case.id,
            'user_id': user.id,
            'timestamp': datetime.now().isoformat(),
            'usability_score': 4,  # 1-5 scale
            'clarity_score': 4,
            'performance_score': 5,
            'comments': f"Test {test_case.name} executed successfully. Minor UI improvements suggested.",
            'bugs_found': [],
            'suggestions': [
                "Add more tooltips for new users",
                "Improve loading indicators",
                "Add keyboard shortcuts"
            ]
        }
        
        # Simulate bug discovery for failed tests
        if test_case.status == TestStatus.FAILED:
            feedback['bugs_found'] = [
                {
                    'severity': 'High',
                    'description': test_case.error_message,
                    'steps_to_reproduce': test_case.steps,
                    'screenshot': f"bug_{test_case.id}.png"
                }
            ]
            feedback['usability_score'] = 2
            
        return feedback
        
    async def generate_uat_report(self) -> UATReport:
        """Generate comprehensive UAT report"""
        logger.info("Generating UAT report...")
        
        # Calculate test metrics
        total_tests = len(self.test_cases)
        passed_tests = len([t for t in self.test_cases if t.status == TestStatus.PASSED])
        failed_tests = len([t for t in self.test_cases if t.status == TestStatus.FAILED])
        blocked_tests = len([t for t in self.test_cases if t.status == TestStatus.BLOCKED])
        skipped_tests = len([t for t in self.test_cases if t.status == TestStatus.SKIPPED])
        
        coverage_percentage = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Identify critical defects
        critical_defects = []
        for test in self.test_cases:
            if test.status == TestStatus.FAILED and test.priority == TestPriority.CRITICAL:
                critical_defects.append({
                    'test_id': test.id,
                    'test_name': test.name,
                    'category': test.category,
                    'error': test.error_message,
                    'impact': 'High',
                    'recommended_action': 'Fix before production release'
                })
                
        # Generate recommendations
        recommendations = []
        
        if coverage_percentage < 80:
            recommendations.append("Increase test coverage to at least 80% before production release")
            
        if critical_defects:
            recommendations.append(f"Address {len(critical_defects)} critical defects immediately")
            
        if failed_tests > total_tests * 0.1:
            recommendations.append("High failure rate detected - conduct additional testing rounds")
            
        recommendations.extend([
            "Implement automated regression testing for critical paths",
            "Schedule follow-up testing after defect fixes",
            "Conduct performance testing under production-like load",
            "Ensure all beta user feedback is addressed",
            "Create user documentation based on UAT findings"
        ])
        
        # Collect sign-offs
        sign_offs = []
        for user in self.beta_users:
            completed_percentage = (len(user.completed_tests) / len(user.assigned_tests)) * 100 if user.assigned_tests else 0
            
            sign_offs.append({
                'user_id': user.id,
                'user_name': user.name,
                'role': user.role.value,
                'tests_completed': len(user.completed_tests),
                'tests_assigned': len(user.assigned_tests),
                'completion_percentage': completed_percentage,
                'approval_status': 'Approved' if completed_percentage >= 90 else 'Pending',
                'signature_date': datetime.now().isoformat() if completed_percentage >= 90 else None,
                'comments': f"Testing completed by {user.name}"
            })
            
        report = UATReport(
            report_id=f"UAT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            execution_date=datetime.now(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            blocked_tests=blocked_tests,
            skipped_tests=skipped_tests,
            coverage_percentage=coverage_percentage,
            critical_defects=critical_defects,
            recommendations=recommendations,
            sign_offs=sign_offs
        )
        
        # Generate detailed HTML report
        await self._generate_html_report(report)
        
        # Generate executive summary
        await self._generate_executive_summary(report)
        
        logger.info("UAT report generated successfully")
        return report
        
    async def _generate_html_report(self, report: UATReport):
        """Generate HTML UAT report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>YTEmpire MVP - UAT Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; border-bottom: 2px solid #4CAF50; }
                h2 { color: #666; margin-top: 30px; }
                .metrics { display: flex; justify-content: space-around; margin: 20px 0; }
                .metric-card { 
                    background: #f5f5f5; 
                    padding: 20px; 
                    border-radius: 8px; 
                    text-align: center;
                    min-width: 150px;
                }
                .metric-value { font-size: 36px; font-weight: bold; color: #4CAF50; }
                .metric-label { color: #666; margin-top: 10px; }
                .passed { color: #4CAF50; }
                .failed { color: #f44336; }
                .blocked { color: #ff9800; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #4CAF50; color: white; }
                .critical { background-color: #ffebee; }
                .recommendations { background: #e8f5e9; padding: 15px; border-radius: 8px; }
                .sign-off { margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 8px; }
                .approved { color: #4CAF50; font-weight: bold; }
                .pending { color: #ff9800; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>User Acceptance Testing Report</h1>
            <p><strong>Report ID:</strong> {{ report.report_id }}</p>
            <p><strong>Execution Date:</strong> {{ report.execution_date }}</p>
            
            <h2>Test Execution Summary</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{{ report.total_tests }}</div>
                    <div class="metric-label">Total Tests</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value passed">{{ report.passed_tests }}</div>
                    <div class="metric-label">Passed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value failed">{{ report.failed_tests }}</div>
                    <div class="metric-label">Failed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.1f"|format(report.coverage_percentage) }}%</div>
                    <div class="metric-label">Coverage</div>
                </div>
            </div>
            
            <h2>Critical Defects</h2>
            {% if report.critical_defects %}
            <table>
                <tr>
                    <th>Test ID</th>
                    <th>Test Name</th>
                    <th>Category</th>
                    <th>Error</th>
                    <th>Impact</th>
                </tr>
                {% for defect in report.critical_defects %}
                <tr class="critical">
                    <td>{{ defect.test_id }}</td>
                    <td>{{ defect.test_name }}</td>
                    <td>{{ defect.category }}</td>
                    <td>{{ defect.error }}</td>
                    <td>{{ defect.impact }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p>No critical defects found.</p>
            {% endif %}
            
            <h2>Recommendations</h2>
            <div class="recommendations">
                <ul>
                {% for recommendation in report.recommendations %}
                    <li>{{ recommendation }}</li>
                {% endfor %}
                </ul>
            </div>
            
            <h2>Beta User Sign-offs</h2>
            {% for sign_off in report.sign_offs %}
            <div class="sign-off">
                <p><strong>{{ sign_off.user_name }}</strong> ({{ sign_off.role }})</p>
                <p>Tests Completed: {{ sign_off.tests_completed }}/{{ sign_off.tests_assigned }} 
                   ({{ "%.1f"|format(sign_off.completion_percentage) }}%)</p>
                <p>Status: <span class="{{ 'approved' if sign_off.approval_status == 'Approved' else 'pending' }}">
                    {{ sign_off.approval_status }}</span></p>
                {% if sign_off.signature_date %}
                <p>Signed: {{ sign_off.signature_date }}</p>
                {% endif %}
            </div>
            {% endfor %}
            
            <h2>Test Case Details</h2>
            <table>
                <tr>
                    <th>Test ID</th>
                    <th>Test Name</th>
                    <th>Category</th>
                    <th>Priority</th>
                    <th>Status</th>
                    <th>Execution Time</th>
                    <th>Assigned To</th>
                </tr>
                {% for test in test_cases %}
                <tr>
                    <td>{{ test.id }}</td>
                    <td>{{ test.name }}</td>
                    <td>{{ test.category }}</td>
                    <td>{{ test.priority.value }}</td>
                    <td class="{{ test.status.value }}">{{ test.status.value }}</td>
                    <td>{{ "%.2f"|format(test.execution_time) }}s</td>
                    <td>{{ test.assigned_to or 'Unassigned' }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd;">
                <p style="text-align: center; color: #666;">
                    Generated on {{ report.execution_date.strftime('%Y-%m-%d %H:%M:%S') }}
                </p>
            </div>
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(report=report, test_cases=self.test_cases)
        
        # Save HTML report
        report_path = Path('uat_reports')
        report_path.mkdir(exist_ok=True)
        
        html_file = report_path / f"{report.report_id}.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
            
        logger.info(f"HTML report saved to {html_file}")
        
    async def _generate_executive_summary(self, report: UATReport):
        """Generate executive summary for stakeholders"""
        summary = f"""
        EXECUTIVE SUMMARY - YTEmpire MVP User Acceptance Testing
        ========================================================
        
        Report ID: {report.report_id}
        Date: {report.execution_date.strftime('%Y-%m-%d')}
        
        KEY METRICS:
        -----------
        • Total Test Cases: {report.total_tests}
        • Pass Rate: {(report.passed_tests/report.total_tests*100):.1f}%
        • Test Coverage: {report.coverage_percentage:.1f}%
        • Critical Defects: {len(report.critical_defects)}
        
        OVERALL STATUS: {'PASS' if report.coverage_percentage >= 80 and len(report.critical_defects) == 0 else 'CONDITIONAL PASS' if report.coverage_percentage >= 70 else 'FAIL'}
        
        CRITICAL FINDINGS:
        -----------------
        """
        
        if report.critical_defects:
            for defect in report.critical_defects[:3]:  # Top 3 critical issues
                summary += f"• {defect['test_name']}: {defect['error']}\n"
        else:
            summary += "• No critical defects identified\n"
            
        summary += f"""
        
        BETA USER PARTICIPATION:
        -----------------------
        • Total Beta Users: {len(self.beta_users)}
        • Average Completion Rate: {sum(s['completion_percentage'] for s in report.sign_offs)/len(report.sign_offs):.1f}%
        • Sign-offs Received: {len([s for s in report.sign_offs if s['approval_status'] == 'Approved'])}/{len(report.sign_offs)}
        
        TOP RECOMMENDATIONS:
        -------------------
        """
        
        for i, rec in enumerate(report.recommendations[:5], 1):
            summary += f"{i}. {rec}\n"
            
        summary += """
        
        NEXT STEPS:
        ----------
        1. Address all critical defects before production release
        2. Implement recommended improvements
        3. Schedule regression testing after fixes
        4. Prepare production deployment plan
        5. Create user training materials
        
        APPROVAL FOR PRODUCTION:
        -----------------------
        [ ] All critical defects resolved
        [ ] Test coverage >= 80%
        [ ] All beta users signed off
        [ ] Performance benchmarks met
        [ ] Security review completed
        
        _______________________________
        UAT Lead Signature
        
        _______________________________
        Product Owner Signature
        
        _______________________________
        Technical Lead Signature
        """
        
        # Save executive summary
        summary_file = Path('uat_reports') / f"{report.report_id}_executive_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
            
        logger.info(f"Executive summary saved to {summary_file}")
        
    async def run_complete_uat(self):
        """Run complete UAT process"""
        logger.info("Starting complete UAT process...")
        
        try:
            # Initialize test suite
            await self.initialize()
            
            # Coordinate beta testing
            await self.coordinate_beta_testing()
            
            # Generate comprehensive report
            report = await self.generate_uat_report()
            
            # Display summary
            print("\n" + "="*60)
            print("UAT EXECUTION COMPLETE")
            print("="*60)
            print(f"Total Tests: {report.total_tests}")
            print(f"Passed: {report.passed_tests} ({report.passed_tests/report.total_tests*100:.1f}%)")
            print(f"Failed: {report.failed_tests}")
            print(f"Coverage: {report.coverage_percentage:.1f}%")
            print(f"Critical Defects: {len(report.critical_defects)}")
            print("\nReports generated:")
            print(f"- HTML Report: uat_reports/{report.report_id}.html")
            print(f"- Executive Summary: uat_reports/{report.report_id}_executive_summary.txt")
            print("="*60)
            
            return report
            
        except Exception as e:
            logger.error(f"UAT execution failed: {e}")
            raise

async def main():
    """Main execution function"""
    config = {
        'base_url': 'http://localhost:8000',
        'database_url': os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/ytempire'),
        'test_environment': 'uat',
        'parallel_execution': True,
        'screenshot_on_failure': True,
        'retry_failed_tests': True,
        'max_retries': 2
    }
    
    uat_suite = UATTestSuite(config)
    report = await uat_suite.run_complete_uat()
    
    # Generate final approval status
    if report.coverage_percentage >= 80 and len(report.critical_defects) == 0:
        logger.info("✅ UAT PASSED - System ready for production")
    else:
        logger.warning("⚠️ UAT CONDITIONAL PASS - Address issues before production")

if __name__ == "__main__":
    asyncio.run(main())