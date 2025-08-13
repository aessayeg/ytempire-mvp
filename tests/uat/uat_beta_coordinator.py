#!/usr/bin/env python3
"""
Beta User Coordination System for UAT
Manages beta user onboarding, communication, and feedback collection
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import aiohttp
import asyncpg
from jinja2 import Template
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BetaUserProfile:
    """Extended beta user profile with tracking"""
    user_id: str
    name: str
    email: str
    company: Optional[str]
    role: str
    expertise_areas: List[str]
    timezone: str
    preferred_contact: str  # email, slack, discord
    onboarding_status: str
    test_environment_access: bool
    nda_signed: bool
    training_completed: bool
    active_since: datetime
    last_activity: Optional[datetime]
    total_tests_assigned: int = 0
    tests_completed: int = 0
    bugs_reported: int = 0
    feedback_quality_score: float = 0.0
    availability_hours: Dict[str, List[str]] = field(default_factory=dict)
    communication_preferences: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class TestingSession:
    """UAT testing session management"""
    session_id: str
    start_date: datetime
    end_date: datetime
    participants: List[str]
    test_scope: List[str]
    environment_url: str
    status: str
    completion_percentage: float = 0.0
    issues_found: int = 0
    feedback_collected: List[Dict[str, Any]] = field(default_factory=list)

class BetaUserCoordinator:
    """Manages beta user coordination for UAT"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.beta_users: List[BetaUserProfile] = []
        self.testing_sessions: List[TestingSession] = []
        self.communication_channels = {
            'email': self._setup_email_channel(),
            'slack': self._setup_slack_channel(),
            'discord': self._setup_discord_channel()
        }
        
    def _setup_email_channel(self):
        """Setup email communication channel"""
        return {
            'smtp_server': self.config.get('smtp_server', 'smtp.gmail.com'),
            'smtp_port': self.config.get('smtp_port', 587),
            'sender_email': self.config.get('sender_email', 'uat@ytempire.com'),
            'templates_dir': Path('templates/emails')
        }
        
    def _setup_slack_channel(self):
        """Setup Slack integration"""
        return {
            'webhook_url': self.config.get('slack_webhook'),
            'channel': '#uat-testing',
            'bot_name': 'UAT Coordinator'
        }
        
    def _setup_discord_channel(self):
        """Setup Discord integration"""
        return {
            'webhook_url': self.config.get('discord_webhook'),
            'server_id': self.config.get('discord_server'),
            'channel_id': self.config.get('discord_channel')
        }
        
    async def onboard_beta_user(self, user_data: Dict[str, Any]) -> BetaUserProfile:
        """Onboard a new beta user"""
        logger.info(f"Onboarding beta user: {user_data['name']}")
        
        # Create user profile
        user = BetaUserProfile(
            user_id=f"BETA-{datetime.now().strftime('%Y%m%d')}-{len(self.beta_users)+1:03d}",
            name=user_data['name'],
            email=user_data['email'],
            company=user_data.get('company'),
            role=user_data['role'],
            expertise_areas=user_data.get('expertise_areas', []),
            timezone=user_data.get('timezone', 'UTC'),
            preferred_contact=user_data.get('preferred_contact', 'email'),
            onboarding_status='pending',
            test_environment_access=False,
            nda_signed=False,
            training_completed=False,
            active_since=datetime.now(),
            last_activity=None,
            availability_hours=user_data.get('availability', {})
        )
        
        # Onboarding steps
        await self._send_welcome_package(user)
        await self._setup_test_environment_access(user)
        await self._schedule_training_session(user)
        
        user.onboarding_status = 'in_progress'
        self.beta_users.append(user)
        
        logger.info(f"Beta user {user.user_id} onboarded successfully")
        return user
        
    async def _send_welcome_package(self, user: BetaUserProfile):
        """Send welcome package to beta user"""
        welcome_content = f"""
        Welcome to YTEmpire UAT Beta Testing Program!
        
        Dear {user.name},
        
        Thank you for joining our beta testing program. Your expertise in {', '.join(user.expertise_areas)} 
        will be invaluable in ensuring our platform meets the highest quality standards.
        
        Next Steps:
        1. Sign the NDA (link provided separately)
        2. Complete the training module (30 minutes)
        3. Access the test environment
        4. Review assigned test cases
        5. Begin testing!
        
        Resources:
        - UAT Guide: https://docs.ytempire.com/uat-guide
        - Test Environment: https://uat.ytempire.com
        - Support Channel: uat-support@ytempire.com
        - Discord Server: [Invite Link]
        
        Your Coordinator: UAT Team
        Contact: uat@ytempire.com
        
        We're excited to have you on board!
        
        Best regards,
        YTEmpire UAT Team
        """
        
        # Send via preferred channel
        await self._send_communication(user, "Welcome to UAT Beta Testing", welcome_content)
        
    async def _setup_test_environment_access(self, user: BetaUserProfile):
        """Setup test environment access for user"""
        logger.info(f"Setting up test environment for {user.user_id}")
        
        # Create test account
        test_credentials = {
            'username': f"{user.email.split('@')[0]}_uat",
            'password': f"UAT2024_{user.user_id[-4:]}!",
            'environment_url': 'https://uat.ytempire.com',
            'api_key': f"uat_key_{user.user_id}",
            'test_data_access': True
        }
        
        # Send credentials securely
        await self._send_secure_credentials(user, test_credentials)
        
        user.test_environment_access = True
        
    async def _schedule_training_session(self, user: BetaUserProfile):
        """Schedule UAT training session"""
        training_schedule = {
            'session_type': 'individual' if user.role in ['admin', 'advanced'] else 'group',
            'duration': '30 minutes',
            'topics': [
                'Platform overview',
                'Testing procedures',
                'Bug reporting',
                'Feedback submission',
                'Tools and resources'
            ],
            'scheduled_time': self._find_suitable_time(user.availability_hours)
        }
        
        await self._send_communication(
            user,
            "UAT Training Session Scheduled",
            f"Your training is scheduled for {training_schedule['scheduled_time']}"
        )
        
    def _find_suitable_time(self, availability: Dict[str, List[str]]) -> str:
        """Find suitable time based on availability"""
        # Simple logic to find first available slot
        for day, hours in availability.items():
            if hours:
                return f"{day} at {hours[0]}"
        return "To be scheduled"
        
    async def _send_communication(self, user: BetaUserProfile, subject: str, content: str):
        """Send communication via user's preferred channel"""
        channel = user.preferred_contact
        
        if channel == 'email':
            await self._send_email(user.email, subject, content)
        elif channel == 'slack':
            await self._send_slack_message(user, content)
        elif channel == 'discord':
            await self._send_discord_message(user, content)
            
    async def _send_email(self, email: str, subject: str, content: str):
        """Send email notification"""
        logger.info(f"Sending email to {email}: {subject}")
        # Email implementation would go here
        
    async def _send_slack_message(self, user: BetaUserProfile, content: str):
        """Send Slack notification"""
        if self.communication_channels['slack']['webhook_url']:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'text': f"@{user.name}: {content}",
                    'channel': self.communication_channels['slack']['channel']
                }
                await session.post(
                    self.communication_channels['slack']['webhook_url'],
                    json=payload
                )
                
    async def _send_discord_message(self, user: BetaUserProfile, content: str):
        """Send Discord notification"""
        if self.communication_channels['discord']['webhook_url']:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'content': f"**{user.name}**: {content}"
                }
                await session.post(
                    self.communication_channels['discord']['webhook_url'],
                    json=payload
                )
                
    async def _send_secure_credentials(self, user: BetaUserProfile, credentials: Dict[str, Any]):
        """Send credentials securely"""
        # In production, use secure credential delivery
        secure_message = f"""
        Your UAT Test Environment Credentials:
        
        Username: {credentials['username']}
        Password: {credentials['password']}
        URL: {credentials['environment_url']}
        API Key: {credentials['api_key']}
        
        Please change your password on first login.
        Do not share these credentials.
        """
        
        await self._send_communication(user, "UAT Credentials", secure_message)
        
    async def create_testing_session(self, session_data: Dict[str, Any]) -> TestingSession:
        """Create a new testing session"""
        session = TestingSession(
            session_id=f"UAT-SESSION-{datetime.now().strftime('%Y%m%d-%H%M')}",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=session_data.get('duration_days', 3)),
            participants=session_data.get('participants', []),
            test_scope=session_data.get('test_scope', []),
            environment_url='https://uat.ytempire.com',
            status='scheduled'
        )
        
        self.testing_sessions.append(session)
        
        # Notify participants
        for participant_id in session.participants:
            user = next((u for u in self.beta_users if u.user_id == participant_id), None)
            if user:
                await self._notify_session_start(user, session)
                
        return session
        
    async def _notify_session_start(self, user: BetaUserProfile, session: TestingSession):
        """Notify user about testing session"""
        notification = f"""
        Testing Session Starting: {session.session_id}
        
        Duration: {session.start_date.strftime('%Y-%m-%d')} to {session.end_date.strftime('%Y-%m-%d')}
        Scope: {', '.join(session.test_scope)}
        Environment: {session.environment_url}
        
        Your assigned tests are available in the testing portal.
        """
        
        await self._send_communication(user, "UAT Session Starting", notification)
        
    async def collect_feedback(self, user_id: str, feedback_data: Dict[str, Any]):
        """Collect and process user feedback"""
        user = next((u for u in self.beta_users if u.user_id == user_id), None)
        if not user:
            logger.error(f"User {user_id} not found")
            return
            
        # Process feedback
        feedback = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'test_id': feedback_data.get('test_id'),
            'rating': feedback_data.get('rating'),
            'comments': feedback_data.get('comments'),
            'bugs': feedback_data.get('bugs', []),
            'suggestions': feedback_data.get('suggestions', []),
            'usability_score': feedback_data.get('usability_score'),
            'performance_score': feedback_data.get('performance_score')
        }
        
        # Update user metrics
        user.tests_completed += 1
        user.bugs_reported += len(feedback.get('bugs', []))
        user.last_activity = datetime.now()
        
        # Calculate feedback quality score
        quality_factors = [
            len(feedback.get('comments', '')) > 50,  # Detailed comments
            len(feedback.get('bugs', [])) > 0,  # Found bugs
            len(feedback.get('suggestions', [])) > 0,  # Provided suggestions
            feedback.get('rating') is not None  # Provided rating
        ]
        user.feedback_quality_score = sum(quality_factors) / len(quality_factors)
        
        # Store feedback
        current_session = self.testing_sessions[-1] if self.testing_sessions else None
        if current_session:
            current_session.feedback_collected.append(feedback)
            current_session.issues_found += len(feedback.get('bugs', []))
            
        logger.info(f"Feedback collected from {user.name}")
        
    async def generate_beta_dashboard(self):
        """Generate beta testing dashboard"""
        logger.info("Generating beta testing dashboard...")
        
        # Create dashboard with plotly
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'User Activity', 'Test Completion Rate',
                'Bug Discovery Trend', 'Feedback Quality',
                'Session Progress', 'User Engagement'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'indicator'}, {'type': 'heatmap'}]
            ]
        )
        
        # User Activity
        user_names = [u.name for u in self.beta_users]
        tests_completed = [u.tests_completed for u in self.beta_users]
        
        fig.add_trace(
            go.Bar(x=user_names, y=tests_completed, name='Tests Completed'),
            row=1, col=1
        )
        
        # Test Completion Rate
        if self.testing_sessions:
            session = self.testing_sessions[-1]
            completed = len([f for f in session.feedback_collected if f])
            total = len(session.participants) * 10  # Assuming 10 tests per user
            
            fig.add_trace(
                go.Pie(
                    labels=['Completed', 'Pending'],
                    values=[completed, total - completed],
                    hole=0.4
                ),
                row=1, col=2
            )
            
        # Bug Discovery Trend
        dates = []
        bug_counts = []
        for session in self.testing_sessions:
            dates.append(session.start_date)
            bug_counts.append(session.issues_found)
            
        fig.add_trace(
            go.Scatter(x=dates, y=bug_counts, mode='lines+markers', name='Bugs Found'),
            row=2, col=1
        )
        
        # Feedback Quality
        quality_scores = [u.feedback_quality_score for u in self.beta_users]
        
        fig.add_trace(
            go.Bar(x=user_names, y=quality_scores, name='Quality Score'),
            row=2, col=2
        )
        
        # Session Progress Indicator
        if self.testing_sessions:
            progress = self.testing_sessions[-1].completion_percentage
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=progress,
                    title={'text': "Overall Progress"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [None, 100]},
                          'bar': {'color': "darkgreen"},
                          'steps': [
                              {'range': [0, 50], 'color': "lightgray"},
                              {'range': [50, 80], 'color': "yellow"},
                              {'range': [80, 100], 'color': "green"}
                          ]}
                ),
                row=3, col=1
            )
            
        # Update layout
        fig.update_layout(
            title="UAT Beta Testing Dashboard",
            showlegend=False,
            height=900,
            template="plotly_white"
        )
        
        # Save dashboard
        dashboard_path = Path('uat_reports') / 'beta_dashboard.html'
        fig.write_html(str(dashboard_path))
        
        logger.info(f"Dashboard saved to {dashboard_path}")
        
    async def send_daily_digest(self):
        """Send daily digest to all beta users"""
        logger.info("Sending daily digest to beta users...")
        
        for user in self.beta_users:
            if user.onboarding_status == 'completed':
                digest = await self._generate_user_digest(user)
                await self._send_communication(user, "UAT Daily Digest", digest)
                
    async def _generate_user_digest(self, user: BetaUserProfile) -> str:
        """Generate personalized daily digest"""
        digest = f"""
        Daily UAT Digest for {user.name}
        Date: {datetime.now().strftime('%Y-%m-%d')}
        
        Your Progress:
        - Tests Completed: {user.tests_completed}/{user.total_tests_assigned}
        - Bugs Reported: {user.bugs_reported}
        - Quality Score: {user.feedback_quality_score:.2f}/1.00
        
        Today's Focus:
        - Complete remaining critical tests
        - Review and update bug reports
        - Provide detailed feedback on usability
        
        Team Updates:
        - Total tests completed: {sum(u.tests_completed for u in self.beta_users)}
        - Critical bugs found: {sum(u.bugs_reported for u in self.beta_users)}
        - Average completion: {sum(u.tests_completed for u in self.beta_users) / max(sum(u.total_tests_assigned for u in self.beta_users), 1) * 100:.1f}%
        
        Need Help?
        Contact UAT support at uat-support@ytempire.com
        """
        
        return digest
        
    async def coordinate_beta_testing_cycle(self):
        """Coordinate complete beta testing cycle"""
        logger.info("Starting beta testing coordination cycle...")
        
        # Phase 1: Recruitment and Onboarding
        logger.info("Phase 1: Recruitment and Onboarding")
        beta_user_data = [
            {
                'name': 'Alex Johnson',
                'email': 'alex.j@example.com',
                'role': 'experienced_creator',
                'expertise_areas': ['content_creation', 'analytics'],
                'timezone': 'PST',
                'availability': {'Monday': ['9am-12pm'], 'Wednesday': ['2pm-5pm']}
            },
            {
                'name': 'Maria Garcia',
                'email': 'maria.g@example.com',
                'role': 'channel_manager',
                'expertise_areas': ['channel_management', 'monetization'],
                'timezone': 'EST',
                'availability': {'Tuesday': ['10am-2pm'], 'Thursday': ['3pm-6pm']}
            },
            {
                'name': 'David Lee',
                'email': 'david.l@example.com',
                'role': 'analytics_expert',
                'expertise_areas': ['data_analysis', 'reporting'],
                'timezone': 'CST',
                'availability': {'Daily': ['Flexible']}
            },
            {
                'name': 'Sophie Chen',
                'email': 'sophie.c@example.com',
                'role': 'new_user',
                'expertise_areas': ['ui_testing', 'usability'],
                'timezone': 'PST',
                'availability': {'Weekends': ['10am-4pm']}
            },
            {
                'name': 'Robert Brown',
                'email': 'robert.b@example.com',
                'role': 'technical_user',
                'expertise_areas': ['api_testing', 'integration'],
                'timezone': 'MST',
                'availability': {'Weekdays': ['Evening']}
            }
        ]
        
        for user_data in beta_user_data:
            await self.onboard_beta_user(user_data)
            
        # Phase 2: Testing Session Setup
        logger.info("Phase 2: Testing Session Setup")
        session = await self.create_testing_session({
            'duration_days': 5,
            'participants': [u.user_id for u in self.beta_users],
            'test_scope': [
                'Authentication',
                'Channel Management',
                'Video Generation',
                'Analytics',
                'Payment Processing'
            ]
        })
        
        # Phase 3: Active Testing
        logger.info("Phase 3: Active Testing")
        session.status = 'active'
        
        # Simulate feedback collection
        for user in self.beta_users:
            user.total_tests_assigned = 10
            user.onboarding_status = 'completed'
            user.nda_signed = True
            user.training_completed = True
            
            # Simulate test execution and feedback
            for i in range(5):  # Each user completes 5 tests
                await self.collect_feedback(user.user_id, {
                    'test_id': f'TEST-{i+1:03d}',
                    'rating': 4,
                    'comments': f'Test executed successfully with minor issues noted.',
                    'bugs': [f'Bug-{i}'] if i % 2 == 0 else [],
                    'suggestions': ['Improve UI response time'],
                    'usability_score': 4.5,
                    'performance_score': 4.0
                })
                
        # Phase 4: Analysis and Reporting
        logger.info("Phase 4: Analysis and Reporting")
        session.completion_percentage = 50.0  # 50% completion
        session.status = 'analysis'
        
        # Generate dashboard
        await self.generate_beta_dashboard()
        
        # Send daily digest
        await self.send_daily_digest()
        
        # Phase 5: Sign-off Collection
        logger.info("Phase 5: Sign-off Collection")
        sign_offs = []
        for user in self.beta_users:
            sign_off = {
                'user_id': user.user_id,
                'user_name': user.name,
                'role': user.role,
                'approval': user.tests_completed >= 5,
                'comments': 'Testing completed satisfactorily',
                'date': datetime.now().isoformat()
            }
            sign_offs.append(sign_off)
            
        logger.info(f"Collected {len(sign_offs)} sign-offs")
        
        # Final status
        session.status = 'completed'
        
        logger.info("Beta testing coordination cycle completed")
        
        return {
            'users_onboarded': len(self.beta_users),
            'tests_completed': sum(u.tests_completed for u in self.beta_users),
            'bugs_found': sum(u.bugs_reported for u in self.beta_users),
            'sign_offs_received': len([s for s in sign_offs if s['approval']]),
            'dashboard_location': 'uat_reports/beta_dashboard.html'
        }

async def main():
    """Main execution"""
    config = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': 'uat@ytempire.com',
        'slack_webhook': None,  # Set if using Slack
        'discord_webhook': None,  # Set if using Discord
    }
    
    coordinator = BetaUserCoordinator(config)
    results = await coordinator.coordinate_beta_testing_cycle()
    
    print("\n" + "="*60)
    print("BETA USER COORDINATION COMPLETE")
    print("="*60)
    print(f"Users Onboarded: {results['users_onboarded']}")
    print(f"Tests Completed: {results['tests_completed']}")
    print(f"Bugs Found: {results['bugs_found']}")
    print(f"Sign-offs Received: {results['sign_offs_received']}")
    print(f"Dashboard: {results['dashboard_location']}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())