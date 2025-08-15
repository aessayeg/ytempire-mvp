"""
Email service for sending verification and notification emails
"""
import os
from typing import Optional, Dict, Any
from datetime import datetime
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from jinja2 import Environment, FileSystemLoader
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmailService:
    """
    Email service for sending transactional emails
    """
    
    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_user = settings.SMTP_USER
        self.smtp_password = settings.SMTP_PASSWORD
        self.from_email = settings.EMAILS_FROM_EMAIL or "noreply@ytempire.com"
        self.from_name = settings.EMAILS_FROM_NAME or "YTEmpire"
        self.base_url = "http://localhost:3000"  # Frontend URL
        
        # Setup Jinja2 for email templates
        template_dir = os.path.join(os.path.dirname(__file__), "../templates/emails")
        self.template_env = Environment(loader=FileSystemLoader(template_dir))
    
    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None
    ) -> bool:
        """
        Send email using SMTP
        """
        try:
            message = MIMEMultipart("alternative")
            message["From"] = f"{self.from_name} <{self.from_email}>"
            message["To"] = to_email
            message["Subject"] = subject
            
            # Add text part
            if text_content:
                text_part = MIMEText(text_content, "plain")
                message.attach(text_part)
            
            # Add HTML part
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)
            
            # Send email
            async with aiosmtplib.SMTP(
                hostname=self.smtp_host,
                port=self.smtp_port,
                use_tls=True
            ) as smtp:
                await smtp.login(self.smtp_user, self.smtp_password)
                await smtp.send_message(message)
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {str(e)}")
            return False
    
    async def send_verification_email(
        self,
        to_email: str,
        full_name: str,
        verification_token: str
    ) -> bool:
        """
        Send email verification link
        """
        verification_url = f"{self.base_url}/verify-email?token={verification_token}"
        
        # Create email content
        subject = "Verify Your YTEmpire Account"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; padding: 15px 30px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to YTEmpire!</h1>
                </div>
                <div class="content">
                    <h2>Hi {full_name},</h2>
                    <p>Thank you for creating an account with YTEmpire! We're excited to help you automate your YouTube content creation.</p>
                    <p>Please verify your email address by clicking the button below:</p>
                    <center>
                        <a href="{verification_url}" class="button">Verify Email Address</a>
                    </center>
                    <p>Or copy and paste this link into your browser:</p>
                    <p style="word-break: break-all; background: #f0f0f0; padding: 10px; border-radius: 5px;">{verification_url}</p>
                    <p>This link will expire in 24 hours for security reasons.</p>
                    <p>If you didn't create an account with YTEmpire, please ignore this email.</p>
                    <p>Best regards,<br>The YTEmpire Team</p>
                </div>
                <div class="footer">
                    <p>© {datetime.now().year} YTEmpire. All rights reserved.</p>
                    <p>This is an automated message, please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        Welcome to YTEmpire!
        
        Hi {full_name},
        
        Thank you for creating an account with YTEmpire!
        
        Please verify your email address by visiting:
        {verification_url}
        
        This link will expire in 24 hours.
        
        If you didn't create an account, please ignore this email.
        
        Best regards,
        The YTEmpire Team
        """
        
        return await self.send_email(to_email, subject, html_content, text_content)
    
    async def send_password_reset_email(
        self,
        to_email: str,
        full_name: str,
        reset_token: str
    ) -> bool:
        """
        Send password reset email
        """
        reset_url = f"{self.base_url}/reset-password?token={reset_token}"
        
        subject = "Reset Your YTEmpire Password"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; padding: 15px 30px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .warning {{ background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Password Reset Request</h1>
                </div>
                <div class="content">
                    <h2>Hi {full_name},</h2>
                    <p>We received a request to reset your YTEmpire account password.</p>
                    <p>Click the button below to reset your password:</p>
                    <center>
                        <a href="{reset_url}" class="button">Reset Password</a>
                    </center>
                    <p>Or copy and paste this link into your browser:</p>
                    <p style="word-break: break-all; background: #f0f0f0; padding: 10px; border-radius: 5px;">{reset_url}</p>
                    <div class="warning">
                        <strong>⚠️ Security Notice:</strong><br>
                        This link will expire in 1 hour for security reasons.<br>
                        If you didn't request a password reset, please ignore this email and your password will remain unchanged.
                    </div>
                    <p>Best regards,<br>The YTEmpire Team</p>
                </div>
                <div class="footer">
                    <p>© {datetime.now().year} YTEmpire. All rights reserved.</p>
                    <p>This is an automated message, please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        Password Reset Request
        
        Hi {full_name},
        
        We received a request to reset your YTEmpire account password.
        
        Reset your password by visiting:
        {reset_url}
        
        This link will expire in 1 hour.
        
        If you didn't request a password reset, please ignore this email.
        
        Best regards,
        The YTEmpire Team
        """
        
        return await self.send_email(to_email, subject, html_content, text_content)
    
    async def send_welcome_email(
        self,
        to_email: str,
        full_name: str
    ) -> bool:
        """
        Send welcome email after successful verification
        """
        subject = "Welcome to YTEmpire - Let's Get Started!"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .feature {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #667eea; }}
                .button {{ display: inline-block; padding: 15px 30px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎉 Welcome to YTEmpire!</h1>
                </div>
                <div class="content">
                    <h2>Hi {full_name},</h2>
                    <p>Your email has been verified and your account is now active!</p>
                    <p>Here's what you can do with YTEmpire:</p>
                    
                    <div class="feature">
                        <strong>🎥 Automated Video Creation</strong><br>
                        Generate professional YouTube videos with AI-powered scripts and voiceovers
                    </div>
                    
                    <div class="feature">
                        <strong>📊 Analytics Dashboard</strong><br>
                        Track your channel's performance and optimize your content strategy
                    </div>
                    
                    <div class="feature">
                        <strong>🚀 Bulk Generation</strong><br>
                        Create multiple videos at once and schedule them for optimal posting times
                    </div>
                    
                    <div class="feature">
                        <strong>💰 Cost Optimization</strong><br>
                        Generate videos for less than $3 each with our efficient AI pipeline
                    </div>
                    
                    <center>
                        <a href="{self.base_url}/dashboard" class="button">Go to Dashboard</a>
                    </center>
                    
                    <p>Need help getting started? Check out our <a href="{self.base_url}/docs">documentation</a> or contact our support team.</p>
                    
                    <p>Best regards,<br>The YTEmpire Team</p>
                </div>
                <div class="footer">
                    <p>© {datetime.now().year} YTEmpire. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return await self.send_email(to_email, subject, html_content)