"""
Alert Service for threshold notifications
"""
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import logging

from app.core.config import settings
from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)


class AlertService:
    """Service for handling threshold alerts and notifications"""

    def __init__(self):
        self.webhook_session: Optional[aiohttp.ClientSession] = None
        self.alert_history: List[Dict] = []

    async def initialize(self):
        """Initialize alert service"""
        if not self.webhook_session:
            self.webhook_session = aiohttp.ClientSession()

    async def send_threshold_alert(
        self,
        alert_type: str,
        threshold_value: float,
        current_value: float,
        service: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """Send threshold exceeded alert through multiple channels"""
        alert_data = {
            "type": alert_type,
            "service": service,
            "threshold": threshold_value,
            "current_value": current_value,
            "exceeded_by": current_value - threshold_value,
            "percentage": (current_value / threshold_value * 100)
            if threshold_value > 0
            else 0,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        # Add to history
        self.alert_history.append(alert_data)

        # Send through different channels
        tasks = []

        # Email alert
        if settings.ALERT_EMAIL_ENABLED:
            tasks.append(self._send_email_alert(alert_data))

        # Webhook alert
        if settings.ALERT_WEBHOOK_URL:
            tasks.append(self._send_webhook_alert(alert_data))

        # Slack alert
        if settings.SLACK_WEBHOOK_URL:
            tasks.append(self._send_slack_alert(alert_data))

        # Discord alert
        if settings.DISCORD_WEBHOOK_URL:
            tasks.append(self._send_discord_alert(alert_data))

        # Execute all alerts concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Log alert
        logger.warning(f"THRESHOLD ALERT: {json.dumps(alert_data, indent=2)}")

    async def _send_email_alert(self, alert_data: Dict):
        """Send email alert"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = settings.SMTP_FROM_EMAIL
            msg["To"] = settings.ALERT_EMAIL_TO
            msg["Subject"] = f"YTEmpire Cost Alert: {alert_data['type'].upper()}"

            # Create body
            body = self._format_email_body(alert_data)
            msg.attach(MIMEText(body, "html"))

            # Send email
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                if settings.SMTP_USE_TLS:
                    server.starttls()
                if settings.SMTP_USERNAME:
                    server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                server.send_message(msg)

            logger.info(f"Email alert sent for {alert_data['type']}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    async def _send_webhook_alert(self, alert_data: Dict):
        """Send webhook alert"""
        try:
            if not self.webhook_session:
                await self.initialize()

            async with self.webhook_session.post(
                settings.ALERT_WEBHOOK_URL, json=alert_data, timeout=10
            ) as response:
                if response.status == 200:
                    logger.info(f"Webhook alert sent for {alert_data['type']}")
                else:
                    logger.error(f"Webhook alert failed with status {response.status}")

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    async def _send_slack_alert(self, alert_data: Dict):
        """Send Slack alert"""
        try:
            if not self.webhook_session:
                await self.initialize()

            # Format for Slack
            slack_message = {
                "text": f"🚨 Cost Alert: {alert_data['type'].upper()}",
                "attachments": [
                    {
                        "color": "danger"
                        if alert_data["percentage"] > 100
                        else "warning",
                        "fields": [
                            {
                                "title": "Type",
                                "value": alert_data["type"],
                                "short": True,
                            },
                            {
                                "title": "Service",
                                "value": alert_data.get("service", "N/A"),
                                "short": True,
                            },
                            {
                                "title": "Current Value",
                                "value": f"${alert_data['current_value']:.2f}",
                                "short": True,
                            },
                            {
                                "title": "Threshold",
                                "value": f"${alert_data['threshold']:.2f}",
                                "short": True,
                            },
                            {
                                "title": "Exceeded By",
                                "value": f"${alert_data['exceeded_by']:.2f}",
                                "short": True,
                            },
                            {
                                "title": "Percentage",
                                "value": f"{alert_data['percentage']:.1f}%",
                                "short": True,
                            },
                        ],
                        "footer": "YTEmpire Cost Tracking",
                        "ts": int(datetime.utcnow().timestamp()),
                    }
                ],
            }

            async with self.webhook_session.post(
                settings.SLACK_WEBHOOK_URL, json=slack_message, timeout=10
            ) as response:
                if response.status == 200:
                    logger.info(f"Slack alert sent for {alert_data['type']}")
                else:
                    logger.error(f"Slack alert failed with status {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    async def _send_discord_alert(self, alert_data: Dict):
        """Send Discord alert"""
        try:
            if not self.webhook_session:
                await self.initialize()

            # Format for Discord
            discord_message = {
                "content": f"@everyone **Cost Alert: {alert_data['type'].upper()}**",
                "embeds": [
                    {
                        "title": "Cost Threshold Exceeded",
                        "color": 15158332
                        if alert_data["percentage"] > 100
                        else 16776960,  # Red or Yellow
                        "fields": [
                            {
                                "name": "Alert Type",
                                "value": alert_data["type"],
                                "inline": True,
                            },
                            {
                                "name": "Service",
                                "value": alert_data.get("service", "N/A"),
                                "inline": True,
                            },
                            {
                                "name": "Current Cost",
                                "value": f"${alert_data['current_value']:.2f}",
                                "inline": True,
                            },
                            {
                                "name": "Threshold",
                                "value": f"${alert_data['threshold']:.2f}",
                                "inline": True,
                            },
                            {
                                "name": "Exceeded By",
                                "value": f"${alert_data['exceeded_by']:.2f}",
                                "inline": True,
                            },
                            {
                                "name": "Usage",
                                "value": f"{alert_data['percentage']:.1f}%",
                                "inline": True,
                            },
                        ],
                        "timestamp": alert_data["timestamp"],
                        "footer": {"text": "YTEmpire Cost Tracking System"},
                    }
                ],
            }

            async with self.webhook_session.post(
                settings.DISCORD_WEBHOOK_URL, json=discord_message, timeout=10
            ) as response:
                if response.status in [200, 204]:
                    logger.info(f"Discord alert sent for {alert_data['type']}")
                else:
                    logger.error(f"Discord alert failed with status {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

    def _format_email_body(self, alert_data: Dict) -> str:
        """Format email body HTML"""
        service_info = (
            f"<tr><td><strong>Service:</strong></td><td>{alert_data.get('service', 'N/A')}</td></tr>"
            if alert_data.get("service")
            else ""
        )

        return f"""
        <html>
            <body style="font-family: Arial, sans-serif;">
                <h2 style="color: #d32f2f;">YTEmpire Cost Alert</h2>
                <p>A cost threshold has been exceeded:</p>
                
                <table style="border-collapse: collapse; width: 100%; max-width: 600px;">
                    <tr style="background-color: #f5f5f5;">
                        <td style="padding: 10px; border: 1px solid #ddd;"><strong>Alert Type:</strong></td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{alert_data['type'].upper()}</td>
                    </tr>
                    {service_info}
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;"><strong>Current Value:</strong></td>
                        <td style="padding: 10px; border: 1px solid #ddd;">${alert_data['current_value']:.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;"><strong>Threshold:</strong></td>
                        <td style="padding: 10px; border: 1px solid #ddd;">${alert_data['threshold']:.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;"><strong>Exceeded By:</strong></td>
                        <td style="padding: 10px; border: 1px solid #ddd; color: #d32f2f;">${alert_data['exceeded_by']:.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;"><strong>Usage:</strong></td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{alert_data['percentage']:.1f}%</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;"><strong>Timestamp:</strong></td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{alert_data['timestamp']}</td>
                    </tr>
                </table>
                
                <p style="margin-top: 20px;">
                    <strong>Action Required:</strong> Please review your API usage and consider adjusting limits or upgrading your plan.
                </p>
                
                <p style="color: #666; font-size: 12px; margin-top: 30px;">
                    This is an automated message from YTEmpire Cost Tracking System.
                </p>
            </body>
        </html>
        """

    async def cleanup(self):
        """Cleanup resources"""
        if self.webhook_session:
            await self.webhook_session.close()


# Celery task for async alert processing
@celery_app.task
def send_cost_alert_task(alert_data: Dict):
    """Celery task to send cost alerts"""
    alert_service = AlertService()
    asyncio.run(
        alert_service.send_threshold_alert(
            alert_type=alert_data["type"],
            threshold_value=alert_data["threshold"],
            current_value=alert_data["current_value"],
            service=alert_data.get("service"),
            metadata=alert_data.get("metadata"),
        )
    )


# Global instance
alert_service = AlertService()
