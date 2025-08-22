import os
import json
import aiohttp
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import asyncio
from ..core.settings import settings

class NotificationManager:
    """Manages notifications for code review findings and analysis completion"""
    
    def __init__(self):
        self.smtp_config = self._load_smtp_config()
        self.slack_config = self._load_slack_config()
        self.discord_config = self._load_discord_config()
        self.webhook_configs = self._load_webhook_configs()
        
        # Notification templates
        self.templates = self._load_notification_templates()
    
    def _load_smtp_config(self) -> Dict:
        """Load SMTP configuration for email notifications"""
        return {
            "enabled": os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "false").lower() == "true",
            "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "username": os.getenv("SMTP_USERNAME", ""),
            "password": os.getenv("SMTP_PASSWORD", ""),
            "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true",
            "from_email": os.getenv("FROM_EMAIL", "code-review@company.com"),
            "from_name": os.getenv("FROM_NAME", "Code Review Agent")
        }
    
    def _load_slack_config(self) -> Dict:
        """Load Slack configuration"""
        return {
            "enabled": os.getenv("SLACK_NOTIFICATIONS_ENABLED", "false").lower() == "true",
            "webhook_url": os.getenv("SLACK_WEBHOOK_URL", ""),
            "channel": os.getenv("SLACK_CHANNEL", "#code-review"),
            "username": os.getenv("SLACK_USERNAME", "Code Review Bot"),
            "icon_emoji": os.getenv("SLACK_ICON_EMOJI", ":robot_face:")
        }
    
    def _load_discord_config(self) -> Dict:
        """Load Discord configuration"""
        return {
            "enabled": os.getenv("DISCORD_NOTIFICATIONS_ENABLED", "false").lower() == "true",
            "webhook_url": os.getenv("DISCORD_WEBHOOK_URL", ""),
            "username": os.getenv("DISCORD_USERNAME", "Code Review Bot"),
            "avatar_url": os.getenv("DISCORD_AVATAR_URL", "")
        }
    
    def _load_webhook_configs(self) -> List[Dict]:
        """Load webhook configurations"""
        webhook_configs = []
        
        # Load from environment variables
        webhook_urls = os.getenv("WEBHOOK_URLS", "")
        if webhook_urls:
            for url in webhook_urls.split(","):
                webhook_configs.append({
                    "url": url.strip(),
                    "enabled": True,
                    "headers": {},
                    "timeout": 30
                })
        
        # Load from config file if exists
        config_file = Path("webhook_config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_configs = json.load(f)
                    webhook_configs.extend(file_configs)
            except Exception as e:
                print(f"Warning: Could not load webhook config: {e}")
        
        return webhook_configs
    
    def _load_notification_templates(self) -> Dict:
        """Load notification message templates"""
        return {
            "critical_finding": {
                "subject": "🚨 Critical Security Finding Detected",
                "email_template": """
                <h2>🚨 Critical Security Finding</h2>
                <p><strong>Repository:</strong> {repo_name}</p>
                <p><strong>File:</strong> {file_path}:{line_number}</p>
                <p><strong>Issue:</strong> {message}</p>
                <p><strong>Tool:</strong> {tool}</p>
                <p><strong>Severity:</strong> {severity}</p>
                <p><strong>Remediation:</strong> {remediation}</p>
                <p><strong>Risk Score:</strong> {risk_score}/10</p>
                <br>
                <p><em>This requires immediate attention. Please review and address as soon as possible.</em></p>
                """,
                "slack_template": """
🚨 *Critical Security Finding Detected*
• *Repository:* {repo_name}
• *File:* {file_path}:{line_number}
• *Issue:* {message}
• *Tool:* {tool}
• *Severity:* {severity}
• *Risk Score:* {risk_score}/10
                """,
                "discord_template": """
🚨 **Critical Security Finding Detected**
• **Repository:** {repo_name}
• **File:** {file_path}:{line_number}
• **Issue:** {message}
• **Tool:** {tool}
• **Severity:** {severity}
• **Risk Score:** {risk_score}/10
                """
            },
            "analysis_complete": {
                "subject": "✅ Code Review Analysis Complete",
                "email_template": """
                <h2>✅ Code Review Analysis Complete</h2>
                <p><strong>Repository:</strong> {repo_name}</p>
                <p><strong>Analysis Date:</strong> {analysis_date}</p>
                <p><strong>Total Findings:</strong> {total_findings}</p>
                <p><strong>Critical Issues:</strong> {critical_count}</p>
                <p><strong>High Issues:</strong> {high_count}</p>
                <p><strong>Overall Risk Score:</strong> {risk_score}/10</p>
                <br>
                <p><strong>Summary:</strong></p>
                <ul>
                    {findings_summary}
                </ul>
                <br>
                <p><em>Review the detailed report for complete analysis.</em></p>
                """,
                "slack_template": """
✅ *Code Review Analysis Complete*
• *Repository:* {repo_name}
• *Total Findings:* {total_findings}
• *Critical Issues:* {critical_count}
• *High Issues:* {high_count}
• *Risk Score:* {risk_score}/10
                """,
                "discord_template": """
✅ **Code Review Analysis Complete**
• **Repository:** {repo_name}
• **Total Findings:** {total_findings}
• **Critical Issues:** {critical_count}
• **High Issues:** {high_count}
• **Risk Score:** {risk_score}/10
                """
            },
            "pr_review_ready": {
                "subject": "🔍 PR Review Ready for {repo_name}",
                "email_template": """
                <h2>🔍 PR Review Ready</h2>
                <p><strong>Repository:</strong> {repo_name}</p>
                <p><strong>PR Number:</strong> #{pr_number}</p>
                <p><strong>Branch:</strong> {branch_name}</p>
                <p><strong>Findings:</strong> {findings_count}</p>
                <p><strong>Risk Level:</strong> {risk_level}</p>
                <br>
                <p><em>Automated code review is complete. Please review the findings and take appropriate action.</em></p>
                """,
                "slack_template": """
🔍 *PR Review Ready*
• *Repository:* {repo_name}
• *PR:* #{pr_number}
• *Branch:* {branch_name}
• *Findings:* {findings_count}
• *Risk Level:* {risk_level}
                """,
                "discord_template": """
🔍 **PR Review Ready**
• **Repository:** {repo_name}
• **PR:** #{pr_number}
• **Branch:** {branch_name}
• **Findings:** {findings_count}
• **Risk Level:** {risk_level}
                """
            }
        }
    
    async def send_critical_finding_notification(self, finding: Dict, repo_info: Dict) -> Dict:
        """Send notification for critical findings"""
        notification_results = {
            "email": {"sent": False, "error": None},
            "slack": {"sent": False, "error": None},
            "discord": {"sent": False, "error": None},
            "webhooks": {"sent": False, "error": None}
        }
        
        # Prepare notification data
        notification_data = {
            "repo_name": repo_info.get("name", "Unknown"),
            "file_path": finding.get("file", "Unknown"),
            "line_number": finding.get("line", 0),
            "message": finding.get("message", "No message"),
            "tool": finding.get("tool", "Unknown"),
            "severity": finding.get("severity", "Unknown"),
            "remediation": finding.get("remediation", "No remediation provided"),
            "risk_score": finding.get("risk_score", 0)
        }
        
        # Send email notification
        if self.smtp_config["enabled"]:
            try:
                await self._send_email_notification(
                    "critical_finding",
                    notification_data,
                    repo_info.get("notification_emails", [])
                )
                notification_results["email"]["sent"] = True
            except Exception as e:
                notification_results["email"]["error"] = str(e)
        
        # Send Slack notification
        if self.slack_config["enabled"]:
            try:
                await self._send_slack_notification(
                    "critical_finding",
                    notification_data
                )
                notification_results["slack"]["sent"] = True
            except Exception as e:
                notification_results["slack"]["error"] = str(e)
        
        # Send Discord notification
        if self.discord_config["enabled"]:
            try:
                await self._send_discord_notification(
                    "critical_finding",
                    notification_data
                )
                notification_results["discord"]["sent"] = True
            except Exception as e:
                notification_results["discord"]["error"] = str(e)
        
        # Send webhook notifications
        if self.webhook_configs:
            try:
                await self._send_webhook_notifications(
                    "critical_finding",
                    notification_data
                )
                notification_results["webhooks"]["sent"] = True
            except Exception as e:
                notification_results["webhooks"]["error"] = str(e)
        
        return notification_results
    
    async def send_analysis_complete_notification(self, analysis_results: Dict, repo_info: Dict) -> Dict:
        """Send notification when analysis is complete"""
        notification_results = {
            "email": {"sent": False, "error": None},
            "slack": {"sent": False, "error": None},
            "discord": {"sent": False, "error": None},
            "webhooks": {"sent": False, "error": None}
        }
        
        # Prepare notification data
        findings = analysis_results.get("findings", [])
        critical_count = len([f for f in findings if f.get("severity") == "critical"])
        high_count = len([f for f in findings if f.get("severity") == "high"])
        
        notification_data = {
            "repo_name": repo_info.get("name", "Unknown"),
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_findings": len(findings),
            "critical_count": critical_count,
            "high_count": high_count,
            "risk_score": analysis_results.get("risk_score", 0),
            "findings_summary": self._generate_findings_summary(findings)
        }
        
        # Send notifications
        if self.smtp_config["enabled"]:
            try:
                await self._send_email_notification(
                    "analysis_complete",
                    notification_data,
                    repo_info.get("notification_emails", [])
                )
                notification_results["email"]["sent"] = True
            except Exception as e:
                notification_results["email"]["error"] = str(e)
        
        if self.slack_config["enabled"]:
            try:
                await self._send_slack_notification(
                    "analysis_complete",
                    notification_data
                )
                notification_results["slack"]["sent"] = True
            except Exception as e:
                notification_results["slack"]["error"] = str(e)
        
        if self.discord_config["enabled"]:
            try:
                await self._send_discord_notification(
                    "analysis_complete",
                    notification_data
                )
                notification_results["discord"]["sent"] = True
            except Exception as e:
                notification_results["discord"]["error"] = str(e)
        
        if self.webhook_configs:
            try:
                await self._send_webhook_notifications(
                    "analysis_complete",
                    notification_data
                )
                notification_results["webhooks"]["sent"] = True
            except Exception as e:
                notification_results["webhooks"]["error"] = str(e)
        
        return notification_results
    
    async def send_pr_review_notification(self, pr_info: Dict, findings: List[Dict]) -> Dict:
        """Send notification when PR review is ready"""
        notification_results = {
            "email": {"sent": False, "error": None},
            "slack": {"sent": False, "error": None},
            "discord": {"sent": False, "error": None},
            "webhooks": {"sent": False, "error": None}
        }
        
        # Calculate risk level
        critical_count = len([f for f in findings if f.get("severity") == "critical"])
        high_count = len([f for f in findings if f.get("severity") == "high"])
        
        if critical_count > 0:
            risk_level = "Critical"
        elif high_count > 0:
            risk_level = "High"
        elif len(findings) > 10:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        notification_data = {
            "repo_name": pr_info.get("repo_name", "Unknown"),
            "pr_number": pr_info.get("pr_number", 0),
            "branch_name": pr_info.get("branch_name", "Unknown"),
            "findings_count": len(findings),
            "risk_level": risk_level
        }
        
        # Send notifications
        if self.smtp_config["enabled"]:
            try:
                await self._send_email_notification(
                    "pr_review_ready",
                    notification_data,
                    pr_info.get("notification_emails", [])
                )
                notification_results["email"]["sent"] = True
            except Exception as e:
                notification_results["email"]["error"] = str(e)
        
        if self.slack_config["enabled"]:
            try:
                await self._send_slack_notification(
                    "pr_review_ready",
                    notification_data
                )
                notification_results["slack"]["sent"] = True
            except Exception as e:
                notification_results["slack"]["error"] = str(e)
        
        if self.discord_config["enabled"]:
            try:
                await self._send_discord_notification(
                    "pr_review_ready",
                    notification_data
                )
                notification_results["discord"]["sent"] = True
            except Exception as e:
                notification_results["discord"]["error"] = str(e)
        
        if self.webhook_configs:
            try:
                await self._send_webhook_notifications(
                    "pr_review_ready",
                    notification_data
                )
                notification_results["webhooks"]["sent"] = True
            except Exception as e:
                notification_results["webhooks"]["error"] = str(e)
        
        return notification_results
    
    async def _send_email_notification(self, template_name: str, data: Dict, recipients: List[str]) -> None:
        """Send email notification using SMTP"""
        if not recipients:
            return
        
        template = self.templates.get(template_name, {})
        subject = template.get("subject", "Code Review Notification")
        html_content = template.get("email_template", "").format(**data)
        
        # Create message
        msg = MimeMultipart()
        msg['From'] = f"{self.smtp_config['from_name']} <{self.smtp_config['from_email']}>"
        msg['To'] = ", ".join(recipients)
        msg['Subject'] = subject
        
        # Add HTML content
        html_part = MimeText(html_content, 'html')
        msg.attach(html_part)
        
        # Send email
        try:
            if self.smtp_config["use_tls"]:
                server = smtplib.SMTP(self.smtp_config["smtp_server"], self.smtp_config["smtp_port"])
                server.starttls()
            else:
                server = smtplib.SMTP(self.smtp_config["smtp_server"], self.smtp_config["smtp_port"])
            
            server.login(self.smtp_config["username"], self.smtp_config["password"])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            raise Exception(f"Failed to send email: {str(e)}")
    
    async def _send_slack_notification(self, template_name: str, data: Dict) -> None:
        """Send Slack notification using webhook"""
        template = self.templates.get(template_name, {})
        text_content = template.get("slack_template", "").format(**data)
        
        payload = {
            "channel": self.slack_config["channel"],
            "username": self.slack_config["username"],
            "icon_emoji": self.slack_config["icon_emoji"],
            "text": text_content
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.slack_config["webhook_url"],
                json=payload,
                timeout=30
            ) as response:
                if response.status != 200:
                    raise Exception(f"Slack API returned status {response.status}")
    
    async def _send_discord_notification(self, template_name: str, data: Dict) -> None:
        """Send Discord notification using webhook"""
        template = self.templates.get(template_name, {})
        content = template.get("discord_template", "").format(**data)
        
        payload = {
            "username": self.discord_config["username"],
            "avatar_url": self.discord_config["avatar_url"],
            "content": content
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.discord_config["webhook_url"],
                json=payload,
                timeout=30
            ) as response:
                if response.status != 204:  # Discord returns 204 on success
                    raise Exception(f"Discord API returned status {response.status}")
    
    async def _send_webhook_notifications(self, template_name: str, data: Dict) -> None:
        """Send notifications to configured webhooks"""
        template = self.templates.get(template_name, {})
        
        for webhook_config in self.webhook_configs:
            if not webhook_config.get("enabled", True):
                continue
            
            try:
                payload = {
                    "event_type": template_name,
                    "timestamp": datetime.now().isoformat(),
                    "data": data,
                    "template": template
                }
                
                headers = webhook_config.get("headers", {})
                headers["Content-Type"] = "application/json"
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook_config["url"],
                        json=payload,
                        headers=headers,
                        timeout=webhook_config.get("timeout", 30)
                    ) as response:
                        if response.status not in [200, 201, 202]:
                            print(f"Warning: Webhook {webhook_config['url']} returned status {response.status}")
                            
            except Exception as e:
                print(f"Warning: Failed to send webhook to {webhook_config['url']}: {str(e)}")
    
    def _generate_findings_summary(self, findings: List[Dict]) -> str:
        """Generate HTML summary of findings for email"""
        if not findings:
            return "<li>No findings detected</li>"
        
        summary_items = []
        for finding in findings[:5]:  # Limit to top 5 findings
            severity = finding.get("severity", "unknown")
            severity_emoji = {
                "critical": "🚨",
                "high": "⚠️",
                "medium": "🔶",
                "low": "ℹ️"
            }.get(severity, "❓")
            
            summary_items.append(
                f"<li>{severity_emoji} <strong>{severity.title()}</strong>: {finding.get('message', 'No message')} "
                f"({finding.get('file', 'Unknown')}:{finding.get('line', 0)})</li>"
            )
        
        if len(findings) > 5:
            summary_items.append(f"<li>... and {len(findings) - 5} more findings</li>")
        
        return "".join(summary_items)
    
    async def test_notifications(self) -> Dict:
        """Test all notification channels"""
        test_data = {
            "repo_name": "Test Repository",
            "file_path": "test.py",
            "line_number": 42,
            "message": "This is a test notification",
            "tool": "test-tool",
            "severity": "medium",
            "remediation": "This is just a test",
            "risk_score": 5
        }
        
        results = await self.send_critical_finding_notification(test_data, {"name": "Test Repo"})
        
        return {
            "test_results": results,
            "config_status": {
                "email": self.smtp_config["enabled"],
                "slack": self.slack_config["enabled"],
                "discord": self.discord_config["enabled"],
                "webhooks": len([w for w in self.webhook_configs if w.get("enabled", True)])
            }
        }

# Global notification manager instance
notification_manager = NotificationManager()

# Convenience functions
async def send_critical_finding_notification(finding: Dict, repo_info: Dict) -> Dict:
    """Send notification for critical findings"""
    return await notification_manager.send_critical_finding_notification(finding, repo_info)

async def send_analysis_complete_notification(analysis_results: Dict, repo_info: Dict) -> Dict:
    """Send notification when analysis is complete"""
    return await notification_manager.send_analysis_complete_notification(analysis_results, repo_info)

async def send_pr_review_notification(pr_info: Dict, findings: List[Dict]) -> Dict:
    """Send notification when PR review is ready"""
    return await notification_manager.send_pr_review_notification(pr_info, findings)

async def test_notification_system() -> Dict:
    """Test all notification channels"""
    return await notification_manager.test_notifications()
