"""
Email Service for Password Reset
Face Recognition Attendance System
"""

import smtplib
import random
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import os


def generate_reset_code(length=6):
    """Generate a random numeric reset code"""
    return ''.join(random.choices(string.digits, k=length))


def generate_reset_token():
    """Generate a secure reset token"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=32))


def send_reset_email(to_email, reset_code, username):
    """
    Send password reset email with code

    Note: For this to work, you need to configure SMTP settings.
    For Gmail, you need to:
    1. Enable 2-Factor Authentication
    2. Create an App Password at https://myaccount.google.com/apppasswords
    3. Set environment variables:
       - SMTP_EMAIL: your Gmail address
       - SMTP_PASSWORD: your App Password (not your regular password)
    """

    # Get SMTP settings from environment or use defaults
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    smtp_email = os.getenv('SMTP_EMAIL', '')
    smtp_password = os.getenv('SMTP_PASSWORD', '')

    if not smtp_email or not smtp_password:
        # Return the code for demo/testing purposes when email is not configured
        return {
            'success': False,
            'message': 'Email not configured. Use this code for testing.',
            'code': reset_code,
            'demo_mode': True
        }

    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = 'Password Reset Code - AttendEase'
        msg['From'] = smtp_email
        msg['To'] = to_email

        # Plain text version
        text = f"""
Hello {username},

You requested a password reset for your AttendEase account.

Your password reset code is: {reset_code}

This code will expire in 15 minutes.

If you didn't request this reset, please ignore this email.

Best regards,
AttendEase Team
        """

        # HTML version
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Inter', Arial, sans-serif; background-color: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 500px; margin: 0 auto; background: white; border-radius: 12px; padding: 40px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .logo {{ font-size: 28px; font-weight: 700; color: #f97316; }}
        .code-box {{ background: #fff7ed; border: 2px solid #f97316; border-radius: 8px; padding: 20px; text-align: center; margin: 20px 0; }}
        .code {{ font-size: 32px; font-weight: 700; color: #ea580c; letter-spacing: 8px; }}
        .message {{ color: #525252; line-height: 1.6; }}
        .footer {{ text-align: center; margin-top: 30px; color: #a3a3a3; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">AttendEase</div>
        </div>
        <p class="message">Hello <strong>{username}</strong>,</p>
        <p class="message">You requested a password reset for your account. Use the code below to reset your password:</p>
        <div class="code-box">
            <div class="code">{reset_code}</div>
        </div>
        <p class="message">This code will expire in <strong>15 minutes</strong>.</p>
        <p class="message">If you didn't request this reset, please ignore this email.</p>
        <div class="footer">
            <p>&copy; 2024 AttendEase - Smart Attendance System</p>
        </div>
    </div>
</body>
</html>
        """

        part1 = MIMEText(text, 'plain')
        part2 = MIMEText(html, 'html')

        msg.attach(part1)
        msg.attach(part2)

        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, to_email, msg.as_string())

        return {
            'success': True,
            'message': f'Reset code sent to {to_email}',
            'demo_mode': False
        }

    except Exception as e:
        return {
            'success': False,
            'message': f'Failed to send email: {str(e)}',
            'code': reset_code,
            'demo_mode': True
        }
