from __future__ import annotations

import smtplib
from email.message import EmailMessage

from app.config import settings


def send_alert_email(subject: str, body: str) -> bool:
    if not settings.smtp_host or not settings.smtp_user or not settings.smtp_password:
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.alert_from
    msg["To"] = settings.alert_to
    msg.set_content(body)

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
        smtp.starttls()
        smtp.login(settings.smtp_user, settings.smtp_password)
        smtp.send_message(msg)
    return True

