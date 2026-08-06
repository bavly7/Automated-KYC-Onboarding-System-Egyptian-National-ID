"""
Sends the mobile-continuation magic link. Falls back to printing the link
to the console if SMTP isn't configured (config.SMTP_HOST is None) — this
is what makes the handoff flow demoable without needing real email
credentials set up before your defense.
"""
import smtplib
from email.message import EmailMessage

from src2 import config


def send_handoff_email(to_email: str, link: str) -> None:
    subject = "Continue your ID verification on your phone"
    body = (
        f"Hi,\n\n"
        f"Tap the link below on your phone to continue your verification:\n\n"
        f"{link}\n\n"
        f"This link expires in {config.HANDOFF_TOKEN_TTL_MINUTES} minutes and can only be used once."
    )

    if not config.SMTP_HOST:
        # Demo fallback — no real SMTP configured yet.
        print(f"[email_utils] SMTP not configured — would have sent to {to_email}:")
        print(f"[email_utils] {link}")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.EMAIL_FROM
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
        server.starttls()
        if config.SMTP_USER and config.SMTP_PASSWORD:
            server.login(config.SMTP_USER, config.SMTP_PASSWORD)
        server.send_message(msg)
