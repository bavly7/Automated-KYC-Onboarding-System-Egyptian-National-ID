import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from src2 import config

def send_handoff_email(to_email: str, handoff_link: str):

    if not config.SMTP_USER or not config.SMTP_PASSWORD:
        print(f"\n[EMAIL MOCK] Sending handoff link to {to_email}:")
        print(f"[EMAIL MOCK] Link: {handoff_link}\n")
        return

    subject = "Action Required: Continue your KYC Verification"
    

    html_body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
            <h2 style="color: #0284c7;">KYC Identity Verification</h2>
            <p>Hello,</p>
            <p>You requested to continue your identity verification on your mobile device.</p>
            <p>Please click the secure button below to resume your session:</p>
            <a href="{handoff_link}" style="display: inline-block; padding: 12px 24px; background-color: #0ea5e9; color: white; text-decoration: none; border-radius: 6px; font-weight: bold; margin: 10px 0;">Resume Verification</a>
            <p style="margin-top: 20px; font-size: 12px; color: #64748b;">
                This link is valid for {config.HANDOFF_TOKEN_TTL_MINUTES} minutes and can only be used once.
            </p>
            <p style="font-size: 12px; color: #64748b;">
                If the button doesn't work, copy and paste this URL into your browser:<br>
                <a href="{handoff_link}" style="color: #0ea5e9;">{handoff_link}</a>
            </p>
        </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = config.EMAIL_FROM
    msg["To"] = to_email

    msg.attach(MIMEText(html_body, "html"))

    try:

        server = smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT)
        server.starttls()  
        server.login(config.SMTP_USER, config.SMTP_PASSWORD)
        server.sendmail(config.EMAIL_FROM, to_email, msg.as_string())
        server.quit()
        print(f"[SUCCESS] Handoff email successfully sent to {to_email}")
    except Exception as e:
        print(f"[ERROR] Failed to send email to {to_email}. Error: {e}")