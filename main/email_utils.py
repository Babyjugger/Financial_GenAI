import os
import ssl
import certifi
import smtplib
import traceback

from config import EMAIL_CONFIG
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication


def send_email_with_attachment(subject, body, to_email, attachment_path, attachment_name=None):
    """
    Send an email with a PDF attachment.

    Args:
        subject: Email subject
        body: Email body content
        to_email: Recipient email address
        attachment_path: Path to the attachment file
        attachment_name: Name to use for the attachment (defaults to filename)

    Returns:
        bool: True if successful, False otherwise
    """
    # Email configuration - store these in environment variables in production
    smtp_server = EMAIL_CONFIG.get("smtp_server", "smtp.gmail.com")
    smtp_port = EMAIL_CONFIG.get("smtp_port", 587)
    smtp_username = EMAIL_CONFIG.get("smtp_username", "")
    smtp_password = EMAIL_CONFIG.get("smtp_password", "")

    # Create message
    message = MIMEMultipart()
    message["Subject"] = subject
    message["From"] = smtp_username
    message["To"] = to_email

    # Add body
    message.attach(MIMEText(body, "plain"))

    # Add attachment
    try:
        with open(attachment_path, "rb") as attachment:
            # If no attachment name specified, use the filename
            if not attachment_name:
                attachment_name = os.path.basename(attachment_path)

            part = MIMEApplication(attachment.read(), Name=attachment_name)
            part['Content-Disposition'] = f'attachment; filename="{attachment_name}"'
            message.attach(part)
    except Exception as e:
        print(f"Error attaching file: {str(e)}")
        return False

    # Send email
    try:
        # Try to create a context with certifi's certificates
        try:
            # First attempt with certifi (may need to install: pip install certifi)
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            # If certifi is not available, try to use the system's certificates
            context = ssl.create_default_context()
            # On macOS, you might need to install certificates
            print("Consider installing certifi: pip install certifi")

        # Special handling for macOS certificate issues
        if os.path.exists('/Applications/Python 3.12/Install Certificates.command'):
            print("You need to install certificates for macOS.")
            print("Run: /Applications/Python 3.12/Install Certificates.command")
            # Temporarily disable verification for this example only
            # WARNING: This is insecure, but a temporary workaround
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(smtp_username, smtp_password)
            server.send_message(message)
        print(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        print(traceback.format_exc())

        # If we got a certificate error, try with verification disabled (not secure, but may work)
        if isinstance(e, ssl.SSLCertVerificationError):
            try:
                print("Attempting to send without certificate verification (insecure, but may work)")
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE

                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls(context=context)
                    server.login(smtp_username, smtp_password)
                    server.send_message(message)
                print(f"Email sent successfully to {to_email} (with verification disabled)")
                return True
            except Exception as fallback_error:
                print(f"Still failed to send email: {str(fallback_error)}")

        return False

