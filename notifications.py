
"""Send email notifications when a new helpdesk ticket is created."""

import logging
import smtplib
from email.message import EmailMessage

import config

log = logging.getLogger("helpdesk.notifications")


def send_ticket_notification(
    ticket_id,
    question,
    category,
    priority,
    summary,
    department
):
    sender = config.get_setting("SMTP_EMAIL")
    app_password = config.get_setting("SMTP_APP_PASSWORD")
    admin_email = config.get_setting("ADMIN_EMAIL")

    if not all([sender, app_password, admin_email]):
        log.info("Email notification skipped: SMTP settings are missing.")
        return False

    message = EmailMessage()
    message["Subject"] = f"New Helpdesk Ticket #{ticket_id}"
    message["From"] = sender
    message["To"] = admin_email

    message.set_content(f"""
A new student helpdesk ticket has been created.

Ticket ID: {ticket_id}
Category: {category}
Priority: {priority}
Department: {department}

Student Query:
{question}

AI Summary:
{summary}

Status: Open

Please review this ticket in the Helpdesk Admin Dashboard.

AI Helpdesk Assistant
""")

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(sender, app_password.replace(" ", ""))
            server.send_message(message)

        log.info("Notification sent for ticket #%s.", ticket_id)
        return True

    except Exception:
        log.exception("Email notification failed for ticket #%s.", ticket_id)
        return False
