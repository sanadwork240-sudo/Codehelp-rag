
import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()


def send_ticket_notification(
    ticket_id,
    question,
    category,
    priority,
    summary,
    department
):
    sender = os.getenv("SMTP_EMAIL")
    app_password = os.getenv("SMTP_APP_PASSWORD")
    admin_email = os.getenv("ADMIN_EMAIL")

    if not all([sender, app_password, admin_email]):
        print("Email notification skipped: SMTP settings are missing.")
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

Greenfield Institute of Technology
AI Helpdesk Assistant
""")

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, app_password.replace(" ", ""))
            server.send_message(message)

        print(f"Notification sent for ticket #{ticket_id}.")
        return True

    except Exception as error:
        print(f"Email notification failed: {error}")
        return False
