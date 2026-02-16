import os
from typing import Dict

import sendgrid
from sendgrid.helpers.mail import Email, Mail, Content, To
from agents import Agent, function_tool
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX


@function_tool
def send_email(subject: str, html_body: str) -> Dict[str, str]:
    """Send an email with the given subject and HTML body"""
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
    from_email = Email("ed@edwarddonner.com")  # put your verified sender here
    to_email = To("ed.donner@gmail.com")  # put your recipient here
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email, subject, content).get()
    response = sg.client.mail.send.post(request_body=mail)
    print("Email response", response.status_code)
    return "success"


INSTRUCTIONS = f"""{RECOMMENDED_PROMPT_PREFIX}

You send research reports by email. You will be given a detailed report in the conversation (from the Research Manager).
Use your tool to send one email: convert the report to clean, well-presented HTML and choose an appropriate subject line.
Once you have sent the email, hand back to the Research Manager so they can show the report to the user."""

email_agent = Agent(
    name="EmailAgent",
    instructions=INSTRUCTIONS,
    tools=[send_email],
    model="gpt-4o-mini",
    handoff_description="Transfer here when the report is written and ready to send by email. You will send it and then hand back.",
)
