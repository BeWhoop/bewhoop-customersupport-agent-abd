import os, uuid, requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# LLM for escalation summaries
llm = ChatOpenAI(
    model_name="openai/gpt-3.5-turbo",
    base_url="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.2,
    max_tokens=500,
)

def escalate_to_slack(contact_info, query, ticket_id):
    summary = llm.invoke(f"Summarize this query for our human escalation, so they can look at a glance and become familiar with the issue: {query}").content

    payload = {
        "text": f":rotating_light: *New Escalation Ticket* :rotating_light:\n"
                f"*Ticket ID:* {ticket_id}\n"
                f"*User Contact:* {contact_info}\n"
                f"*Question:* {query}\n"
                f"*Issue Summary:* {summary}"
    }
    
    try:
        resp = requests.post(os.getenv("SLACK_WEBHOOK_URL"), json=payload)
        if resp.status_code == 200:
            return f"Escalated to support team with Ticket ID: {ticket_id}"
        else:
            return "Failed to escalate to Slack. Please try again later."
    except Exception as e:
        return f"Error escalating to Slack: {e}"

def create_support_ticket(question, contact_number, email_address):
    """Create a support ticket with contact information"""
    contact_info = {"contact_number": contact_number, "email_address": email_address}
    ticket_id = str(uuid.uuid4())[:8]
    
    try:
        escalate_to_slack(contact_info, question, ticket_id)
        return f"I couldn't find an answer, so I've escalated your request to support. Ticket ID: {ticket_id}. Our team will contact you shortly."
    except Exception as e:
        return f"Error creating support ticket: {e}. Please try again later." 