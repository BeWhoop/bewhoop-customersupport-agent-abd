import os, uuid, requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from .models import ConversationState

load_dotenv()

# LLM for escalation summaries
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
    max_output_tokens=500,
)

# =============================================================================
# MOST FREQUENTLY USED FUNCTIONS (Main Flow)
# =============================================================================

def handle_escalation_flow(state: ConversationState) -> tuple[bool, str]:
    """Handle the complete escalation flow - MAIN ENTRY POINT"""
    # Ask if user wants escalation
    if not ask_for_escalation():
        return False, "I might not be able to find the solution. Could you provide a more detailed question?"
    
    # Collect escalation information
    if collect_escalation_info(state):
        # Create ticket
        result = create_support_ticket(state)
        state.escalation_needed = True
        return True, result
    
    return False, "Unable to collect escalation information. Please try again."

def ask_for_escalation() -> bool:
    """Ask user if they want to escalate to human support"""
    print("\nI couldn't find an answer to your question after trying multiple approaches.")
    while True:
        response = input("Would you like me to escalate this to our human support team? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please answer 'yes' or 'no'.")

def collect_escalation_info(state: ConversationState) -> bool:
    """Collect user information for escalation"""
    print("\n--- Escalation Process ---")
    print("I'll need some information to create a support ticket for you.")
    
    # Collect email
    while not state.email:
        email = input("Please enter your email address: ").strip()
        if email and "@" in email:
            state.email = email
        else:
            print("Please enter a valid email address.")
    
    # Collect phone number
    while not state.number:
        number = input("Please enter your contact number: ").strip()
        if number:
            state.number = number
        else:
            print("Please enter a valid contact number.")
    
    # Generate issue summary
    try:
        summary_prompt = f"Create a concise professional summary of this customer support issue for our human agents to understand: {state.original_question,state.question}. It contains both original and most recent question. Return only the summary content without any headings or formatting."
        response = llm.invoke(summary_prompt)
        state.issue_summary = response.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        state.issue_summary = f"Customer inquiry: {state.question}"
    
    return True

def create_support_ticket(state: ConversationState) -> str:
    """Create a support ticket with collected information"""
    ticket_id = str(uuid.uuid4())[:8].upper()
    
    contact_info = {
        "contact_number": state.number,
        "email_address": state.email
    }
    
    # Try to send to Slack
    slack_success = escalate_to_slack(contact_info, state.original_question, state.question,ticket_id, state.issue_summary)
    
    if slack_success:
        return (f"✅ Support ticket created successfully!\n"
                f"Ticket ID: {ticket_id}\n"
                f"Our support team will contact you at {state.email} or {state.number} within 24 hours.\n"
                f"Please save your ticket ID for reference.")
    else:
        return (f"⚠️ Ticket created with ID: {ticket_id}, but there was an issue notifying our team.\n"
                f"Please contact our support directly at support@bewhoop.com with your ticket ID.\n"
                f"We'll respond to {state.email} as soon as possible.")

# =============================================================================
# UTILITY FUNCTIONS (Supporting Functions)
# =============================================================================

def escalate_to_slack(contact_info, original_question, query, ticket_id, issue_summary):
    """Send escalation notification to Slack"""
    try:
        payload = {
            "text": f":rotating_light: *New Escalation Ticket* :rotating_light:\n"
                    f"*Ticket ID:* {ticket_id}\n"
                    f"*User Contact Information:*\n"
                    f" • Phone: {contact_info.get('contact_number', 'N/A')}\n"
                    f" • Email: {contact_info.get('email_address', 'N/A')}\n"
                    f"*Original Question:* {original_question}\n"
                    f"*Most Recent Query:* {query}"
                    f"*Issue Summary:* {issue_summary}"
        }
        
        resp = requests.post(os.getenv("SLACK_WEBHOOK_URL"), json=payload)
        if resp.status_code == 200:
            return True
        else:
            print(f"Slack webhook failed with status: {resp.status_code}")
            return False
    except Exception as e:
        print(f"Error escalating to Slack: {e}")
        return False

# =============================================================================
# LEGACY/TESTING FUNCTIONS (Least Used)
# =============================================================================

def create_support_ticket_legacy(question, contact_number, email_address):
    """Legacy function for backward compatibility (Mainly for testing)"""
    state = ConversationState()
    state.question = question
    state.number = contact_number
    state.email = email_address
    return create_support_ticket(state) 
