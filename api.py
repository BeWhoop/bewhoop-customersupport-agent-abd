import os
from dotenv import load_dotenv
from fastapi import FastAPI, Form
from pydantic import BaseModel

from core import (
    ConversationState, 
    set_conversation_state,
    process_with_langgraph,
    reset_conversation,
    is_waiting_for_clarification
)

# Load environment variables
load_dotenv()

# Global state
conversation_state = ConversationState()
MAX_CLARIFICATION_ATTEMPTS = 1
set_conversation_state(conversation_state)

# Temporary in-memory ticket store (replace with DB or external ticketing system)
tickets = []

# FastAPI app
app = FastAPI(title="BeWhoop Support Assistant API")

# ======================
# Schemas
# ======================
class Query(BaseModel):
    question: str

# ======================
# Routes
# ======================

@app.get("/")
def root():
    """Health check endpoint"""
    return {"message": "BeWhoop Support Assistant API is running"}

@app.post("/ask")
def ask_question(query: Query):
    """
    Main endpoint for customer queries.
    Handles conversation state and clarification logic.
    """
    global conversation_state

    user_input = query.question.strip()
    if not user_input:
        return {"error": "Question cannot be empty."}

    try:
        # Handle clarification
        if is_waiting_for_clarification(conversation_state, MAX_CLARIFICATION_ATTEMPTS):
            response = process_with_langgraph(user_input, is_clarification=True)
        else:
            response = process_with_langgraph(user_input)

        # Reset if escalation happened
        if conversation_state.escalation_needed:
            conversation_state = reset_conversation()
            set_conversation_state(conversation_state)

        return {"answer": response}

    except Exception as e:
        return {"error": str(e)}

@app.post("/reset")
def reset_conversation_api():
    """
    Reset the current conversation state manually.
    """
    global conversation_state
    conversation_state = reset_conversation()
    set_conversation_state(conversation_state)
    return {"message": "Conversation has been reset."}

@app.post("/escalate")
def escalate_ticket(user_id: str = Form(...), issue: str = Form(...)):
    """
    Explicitly create a support ticket if the AI agent fails or
    if the user requests escalation.
    """
    ticket = {
        "ticket_id": len(tickets) + 1,
        "user_id": user_id,
        "issue": issue,
        "status": "open"
    }
    tickets.append(ticket)

    # 🚀 Ready for production:
    # Replace this with DB insert (with SQLAlchemy or supabase)
    return {"message": "Ticket created successfully", "ticket": ticket}


