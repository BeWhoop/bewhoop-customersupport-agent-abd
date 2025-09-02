import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from core import (
    ConversationState, 
    query_tools_parallel, 
    process_tool_results, 
    handle_escalation_flow,
    semantic_memory_upsert
)

load_dotenv()

# Global state - much simpler than a class
conversation_state = ConversationState()
MAX_CLARIFICATION_ATTEMPTS = 1

# LLM for answering
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
    max_output_tokens=500,
)

def answer_with_llm(question: str, context: str) -> str:
    """Use LLM to answer question with context - returns CANNOT_ANSWER if context is insufficient"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a BeWhoop Assistant. BeWhoop is a social platform that connects vendors with event organizers and event seekersq with their favourite genre events, providing services for event seekers, vendor registration, event management, and facility of booking events for event seekers with ease.

        Your job:
        1. If the question is about BeWhoop services/platform OR about who you are/your capabilities - answer it helpfully
        2. If the question is completely unrelated to BeWhoop (like "who was Albert Einstein", "what's the weather") - politely decline and redirect to BeWhoop topics
        3. Use the provided context to give detailed, comprehensive answers
        4. Be conversational and helpful, not robotic

        CRITICAL: Only provide a real answer if the context actually contains SPECIFIC, ACTIONABLE information needed to answer the question. Generic mentions or vague statements are NOT sufficient. If the provided context doesn't contain DETAILED, SPECIFIC information to properly answer the user's question, you MUST respond with exactly: "CANNOT_ANSWER_WITH_CONTEXT"
        
        Examples of INSUFFICIENT context:
        - "you can book events with ease" (too vague, no steps)
        - "BeWhoop provides booking services" (no how-to details)  
        - General platform descriptions without specific instructions"""),
        ("human", "Question: {question}\n\nContext: {context}\n\nPlease respond:")
        ])
    
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})
    return response.content.strip()

def is_escalation_request(user_input: str) -> bool:
    """Check if user is specifically requesting escalation"""
    escalation_keywords = [
        "escalate", "escalation", "human support", "human agent", 
        "talk to human", "speak to human", "real person", "live agent", "escalate this",
        "escalate my issue", "i want to escalate", "need human help"
    ]
    
    user_input_lower = user_input.lower().strip()
    
    # Check for direct escalation requests
    if any(keyword in user_input_lower for keyword in escalation_keywords):
        return True
    
    # Check for phrases
    # if user_input_lower in ["yes escalate", "escalate please", "escalate", "yes, escalate","can you transfer me to human support"]:
    #     return True
        
    return False


def ask_for_clarification() -> str:
    """Ask user for more context based on clarification attempt"""
    # print(f"DEBUG: Clarification attempt {conversation_state.clarification_attempts}")
    
    if conversation_state.clarification_attempts == 1:
        return ("I couldn't find answer about your question."
               "Could you please rephrase your question or provide more details about what you're looking for?")
    else:
        return ""  # Max attempts reached

def process_user_input(user_input: str, is_clarification: bool = False) -> str:
    """Unified function to process user input (either new question or clarification)"""
    global conversation_state
    
    # Check if user is specifically requesting escalation
    if is_escalation_request(user_input):
        conversation_state.question = user_input.strip()
        escalated, message = handle_escalation_flow(conversation_state)
        if escalated:
            return message
        else:
            # User declined escalation, reset for new question
            reset_conversation()
            return message
    
    # Prepare the question based on whether it's a clarification or new question
    if is_clarification:
        # Combine original question with clarification
        question_to_process = f"{conversation_state.original_question} {user_input}"
        print(f"DEBUG: handle_clarification called, current attempts: {conversation_state.clarification_attempts}")
    else:
        # New question - update state and reset search results
        question_to_process = user_input.strip()
        conversation_state.question = question_to_process
        # Only set original_question if this is a fresh conversation (no clarification attempts yet)
        if conversation_state.clarification_attempts == 0:
            conversation_state.original_question = question_to_process
        conversation_state.reset_search_results()
    
    # Query tools in parallel (Memory + KB) - only fetch chunks
    memory_result, kb_result = query_tools_parallel(question_to_process)
    
    # Process results and update state
    conversation_state = process_tool_results(conversation_state, memory_result, kb_result)
    
    debug_prefix = "After clarification - " if is_clarification else ""
    print(f"DEBUG: {debug_prefix}Memory found: {conversation_state.qa_found}, KB found: {conversation_state.kb_found}")
    
    # Check chunks in priority order: Memory → KB → Clarification
    
    # First check memory chunks
    if conversation_state.qa_found and conversation_state.qa_chunks:
        context = f"From Memory: {conversation_state.qa_chunks[0].get('answer', '')}"
        answer = answer_with_llm(question_to_process, context)
        
        # Check if LLM couldn't answer with the context
        if answer == "CANNOT_ANSWER_WITH_CONTEXT":
            debug_msg = "after clarification" if is_clarification else "treating as no results"
            print(f"DEBUG: LLM couldn't answer with memory context {debug_msg}")
            return handle_no_results()
        
        return answer
    
    # Then check KB chunks
    elif conversation_state.kb_found and conversation_state.kb_chunks:
        context = f"From Knowledge Base: {' '.join([doc.page_content for doc in conversation_state.kb_chunks])}"
        answer = answer_with_llm(question_to_process, context)
        
        # Check if LLM couldn't answer with the context
        if answer == "CANNOT_ANSWER_WITH_CONTEXT":
            debug_msg = "after clarification" if is_clarification else "treating as no results"
            print(f"DEBUG: LLM couldn't answer with KB context {debug_msg}")
            return handle_no_results()
        
        # Store this Q&A in memory for future use (only if successfully answered)
        semantic_memory_upsert(question_to_process, answer)
        return answer
    
    # No chunks found - ask for clarification or escalate
    else:
        debug_msg = "Still no chunks after clarification" if is_clarification else "No chunks found, going to handle_no_results"
        print(f"DEBUG: {debug_msg}")
        return handle_no_results()

def handle_no_results() -> str:
    """Handle cases where no results were found"""
    global conversation_state
    
    # print(f"DEBUG: handle_no_results called, current attempts: {conversation_state.clarification_attempts}")
    
    # Check if we've reached max clarification attempts
    if conversation_state.clarification_attempts >= MAX_CLARIFICATION_ATTEMPTS:
        print(f"DEBUG: Max attempts reached, escalating")
        # Proceed to escalation flow
        escalated, message = handle_escalation_flow(conversation_state)
        if escalated:
            return message
        else:
            # User declined escalation, reset for new question
            reset_conversation()
            return message
    
    # Ask for clarification
    conversation_state.clarification_attempts += 1
    print(f"DEBUG: Incremented to attempt {conversation_state.clarification_attempts}")
    return ask_for_clarification()


def reset_conversation():
    """Reset conversation state for new interaction"""
    global conversation_state
    conversation_state = ConversationState()

def is_waiting_for_clarification() -> bool:
    """Check if we're in a clarification state"""
    return (conversation_state.clarification_attempts > 0 and 
            conversation_state.clarification_attempts < MAX_CLARIFICATION_ATTEMPTS and
            not conversation_state.escalation_needed)

def main():
    """Main application loop"""
    print("BeWhoop Support Assistant")
    print("I'm here to help you with BeWhoop-related questions!")
    print("Type 'exit' to quit")
    print("-" * 50)
    
    while True:
        # Get user input
        if is_waiting_for_clarification():
            user_input = input("\nPlease provide more details: ").strip()
        else:
            user_input = input("\nAsk a question: ").strip()
        
        # Handle special commands
        if user_input.lower() == "exit":
            print("\n👋 Thank you for using BeWhoop Support!")
            break
        
        if not user_input:
            continue
        
        try:
            # Process the input
            if is_waiting_for_clarification():
                response = process_user_input(user_input, is_clarification=True)
            else:
                response = process_user_input(user_input)
            
            print(f"\n🤖 BeWhoop Assistant: {response}")
            
            # If escalation was completed, reset for next conversation
            if conversation_state.escalation_needed:
                reset_conversation()
                print("\n" + "="*50)
                print("Ready for your next question!")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'reset' to start over.")

if __name__ == "__main__":
    main()
