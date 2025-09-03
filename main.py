import os
from dotenv import load_dotenv
from core import (
    ConversationState, 
    set_conversation_state,
    process_with_langgraph,
    reset_conversation,
    is_waiting_for_clarification
)

load_dotenv()

# Global state - much simpler than a class
conversation_state = ConversationState()
MAX_CLARIFICATION_ATTEMPTS = 1

def main():
    """Main application loop"""
    global conversation_state
    
    print("BeWhoop Support Assistant")
    print("I'm here to help you with BeWhoop-related questions!")
    print("Type 'exit' to quit")
    print("-" * 50)
    
    # Set the conversation state in graph_nodes module
    set_conversation_state(conversation_state)
    
    while True:
        # Get user input
        if is_waiting_for_clarification(conversation_state, MAX_CLARIFICATION_ATTEMPTS):
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
            # Process the input using intelligent LangGraph
            if is_waiting_for_clarification(conversation_state, MAX_CLARIFICATION_ATTEMPTS):
                response = process_with_langgraph(user_input, is_clarification=True)
            else:
                response = process_with_langgraph(user_input)
            
            print(f"\n🤖 BeWhoop Assistant: {response}")
            
            # If escalation was completed, reset for next conversation
            if conversation_state.escalation_needed:
                conversation_state = reset_conversation()
                set_conversation_state(conversation_state)
                print("\n" + "="*50)
                print("Ready for your next question!")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'reset' to start over.")

if __name__ == "__main__":
    main()
