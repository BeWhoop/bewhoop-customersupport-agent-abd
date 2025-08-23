from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from core import tools, create_support_ticket
import os
from dotenv import load_dotenv

load_dotenv()

# LLM for the agent
llm = ChatOpenAI(
    model_name="openai/gpt-3.5-turbo",
    base_url="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.2,
    max_tokens=500,
)

# Agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are BeWhoop's Support AI assistant. You help customers with questions about our services.

Your workflow:
1. ALWAYS start with search_memory to check for previously answered questions
2. If memory search finds a relevant answer, return it directly to the user
3. If no memory result, use search_knowledge_base to search the knowledge base
4. If knowledge base returns a clear answer, provide it to the user - DO NOT ask for escalation
5. If knowledge base returns "NO_ANSWER_FOUND_IN_KB", you MUST say: "I couldn't find an answer to your question. I will need to escalate this to our support team. Could you please provide your contact number and email address for further assistance?"

Critical Rules:
- When you see "NO_ANSWER_FOUND_IN_KB", do NOT ask for more context - escalate immediately
- If you find relevant information in the knowledge base, provide it without asking for escalation
- Never hallucinate or invent information
- Be professional and concise"""),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def handle_escalation(question):
    """Handle the escalation process by collecting contact info"""
    print("\n--- Escalation Process ---")
    contact_number = input("Please enter your contact number: ").strip()
    email_address = input("Please enter your email address: ").strip()
    
    if contact_number and email_address:
        result = create_support_ticket(question, contact_number, email_address)
        print(f"\nBeWhoop Assistant: {result}")
        return True
    else:
        print("\nBeWhoop Assistant: Both contact number and email are required for escalation. Please try again.")
        return False

def main():
    print("BeWhoop Support Assistant")
    print("Type 'exit' to quit")
    print("-" * 30)
    
    while True:
        question = input("\nAsk a question: ").strip()
        if question.lower() == "exit":
            print("Thank you for using BeWhoop Support!")
            break
        
        if not question:
            continue
            
        try:
            resp = agent_executor.invoke({"input": question})
            output = resp['output']
            print(f"\nBeWhoop Assistant: {output}")
            
            # Check if escalation is needed - only escalate if no answer found in KB
            if ("couldn't find an answer" in output.lower() and 
                "escalate" in output.lower() and 
                "contact number" in output.lower() and 
                "email" in output.lower()):
                if handle_escalation(question):
                    break
                
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or contact support directly.")

if __name__ == "__main__":
    main()
