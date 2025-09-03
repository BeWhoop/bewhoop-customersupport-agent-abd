from .memory import semantic_memory_lookup, semantic_memory_upsert, search_knowledge_base_internal
from .models import Answer, ConversationState
from concurrent.futures import ThreadPoolExecutor
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# LLM for answering
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
    max_output_tokens=500,
)

# Agent Decision LLM - for smart routing
agent_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
    max_output_tokens=200,
)


def query_tools_parallel(query: str) -> tuple[Answer, Answer]:
    """Query Memory and Knowledge Base in parallel - only fetch chunks"""
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both queries simultaneously
        memory_future = executor.submit(semantic_memory_lookup, query)
        kb_future = executor.submit(search_knowledge_base_internal, query)
        
        # Get results
        memory_result = memory_future.result()
        kb_result = kb_future.result()
        
        return memory_result, kb_result

def process_tool_results(state: ConversationState, memory_result: Answer, kb_result: Answer) -> ConversationState:
    """Process and update state with tool results - NO LLM processing here"""
    state.qa_found = memory_result.found
    state.kb_found = kb_result.found
    state.qa_chunks = memory_result.chunks or []
    state.kb_chunks = kb_result.chunks or []
    
    return state

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
        
    return False

def ask_for_clarification(clarification_attempts: int) -> str:
    """Ask user for more context based on clarification attempt"""
    if clarification_attempts == 1:
        return ("I couldn't find answer about your question."
               "Could you please rephrase your question or provide more details about what you're looking for?")
    else:
        return ""  # Max attempts reached

def reset_conversation() -> ConversationState:
    """Reset conversation state for new interaction"""
    return ConversationState()

def is_waiting_for_clarification(conversation_state: ConversationState, max_attempts: int = 1) -> bool:
    """Check if we're in a clarification state"""
    return (conversation_state.clarification_attempts > 0 and 
            conversation_state.clarification_attempts < max_attempts and
            not conversation_state.escalation_needed)

def make_agent_decision(question: str, is_clarification: bool, clarification_attempts: int) -> str:
    """Intelligent agent that decides which tools to use"""
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a smart routing agent for a BeWhoop support system. Analyze the user's question and decide the best approach.

        Your options:
        1. "direct_answer" - If question is very basic and you can answer directly about BeWhoop
        2. "need_memory" - If this looks like a question that might have been asked before
        3. "need_kb_search" - If this needs specific information from knowledge base
        4. "need_both" - If you should search both memory and knowledge base
        5. "need_clarification" - If the question is too vague or unclear
        6. "escalate" - If this seems like a complex issue needing human help
        
        Context: This is a {context_type}.
        Current clarification attempts: {attempts}
        
        Be smart about routing:
        - Vague questions like "I need help" → need_clarification
        - Basic questions like "what is BeWhoop", "tell me about BeWhoop" → direct_answer
        - Specific how-to questions about BeWhoop features → need_kb_search or need_both
        - Questions that sound like repeats → need_memory first
        - Complex technical issues → escalate
        - Non-BeWhoop questions (weather, celebrities, etc) → direct_answer (to politely decline)
        
        Respond with ONLY the decision keyword."""),
        ("human", "Question: {question}")
    ])
    
    chain = decision_prompt | agent_llm
    context_type = "clarification attempt" if is_clarification else "new question"
    decision = chain.invoke({
        "question": question, 
        "context_type": context_type,
        "attempts": clarification_attempts
    }).content.strip().lower()
    
    return decision 