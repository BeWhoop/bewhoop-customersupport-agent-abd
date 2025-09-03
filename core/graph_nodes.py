"""
LangGraph nodes and workflow management for BeWhoop Support Agent
"""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from .memory import semantic_memory_lookup, search_knowledge_base_internal, semantic_memory_upsert
from .tools import (
    query_tools_parallel, 
    process_tool_results, 
    answer_with_llm, 
    is_escalation_request,
    ask_for_clarification,
    make_agent_decision
)
from .escalation import handle_escalation_flow
from .models import ConversationState

# LangGraph State Schema
class AgentState(TypedDict):
    user_input: str
    is_clarification: bool
    processed_question: str
    agent_decision: str
    memory_results: Optional[dict]
    kb_results: Optional[dict]
    response: str
    should_continue: str
    needs_storage: bool
    debug_info: str

# Global state variable (will be managed from main.py)
conversation_state = None
MAX_CLARIFICATION_ATTEMPTS = 1

def set_conversation_state(state: ConversationState):
    """Set the global conversation state"""
    global conversation_state
    conversation_state = state

def input_processor_node(state: AgentState) -> AgentState:
    """Process and prepare user input"""
    global conversation_state
    
    user_input = state["user_input"]
    is_clarification = state.get("is_clarification", False)
    
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
    
    state["processed_question"] = question_to_process
    state["should_continue"] = "agent_decision"
    return state

def agent_decision_node(state: AgentState) -> AgentState:
    """Intelligent agent that decides which tools to use"""
    global conversation_state
    
    question = state["processed_question"]
    user_input = state["user_input"]
    is_clarification = state.get("is_clarification", False)
    
    # Check for direct escalation request first
    if is_escalation_request(user_input):
        state["agent_decision"] = "escalate"
        state["should_continue"] = "escalation_tool"
        return state
    
    # Use LLM to make intelligent routing decision
    decision = make_agent_decision(question, is_clarification, conversation_state.clarification_attempts)
    
    print(f"DEBUG: Agent decision: {decision}")
    
    # Route based on agent decision
    if decision == "direct_answer":
        state["agent_decision"] = "direct_answer"
        state["should_continue"] = "answer_node"
    elif decision == "need_memory":
        state["agent_decision"] = "need_memory"
        state["should_continue"] = "memory_tool"
    elif decision == "need_kb_search":
        state["agent_decision"] = "need_kb_search"
        state["should_continue"] = "kb_tool"
    elif decision == "need_both":
        state["agent_decision"] = "need_both"
        state["should_continue"] = "parallel_search"
    elif decision == "need_clarification":
        state["agent_decision"] = "need_clarification"
        state["should_continue"] = "clarification_tool"
    elif decision == "escalate":
        state["agent_decision"] = "escalate"
        state["should_continue"] = "escalation_tool"
    else:
        # Default fallback
        state["agent_decision"] = "need_both"
        state["should_continue"] = "parallel_search"
    
    return state

def memory_tool_node(state: AgentState) -> AgentState:
    """Search semantic memory"""
    global conversation_state
    
    question = state["processed_question"]
    
    memory_result = semantic_memory_lookup(question)
    
    state["memory_results"] = {"found": memory_result.found, "chunks": memory_result.chunks}
    
    if memory_result.found:
        conversation_state.qa_found = True
        conversation_state.qa_chunks = memory_result.chunks
        state["should_continue"] = "answer_node"
    else:
        # Memory didn't have it, try KB
        state["should_continue"] = "kb_tool"
    
    print(f"DEBUG: Memory search - Found: {memory_result.found}")
    return state

def kb_tool_node(state: AgentState) -> AgentState:
    """Search knowledge base"""
    global conversation_state
    
    question = state["processed_question"]
    
    kb_result = search_knowledge_base_internal(question)
    
    state["kb_results"] = {"found": kb_result.found, "chunks": kb_result.chunks}
    
    if kb_result.found:
        conversation_state.kb_found = True
        conversation_state.kb_chunks = kb_result.chunks
        state["should_continue"] = "answer_node"
        state["needs_storage"] = True  # Store KB answers in memory
    else:
        # No results from KB either
        state["should_continue"] = "clarification_tool"
    
    print(f"DEBUG: KB search - Found: {kb_result.found}")
    return state

def parallel_search_node(state: AgentState) -> AgentState:
    """Search both memory and KB in parallel"""
    global conversation_state
    
    question = state["processed_question"]
    is_clarification = state.get("is_clarification", False)
    
    # Query tools in parallel (Memory + KB)
    memory_result, kb_result = query_tools_parallel(question)
    
    # Process results and update state
    conversation_state = process_tool_results(conversation_state, memory_result, kb_result)
    
    state["memory_results"] = {"found": memory_result.found, "chunks": memory_result.chunks}
    state["kb_results"] = {"found": kb_result.found, "chunks": kb_result.chunks}
    
    debug_prefix = "After clarification - " if is_clarification else ""
    debug_msg = f"DEBUG: {debug_prefix}Memory found: {conversation_state.qa_found}, KB found: {conversation_state.kb_found}"
    print(debug_msg)
    
    if conversation_state.qa_found or conversation_state.kb_found:
        state["should_continue"] = "answer_node"
        state["needs_storage"] = conversation_state.kb_found and not conversation_state.qa_found
    else:
        state["should_continue"] = "clarification_tool"
    
    return state

def answer_node(state: AgentState) -> AgentState:
    """Generate answer from available information"""
    global conversation_state
    
    question = state["processed_question"]
    agent_decision = state.get("agent_decision", "")
    is_clarification = state.get("is_clarification", False)
    
    # Handle direct answer case
    if agent_decision == "direct_answer":
        # Use LLM to provide basic BeWhoop info or politely decline non-BeWhoop questions
        basic_context = """BeWhoop is a social platform that connects vendors with event organizers and event seekers. We help you discover events in your favorite genres and provide easy booking services. 

Key features:
- Event discovery and booking for event seekers
- Vendor registration and management
- Event organization tools
- Seamless connection between all parties

For non-BeWhoop questions, politely decline and redirect to BeWhoop topics."""
        
        answer = answer_with_llm(question, basic_context)
        state["response"] = answer
        state["should_continue"] = "end"
        return state
    
    # Priority: Memory → KB → No results
    if conversation_state.qa_found and conversation_state.qa_chunks:
        context = f"From Memory: {conversation_state.qa_chunks[0].get('answer', '')}"
        answer = answer_with_llm(question, context)
        
        if answer == "CANNOT_ANSWER_WITH_CONTEXT":
            debug_msg = "after clarification" if is_clarification else "treating as no results"
            print(f"DEBUG: LLM couldn't answer with memory context {debug_msg}")
            state["should_continue"] = "clarification_tool"
            return state
        
        state["response"] = answer
        state["should_continue"] = "end"
        return state
    
    elif conversation_state.kb_found and conversation_state.kb_chunks:
        context = f"From Knowledge Base: {' '.join([doc.page_content for doc in conversation_state.kb_chunks])}"
        answer = answer_with_llm(question, context)
        
        if answer == "CANNOT_ANSWER_WITH_CONTEXT":
            debug_msg = "after clarification" if is_clarification else "treating as no results"
            print(f"DEBUG: LLM couldn't answer with KB context {debug_msg}")
            state["should_continue"] = "clarification_tool"
            return state
        
        # Store successful KB answer in memory
        if state.get("needs_storage", False):
            semantic_memory_upsert(question, answer)
        
        state["response"] = answer
        state["should_continue"] = "end"
        return state
    
    # No results available
    else:
        state["should_continue"] = "clarification_tool"
        return state

def clarification_tool_node(state: AgentState) -> AgentState:
    """Handle clarification or escalation"""
    global conversation_state
    
    # Check if we've reached max clarification attempts
    if conversation_state.clarification_attempts >= MAX_CLARIFICATION_ATTEMPTS:
        print(f"DEBUG: Max attempts reached, escalating")
        state["should_continue"] = "escalation_tool"
        return state
    
    # Ask for clarification
    conversation_state.clarification_attempts += 1
    print(f"DEBUG: Incremented to attempt {conversation_state.clarification_attempts}")
    state["response"] = ask_for_clarification(conversation_state.clarification_attempts)
    state["should_continue"] = "end"
    return state

def escalation_tool_node(state: AgentState) -> AgentState:
    """Handle escalation to human support"""
    global conversation_state
    
    if state.get("agent_decision") == "escalate" and state.get("user_input"):
        conversation_state.question = state["user_input"].strip()
    
    escalated, message = handle_escalation_flow(conversation_state)
    
    if escalated:
        state["response"] = message
        state["should_continue"] = "end"
        return state
    else:
        # User declined escalation, reset for new question
        from .tools import reset_conversation
        conversation_state = reset_conversation()
        state["response"] = message
        state["should_continue"] = "end"
        return state

def route_next(state: AgentState) -> str:
    """Route to next node based on state"""
    return state["should_continue"]

def create_support_graph():
    """Create the intelligent LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("input_processor", input_processor_node)
    workflow.add_node("agent_decision", agent_decision_node)
    workflow.add_node("memory_tool", memory_tool_node)
    workflow.add_node("kb_tool", kb_tool_node)
    workflow.add_node("parallel_search", parallel_search_node)
    workflow.add_node("answer_node", answer_node)
    workflow.add_node("clarification_tool", clarification_tool_node)
    workflow.add_node("escalation_tool", escalation_tool_node)
    
    # Add edges
    workflow.add_edge(START, "input_processor")
    
    workflow.add_conditional_edges(
        "input_processor",
        route_next,
        {
            "agent_decision": "agent_decision"
        }
    )
    
    workflow.add_conditional_edges(
        "agent_decision",
        route_next,
        {
            "memory_tool": "memory_tool",
            "kb_tool": "kb_tool", 
            "parallel_search": "parallel_search",
            "answer_node": "answer_node",
            "clarification_tool": "clarification_tool",
            "escalation_tool": "escalation_tool"
        }
    )
    
    workflow.add_conditional_edges(
        "memory_tool",
        route_next,
        {
            "answer_node": "answer_node",
            "kb_tool": "kb_tool"
        }
    )
    
    workflow.add_conditional_edges(
        "kb_tool",
        route_next,
        {
            "answer_node": "answer_node",
            "clarification_tool": "clarification_tool"
        }
    )
    
    workflow.add_conditional_edges(
        "parallel_search",
        route_next,
        {
            "answer_node": "answer_node",
            "clarification_tool": "clarification_tool"
        }
    )
    
    workflow.add_conditional_edges(
        "answer_node",
        route_next,
        {
            "clarification_tool": "clarification_tool",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "clarification_tool",
        route_next,
        {
            "escalation_tool": "escalation_tool",
            "end": END
        }
    )
    
    workflow.add_edge("escalation_tool", END)
    
    return workflow.compile()

def process_with_langgraph(user_input: str, is_clarification: bool = False):
    """Process user input using intelligent LangGraph workflow"""
    initial_state = {
        "user_input": user_input,
        "is_clarification": is_clarification,
        "processed_question": "",
        "agent_decision": "",
        "memory_results": None,
        "kb_results": None,
        "response": "",
        "should_continue": "",
        "needs_storage": False,
        "debug_info": ""
    }
    
    # Create and run the graph
    support_graph = create_support_graph()
    result = support_graph.invoke(initial_state)
    return result["response"] 