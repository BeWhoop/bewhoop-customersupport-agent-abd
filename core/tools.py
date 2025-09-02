from .memory import semantic_memory_lookup, semantic_memory_upsert, search_knowledge_base_internal
from .models import Answer, ConversationState
from concurrent.futures import ThreadPoolExecutor


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