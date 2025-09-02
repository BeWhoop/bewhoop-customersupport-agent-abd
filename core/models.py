from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Answer:
    """Standard response format for Memory and KB tools"""
    found: bool  # true | false
    chunks: Optional[List] = None

@dataclass
class ConversationState:
    """State schema for tracking conversation flow"""
    original_question: str = "asdqadqasdq"
    question: str = ""
    answer: str = ""
    qa_chunks: List = None
    kb_chunks: List = None
    qa_found: bool = False
    kb_found: bool = False
    escalation_needed: bool = False
    email: str = ""
    number: str = ""
    issue_summary: str = ""
    clarification_attempts: int = 0
    
    def __post_init__(self):
        if self.qa_chunks is None:
            self.qa_chunks = []
        if self.kb_chunks is None:
            self.kb_chunks = []
    
    def reset_search_results(self):
        """Reset search-related fields for new queries"""
        self.qa_chunks = []
        self.kb_chunks = []
        self.qa_found = False
        self.kb_found = False
        self.answer = ""
    
    def has_results(self) -> bool:
        """Check if either memory or KB found results"""
        return self.qa_found or self.kb_found 