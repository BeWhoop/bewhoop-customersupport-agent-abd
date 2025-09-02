from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from db.db import supabase_client
from .models import Answer
import os
from dotenv import load_dotenv

load_dotenv()

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase_client,
    table_name="documents",
    query_name="match_documents",
)

def semantic_memory_lookup(query: str, threshold: float = 0.82) -> Answer:
    """Search for previously answered questions in semantic memory"""
    # Turning query to vector for semantic search
    query_vec = embeddings.embed_query(query)
    # Searching
    response = supabase_client.rpc(
        "match_qa_memory",
        {"query_embedding": query_vec, "match_threshold": threshold, "match_count": 1}
    ).execute()
    data = getattr(response, "data", None) or []
    
    if data and len(data) > 0:
        result = data[0]
        # Validate the answer quality
        answer = result.get('answer', '').strip()
        if (answer and 
            len(answer) > 20 ):
            return Answer(found=True, chunks=[result])
    
    return Answer(found=False, chunks=[])

def search_knowledge_base_internal(query: str) -> Answer:
    """Search the knowledge base and return raw chunks - no LLM processing"""
    try:
        # Fetching chunks
        relevant_docs = vector_store.similarity_search(query, k=3)
        if not relevant_docs:
            return Answer(found=False, chunks=[])
        
        # Check if documents contain meaningful content
        doc_content = " ".join([doc.page_content for doc in relevant_docs])
        if len(doc_content.strip()) < 400:
            return Answer(found=False, chunks=[])
        
        # Return raw chunks - let the main LLM process them
        return Answer(found=True, chunks=relevant_docs)
        
    except Exception as e:
        print(f"Error in knowledge base search: {e}")
        return Answer(found=False, chunks=[])

def semantic_memory_upsert(question: str, answer: str):
    """Store question-answer pair in semantic memory"""
    # Converting Query to vectors
    q_vec = embeddings.embed_query(question)
    # Making payload as json, because upsert accepts json
    payload = {"question": question, "answer": answer, "q_embedding": q_vec}
    # Uploading Q/A to qa_memory table with question as a unique value
    supabase_client.table("qa_memory").upsert(payload, on_conflict="question").execute() 