from langchain.agents import tool
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from db.db import supabase_client
from .memory import embeddings, semantic_memory_lookup, semantic_memory_upsert
import os
from dotenv import load_dotenv

load_dotenv()

# Vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase_client,
    table_name="documents",
    query_name="match_documents",
)

# LLM
llm = ChatOpenAI(
    model_name="openai/gpt-3.5-turbo",
    base_url="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.2,
    max_tokens=500,
)

@tool
def search_memory(query: str) -> str:
    """Search for previously answered questions in semantic memory. Use this first to check if the question has been answered before."""
    
    res = semantic_memory_lookup(query)
    if (res and 
        len(res['answer'].strip()) > 20 and 
        not res['answer'].strip().lower().startswith("i don't know") and
        not res['answer'].strip().lower().startswith("sorry") and
        not res['answer'].strip().lower().startswith("no_answer_found")): # we can remove this extra conditions, just for safety.
        return f"Found in memory: {res['answer']}"
    
    return "No relevant answer found in memory."

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base using RAG. Use this when memory search returns no results."""
    # Fetching chunks
    relevant_docs = vector_store.similarity_search(query, k=3)
    if not relevant_docs:
        return "NO_ANSWER_FOUND_IN_KB"
    
    # Check if documents contain meaningful content / We can also remove this
    doc_content = " ".join([doc.page_content for doc in relevant_docs])
    if len(doc_content.strip()) < 20:
        return "NO_ANSWER_FOUND_IN_KB"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for BeWhoop services. Answer questions related to account creation, vendor information, events, etc. Be concise and helpful. If you cannot find a clear answer in the provided context, respond with 'NO_CLEAR_ANSWER'."),
        ("human", "Question: {question} \n Knowledge Base: {context} \n Please answer the user's question based on the above context.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": relevant_docs, "question": query})
    answer = response.content.strip()
    
    # Validate the answer quality
    if (answer and 
        not answer.lower().startswith("i don't know") and
        not answer.lower().startswith("no_clear_answer") and
        not answer.lower().startswith("sorry") and
        len(answer) > 15): # Made this extra conditions just because it does'nt store not-found answers in the db
        
        # Store the validated Q&A in memory for future use
        semantic_memory_upsert(query, answer)
        return answer
    
    return "NO_ANSWER_FOUND_IN_KB"

# Export tools list
tools = [search_memory, search_knowledge_base] 