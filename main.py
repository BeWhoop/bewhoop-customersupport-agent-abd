from langchain_community.vectorstores import SupabaseVectorStore
from db import supabase_client
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent

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

# LLM
llm = ChatOpenAI(
    model_name="openai/gpt-3.5-turbo",
    base_url="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.2,
    max_tokens=500,
)

def semantic_memory_lookup(query:str, threshold:float = 0.82):
    query_vec = embeddings.embed_query(query)
    response = supabase_client.rpc(
        "match_qa_memory",
        {"query_embedding": query_vec, "match_threshold": threshold, "match_count": 1}
    ).execute()
    data = getattr(response, "data", None) or []
    return data[0] if data else None

def semantic_memory_upsert(question: str, answer: str):
    # Converting Query to vectors
    q_vec = embeddings.embed_query(question)
    # Making payload as json, because upsert accepts json
    payload = {"question" : question, "answer" : answer, "q_embedding" : q_vec}
    # Uploading Q/A to qa_memory table with question as a unique value
    supabase_client.table("qa_memory").upsert(payload, on_conflict="question").execute()
    
# ------- TOOLS -------
@tool
def search_memory(query: str) -> str:
    """Search for previously answered questions in semantic memory. Use this first to check if the question has been answered before."""
    
    res = semantic_memory_lookup(query)
    if res and len(res['answer'].strip()) > 20 and not res['answer'].strip().lower().startswith("i don't know"):
        return f"Found in memory: {res['answer']}"
    
    return "No relevant answer found in memory."

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base using RAG and store the result in memory for future use. Use this when memory search returns no results."""
    
    relevant_docs = vector_store.similarity_search(query, k=3)
    if not relevant_docs:
        return "No relevant information found in the knowledge base."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for BeWhoop services. Answer questions related to account creation, vendor information, events, etc. Be concise where you can be"),
        ("human", "Question: {question} \n Knowledge Base: {context} \n Please answer the user's question based on the above context.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": relevant_docs, "question": query})
    answer = response.content
    
    if answer.strip() and not answer.strip().lower().startswith("i don't know"):
        # Store the new Q&A in memory for future use
        semantic_memory_upsert(query, answer)
        return answer
    
    return "Sorry, I couldn't find relevant information in the knowledge base."


tools = [search_memory, search_knowledge_base]

# Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are BeWhoop's support AI assistant. You have access to two {tools}:

Workflow:
- Always start with search_memory
- If memory search finds a relevant answer, return it directly
- If memory search finds no results, then use search_knowledge_base
- Return the answer from whichever tool provides useful information
- Be helpful and concise in your responses"""),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    while True:
        question = input("Ask a question: ").strip()
        if question.lower() == "exit":
            break
        resp = agent_executor.invoke({"input": question, "tools": tools})
        print(resp['output'])
