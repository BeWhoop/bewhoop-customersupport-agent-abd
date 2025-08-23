from langchain_huggingface import HuggingFaceEmbeddings
from db.db import supabase_client

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def semantic_memory_lookup(query: str, threshold: float = 0.82):
    # Turning query to vector for semantic search
    query_vec = embeddings.embed_query(query)
    # Searching
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
    payload = {"question": question, "answer": answer, "q_embedding": q_vec}
    # Uploading Q/A to qa_memory table with question as a unique value
    supabase_client.table("qa_memory").upsert(payload, on_conflict="question").execute() 