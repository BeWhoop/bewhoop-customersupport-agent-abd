from langchain_community.vectorstores import SupabaseVectorStore
from db import supabase_client
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def chat():
    # This embedding_model is needed for retrieving from supabase
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2", 
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': False}
    )
    # Retrieving
    vector_store = SupabaseVectorStore(
        embedding=embedding_model,
        client=supabase_client,
        table_name="documents",
        query_name="match_documents",
    )

    # Initialising LLM
    llm = ChatOpenAI(model_name="openai/gpt-3.5-turbo",
                    base_url="https://openrouter.ai/api/v1",
                    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                    temperature=0.4,
                    max_tokens=500,
                    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful assistant for BeWhoop services. Answer questions related to account creation, vendor information, events, etc."),
        ("human", "Question: {input} \n Knowledge Base: {context} \n Please answer the user's question based on the above context.")
    ])

    # When we want to retreive something, like chunks, we should use create_stuff_documents_chain rather than simple chaining
    chain = create_stuff_documents_chain(llm,prompt)
        
    # Retreiving Data from vector db as per query
    retriever = vector_store.as_retriever(search_kwargs={"k" : 3})
    retrieval_chain = create_retrieval_chain(
            retriever,
            chain
        )
    return retrieval_chain

chain = chat()


if __name__ == "__main__":
    chat()
    response = chain.invoke({
    "input" : "How do i get in touch with the support?"
    })

    print(response["answer"])